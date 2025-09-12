import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


# --- Normalizing Flows ---

class _PlanarFlow(nn.Module):
    """
    Planar flow: z' = z + u_hat * tanh(w^T z + b)
    Works on inputs shaped [batch, 1, dim]. Returns (z', log|det J|) with logdet per batch.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.u = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _m(w_dot_u: torch.Tensor) -> torch.Tensor:
        # Enforce invertibility via u_hat: u_hat = u + (m - w^T u) * w / ||w||^2, m = -1 + softplus(w^T u)
        return -1.0 + F.softplus(w_dot_u)

    def forward(self, z: torch.Tensor):
        # z: [B, 1, D]
        assert z.dim() == 3 and z.size(1) == 1, "_PlanarFlow expects [B,1,D] input"
        B, _, D = z.shape
        w = self.w.view(1, 1, D)
        u = self.u.view(1, 1, D)
        b = self.b.view(1, 1, 1)
        # Compute u_hat
        w_dot_u = torch.sum(self.w * self.u)
        m = self._m(w_dot_u)
        w_norm_sq = torch.sum(self.w * self.w) + 1e-8
        u_hat = self.u + ((m - w_dot_u) * self.w) / w_norm_sq
        u_hat = u_hat.view(1, 1, D)

        # a = w^T z + b, h = tanh(a)
        a = torch.sum(w * z, dim=-1, keepdim=True) + b  # [B,1,1]
        h = torch.tanh(a)  # [B,1,1]
        z_new = z + u_hat * h  # broadcast over last dim

        # log|det J| = log|1 + u_hat^T psi|, psi = h'(a) * w
        h_prime = 1.0 - torch.tanh(a) ** 2  # [B,1,1]
        psi = h_prime * w  # [B,1,D]
        inner = torch.sum(u_hat * psi, dim=-1).squeeze(-1)  # [B,1]
        logdet = torch.log(torch.abs(1.0 + inner) + 1e-8).squeeze(-1)  # [B]
        return z_new, logdet


# MAB - Multi-head Attention Block
class MAB(nn.Module):
    def __init__(self, dim, heads, dropout: float = 0.1):
        super().__init__()
        self.multihead = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        q_t = q.transpose(0, 1)
        kv_t = kv.transpose(0, 1)
        attn_output, _ = self.multihead(q_t, kv_t, kv_t)
        attn_output = attn_output.transpose(0, 1)
        # Residual + Pre-LN
        a = self.ln1(q + self.dropout(attn_output))
        ff_out = self.ff(a)
        return self.ln2(a + self.dropout(ff_out))


# ISAB - Induced Set Attention Block
class ISAB(nn.Module):
    def __init__(self, dim, heads, m):
        super().__init__()
        self.mab1 = MAB(dim, heads)
        self.mab2 = MAB(dim, heads)
        self.inducing = nn.Parameter(torch.randn(m, dim))

    def forward(self, x):
        batch_size = x.size(0)
        inducing = self.inducing.unsqueeze(0).expand(batch_size, -1, -1)
        h = self.mab1(inducing, x)
        return self.mab2(x, h)


# PoolingByMultiheadAttention
class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(dim, heads, dropout=dropout)
        self.ff = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # Learnable base query to make decoding deterministic; tiled to target_n
        self.base_query = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, z, target_n, noise_std=0.5):
        batch_size = z.size(0)
        seed = self.base_query.expand(batch_size, target_n, -1)
        if self.training and noise_std > 0:
            seed = seed + torch.randn_like(seed) * noise_std
        ff_z = F.gelu(self.ff(z))
        ln_ff = self.ln(ff_z)
        mab_out = self.mab(seed, ln_ff)
        return self.dropout(mab_out + seed)


# AttentiveBottleneckLayer
class AttentiveBottleneckLayer(nn.Module):
    def __init__(self, dim, heads, m):
        super().__init__()
        self.isab = ISAB(dim, heads, m)
        self.pma = PoolingByMultiheadAttention(dim, heads)

    def forward(self, z_prev, target_n, x=None, noise_std=0.5):
        if x is not None:
            z = self.isab(torch.cat([z_prev, x], dim=1))
        else:
            z = self.isab(z_prev)
        return self.pma(z, target_n, noise_std=noise_std)


# SetVAE Module
class SetVAEModule(nn.Module):
    def __init__(self, input_dim, reduced_dim, latent_dim, levels, heads, m, enable_posterior_std_augmentation: bool = True, use_flows: bool = False, num_flows: int = 0):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.levels = levels
        self.enable_posterior_std_augmentation = enable_posterior_std_augmentation
        self.use_flows = bool(use_flows)
        self.num_flows = int(num_flows) if self.use_flows else 0
        if reduced_dim is not None:
            self.dim_reducer = nn.Linear(input_dim, reduced_dim)
            embed_input = reduced_dim
            out_output = reduced_dim
        else:
            self.dim_reducer = None
            embed_input = input_dim
            out_output = input_dim
        
        self.embed = nn.Sequential(
            nn.Linear(embed_input, latent_dim),
            nn.LayerNorm(latent_dim),  # Add layer normalization
            nn.GELU()  # Use GELU activation function
        )
        
        self.encoder_layers = nn.ModuleList(
            [ISAB(latent_dim, heads, m) for _ in range(levels)]
        )
        self.decoder_layers = nn.ModuleList(
            [AttentiveBottleneckLayer(latent_dim, heads, m) for _ in range(levels)]
        )
        
        # Improved mu and logvar networks
        self.mu_logvar = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim * 2),
        )
        
        self.out = nn.Sequential(
            nn.Linear(latent_dim, out_output)
        )

        # Optional: planar flow stack for posterior transform
        if self.use_flows and self.num_flows > 0:
            self.flows = nn.ModuleList([_PlanarFlow(latent_dim) for _ in range(self.num_flows)])
        else:
            self.flows = nn.ModuleList()

        # Improved initialization
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x):
        embeds = self.embed(x)
        current = embeds
        z_list = []
        for layer in self.encoder_layers:
            # Add residual connection
            residual = current
            current = layer(current)
            current = current + residual  # Residual connection
            
            # Aggregate and generate latent variables
            agg = current.mean(dim=1, keepdim=True)
            mu_logvar = self.mu_logvar(agg)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            
            # Clamp logvar range to prevent numerical instability
            logvar = torch.clamp(logvar, min=-10, max=10)
            std = torch.exp(0.5 * logvar)
            
            # Optional std augmentation (disabled for truthful SetVAE-only)
            if self.training and self.enable_posterior_std_augmentation:
                eps = torch.randn_like(std) * 0.1
                std = std + eps.abs()
                
            dist = Normal(mu, std)
            z_sampled = dist.rsample()
            z_list.append((z_sampled, mu, logvar))
        return (z_list, current)

    def encode_from_var_val(self, var, val):
        """
        Encode directly from raw (var, val) without decoding.
        This mirrors the preprocessing in forward() but stops after encode().
        Returns (z_list, encoded).
        """
        if self.dim_reducer is not None:
            reduced = self.dim_reducer(var)
        else:
            reduced = var
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        reduced_normalized = reduced / (norms + 1e-8)
        x = reduced_normalized * val
        if self.training:
            x = F.dropout(x, p=0.05, training=True)
        return self.encode(x)

    def decode(self, z_list, target_n, use_mean=False, noise_std=0.5):
        idx = 1 if use_mean else 0
        current = z_list[-1][idx]
        # Apply optional normalizing flows to the last-layer latent sample before decoding
        if self.use_flows and len(self.flows) > 0 and idx == 0:
            current, _ = self.apply_flows(current)
        for l in range(self.levels - 1, -1, -1):
            # Add residual connection
            residual = current
            layer_out = self.decoder_layers[l](current, target_n, noise_std=noise_std)
            current = layer_out + residual.expand_as(layer_out)
        recon = self.out(current)
        return recon

    # --- Flow helpers ---
    def apply_flows(self, z: torch.Tensor):
        """
        Apply the planar flow stack to latent sample z.
        Args:
            z: [batch, 1, latent_dim]
        Returns:
            z_k: flowed latent with same shape
            sum_logdet: [batch] log-determinant sums (useful if one wants exact flow KL later)
        """
        if not self.use_flows or len(self.flows) == 0:
            batch = z.size(0)
            return z, torch.zeros(batch, device=z.device, dtype=z.dtype)
        sum_logdet = 0.0
        z_k = z
        for flow in self.flows:
            z_k, logdet = flow(z_k)
            sum_logdet = sum_logdet + logdet
        return z_k, sum_logdet

    def forward(self, var, val):
        if self.dim_reducer is not None:
            reduced = self.dim_reducer(var)
        else:
            reduced = var
        
        # Improved normalization method
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        # Add small random noise to prevent division by zero
        norms = norms + torch.randn_like(norms) * 1e-8
        reduced_normalized = reduced / (norms + 1e-8)
        x = reduced_normalized * val
        
        # Add input dropout
        if self.training:
            x = F.dropout(x, p=0.05, training=True)
            
        z_list, encoded = self.encode(x)
        target_n = x.size(1)
        recon = self.decode(z_list, target_n)
        return recon, z_list, encoded


# Loss functions


def recon_loss(
    recon,
    target,
    alpha=1.0,
    beta=1.0,
    gamma=3.0,
    beta_var=0.0,
    epsilon=1e-8,
):
    """
    Reconstruction loss for unordered sets, using Chamfer distances for direction, magnitude, and full vectors.
    This ensures permutation-invariant matching, where each target direction/magnitude is covered by the closest recon point.

    Args:
        recon: Reconstructed events [batch, n, dim]
        target: Target events [batch, n, dim]
        alpha: Weight for direction loss
        beta: Weight for magnitude loss
        gamma: Weight for full Chamfer loss
        beta_var: Weight for variance regularization (encourages higher recon variance)
        epsilon: Small value to avoid division by zero

    Returns:
        total_loss: Scalar tensor
    """
    assert (
        recon.shape == target.shape
    ), f"Shape mismatch: recon {recon.shape}, target {target.shape}"

    # Normalize for directions
    recon_norm = torch.norm(recon, dim=-1, keepdim=True)
    target_norm = torch.norm(target, dim=-1, keepdim=True)
    recon_unit = recon / (recon_norm + epsilon)
    target_unit = target / (target_norm + epsilon)

    # Direction Chamfer
    sim_matrix = torch.bmm(recon_unit, target_unit.transpose(1, 2))
    dissim_matrix = 1 - sim_matrix
    min_dissim_recon = torch.min(dissim_matrix, dim=2)[0].mean(dim=1)
    min_dissim_target = torch.min(dissim_matrix, dim=1)[0].mean(dim=1)
    dir_chamfer = torch.mean(min_dissim_recon + min_dissim_target) / 2

    # Magnitude Chamfer (absolute difference)
    mag_distances = torch.abs(
        recon_norm.unsqueeze(2) - target_norm.unsqueeze(1)
    ).squeeze(-1)
    min_mag_recon = torch.min(mag_distances, dim=2)[0].mean(dim=1)
    min_mag_target = torch.min(mag_distances, dim=1)[0].mean(dim=1)
    mag_chamfer = torch.mean(min_mag_recon + min_mag_target) / 2

    # Full Chamfer (L2 on vectors, for global position constraint)
    recon_exp = recon.unsqueeze(2)
    target_exp = target.unsqueeze(1)
    distances = torch.sum((recon_exp - target_exp) ** 2, dim=-1)
    min_dist_recon_to_target = torch.min(distances, dim=2)[0].mean(dim=1)
    min_dist_target_to_recon = torch.min(distances, dim=1)[0].mean(dim=1)
    chamfer_loss = torch.mean(min_dist_recon_to_target + min_dist_target_to_recon) / 2

    # Variance regularization: Encourage higher variance in recon (negative to minimize)
    var_term = (
        -beta_var * torch.var(recon, dim=1, unbiased=False).mean(dim=-1).mean()
    )

    # Total loss
    total_loss = (
        alpha * dir_chamfer + beta * mag_chamfer + gamma * chamfer_loss + var_term
    )
    return total_loss


def elbo_loss(
    recon,
    target,
    z_list,
    free_bits: float = 0.1,
    recon_alpha: float = 1.0,
    recon_beta: float = 1.0,
    recon_gamma: float = 3.0,
    beta_var: float = 0.0,
):
    """
    ELBO with per-dimension free-bits and configurable reconstruction weights.
    - free_bits is applied per latent dimension before summation.
    - beta_var encourages higher reconstruction variance (mitigates variance collapse).
    """
    r_loss = recon_loss(
        recon,
        target,
        alpha=recon_alpha,
        beta=recon_beta,
        gamma=recon_gamma,
        beta_var=beta_var,
    )
    kl_accum = 0.0
    for (_z, mu, logvar) in z_list:
        # Per-dimension KL for diagonal Gaussians
        per_dim_kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        # Apply per-dimension free-bits
        if free_bits is not None and free_bits > 0.0:
            per_dim_kl = torch.clamp(per_dim_kl, min=free_bits)
        # Sum over dimensions, then average over batch
        kl_div = torch.sum(per_dim_kl, dim=-1)
        kl_accum = kl_accum + kl_div.mean()
    total_kl = kl_accum / max(1, len(z_list))
    return r_loss, total_kl

