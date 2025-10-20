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
    def __init__(self, dim, heads):
        super().__init__()
        # Prefer batch_first to avoid transposes; safe fallback for older Torch
        self._mha_batch_first = True
        try:
            self.multihead = nn.MultiheadAttention(dim, heads, batch_first=True)
        except TypeError:
            self.multihead = nn.MultiheadAttention(dim, heads)
            self._mha_batch_first = False
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        if self._mha_batch_first:
            attn_output, _ = self.multihead(q, kv, kv)
        else:
            q_t = q.transpose(0, 1)
            kv_t = kv.transpose(0, 1)
            attn_output, _ = self.multihead(q_t, kv_t, kv_t)
            attn_output = attn_output.transpose(0, 1)
        a = self.ln1(attn_output)
        ff_out = self.ff(a)
        return self.ln2(ff_out)


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
        self.mab = MAB(dim, heads)
        self.ff = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, target_n, noise_std=0.5):
        batch_size = z.size(0)
        seed = torch.randn(batch_size, target_n, z.size(-1), device=z.device)
        noise = torch.randn_like(seed) * noise_std
        seed = seed + noise
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
    def __init__(
        self,
        input_dim,
        reduced_dim,
        latent_dim,
        levels,
        heads,
        m,
        use_flows: bool = False,
        num_flows: int = 0,
        posterior_logvar_min: float = -2.5,
        posterior_logvar_max: float = 2.5,
        enable_posterior_std_augmentation: bool = False,
        posterior_std_aug_sigma: float = 0.0,
    ):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.levels = levels
        self.use_flows = bool(use_flows)
        self.num_flows = int(num_flows) if use_flows else 0
        # Posterior stability controls
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.enable_posterior_std_augmentation = bool(enable_posterior_std_augmentation)
        self.posterior_std_aug_sigma = float(posterior_std_aug_sigma)
        # Treat reduced_dim equal to input_dim (or non-positive/None) as "no reducer"
        rd_effective = None
        try:
            if reduced_dim is not None:
                rd_val = int(reduced_dim)
                if rd_val > 0 and rd_val != int(input_dim):
                    rd_effective = rd_val
        except Exception:
            rd_effective = None

        if rd_effective is not None:
            self.dim_reducer = nn.Linear(input_dim, rd_effective)
            embed_input = rd_effective
            out_output = rd_effective
        else:
            self.dim_reducer = None
            embed_input = input_dim
            out_output = input_dim

        self.embed = nn.Sequential(
            nn.Linear(embed_input, latent_dim),
            nn.LayerNorm(latent_dim),  # Add layer normalization
            nn.GELU(),  # Use GELU activation function
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

        # Decoder output head (no tanh limit to preserve amplitude capacity)
        self.out = nn.Linear(latent_dim, out_output)

        # Optional: simple planar flow stack for posterior transform
        # z_K = f_K \circ ... \circ f_1 (z_0)
        if self.use_flows and self.num_flows > 0:
            self.flows = nn.ModuleList(
                [_PlanarFlow(latent_dim) for _ in range(self.num_flows)]
            )
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

            # Clamp logvar range to prevent pathological variances
            logvar = torch.clamp(
                logvar,
                min=self.posterior_logvar_min,
                max=self.posterior_logvar_max,
            )
            std = torch.exp(0.5 * logvar)

            # Optional std augmentation for robustness (disabled by default)
            if (
                self.training
                and self.enable_posterior_std_augmentation
                and self.posterior_std_aug_sigma > 0.0
            ):
                std = std + self.posterior_std_aug_sigma * torch.randn_like(std).abs()

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
            x = F.dropout(x, p=0.1, training=True)
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
            # Inject corresponding encoder layer latent sample to encourage usage across layers
            inject_x = z_list[l][0] if l < len(z_list) else None
            layer_out = self.decoder_layers[l](
                current, target_n, x=inject_x, noise_std=noise_std
            )
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

        # Improved normalization method (deterministic epsilon for stability)
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        reduced_normalized = reduced / (norms + 1e-8)
        x = reduced_normalized * val

        # Add input dropout
        if self.training:
            x = F.dropout(x, p=0.1, training=True)

        z_list, encoded = self.encode(x)
        target_n = x.size(1)
        recon = self.decode(z_list, target_n)
        return recon, z_list, encoded


# Loss functions


def recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 3.0,
    beta_var: float = 0.1,
    scale_calib_weight: float = 0.0,
    epsilon: float = 1e-8,
    # Sinkhorn-OT options (方案A)
    use_sinkhorn: bool = False,
    sinkhorn_eps: float = 0.1,
    sinkhorn_iters: int = 100,
    # Compatibility no-ops for callers that pass extra arguments
    amp_weight: float = 1.0,
    base_unit: torch.Tensor | None = None,
    target_val: torch.Tensor | None = None,
):
    """
    Permutation-invariant reconstruction loss for sets.

    Two modes:
      - Chamfer (default, multi-to-one): original behavior using nearest-neighbor Chamfer
      - Sinkhorn-OT (soft one-to-one): entropic OT with cost = α·dir + β·mag + γ·vec

    Args:
      recon: [B, N, D]
      target: [B, N, D]
      alpha/beta/gamma: weights for direction/magnitude/vector components
      use_sinkhorn: enable differentiable soft assignment with Sinkhorn (方案A)
      sinkhorn_eps: entropic regularization strength (higher = softer)
      sinkhorn_iters: Sinkhorn iterations
    """
    assert recon.shape == target.shape, f"Shape mismatch: recon {recon.shape}, target {target.shape}"

    # Normalize for directions
    recon_norm = torch.norm(recon, dim=-1, keepdim=True)
    target_norm = torch.norm(target, dim=-1, keepdim=True)
    recon_unit = recon / (recon_norm + epsilon)
    target_unit = target / (target_norm + epsilon)

    if use_sinkhorn:
        # Run Sinkhorn in float32 for numerical stability under mixed precision.
        orig_dtype = recon.dtype
        recon32 = recon.to(torch.float32)
        target32 = target.to(torch.float32)

        # --- Build pairwise costs (float32) ---
        recon_norm32 = torch.norm(recon32, dim=-1, keepdim=True)
        target_norm32 = torch.norm(target32, dim=-1, keepdim=True)
        recon_unit32 = recon32 / (recon_norm32 + epsilon)
        target_unit32 = target32 / (target_norm32 + epsilon)

        # Direction (unsigned cosine dissimilarity)
        sim = torch.bmm(recon_unit32, target_unit32.transpose(1, 2)).abs()
        c_dir = 1.0 - sim  # [B,N,N]
        # Magnitude (absolute difference)
        mag_r = recon_norm32.squeeze(-1)  # [B,N]
        mag_t = target_norm32.squeeze(-1)  # [B,N]
        c_mag = (mag_r.unsqueeze(-1) - mag_t.unsqueeze(1)).abs()  # [B,N,N]
        # Vector (sign-invariant L2)
        r_exp = recon32.unsqueeze(2)  # [B,N,1,D]
        t_exp = target32.unsqueeze(1)  # [B,1,N,D]
        d_pos = ((r_exp - t_exp) ** 2).sum(dim=-1)
        d_neg = ((r_exp + t_exp) ** 2).sum(dim=-1)
        c_vec = torch.minimum(d_pos, d_neg)  # [B,N,N]
        # Combined cost
        C = alpha * c_dir + beta * c_mag + gamma * c_vec  # [B,N,N]

        # --- Sinkhorn-Knopp in float32 with stabilized exponent ---
        B, N, M = C.shape
        assert N == M, "Sinkhorn-OT currently expects square cost (same #tokens)"
        # Clamp exponent to avoid under/overflow in exp()
        inv_eps = 1.0 / max(1e-6, float(sinkhorn_eps))
        expo = (-C * inv_eps).clamp(min=-50.0, max=50.0)
        K = torch.exp(expo).clamp_min(1e-12)  # [B,N,N]
        # Uniform marginals a,b
        a = torch.full((B, N), 1.0 / float(N), device=C.device, dtype=C.dtype)
        b = torch.full((B, M), 1.0 / float(M), device=C.device, dtype=C.dtype)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(int(sinkhorn_iters)):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            u = a / Kv
            Ktu = (
                torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            )
            v = b / Ktu
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [B,N,N]
        ot_cost = (P * C).sum(dim=(1, 2)).mean()

        # If still unstable, fall back to Chamfer (rare)
        if not torch.isfinite(ot_cost):
            # Recompute Chamfer in float32 and return
            # Direction Chamfer (unsigned)
            sim_matrix = torch.bmm(recon_unit32, target_unit32.transpose(1, 2)).abs()
            dissim_matrix = 1 - sim_matrix
            min_dissim_recon = torch.min(dissim_matrix, dim=2)[0].mean(dim=1)
            min_dissim_target = torch.min(dissim_matrix, dim=1)[0].mean(dim=1)
            dir_chamfer = torch.mean(min_dissim_recon + min_dissim_target) / 2
            # Magnitude Chamfer (absolute difference)
            mag_distances = (
                recon_norm32.squeeze(-1).unsqueeze(2) - target_norm32.squeeze(-1).unsqueeze(1)
            ).abs()
            min_mag_recon = torch.min(mag_distances, dim=2)[0].mean(dim=1)
            min_mag_target = torch.min(mag_distances, dim=1)[0].mean(dim=1)
            mag_chamfer = torch.mean(min_mag_recon + min_mag_target) / 2
            # Full Chamfer (sign-invariant L2 on vectors)
            recon_exp = recon32.unsqueeze(2)
            target_exp = target32.unsqueeze(1)
            dist_pos = torch.sum((recon_exp - target_exp) ** 2, dim=-1)
            dist_neg = torch.sum((recon_exp + target_exp) ** 2, dim=-1)
            distances = torch.minimum(dist_pos, dist_neg)
            min_dist_recon_to_target = torch.min(distances, dim=2)[0].mean(dim=1)
            min_dist_target_to_recon = torch.min(distances, dim=1)[0].mean(dim=1)
            chamfer_loss = torch.mean(min_dist_recon_to_target + min_dist_target_to_recon) / 2
            var_term_f32 = -beta_var * torch.var(recon32, dim=1, unbiased=False).mean(dim=-1).mean()
            mean_recon_norm_f32 = torch.mean(recon_norm32 + 1e-8)
            mean_target_norm_f32 = torch.mean(target_norm32 + 1e-8)
            scale_ratio_f32 = mean_recon_norm_f32 / (mean_target_norm_f32 + 1e-8)
            scale_penalty_f32 = scale_calib_weight * (scale_ratio_f32 - 1.0) ** 2
            total_f32 = alpha * dir_chamfer + beta * mag_chamfer + gamma * chamfer_loss + var_term_f32 + scale_penalty_f32
            return total_f32.to(orig_dtype)

        # Variance regularization and scale penalty (float32)
        var_term = -beta_var * torch.var(recon32, dim=1, unbiased=False).mean(dim=-1).mean()
        mean_recon_norm = torch.mean(recon_norm32 + 1e-8)
        mean_target_norm = torch.mean(target_norm32 + 1e-8)
        scale_ratio = mean_recon_norm / (mean_target_norm + 1e-8)
        scale_penalty = scale_calib_weight * (scale_ratio - 1.0) ** 2
        return (ot_cost + var_term + scale_penalty).to(orig_dtype)

    # ---- Fallback: original Chamfer losses (multi-to-one) ----
    # Direction Chamfer (unsigned)
    sim_matrix = torch.bmm(recon_unit, target_unit.transpose(1, 2)).abs()
    dissim_matrix = 1 - sim_matrix
    min_dissim_recon = torch.min(dissim_matrix, dim=2)[0].mean(dim=1)
    min_dissim_target = torch.min(dissim_matrix, dim=1)[0].mean(dim=1)
    dir_chamfer = torch.mean(min_dissim_recon + min_dissim_target) / 2

    # Magnitude Chamfer (absolute difference)
    mag_distances = (recon_norm.squeeze(-1).unsqueeze(2) - target_norm.squeeze(-1).unsqueeze(1)).abs()
    min_mag_recon = torch.min(mag_distances, dim=2)[0].mean(dim=1)
    min_mag_target = torch.min(mag_distances, dim=1)[0].mean(dim=1)
    mag_chamfer = torch.mean(min_mag_recon + min_mag_target) / 2

    # Full Chamfer (sign-invariant L2 on vectors)
    recon_exp = recon.unsqueeze(2)
    target_exp = target.unsqueeze(1)
    dist_pos = torch.sum((recon_exp - target_exp) ** 2, dim=-1)
    dist_neg = torch.sum((recon_exp + target_exp) ** 2, dim=-1)
    distances = torch.minimum(dist_pos, dist_neg)
    min_dist_recon_to_target = torch.min(distances, dim=2)[0].mean(dim=1)
    min_dist_target_to_recon = torch.min(distances, dim=1)[0].mean(dim=1)
    chamfer_loss = torch.mean(min_dist_recon_to_target + min_dist_target_to_recon) / 2

    # Variance regularization
    var_term = -beta_var * torch.var(recon, dim=1, unbiased=False).mean(dim=-1).mean()
    # Scale calibration
    mean_recon_norm = torch.mean(recon_norm + 1e-8)
    mean_target_norm = torch.mean(target_norm + 1e-8)
    scale_ratio = mean_recon_norm / (mean_target_norm + 1e-8)
    scale_penalty = scale_calib_weight * (scale_ratio - 1.0) ** 2

    total_loss = alpha * dir_chamfer + beta * mag_chamfer + gamma * chamfer_loss + var_term + scale_penalty
    return total_loss


def elbo_loss(recon, target, z_list, free_bits=0.1):
    """
    Improved ELBO loss function to prevent posterior collapse
    """
    r_loss = recon_loss(recon, target)

    total_kl = 0
    latent_dim = z_list[0][1].size(-1)
    min_kl = free_bits * latent_dim

    for layer_idx, (z_sample, mu, logvar) in enumerate(z_list):
        # Standard KL divergence
        kl_div = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar, dim=-1)

        # Apply free bits
        kl_div = torch.clamp(kl_div, min=min_kl)

        # Add layer weights, deeper layers have higher KL loss weights
        layer_weight = (layer_idx + 1) / len(z_list)

        # Regularization term to prevent variance collapse
        var_reg = -0.1 * torch.mean(logvar)  # Encourage larger variance

        # Regularization term to prevent mean collapse
        mean_reg = 0.01 * torch.mean(mu.pow(2))  # Slightly penalize large means

        total_kl += layer_weight * (kl_div.mean() + var_reg + mean_reg)

    return r_loss, total_kl


# --------------------- Test functions for each module ----------------------------
def test_recon_loss():
    batch_size, n_x, dim = 1, 3, 3
    x = torch.randn(batch_size, n_x, dim)
    y = torch.randn(batch_size, n_x, dim)
    loss = recon_loss(x, y)
    print(
        f"Recon Loss Test - Input x: {x.shape}, y: {y.shape}, Output: {loss.shape}, Value: {loss.item()}"
    )
    assert loss.dim() == 0, "Recon loss should return a scalar"


def test_multihead_attention_block():
    batch_size, seq_len, dim, heads = 1, 3, 64, 4
    q = torch.randn(batch_size, seq_len, dim)
    kv = torch.randn(batch_size, seq_len, dim)
    mab = MAB(dim, heads)
    output = mab(q, kv)
    print(
        f"MultiheadAttentionBlock Test - Input q: {q.shape}, kv: {kv.shape}, Output: {output.shape}"
    )
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"


def test_induced_set_attention_block():
    batch_size, seq_len, dim, heads, m = 1, 3, 64, 4, 64
    x = torch.randn(batch_size, seq_len, dim)
    isab = ISAB(dim, heads, m)
    output = isab(x)
    print(f"InducedSetAttentionBlock Test - Input x: {x.shape}, Output: {output.shape}")
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"


def test_pooling_by_multihead_attention():
    batch_size, seq_len, dim, heads = 1, 3, 64, 4
    z = torch.randn(batch_size, seq_len, dim)
    pma = PoolingByMultiheadAttention(dim, heads)
    target_n = 4
    output = pma(z, target_n)
    print(
        f"PoolingByMultiheadAttention Test - Input z: {z.shape}, target_n: {target_n}, Output: {output.shape}"
    )
    assert output.shape == (batch_size, target_n, dim), "Output shape mismatch"


def test_attentive_bottleneck_layer():
    batch_size, seq_len, dim, heads, m = 1, 3, 64, 4, 64
    z_prev = torch.randn(batch_size, seq_len, dim)
    abl = AttentiveBottleneckLayer(dim, heads, m)
    target_n = 4
    output = abl(z_prev, target_n)
    print(
        f"AttentiveBottleneckLayer Test - Input z_prev: {z_prev.shape}, target_n: {target_n}, Output: {output.shape}"
    )
    assert output.shape == (batch_size, target_n, dim), "Output shape mismatch"


def test_setvae():
    batch_size, seq_len, input_dim, reduced_dim, latent_dim = 1, 3, 768, 256, 256
    var = torch.randn(batch_size, seq_len, input_dim)
    val = torch.randn(batch_size, seq_len, 1)
    model = SetVAEModule(
        input_dim=input_dim,
        reduced_dim=reduced_dim,
        latent_dim=latent_dim,
        levels=2,
        heads=2,
        m=16,
    )
    recon, z_list, _ = model(var, val)

    print(
        f"SetVAE Test - Input var: {var.shape}, Input val: {val.shape}, Output recon: {recon.shape}, z_list length: {len(z_list)}"
    )
    assert recon.shape == (
        batch_size,
        seq_len,
        reduced_dim,
    ), "Reconstruction shape mismatch"
    assert all(
        z[0].shape == (batch_size, 1, latent_dim) for z in z_list
    ), "Z_list shape mismatch"


def test_setvae_variable_n():
    batch_size = 1
    input_dim = 768
    reduced_dim = 256
    latent_dim = 64
    levels = 3
    heads = 4
    m = 32
    model = SetVAEModule(
        input_dim=input_dim,
        latent_dim=latent_dim,
        levels=levels,
        heads=heads,
        m=m,
        reduced_dim=reduced_dim,
    )

    for n in [5, 10, 20, 50]:
        var = torch.randn(batch_size, n, input_dim)
        val = torch.randn(batch_size, n, 1)
        recon, z_list, _ = model(var, val)
        print(
            f"Test with n={n} - Input var: {var.shape}, Input val: {val.shape}, Recon: {recon.shape}, z_list length: {len(z_list)}"
        )
        assert recon.shape == (batch_size, n, reduced_dim), "Shape mismatch"
        assert all(
            z[0].shape == (batch_size, 1, latent_dim) for z in z_list
        ), "z shape mismatch"
    print("Variable n test passed!")


if __name__ == "__main__":
    print("Running module tests...")
    test_recon_loss()
    test_multihead_attention_block()
    test_induced_set_attention_block()
    test_pooling_by_multihead_attention()
    test_attentive_bottleneck_layer()
    test_setvae()
    test_setvae_variable_n()
    print("All tests completed successfully!")
