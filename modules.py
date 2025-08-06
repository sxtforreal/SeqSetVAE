import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


# MAB - Multi-head Attention Block
class MAB(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.multihead = nn.MultiheadAttention(dim, heads)
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
    def __init__(self, input_dim, reduced_dim, latent_dim, levels, heads, m):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.levels = levels
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
            nn.Linear(latent_dim, out_output),
            nn.Tanh()  # Add output activation function to limit output range
        )

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
            
            # Add noise to prevent overfitting
            if self.training:
                eps = torch.randn_like(std) * 0.1
                std = std + eps.abs()
                
            dist = Normal(mu, std)
            z_sampled = dist.rsample()
            z_list.append((z_sampled, mu, logvar))
        return (z_list, current)

    def decode(self, z_list, target_n, use_mean=False, noise_std=0.5):
        idx = 1 if use_mean else 0
        current = z_list[-1][idx]
        for l in range(self.levels - 1, -1, -1):
            # Add residual connection
            residual = current
            layer_out = self.decoder_layers[l](current, target_n, noise_std=noise_std)
            current = layer_out + residual.expand_as(layer_out)
        recon = self.out(current)
        return recon

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
            x = F.dropout(x, p=0.1, training=True)
            
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
    beta_var=0.1,
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
    )  # Mean over dims, tokens, batch

    # Total loss
    total_loss = (
        alpha * dir_chamfer + beta * mag_chamfer + gamma * chamfer_loss + var_term
    )
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
