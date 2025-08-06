import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

# 添加谱归一化装饰器
def spectral_norm_layer(layer):
    """Apply spectral normalization to a layer"""
    return nn.utils.spectral_norm(layer)

# 后验坍缩诊断工具
class PosteriorCollapseMonitor:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.kl_values = []
        self.active_units = []
        self.mutual_info = []
    
    def update(self, z_list, reconstruction_loss):
        """更新后验坍缩监控指标"""
        total_kl = 0
        active_count = 0
        
        for z_sample, mu, logvar in z_list:
            # 计算每个维度的KL散度
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_mean = kl_per_dim.mean(dim=0)  # 平均每个维度
            
            # 统计活跃单元（KL > threshold的维度）
            active_dims = (kl_mean > self.threshold).sum().item()
            active_count += active_dims
            total_kl += kl_per_dim.sum()
        
        self.kl_values.append(total_kl.item())
        self.active_units.append(active_count)
        
        # 简单的互信息估计（基于重构损失和KL的比值）
        if reconstruction_loss > 0:
            mi_estimate = total_kl.item() / (reconstruction_loss + 1e-8)
            self.mutual_info.append(mi_estimate)
    
    def get_metrics(self):
        if not self.kl_values:
            return {}
        return {
            'avg_kl': sum(self.kl_values) / len(self.kl_values),
            'avg_active_units': sum(self.active_units) / len(self.active_units),
            'avg_mutual_info': sum(self.mutual_info) / len(self.mutual_info) if self.mutual_info else 0,
            'collapse_ratio': 1.0 - (sum(self.active_units) / len(self.active_units)) / (len(self.kl_values) * 256)  # 假设256维
        }

# 改进的β退火策略
class BetaScheduler:
    def __init__(self, strategy='cyclical', max_beta=1.0, min_beta=0.0, cycle_length=10000, warmup_steps=1000):
        self.strategy = strategy
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.cycle_length = cycle_length
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def get_beta(self):
        if self.strategy == 'linear':
            return self._linear_annealing()
        elif self.strategy == 'cyclical':
            return self._cyclical_annealing()
        elif self.strategy == 'sigmoid':
            return self._sigmoid_annealing()
        else:
            return self.max_beta
    
    def _linear_annealing(self):
        if self.step_count < self.warmup_steps:
            return self.min_beta + (self.max_beta - self.min_beta) * (self.step_count / self.warmup_steps)
        return self.max_beta
    
    def _cyclical_annealing(self):
        cycle_pos = (self.step_count % self.cycle_length) / self.cycle_length
        if cycle_pos < 0.5:
            # 前半周期：从min_beta增长到max_beta
            return self.min_beta + (self.max_beta - self.min_beta) * (cycle_pos * 2)
        else:
            # 后半周期：从max_beta降到min_beta
            return self.max_beta - (self.max_beta - self.min_beta) * ((cycle_pos - 0.5) * 2)
    
    def _sigmoid_annealing(self):
        # Sigmoid退火
        x = (self.step_count - self.warmup_steps / 2) / (self.warmup_steps / 4)
        sigmoid_val = 1 / (1 + math.exp(-x))
        return self.min_beta + (self.max_beta - self.min_beta) * sigmoid_val
    
    def step(self):
        self.step_count += 1


# MAB
class MAB(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.multihead = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        assert (
            q.dim() == 3 and kv.dim() == 3
        ), f"MAB input dims invalid: q={q.shape}, kv={kv.shape}"
        attn_output, _ = self.multihead(q, kv, kv)  # (B, N, D)
        h = self.ln1(q + attn_output)  # residual + norm
        return self.ln2(h + self.ff(h))


# ISAB
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


# SetVAE
class SetVAEModule(nn.Module):
    def __init__(self, input_dim, reduced_dim, latent_dim, levels, heads, m, use_spectral_norm=True):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.levels = levels
        self.use_spectral_norm = use_spectral_norm
        
        if reduced_dim is not None:
            self.dim_reducer = spectral_norm_layer(nn.Linear(input_dim, reduced_dim)) if use_spectral_norm else nn.Linear(input_dim, reduced_dim)
            embed_input = reduced_dim
            out_output = reduced_dim
        else:
            self.dim_reducer = None
            embed_input = input_dim
            out_output = input_dim
            
        self.embed = spectral_norm_layer(nn.Linear(embed_input, latent_dim)) if use_spectral_norm else nn.Linear(embed_input, latent_dim)
        self.encoder_layers = nn.ModuleList(
            [ISAB(latent_dim, heads, m) for _ in range(levels)]
        )
        self.decoder_layers = nn.ModuleList(
            [AttentiveBottleneckLayer(latent_dim, heads, m) for _ in range(levels)]
        )
        
        # 改进的μ和logvar网络，增加更多非线性
        self.mu_logvar = nn.Sequential(
            spectral_norm_layer(nn.Linear(latent_dim, latent_dim * 2)) if use_spectral_norm else nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            spectral_norm_layer(nn.Linear(latent_dim * 2, latent_dim * 2)) if use_spectral_norm else nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        
        self.out = spectral_norm_layer(nn.Linear(latent_dim, out_output)) if use_spectral_norm else nn.Linear(latent_dim, out_output)
        self.pma_agg = PoolingByMultiheadAttention(latent_dim, heads)

        # 改进的初始化策略
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化策略"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用He初始化对于GELU激活函数更合适
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def encode(self, x):
        embeds = self.embed(x)
        current = embeds
        z_list = []
        for layer in self.encoder_layers:
            layer_out = layer(current)
            current = layer_out + current + embeds
            agg = self.pma_agg(current, target_n=1).squeeze(1)
            mu_logvar = self.mu_logvar(agg)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            
            # 限制logvar的范围以避免数值不稳定
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            
            # 添加一个小的噪声来鼓励多样性
            std = torch.exp(0.5 * logvar) + 1e-6
            dist = Normal(mu, std)
            z_sampled = dist.rsample()
            z_list.append((z_sampled, mu, logvar))
        return (z_list, current)

    def decode(self, z_list, target_n, use_mean=False, noise_std=0.5):
        idx = 1 if use_mean else 0
        current = z_list[-1][idx].unsqueeze(1)  # [B, 1, D] instead of [B, D]
        for l in range(self.levels - 1, -1, -1):
            layer_out = self.decoder_layers[l](current, target_n, noise_std=noise_std)
            current = layer_out + current  # broadcasting [B, 1, D] over [B, N, D]
        recon = self.out(current)
        return recon

    def forward(self, var, val):
        if self.dim_reducer is not None:
            reduced = self.dim_reducer(var)
        else:
            reduced = var
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        reduced_normalized = reduced / (norms + 1e-8)
        x = reduced_normalized * val
        z_list, encoded = self.encode(x)
        target_n = x.size(1)
        recon = self.decode(z_list, target_n)
        return recon, z_list, encoded


# Loss function


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


def elbo_loss(recon, target, z_list, free_bits=0.1, beta=1.0, use_tc_decomposition=False):
    """
    改进的ELBO损失，包含多种后验坍缩防护措施
    
    Args:
        recon: 重构输出
        target: 目标输入
        z_list: 潜在变量列表
        free_bits: 自由位数
        beta: KL权重
        use_tc_decomposition: 是否使用Total Correlation分解
    """
    r_loss = recon_loss(recon, target)
    kl_total = 0
    latent_dim = z_list[0][1].size(-1)
    
    for level_idx, (z_sample, mu, logvar) in enumerate(z_list):
        # 标准KL散度
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Free bits策略 - 每个维度独立应用
        min_kl_per_dim = free_bits
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=min_kl_per_dim)
        
        # 层次化的KL权重（深层权重更大）
        level_weight = 1.0 + 0.1 * level_idx
        
        if use_tc_decomposition and z_sample.size(0) > 1:
            # Total Correlation分解
            kl_loss = beta_tc_vae_loss(z_sample, mu, logvar, beta=beta)
        else:
            kl_loss = kl_per_dim_clamped.sum(dim=-1).mean()
        
        kl_total += level_weight * kl_loss
    
    return r_loss, kl_total

def beta_tc_vae_loss(z, mu, logvar, beta=1.0, alpha=1.0, gamma=1.0):
    """
    β-TC-VAE损失，分解KL散度为三个部分：
    - Index-code MI: I(z; n)
    - Total Correlation: TC(z)  
    - Dimension-wise KL: ∑KL(q(zi|n)||p(zi))
    """
    batch_size = z.size(0)
    latent_dim = z.size(-1)
    
    # 标准KL散度
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    # 计算log q(z|x)
    log_qz_condx = gaussian_log_density(z, mu, logvar)
    
    # 计算log q(z) = log ∫ q(z|x)p(x)dx ≈ log (1/N ∑ q(z|xi))
    # 使用重要性采样近似
    _logqz = gaussian_log_density(
        z.view(batch_size, 1, latent_dim),
        mu.view(1, batch_size, latent_dim),
        logvar.view(1, batch_size, latent_dim)
    )
    logqz_prodmarginals = torch.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size)
    logqz = torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size)
    
    # 计算各个项
    index_code_mi = (log_qz_condx - logqz).mean()
    total_corr = (logqz - logqz_prodmarginals.sum(1)).mean()
    dim_wise_kl = kl_div.mean()
    
    # β-TC-VAE损失
    return alpha * index_code_mi + beta * total_corr + gamma * dim_wise_kl

def gaussian_log_density(samples, mu, logvar):
    """计算高斯分布的对数密度"""
    pi = torch.tensor(math.pi)
    normalization = -0.5 * (logvar + torch.log(2 * pi))
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((samples - mu) ** 2 * inv_var)
    return log_density


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
