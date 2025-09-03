from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import mask_last_index, time_encoding_sin


class LogisticProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def build_tokens(mu: torch.Tensor, logvar: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor, add_time: bool, time_dim: int = 8):
    # mu/logvar: [B,T,D], dt: [B,T,1], mask: [B,T]
    tokens = [mu, torch.log(torch.clamp(torch.exp(logvar), min=1e-8))]
    if add_time:
        phi = time_encoding_sin(dt, num_feats=time_dim)  # [B,T,2*time_dim]
        tokens.append(phi)
    return torch.cat(tokens, dim=-1)


class AttentionPoolingHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 1, use_time_decay: bool = False):
        super().__init__()
        self.use_time_decay = use_time_decay
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, dt: Optional[torch.Tensor] = None):
        # tokens: [B,T,H]
        h = self.mlp(tokens)
        score = self.score(h).squeeze(-1)  # [B,T]
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1)  # [B,T]
        pooled = torch.sum(alpha.unsqueeze(-1) * h, dim=1)
        logit = self.out(pooled).squeeze(-1)
        return logit


class ExpectationLogit(nn.Module):
    """Route B: Expected-logit linear model.
    a_t = w^T z_t + b; E[sigmoid(a_t)] ≈ sigma((w^T μ + b) / sqrt(1 + π/8 w^T Σ w)).
    We implement w, b and use a closed-form approximation with μ and Σ=diag(exp(logvar))
    to obtain p_t in the forward pass, then aggregate by averaging across time.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor):
        # mu/logvar: [B,T,D]
        var = torch.exp(logvar)
        # w^T μ + b
        a = (mu * self.w.view(1, 1, -1)).sum(-1) + self.b  # [B,T]
        # denom = sqrt(1 + pi/8 * w^T Σ w)
        quad = (var * (self.w.view(1, 1, -1) ** 2)).sum(-1)
        denom = torch.sqrt(1.0 + (3.1415926535 / 8.0) * quad)
        p_t = torch.sigmoid(a / denom)
        p_t = p_t * mask.float()
        denom2 = mask.float().sum(dim=1).clamp(min=1.0)
        p = p_t.sum(dim=1) / denom2
        logit = torch.log(p.clamp(min=1e-6)) - torch.log((1 - p).clamp(min=1e-6))
        return logit


class TimeWeightedPoE(nn.Module):
    """Route C: Time-weighted Product-of-Experts, followed by a small MLP/LogReg"""

    def __init__(self, dim: int, lambda_decay: float = 0.99, mlp_hidden: int = 128):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.head = nn.Sequential(nn.Linear(2 * dim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor):
        B, T, D = mu.shape
        t = torch.arange(T, device=mu.device).float()
        gamma = (self.lambda_decay ** (T - 1 - t)).view(1, T, 1)
        prec = torch.exp(-logvar) * gamma * mask.unsqueeze(-1).float()
        prec_sum = prec.sum(dim=1).clamp(min=1e-6)
        mu_star = (prec * mu).sum(dim=1) / prec_sum
        logvar_star = -torch.log(prec_sum)
        x = torch.cat([mu_star, logvar_star], dim=-1)
        return self.head(x).squeeze(-1)


class WassersteinBarycenter(nn.Module):
    """Route D: Approximate W2 barycenter per dimension (diagonal covariance approximation):
    μ* = mean(μ), σ* = mean(σ) then squared -> diagonal approximation.
    Followed by a LogReg/MLP head.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor):
        mask_f = mask.float()
        denom = mask_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        mu_bar = (mu * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        sigma = torch.sqrt(torch.exp(logvar))
        sigma_bar = (sigma * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        logvar_star = 2.0 * torch.log(sigma_bar.clamp(min=1e-6))
        x = torch.cat([mu_bar, logvar_star], dim=-1)
        return self.head(x).squeeze(-1)


class KMEPooling(nn.Module):
    """Route E: Kernel Mean Embedding + MLP."""

    def __init__(self, dim: int, kernel_scale: float = 1.0, hidden: int = 128):
        super().__init__()
        self.kernel_scale = kernel_scale
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor):
        # Use an RBF feature to approximate the mean embedding of μ:
        # phi(x) ~ exp(-||x||^2 / (2 s^2)) * x
        s2 = self.kernel_scale ** 2
        feat = torch.exp(-(mu ** 2).sum(dim=-1, keepdim=True) / (2 * s2)) * mu
        feat = feat * mask.unsqueeze(-1).float()
        denom = mask.float().sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        pooled = feat.sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


class ShallowSequenceModel(nn.Module):
    """Route F: 1–2 layer GRU/Transformer (lightweight)."""

    def __init__(self, token_dim: int, hidden: int = 64, num_layers: int = 1, use_gru: bool = True):
        super().__init__()
        self.use_gru = use_gru
        if use_gru:
            self.rnn = nn.GRU(token_dim, hidden, num_layers=num_layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=4, dim_feedforward=128, batch_first=True)
            self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.Linear(token_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if self.use_gru:
            lengths = mask.long().sum(dim=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(tokens, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, h = self.rnn(packed)
            _, h_n = out, h[-1]
            feat = h_n
        else:
            # simple attention mask
            key_mask = ~mask
            out = self.trans(tokens, src_key_padding_mask=key_mask)
            feat = out[:, 0]  # use first token as [CLS]-like
        return self.head(feat).squeeze(-1)

