#!/usr/bin/env python3
"""
Combined feasibility + Gaussian-MIL ablation tests over cached per-set posteriors.

What this script does (time window = all only):
- Reuses the feasibility harness: linear probes, pooled-MLP (mean/PoE/W2),
  closed-form Logistic–Gaussian on PoE, baseline Gaussian-MIL head, and
  light MC sampling + pooled MLP.
- Runs the focused Gaussian-MIL ablations (10 small variants) and aggregates
  metrics across seeds.
- Reports AUROC/AUPRC/Recall@95% specificity on valid/test (+ temperature calibration)
- Saves a single JSON with all results and an English guide describing tests.

Inputs:
- --cached_dir: directory produced by exp/cache_features.py with train/valid/test
"""

import os
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import AUROC, AveragePrecision
from tqdm import tqdm

import sys

# Ensure main module directory is importable when running from exp/
_MAIN_DIR = "/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/main"
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from model import GaussianMILHead  # baseline MIL head
import config


# --------------------------
# Data utilities
# --------------------------
class CachedSeqDataset(Dataset):
    def __init__(self, part_dir: str):
        manifest_path = os.path.join(part_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            self.file_map = json.load(f)
        self.paths = list(self.file_map.values())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obj = torch.load(self.paths[idx], map_location="cpu")
        return obj


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_s = max(item["mu"].shape[0] for item in batch)
    D = batch[0]["mu"].shape[1]
    B = len(batch)
    mu = torch.zeros(B, max_s, D)
    logvar = torch.zeros(B, max_s, D)
    minutes = torch.zeros(B, max_s)
    padding_mask = torch.ones(B, max_s, dtype=torch.bool)
    labels = torch.tensor([item.get("label", 0) for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        S = item["mu"].shape[0]
        mu[i, :S] = item["mu"]
        logvar[i, :S] = item["logvar"]
        minutes[i, :S] = item["minutes"]
        padding_mask[i, :S] = False
    return {
        "mu": mu,
        "logvar": logvar,
        "minutes": minutes,
        "padding_mask": padding_mask,
        "label": labels,
    }


def masked_select_last(mu: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = mu.shape
    last_indices = (~padding_mask).sum(dim=1).clamp(min=1) - 1
    idx = last_indices.view(-1, 1, 1).expand(-1, 1, D)
    gathered = mu.gather(dim=1, index=idx).squeeze(1)
    return gathered


def pool_mean(mu: torch.Tensor, logvar: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = (~padding_mask).float().unsqueeze(-1)
    mu_mean = (mu * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    logvar_mean = (logvar * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return mu_mean, logvar_mean


def pool_poe(mu: torch.Tensor, logvar: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    var = logvar.exp().clamp(min=1e-6)
    precision = 1.0 / var
    mask = (~padding_mask).float().unsqueeze(-1)
    precision = precision * mask
    tau_star = precision.sum(dim=1).clamp(min=1e-6)
    num = (precision * mu).sum(dim=1)
    mu_star = num / tau_star
    var_star = 1.0 / tau_star
    logvar_star = var_star.clamp(min=1e-8).log()
    return mu_star, logvar_star


def pool_w2(mu: torch.Tensor, logvar: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    var = logvar.exp().clamp(min=1e-6)
    sd = var.sqrt()
    mask = (~padding_mask).float().unsqueeze(-1)
    weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    mu_bar = (weights * mu).sum(dim=1)
    sd_bar = (weights * sd).sum(dim=1)
    var_bar = sd_bar.pow(2)
    logvar_bar = var_bar.clamp(min=1e-8).log()
    return mu_bar, logvar_bar


# --------------------------
# Heads and helpers
# --------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [256, 128], dropout: float = 0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LogisticGaussianHead(nn.Module):
    """Closed-form Logistic–Gaussian approximation: logit ≈ κ*(w^T μ + b)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(in_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = logvar.exp().clamp(min=1e-8)
        s = torch.matmul(mu, self.w) + self.b  # [B]
        v = torch.matmul(var, self.w.pow(2))  # [B]
        kappa = 1.0 / torch.sqrt(1.0 + (math.pi / 8.0) * v)
        return kappa * s


# --------------------------
# Metrics utilities (shared)
# --------------------------
def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    auroc = AUROC(task="binary")(logits.sigmoid(), labels)
    auprc = AveragePrecision(task="binary")(logits.sigmoid(), labels)
    rec95 = recall_at_specificity(logits, labels, target_spec=0.95)
    return {"auroc": float(auroc.item()), "auprc": float(auprc), "recall@95spec": float(rec95)}


def recall_at_specificity(logits: torch.Tensor, labels: torch.Tensor, target_spec: float = 0.95) -> float:
    logits = logits.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1).long()
    neg_mask = labels == 0
    if neg_mask.sum() == 0 or (~neg_mask).sum() == 0:
        return float("nan")
    min_l, max_l = float(logits.min()), float(logits.max())
    thresholds = torch.linspace(min_l, max_l, steps=200)
    best_rec = 0.0
    for thr in thresholds:
        preds = (logits >= thr).long()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        spec = tn / max(1, (tn + fp))
        if spec >= target_spec:
            rec = tp / max(1, (tp + fn))
            if rec > best_rec:
                best_rec = rec
    return best_rec


def calibrate_temperature(valid_logits: torch.Tensor, valid_labels: torch.Tensor, max_iter: int = 200, lr: float = 0.01) -> float:
    temp = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    y = valid_labels.float()

    def closure():
        opt.zero_grad()
        scaled = valid_logits / temp.clamp(min=1e-3)
        loss = F.binary_cross_entropy_with_logits(scaled, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        opt_sgd = torch.optim.SGD([temp], lr=lr)
        for _ in range(max_iter):
            opt_sgd.zero_grad()
            scaled = valid_logits / temp.clamp(min=1e-3)
            loss = F.binary_cross_entropy_with_logits(scaled, y)
            loss.backward()
            opt_sgd.step()
    return float(temp.detach().clamp(min=1e-3).item())


@torch.no_grad()
def eval_loader(model_fn, loader) -> Tuple[Dict[str, float], Dict[str, float]]:
    all_logits, all_labels = [], []
    for batch in loader:
        logits = model_fn(batch)
        labels = batch["label"].float()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).long()
    base = evaluate_logits(logits, labels)
    T = calibrate_temperature(logits, labels)
    base_cal = evaluate_logits(logits / T, labels)
    return base, base_cal


# --------------------------
# Gaussian-MIL Variant (ablation head)
# --------------------------
class PosLinear(nn.Module):
    """Linear layer with positive weights via softplus parameterization."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(out_features, in_features))
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.W), self.b)


class GaussianMILVariant(nn.Module):
    """
    Self-contained Gaussian discriminant head with configurable gate/aggregation.

    Differences vs model.GaussianMILHead:
    - Aggregators: mean, logsumexp (LSE), generalized power mean (learnable p)
    - Optional Noisy-OR mixture (alpha); optional top-k by gate
    - Gate options: precision-attention mixture, SNR features, DeepSets context,
      monotone gate (PosLinear), optional time decay multiplier
    - Group Lasso penalty on first gate layer (helper method)
    """

    def __init__(
        self,
        latent_dim: int,
        use_time: bool = True,
        gate_hidden_dim: int = 128,
        aggregator: str = "mean",  # mean | lse | pmean
        mix_with_or: bool = True,
        topk_k: Optional[int] = None,
        precision_attention: bool = False,
        snr_features: bool = False,
        deepsets_ctx: bool = False,
        monotone_gate: bool = False,
        time_decay: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_time = use_time
        self.aggregator = aggregator
        self.mix_with_or = mix_with_or
        self.topk_k = topk_k
        self.precision_attention = precision_attention
        self.snr_features = snr_features
        self.deepsets_ctx = deepsets_ctx
        self.monotone_gate = monotone_gate
        self.time_decay = time_decay

        # Gaussian class prototypes
        self.mu = nn.Parameter(torch.zeros(2, latent_dim))
        self.log_tau2 = nn.Parameter(torch.zeros(2, latent_dim))
        self.log_prior = nn.Parameter(torch.zeros(2))

        # DeepSets context (phi) -> 64 dims
        self.ctx_dim = 0
        if self.deepsets_ctx:
            self.phi = nn.Sequential(
                nn.Linear(2 * latent_dim, 64),
                nn.ReLU(),
            )
            self.ctx_dim = 64

        # Gate features: mu, logvar, s_i, optional time, optional SNR bundle (3 dims), optional ctx
        extra_dims = (1 if use_time else 0) + (3 if snr_features else 0) + self.ctx_dim
        gate_in_dim = 2 * latent_dim + 1 + extra_dims
        if monotone_gate:
            self.gate = nn.Sequential(
                PosLinear(gate_in_dim, gate_hidden_dim),
                nn.Softplus(),
                PosLinear(gate_hidden_dim, 1),
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )

        # Aggregation mix weights
        self.alpha_logits = nn.Parameter(torch.tensor(0.5).logit())  # mix with OR

        # Power-mean parameter p (initialize near arithmetic mean)
        self.p_log = nn.Parameter(torch.tensor(0.0))

        # Precision-attention interpolation (gate vs precision softmax)
        if self.precision_attention:
            self.attn_lambda_logit = nn.Parameter(torch.tensor(0.5).logit())

        # Time-decay rate (per hour)
        if self.time_decay:
            self.decay_rate_log = nn.Parameter(torch.tensor(0.1).log())

    @staticmethod
    def _expected_loglik(mu: torch.Tensor, logvar: torch.Tensor, mu_c: torch.Tensor, log_tau2_c: torch.Tensor) -> torch.Tensor:
        var = logvar.exp()
        tau2 = log_tau2_c.exp()
        diff2 = (mu - mu_c) ** 2
        term = (var + diff2) / tau2
        ell = -0.5 * (term + log_tau2_c).sum(dim=-1)
        return ell  # [B, S]

    def per_set_logit(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        ell1 = self._expected_loglik(mu, logvar, self.mu[1], self.log_tau2[1]) + self.log_prior[1]
        ell0 = self._expected_loglik(mu, logvar, self.mu[0], self.log_tau2[0]) + self.log_prior[0]
        s = ell1 - ell0
        return s  # [B, S]

    def _snr_bundle(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        B, S, D = mu.shape
        var = logvar.exp().clamp(min=1e-6)
        snr1 = mu.norm(dim=-1) / math.sqrt(D)
        snr2 = logvar.mean(dim=-1)
        snr3 = (mu.pow(2) / var).mean(dim=-1)
        return torch.stack([snr1, snr2, snr3], dim=-1)

    def _build_gate_inputs(self, mu: torch.Tensor, logvar: torch.Tensor, s: torch.Tensor, minutes: Optional[torch.Tensor]) -> torch.Tensor:
        feats = [mu, logvar, s.unsqueeze(-1)]
        if self.use_time and minutes is not None:
            feats.append(minutes.unsqueeze(-1))
        if self.snr_features:
            feats.append(self._snr_bundle(mu, logvar))
        gate_in = torch.cat(feats, dim=-1)
        if self.deepsets_ctx:
            x = torch.cat([mu, logvar], dim=-1)
            h_i = self.phi(x)
            h_bar = h_i.mean(dim=1, keepdim=True).expand_as(h_i)
            gate_in = torch.cat([gate_in, h_bar], dim=-1)
        return gate_in

    def _compute_gate_weights(self, gate_in: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        a_raw = torch.sigmoid(self.gate(gate_in)).squeeze(-1)
        if self.precision_attention:
            var = logvar.exp().clamp(min=1e-6)
            precision = 1.0 / var
            score = (mu.pow(2) * precision).sum(dim=-1)
            a_prec = torch.softmax(score, dim=1)
            a_mlp = a_raw / a_raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
            lam = torch.sigmoid(self.attn_lambda_logit)
            a = lam * a_mlp + (1.0 - lam) * a_prec
        else:
            a = a_raw / a_raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return a

    def _apply_time_decay(self, minutes: Optional[torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        if (not self.time_decay) or (minutes is None):
            return a
        last_time = minutes.max(dim=1, keepdim=True).values
        delta_h = (last_time - minutes).clamp(min=0.0) / 60.0
        rate = F.softplus(self.decay_rate_log)
        decay = torch.exp(-rate * delta_h)
        w = a * decay
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return w

    def group_lasso_penalty(self) -> torch.Tensor:
        first = self.gate[0]
        if isinstance(first, PosLinear):
            W = F.softplus(first.W)
        else:
            W = first.weight  # type: ignore[attr-defined]
        return (W.pow(2).sum(dim=0).sqrt()).sum()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, minutes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.per_set_logit(mu, logvar)
        gate_in = self._build_gate_inputs(mu, logvar, s, minutes)
        a = self._compute_gate_weights(gate_in, mu, logvar)
        a = self._apply_time_decay(minutes, a)

        if self.topk_k is not None and self.topk_k > 0:
            k = min(self.topk_k, a.size(1))
            topk_vals, topk_idx = torch.topk(a, k=k, dim=1)
            mask = torch.zeros_like(a)
            mask.scatter_(1, topk_idx, 1.0)
            a = a * mask
            a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-6)

        if self.aggregator == "mean":
            s_main = (a * s).sum(dim=1)
        elif self.aggregator == "lse":
            s_main = torch.logsumexp(s + torch.log(a.clamp(min=1e-6)), dim=1)
        elif self.aggregator == "pmean":
            p = torch.exp(self.p_log).clamp(min=1e-3, max=50.0)
            prob = torch.sigmoid(s)
            m_p = (a * (prob.pow(p))).sum(dim=1).clamp(min=1e-6) ** (1.0 / p)
            s_main = torch.logit(m_p.clamp(1e-6, 1 - 1e-6))
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        p_i = torch.sigmoid(s) * a
        p_i = p_i.clamp(1e-6, 1 - 1e-6)
        log1m = torch.log1p(-p_i).sum(dim=1)
        p_or = 1 - torch.exp(log1m)
        s_or = torch.logit(p_or.clamp(1e-6, 1 - 1e-6))

        if self.mix_with_or:
            alpha = torch.sigmoid(self.alpha_logits)
            s_total = alpha * s_main + (1.0 - alpha) * s_or
        else:
            s_total = s_main

        return s_total, s, a


# --------------------------
# Runner
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pos_weight", type=float, default=9.0)
    parser.add_argument("--epochs_linear", type=int, default=3)
    parser.add_argument("--epochs_mlp", type=int, default=5)
    parser.add_argument("--epochs_mil", type=int, default=8)
    parser.add_argument("--mc_k", type=int, default=4)
    # Ablation-specific
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--group_lasso_lambda", type=float, default=0.0)
    parser.add_argument("--spec_hinge_lambda", type=float, default=0.0)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=getattr(config, "focal_alpha", 0.35))
    parser.add_argument("--focal_gamma", type=float, default=getattr(config, "focal_gamma", 2.0))
    args = parser.parse_args()

    device = torch.device(args.device)

    def build_loader(split):
        ds = CachedSeqDataset(os.path.join(args.cached_dir, split))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=args.num_workers, collate_fn=collate_fn)

    train_loader = build_loader("train")
    valid_loader = build_loader("valid")
    test_loader = build_loader("test")

    latent_dim = config.latent_dim

    # Unified results container
    results: Dict[str, Dict[str, float]] = {}

    # --------------------------
    # A) Feasibility suite (time window = all)
    # --------------------------
    # 1) Linear probe: last mu
    lin = nn.Linear(latent_dim, 1).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=5e-3, weight_decay=1e-4)
    for epoch_idx in range(1, args.epochs_linear + 1):
        lin.train()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"linear_last_mu epoch {epoch_idx}/{args.epochs_linear}", leave=False)
        for batch_idx, batch in pbar:
            mu = batch["mu"].to(device)
            pm = batch["padding_mask"].to(device)
            y = batch["label"].float().to(device)
            x = masked_select_last(mu, pm)
            logit = lin(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")
    lin.eval()

    def lin_fn(b):
        mu = b["mu"].to(device)
        pm = b["padding_mask"].to(device)
        return lin(masked_select_last(mu, pm)).squeeze(-1)

    base, cal = eval_loader(lin_fn, valid_loader)
    tbase, tcal = eval_loader(lin_fn, test_loader)
    results["linear_last_mu_valid"] = base
    results["linear_last_mu_valid_cal"] = cal
    results["linear_last_mu_test"] = tbase
    results["linear_last_mu_test_cal"] = tcal

    # 2) Linear probe: last [mu||logvar]
    lin2 = nn.Linear(2 * latent_dim, 1).to(device)
    opt2 = torch.optim.AdamW(lin2.parameters(), lr=5e-3, weight_decay=1e-4)
    for epoch_idx in range(1, args.epochs_linear + 1):
        lin2.train()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"linear_last_mu_logvar epoch {epoch_idx}/{args.epochs_linear}", leave=False)
        for batch_idx, batch in pbar:
            mu = batch["mu"].to(device)
            logvar = batch["logvar"].to(device)
            pm = batch["padding_mask"].to(device)
            y = batch["label"].float().to(device)
            x = torch.cat([masked_select_last(mu, pm), masked_select_last(logvar, pm)], dim=-1)
            logit = lin2(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
            opt2.zero_grad(); loss.backward(); opt2.step()
            pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

    def lin2_fn(b):
        mu = b["mu"].to(device)
        logvar = b["logvar"].to(device)
        pm = b["padding_mask"].to(device)
        x = torch.cat([masked_select_last(mu, pm), masked_select_last(logvar, pm)], dim=-1)
        return lin2(x).squeeze(-1)

    base, cal = eval_loader(lin2_fn, valid_loader)
    tbase, tcal = eval_loader(lin2_fn, test_loader)
    results["linear_last_mu_logvar_valid"] = base
    results["linear_last_mu_logvar_valid_cal"] = cal
    results["linear_last_mu_logvar_test"] = tbase
    results["linear_last_mu_logvar_test_cal"] = tcal

    # 3) MLP over pooled features: mean/PoE/W2
    def run_mlp(pool_fn, name: str):
        mlp = MLPHead(in_dim=2 * latent_dim).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
        for epoch_idx in range(1, args.epochs_mlp + 1):
            mlp.train()
            pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"mlp_{name} epoch {epoch_idx}/{args.epochs_mlp}", leave=False)
            for batch_idx, batch in pbar:
                mu = batch["mu"].to(device)
                logvar = batch["logvar"].to(device)
                pm = batch["padding_mask"].to(device)
                y = batch["label"].float().to(device)
                mu_p, logvar_p = pool_fn(mu, logvar, pm)
                x = torch.cat([mu_p, logvar_p], dim=-1)
                logit = mlp(x)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

        def fn(b):
            mu = b["mu"].to(device)
            logvar = b["logvar"].to(device)
            pm = b["padding_mask"].to(device)
            mu_p, logvar_p = pool_fn(mu, logvar, pm)
            return mlp(torch.cat([mu_p, logvar_p], dim=-1))

        base, cal = eval_loader(fn, valid_loader)
        tbase, tcal = eval_loader(fn, test_loader)
        results[f"mlp_{name}_valid"] = base
        results[f"mlp_{name}_valid_cal"] = cal
        results[f"mlp_{name}_test"] = tbase
        results[f"mlp_{name}_test_cal"] = tcal

    run_mlp(pool_mean, "mean")
    run_mlp(pool_poe, "poe")
    run_mlp(pool_w2, "w2")

    # 4) Closed-form Logistic–Gaussian on PoE aggregate
    lg = LogisticGaussianHead(in_dim=latent_dim).to(device)
    opt_lg = torch.optim.AdamW(lg.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch_idx in range(1, args.epochs_mlp + 1):
        lg.train()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"logistic_gaussian_poe epoch {epoch_idx}/{args.epochs_mlp}", leave=False)
        for batch_idx, batch in pbar:
            mu = batch["mu"].to(device)
            logvar = batch["logvar"].to(device)
            pm = batch["padding_mask"].to(device)
            y = batch["label"].float().to(device)
            mu_p, logvar_p = pool_poe(mu, logvar, pm)
            logit = lg(mu_p, logvar_p)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
            opt_lg.zero_grad(); loss.backward(); opt_lg.step()
            pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

    def lg_fn(b):
        mu = b["mu"].to(device)
        logvar = b["logvar"].to(device)
        pm = b["padding_mask"].to(device)
        mu_p, logvar_p = pool_poe(mu, logvar, pm)
        return lg(mu_p, logvar_p)

    base, cal = eval_loader(lg_fn, valid_loader)
    tbase, tcal = eval_loader(lg_fn, test_loader)
    results["logistic_gaussian_poe_valid"] = base
    results["logistic_gaussian_poe_valid_cal"] = cal
    results["logistic_gaussian_poe_test"] = tbase
    results["logistic_gaussian_poe_test_cal"] = tcal

    # 5) Gaussian-MIL baseline head
    mil = GaussianMILHead(latent_dim=latent_dim, num_classes=2, use_time=True, gate_hidden_dim=128).to(device)
    opt_mil = torch.optim.AdamW(mil.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch_idx in range(1, args.epochs_mil + 1):
        mil.train()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"gaussian_mil epoch {epoch_idx}/{args.epochs_mil}", leave=False)
        for batch_idx, batch in pbar:
            mu = batch["mu"].to(device)
            logvar = batch["logvar"].to(device)
            minutes = batch["minutes"].to(device)
            pm = batch["padding_mask"].to(device)
            y = batch["label"].float().to(device)
            mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
            logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
            minutes = minutes.masked_fill(pm, 0.0)
            logits, _, _ = mil(mu, logvar, minutes)
            logit_pos = logits[:, 1]
            loss = F.binary_cross_entropy_with_logits(logit_pos, y, pos_weight=torch.tensor(args.pos_weight, device=device))
            opt_mil.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mil.parameters(), 1.0); opt_mil.step()
            pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

    def mil_fn(b):
        mu = b["mu"].to(device)
        logvar = b["logvar"].to(device)
        minutes = b["minutes"].to(device)
        pm = b["padding_mask"].to(device)
        mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
        logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
        minutes = minutes.masked_fill(pm, 0.0)
        logits, _, _ = mil(mu, logvar, minutes)
        return logits[:, 1]

    base, cal = eval_loader(mil_fn, valid_loader)
    tbase, tcal = eval_loader(mil_fn, test_loader)
    results["gaussian_mil_valid"] = base
    results["gaussian_mil_valid_cal"] = cal
    results["gaussian_mil_test"] = tbase
    results["gaussian_mil_test_cal"] = tcal

    # 6) Light MC sampling + mean pooling + MLP
    mlp_mc = MLPHead(in_dim=latent_dim).to(device)
    opt_mc = torch.optim.AdamW(mlp_mc.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch_idx in range(1, args.epochs_mlp + 1):
        mlp_mc.train()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"mlp_mc_mean epoch {epoch_idx}/{args.epochs_mlp}", leave=False)
        for batch_idx, batch in pbar:
            mu = batch["mu"].to(device)
            logvar = batch["logvar"].to(device)
            pm = batch["padding_mask"].to(device)
            y = batch["label"].float().to(device)
            var = logvar.exp().clamp(min=1e-6)
            mask = (~pm).float().unsqueeze(-1)
            logits_acc = 0.0
            for _k in range(args.mc_k):
                eps = torch.randn_like(mu)
                z = mu + (var.sqrt() * eps)
                z = z * mask
                z_pooled = z.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                logits_acc = logits_acc + mlp_mc(z_pooled)
            logit = logits_acc / float(args.mc_k)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
            opt_mc.zero_grad(); loss.backward(); opt_mc.step()
            pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

    def mlp_mc_fn(b):
        mu = b["mu"].to(device)
        logvar = b["logvar"].to(device)
        pm = b["padding_mask"].to(device)
        var = logvar.exp().clamp(min=1e-6)
        mask = (~pm).float().unsqueeze(-1)
        logits_acc = 0.0
        for _k in range(args.mc_k):
            eps = torch.randn_like(mu)
            z = mu + (var.sqrt() * eps)
            z = z * mask
            z_pooled = z.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            logits_acc = logits_acc + mlp_mc(z_pooled)
        return logits_acc / float(args.mc_k)

    base, cal = eval_loader(mlp_mc_fn, valid_loader)
    tbase, tcal = eval_loader(mlp_mc_fn, test_loader)
    results["mlp_mc_mean_valid"] = base
    results["mlp_mc_mean_valid_cal"] = cal
    results["mlp_mc_mean_test"] = tbase
    results["mlp_mc_mean_test_cal"] = tcal

    # --------------------------
    # B) Gaussian-MIL ablations (time window = all)
    # --------------------------
    ablations = [
        ("baseline_mean_or", {"aggregator": "mean", "mix_with_or": True}),
        ("lse_or", {"aggregator": "lse", "mix_with_or": True}),
        ("pmean_or", {"aggregator": "pmean", "mix_with_or": True}),
        ("topk2_mean_or", {"aggregator": "mean", "mix_with_or": True, "topk_k": 2}),
        ("topk3_mean_or", {"aggregator": "mean", "mix_with_or": True, "topk_k": 3}),
        ("precision_attention", {"aggregator": "mean", "mix_with_or": True, "precision_attention": True}),
        ("snr_features", {"aggregator": "mean", "mix_with_or": True, "snr_features": True}),
        ("group_lasso", {"aggregator": "mean", "mix_with_or": True}, True),
        ("deepsets_ctx", {"aggregator": "mean", "mix_with_or": True, "deepsets_ctx": True}),
        ("focal_bce_hinge", {"aggregator": "mean", "mix_with_or": True}, False, True),
    ]

    def make_model(model_kwargs: Dict) -> GaussianMILVariant:
        model = GaussianMILVariant(
            latent_dim=latent_dim,
            use_time=True,
            gate_hidden_dim=128,
            **model_kwargs,
        ).to(device)
        return model

    def compute_spec_hinge(logits: torch.Tensor, labels: torch.Tensor, target_spec: float = 0.95) -> torch.Tensor:
        with torch.no_grad():
            neg_logits = logits[labels == 0]
            if neg_logits.numel() == 0:
                thr = logits.new_tensor(float("inf"))
            else:
                k = max(1, int(math.ceil((1.0 - target_spec) * neg_logits.numel())))
                topk_vals, _ = torch.topk(neg_logits, k=k)
                thr = topk_vals.min()
        neg_mask = (labels == 0)
        hinge = F.relu(logits[neg_mask] - thr).mean() if neg_mask.any() else logits.new_tensor(0.0)
        return hinge

    def build_loss():
        if args.use_focal:
            from losses import FocalLoss  # optional dependency

            alpha = args.focal_alpha if args.focal_alpha is not None else 0.35
            return FocalLoss(alpha=alpha, gamma=args.focal_gamma, reduction="mean")
        else:
            return None

    def run_one(name: str, model_kwargs: Dict, seed: int, use_group_lasso_flag: bool = False, use_hinge_flag: bool = False):
        torch.manual_seed(seed)
        model = make_model(model_kwargs)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        focal = build_loss()
        pos_weight_tensor = torch.tensor(args.pos_weight, device=device)

        for epoch_idx in range(1, args.epochs_mil + 1):
            model.train()
            pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"{name} epoch {epoch_idx}/{args.epochs_mil}", leave=False)
            for batch_idx, batch in pbar:
                mu = batch["mu"].to(device)
                logvar = batch["logvar"].to(device)
                minutes = batch["minutes"].to(device)
                pm = batch["padding_mask"].to(device)
                y = batch["label"].float().to(device)

                mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
                logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
                minutes = minutes.masked_fill(pm, 0.0)

                logits, _, _ = model(mu, logvar, minutes)
                logit_pos = logits

                if focal is None:
                    loss = F.binary_cross_entropy_with_logits(logit_pos, y, pos_weight=pos_weight_tensor)
                else:
                    twoc = torch.stack([-logit_pos, logit_pos], dim=1)
                    loss = focal(twoc, y.long())

                if use_hinge_flag and args.spec_hinge_lambda > 0.0:
                    hinge = compute_spec_hinge(logit_pos.detach(), y)
                    loss = loss + args.spec_hinge_lambda * hinge

                if use_group_lasso_flag and args.group_lasso_lambda > 0.0:
                    gl = model.group_lasso_penalty()
                    loss = loss + args.group_lasso_lambda * gl

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                pbar.set_postfix(epoch=epoch_idx, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

        @torch.no_grad()
        def eval_fn(b):
            mu = b["mu"].to(device)
            logvar = b["logvar"].to(device)
            minutes = b["minutes"].to(device)
            pm = b["padding_mask"].to(device)
            mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
            logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
            minutes = minutes.masked_fill(pm, 0.0)
            logits, _, _ = model(mu, logvar, minutes)
            return logits

        base, cal = eval_loader(eval_fn, valid_loader)
        tbase, tcal = eval_loader(eval_fn, test_loader)
        return base, cal, tbase, tcal

    def reduce_stats(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(dict_list) == 0:
            return out
        keys = dict_list[0].keys()
        for k in keys:
            vals = torch.tensor([d[k] for d in dict_list], dtype=torch.float)
            out[f"{k}_mean"] = float(vals.mean().item())
            out[f"{k}_std"] = float(vals.std(unbiased=False).item())
        return out

    for entry in ablations:
        if len(entry) == 2:
            name, kwargs = entry
            use_group_lasso_flag = False
            use_hinge_flag = False
        elif len(entry) == 3:
            name, kwargs, use_group_lasso_flag = entry
            use_hinge_flag = False
        else:
            name, kwargs, use_group_lasso_flag, use_hinge_flag = entry

        seed_results_valid = []
        seed_results_valid_cal = []
        seed_results_test = []
        seed_results_test_cal = []
        for i in range(args.num_seeds):
            seed = args.seed_base + i
            b, c, tb, tc = run_one(name, kwargs, seed, use_group_lasso_flag, use_hinge_flag)
            seed_results_valid.append(b)
            seed_results_valid_cal.append(c)
            seed_results_test.append(tb)
            seed_results_test_cal.append(tc)

        results[f"ablation.{name}_valid"] = reduce_stats(seed_results_valid)
        results[f"ablation.{name}_valid_cal"] = reduce_stats(seed_results_valid_cal)
        results[f"ablation.{name}_test"] = reduce_stats(seed_results_test)
        results[f"ablation.{name}_test_cal"] = reduce_stats(seed_results_test_cal)

    # --------------------------
    # Save results and guide
    # --------------------------
    print("\n=== Combined Results (Feasibility + Gaussian-MIL Ablations) ===")
    for k, v in results.items():
        if isinstance(v, dict):
            auroc_k = next((vv for kk, vv in v.items() if kk.startswith("auroc")), float("nan"))
            auprc_k = next((vv for kk, vv in v.items() if kk.startswith("auprc")), float("nan"))
            rec_k = next((vv for kk, vv in v.items() if kk.startswith("recall@95spec")), float("nan"))
            print(f"{k}: AUROC={auroc_k:.4f}, AUPRC={auprc_k:.4f}, R@95spec={rec_k:.4f}")
        else:
            print(f"{k}: {v}")

    out_json = os.path.join(args.cached_dir, "combined_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")

    guide_path = os.path.join(args.cached_dir, "combined_tests_guide.md")
    guide_text = _build_english_guide()
    with open(guide_path, "w") as f:
        f.write(guide_text)
    print(f"Saved guide to {guide_path}")


def _build_english_guide() -> str:
    return (
        "# Combined Feasibility + Gaussian-MIL Ablation Tests\n\n"
        "This run evaluates only the full history (time window = all). No 24h/72h truncation is used.\n\n"
        "## Datasets and Inputs\n"
        "- Cached per-set posteriors produced by `exp/cache_features.py` under `train/valid/test`.\n"
        "- Each sample contains tensors: `mu [S,D]`, `logvar [S,D]`, `minutes [S]`, `label`, and a padding mask is derived at collate time.\n\n"
        "## Metrics\n"
        "- AUROC, AUPRC, and Recall at 95% specificity are reported.\n"
        "- We also perform temperature calibration on the validation split and report calibrated metrics.\n\n"
        "## Part A — Feasibility Suite (single pass, all-time)\n"
        "1. Linear probe (last mu):\n"
        "   - Selects the last non-padded time step's latent mean per sequence and fits a single linear layer.\n"
        "2. Linear probe (last [mu||logvar]):\n"
        "   - Concatenates last-step `mu` and `logvar` and trains a linear head.\n"
        "3. Pooled MLPs over per-set posteriors: mean / Product-of-Experts (PoE) / W2:\n"
        "   - Pools `mu` and `logvar` across sets via mean, precision-weighted PoE, or Wasserstein-2 style average; then feeds `[mu_pool||logvar_pool]` to a small MLP.\n"
        "4. Closed-form Logistic–Gaussian on PoE aggregate:\n"
        "   - Applies a MacKay-style scaling for a logistic head directly on PoE `(mu, logvar)` aggregates.\n"
        "5. Baseline Gaussian-MIL head:\n"
        "   - Uses the project’s `model.GaussianMILHead` with time as an input, trained with BCE-with-logits.\n"
        "6. Light Monte-Carlo (MC) sampling + mean pooling + MLP:\n"
        "   - Draws K samples from each per-set Gaussian, mean-pools sampled `z`, and feeds to an MLP; logits are averaged over K.\n\n"
        "## Part B — Gaussian-MIL Ablations (all-time)\n"
        "Each ablation modifies the Gaussian-MIL variant head while keeping the dataset and training loop fixed. Results are averaged across multiple seeds.\n\n"
        "- baseline_mean_or: Mean aggregation mixed with a Noisy-OR branch.\n"
        "- lse_or: LogSumExp aggregation mixed with Noisy-OR.\n"
        "- pmean_or: Learnable power-mean aggregation mixed with Noisy-OR.\n"
        "- topk2_mean_or / topk3_mean_or: Keep only the top-2/top-3 instances according to the gate, then normalize weights and aggregate with mean, mixed with Noisy-OR.\n"
        "- precision_attention: Interpolates gate weights with precision-based attention (`mu^2/var`).\n"
        "- snr_features: Augments gate inputs with three SNR-like features: `||mu||/sqrt(D)`, `mean(logvar)`, and `mean(mu^2/var)`.\n"
        "- group_lasso: Adds a group lasso penalty on the first gate layer’s input groups (controlled by `--group_lasso_lambda`).\n"
        "- deepsets_ctx: Adds a DeepSets context vector computed from per-instance `[mu||logvar]`.\n"
        "- focal_bce_hinge: Optionally swaps BCE for focal loss (controlled by flags) and adds a small hinge penalty nudging negatives below a batch-wise specificity threshold.\n\n"
        "## Outputs\n"
        "- JSON: `combined_results.json` aggregates all metrics for both Part A and Part B.\n"
        "- Guide: `combined_tests_guide.md` (this file) summarizes what was tested.\n"
    )


if __name__ == "__main__":
    main()

