#!/usr/bin/env python3
"""
Gaussian-MIL focused ablation tests over cached per-set posteriors.

What this script does (all with time window = all):
- Builds a lightweight harness similar to exp/feasibility_tests.py
- Trains/evaluates a series of 10 Gaussian-MIL variants (small ablations)
- Reports AUROC/AUPRC/Recall@95%spec on valid/test (+ temp calibration)
- Saves JSON results under the cached feature directory

Inputs:
- --cached_dir: directory produced by exp/cache_features.py with train/valid/test

Notes:
- This file is self-contained; it re-implements a configurable Gaussian-MIL head
  so we can run gate/aggregation ablations without touching the main model file.
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


# --------------------------
# Metrics utilities
# --------------------------
def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    auroc = AUROC(task="binary")(logits.sigmoid(), labels)
    auprc = AveragePrecision(task="binary")(logits.sigmoid(), labels)
    rec95 = recall_at_specificity(logits, labels, target_spec=0.95)
    return {
        "auroc": float(auroc.item()),
        "auprc": float(auprc),
        "recall@95spec": float(rec95),
    }


def recall_at_specificity(logits: torch.Tensor, labels: torch.Tensor, target_spec: float = 0.95) -> float:
    # Approximate by scanning quantile thresholds
    logits = logits.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1).long()
    neg_mask = labels == 0
    if neg_mask.sum() == 0 or (~neg_mask).sum() == 0:
        return float("nan")
    # Scan thresholds between min and max
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
        # Fallback simple SGD
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
    # Temperature scaling
    T = calibrate_temperature(logits, labels)
    base_cal = evaluate_logits(logits / T, labels)
    return base, base_cal


# --------------------------
# Gaussian-MIL variant head for ablations
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
        # Three summary features per set: ||mu||/sqrt(D), mean(logvar), mean(mu^2/var)
        B, S, D = mu.shape
        var = logvar.exp().clamp(min=1e-6)
        snr1 = mu.norm(dim=-1) / math.sqrt(D)  # [B, S]
        snr2 = logvar.mean(dim=-1)  # [B, S]
        snr3 = (mu.pow(2) / var).mean(dim=-1)  # [B, S]
        return torch.stack([snr1, snr2, snr3], dim=-1)  # [B, S, 3]

    def _build_gate_inputs(self, mu: torch.Tensor, logvar: torch.Tensor, s: torch.Tensor, minutes: Optional[torch.Tensor]) -> torch.Tensor:
        feats = [mu, logvar, s.unsqueeze(-1)]
        if self.use_time and minutes is not None:
            feats.append(minutes.unsqueeze(-1))
        if self.snr_features:
            feats.append(self._snr_bundle(mu, logvar))
        gate_in = torch.cat(feats, dim=-1)  # [B, S, 2D+1(+extras)]
        if self.deepsets_ctx:
            x = torch.cat([mu, logvar], dim=-1)
            h_i = self.phi(x)
            # Average over sets (valid positions assumed non-zero from caller)
            h_bar = h_i.mean(dim=1, keepdim=True).expand_as(h_i)
            gate_in = torch.cat([gate_in, h_bar], dim=-1)
        return gate_in

    def _compute_gate_weights(self, gate_in: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Gate prediction
        a_raw = torch.sigmoid(self.gate(gate_in)).squeeze(-1)  # [B, S]
        # Optional precision attention mixture
        if self.precision_attention:
            var = logvar.exp().clamp(min=1e-6)
            precision = 1.0 / var
            score = (mu.pow(2) * precision).sum(dim=-1)  # [B, S]
            a_prec = torch.softmax(score, dim=1)  # normalized across S
            a_mlp = a_raw / a_raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
            lam = torch.sigmoid(self.attn_lambda_logit)
            a = lam * a_mlp + (1.0 - lam) * a_prec
        else:
            a = a_raw / a_raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return a  # normalized weights

    def _apply_time_decay(self, minutes: Optional[torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        if (not self.time_decay) or (minutes is None):
            return a
        # Exponential decay based on hours since last event
        last_time = minutes.max(dim=1, keepdim=True).values  # [B,1]
        delta_h = (last_time - minutes).clamp(min=0.0) / 60.0
        rate = F.softplus(self.decay_rate_log)  # positive
        decay = torch.exp(-rate * delta_h)  # [B, S]
        w = a * decay
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return w

    def group_lasso_penalty(self) -> torch.Tensor:
        # Group lasso across input dimensions on first gate layer
        first = self.gate[0]
        if isinstance(first, PosLinear):
            W = F.softplus(first.W)
        else:
            W = first.weight  # type: ignore[attr-defined]
        # Sum of L2 norms of columns
        return (W.pow(2).sum(dim=0).sqrt()).sum()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, minutes: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.per_set_logit(mu, logvar)  # [B, S]
        gate_in = self._build_gate_inputs(mu, logvar, s, minutes)
        a = self._compute_gate_weights(gate_in, mu, logvar)  # normalized across S
        a = self._apply_time_decay(minutes, a)

        # Optional top-k by gate weight
        if self.topk_k is not None and self.topk_k > 0:
            k = min(self.topk_k, a.size(1))
            topk_vals, topk_idx = torch.topk(a, k=k, dim=1)
            mask = torch.zeros_like(a)
            mask.scatter_(1, topk_idx, 1.0)
            a = a * mask
            a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Aggregators
        if self.aggregator == "mean":
            s_main = (a * s).sum(dim=1)  # weighted mean as weights are normalized
        elif self.aggregator == "lse":
            s_main = torch.logsumexp(s + torch.log(a.clamp(min=1e-6)), dim=1)
        elif self.aggregator == "pmean":
            p = torch.exp(self.p_log).clamp(min=1e-3, max=50.0)
            prob = torch.sigmoid(s)
            m_p = (a * (prob.pow(p))).sum(dim=1).clamp(min=1e-6) ** (1.0 / p)
            s_main = torch.logit(m_p.clamp(1e-6, 1 - 1e-6))
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        # Noisy-OR branch
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

        # Return binary logit [B]; s and a for analysis
        return s_total, s, a


# --------------------------
# Experiment runner
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pos_weight", type=float, default=9.0)
    parser.add_argument("--epochs_mil", type=int, default=8)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--group_lasso_lambda", type=float, default=0.0)
    parser.add_argument("--spec_hinge_lambda", type=float, default=0.0)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=getattr(config, "focal_alpha", 0.35))
    parser.add_argument("--focal_gamma", type=float, default=getattr(config, "focal_gamma", 2.0))
    parser.add_argument("--progress_file", type=str, default=None, help="Write JSONL progress records to this path")
    parser.add_argument("--no_partial", action="store_true", help="Disable partial results saving after each ablation")
    args = parser.parse_args()

    device = torch.device(args.device)

    def build_loader(split):
        ds = CachedSeqDataset(os.path.join(args.cached_dir, split))
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    train_loader = build_loader("train")
    valid_loader = build_loader("valid")
    test_loader = build_loader("test")

    D = config.latent_dim

    # Define the 10 ablations to run (all time window = all)
    ablations = [
        ("baseline_mean_or", {"aggregator": "mean", "mix_with_or": True}),
        ("lse_or", {"aggregator": "lse", "mix_with_or": True}),
        ("pmean_or", {"aggregator": "pmean", "mix_with_or": True}),
        ("topk2_mean_or", {"aggregator": "mean", "mix_with_or": True, "topk_k": 2}),
        ("topk3_mean_or", {"aggregator": "mean", "mix_with_or": True, "topk_k": 3}),
        ("precision_attention", {"aggregator": "mean", "mix_with_or": True, "precision_attention": True}),
        ("snr_features", {"aggregator": "mean", "mix_with_or": True, "snr_features": True}),
        ("group_lasso", {"aggregator": "mean", "mix_with_or": True}, True),  # marks to apply group lasso lambda
        ("deepsets_ctx", {"aggregator": "mean", "mix_with_or": True, "deepsets_ctx": True}),
        ("focal_bce_hinge", {"aggregator": "mean", "mix_with_or": True}, False, True),  # focal + hinge
    ]
    # Each tuple is (name, model_kwargs[, use_group_lasso][, use_hinge]) with optional flags

    results: Dict[str, Dict[str, float]] = {}

    # Progress logging helper (JSONL)
    progress_fp = None
    if args.progress_file is None:
        progress_path = os.path.join(args.cached_dir, "gaussianmil_progress.jsonl")
    else:
        progress_path = args.progress_file
    try:
        progress_fp = open(progress_path, "a", buffering=1)
    except Exception:
        progress_fp = None

    def log_progress(event: str, payload: Dict):
        rec = {"event": event, **payload}
        if progress_fp is not None:
            try:
                progress_fp.write(json.dumps(rec) + "\n")
            except Exception:
                pass

    def make_model(model_kwargs: Dict) -> GaussianMILVariant:
        model = GaussianMILVariant(
            latent_dim=D,
            use_time=True,
            gate_hidden_dim=128,
            **model_kwargs,
        ).to(device)
        return model

    def compute_spec_hinge(logits: torch.Tensor, labels: torch.Tensor, target_spec: float = 0.95) -> torch.Tensor:
        # Small hinge penalty on negatives above threshold achieving ~target specificity (batch-wise approx)
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

    # Loss selector
    def build_loss():
        if args.use_focal:
            from losses import FocalLoss  # local import to avoid unused on CPU

            alpha = args.focal_alpha if args.focal_alpha is not None else 0.35
            return FocalLoss(alpha=alpha, gamma=args.focal_gamma, reduction="mean")
        else:
            return None  # use BCEWithLogits with pos_weight

    # Train/eval helper for one configuration and one seed
    def run_one(name: str, model_kwargs: Dict, seed: int, use_group_lasso_flag: bool = False, use_hinge_flag: bool = False):
        torch.manual_seed(seed)
        model = make_model(model_kwargs)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        focal = build_loss()
        pos_weight_tensor = torch.tensor(args.pos_weight, device=device)

        for epoch_idx in range(1, args.epochs_mil + 1):
            model.train()
            pbar = tqdm(
                enumerate(train_loader, 1),
                total=len(train_loader),
                desc=f"{name} epoch {epoch_idx}/{args.epochs_mil}",
                leave=False,
            )
            for batch_idx, batch in pbar:
                mu = batch["mu"].to(device)
                logvar = batch["logvar"].to(device)
                minutes = batch["minutes"].to(device)
                pm = batch["padding_mask"].to(device)
                y = batch["label"].float().to(device)

                # Mask padded positions to zeros for stability
                mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
                logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
                minutes = minutes.masked_fill(pm, 0.0)

                logits, s_per_set, gate_w = model(mu, logvar, minutes)
                logit_pos = logits  # [B]

                if focal is None:
                    loss = F.binary_cross_entropy_with_logits(logit_pos, y, pos_weight=pos_weight_tensor)
                else:
                    # Convert to 2-class logits for focal implementation if needed
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
            # epoch progress record
            log_progress(
                "train_epoch_end",
                {
                    "name": name,
                    "seed": seed,
                    "epoch": epoch_idx,
                    "loss": float(loss.item()),
                },
            )

        # Evaluation (valid/test) with calibrated and uncalibrated metrics
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
        log_progress(
            "eval_complete",
            {
                "name": name,
                "seed": seed,
                "valid": base,
                "valid_cal": cal,
                "test": tbase,
                "test_cal": tcal,
            },
        )
        return base, cal, tbase, tcal

    # Run all ablations across seeds and aggregate mean/std
    for entry in ablations:
        # Unpack optional flags
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

        def reduce_stats(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            keys = dict_list[0].keys()
            for k in keys:
                vals = torch.tensor([d[k] for d in dict_list], dtype=torch.float)
                out[f"{k}_mean"] = float(vals.mean().item())
                out[f"{k}_std"] = float(vals.std(unbiased=False).item())
            return out

        results[f"{name}_valid"] = reduce_stats(seed_results_valid)
        results[f"{name}_valid_cal"] = reduce_stats(seed_results_valid_cal)
        results[f"{name}_test"] = reduce_stats(seed_results_test)
        results[f"{name}_test_cal"] = reduce_stats(seed_results_test_cal)

        # Save partial results after each ablation for robustness
        if not args.no_partial:
            out_json_partial = os.path.join(args.cached_dir, "gaussianmil_results_partial.json")
            try:
                with open(out_json_partial, "w") as f:
                    json.dump(results, f, indent=2)
                log_progress("partial_saved", {"file": out_json_partial, "upto": name})
            except Exception:
                pass

    # Print concise report
    print("\n=== Gaussian-MIL Ablation Results ===")
    for k, v in results.items():
        auroc_m = v.get("auroc_mean", float("nan"))
        auprc_m = v.get("auprc_mean", float("nan"))
        rec_m = v.get("recall@95spec_mean", float("nan"))
        print(f"{k}: AUROC={auroc_m:.4f} (±{v.get('auroc_std', float('nan')):.4f}), "
              f"AUPRC={auprc_m:.4f} (±{v.get('auprc_std', float('nan')):.4f}), "
              f"R@95spec={rec_m:.4f} (±{v.get('recall@95spec_std', float('nan')):.4f})")

    out_json = os.path.join(args.cached_dir, "gaussianmil_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")

    # Ensure progress file gets closed
    if progress_fp is not None:
        try:
            progress_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

