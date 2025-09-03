#!/usr/bin/env python3
"""
Feasibility test harness over cached per-set posteriors.

Models compared:
- Linear probes: last-step mu; last-step [mu||logvar]
- MLP on pooled features with pooling rules: mean, PoE, W2
- Closed-form Logistic–Gaussian head on PoE aggregate (MacKay scaling)
- Gaussian-MIL head using per-set [mu, logvar, time]
- Light MC sampling + mean pooling + MLP (K small)

Extras:
- Time-window ablations (e.g., last 24h/72h/all)
- Metrics: AUROC, AUPRC, recall@95% specificity
- Optional temperature scaling on validation set

Inputs:
- --cached_dir: directory produced by cache_features.py with train/valid/test subdirs
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

from model import GaussianMILHead
import config


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


def apply_time_window(mu, logvar, minutes, padding_mask, hours: Optional[float]):
    if hours is None:
        return mu, logvar, minutes, padding_mask
    B, S = minutes.shape
    pm = padding_mask.clone()
    # last real time per sample
    lengths = (~pm).sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1)
    last_time = minutes.gather(dim=1, index=idx)
    cutoff = last_time - hours * 60.0
    keep = minutes >= cutoff
    pm = pm | (~keep)
    mu = mu.clone().masked_fill(pm.unsqueeze(-1), 0.0)
    logvar = logvar.clone().masked_fill(pm.unsqueeze(-1), 0.0)
    minutes = minutes.clone().masked_fill(pm, 0.0)
    return mu, logvar, minutes, pm


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
        v = torch.matmul(var, self.w.pow(2))   # [B]
        kappa = 1.0 / torch.sqrt(1.0 + (math.pi / 8.0) * v)
        return kappa * s


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    auroc = AUROC(task="binary")(logits.sigmoid(), labels)
    auprc = AveragePrecision(task="binary")(logits.sigmoid(), labels)
    rec95 = recall_at_specificity(logits, labels, target_spec=0.95)
    return {"auroc": float(auroc.item()), "auprc": float(auprc), "recall@95spec": float(rec95)}


def recall_at_specificity(logits: torch.Tensor, labels: torch.Tensor, target_spec: float = 0.95) -> float:
    # Approximate by scanning quantile thresholds
    logits = logits.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1).long()
    neg_mask = labels == 0
    if neg_mask.sum() == 0 or (~neg_mask).sum() == 0:
        return float('nan')
    # Scan 200 thresholds between min and max
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time_windows", type=int, nargs="*", default=[0, 24, 72], help="Hours; 0 means all history")
    parser.add_argument("--pos_weight", type=float, default=9.0)
    parser.add_argument("--epochs_linear", type=int, default=3)
    parser.add_argument("--epochs_mlp", type=int, default=5)
    parser.add_argument("--epochs_mil", type=int, default=8)
    parser.add_argument("--mc_k", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)

    def build_loader(split):
        ds = CachedSeqDataset(os.path.join(args.cached_dir, split))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=(split=="train"), num_workers=args.num_workers, collate_fn=collate_fn)

    train_loader = build_loader("train")
    valid_loader = build_loader("valid")
    test_loader = build_loader("test")

    D = config.latent_dim
    results = {}

    time_windows = [None if h == 0 else float(h) for h in args.time_windows]

    # Helpers to apply time window inside a model_fn
    def tw_apply(batch, hours: Optional[float]):
        mu = batch["mu"].to(device)
        logvar = batch["logvar"].to(device)
        minutes = batch["minutes"].to(device)
        pm = batch["padding_mask"].to(device)
        return apply_time_window(mu, logvar, minutes, pm, hours)

    # 1) Linear probe: last mu; last [mu||logvar]
    for hours in time_windows:
        tag = f"tw_{'all' if hours is None else int(hours)}h"

        # last mu
        lin = nn.Linear(D, 1).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=5e-3, weight_decay=1e-4)
        for _ in range(args.epochs_linear):
            lin.train()
            for batch in train_loader:
                mu, _, _, pm = tw_apply(batch, hours)
                y = batch["label"].float().to(device)
                x = masked_select_last(mu, pm)
                logit = lin(x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
                opt.zero_grad(); loss.backward(); opt.step()
        lin.eval()
        base, cal = eval_loader(lambda b: lin(masked_select_last(*tw_apply(b, hours)[:2:1],).squeeze(-1) if False else lin(masked_select_last(tw_apply(b, hours)[0], tw_apply(b, hours)[3])).squeeze(-1), valid_loader)  # placeholder, replaced below
        # above line was a placeholder; redefining cleanly:
        def lin_fn(b):
            mu, _, _, pm = tw_apply(b, hours)
            return lin(masked_select_last(mu, pm)).squeeze(-1)
        base, cal = eval_loader(lin_fn, valid_loader)
        tbase, tcal = eval_loader(lin_fn, test_loader)
        results[f"linear_last_mu_{tag}_valid"] = base
        results[f"linear_last_mu_{tag}_valid_cal"] = cal
        results[f"linear_last_mu_{tag}_test"] = tbase
        results[f"linear_last_mu_{tag}_test_cal"] = tcal

        # last [mu||logvar]
        lin2 = nn.Linear(2*D, 1).to(device)
        opt2 = torch.optim.AdamW(lin2.parameters(), lr=5e-3, weight_decay=1e-4)
        for _ in range(args.epochs_linear):
            lin2.train()
            for batch in train_loader:
                mu, logvar, _, pm = tw_apply(batch, hours)
                y = batch["label"].float().to(device)
                x = torch.cat([masked_select_last(mu, pm), masked_select_last(logvar, pm)], dim=-1)
                logit = lin2(x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
                opt2.zero_grad(); loss.backward(); opt2.step()
        def lin2_fn(b):
            mu, logvar, _, pm = tw_apply(b, hours)
            x = torch.cat([masked_select_last(mu, pm), masked_select_last(logvar, pm)], dim=-1)
            return lin2(x).squeeze(-1)
        base, cal = eval_loader(lin2_fn, valid_loader)
        tbase, tcal = eval_loader(lin2_fn, test_loader)
        results[f"linear_last_mu_logvar_{tag}_valid"] = base
        results[f"linear_last_mu_logvar_{tag}_valid_cal"] = cal
        results[f"linear_last_mu_logvar_{tag}_test"] = tbase
        results[f"linear_last_mu_logvar_{tag}_test_cal"] = tcal

        # 2) MLP over pooled features: mean/PoE/W2
        def run_mlp(pool_fn, name: str):
            mlp = MLPHead(in_dim=2*D).to(device)
            opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
            for _ in range(args.epochs_mlp):
                mlp.train()
                for batch in train_loader:
                    mu, logvar, _, pm = tw_apply(batch, hours)
                    y = batch["label"].float().to(device)
                    mu_p, logvar_p = pool_fn(mu, logvar, pm)
                    x = torch.cat([mu_p, logvar_p], dim=-1)
                    logit = mlp(x)
                    loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
                    opt.zero_grad(); loss.backward(); opt.step()
            def fn(b):
                mu, logvar, _, pm = tw_apply(b, hours)
                mu_p, logvar_p = pool_fn(mu, logvar, pm)
                return mlp(torch.cat([mu_p, logvar_p], dim=-1))
            base, cal = eval_loader(fn, valid_loader)
            tbase, tcal = eval_loader(fn, test_loader)
            results[f"mlp_{name}_{tag}_valid"] = base
            results[f"mlp_{name}_{tag}_valid_cal"] = cal
            results[f"mlp_{name}_{tag}_test"] = tbase
            results[f"mlp_{name}_{tag}_test_cal"] = tcal

        run_mlp(pool_mean, "mean")
        run_mlp(pool_poe, "poe")
        run_mlp(pool_w2, "w2")

        # 3) Closed-form Logistic–Gaussian on PoE aggregate
        lg = LogisticGaussianHead(in_dim=D).to(device)
        opt_lg = torch.optim.AdamW(lg.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(args.epochs_mlp):
            lg.train()
            for batch in train_loader:
                mu, logvar, _, pm = tw_apply(batch, hours)
                y = batch["label"].float().to(device)
                mu_p, logvar_p = pool_poe(mu, logvar, pm)
                logit = lg(mu_p, logvar_p)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(args.pos_weight, device=device))
                opt_lg.zero_grad(); loss.backward(); opt_lg.step()
        def lg_fn(b):
            mu, logvar, _, pm = tw_apply(b, hours)
            mu_p, logvar_p = pool_poe(mu, logvar, pm)
            return lg(mu_p, logvar_p)
        base, cal = eval_loader(lg_fn, valid_loader)
        tbase, tcal = eval_loader(lg_fn, test_loader)
        results[f"logistic_gaussian_poe_{tag}_valid"] = base
        results[f"logistic_gaussian_poe_{tag}_valid_cal"] = cal
        results[f"logistic_gaussian_poe_{tag}_test"] = tbase
        results[f"logistic_gaussian_poe_{tag}_test_cal"] = tcal

        # 4) Gaussian-MIL head
        mil = GaussianMILHead(latent_dim=D, num_classes=2, use_time=True, gate_hidden_dim=128).to(device)
        opt_mil = torch.optim.AdamW(mil.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(args.epochs_mil):
            mil.train()
            for batch in train_loader:
                mu, logvar, minutes, pm = tw_apply(batch, hours)
                y = batch["label"].float().to(device)
                mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
                logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
                minutes = minutes.masked_fill(pm, 0.0)
                logits, _, _ = mil(mu, logvar, minutes)
                logit_pos = logits[:, 1]
                loss = F.binary_cross_entropy_with_logits(logit_pos, y, pos_weight=torch.tensor(args.pos_weight, device=device))
                opt_mil.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mil.parameters(), 1.0); opt_mil.step()
        def mil_fn(b):
            mu, logvar, minutes, pm = tw_apply(b, hours)
            mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
            logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
            minutes = minutes.masked_fill(pm, 0.0)
            logits, _, _ = mil(mu, logvar, minutes)
            return logits[:, 1]
        base, cal = eval_loader(mil_fn, valid_loader)
        tbase, tcal = eval_loader(mil_fn, test_loader)
        results[f"gaussian_mil_{tag}_valid"] = base
        results[f"gaussian_mil_{tag}_valid_cal"] = cal
        results[f"gaussian_mil_{tag}_test"] = tbase
        results[f"gaussian_mil_{tag}_test_cal"] = tcal

        # 5) Light MC sampling + mean pooling + MLP
        mlp_mc = MLPHead(in_dim=D).to(device)
        opt_mc = torch.optim.AdamW(mlp_mc.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(args.epochs_mlp):
            mlp_mc.train()
            for batch in train_loader:
                mu, logvar, _, pm = tw_apply(batch, hours)
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
        def mlp_mc_fn(b):
            mu, logvar, _, pm = tw_apply(b, hours)
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
        results[f"mlp_mc_mean_{tag}_valid"] = base
        results[f"mlp_mc_mean_{tag}_valid_cal"] = cal
        results[f"mlp_mc_mean_{tag}_test"] = tbase
        results[f"mlp_mc_mean_{tag}_test_cal"] = tcal

    # Print concise report
    print("\n=== Feasibility Results ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}: AUROC={v.get('auroc', float('nan')):.4f}, AUPRC={v.get('auprc', float('nan')):.4f}, R@95spec={v.get('recall@95spec', float('nan')):.4f}")
        else:
            print(f"{k}: {v}")

    # Save JSON
    out_json = os.path.join(args.cached_dir, "feasibility_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()

