#!/usr/bin/env python3
"""
Feasibility test harness over cached per-set posteriors.

Models compared:
- Linear probe on last-step mu (baseline)
- MLP on pooled features with three pooling rules: mean, PoE, Wasserstein-2 barycenter
- Gaussian-MIL head (from model.GaussianMILHead) using per-set [mu, logvar, minutes]

Metrics:
- AUROC, Average Precision (AUPRC)
- Optionally report recall at 95% specificity

Inputs:
- --cached_dir: directory produced by cache_features.py with train/valid/test subdirs
"""

import os
import json
import argparse
from typing import List, Tuple, Dict

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
    # Variable S across batch; pad to max S
    max_s = max(item["mu"].shape[0] for item in batch)
    D = batch[0]["mu"].shape[1]
    B = len(batch)
    mu = torch.zeros(B, max_s, D)
    logvar = torch.zeros(B, max_s, D)
    minutes = torch.zeros(B, max_s)
    padding_mask = torch.ones(B, max_s, dtype=torch.bool)
    labels = torch.tensor([item.get("label", 0) for item in batch], dtype=torch.long)
    lengths = []
    for i, item in enumerate(batch):
        S = item["mu"].shape[0]
        mu[i, :S] = item["mu"]
        logvar[i, :S] = item["logvar"]
        minutes[i, :S] = item["minutes"]
        padding_mask[i, :S] = False
        lengths.append(S)
    return {
        "mu": mu,
        "logvar": logvar,
        "minutes": minutes,
        "padding_mask": padding_mask,
        "label": labels,
        "lengths": torch.tensor(lengths)
    }


def masked_select_last(mu: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    # mu: [B,S,D], padding_mask: True for pad
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
    # tempered PoE with weights proportional to inverse variance, ignoring padded
    var = logvar.exp()
    precision = 1.0 / var.clamp(min=1e-6)
    mask = (~padding_mask).float().unsqueeze(-1)
    precision = precision * mask
    tau_star = precision.sum(dim=1).clamp(min=1e-6)
    num = (precision * mu).sum(dim=1)
    mu_star = num / tau_star
    var_star = 1.0 / tau_star
    logvar_star = var_star.clamp(min=1e-6).log()
    return mu_star, logvar_star


def pool_w2(mu: torch.Tensor, logvar: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Wasserstein-2 barycenter (diagonal Gaussian, per-dim)
    var = logvar.exp().clamp(min=1e-6)
    sd = var.sqrt()
    mask = (~padding_mask).float().unsqueeze(-1)
    weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    mu_bar = (weights * mu).sum(dim=1)
    sd_bar = (weights * sd).sum(dim=1)
    var_bar = sd_bar.pow(2)
    logvar_bar = var_bar.clamp(min=1e-8).log()
    return mu_bar, logvar_bar


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


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    auroc = AUROC(task="binary")(logits.sigmoid(), labels)
    auprc = AveragePrecision(task="binary")(logits.sigmoid(), labels)
    return {"auroc": float(auroc.item()), "auprc": float(auprc.item())}


@torch.no_grad()
def eval_loader(model_fn, loader) -> Dict[str, float]:
    all_logits, all_labels = [], []
    for batch in loader:
        logits = model_fn(batch)
        labels = batch["label"].float()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).long()
    return evaluate_logits(logits, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    # 1) Linear probe on last-step mu
    lin = nn.Linear(D, 1).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=5e-3, weight_decay=1e-4)
    for epoch in range(3):
        lin.train()
        for batch in train_loader:
            mu = batch["mu"].to(device)
            y = batch["label"].float().to(device)
            last = masked_select_last(mu, batch["padding_mask"]).to(device)
            logit = lin(last).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
    lin.eval()
    results["linear_last_mu_valid"] = eval_loader(lambda b: lin(masked_select_last(b["mu"].to(device), b["padding_mask"]).to(device)).squeeze(-1), valid_loader)
    results["linear_last_mu_test"] = eval_loader(lambda b: lin(masked_select_last(b["mu"].to(device), b["padding_mask"]).to(device)).squeeze(-1), test_loader)

    # 2) MLP over pooled features
    def run_mlp(pool_fn, name: str):
        mlp = MLPHead(in_dim=2*D).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
        for epoch in range(5):
            mlp.train()
            for batch in train_loader:
                mu = batch["mu"].to(device)
                logvar = batch["logvar"].to(device)
                pm = batch["padding_mask"].to(device)
                y = batch["label"].float().to(device)
                mu_p, logvar_p = pool_fn(mu, logvar, pm)
                x = torch.cat([mu_p, logvar_p], dim=-1)
                logit = mlp(x)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        model_fn = lambda b: mlp(torch.cat(pool_fn(b["mu"].to(device), b["logvar"].to(device), b["padding_mask"].to(device)), dim=-1))
        results[f"mlp_{name}_valid"] = eval_loader(model_fn, valid_loader)
        results[f"mlp_{name}_test"] = eval_loader(model_fn, test_loader)

    run_mlp(pool_mean, "mean")
    run_mlp(pool_poe, "poe")
    run_mlp(pool_w2, "w2")

    # 3) Gaussian-MIL head
    mil = GaussianMILHead(latent_dim=D, num_classes=2, use_time=True, gate_hidden_dim=128).to(device)
    opt = torch.optim.AdamW(mil.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch in range(8):
        mil.train()
        for batch in train_loader:
            mu = batch["mu"].to(device)
            logvar = batch["logvar"].to(device)
            minutes = batch["minutes"].to(device)
            pm = batch["padding_mask"].to(device)
            y = batch["label"].to(device)
            # Mask out pads by zeroing them; gating should learn to downweight
            mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
            logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
            minutes = minutes.masked_fill(pm, 0.0)
            logits, s_per_set, a = mil(mu, logvar, minutes)
            logit_pos = logits[:, 1]
            # Class imbalance aware loss (weighted BCE)
            pos_weight = torch.tensor([9.0], device=device)
            loss = F.binary_cross_entropy_with_logits(logit_pos, y.float(), pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mil.parameters(), 1.0); opt.step()
    mil.eval()
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
    results["gaussian_mil_valid"] = eval_loader(mil_fn, valid_loader)
    results["gaussian_mil_test"] = eval_loader(mil_fn, test_loader)

    # Print concise report
    print("\n=== Feasibility Results ===")
    for k, v in results.items():
        print(f"{k}: AUROC={v['auroc']:.4f}, AUPRC={v['auprc']:.4f}")

    # Save JSON
    out_json = os.path.join(args.cached_dir, "feasibility_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()

