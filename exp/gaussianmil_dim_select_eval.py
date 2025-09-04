#!/usr/bin/env python3
"""
Run Gaussian-MIL head over top-K selected latent dimensions using cached per-set posteriors.

- Loads cached features from --cached_dir with subdirs train/valid/test each having manifest.json
- Selects top-K dims by a criterion from train dim_stats (default: mean_kl)
- Trains a Gaussian-MIL head on sliced mu/logvar (K dims) and evaluates on valid/test
- Writes results into cached_dir/dim_select_results.json with keys like gaussian_mil_top{K}_{split}

Usage:
  python exp/gaussianmil_dim_select_eval.py --cached_dir /path/to/cached_features \
      --topk 32 64 128 --criterion mean_kl --epochs 8 --batch_size 64 --device cuda
"""

import os
import json
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import AUROC, AveragePrecision
from tqdm import tqdm

from model import GaussianMILHead


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
        return torch.load(self.paths[idx], map_location="cpu")


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
    return {"mu": mu, "logvar": logvar, "minutes": minutes, "padding_mask": padding_mask, "label": labels}


def select_dims_from_stats(stats_path: str, topk: int, criterion: str = "mean_kl") -> List[int]:
    stats = torch.load(stats_path, map_location="cpu") if stats_path.endswith('.pt') else json.load(open(stats_path))
    scores = torch.tensor(stats[criterion])
    vals, idx = torch.sort(scores, descending=True)
    return idx[:topk].tolist()


@torch.no_grad()
def evaluate(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    auroc = AUROC(task="binary")(logits.sigmoid(), labels)
    auprc = AveragePrecision(task="binary")(logits.sigmoid(), labels)
    return {"auroc": float(auroc.item()), "auprc": float(auprc.item())}


@torch.no_grad()
def _eval_loader(model_fn, loader) -> Dict[str, float]:
    logits, labels = [], []
    pbar = tqdm(enumerate(loader, 1), total=len(loader), desc="eval", leave=False)
    for _idx, batch in pbar:
        logit = model_fn(batch).detach().cpu()
        y = batch["label"].detach().cpu()
        logits.append(logit)
        labels.append(y)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0).long()
    return evaluate(logits, labels)


def run_eval(cached_dir: str, topk_values: List[int], criterion: str, device: str = None, batch_size: int = 64, num_workers: int = 2, epochs: int = 8, pos_weight: float = 9.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    train_dir = os.path.join(cached_dir, "train")
    valid_dir = os.path.join(cached_dir, "valid")
    test_dir = os.path.join(cached_dir, "test")

    dim_stats_path = os.path.join(train_dir, "dim_stats.pt")

    def build_loader(d):
        ds = CachedSeqDataset(d)
        return DataLoader(ds, batch_size=batch_size, shuffle=(d==train_dir), num_workers=num_workers, collate_fn=collate_fn)

    train_loader = build_loader(train_dir)
    valid_loader = build_loader(valid_dir)
    test_loader = build_loader(test_dir)

    all_results: Dict[str, Dict[str, float]] = {}

    for K in topk_values:
        dims = select_dims_from_stats(dim_stats_path, K, criterion)
        dims_tensor = torch.tensor(dims, dtype=torch.long)

        mil = GaussianMILHead(latent_dim=K, num_classes=2, use_time=True, gate_hidden_dim=128).to(device)
        opt = torch.optim.AdamW(mil.parameters(), lr=3e-3, weight_decay=1e-4)
        pos_w = torch.tensor(pos_weight, device=device)

        for epoch in range(1, epochs + 1):
            mil.train()
            pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"gaussian_mil top{K} epoch {epoch}/{epochs}", leave=False)
            for batch_idx, batch in pbar:
                mu = batch["mu"][:, :, dims_tensor].to(device)
                logvar = batch["logvar"][:, :, dims_tensor].to(device)
                minutes = batch["minutes"].to(device)
                pm = batch["padding_mask"].to(device)
                y = batch["label"].float().to(device)

                mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
                logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
                minutes = minutes.masked_fill(pm, 0.0)

                logits2, _, _ = mil(mu, logvar, minutes)
                logit_pos = logits2[:, 1]
                loss = F.binary_cross_entropy_with_logits(logit_pos, y, pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mil.parameters(), 1.0); opt.step()
                pbar.set_postfix(epoch=epoch, batch=f"{batch_idx}/{len(train_loader)}", loss=f"{loss.item():.4f}")

        mil.eval()

        def mil_fn(b):
            mu = b["mu"][:, :, dims_tensor].to(device)
            logvar = b["logvar"][:, :, dims_tensor].to(device)
            minutes = b["minutes"].to(device)
            pm = b["padding_mask"].to(device)
            mu = mu.masked_fill(pm.unsqueeze(-1), 0.0)
            logvar = logvar.masked_fill(pm.unsqueeze(-1), 0.0)
            minutes = minutes.masked_fill(pm, 0.0)
            logits2, _, _ = mil(mu, logvar, minutes)
            return logits2[:, 1]

        all_results[f"gaussian_mil_top{K}_valid"] = _eval_loader(mil_fn, valid_loader)
        all_results[f"gaussian_mil_top{K}_test"] = _eval_loader(mil_fn, test_loader)

    # Merge into dim_select_results.json under cached_dir
    out_json = os.path.join(cached_dir, "dim_select_results.json")
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    else:
        prev = {}
    prev.update(all_results)
    with open(out_json, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"Saved updated results to {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, required=True)
    parser.add_argument("--criterion", type=str, default="mean_kl", choices=["mean_kl", "mean_mu2", "mean_var", "mean_logvar"])
    parser.add_argument("--topk", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--pos_weight", type=float, default=9.0)
    args = parser.parse_args()

    run_eval(
        cached_dir=args.cached_dir,
        topk_values=args.topk,
        criterion=args.criterion,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        pos_weight=args.pos_weight,
    )


if __name__ == "__main__":
    main()

