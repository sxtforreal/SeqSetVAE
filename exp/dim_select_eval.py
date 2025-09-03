#!/usr/bin/env python3
"""
Dimension-selection evaluation over cached features.

Procedure:
1) Load train/valid/test cached features and the train partition's dim_stats.
2) Rank latent dimensions by train mean_kl (descending) or alternative criteria.
3) Select top-K dims, slice mu/logvar accordingly, and re-run:
   - Linear probe (last-step mu)
   - MLP with PoE pooling

Outputs metrics for different K values to assess whether pruning helps.
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


def masked_select_last(mu: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
    B, S, D = mu.shape
    last = (~pm).sum(dim=1).clamp(min=1) - 1
    idx = last.view(-1, 1, 1).expand(-1, 1, D)
    return mu.gather(dim=1, index=idx).squeeze(1)


def pool_poe(mu: torch.Tensor, logvar: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
    var = logvar.exp().clamp(min=1e-6)
    precision = 1.0 / var
    mask = (~pm).float().unsqueeze(-1)
    precision = precision * mask
    tau_star = precision.sum(dim=1).clamp(min=1e-6)
    num = (precision * mu).sum(dim=1)
    mu_star = num / tau_star
    return mu_star


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64], dropout: float = 0.2):
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


def evaluate(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    auroc = AUROC(task="binary")(logits.sigmoid(), labels)
    auprc = AveragePrecision(task="binary")(logits.sigmoid(), labels)
    return {"auroc": float(auroc.item()), "auprc": float(auprc.item())}


def run_eval(cached_dir: str, topk_values: List[int], criterion: str, device: str = None, batch_size: int = 64, num_workers: int = 2):
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

    all_results = {}

    for K in topk_values:
        dims = select_dims_from_stats(dim_stats_path, K, criterion)
        dims_tensor = torch.tensor(dims, dtype=torch.long)

        # Linear probe on last-step mu[:, :, dims]
        lin = nn.Linear(K, 1).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=5e-3, weight_decay=1e-4)
        for epoch in range(3):
            lin.train()
            for batch in train_loader:
                mu = batch["mu"][:, :, dims_tensor].to(device)
                y = batch["label"].float().to(device)
                last = masked_select_last(mu, batch["padding_mask"]).to(device)
                logit = lin(last).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
        lin.eval()
        def lin_fn(b):
            mu = b["mu"][:, :, dims_tensor].to(device)
            last = masked_select_last(mu, b["padding_mask"]).to(device)
            return lin(last).squeeze(-1)
        all_results[f"linear_last_mu_top{K}_valid"] = _eval_loader(lin_fn, valid_loader)
        all_results[f"linear_last_mu_top{K}_test"] = _eval_loader(lin_fn, test_loader)

        # PoE + MLP on mu/logvar sliced
        mlp = MLPHead(in_dim=K).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
        for epoch in range(5):
            mlp.train()
            for batch in train_loader:
                mu = batch["mu"][:, :, dims_tensor].to(device)
                logvar = batch["logvar"][:, :, dims_tensor].to(device)
                pm = batch["padding_mask"].to(device)
                y = batch["label"].float().to(device)
                mu_p = pool_poe(mu, logvar, pm)
                logit = mlp(mu_p)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        def mlp_poe_fn(b):
            mu = b["mu"][:, :, dims_tensor].to(device)
            logvar = b["logvar"][:, :, dims_tensor].to(device)
            pm = b["padding_mask"].to(device)
            return mlp(pool_poe(mu, logvar, pm))
        all_results[f"mlp_poe_top{K}_valid"] = _eval_loader(mlp_poe_fn, valid_loader)
        all_results[f"mlp_poe_top{K}_test"] = _eval_loader(mlp_poe_fn, test_loader)

    # Save
    out_json = os.path.join(cached_dir, "dim_select_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {out_json}")


@torch.no_grad()
def _eval_loader(model_fn, loader) -> Dict[str, float]:
    logits, labels = [], []
    for batch in loader:
        logit = model_fn(batch).detach().cpu()
        y = batch["label"].detach().cpu()
        logits.append(logit)
        labels.append(y)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0).long()
    return evaluate(logits, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, required=True)
    parser.add_argument("--criterion", type=str, default="mean_kl", choices=["mean_kl", "mean_mu2", "mean_var", "mean_logvar"])
    parser.add_argument("--topk", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    run_eval(args.cached_dir, args.topk, args.criterion, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()

