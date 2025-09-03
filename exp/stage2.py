import argparse
import itertools
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp.utils import ensure_dir, load_npz_data, set_global_seed, save_json
from exp.data import split_indices, make_loaders
from exp.metrics import evaluate_all_metrics
from exp.models import (
    AttentionPoolingHead,
    ExpectationLogit,
    TimeWeightedPoE,
    WassersteinBarycenter,
    KMEPooling,
    ShallowSequenceModel,
    build_tokens,
)


def train_one(model, loaders, device, pos_weight_value, lr, wd, max_epochs, patience):
    train_loader, val_loader, test_loader = loaders
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val = -np.inf
    best_state = None
    no_imp = 0
    for _ in range(max_epochs):
        model.train()
        for mu, logvar, dt, mask, y in train_loader:
            mu, logvar, dt, mask, y = mu.to(device), logvar.to(device), dt.to(device), mask.to(device), y.to(device)
            logits = forward_route(model, mu, logvar, dt, mask)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_model(model, val_loader, device)
        if val_metrics["auprc"] > best_val + 1e-6:
            best_val = val_metrics["auprc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device)
    return test_metrics


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_logits = [], []
    for mu, logvar, dt, mask, y in loader:
        mu, logvar, dt, mask = mu.to(device), logvar.to(device), dt.to(device), mask.to(device)
        logits = forward_route(model, mu, logvar, dt, mask)
        y_true.append(y.numpy())
        y_logits.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_logits = np.concatenate(y_logits)
    return evaluate_all_metrics(y_true, y_logits)


def forward_route(model, mu, logvar, dt, mask):
    if isinstance(model, AttentionPoolingHead):
        tokens = torch.cat([mu, torch.log(torch.clamp(torch.exp(logvar), min=1e-8))], dim=-1)
        return model(tokens, mask, dt)
    if isinstance(model, ExpectationLogit):
        return model(mu, logvar, mask)
    if isinstance(model, TimeWeightedPoE):
        return model(mu, logvar, mask)
    if isinstance(model, WassersteinBarycenter):
        return model(mu, logvar, mask)
    if isinstance(model, KMEPooling):
        return model(mu, logvar, mask)
    if isinstance(model, ShallowSequenceModel):
        tokens = torch.cat([mu, torch.log(torch.clamp(torch.exp(logvar), min=1e-8))], dim=-1)
        return model(tokens, mask)
    raise ValueError("Unknown model type")


def grid_for_route(route: str, dim: int):
    if route == "A":
        # Attention pooling: hidden size in {64, 128}, number of layers in {1, 2}
        for hidden in [64, 128]:
            for layers in [1, 2]:
                yield AttentionPoolingHead(in_dim=2 * dim, hidden=hidden, num_layers=layers)
    elif route == "B":
        yield ExpectationLogit(dim=dim)
    elif route == "C":
        for lam in [0.95, 0.97, 0.99]:
            yield TimeWeightedPoE(dim=dim, lambda_decay=lam, mlp_hidden=128)
    elif route == "D":
        yield WassersteinBarycenter(dim=dim, hidden=128)
    elif route == "E":
        for scale in [0.5, 1.0, 2.0]:
            yield KMEPooling(dim=dim, kernel_scale=scale, hidden=128)
    elif route == "F":
        for layers in [1, 2]:
            yield ShallowSequenceModel(token_dim=2 * dim, hidden=64, num_layers=layers, use_gru=True)
    else:
        raise ValueError(f"Unknown route {route}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Minimal implementations comparison across candidate routes")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--routes", type=str, default="A,B,C")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    arrays = load_npz_data(args.data_path)
    B = arrays["mu"].shape[0]
    dim = arrays["mu"].shape[-1]
    routes = [r.strip().upper() for r in args.routes.split(",") if r.strip()]

    all_results: List[Dict] = []
    for seed in args.seeds:
        set_global_seed(seed)
        train_idx, val_idx, test_idx = split_indices(B, args.val_ratio, args.test_ratio, seed)
        loaders = make_loaders(arrays, train_idx, val_idx, test_idx, args.batch_size)
        loader_train, loader_val, loader_test, class_counts = loaders
        pos_weight_value = float(class_counts[0]) / max(1, class_counts[1])
        for route in routes:
            for model in grid_for_route(route, dim):
                model = model.to(args.device)
                metrics = train_one(model, (loader_train, loader_val, loader_test), args.device, pos_weight_value, args.lr, args.wd, args.max_epochs, args.patience)
                all_results.append({"seed": seed, "route": route, "model": type(model).__name__, **metrics})

    out_path = os.path.join(args.out_dir, "stage2_results.json")
    save_json(all_results, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

