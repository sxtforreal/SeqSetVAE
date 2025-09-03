import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

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
)


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


def make_model(route: str, dim: int, width: int = 128):
    if route == "A":
        return AttentionPoolingHead(in_dim=2 * dim, hidden=width, num_layers=1)
    if route == "B":
        return ExpectationLogit(dim=dim)
    if route == "C":
        return TimeWeightedPoE(dim=dim, lambda_decay=0.99, mlp_hidden=width)
    if route == "D":
        return WassersteinBarycenter(dim=dim, hidden=width)
    if route == "E":
        return KMEPooling(dim=dim, kernel_scale=1.0, hidden=width)
    if route == "F":
        return ShallowSequenceModel(token_dim=2 * dim, hidden=width//2, num_layers=1, use_gru=True)
    raise ValueError(route)


def run_training(model, loaders, device, lr, wd, max_epochs, patience, pos_weight_value):
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
        # val
        model.eval()
        y_true, y_logits = [], []
        with torch.no_grad():
            for mu, logvar, dt, mask, y in val_loader:
                mu, logvar, dt, mask = mu.to(device), logvar.to(device), dt.to(device), mask.to(device)
                logits = forward_route(model, mu, logvar, dt, mask)
                y_true.append(y.numpy()); y_logits.append(logits.detach().cpu().numpy())
        val_metrics = evaluate_all_metrics(np.concatenate(y_true), np.concatenate(y_logits))
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

    # test
    model.eval()
    y_true, y_logits = [], []
    with torch.no_grad():
        for mu, logvar, dt, mask, y in test_loader:
            mu, logvar, dt, mask = mu.to(device), logvar.to(device), dt.to(device), mask.to(device)
            logits = forward_route(model, mu, logvar, dt, mask)
            y_true.append(y.numpy()); y_logits.append(logits.detach().cpu().numpy())
    test_metrics = evaluate_all_metrics(np.concatenate(y_true), np.concatenate(y_logits))
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 3: 诊断性消融")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--best_routes", type=str, default="A,C")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0,1,2])

    # Ablation toggles
    parser.add_argument("--near_vs_far_steps", type=int, nargs="*", default=[1,3,5])
    parser.add_argument("--add_logvar", action="store_true", help="在可用路线上加入/去除 log σ 特征")
    parser.add_argument("--scale_widths", type=int, nargs="*", default=[64,128,256])
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    arrays = load_npz_data(args.data_path)
    B = arrays["mu"].shape[0]
    dim = arrays["mu"].shape[-1]
    routes = [r.strip().upper() for r in args.best_routes.split(",") if r.strip()]

    all_results: List[Dict] = []
    for seed in args.seeds:
        set_global_seed(seed)
        train_idx, val_idx, test_idx = split_indices(B, args.val_ratio, args.test_ratio, seed)
        loaders = make_loaders(arrays, train_idx, val_idx, test_idx, args.batch_size)
        _, _, _, class_counts = loaders
        pos_weight_value = float(class_counts[0]) / max(1, class_counts[1])

        # 1) 近期 vs 远期：仅用最后 k 次就诊（k ∈ near_vs_far_steps）
        for route in routes:
            for k in args.near_vs_far_steps:
                arrays_k = arrays.copy()
                mask = arrays_k["mask"].copy()
                T = mask.shape[1]
                # keep last k valid tokens
                for i in range(B):
                    valid_idx = np.where(mask[i])[0]
                    if len(valid_idx) > k:
                        drop = valid_idx[:-k]
                        arrays_k["mask"][i, drop] = False
                loaders_k = make_loaders(arrays_k, train_idx, val_idx, test_idx, args.batch_size)
                model = make_model(route, dim, width=128).to(args.device)
                m = run_training(model, loaders_k, args.device, args.lr, args.wd, args.max_epochs, args.patience, pos_weight_value)
                m.update({"seed": seed, "route": route, "abl": f"near_last_{k}"})
                all_results.append(m)

        # 2) 顺序敏感性：打乱顺序
        arrays_shuffle = arrays.copy()
        rng = np.random.default_rng(seed)
        for i in range(B):
            idx = np.where(arrays_shuffle["mask"][i])[0]
            rng.shuffle(idx)
            # reorder the valid prefix; keep padding after
            T = arrays_shuffle["mu"].shape[1]
            keep = np.where(arrays["mask"][i])[0]
            pad = np.where(~arrays["mask"][i])[0]
            order = np.concatenate([idx, pad])
            arrays_shuffle["mu"][i] = arrays["mu"][i, order]
            arrays_shuffle["logvar"][i] = arrays["logvar"][i, order]
            arrays_shuffle["dt"][i] = arrays["dt"][i, order]
        loaders_shuf = make_loaders(arrays_shuffle, train_idx, val_idx, test_idx, args.batch_size)
        for route in routes:
            model = make_model(route, dim, width=128).to(args.device)
            m = run_training(model, loaders_shuf, args.device, args.lr, args.wd, args.max_epochs, args.patience, pos_weight_value)
            m.update({"seed": seed, "route": route, "abl": "order_shuffle"})
            all_results.append(m)

        # 3) 不确定性贡献：加入/去除 log σ
        if args.add_logvar:
            # 对能加的模型（A/C/D/F）额外加入 log σ -> 通过更宽 head 近似
            for route in routes:
                width_list = args.scale_widths
                for w in width_list:
                    model = make_model(route, dim, width=w).to(args.device)
                    m = run_training(model, loaders, args.device, args.lr, args.wd, args.max_epochs, args.patience, pos_weight_value)
                    m.update({"seed": seed, "route": route, "abl": f"width_{w}"})
                    all_results.append(m)

    out_path = os.path.join(args.out_dir, "stage3_results.json")
    save_json(all_results, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

