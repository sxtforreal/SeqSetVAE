import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from exp.utils import set_global_seed, ensure_dir, save_json, load_npz_data, mask_last_index, per_dim_kl_to_standard_normal, make_dim_gate
from exp.metrics import evaluate_all_metrics


class SeqDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.mu = arrays["mu"]
        self.logvar = arrays["logvar"]
        self.dt = arrays["dt"]
        self.mask = arrays["mask"]
        self.y = arrays["y"]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.mu[i],
            self.logvar[i],
            self.dt[i],
            self.mask[i],
            self.y[i].astype(np.float32),
        )


class LogisticProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def build_feature(mu, logvar, mask, mode: str, use_logvar: bool, use_dim_gate: bool = False, gate_scale: float = 5.0, gate_bias: float = 0.0):
    # mu, logvar, mask: [B,T,D], [B,T,D], [B,T]
    device = mu.device
    B, T, D = mu.shape
    eps = 1e-8
    mask_f = mask.float()

    if mode == "last":
        last_idx = mask_last_index(mask)  # [B]
        batch_idx = torch.arange(B, device=device)
        feats_mu = mu[batch_idx, last_idx]
        feats_lv = logvar[batch_idx, last_idx]
    elif mode == "mean":
        denom = mask_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        feats_mu = (mu * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        feats_lv = (logvar * mask_f.unsqueeze(-1)).sum(dim=1) / denom
    elif mode == "poe":
        prec_t = torch.exp(-logvar) * mask_f.unsqueeze(-1)
        prec_sum = prec_t.sum(dim=1).clamp(min=eps)
        feats_mu = (prec_t * mu).sum(dim=1) / prec_sum
        feats_lv = -torch.log(prec_sum + eps)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if use_dim_gate:
        kl_mean = per_dim_kl_to_standard_normal(mu, logvar, mask)  # [B,D]
        gate = make_dim_gate(kl_mean, scale=gate_scale, bias=gate_bias)  # [B,D]
        feats_mu = feats_mu * gate
        feats_lv = feats_lv * gate

    features = [feats_mu]
    if use_logvar:
        features.append(feats_lv)
    x = torch.cat(features, dim=-1)
    return x


def collate(batch):
    mu = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    logvar = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    dt = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
    mask = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.bool)
    y = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32)
    return mu, logvar, dt, mask, y


def maybe_shuffle_time(mu, logvar, dt, mask):
    # Shuffle valid time steps per sample while keeping the number of valid steps (mask) fixed
    B, T, D = mu.shape
    device = mu.device
    rng = torch.Generator(device=device)
    out_mu, out_lv, out_dt = [], [], []
    for i in range(B):
        valid = mask[i].nonzero(as_tuple=False).squeeze(-1)
        p = torch.randperm(len(valid), generator=rng, device=device)
        idx = valid[p]
        pad = torch.arange(T, device=device)[~mask[i]]
        new_order = torch.cat([idx, pad], dim=0)
        out_mu.append(mu[i, new_order])
        out_lv.append(logvar[i, new_order])
        if dt.ndim == 3:
            out_dt.append(dt[i, new_order])
        else:
            out_dt.append(dt[i, new_order])
    return torch.stack(out_mu), torch.stack(out_lv), torch.stack(out_dt)


def train_epoch(model, loader, optimizer, criterion, device, feature_mode, use_logvar, shuffle_time=False, use_dim_gate: bool = False, gate_scale: float = 5.0, gate_bias: float = 0.0):
    model.train()
    losses = []
    for mu, logvar, dt, mask, y in loader:
        mu = mu.to(device)
        logvar = logvar.to(device)
        mask = mask.to(device)
        y = y.to(device)
        if shuffle_time:
            mu, logvar, dt = maybe_shuffle_time(mu, logvar, dt, mask)
        x = build_feature(mu, logvar, mask, feature_mode, use_logvar, use_dim_gate=use_dim_gate, gate_scale=gate_scale, gate_bias=gate_bias)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, feature_mode, use_logvar, use_dim_gate: bool = False, gate_scale: float = 5.0, gate_bias: float = 0.0):
    model.eval()
    y_true = []
    y_logits = []
    for mu, logvar, dt, mask, y in loader:
        mu = mu.to(device)
        logvar = logvar.to(device)
        mask = mask.to(device)
        x = build_feature(mu, logvar, mask, feature_mode, use_logvar, use_dim_gate=use_dim_gate, gate_scale=gate_scale, gate_bias=gate_bias)
        logits = model(x)
        y_true.append(y.numpy())
        y_logits.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_logits = np.concatenate(y_logits)
    return evaluate_all_metrics(y_true, y_logits)


def split_indices(num_samples: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    n_val = int(num_samples * val_ratio)
    n_test = int(num_samples * test_ratio)
    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val+n_test]
    train_idx = indices[n_val+n_test:]
    return train_idx, val_idx, test_idx


def run_once(args, seed, feature_mode: str, use_logvar: bool, shuffle_time_flag: bool):
    set_global_seed(seed)
    arrays = load_npz_data(args.data_path)
    B = arrays["mu"].shape[0]
    train_idx, val_idx, test_idx = split_indices(B, args.val_ratio, args.test_ratio, seed)

    ds_train = SeqDataset(arrays, train_idx)
    ds_val = SeqDataset(arrays, val_idx)
    ds_test = SeqDataset(arrays, test_idx)

    # Weighted sampler for positive class balancing
    y_train = arrays["y"][train_idx].astype(np.int64)
    class_counts = np.bincount(y_train, minlength=2)
    pos_weight_value = args.pos_weight if args.pos_weight > 0 else float(class_counts[0]) / max(1, class_counts[1])
    samples_weight = np.where(y_train == 1, class_counts.sum()/(2*max(1,class_counts[1])), class_counts.sum()/(2*max(1,class_counts[0])))
    sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), num_samples=len(samples_weight), replacement=True)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    D = arrays["mu"].shape[-1]
    in_dim = D + (D if use_logvar else 0)
    model = LogisticProbe(in_dim).to(args.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = -np.inf
    best_state = None
    patience = args.patience
    no_improve = 0
    for epoch in range(args.max_epochs):
        train_epoch(
            model,
            loader_train,
            optimizer,
            criterion,
            args.device,
            feature_mode,
            use_logvar,
            shuffle_time=shuffle_time_flag,
            use_dim_gate=args.use_dim_gate,
            gate_scale=args.gate_scale,
            gate_bias=args.gate_bias,
        )
        val_metrics = evaluate(
            model,
            loader_val,
            args.device,
            feature_mode,
            use_logvar,
            use_dim_gate=args.use_dim_gate,
            gate_scale=args.gate_scale,
            gate_bias=args.gate_bias,
        )
        if val_metrics["auprc"] > best_val + 1e-6:
            best_val = val_metrics["auprc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        loader_test,
        args.device,
        feature_mode,
        use_logvar,
        use_dim_gate=args.use_dim_gate,
        gate_scale=args.gate_scale,
        gate_bias=args.gate_bias,
    )
    return {
        "seed": seed,
        "feature_mode": feature_mode,
        "use_logvar": use_logvar,
        "shuffle_time": shuffle_time_flag,
        **test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Three linear probes + order/uncertainty checks")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--pos_weight", type=float, default=-1.0, help="If <0, set from training set class frequency")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0,1,2])
    parser.add_argument("--modes", type=str, default="last,mean,poe")
    parser.add_argument("--with_logvar", action="store_true")
    parser.add_argument("--order_check", action="store_true", help="For Mean/PoE, validate by shuffling order")
    # Dimension gating for prior-like dimensions
    parser.add_argument("--use_dim_gate", action="store_true")
    parser.add_argument("--gate_scale", type=float, default=5.0)
    parser.add_argument("--gate_bias", type=float, default=0.0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    results = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for seed in args.seeds:
        for mode in modes:
            res = run_once(args, seed, mode, args.with_logvar, shuffle_time_flag=False)
            results.append(res)
            if args.order_check and mode in {"mean", "poe"}:
                res_shuffle = run_once(args, seed, mode, args.with_logvar, shuffle_time_flag=True)
                results.append(res_shuffle)

    out_path = os.path.join(args.out_dir, "stage1_results.json")
    save_json(results, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

