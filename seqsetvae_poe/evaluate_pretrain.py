#!/usr/bin/env python3
"""
Evaluate SetVAE pretraining results: posterior collapse and reconstruction quality.

Features:
- Dataset-level posterior metrics: KL per-dimension, active ratio, KL coverage (dim@90/95%).
- Reconstruction metrics on a few samples: NN-L2, Chamfer(L2), cosine dir, magnitude errors.
- Visualizations: KL histogram + CDF, per-sample latent (mu/var) heatmaps across sets,
  and Original vs Reconstruction 2D overlay (UMAP if available else PCA).

Usage example:
  python -u seqsetvae_poe/evaluate_pretrain.py \
    --checkpoint /path/to/outputs/SetVAE-Only-PT/checkpoints/setvae_PT.ckpt \
    --data_dir /path/to/SeqSetVAE \
    --split valid \
    --num_eval_batches 50 \
    --num_vis_samples 2 \
    --output_dir ./outputs/setvae_pretrain_eval
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Optional visualization deps
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
except Exception:
    PCA = None  # type: ignore
    StandardScaler = None  # type: ignore
    NearestNeighbors = None  # type: ignore
try:
    import umap  # type: ignore
except Exception:
    umap = None  # type: ignore


# Ensure local imports work even if launched from project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

import config as cfg  # type: ignore
from dataset import DataModule  # type: ignore
from model import SetVAEOnlyPretrain, PoESeqSetVAEPretrain  # type: ignore


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict")
    # If keys look like 'model.xxx', strip leading 'model.' prefix from Lightning 2.x
    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    return state


def _detect_ckpt_type(state: Dict[str, torch.Tensor], prefer: str = "auto") -> str:
    if prefer in {"poe", "setvae"}:
        return prefer
    keys = list(state.keys())
    has_transformer = any(k.startswith("transformer.") for k in keys)
    has_prior_head = any(k.startswith("prior_head.") for k in keys)
    if has_transformer or has_prior_head:
        return "poe"
    return "setvae"


def _build_model(ckpt_type: str, lr: float) -> torch.nn.Module:
    if ckpt_type == "poe":
        model = PoESeqSetVAEPretrain(
            input_dim=getattr(cfg, "input_dim", 768),
            reduced_dim=getattr(cfg, "reduced_dim", 256),
            latent_dim=getattr(cfg, "latent_dim", 128),
            levels=getattr(cfg, "levels", 2),
            heads=getattr(cfg, "heads", 2),
            m=getattr(cfg, "m", 16),
            beta=getattr(cfg, "beta", 0.1),
            lr=lr,
            ff_dim=getattr(cfg, "ff_dim", 512),
            transformer_heads=getattr(cfg, "transformer_heads", 8),
            transformer_layers=getattr(cfg, "transformer_layers", 4),
            transformer_dropout=getattr(cfg, "transformer_dropout", 0.15),
            warmup_beta=getattr(cfg, "warmup_beta", True),
            max_beta=getattr(cfg, "max_beta", 0.05),
            beta_warmup_steps=getattr(cfg, "beta_warmup_steps", 8000),
            free_bits=getattr(cfg, "free_bits", 0.03),
            stale_dropout_p=getattr(cfg, "stale_dropout_p", 0.2),
            set_mae_ratio=getattr(cfg, "set_mae_ratio", 0.0),
            enable_next_change=True,
            next_change_weight=0.3,
        )
    else:
        model = SetVAEOnlyPretrain(
            input_dim=getattr(cfg, "input_dim", 768),
            reduced_dim=getattr(cfg, "reduced_dim", 256),
            latent_dim=getattr(cfg, "latent_dim", 128),
            levels=getattr(cfg, "levels", 2),
            heads=getattr(cfg, "heads", 2),
            m=getattr(cfg, "m", 16),
            beta=getattr(cfg, "beta", 0.1),
            lr=lr,
            warmup_beta=getattr(cfg, "warmup_beta", True),
            max_beta=getattr(cfg, "max_beta", 0.2),
            beta_warmup_steps=getattr(cfg, "beta_warmup_steps", 8000),
            free_bits=getattr(cfg, "free_bits", 0.05),
            p_stale=getattr(cfg, "stale_dropout_p", 0.5),
            p_live=0.05,
            set_mae_ratio=0.0,
            small_set_mask_prob=0.0,
            small_set_threshold=5,
            max_masks_per_set=0,
            val_noise_std=0.0,
            dir_noise_std=0.0,
            train_decoder_noise_std=0.0,
            eval_decoder_noise_std=0.0,
        )
    return model


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _nn_from_a_to_b(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if NearestNeighbors is None or len(b) == 0:
        return np.array([]), np.array([])
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(b)
    dists, idxs = nn.kneighbors(a)
    return dists.squeeze(1), idxs.squeeze(1)


def compute_recon_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if orig.size == 0 or recon.size == 0:
        return {"error": 1.0}
    # Nearest neighbor from recon->orig and orig->recon
    d_ro, _ = _nn_from_a_to_b(recon, orig)
    d_or, _ = _nn_from_a_to_b(orig, recon)
    if d_ro.size == 0 or d_or.size == 0:
        return {"error": 1.0}
    chamfer = float(np.mean(d_ro ** 2) + np.mean(d_or ** 2)) / 2.0
    metrics["chamfer_l2_mean"] = chamfer
    metrics.update({
        "nn_l2_mean": float(np.mean(d_ro)),
        "nn_l2_median": float(np.median(d_ro)),
        "nn_l2_p95": float(np.percentile(d_ro, 95)),
    })
    # Direction cosine similarity
    def _norm(x):
        return np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    u_recon = recon / _norm(recon)
    u_orig = orig / _norm(orig)
    # Pair with nearest orig for each recon
    _, idx_ro = _nn_from_a_to_b(recon, orig)
    if idx_ro.size > 0:
        paired_orig = u_orig[idx_ro]
        cos_sim = np.sum(u_recon * paired_orig, axis=1)
        metrics["dir_cosine_mean"] = float(np.mean(cos_sim))
        metrics["dir_cosine_median"] = float(np.median(cos_sim))
    # Magnitude agreement
    norm_recon = np.linalg.norm(recon, axis=1)
    norm_orig = np.linalg.norm(orig, axis=1)
    if idx_ro.size > 0:
        norm_orig_matched = norm_orig[idx_ro]
        metrics["mag_mae"] = float(np.mean(np.abs(norm_recon - norm_orig_matched)))
        metrics["mag_rmse"] = float(np.sqrt(np.mean((norm_recon - norm_orig_matched) ** 2)))
        # correlation
        try:
            if np.std(norm_recon) > 1e-8 and np.std(norm_orig_matched) > 1e-8:
                metrics["mag_corr"] = float(np.corrcoef(norm_recon, norm_orig_matched)[0, 1])
        except Exception:
            pass
    metrics["scale_ratio"] = float((np.mean(norm_recon) + 1e-8) / (np.mean(norm_orig) + 1e-8))
    # Coverage at thresholds
    for th in [0.25, 0.5, 1.0, 2.0]:
        metrics[f"coverage@{th}"] = float(np.mean(d_ro <= th))
    # Global norms
    metrics["orig_norm_mean"] = float(np.mean(norm_orig))
    metrics["recon_norm_mean"] = float(np.mean(norm_recon))
    metrics["orig_norm_std"] = float(np.std(norm_orig))
    metrics["recon_norm_std"] = float(np.std(norm_recon))
    return metrics


@torch.no_grad()
def compute_kl_dim_stats_for_batch(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    # If model provides a helper, use it
    if hasattr(model, "_compute_kl_dim_stats") and callable(getattr(model, "_compute_kl_dim_stats")):
        try:
            return model._compute_kl_dim_stats(batch)
        except Exception:
            pass
    # Fallback: manual compute using SetVAE encoder on clean inputs
    var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    # Use the split_sets helper if present
    if hasattr(model, "_split_sets"):
        all_sets = model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
    else:
        # Minimal split: treat each contiguous set_id as one set
        all_sets = []
        B = var.size(0)
        for b in range(B):
            v = var[b]
            x = val[b]
            t = minutes[b]
            s = set_id[b]
            if padding_mask is not None:
                mask = ~padding_mask[b]
                v, x, t, s = v[mask], x[mask], t[mask], s[mask]
            if s.dim() > 1:
                s = s.squeeze(-1)
            uniq, counts = torch.unique_consecutive(s.long(), return_counts=True)
            idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(c) for c in counts])
            patient_sets = []
            for idx in idx_splits:
                patient_sets.append({
                    "var": v[idx].unsqueeze(0),
                    "val": x[idx].unsqueeze(0),
                })
            all_sets.append(patient_sets)

    latent_dim = int(getattr(model, "latent_dim", 128))
    kl_dim_sum: Optional[torch.Tensor] = None
    count = 0
    device = var.device

    for sets in all_sets:
        for s in sets:
            if hasattr(model, "set_encoder"):
                encoder = model.set_encoder
            else:
                # Try attribute nesting
                encoder = getattr(model, "setvae", None)
                if encoder is None or not hasattr(encoder, "set_encoder"):
                    return torch.zeros(latent_dim, device=device)
                encoder = encoder.set_encoder
            # Build clean target x = normalize(var) * val
            if getattr(encoder, "dim_reducer", None) is not None:
                reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
            else:
                reduced = s["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            v_norm = reduced / (norms + 1e-8)
            x_clean = v_norm * s["val"]
            z_list, _ = encoder.encode(x_clean)
            _z, mu, logvar = z_list[-1]
            # KL to N(0,I) per-dim
            kl_dim = 0.5 * (logvar.exp().squeeze(0).squeeze(0) + mu.squeeze(0).squeeze(0).pow(2) - 1.0 - logvar.squeeze(0).squeeze(0))
            if kl_dim_sum is None:
                kl_dim_sum = kl_dim
            else:
                kl_dim_sum = kl_dim_sum + kl_dim
            count += 1

    if kl_dim_sum is None or count == 0:
        return torch.zeros(latent_dim, device=device)
    return kl_dim_sum / float(count)


@torch.no_grad()
def compute_kl_dataset(model: torch.nn.Module, dataloader, num_batches: Optional[int] = None) -> torch.Tensor:
    device = next(model.parameters()).device
    kl_sum: Optional[torch.Tensor] = None
    seen = 0
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        kd = compute_kl_dim_stats_for_batch(model, batch)
        if kd is None:
            continue
        kd = kd.detach()
        if kl_sum is None:
            kl_sum = kd
        else:
            kl_sum = kl_sum + kd
        seen += 1
    if kl_sum is None or seen == 0:
        latent_dim = int(getattr(model, "latent_dim", 128))
        return torch.zeros(latent_dim, device=device)
    return kl_sum / float(seen)


@torch.no_grad()
def collect_mu_var_heatmaps(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    # Expect batch_size=1 for visualization
    var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    if hasattr(model, "_split_sets"):
        pats = model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
    else:
        pats = [[{"var": var[0:1, :1, :], "val": val[0:1, :1, :]}]]
    if len(pats) == 0 or len(pats[0]) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    sets = pats[0]
    mu_list: List[np.ndarray] = []
    var_list: List[np.ndarray] = []
    encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    if encoder is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    for s in sets:
        if getattr(encoder, "dim_reducer", None) is not None:
            reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
        else:
            reduced = s["var"]
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        v_norm = reduced / (norms + 1e-8)
        x_clean = v_norm * s["val"]
        z_list, _ = encoder.encode(x_clean)
        _z, mu, logvar = z_list[-1]
        mu_list.append(_to_numpy(mu.squeeze(0).squeeze(0)))  # [D]
        var_list.append(_to_numpy(logvar.exp().squeeze(0).squeeze(0)))  # [D]
    mu_mat = np.stack(mu_list, axis=1)  # [D, S]
    var_mat = np.stack(var_list, axis=1)  # [D, S]
    return mu_mat, var_mat


@torch.no_grad()
def collect_recon_events(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    if hasattr(model, "_split_sets"):
        pats = model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
    else:
        pats = [[{"var": var[0:1, :1, :], "val": val[0:1, :1, :]}]]
    if len(pats) == 0 or len(pats[0]) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    sets = pats[0]
    encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    if encoder is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    orig_cat: List[np.ndarray] = []
    recon_cat: List[np.ndarray] = []

    for s in sets:
        # Build clean targets in reduced space
        if getattr(encoder, "dim_reducer", None) is not None:
            reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
        else:
            reduced = s["var"]
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        v_norm = reduced / (norms + 1e-8)
        x_target = v_norm * s["val"]
        z_list, _ = encoder.encode(x_target)
        recon = encoder.decode(z_list, target_n=x_target.size(1), use_mean=True, noise_std=0.0)
        orig_cat.append(_to_numpy(x_target.squeeze(0)))
        recon_cat.append(_to_numpy(recon.squeeze(0)))

    orig = np.concatenate(orig_cat, axis=0) if orig_cat else np.zeros((0, 0), dtype=np.float32)
    recon = np.concatenate(recon_cat, axis=0) if recon_cat else np.zeros((0, 0), dtype=np.float32)
    return orig, recon


def _plot_kl_hist_and_cdf(kl_vec: np.ndarray, out_dir: str):
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    # Histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(kl_vec, bins=40, color="tab:blue", alpha=0.85)
    ax.set_title("KL per-dimension (hist)")
    ax.set_xlabel("KL (nats)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kl_hist.png"), dpi=200)
    plt.close(fig)
    # CDF and coverage
    sorted_vals = np.sort(kl_vec)[::-1]
    cumsum = np.cumsum(sorted_vals)
    total = float(cumsum[-1]) if len(cumsum) > 0 else 0.0
    frac = cumsum / (total + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(len(frac)), frac, color="tab:orange")
    ax.set_title("KL mass CDF (dims sorted desc)")
    ax.set_xlabel("# top dimensions")
    ax.set_ylabel("Cumulative fraction of KL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kl_cdf.png"), dpi=200)
    plt.close(fig)


def _plot_heatmap(matrix: np.ndarray, title: str, out_file: str, value_type: str = "mean"):
    if plt is None:
        return
    if matrix.size == 0:
        return
    plt.figure(figsize=(10, 6))
    if value_type == "mean":
        vmax = float(np.nanmax(np.abs(matrix))) if matrix.size > 0 else 1.0
        vmax = vmax if vmax > 1e-6 else 1.0
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin = 0.0
        vmax = float(np.percentile(matrix, 95)) if matrix.size > 0 else 1.0
        vmax = vmax if vmax > 1e-6 else float(np.nanmax(matrix) + 1e-6)
        cmap = "viridis"
    im = plt.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("set index")
    plt.ylabel("latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def _plot_overlay_2d(orig: np.ndarray, recon: np.ndarray, out_file: str):
    if plt is None:
        return
    if orig.size == 0 or recon.size == 0:
        return
    # Standardize using original only
    if StandardScaler is not None:
        try:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(orig)
            orig_s = scaler.transform(orig)
            recon_s = scaler.transform(recon)
        except Exception:
            orig_s, recon_s = orig, recon
    else:
        orig_s, recon_s = orig, recon
    # Reduce
    try:
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            orig_emb = reducer.fit_transform(orig_s)
            recon_emb = reducer.transform(recon_s)
        elif PCA is not None:
            pca = PCA(n_components=2)
            orig_emb = pca.fit_transform(orig_s)
            recon_emb = pca.transform(recon_s)
        else:
            # fallback: first two dims
            orig_emb = orig_s[:, :2]
            recon_emb = recon_s[:, :2]
    except Exception:
        if PCA is not None:
            pca = PCA(n_components=2)
            orig_emb = pca.fit_transform(orig_s)
            recon_emb = pca.transform(recon_s)
        else:
            orig_emb = orig_s[:, :2]
            recon_emb = recon_s[:, :2]

    n = min(len(orig_emb), len(recon_emb))
    orig_emb = orig_emb[:n]
    recon_emb = recon_emb[:n]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(orig_emb[:, 0], orig_emb[:, 1], s=8, alpha=0.7, c="tab:blue", label="Original")
    ax.scatter(recon_emb[:, 0], recon_emb[:, 1], s=8, alpha=0.7, c="tab:orange", label="Reconstruction")
    ax.set_title("Original vs Reconstruction (2D projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=260)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate SetVAE pretraining: collapse + reconstruction")
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to pretraining checkpoint (.ckpt)")
    ap.add_argument("--ckpt_type", type=str, choices=["auto", "setvae", "poe"], default="auto")
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--batch_size", type=int, default=getattr(cfg, "batch_size", 8))
    ap.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 0))
    ap.add_argument("--num_eval_batches", type=int, default=50, help="Number of batches to estimate KL stats")
    ap.add_argument("--num_vis_samples", type=int, default=2, help="#samples for detailed recon/heatmaps")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--active_threshold", type=float, default=0.01, help="KL threshold for active units")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Derive default output_dir as sibling to checkpoint version: version_X/eval
    if args.output_dir is None:
        ckpt_abs = os.path.abspath(args.checkpoint)
        ckpt_dir = os.path.dirname(ckpt_abs)
        # Expect structure: .../setvae-PT/version_X/checkpoints/*.ckpt
        # We will go up to version_X and create/use version_X/eval
        version_dir = os.path.dirname(ckpt_dir)
        # If path contains nested 'checkpoints', handle robustly
        # Search upward for 'version_*'
        def _find_version_dir(p: str) -> str:
            q = p
            while True:
                parent, tail = os.path.split(q)
                if tail.startswith("version_"):
                    return q
                if parent == q or tail == "":
                    return p  # fallback to given
                q = parent
        version_dir = _find_version_dir(version_dir)
        args.output_dir = os.path.join(version_dir, "eval")
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed (CPU only for deterministic plotting)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load checkpoint and instantiate model
    state = _load_state_dict(args.checkpoint)
    ckpt_type = _detect_ckpt_type(state, prefer=args.ckpt_type)
    model = _build_model(ckpt_type, lr=getattr(cfg, "lr", 3e-4))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    # Device
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data
    dm = DataModule(
        saved_dir=args.data_dir,
        params_map_path=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        smoke=False,
        apply_A=False,
    )
    dm.setup()
    dl = dm.val_dataloader() if args.split == "valid" else dm.test_dataloader()

    # 1) Posterior KL stats across dataset
    print("Computing KL per-dimension stats across dataset ...")
    kl_dim_t = compute_kl_dataset(model, dl, num_batches=args.num_eval_batches)
    kl_dim = kl_dim_t.detach().cpu().numpy().astype(np.float32)
    total_kl = float(kl_dim.sum())
    active_ratio = float(np.mean(kl_dim > args.active_threshold)) if kl_dim.size > 0 else 0.0
    sorted_vals = np.sort(kl_dim)[::-1]
    mass = np.cumsum(sorted_vals)
    denom = mass[-1] if mass.size > 0 else 1.0
    frac = mass / (denom + 1e-8)
    dim90 = int(np.sum(frac < 0.90) + 1) if frac.size > 0 else 0
    dim95 = int(np.sum(frac < 0.95) + 1) if frac.size > 0 else 0

    # Save KL plots
    _plot_kl_hist_and_cdf(kl_dim, args.output_dir)

    # 2) Detailed per-sample visualizations and recon metrics (batch_size=1 loader)
    print("Collecting per-sample visualizations and reconstruction metrics ...")
    dm_vis = DataModule(
        saved_dir=args.data_dir,
        params_map_path=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        smoke=False,
        apply_A=False,
    )
    dm_vis.setup()
    dl_vis = dm_vis.val_dataloader() if args.split == "valid" else dm_vis.test_dataloader()

    recon_results: List[Dict[str, float]] = []
    vis_count = 0
    for i, batch in enumerate(dl_vis):
        if vis_count >= args.num_vis_samples:
            break
        # Heatmaps
        mu_mat, var_mat = collect_mu_var_heatmaps(model, batch)
        _plot_heatmap(mu_mat, title=f"Sample {i} - μ across sets", out_file=os.path.join(args.output_dir, f"heatmap_mu_sample_{i}.png"), value_type="mean")
        _plot_heatmap(var_mat, title=f"Sample {i} - σ² across sets", out_file=os.path.join(args.output_dir, f"heatmap_var_sample_{i}.png"), value_type="var")
        # Recon overlay and metrics
        orig, recon = collect_recon_events(model, batch)
        _plot_overlay_2d(orig, recon, out_file=os.path.join(args.output_dir, f"overlay_orig_recon_sample_{i}.png"))
        metrics = compute_recon_metrics(orig, recon)
        metrics["sample_index"] = i
        recon_results.append(metrics)
        vis_count += 1

    # Aggregate recon summary
    def _agg(key: str) -> float:
        vals = [r[key] for r in recon_results if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    recon_summary = {
        "num_samples": vis_count,
        "nn_l2_mean": _agg("nn_l2_mean"),
        "nn_l2_median": _agg("nn_l2_median"),
        "nn_l2_p95": _agg("nn_l2_p95"),
        "dir_cosine_mean": _agg("dir_cosine_mean"),
        "mag_mae": _agg("mag_mae"),
        "mag_rmse": _agg("mag_rmse"),
        "mag_corr": _agg("mag_corr"),
        "scale_ratio": _agg("scale_ratio"),
        "chamfer_l2_mean": _agg("chamfer_l2_mean"),
    }

    # Collapse heuristic
    collapse_flags = {
        "kl_active_ratio_below_0.1": active_ratio < 0.1,
        "total_kl_below_1e-2": total_kl < 1e-2,
        "risk": (active_ratio < 0.1) or (total_kl < 1e-2),
    }

    # Persist JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "checkpoint": args.checkpoint,
        "ckpt_type": ckpt_type,
        "split": args.split,
        "active_threshold": args.active_threshold,
        "latent_dim": int(getattr(model, "latent_dim", len(kl_dim))),
        "kl": {
            "mean_per_dim": kl_dim.tolist(),
            "total_kl": total_kl,
            "active_ratio": active_ratio,
            "dim90": dim90,
            "dim95": dim95,
        },
        "reconstruction": {
            "summary": recon_summary,
            "details": recon_results,
        },
        "collapse_flags": collapse_flags,
    }
    out_json = os.path.join(args.output_dir, f"pretrain_eval_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {out_json}")


if __name__ == "__main__":
    main()

