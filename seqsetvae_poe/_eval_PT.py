#!/usr/bin/env python3
"""
Evaluate SetVAE pretraining results: posterior collapse and reconstruction quality.

Features:
- Dataset-level posterior metrics: KL per-dimension, active ratio, KL coverage (dim@90/95%).
- Reconstruction metrics on a few samples: NN-L2, Chamfer(L2), cosine dir, magnitude errors.
- Visualizations: KL histogram + CDF, per-sample latent (mu/var) heatmaps across sets,
  and Original vs Reconstruction 2D overlay (UMAP if available else PCA).

Usage example:
  python -u seqsetvae_poe/_eval_PT.py \
    --checkpoint /path/to/output/setvae-PT/version_0/checkpoints/setvae_PT.ckpt \
    --data_dir /path/to/SeqSetVAE \
    --split valid \
    --num_eval_batches 50 \
    --num_vis_samples 2
"""

import os
import sys
import json
import csv
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
from model import SetVAEOnlyPretrain  # type: ignore


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


def _build_model(lr: float) -> torch.nn.Module:
    return SetVAEOnlyPretrain(
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
        use_kl_capacity=getattr(cfg, "use_kl_capacity", True),
        capacity_per_dim_end=getattr(cfg, "capacity_per_dim_end", 0.03),
        capacity_warmup_steps=getattr(cfg, "capacity_warmup_steps", 20000),
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
        use_flows=False,
        num_flows=0,
        mmd_weight=0.0,
    )


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
    def _norm(x):
        return np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    u_recon = recon / _norm(recon)
    u_orig = orig / _norm(orig)
    _, idx_ro = _nn_from_a_to_b(recon, orig)
    if idx_ro.size > 0:
        paired_orig = u_orig[idx_ro]
        cos_sim = np.sum(u_recon * paired_orig, axis=1)
        metrics["dir_cosine_mean"] = float(np.mean(cos_sim))
        metrics["dir_cosine_median"] = float(np.median(cos_sim))
    norm_recon = np.linalg.norm(recon, axis=1)
    norm_orig = np.linalg.norm(orig, axis=1)
    if idx_ro.size > 0:
        norm_orig_matched = norm_orig[idx_ro]
        metrics["mag_mae"] = float(np.mean(np.abs(norm_recon - norm_orig_matched)))
        metrics["mag_rmse"] = float(np.sqrt(np.mean((norm_recon - norm_orig_matched) ** 2)))
        try:
            if np.std(norm_recon) > 1e-8 and np.std(norm_orig_matched) > 1e-8:
                metrics["mag_corr"] = float(np.corrcoef(norm_recon, norm_orig_matched)[0, 1])
        except Exception:
            pass
    metrics["scale_ratio"] = float((np.mean(norm_recon) + 1e-8) / (np.mean(norm_orig) + 1e-8))
    for th in [0.25, 0.5, 1.0, 2.0]:
        metrics[f"coverage@{th}"] = float(np.mean(d_ro <= th))
    metrics["orig_norm_mean"] = float(np.mean(norm_orig))
    metrics["recon_norm_mean"] = float(np.mean(norm_recon))
    metrics["orig_norm_std"] = float(np.std(norm_orig))
    metrics["recon_norm_std"] = float(np.std(norm_recon))
    return metrics


@torch.no_grad()
def compute_kl_dim_stats_for_batch(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "_compute_kl_dim_stats") and callable(getattr(model, "_compute_kl_dim_stats")):
        try:
            return model._compute_kl_dim_stats(batch)
        except Exception:
            pass
    var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    if hasattr(model, "_split_sets"):
        all_sets = model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
    else:
        all_sets = []
        B = var.size(0)
        for b in range(B):
            v = var[b]; x = val[b]; t = minutes[b]; s = set_id[b]
            if padding_mask is not None:
                mask = ~padding_mask[b]
                v, x, t, s = v[mask], x[mask], t[mask], s[mask]
            if s.dim() > 1:
                s = s.squeeze(-1)
            uniq, counts = torch.unique_consecutive(s.long(), return_counts=True)
            idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(c) for c in counts])
            patient_sets = []
            for idx in idx_splits:
                patient_sets.append({"var": v[idx].unsqueeze(0), "val": x[idx].unsqueeze(0)})
            all_sets.append(patient_sets)
    latent_dim = int(getattr(model, "latent_dim", 128))
    kl_dim_sum: Optional[torch.Tensor] = None
    count = 0
    device = var.device
    for sets in all_sets:
        for s in sets:
            encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
            if encoder is None:
                return torch.zeros(latent_dim, device=device)
            if getattr(encoder, "dim_reducer", None) is not None:
                reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
            else:
                reduced = s["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            v_norm = reduced / (norms + 1e-8)
            x_clean = v_norm * s["val"]
            z_list, _ = encoder.encode(x_clean)
            _z, mu, logvar = z_list[-1]
            kl_dim = 0.5 * (logvar.exp().squeeze(0).squeeze(0) + mu.squeeze(0).squeeze(0).pow(2) - 1.0 - logvar.squeeze(0).squeeze(0))
            kl_dim_sum = kl_dim if kl_dim_sum is None else (kl_dim_sum + kl_dim)
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
        kl_sum = kd if kl_sum is None else (kl_sum + kd)
        seen += 1
    if kl_sum is None or seen == 0:
        latent_dim = int(getattr(model, "latent_dim", 128))
        return torch.zeros(latent_dim, device=device)
    return kl_sum / float(seen)


@torch.no_grad()
def compute_posterior_moments_dataset(model: torch.nn.Module, dataloader, num_batches: Optional[int] = None) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    mu_abs_sum: Optional[torch.Tensor] = None
    mu_sum: Optional[torch.Tensor] = None
    mu_sq_sum: Optional[torch.Tensor] = None
    var_sum: Optional[torch.Tensor] = None
    var_sq_sum: Optional[torch.Tensor] = None
    count_sets = 0
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        var, val, minutes, set_id = batch["var"], batch["val"], batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        if hasattr(model, "_split_sets"):
            all_sets = model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
        else:
            all_sets = []
            B = var.size(0)
            for b in range(B):
                v = var[b]; x = val[b]; t = minutes[b]; s = set_id[b]
                if padding_mask is not None:
                    mask = ~padding_mask[b]
                    v, x, t, s = v[mask], x[mask], t[mask], s[mask]
                if s.dim() > 1:
                    s = s.squeeze(-1)
                uniq, counts = torch.unique_consecutive(s.long(), return_counts=True)
                idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(c) for c in counts])
                patient_sets = []
                for idx in idx_splits:
                    patient_sets.append({"var": v[idx].unsqueeze(0), "val": x[idx].unsqueeze(0)})
                all_sets.append(patient_sets)
        for sets in all_sets:
            for s in sets:
                encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
                if encoder is None:
                    continue
                if getattr(encoder, "dim_reducer", None) is not None:
                    reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
                else:
                    reduced = s["var"]
                norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
                v_norm = reduced / (norms + 1e-8)
                x_clean = v_norm * s["val"]
                z_list, _ = encoder.encode(x_clean)
                _z, mu, logvar = z_list[-1]
                mu_vec = mu.squeeze(0).squeeze(0)
                var_vec = logvar.exp().squeeze(0).squeeze(0)
                if mu_abs_sum is None:
                    mu_abs_sum = mu_vec.abs().clone(); mu_sum = mu_vec.clone(); mu_sq_sum = mu_vec.pow(2).clone(); var_sum = var_vec.clone(); var_sq_sum = var_vec.pow(2).clone()
                else:
                    mu_abs_sum = mu_abs_sum + mu_vec.abs(); mu_sum = mu_sum + mu_vec; mu_sq_sum = mu_sq_sum + mu_vec.pow(2); var_sum = var_sum + var_vec; var_sq_sum = var_sq_sum + var_vec.pow(2)
                count_sets += 1
    if count_sets == 0 or mu_abs_sum is None or mu_sum is None or mu_sq_sum is None or var_sum is None or var_sq_sum is None:
        latent_dim = int(getattr(model, "latent_dim", 128))
        zeros = np.zeros(latent_dim, dtype=np.float32)
        return {"mu_abs_mean": zeros, "mu_mean": zeros, "mu_std": zeros, "var_mean": zeros, "var_std": zeros}
    cs = float(count_sets)
    mu_abs_mean = (mu_abs_sum / cs).detach().cpu().numpy().astype(np.float32)
    mu_mean_t = (mu_sum / cs)
    mu_var_t = (mu_sq_sum / cs) - mu_mean_t.pow(2)
    mu_std = torch.clamp(mu_var_t, min=0.0).sqrt().detach().cpu().numpy().astype(np.float32)
    mu_mean = mu_mean_t.detach().cpu().numpy().astype(np.float32)
    var_mean_t = (var_sum / cs)
    var_var_t = (var_sq_sum / cs) - var_mean_t.pow(2)
    var_std = torch.clamp(var_var_t, min=0.0).sqrt().detach().cpu().numpy().astype(np.float32)
    var_mean = var_mean_t.detach().cpu().numpy().astype(np.float32)
    return {"mu_abs_mean": mu_abs_mean, "mu_mean": mu_mean, "mu_std": mu_std, "var_mean": var_mean, "var_std": var_std}


def compute_active_ratios(kl_dim: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    return {str(th): float(np.mean(kl_dim > th)) if kl_dim.size > 0 else 0.0 for th in thresholds}


def compute_effective_dimensions(kl_dim: np.ndarray) -> Dict[str, float]:
    if kl_dim.size == 0:
        return {"participation_ratio": 0.0, "entropy_dim": 0.0}
    s1 = float(np.sum(kl_dim)); s2 = float(np.sum(kl_dim ** 2))
    pr = (s1 * s1) / (s2 + 1e-12) if s2 > 0 else 0.0
    if s1 <= 0:
        ent_dim = 0.0
    else:
        p = kl_dim / (s1 + 1e-12)
        h = float(-(p * (np.log(p + 1e-12))).sum())
        ent_dim = float(np.exp(h))
    return {"participation_ratio": pr, "entropy_dim": ent_dim}


def _plot_hist_generic(values: np.ndarray, title: str, out_file: str, xlabel: str):
    if plt is None or values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=40, color="tab:green", alpha=0.85)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_file, dpi=200); plt.close(fig)


def _plot_scatter_generic(x: np.ndarray, y: np.ndarray, title: str, out_file: str, xlabel: str, ylabel: str):
    if plt is None or x.size == 0 or y.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.7, c="tab:purple")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_file, dpi=200); plt.close(fig)


def _plot_bar_topk(values: np.ndarray, k: int, title: str, out_file: str, xlabel: str = "ranked dims", ylabel: str = "value"):
    if plt is None or values.size == 0:
        return
    idx = np.argsort(values)[::-1]; vals = values[idx][:k]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(vals)), vals, color="tab:blue", alpha=0.85)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_file, dpi=200); plt.close(fig)


@torch.no_grad()
def collect_mu_var_heatmaps(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
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
    mu_list: List[np.ndarray] = []; var_list: List[np.ndarray] = []
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
        mu_list.append(_to_numpy(mu.squeeze(0).squeeze(0)))
        var_list.append(_to_numpy(logvar.exp().squeeze(0).squeeze(0)))
    mu_mat = np.stack(mu_list, axis=1); var_mat = np.stack(var_list, axis=1)
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
    orig_cat: List[np.ndarray] = []; recon_cat: List[np.ndarray] = []
    for s in sets:
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
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(kl_vec, bins=40, color="tab:blue", alpha=0.85)
    ax.set_title("KL per-dimension (hist)"); ax.set_xlabel("KL (nats)"); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "kl_hist.png"), dpi=200); plt.close(fig)
    sorted_vals = np.sort(kl_vec)[::-1]; cumsum = np.cumsum(sorted_vals); total = float(cumsum[-1]) if len(cumsum) > 0 else 0.0
    frac = cumsum / (total + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(len(frac)), frac, color="tab:orange")
    ax.set_title("KL mass CDF (dims sorted desc)"); ax.set_xlabel("# top dimensions"); ax.set_ylabel("Cumulative fraction of KL"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "kl_cdf.png"), dpi=200); plt.close(fig)


def _plot_heatmap(matrix: np.ndarray, title: str, out_file: str, value_type: str = "mean"):
    if plt is None or matrix.size == 0:
        return
    plt.figure(figsize=(10, 6))
    if value_type == "mean":
        vmax = float(np.nanmax(np.abs(matrix))) if matrix.size > 0 else 1.0; vmax = vmax if vmax > 1e-6 else 1.0; vmin = -vmax; cmap = "RdBu_r"
    else:
        vmin = 0.0; vmax = float(np.percentile(matrix, 95)) if matrix.size > 0 else 1.0; vmax = vmax if vmax > 1e-6 else float(np.nanmax(matrix) + 1e-6); cmap = "viridis"
    im = plt.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("set index"); plt.ylabel("latent dimension"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_file, dpi=220); plt.close()


def _plot_overlay_2d(orig: np.ndarray, recon: np.ndarray, out_file: str):
    if plt is None or orig.size == 0 or recon.size == 0:
        return
    if StandardScaler is not None:
        try:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(orig)
            orig_s = scaler.transform(orig); recon_s = scaler.transform(recon)
        except Exception:
            orig_s, recon_s = orig, recon
    else:
        orig_s, recon_s = orig, recon
    try:
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            orig_emb = reducer.fit_transform(orig_s); recon_emb = reducer.transform(recon_s)
        elif PCA is not None:
            pca = PCA(n_components=2)
            orig_emb = pca.fit_transform(orig_s); recon_emb = pca.transform(recon_s)
        else:
            orig_emb = orig_s[:, :2]; recon_emb = recon_s[:, :2]
    except Exception:
        if PCA is not None:
            pca = PCA(n_components=2)
            orig_emb = pca.fit_transform(orig_s); recon_emb = pca.transform(recon_s)
        else:
            orig_emb = orig_s[:, :2]; recon_emb = recon_s[:, :2]
    n = min(len(orig_emb), len(recon_emb)); orig_emb = orig_emb[:n]; recon_emb = recon_emb[:n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(orig_emb[:, 0], orig_emb[:, 1], s=8, alpha=0.7, c="tab:blue", label="Original")
    ax.scatter(recon_emb[:, 0], recon_emb[:, 1], s=8, alpha=0.7, c="tab:orange", label="Reconstruction")
    ax.set_title("Original vs Reconstruction (2D projection)"); ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.legend(frameon=True); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_file, dpi=260); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate SetVAE pretraining: collapse + reconstruction")
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to pretraining checkpoint (.ckpt)")
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--batch_size", type=int, default=getattr(cfg, "batch_size", 8))
    ap.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 0))
    ap.add_argument("--num_eval_batches", type=int, default=50, help="Number of batches to estimate KL stats")
    ap.add_argument("--num_vis_samples", type=int, default=2, help="#samples for detailed recon/heatmaps")
    ap.add_argument("--output_dir", type=str, default=None, help="If not set, saves to <ckpt_version_dir>/eval")
    ap.add_argument("--active_thresholds", type=float, nargs="*", default=[0.001, 0.005, 0.01, 0.02, 0.05], help="KL thresholds for active ratio diagnostics")
    ap.add_argument("--topk_plot", type=int, default=20, help="Top-K bars to show for KL and |mu| means")
    ap.add_argument("--collapse_kl_thresh", type=float, default=0.001, help="Dim considered collapsed if KL below this")
    ap.add_argument("--collapse_mu_abs_thresh", type=float, default=0.02, help="Dim considered collapsed if mean |mu| below this")
    ap.add_argument("--collapse_var_tol", type=float, default=0.05, help="Dim considered collapsed if |mean var - 1| below this tolerance")
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--active_threshold", type=float, default=0.01, help="KL threshold for active units")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.output_dir is None:
        ckpt_abs = os.path.abspath(args.checkpoint); ckpt_dir = os.path.dirname(ckpt_abs); base = os.path.basename(ckpt_dir)
        if base == "checkpoints":
            version_dir = os.path.dirname(ckpt_dir)
        elif base.startswith("version_"):
            version_dir = ckpt_dir
        else:
            version_dir = os.path.dirname(ckpt_dir)
        args.output_dir = os.path.join(version_dir, "eval")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    state = _load_state_dict(args.checkpoint)
    model = _build_model(lr=getattr(cfg, "lr", 3e-4))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dm = DataModule(saved_dir=args.data_dir, params_map_path=None, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False, smoke=False, apply_A=False)
    dm.setup()
    dl = dm.val_dataloader() if args.split == "valid" else dm.test_dataloader()

    print("Computing KL per-dimension stats across dataset ...")
    kl_dim_t = compute_kl_dataset(model, dl, num_batches=args.num_eval_batches)
    kl_dim = kl_dim_t.detach().cpu().numpy().astype(np.float32)
    total_kl = float(kl_dim.sum())
    active_ratio = float(np.mean(kl_dim > args.active_threshold)) if kl_dim.size > 0 else 0.0
    active_ratios_multi = compute_active_ratios(kl_dim, args.active_thresholds)
    sorted_vals = np.sort(kl_dim)[::-1]; mass = np.cumsum(sorted_vals); denom = mass[-1] if mass.size > 0 else 1.0; frac = mass / (denom + 1e-8)
    dim90 = int(np.sum(frac < 0.90) + 1) if frac.size > 0 else 0
    dim95 = int(np.sum(frac < 0.95) + 1) if frac.size > 0 else 0
    eff_dims = compute_effective_dimensions(kl_dim)
    _plot_kl_hist_and_cdf(kl_dim, args.output_dir)
    _plot_bar_topk(kl_dim, args.topk_plot, title="Top-K KL per-dimension", out_file=os.path.join(args.output_dir, "kl_topk.png"), ylabel="KL (nats)")

    print("Computing posterior moments across dataset ...")
    moments = compute_posterior_moments_dataset(model, dl, num_batches=args.num_eval_batches)
    mu_abs_mean = moments["mu_abs_mean"]; mu_mean = moments["mu_mean"]; mu_std = moments["mu_std"]; var_mean = moments["var_mean"]; var_std = moments["var_std"]
    _plot_hist_generic(mu_abs_mean, "Mean |μ| per dimension", os.path.join(args.output_dir, "mu_abs_mean_hist.png"), "mean |μ|")
    _plot_hist_generic(var_mean, "Mean σ² per dimension", os.path.join(args.output_dir, "var_mean_hist.png"), "mean σ²")
    _plot_scatter_generic(mu_abs_mean, np.abs(var_mean - 1.0), "|μ| vs |σ²-1| per-dimension", os.path.join(args.output_dir, "muabs_vs_varminus1_scatter.png"), "mean |μ|", "|mean σ² - 1|")
    _plot_bar_topk(mu_abs_mean, args.topk_plot, title="Top-K mean |μ| per-dimension", out_file=os.path.join(args.output_dir, "mu_abs_topk.png"), ylabel="mean |μ|")
    collapsed_mask = (kl_dim < args.collapse_kl_thresh) & (mu_abs_mean < args.collapse_mu_abs_thresh) & (np.abs(var_mean - 1.0) <= args.collapse_var_tol)
    collapsed_indices = np.nonzero(collapsed_mask)[0].tolist(); collapsed_count = int(len(collapsed_indices))
    dim_csv = os.path.join(args.output_dir, "dim_diagnostics.csv")
    with open(dim_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv); writer.writerow(["dim", "kl", "mu_abs_mean", "mu_mean", "mu_std", "var_mean", "var_std", "collapsed"])
        for d in range(len(kl_dim)):
            writer.writerow([d, float(kl_dim[d]), float(mu_abs_mean[d]), float(mu_mean[d]), float(mu_std[d]), float(var_mean[d]), float(var_std[d]), bool(collapsed_mask[d])])

    print("Collecting per-sample visualizations and reconstruction metrics ...")
    dm_vis = DataModule(saved_dir=args.data_dir, params_map_path=None, batch_size=1, num_workers=0, pin_memory=False, smoke=False, apply_A=False)
    dm_vis.setup(); dl_vis = dm_vis.val_dataloader() if args.split == "valid" else dm_vis.test_dataloader()
    recon_results: List[Dict[str, float]] = []; vis_count = 0; sample_rows: List[List[object]] = []
    for i, batch in enumerate(dl_vis):
        if vis_count >= args.num_vis_samples:
            break
        mu_mat, var_mat = collect_mu_var_heatmaps(model, batch)
        _plot_heatmap(mu_mat, title=f"Sample {i} - μ across sets", out_file=os.path.join(args.output_dir, f"heatmap_mu_sample_{i}.png"), value_type="mean")
        _plot_heatmap(var_mat, title=f"Sample {i} - σ² across sets", out_file=os.path.join(args.output_dir, f"heatmap_var_sample_{i}.png"), value_type="var")
        orig, recon = collect_recon_events(model, batch)
        _plot_overlay_2d(orig, recon, out_file=os.path.join(args.output_dir, f"overlay_orig_recon_sample_{i}.png"))
        metrics = compute_recon_metrics(orig, recon)
        kd = compute_kl_dim_stats_for_batch(model, batch); kd_np = kd.detach().cpu().numpy().astype(np.float32)
        metrics["kl_total"] = float(kd_np.sum()); metrics["active_ratio"] = float(np.mean(kd_np > args.active_threshold)); metrics["sample_index"] = i
        recon_results.append(metrics); vis_count += 1
        sample_rows.append([i, metrics.get("kl_total", float("nan")), metrics.get("nn_l2_mean", float("nan")), metrics.get("nn_l2_median", float("nan")), metrics.get("nn_l2_p95", float("nan")), metrics.get("dir_cosine_mean", float("nan")), metrics.get("mag_mae", float("nan")), metrics.get("mag_rmse", float("nan")), metrics.get("mag_corr", float("nan")), metrics.get("scale_ratio", float("nan")), metrics.get("chamfer_l2_mean", float("nan")), metrics.get("active_ratio", float("nan"))])

    def _agg(key: str) -> float:
        vals = [r[key] for r in recon_results if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")
    recon_summary = {"num_samples": vis_count, "nn_l2_mean": _agg("nn_l2_mean"), "nn_l2_median": _agg("nn_l2_median"), "nn_l2_p95": _agg("nn_l2_p95"), "dir_cosine_mean": _agg("dir_cosine_mean"), "mag_mae": _agg("mag_mae"), "mag_rmse": _agg("mag_rmse"), "mag_corr": _agg("mag_corr"), "scale_ratio": _agg("scale_ratio"), "chamfer_l2_mean": _agg("chamfer_l2_mean"), "kl_total_mean": _agg("kl_total"), "active_ratio_mean": _agg("active_ratio")}

    def _corr(a_key: str, b_key: str) -> float:
        a = [m.get(a_key) for m in recon_results if isinstance(m.get(a_key), (int, float)) and math.isfinite(m.get(a_key))]
        b = [m.get(b_key) for m in recon_results if isinstance(m.get(b_key), (int, float)) and math.isfinite(m.get(b_key))]
        n = min(len(a), len(b));
        if n < 2:
            return float("nan")
        a = np.array(a[:n], dtype=np.float32); b = np.array(b[:n], dtype=np.float32)
        try:
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            return float("nan")
    sample_correlations = {"kl_total_vs_nn_l2_mean": _corr("kl_total", "nn_l2_mean"), "kl_total_vs_chamfer_l2_mean": _corr("kl_total", "chamfer_l2_mean"), "kl_total_vs_dir_cosine_mean": _corr("kl_total", "dir_cosine_mean"), "kl_total_vs_mag_mae": _corr("kl_total", "mag_mae"), "kl_total_vs_mag_rmse": _corr("kl_total", "mag_rmse")}

    sample_csv = os.path.join(args.output_dir, "per_sample_metrics.csv")
    with open(sample_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv); writer.writerow(["sample_index", "kl_total", "nn_l2_mean", "nn_l2_median", "nn_l2_p95", "dir_cosine_mean", "mag_mae", "mag_rmse", "mag_corr", "scale_ratio", "chamfer_l2_mean", "active_ratio"]); writer.writerows(sample_rows)

    collapse_flags = {"kl_active_ratio_below_0.1": active_ratio < 0.1, "total_kl_below_1e-2": total_kl < 1e-2, "risk": (active_ratio < 0.1) or (total_kl < 1e-2)}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "active_threshold": args.active_threshold,
        "active_thresholds": args.active_thresholds,
        "latent_dim": int(getattr(model, "latent_dim", len(kl_dim))),
        "kl": {"mean_per_dim": kl_dim.tolist(), "total_kl": total_kl, "active_ratio": active_ratio, "active_ratios": active_ratios_multi, "dim90": dim90, "dim95": dim95, "effective_dimensions": eff_dims},
        "posterior_moments": {
            "mu_abs_mean": mu_abs_mean.tolist(),
            "mu_mean": mu_mean.tolist(),
            "mu_std": mu_std.tolist(),
            "var_mean": var_mean.tolist(),
            "var_std": var_std.tolist(),
            "mu_abs_mean_stats": {"mean": float(np.mean(mu_abs_mean)) if mu_abs_mean.size > 0 else 0.0, "median": float(np.median(mu_abs_mean)) if mu_abs_mean.size > 0 else 0.0, "p95": float(np.percentile(mu_abs_mean, 95)) if mu_abs_mean.size > 0 else 0.0},
            "var_mean_stats": {"mean": float(np.mean(var_mean)) if var_mean.size > 0 else 0.0, "median": float(np.median(var_mean)) if var_mean.size > 0 else 0.0, "p95_abs_var_minus_1": float(np.percentile(np.abs(var_mean - 1.0), 95)) if var_mean.size > 0 else 0.0},
        },
        "reconstruction": {"summary": recon_summary, "details": recon_results},
        "collapsed_dims": {"criteria": {"kl_thresh": args.collapse_kl_thresh, "mu_abs_thresh": args.collapse_mu_abs_thresh, "var_tol": args.collapse_var_tol}, "count": collapsed_count, "indices": collapsed_indices},
        "sample_correlations": sample_correlations,
        "collapse_flags": collapse_flags,
    }
    out_json = os.path.join(args.output_dir, f"pretrain_eval_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {out_json}")


if __name__ == "__main__":
    main()

import json
import csv
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
    ap.add_argument("--output_dir", type=str, default=None, help="If not set, saves to <ckpt_version_dir>/eval")
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--active_threshold", type=float, default=0.01, help="KL threshold for active units")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve output directory: default to <ckpt_version_dir>/eval
    if args.output_dir is None:
        ckpt_abs = os.path.abspath(args.checkpoint)
        ckpt_dir = os.path.dirname(ckpt_abs)
        base = os.path.basename(ckpt_dir)
        if base == "checkpoints":
            version_dir = os.path.dirname(ckpt_dir)
        elif base.startswith("version_"):
            version_dir = ckpt_dir
        else:
            version_dir = os.path.dirname(ckpt_dir)
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

