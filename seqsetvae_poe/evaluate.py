#!/usr/bin/env python3
"""
Unified eval entrypoint with --stage {A,B,C} (legacy) or new branches.
- Pretraining eval (formerly A/B): collapse + reconstruction metrics
- Discriminative eval (formerly C): classifier metrics on test split
"""
from __future__ import annotations
import argparse
import sys
import os
import json
import csv
import math
import re
from typing import Dict, List, Tuple, Optional

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar

import numpy as np
import torch
import torch.nn.functional as F
import warnings
import random
import glob
import pandas as pd

import config as cfg  # type: ignore
from dataset import MortalityDataModule, DataModule, PatientDataset, _detect_vcols  # type: ignore
from model import MortalityClassifier, _load_state_dict, _build_poe_from_state, SetVAEOnlyPretrain, PoESeqSetVAEPretrain  # type: ignore
from modules import recon_loss as _set_recon_loss  # type: ignore

# Optional visualization deps for A/B
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore
try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover - optional
    PCA = None  # type: ignore
    StandardScaler = None  # type: ignore
    NearestNeighbors = None  # type: ignore
try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional
    umap = None  # type: ignore

# Optional metrics for OOD/consistency eval
try:
    from sklearn.metrics import roc_auc_score as _roc_auc, average_precision_score as _avg_prec  # type: ignore
    from sklearn.metrics import roc_curve as _roc_curve, precision_recall_curve as _pr_curve  # type: ignore
except Exception:  # pragma: no cover - optional
    _roc_auc = None  # type: ignore
    _avg_prec = None  # type: ignore
    _roc_curve = None  # type: ignore
    _pr_curve = None  # type: ignore

# Targeted warning filters to keep logs clean without changing behavior
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to 1 by setting random_state\. Use no seed for parallelism\.",
    category=UserWarning,
    module="umap.umap_",
)
warnings.filterwarnings(
    "ignore",
    message=r"Graph is not fully connected, spectral embedding may not work as expected\.",
    category=UserWarning,
    module="sklearn.manifold._spectral_embedding",
)

# ---------------- Stage A/B helper functions (pretraining eval) ---------------- #

def _load_state_dict_pretrain(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict")
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


def _detect_num_flows_in_state(state: Dict[str, torch.Tensor]) -> int:
    flow_indices = set()
    for k in state.keys():
        if ".flows." in k:
            m = re.search(r"\.flows\.(\d+)\.", k)
            if m:
                try:
                    flow_indices.add(int(m.group(1)))
                except Exception:
                    pass
    if not flow_indices:
        return 0
    return max(flow_indices) + 1


def _infer_dims_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Best-effort inference of input/reduced/latent dims from a pretraining state_dict."""
    dims: Dict[str, int] = {}
    # Prefer explicit linear reducer if present: weight shape [R, D]
    w = state.get("set_encoder.dim_reducer.weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        R, D = w.shape
        dims["reduced_dim"] = int(R)
        dims["input_dim"] = int(D)
    # Embed/out layers always exist; their shapes imply dims
    ew = state.get("set_encoder.embed.0.weight", None)  # [latent_dim, embed_input]
    if isinstance(ew, torch.Tensor) and ew.dim() == 2:
        ld, embed_in = ew.shape
        dims["latent_dim"] = int(ld)
        # If reducer absent, embed_in corresponds to effective input/reduced dim
        dims.setdefault("reduced_dim", int(embed_in))
        dims.setdefault("input_dim", int(embed_in))
    ow = state.get("set_encoder.out.weight", None)  # [out_output, latent_dim]
    if isinstance(ow, torch.Tensor) and ow.dim() == 2:
        out_out, ld2 = ow.shape
        dims["latent_dim"] = int(ld2)
        # If reducer absent, out_out corresponds to effective input/reduced dim
        dims.setdefault("reduced_dim", int(out_out))
        dims.setdefault("input_dim", int(out_out))
    # Fall back to mu_logvar shapes to confirm latent dim
    mlw = state.get("set_encoder.mu_logvar.4.weight", None)  # [2*latent_dim, latent_dim]
    if isinstance(mlw, torch.Tensor) and mlw.dim() == 2:
        dims["latent_dim"] = int(mlw.shape[1])
    return dims


def _build_model_pretrain(
    ckpt_type: str,
    lr: float,
    dims: Dict[str, int],
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.nn.Module:
    if ckpt_type == "poe":
        model = PoESeqSetVAEPretrain(
            input_dim=int(dims.get("input_dim", getattr(cfg, "input_dim", 768))),
            reduced_dim=int(dims.get("reduced_dim", getattr(cfg, "reduced_dim", 256))),
            latent_dim=int(dims.get("latent_dim", getattr(cfg, "latent_dim", 128))),
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
        num_flows = _detect_num_flows_in_state(state) if state is not None else 0
        use_flows = num_flows > 0
        model = SetVAEOnlyPretrain(
            input_dim=int(dims.get("input_dim", getattr(cfg, "input_dim", 768))),
            reduced_dim=int(dims.get("reduced_dim", getattr(cfg, "reduced_dim", 256))),
            latent_dim=int(dims.get("latent_dim", getattr(cfg, "latent_dim", 128))),
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
            use_flows=use_flows,
            num_flows=num_flows,
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
    d_ro, _ = _nn_from_a_to_b(recon, orig)
    d_or, _ = _nn_from_a_to_b(orig, recon)
    if d_ro.size == 0 or d_or.size == 0:
        return {"error": 1.0}
    chamfer = float(np.mean(d_ro**2) + np.mean(d_or**2)) / 2.0
    metrics["chamfer_l2_mean"] = chamfer
    metrics.update(
        {
            "nn_l2_mean": float(np.mean(d_ro)),
            "nn_l2_median": float(np.median(d_ro)),
            "nn_l2_p95": float(np.percentile(d_ro, 95)),
        }
    )
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
        metrics["mag_rmse"] = float(
            np.sqrt(np.mean((norm_recon - norm_orig_matched) ** 2))
        )
        try:
            if np.std(norm_recon) > 1e-8 and np.std(norm_orig_matched) > 1e-8:
                metrics["mag_corr"] = float(
                    np.corrcoef(norm_recon, norm_orig_matched)[0, 1]
                )
        except Exception:
            pass
    metrics["scale_ratio"] = float(
        (np.mean(norm_recon) + 1e-8) / (np.mean(norm_orig) + 1e-8)
    )
    for th in [0.25, 0.5, 1.0, 2.0]:
        metrics[f"coverage@{th}"] = float(np.mean(d_ro <= th))
    metrics["orig_norm_mean"] = float(np.mean(norm_orig))
    metrics["recon_norm_mean"] = float(np.mean(norm_recon))
    metrics["orig_norm_std"] = float(np.std(norm_orig))
    metrics["recon_norm_std"] = float(np.std(norm_recon))
    return metrics


@torch.no_grad()
def compute_kl_dim_stats_for_batch(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    if hasattr(model, "_compute_kl_dim_stats") and callable(
        getattr(model, "_compute_kl_dim_stats")
    ):
        try:
            return model._compute_kl_dim_stats(batch)
        except Exception:
            pass
    var, val, minutes, set_id = (
        batch["var"],
        batch["val"],
        batch["minute"],
        batch["set_id"],
    )
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    if hasattr(model, "_split_sets"):
        all_sets = model._split_sets(
            var, val, minutes, set_id, padding_mask, carry_mask
        )
    else:
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
            idx_splits = torch.split(
                torch.arange(len(s), device=s.device), [int(c) for c in counts]
            )
            patient_sets = []
            for idx in idx_splits:
                patient_sets.append(
                    {
                        "var": v[idx].unsqueeze(0),
                        "val": x[idx].unsqueeze(0),
                    }
                )
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
                encoder = getattr(model, "setvae", None)
                if encoder is None or not hasattr(encoder, "set_encoder"):
                    return torch.zeros(latent_dim, device=device)
                encoder = encoder.set_encoder
            if getattr(encoder, "dim_reducer", None) is not None:
                reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
            else:
                reduced = s["var"]
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            v_norm = reduced / (norms + 1e-8)
            x_clean = v_norm * s["val"]
            z_list, _ = encoder.encode(x_clean)
            _z, mu, logvar = z_list[-1]
            kl_dim = 0.5 * (
                logvar.exp().squeeze(0).squeeze(0)
                + mu.squeeze(0).squeeze(0).pow(2)
                - 1.0
                - logvar.squeeze(0).squeeze(0)
            )
            if kl_dim_sum is None:
                kl_dim_sum = kl_dim
            else:
                kl_dim_sum = kl_dim_sum + kl_dim
            count += 1

    if kl_dim_sum is None or count == 0:
        return torch.zeros(latent_dim, device=device)
    return kl_dim_sum / float(count)


@torch.no_grad()
def compute_kl_dataset(
    model: torch.nn.Module, dataloader, num_batches: Optional[int] = None
) -> torch.Tensor:
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
def compute_posterior_moments_dataset(
    model: torch.nn.Module, dataloader, num_batches: Optional[int] = None
) -> Dict[str, np.ndarray]:
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

        var, val, minutes, set_id = (
            batch["var"],
            batch["val"],
            batch["minute"],
            batch["set_id"],
        )
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        if hasattr(model, "_split_sets"):
            all_sets = model._split_sets(
                var, val, minutes, set_id, padding_mask, carry_mask
            )
        else:
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
                idx_splits = torch.split(
                    torch.arange(len(s), device=s.device), [int(c) for c in counts]
                )
                patient_sets = []
                for idx in idx_splits:
                    patient_sets.append(
                        {
                            "var": v[idx].unsqueeze(0),
                            "val": x[idx].unsqueeze(0),
                        }
                    )
                all_sets.append(patient_sets)

        for sets in all_sets:
            for s in sets:
                if hasattr(model, "set_encoder"):
                    encoder = model.set_encoder
                else:
                    encoder = getattr(model, "setvae", None)
                    if encoder is None or not hasattr(encoder, "set_encoder"):
                        continue
                    encoder = encoder.set_encoder
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
                    mu_abs_sum = mu_vec.abs().clone()
                    mu_sum = mu_vec.clone()
                    mu_sq_sum = mu_vec.pow(2).clone()
                    var_sum = var_vec.clone()
                    var_sq_sum = var_vec.pow(2).clone()
                else:
                    mu_abs_sum = mu_abs_sum + mu_vec.abs()
                    mu_sum = mu_sum + mu_vec
                    mu_sq_sum = mu_sq_sum + mu_vec.pow(2)
                    var_sum = var_sum + var_vec
                    var_sq_sum = var_sq_sum + var_vec.pow(2)
                count_sets += 1

    if count_sets == 0 or mu_abs_sum is None or mu_sum is None or mu_sq_sum is None or var_sum is None or var_sq_sum is None:
        latent_dim = int(getattr(model, "latent_dim", 128))
        zeros = np.zeros(latent_dim, dtype=np.float32)
        return {
            "mu_abs_mean": zeros,
            "mu_mean": zeros,
            "mu_std": zeros,
            "var_mean": zeros,
            "var_std": zeros,
        }

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

    return {
        "mu_abs_mean": mu_abs_mean,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "var_mean": var_mean,
        "var_std": var_std,
    }


def compute_active_ratios(kl_dim: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for th in thresholds:
        ratios[str(th)] = float(np.mean(kl_dim > th)) if kl_dim.size > 0 else 0.0
    return ratios


def compute_effective_dimensions(kl_dim: np.ndarray) -> Dict[str, float]:
    if kl_dim.size == 0:
        return {"participation_ratio": 0.0, "entropy_dim": 0.0}
    s1 = float(np.sum(kl_dim))
    s2 = float(np.sum(kl_dim ** 2))
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
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _plot_scatter_generic(x: np.ndarray, y: np.ndarray, title: str, out_file: str, xlabel: str, ylabel: str):
    if plt is None or x.size == 0 or y.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.7, c="tab:purple")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _plot_bar_topk(values: np.ndarray, k: int, title: str, out_file: str, xlabel: str = "ranked dims", ylabel: str = "value"):
    if plt is None or values.size == 0:
        return
    idx = np.argsort(values)[::-1]
    vals = values[idx][:k]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(vals)), vals, color="tab:blue", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


@torch.no_grad()
def collect_mu_var_heatmaps(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    var, val, minutes, set_id = (
        batch["var"],
        batch["val"],
        batch["minute"],
        batch["set_id"],
    )
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
    encoder = (
        model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    )
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
def _collect_recon_events_poe_from_sets(
    model: torch.nn.Module, sets: List[Dict[str, torch.Tensor]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect original and reconstruction arrays using PoE dynamics + decoder.

    This mirrors the PoE model's temporal prior and posterior fusion, decoding
    each set from the evolving hidden state h_t. Returns (orig, recon) in the
    reduced/input space expected by compute_recon_metrics.
    """
    # Validate minimal PoE attributes
    if not hasattr(model, "set_encoder") or not hasattr(model, "decoder") or not hasattr(model, "prior_head"):
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    device = next(model.parameters()).device

    # Pre-encode per-set q_x(z|x)
    mu_qx_list: List[torch.Tensor] = []
    logvar_qx_list: List[torch.Tensor] = []
    set_aggs: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    for s in sets:
        z_list_enc, enc_tokens = model.set_encoder.encode_from_var_val(s["var"], s["val"])  # [1,1,D]
        _z, mu_qx, logvar_qx = z_list_enc[-1]
        mu_qx_list.append(mu_qx.squeeze(1))       # [1,D]
        logvar_qx_list.append(logvar_qx.squeeze(1))  # [1,D]
        pos_list.append(s["set_time"].float())
        if getattr(model, "poe_mode", "conditional") == "conditional":
            set_aggs.append(enc_tokens.mean(dim=1) if enc_tokens is not None else mu_qx)

    if not mu_qx_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    mu_qx = torch.stack(mu_qx_list, dim=1)  # [1,S,D]
    logvar_qx = torch.stack(logvar_qx_list, dim=1)  # [1,S,D]
    minutes_seq = torch.stack(pos_list, dim=1)
    if minutes_seq.dim() == 1:
        minutes_seq = minutes_seq.unsqueeze(0)

    B = 1
    latent_dim = int(getattr(model, "latent_dim", mu_qx.size(-1)))
    h = torch.zeros(B, latent_dim, device=device)
    time_emb = model._relative_time_bucket_embedding(minutes_seq) if hasattr(model, "_relative_time_bucket_embedding") else torch.zeros_like(mu_qx)
    if getattr(model, "poe_mode", "conditional") == "conditional" and set_aggs:
        set_aggs_t = torch.stack([set_aggs[t] for t in range(mu_qx.size(1))], dim=1)  # [1,S,D]

    orig_cat: List[np.ndarray] = []
    recon_cat: List[np.ndarray] = []

    # Step through sets
    S = mu_qx.size(1)
    for t in range(S):
        prior_params_t = model.prior_head(h)
        mu_p_t, logvar_p_t = prior_params_t.chunk(2, dim=-1)  # [1,D]

        # Optional adaptive observation gate
        beta_t = None
        if getattr(model, "use_adaptive_poe", False):
            mu_qx_t = mu_qx[:, t, :]
            logvar_qx_t = logvar_qx[:, t, :]
            var_sum = (logvar_p_t.exp() + logvar_qx_t.exp()).clamp(min=1e-8)
            d2 = ((mu_qx_t - mu_p_t) ** 2 / var_sum).mean(dim=-1, keepdim=True)
            delta_H = (logvar_p_t - logvar_qx_t).mean(dim=-1, keepdim=True)
            if minutes_seq.size(1) > 0:
                dt = minutes_seq[:, t : t + 1] - (minutes_seq[:, t - 1 : t] if t > 0 else minutes_seq[:, t : t + 1])
            else:
                dt = torch.zeros_like(d2)
            log_dt1p = torch.log1p(dt.clamp(min=0.0))
            if hasattr(model, "obs_gate"):
                gate_inp = torch.cat([d2, delta_H, log_dt1p], dim=-1)
                beta_t = float(model.poe_beta_min) + (float(model.poe_beta_max) - float(model.poe_beta_min)) * torch.sigmoid(model.obs_gate(gate_inp))

        # Form posterior
        poe_mode = getattr(model, "poe_mode", "conditional")
        if poe_mode == "conditional" and 'set_aggs_t' in locals():
            set_agg_t = set_aggs_t[:, t, :]
            like_inp = torch.cat([set_agg_t, mu_p_t, logvar_p_t], dim=-1)
            like_out = model.like_head(like_inp)
            log_prec_like_t, h_like_t = like_out.chunk(2, dim=-1)
            prec_like_t = F.softplus(log_prec_like_t) + 1e-4
            Lambda_p_t = torch.exp(-logvar_p_t)
            if beta_t is not None:
                prec_like_t = prec_like_t * beta_t
                h_like_t = h_like_t * beta_t
            Lambda_post_t = Lambda_p_t + prec_like_t
            h_post_t = Lambda_p_t * mu_p_t + h_like_t
            mu_post_t = h_post_t / (Lambda_post_t + 1e-8)
        else:
            # naive PoE
            mu_qx_t = mu_qx[:, t, :]
            logvar_qx_t = logvar_qx[:, t, :]
            var_qx = logvar_qx_t.exp()
            var_p = logvar_p_t.exp()
            if beta_t is not None:
                var_qx = var_qx / beta_t
            var_post = 1.0 / (1.0 / var_qx + 1.0 / var_p)
            mu_post_t = var_post * (mu_qx_t / var_qx + mu_p_t / var_p)

        # Decode current set from h_t (after updating with posterior mean)
        N_t = int(sets[t]["var"].size(1))
        # Update GRU state with posterior + time embedding first (to mirror training)
        if time_emb is not None and isinstance(time_emb, torch.Tensor) and time_emb.numel() > 0:
            h = model.gru_cell(mu_post_t + time_emb[:, t, :], h)
        else:
            h = model.gru_cell(mu_post_t, h)
        recon_t = model.decoder(h, N_t, noise_std=0.0)

        # Build target in reduced space
        if getattr(model.set_encoder, "dim_reducer", None) is not None:
            reduced = model.set_encoder.dim_reducer(sets[t]["var"])  # [1,N,R]
        else:
            reduced = sets[t]["var"]
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        target_x = reduced / (norms + 1e-8)
        target_x = target_x * sets[t]["val"]

        orig_cat.append(_to_numpy(target_x.squeeze(0)))
        recon_cat.append(_to_numpy(recon_t.squeeze(0)))

    orig = np.concatenate(orig_cat, axis=0) if orig_cat else np.zeros((0, 0), dtype=np.float32)
    recon = np.concatenate(recon_cat, axis=0) if recon_cat else np.zeros((0, 0), dtype=np.float32)
    return orig, recon


@torch.no_grad()
def collect_recon_events(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    recon_from: str = "setvae",
) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    var, val, minutes, set_id = (
        batch["var"],
        batch["val"],
        batch["minute"],
        batch["set_id"],
    )
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    if hasattr(model, "_split_sets"):
        pats = model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)
    else:
        pats = [[{"var": var[0:1, :1, :], "val": val[0:1, :1, :], "set_time": torch.tensor([0.0], device=var.device)}]]
    if len(pats) == 0 or len(pats[0]) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    sets = pats[0]

    # If requested, try PoE-based reconstruction for models that support it
    if recon_from in {"poe", "auto"}:
        is_poe = hasattr(model, "decoder") and hasattr(model, "prior_head")
        if is_poe:
            return _collect_recon_events_poe_from_sets(model, sets)
    encoder = (
        model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    )
    if encoder is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    orig_cat: List[np.ndarray] = []
    recon_cat: List[np.ndarray] = []

    for s in sets:
        if getattr(encoder, "dim_reducer", None) is not None:
            reduced = encoder.dim_reducer(s["var"])  # [1,N,R]
        else:
            reduced = s["var"]
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        v_norm = reduced / (norms + 1e-8)
        x_target = v_norm * s["val"]
        z_list, _ = encoder.encode(x_target)
        recon = encoder.decode(
            z_list, target_n=x_target.size(1), use_mean=True, noise_std=0.0
        )
        orig_cat.append(_to_numpy(x_target.squeeze(0)))
        recon_cat.append(_to_numpy(recon.squeeze(0)))

    orig = (
        np.concatenate(orig_cat, axis=0) if orig_cat else np.zeros((0, 0), dtype=np.float32)
    )
    recon = (
        np.concatenate(recon_cat, axis=0) if recon_cat else np.zeros((0, 0), dtype=np.float32)
    )
    return orig, recon


# ---------------- Consistency & OOD utilities (mask invariance, scramble) ---------------- #

@torch.no_grad()
def _get_encoder_from_model(model: torch.nn.Module):
    if hasattr(model, "set_encoder"):
        return model.set_encoder
    enc = getattr(model, "setvae", None)
    if enc is not None and hasattr(enc, "set_encoder"):
        return enc.set_encoder
    return None


@torch.no_grad()
def _extract_sets_from_batch(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> List[List[Dict[str, torch.Tensor]]]:
    var, val, minutes, set_id = (
        batch["var"],
        batch["val"],
        batch["minute"],
        batch["set_id"],
    )
    padding_mask = batch.get("padding_mask", None)
    carry_mask = batch.get("carry_mask", None)
    if hasattr(model, "_split_sets"):
        return model._split_sets(var, val, minutes, set_id, padding_mask, carry_mask)  # type: ignore
    # Fallback best-effort split: consecutive identical set_id values define a set
    all_sets: List[List[Dict[str, torch.Tensor]]] = []
    B = var.size(0)
    for b in range(B):
        v = var[b]
        x = val[b]
        s = set_id[b]
        if padding_mask is not None:
            mask = ~padding_mask[b]
            v, x, s = v[mask], x[mask], s[mask]
        if s.dim() > 1:
            s = s.squeeze(-1)
        _, counts = torch.unique_consecutive(s.long(), return_counts=True)
        idx_splits = torch.split(torch.arange(len(s), device=s.device), [int(c) for c in counts])
        patient_sets: List[Dict[str, torch.Tensor]] = []
        for idx in idx_splits:
            patient_sets.append({"var": v[idx].unsqueeze(0), "val": x[idx].unsqueeze(0)})
        all_sets.append(patient_sets)
    return all_sets


@torch.no_grad()
def _encode_mu_logvar(encoder, var: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Shapes: var [1,N,D], val [1,N,1]
    if getattr(encoder, "dim_reducer", None) is not None:
        reduced = encoder.dim_reducer(var)
    else:
        reduced = var
    norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
    x = (reduced / (norms + 1e-8)) * val
    z_list, _ = encoder.encode(x)
    _z, mu, logvar = z_list[-1]
    return mu.squeeze(0).squeeze(0), logvar.squeeze(0).squeeze(0)


@torch.no_grad()
def _w2_diag(mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    # 2-Wasserstein distance squared between diagonal Gaussians
    var1 = logvar1.exp().clamp(min=1e-8)
    var2 = logvar2.exp().clamp(min=1e-8)
    term_mean = (mu1 - mu2).pow(2).sum()
    term_cov = (var1.sqrt() - var2.sqrt()).pow(2).sum()
    return term_mean + term_cov


@torch.no_grad()
def _poe_combine_diag(
    mu_a: torch.Tensor, logvar_a: torch.Tensor, mu_b: torch.Tensor, logvar_b: torch.Tensor, prior_prec: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Combine two posteriors approximately: q_AB ∝ q_A * q_B / p, with p=N(0,I)
    var_a = logvar_a.exp().clamp(min=1e-8)
    var_b = logvar_b.exp().clamp(min=1e-8)
    prec_a = 1.0 / var_a
    prec_b = 1.0 / var_b
    prec = (prec_a + prec_b - prior_prec)
    prec = torch.clamp(prec, min=1e-6)
    mu = (prec_a * mu_a + prec_b * mu_b) / prec
    logvar = torch.log(1.0 / prec)
    return mu, logvar


@torch.no_grad()
def evaluate_mask_consistency(
    model: torch.nn.Module,
    dataloader,
    mask_ratios: List[float] = [0.3, 0.5, 0.7],
    num_sets: int = 200,
    repeats: int = 3,
    disjoint: bool = True,
) -> Dict[str, Dict[str, float]]:
    encoder = _get_encoder_from_model(model)
    if encoder is None:
        return {}
    device = next(model.parameters()).device
    rng = random.Random(123)

    # Accumulators per ratio
    acc: Dict[float, Dict[str, List[float]]] = {
        float(r): {"w2_AB": [], "poe_gap": [], "var_shrink_ok": []} for r in mask_ratios
    }

    seen = 0
    for batch in dataloader:
        # Move tensors
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        all_sets = _extract_sets_from_batch(model, batch)
        for sets in all_sets:
            for s in sets:
                var = s["var"].to(device)
                val = s["val"].to(device)
                N = int(var.size(1))
                if N < 4:
                    continue
                # Pre-encode union (full set) once
                mu_full, logvar_full = _encode_mu_logvar(encoder, var, val)
                for r in mask_ratios:
                    k = max(1, int(round(float(r) * N)))
                    for _ in range(repeats):
                        idx = list(range(N))
                        rng.shuffle(idx)
                        if disjoint:
                            idx_a = sorted(idx[:k])
                            idx_b = sorted(idx[k : 2 * k]) if 2 * k <= N else sorted(idx[-k:])
                        else:
                            idx_a = sorted([i for i in idx[:k]])
                            rng.shuffle(idx)
                            idx_b = sorted([i for i in idx[:k]])
                        if len(idx_a) == 0 or len(idx_b) == 0:
                            continue
                        idx_u = sorted(set(idx_a) | set(idx_b))

                        va = var[:, idx_a, :]
                        xa = val[:, idx_a, :]
                        vb = var[:, idx_b, :]
                        xb = val[:, idx_b, :]
                        vu = var[:, idx_u, :]
                        xu = val[:, idx_u, :]

                        mu_a, logvar_a = _encode_mu_logvar(encoder, va, xa)
                        mu_b, logvar_b = _encode_mu_logvar(encoder, vb, xb)
                        mu_u, logvar_u = _encode_mu_logvar(encoder, vu, xu)

                        w2_ab = float(_w2_diag(mu_a, logvar_a, mu_b, logvar_b).item())
                        mu_p, logvar_p = _poe_combine_diag(mu_a, logvar_a, mu_b, logvar_b)
                        poe_gap = float(_w2_diag(mu_p, logvar_p, mu_u, logvar_u).item())

                        var_a = float(logvar_a.exp().mean().item())
                        var_b = float(logvar_b.exp().mean().item())
                        var_u = float(logvar_u.exp().mean().item())
                        shrink_ok = 1.0 if var_u <= min(var_a, var_b) + 1e-6 else 0.0

                        acc[float(r)]["w2_AB"].append(w2_ab)
                        acc[float(r)]["poe_gap"].append(poe_gap)
                        acc[float(r)]["var_shrink_ok"].append(shrink_ok)

                seen += 1
                if seen >= num_sets:
                    break
            if seen >= num_sets:
                break
        if seen >= num_sets:
            break

    out: Dict[str, Dict[str, float]] = {}
    for r, m in acc.items():
        def _m(name: str) -> float:
            arr = m.get(name, [])
            return float(np.mean(arr)) if len(arr) > 0 else float("nan")
        out[str(r)] = {
            "w2_AB_mean": _m("w2_AB"),
            "poe_gap_mean": _m("poe_gap"),
            "var_shrink_ok_frac": _m("var_shrink_ok"),
        }
    return out


@torch.no_grad()
def _per_set_elbo_score(encoder, var: torch.Tensor, val: torch.Tensor) -> float:
    # Compute reconstruction loss (Chamfer) + KL(q||p) with p=N(0,I)
    if getattr(encoder, "dim_reducer", None) is not None:
        reduced = encoder.dim_reducer(var)
    else:
        reduced = var
    norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
    x_target = (reduced / (norms + 1e-8)) * val
    z_list, _ = encoder.encode(x_target)
    recon = encoder.decode(z_list, target_n=int(x_target.size(1)), use_mean=True, noise_std=0.0)
    rec = _set_recon_loss(recon, x_target)
    _z, mu, logvar = z_list[-1]
    mu = mu.squeeze(0).squeeze(0)
    logvar = logvar.squeeze(0).squeeze(0)
    kl = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar)
    return float((rec + kl).item())


@torch.no_grad()
def evaluate_ood_scramble(
    model: torch.nn.Module,
    dataloader,
    num_sets: int = 200,
    scheme: str = "sample_from_stats",
    # Optional resources for stats-based scrambling
    stats_map: Optional[Dict[str, Tuple[float, float]]] = None,
    vocab_event_names: Optional[List[str]] = None,
    vocab_dirs_np: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    encoder = _get_encoder_from_model(model)
    if encoder is None:
        return {}
    device = next(model.parameters()).device
    rng = np.random.default_rng(12345)

    scores: List[float] = []
    labels: List[int] = []  # 0 = real, 1 = scrambled
    seen = 0
    for batch in dataloader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        all_sets = _extract_sets_from_batch(model, batch)
        for sets in all_sets:
            for s in sets:
                var = s["var"].to(device)
                val = s["val"].to(device)
                N = int(var.size(1))
                if N < 4:
                    continue
                # Real score
                score_real = _per_set_elbo_score(encoder, var, val)
                scores.append(score_real)
                labels.append(0)

                # Scrambled score (single supported scheme)
                if scheme == "sample_from_stats":
                    # Require stats_map and vocab resources
                    if stats_map is None or vocab_event_names is None or vocab_dirs_np is None:
                        # If resources missing, skip this set
                        continue
                    else:
                        # Map each token's variable embedding to a variable name via cosine on unit directions
                        with torch.no_grad():
                            vdirs = var / (torch.norm(var, p=2, dim=-1, keepdim=True) + 1e-8)
                        vdirs_np = vdirs.squeeze(0).detach().cpu().numpy()  # [N,D]
                        # Cosine similarity matrix [N,V]
                        sims = np.matmul(vdirs_np, vocab_dirs_np.T)
                        sims = np.abs(sims)
                        best = np.argmax(sims, axis=1)  # [N]
                        # Sample per variable from original (raw) distribution ~ N(mean,std), then normalize
                        sampled = np.zeros((N,), dtype=np.float32)
                        for i in range(N):
                            name = str(vocab_event_names[best[i]])
                            m, s_std = stats_map.get(name, (0.0, 1.0))
                            s_eff = s_std if abs(s_std) > 1e-12 else 1.0
                            x_raw = float(rng.normal(loc=m, scale=s_eff))
                            x_norm = (x_raw - m) / s_eff
                            sampled[i] = np.float32(x_norm)
                        val_scr = torch.from_numpy(sampled.reshape(1, N, 1)).to(device)
                else:
                    # Unknown scheme -> skip
                    continue

                score_scr = _per_set_elbo_score(encoder, var, val_scr)
                scores.append(score_scr)
                labels.append(1)

                seen += 1
                if seen >= num_sets:
                    break
            if seen >= num_sets:
                break
        if seen >= num_sets:
            break

    if len(scores) == 0:
        return {}
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int32)
    # Higher score => more outlier-like (ELBO approx)
    # Compute AUROC/AUPRC if available
    try:
        auroc = float(_roc_auc(labels_np, scores_np)) if _roc_auc is not None else float("nan")
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(_avg_prec(labels_np, scores_np)) if _avg_prec is not None else float("nan")
    except Exception:
        auprc = float("nan")
    # Simple separation metrics
    real_mean = float(scores_np[labels_np == 0].mean()) if (labels_np == 0).any() else float("nan")
    scr_mean = float(scores_np[labels_np == 1].mean()) if (labels_np == 1).any() else float("nan")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "mean_score_real": real_mean,
        "mean_score_scrambled": scr_mean,
        "mean_gap": (scr_mean - real_mean) if np.isfinite(real_mean) and np.isfinite(scr_mean) else float("nan"),
    }


# ---------------- Visualization helpers for consistency & OOD ---------------- #

def _plot_hist(values: List[float], title: str, out_file: str, xlabel: str):
    if plt is None or len(values) == 0:
        return
    arr = np.asarray(values, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arr, bins=40, color="tab:blue", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def _plot_hist_overlay(a: List[float], b: List[float], labels: Tuple[str, str], title: str, out_file: str, xlabel: str):
    if plt is None or (len(a) == 0 and len(b) == 0):
        return
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = 40
    if len(arr_a) > 0:
        ax.hist(arr_a, bins=bins, color="tab:blue", alpha=0.55, label=labels[0])
    if len(arr_b) > 0:
        ax.hist(arr_b, bins=bins, color="tab:orange", alpha=0.55, label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


@torch.no_grad()
def visualize_mask_consistency(
    model: torch.nn.Module,
    dataloader,
    out_dir: str,
    mask_ratios: List[float],
    num_sets: int,
    repeats: int,
    disjoint: bool,
):
    encoder = _get_encoder_from_model(model)
    if encoder is None or plt is None:
        return
    device = next(model.parameters()).device
    rng = random.Random(123)

    store: Dict[float, Dict[str, List[float]]] = {float(r): {"w2_AB": [], "poe_gap": [], "var_shrink_ok": []} for r in mask_ratios}

    seen = 0
    for batch in dataloader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        all_sets = _extract_sets_from_batch(model, batch)
        for sets in all_sets:
            for s in sets:
                var = s["var"].to(device)
                val = s["val"].to(device)
                N = int(var.size(1))
                if N < 4:
                    continue
                # Pre-encode union once for speed
                mu_full, logvar_full = _encode_mu_logvar(encoder, var, val)
                for r in mask_ratios:
                    k = max(1, int(round(float(r) * N)))
                    for _ in range(repeats):
                        idx = list(range(N))
                        rng.shuffle(idx)
                        if disjoint:
                            idx_a = sorted(idx[:k])
                            idx_b = sorted(idx[k : 2 * k]) if 2 * k <= N else sorted(idx[-k:])
                        else:
                            idx_a = sorted(idx[:k])
                            rng.shuffle(idx)
                            idx_b = sorted(idx[:k])
                        if len(idx_a) == 0 or len(idx_b) == 0:
                            continue
                        idx_u = sorted(set(idx_a) | set(idx_b))

                        va, xa = var[:, idx_a, :], val[:, idx_a, :]
                        vb, xb = var[:, idx_b, :], val[:, idx_b, :]
                        vu, xu = var[:, idx_u, :], val[:, idx_u, :]

                        mu_a, log_a = _encode_mu_logvar(encoder, va, xa)
                        mu_b, log_b = _encode_mu_logvar(encoder, vb, xb)
                        mu_u, log_u = _encode_mu_logvar(encoder, vu, xu)

                        w2 = float(_w2_diag(mu_a, log_a, mu_b, log_b).item())
                        mu_p, log_p = _poe_combine_diag(mu_a, log_a, mu_b, log_b)
                        gap = float(_w2_diag(mu_p, log_p, mu_u, log_u).item())
                        var_a = float(log_a.exp().mean().item())
                        var_b = float(log_b.exp().mean().item())
                        var_u = float(log_u.exp().mean().item())
                        shrink_ok = 1.0 if var_u <= min(var_a, var_b) + 1e-6 else 0.0

                        store[float(r)]["w2_AB"].append(w2)
                        store[float(r)]["poe_gap"].append(gap)
                        store[float(r)]["var_shrink_ok"].append(shrink_ok)

                seen += 1
                if seen >= num_sets:
                    break
            if seen >= num_sets:
                break
        if seen >= num_sets:
            break

    # Per-ratio histograms
    for r, m in store.items():
        _plot_hist(m["w2_AB"], title=f"W2(A,B) histogram (mask={r:.2f})", out_file=os.path.join(out_dir, f"mask_consistency_w2_hist_r{int(r*100)}.png"), xlabel="W2 distance")
        _plot_hist(m["poe_gap"], title=f"W2(AB,U) histogram (mask={r:.2f})", out_file=os.path.join(out_dir, f"mask_consistency_poe_gap_hist_r{int(r*100)}.png"), xlabel="W2 distance")

    # Trends across ratios
    ratios_sorted = sorted(store.keys())
    mean_w2 = [float(np.mean(store[r]["w2_AB"])) if len(store[r]["w2_AB"]) else np.nan for r in ratios_sorted]
    mean_gap = [float(np.mean(store[r]["poe_gap"])) if len(store[r]["poe_gap"]) else np.nan for r in ratios_sorted]
    frac_shrink = [float(np.mean(store[r]["var_shrink_ok"])) if len(store[r]["var_shrink_ok"]) else np.nan for r in ratios_sorted]

    if plt is not None and len(ratios_sorted) > 0:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = ax1.twinx()
        ax1.plot(ratios_sorted, mean_w2, marker="o", color="tab:blue", label="mean W2(A,B)")
        ax1.plot(ratios_sorted, mean_gap, marker="s", color="tab:orange", label="mean W2(AB,U)")
        ax2.plot(ratios_sorted, frac_shrink, marker="^", color="tab:green", label="var shrink frac")
        ax1.set_xlabel("mask ratio")
        ax1.set_ylabel("distance (W2)")
        ax2.set_ylabel("fraction")
        ax1.grid(True, alpha=0.3)
        lines, labels = [], []
        for ax in (ax1, ax2):
            line, lab = ax.get_legend_handles_labels()
            lines += line; labels += lab
        ax1.legend(lines, labels, frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "mask_consistency_trends.png"), dpi=240)
        plt.close(fig)


@torch.no_grad()
def visualize_ood_scramble(
    model: torch.nn.Module,
    dataloader,
    out_dir: str,
    num_sets: int,
    scheme: str,
):
    encoder = _get_encoder_from_model(model)
    if encoder is None or plt is None:
        return
    device = next(model.parameters()).device
    rng = np.random.default_rng(12345)

    scores_real: List[float] = []
    scores_scr: List[float] = []
    seen = 0
    for batch in dataloader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        all_sets = _extract_sets_from_batch(model, batch)
        for sets in all_sets:
            for s in sets:
                var = s["var"].to(device)
                val = s["val"].to(device)
                N = int(var.size(1))
                if N < 4:
                    continue
                # Real
                scores_real.append(_per_set_elbo_score(encoder, var, val))
                # For now, do not visualize stats-based scramble without extra resources.
                # Skip plotting detailed scramble overlay to avoid misinterpretation.
                seen += 1
                if seen >= num_sets:
                    break
            if seen >= num_sets:
                break
        if seen >= num_sets:
            break

    # Overlay histogram
    _plot_hist_overlay(scores_real, scores_scr, ("real", "scrambled"), title="OOD scramble ELBO-like scores", out_file=os.path.join(out_dir, "ood_score_hist.png"), xlabel="recon+KL (higher=worse)")

    # ROC & PR curves (if sklearn available)
    if _roc_curve is not None and _pr_curve is not None:
        try:
            y = np.array([0] * len(scores_real) + [1] * len(scores_scr), dtype=np.int32)
            s = np.array(scores_real + scores_scr, dtype=np.float64)
            fpr, tpr, _ = _roc_curve(y, s)
            prec, rec, _ = _pr_curve(y, s)
            # ROC
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="tab:blue", label="ROC")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("OOD ROC (scrambled as positive)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "ood_roc.png"), dpi=240)
            plt.close(fig)
            # PR
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec, prec, color="tab:orange", label="PR")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("OOD PR (scrambled as positive)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "ood_pr.png"), dpi=240)
            plt.close(fig)
        except Exception:
            pass


@torch.no_grad()
def _visualize_per_set_masking_curves(
    model: torch.nn.Module,
    dataloader,
    out_dir: str,
    num_sets: int = 10,
    min_len: int = 2,
):
    """Plot per-set consistency curves by progressively masking tokens.

    For each selected set, compute the posterior on the full set as reference,
    then iteratively remove one token at a time following a random order. At each
    masked count k, compute consistency = 1 / (1 + W2(q_subset, q_full)).
    """
    encoder = _get_encoder_from_model(model)
    if encoder is None or plt is None:
        return
    device = next(model.parameters()).device
    rng = random.Random(4321)

    curves: List[Tuple[List[int], List[float], int]] = []
    collected = 0
    for batch in dataloader:
        # move to device
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        all_sets = _extract_sets_from_batch(model, batch)
        for sets in all_sets:
            for s in sets:
                var = s["var"].to(device)
                val = s["val"].to(device)
                N = int(var.size(1))
                # need at least 3 tokens to form a curve (mask up to N-1)
                if N < max(3, int(min_len)):
                    continue
                # reference posterior for full set
                mu_full, log_full = _encode_mu_logvar(encoder, var, val)
                order = list(range(N))
                rng.shuffle(order)
                xs: List[int] = []
                ys: List[float] = []
                # progressively mask first k tokens in 'order'; avoid empty set
                for masked in range(1, N):
                    remain_idx = sorted(order[masked:])
                    if len(remain_idx) == 0:
                        break
                    v_sub = var[:, remain_idx, :]
                    x_sub = val[:, remain_idx, :]
                    mu_sub, log_sub = _encode_mu_logvar(encoder, v_sub, x_sub)
                    w2 = float(_w2_diag(mu_sub, log_sub, mu_full, log_full).item())
                    consistency = 1.0 / (1.0 + w2)
                    xs.append(masked)
                    ys.append(consistency)
                if len(xs) > 0:
                    curves.append((xs, ys, N))
                    collected += 1
                if collected >= num_sets:
                    break
            if collected >= num_sets:
                break
        if collected >= num_sets:
            break

    if len(curves) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (xs, ys, N) in enumerate(curves):
        ax.plot(xs, ys, marker="o", alpha=0.9, label=f"set {i+1} (N={N})")
    ax.set_xlabel("masked token count")
    ax.set_ylabel("consistency 1/(1+W2(q_subset, q_full))")
    ax.set_title("Per-set masking consistency curves")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mask_consistency_per_set_curves.png"), dpi=240)
    plt.close(fig)


def _plot_kl_hist_and_cdf(kl_vec: np.ndarray, out_dir: str):
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(kl_vec, bins=40, color="tab:blue", alpha=0.85)
    ax.set_title("KL per-dimension (hist)")
    ax.set_xlabel("KL (nats)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kl_hist.png"), dpi=200)
    plt.close(fig)
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


def _plot_heatmap(
    matrix: np.ndarray, title: str, out_file: str, value_type: str = "mean"
):
    if plt is None or matrix.size == 0:
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
    im = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("set index")
    plt.ylabel("latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def _plot_overlay_2d(orig: np.ndarray, recon: np.ndarray, out_file: str):
    if plt is None or orig.size == 0 or recon.size == 0:
        return
    if StandardScaler is not None:
        try:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(orig)
            orig_s = scaler.transform(orig)
            recon_s = scaler.transform(recon)
        except Exception:
            orig_s, recon_s = orig, recon
    else:
        orig_s, recon_s = orig, recon
    try:
        if umap is not None:
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, init="random"
            )
            orig_emb = reducer.fit_transform(orig_s)
            recon_emb = reducer.transform(recon_s)
        elif PCA is not None:
            pca = PCA(n_components=2)
            orig_emb = pca.fit_transform(orig_s)
            recon_emb = pca.transform(recon_s)
        else:
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
    ax.scatter(
        orig_emb[:, 0], orig_emb[:, 1], s=8, alpha=0.7, c="tab:blue", label="Original"
    )
    ax.scatter(
        recon_emb[:, 0],
        recon_emb[:, 1],
        s=8,
        alpha=0.7,
        c="tab:orange",
        label="Reconstruction",
    )
    ax.set_title("Original vs Reconstruction (2D projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=260)
    plt.close(fig)


def _detect_key_column(df: pd.DataFrame) -> str:
    for c in ["Key", "variable", "event", "name", "key"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _load_global_vocab_embeddings(data_dir: str, event_emb_csv: Optional[str]):
    if event_emb_csv and os.path.isfile(event_emb_csv):
        df = pd.read_csv(event_emb_csv)
        key_col = _detect_key_column(df)
        num_cols = [
            c
            for c in df.columns
            if c != key_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(num_cols) == 0:
            raise ValueError("event_emb_csv has no numeric embedding columns")
        num_cols = sorted(num_cols, key=lambda x: (len(x), x))
        names = df[key_col].astype(str).tolist()
        vecs = df[num_cols].to_numpy(dtype=np.float32, copy=False)
        return names, vecs
    parts = ["train", "valid", "test"]
    files: List[str] = []
    for p in parts:
        pattern = os.path.join(data_dir, p, "*.parquet")
        files.extend(sorted(glob.glob(pattern)))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No parquet files under {data_dir}/(train|valid|test)"
        )
    sample = pd.read_parquet(files[0], engine="pyarrow")
    vcols = _detect_vcols(sample)
    keep_cols = ["variable"] + vcols
    last: Dict[str, np.ndarray] = {}
    for fp in files:
        dfp = pd.read_parquet(fp, columns=keep_cols, engine="pyarrow")
        dfp = dfp.drop_duplicates(subset=["variable"], keep="last")
        names = dfp["variable"].astype(str).tolist()
        arr = dfp[vcols].to_numpy(dtype=np.float32, copy=False)
        for n, v in zip(names, arr):
            last[n] = v
    names_all = sorted(last.keys())
    vecs_all = (
        np.stack([last[n] for n in names_all], axis=0).astype(np.float32, copy=False)
    )
    return names_all, vecs_all


@torch.no_grad()
def _encode_decode_set(encoder, var_reduced: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(var_reduced, p=2, dim=-1, keepdim=True)
    var_dirs = var_reduced / (norms + 1e-8)
    x_target = var_dirs * values
    z_list, _ = encoder.encode(x_target)
    recon = encoder.decode(
        z_list, target_n=x_target.size(1), use_mean=True, noise_std=0.0
    )
    return recon


def _greedy_match_recon_to_vars(
    recon: np.ndarray,
    var_dirs: np.ndarray,
    variable_names: List[str],
):
    eps = 1e-8
    recon_norm = np.linalg.norm(recon, axis=1) + eps
    recon_dir = recon / recon_norm[:, None]
    # Unsigned cosine for var matching to be sign-invariant
    sim = np.abs(np.matmul(recon_dir, var_dirs.T))
    N, M = sim.shape
    recon_order = np.argsort(recon_norm)[::-1]
    taken_vars = set()
    assignments: List[Tuple[str, float, float]] = []
    for i in recon_order:
        best_j, best_s = -1, -1.0
        for j in range(M):
            if j in taken_vars:
                continue
            s = float(sim[i, j])
            if s > best_s:
                best_s = s
                best_j = j
        if best_j < 0:
            best_j = int(np.argmax(sim[i]))
            best_s = float(sim[i, best_j])
        taken_vars.add(best_j)
        pred_val = float(np.linalg.norm(recon[i]))
        assignments.append((variable_names[best_j], pred_val, best_s))
    out = [None] * N  # type: ignore
    for rank, i in enumerate(recon_order):
        out[i] = assignments[rank]
    return out  # type: ignore


def _run_named_recon_print(model: torch.nn.Module, args):
    print("Running named reconstruction printing ...")
    encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    if encoder is None:
        print("[WARN] Model does not expose a SetVAE encoder; skip named recon")
        return

    stats_map: Dict[str, Tuple[float, float]] = {}
    if getattr(args, "stats_csv", None) and os.path.exists(args.stats_csv):
        try:
            stats_df = pd.read_csv(args.stats_csv)
            key_col = _detect_key_column(stats_df)
            rename = {}
            for c in stats_df.columns:
                lc = c.lower()
                if lc == "mean":
                    rename[c] = "mean"
                if lc in ("std", "stdev", "stddev"):
                    rename[c] = "std"
            stats_df = stats_df.rename(columns=rename)
            if "mean" in stats_df.columns and "std" in stats_df.columns:
                for _, r in stats_df.iterrows():
                    v = str(r[key_col])
                    m = float(r["mean"]) if pd.notna(r["mean"]) else 0.0
                    s_raw = float(r["std"]) if pd.notna(r["std"]) else 1.0
                    s = s_raw if abs(s_raw) > 1e-12 else 1.0
                    stats_map[v] = (m, s)
        except Exception as e:
            print(f"[WARN] Failed to load stats_csv '{args.stats_csv}': {e}")

    def _denorm_value(var_name: str, val_norm: float) -> float:
        m, s = stats_map.get(var_name, (0.0, 1.0))
        return val_norm * s + m

    vocab_event_names: Optional[List[str]] = None
    vocab_dirs_np: Optional[np.ndarray] = None
    device = next(model.parameters()).device
    # For evaluation-only matching in original input space, prepare a linear lift
    # from reduced_dim back to input_dim using pseudoinverse of the learned reducer.
    lift_A: Optional[torch.Tensor] = None  # shape [reduced_dim, input_dim]
    lift_b: Optional[torch.Tensor] = None  # bias vector (reduced_dim)
    try:
        dim_red = getattr(encoder, "dim_reducer", None)
        if dim_red is not None and hasattr(dim_red, "weight") and dim_red.weight is not None:
            # dim_reducer: y = W x + b, W shape [R, D]. To lift y->x_hat in original D,
            # use x_hat ≈ (pinv(W)) y  (we implement as (y - b) @ (pinv(W))^T).
            W = dim_red.weight  # [R, D]
            lift_A = torch.linalg.pinv(W).t().contiguous().to(device)  # (pinv(W))^T => [R, D]
            lift_b = (dim_red.bias.to(device) if getattr(dim_red, "bias", None) is not None else None)
    except Exception:
        lift_A, lift_b = None, None
    # Decide matching space
    req_space = getattr(args, "match_space", "auto")
    has_reducer = getattr(encoder, "dim_reducer", None) is not None
    if req_space == "original":
        match_space = "original"
    elif req_space == "reduced":
        match_space = "reduced" if has_reducer else "original"
    else:
        match_space = "reduced" if has_reducer else "original"
    if getattr(args, "named_recon", "off") == "global":
        vocab_names, vocab_vecs = _load_global_vocab_embeddings(args.data_dir, getattr(args, "event_emb_csv", ""))
        vocab_t = torch.from_numpy(vocab_vecs).unsqueeze(0).to(device)
        if match_space == "reduced" and has_reducer:
            try:
                vocab_red = encoder.dim_reducer(vocab_t)
            except Exception:
                vocab_red = vocab_t
            vocab_dirs_t = vocab_red / (torch.norm(vocab_red, p=2, dim=-1, keepdim=True) + 1e-8)
        else:
            vocab_dirs_t = vocab_t / (torch.norm(vocab_t, p=2, dim=-1, keepdim=True) + 1e-8)
        vocab_dirs_np = vocab_dirs_t.squeeze(0).detach().cpu().numpy()
        vocab_event_names = [str(n) for n in vocab_names]

    ds = PatientDataset("test" if args.split == "test" else "valid", saved_dir=args.data_dir)
    rng = random.Random(args.seed)

    def _process_one_set(df_set: pd.DataFrame, patient_id, chosen_set, sample_idx: Optional[int], total_samples: Optional[int]):
        vcols_local = _detect_vcols(df_set)
        var_np = df_set[vcols_local].to_numpy(dtype=np.float32, copy=False)
        val_np = df_set["value"].to_numpy(dtype=np.float32)
        var_t = torch.from_numpy(var_np).unsqueeze(0).to(device)
        val_t = torch.from_numpy(val_np).view(1, -1, 1).to(device)
        if getattr(encoder, "dim_reducer", None) is not None:
            var_red = encoder.dim_reducer(var_t)
        else:
            var_red = var_t
        recon_nomask_t = _encode_decode_set(encoder, var_red, val_t)
        # Prepare directions and recon in the chosen matching space
        with torch.no_grad():
            if match_space == "reduced" and has_reducer:
                var_dirs_t = var_red / (torch.norm(var_red, p=2, dim=-1, keepdim=True) + 1e-8)
                recon_space = recon_nomask_t
            else:
                var_dirs_t = var_t / (torch.norm(var_t, p=2, dim=-1, keepdim=True) + 1e-8)
                if lift_A is not None and recon_nomask_t.size(-1) == lift_A.size(0):
                    y = recon_nomask_t
                    if lift_b is not None and y.size(-1) == lift_b.numel():
                        y = y - lift_b.view(1, 1, -1)
                    recon_space = torch.matmul(y, lift_A)  # [1,N,D]
                else:
                    recon_space = recon_nomask_t
        recon_nomask_np = recon_space.squeeze(0).detach().cpu().numpy()

        if getattr(args, "named_recon", "off") == "global":
            assert vocab_dirs_np is not None and vocab_event_names is not None
            var_dirs_np_local = vocab_dirs_np
            event_names_local: List[str] = vocab_event_names
        else:
            var_dirs_np_local = (var_dirs_t.squeeze(0).detach().cpu().numpy())
            event_names_local = df_set["variable"].astype(str).tolist()

        assignments_nomask = _greedy_match_recon_to_vars(
            recon_nomask_np, var_dirs_np_local, event_names_local
        )

        set_event_names: List[str] = df_set["variable"].astype(str).tolist()
        def _split(assignments_list: List[Tuple[str, float, float]]):
            matched: List[Tuple[str, float, float]] = []
            unmatched: List[Tuple[str, float, float]] = []
            names_set = set(set_event_names)
            for (nm, valn, cs) in assignments_list:
                if nm in names_set:
                    matched.append((nm, valn, cs))
                else:
                    unmatched.append((nm, valn, cs))
            return matched, unmatched
        matched_nomask, unmatched_nomask = _split(assignments_nomask)

        print("================ Named Reconstruction ================")
        if sample_idx is not None and total_samples is not None:
            print(f"Sample:     {sample_idx+1}/{total_samples}")
        print(f"Patient ID: {patient_id}")
        print(f"Set index:  {chosen_set}")
        vocab_len = len(event_names_local)
        print(
            f"#Events:    {len(set_event_names)} (match scope: {getattr(args, 'named_recon', 'off')}, "
            f"vocab={vocab_len}, space={match_space})"
        )
        print("---------------------------------------------------------------")
        print("Original events (name -> de-normalized original value):")
        carried_flags: List[int]
        if "is_carry" in df_set.columns:
            try:
                carried_flags = df_set["is_carry"].astype(float).fillna(0.0).astype(int).tolist()  # type: ignore
            except Exception:
                carried_flags = [0] * len(df_set)
        else:
            carried_flags = [0] * len(df_set)
        for idx, (name, val_norm) in enumerate(zip(set_event_names, val_np.tolist())):
            val_orig = _denorm_value(name, float(val_norm))
            cflag = carried_flags[idx] if 0 <= idx < len(carried_flags) else 0
            carry_str = "carried" if cflag else "not carried"
            print(f"  - {name}: {val_orig:.6f} ({carry_str})")
        print("---------------------------------------------------------------")
        print("Reconstruction matched results (No-mask):")
        # Map original variable -> carried (any occurrence carried => carried)
        carry_map: Dict[str, bool] = {}
        if "is_carry" in df_set.columns:
            try:
                for nm, cf in zip(df_set["variable"].astype(str).tolist(), carried_flags):
                    carry_map[nm] = carry_map.get(nm, False) or bool(cf)
            except Exception:
                carry_map = {}
        col_header = ["variable", "Pred value (denorm)", "cosine", "carried"]
        print(" | ".join(col_header))
        for (n, v, c) in matched_nomask:
            cflag = carry_map.get(n, False)
            print(f"{n} | {_denorm_value(n, float(v)):.6f} | {c:.3f} | {int(cflag)}")
        print("---------------------------------------------------------------")
        print("Unmatched predictions (No-mask):")
        for (n, v, c) in unmatched_nomask:
            print(f"{n}: {_denorm_value(n, float(v)):.6f}  (cos={c:.3f})")
        print("===============================================================")

        # Unmatched/recon analysis (revised):
        # 1) For original variables that were not matched, print per-variable
        #    min/max absolute cosine to any recon, labeling the corresponding recon var.
        # 2) For recon-only predictions (names not in original set), compute global
        #    min/max absolute cosine to any original variable and print the values
        #    along with the recon variable names.
        try:
            set_var_dirs_np = var_dirs_t.squeeze(0).detach().cpu().numpy()
            # Build the set of original variable names that were matched
            matched_in_set = set(n for (n, _v, _c) in matched_nomask)
            # Unit-direction normalization
            eps = 1e-8
            recon_norm = np.linalg.norm(recon_nomask_np, axis=1, keepdims=True) + eps
            recon_dir = recon_nomask_np / recon_norm
            # Mapping from recon index to its matched/assigned variable name (label)
            recon_var_labels = [nm for (nm, _pv, _cs) in assignments_nomask]

            # (1) Per-original unmatched variables
            for j, orig_name in enumerate(set_event_names):
                if orig_name in matched_in_set:
                    continue
                v = set_var_dirs_np[j]
                v_norm = v / (np.linalg.norm(v) + eps)
                cosines = np.abs(recon_dir @ v_norm)
                if cosines.size == 0:
                    cflag = carried_flags[j] if 0 <= j < len(carried_flags) else 0
                    print(f"{orig_name} [carried={cflag}]: min_cos_to_recon = nan | max_cos_to_recon = nan")
                    continue
                min_idx = int(np.argmin(cosines))
                max_idx = int(np.argmax(cosines))
                min_cos = float(cosines[min_idx])
                max_cos = float(cosines[max_idx])
                min_label = recon_var_labels[min_idx] if 0 <= min_idx < len(recon_var_labels) else "?"
                max_label = recon_var_labels[max_idx] if 0 <= max_idx < len(recon_var_labels) else "?"
                cflag = carried_flags[j] if 0 <= j < len(carried_flags) else 0
                print(
                    f"{orig_name} [carried={cflag}]: min_cos_to_recon = {min_cos:.3f} (recon_var={min_label}) | "
                    f"max_cos_to_recon = {max_cos:.3f} (recon_var={max_label})"
                )
        except Exception as e:
            print(f"[WARN] unmatched-orig analysis failed: {e}")

        # (2) Recon-only global extremes to any original variable
        try:
            # Prepare normalized original directions
            eps = 1e-8
            set_var_dirs_np = var_dirs_t.squeeze(0).detach().cpu().numpy()
            set_var_norms = np.linalg.norm(set_var_dirs_np, axis=1, keepdims=True) + eps
            set_var_dirs_np = set_var_dirs_np / set_var_norms
            # Normalized recon directions
            recon_norm = np.linalg.norm(recon_nomask_np, axis=1, keepdims=True) + eps
            recon_dir = recon_nomask_np / recon_norm
            names_set = set(set_event_names)
            recon_only_idxs = [i for i, nm in enumerate(recon_var_labels) if nm not in names_set]
            if len(recon_only_idxs) > 0 and set_var_dirs_np.size > 0:
                sub = recon_dir[recon_only_idxs]
                cos_mat = np.abs(sub @ set_var_dirs_np.T)  # [R_only, N_orig]
                # Global min and max over all pairs
                flat_min = int(np.argmin(cos_mat))
                flat_max = int(np.argmax(cos_mat))
                r_only_count, _N = cos_mat.shape
                r_min = flat_min // _N
                r_max = flat_max // _N
                min_val = float(cos_mat.ravel()[flat_min])
                max_val = float(cos_mat.ravel()[flat_max])
                min_recon_name = recon_var_labels[recon_only_idxs[r_min]]
                max_recon_name = recon_var_labels[recon_only_idxs[r_max]]
                print(f"Recon-only: min_cos_to_any_original = {min_val:.3f} (recon_var={min_recon_name})")
                print(f"Recon-only: max_cos_to_any_original = {max_val:.3f} (recon_var={max_recon_name})")
            else:
                print("Recon-only: no recon-only predictions or empty originals; skip cosine extremes")
        except Exception as e:
            print(f"[WARN] recon-only extremes analysis failed: {e}")

    if getattr(args, "patient_index", None) is not None and getattr(args, "set_index", None) is not None:
        if args.patient_index < 0 or args.patient_index >= len(ds):
            print(f"[WARN] patient_index out of range [0,{len(ds)-1}]")
            return
        df, patient_id = ds[int(args.patient_index)]
        if int(args.set_index) not in set(int(s) for s in df["set_index"].tolist()):
            print(f"[WARN] set_index {args.set_index} not found in patient {patient_id}")
            return
        chosen_set = int(args.set_index)
        df_set = df[df["set_index"] == chosen_set].reset_index(drop=True)
        _process_one_set(df_set, patient_id, chosen_set, sample_idx=0, total_samples=1)
    elif getattr(args, "patient_index", None) is not None and getattr(args, "set_index", None) is None:
        if args.patient_index < 0 or args.patient_index >= len(ds):
            print(f"[WARN] patient_index out of range [0,{len(ds)-1}]")
            return
        df, patient_id = ds[int(args.patient_index)]
        uniq_sets = sorted(set(int(s) for s in df["set_index"].tolist()))
        if not uniq_sets:
            print("[WARN] Selected patient has no sets")
            return
        k = min(len(uniq_sets), max(1, int(getattr(args, "num_named_samples", 10))))
        chosen_sets = rng.sample(uniq_sets, k=k)
        for idx, chosen_set in enumerate(chosen_sets):
            df_set = df[df["set_index"] == chosen_set].reset_index(drop=True)
            _process_one_set(df_set, patient_id, chosen_set, sample_idx=idx, total_samples=len(chosen_sets))
    else:
        total = max(1, int(getattr(args, "num_named_samples", 10)))
        for idx in range(total):
            pidx = rng.randrange(len(ds))
            df, patient_id = ds[pidx]
            uniq_sets = sorted(set(int(s) for s in df["set_index"].tolist()))
            if not uniq_sets:
                continue
            chosen_set = rng.choice(uniq_sets)
            df_set = df[df["set_index"] == chosen_set].reset_index(drop=True)
            _process_one_set(df_set, patient_id, chosen_set, sample_idx=idx, total_samples=total)


def _run_var_scale_audit(model: torch.nn.Module, args):
    print("Running per-variable scale audit ...")
    encoder = model.set_encoder if hasattr(model, "set_encoder") else getattr(model, "setvae", None)
    if encoder is None:
        print("[WARN] Model does not expose a SetVAE encoder; skip audit")
        return
    device = next(model.parameters()).device

    part = getattr(args, "audit_partition", "train")
    small_thr = float(getattr(args, "audit_small_thr", 0.2))
    max_patients = int(getattr(args, "audit_max_patients", 200))
    min_count = int(getattr(args, "audit_min_count", 10))
    seed = int(getattr(args, "seed", 42))

    ds = PatientDataset(part, saved_dir=args.data_dir)
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if max_patients is not None and max_patients > 0:
        idxs = idxs[:max_patients]

    var_to_vals: Dict[str, List[float]] = {}
    var_to_ratios: Dict[str, List[float]] = {}
    var_to_cos: Dict[str, List[float]] = {}
    keep_first_n = int(getattr(args, "audit_keep_first_n_vars", 0))
    seen_vars: set[str] = set()

    for i in idxs:
        df, _pid = ds[i]
        try:
            vcols = _detect_vcols(df)
        except Exception:
            continue
        uniq_sets = sorted(set(int(s) for s in df["set_index"].tolist()))
        for sid in uniq_sets:
            df_set = df[df["set_index"] == sid]
            if len(df_set) == 0:
                continue
            names = df_set["variable"].astype(str).tolist()
            val_np = df_set["value"].to_numpy(dtype=np.float32)
            if val_np.size == 0:
                continue
            var_np = df_set[vcols].to_numpy(dtype=np.float32, copy=False)

            var_t = torch.from_numpy(var_np).unsqueeze(0).to(device)
            val_t = torch.from_numpy(val_np).reshape(1, -1, 1).to(device)

            if getattr(encoder, "dim_reducer", None) is not None:
                var_red = encoder.dim_reducer(var_t)
            else:
                var_red = var_t
            with torch.no_grad():
                recon_t = _encode_decode_set(encoder, var_red, val_t)
                var_dirs_t = var_red / (torch.norm(var_red, p=2, dim=-1, keepdim=True) + 1e-8)
            recon_np = recon_t.squeeze(0).detach().cpu().numpy()
            var_dirs_np = var_dirs_t.squeeze(0).detach().cpu().numpy()

            assigns = _greedy_match_recon_to_vars(recon_np, var_dirs_np, names)
            pred_map: Dict[str, Tuple[float, float]] = {n: (pv, cs) for (n, pv, cs) in assigns}

            for name, v in zip(names, val_np.tolist()):
                if keep_first_n > 0 and name not in seen_vars and len(seen_vars) >= keep_first_n:
                    # skip new variables if we already have N distinct vars collected
                    continue
                abs_v = float(abs(v))
                pv, cs = pred_map.get(name, (float("nan"), float("nan")))
                if not math.isfinite(abs_v) or abs_v < 1e-8 or not math.isfinite(pv):
                    continue
                ratio = float(pv) / (abs_v + 1e-8)
                var_to_vals.setdefault(name, []).append(abs_v)
                var_to_ratios.setdefault(name, []).append(ratio)
                if math.isfinite(cs):
                    var_to_cos.setdefault(name, []).append(float(cs))
                seen_vars.add(name)

    rows: List[Dict[str, float]] = []
    for name in var_to_vals.keys():
        vals = np.array(var_to_vals.get(name, []), dtype=np.float32)
        rats = np.array(var_to_ratios.get(name, []), dtype=np.float32)
        coss = np.array(var_to_cos.get(name, []), dtype=np.float32) if name in var_to_cos else np.array([], dtype=np.float32)
        if len(vals) < min_count:
            continue
        row = {
            "variable": name,
            "count": int(len(vals)),
            "median_abs_val": float(np.median(vals)),
            f"prop_abs_val_le_{small_thr}": float(np.mean(vals <= small_thr)),
            "mean_scale_ratio": float(np.mean(rats)),
            "median_scale_ratio": float(np.median(rats)),
            "cos_median": float(np.median(coss)) if coss.size > 0 else float("nan"),
        }
        rows.append(row)
    if not rows:
        print("[WARN] No variables met min_count for audit; skipping save")
        return
    df_var = pd.DataFrame(rows).sort_values("count", ascending=False)
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "var_scale_audit.csv")
    df_var.to_csv(out_csv, index=False)
    print(f"Saved var-scale audit CSV to: {out_csv}")

    # Scatter plot
    if plt is not None:
        try:
            x = df_var[f"prop_abs_val_le_{small_thr}"].to_numpy(dtype=float)
            y = df_var["mean_scale_ratio"].to_numpy(dtype=float)
            sizes = np.sqrt(df_var["count"].to_numpy(dtype=float)).clip(min=1.0) * 2.0
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x, y, s=sizes, alpha=0.65, c="tab:blue")
            ax.set_xlabel(f"Prop(|val| ≤ {small_thr})")
            ax.set_ylabel("Mean scale ratio (recon/|val|)")
            ax.set_title(f"Var-scale audit ({part} set)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_png = os.path.join(args.output_dir, "var_scale_audit_scatter.png")
            fig.savefig(out_png, dpi=220)
            plt.close(fig)
            # Simple Pearson correlation
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                try:
                    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
                    print(f"Pearson corr(prop_small, mean_scale_ratio) = {corr:.4f}")
                except Exception:
                    pass
            print(f"Saved var-scale scatter to: {out_png}")
        except Exception as e:
            print(f"[WARN] Failed to plot var-scale audit: {e}")
    else:
        print("[INFO] matplotlib not available; skipping scatter plot")

def _run_pretrain_eval():
    ap = argparse.ArgumentParser(
        add_help=False,
        description="Evaluate setvae/poe pretraining: collapse + reconstruction (formerly Stage A/B)",
    )
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to pretraining checkpoint (.ckpt)")
    ap.add_argument("--ckpt_type", type=str, choices=["auto", "setvae", "poe"], default="auto")
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--batch_size", type=int, default=getattr(cfg, "batch_size", 8))
    ap.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 0))
    ap.add_argument("--num_eval_batches", type=int, default=50, help="Number of batches to estimate KL stats")
    ap.add_argument("--num_vis_samples", type=int, default=2, help="#samples for detailed recon/heatmaps")
    ap.add_argument(
        "--recon_from",
        type=str,
        choices=["setvae", "poe", "auto"],
        default="auto",
        help=(
            "Source for reconstruction: 'setvae' uses the frozen SetVAE encoder/decoder;\n"
            "'poe' decodes from PoE dynamic posterior via model.decoder(h_t);\n"
            "'auto' chooses 'poe' if model has PoE components, else 'setvae'."
        ),
    )
    ap.add_argument("--output_dir", type=str, default=None, help="If not set, saves to <ckpt_version_dir>/eval")
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--active_threshold", type=float, default=0.01, help="KL threshold for active units")
    ap.add_argument("--active_thresholds", type=float, nargs="*", default=[0.001, 0.005, 0.01, 0.02, 0.05], help="KL thresholds for active ratio diagnostics")
    ap.add_argument("--topk_plot", type=int, default=20, help="Top-K bars to show for KL and |mu| means")
    ap.add_argument("--collapse_kl_thresh", type=float, default=0.001, help="Dim considered collapsed if KL below this")
    ap.add_argument("--collapse_mu_abs_thresh", type=float, default=0.02, help="Dim considered collapsed if mean |mu| below this")
    ap.add_argument("--collapse_var_tol", type=float, default=0.05, help="Dim considered collapsed if |mean var - 1| below this tolerance")
    ap.add_argument("--seed", type=int, default=42)
    # Named recon options
    ap.add_argument("--named_recon", type=str, choices=["off", "set", "global"], default="off")
    ap.add_argument("--num_named_samples", type=int, default=10)
    ap.add_argument("--patient_index", type=int, default=None)
    ap.add_argument("--set_index", type=int, default=None)
    ap.add_argument("--stats_csv", type=str, default=getattr(cfg, "params_map_path", ""))
    ap.add_argument("--event_emb_csv", type=str, default="")
    # Matching space for named reconstruction
    ap.add_argument(
        "--match_space",
        type=str,
        choices=["auto", "reduced", "original"],
        default="auto",
        help=(
            "Space for named recon matching: 'reduced' uses reduced_dim if available; "
            "'original' uses input_dim; 'auto' prefers 'reduced' when reducer exists."
        ),
    )
    # Per-variable scale audit (optional)
    ap.add_argument("--audit_var_scale", action="store_true", default=False, help="Run per-variable scale audit and scatter plot")
    ap.add_argument("--audit_partition", type=str, choices=["train", "valid", "test"], default="train", help="Dataset partition to audit")
    ap.add_argument("--audit_small_thr", type=float, default=0.2, help="Threshold for small |val| proportion")
    ap.add_argument("--audit_max_patients", type=int, default=200, help="Max patients to sample for audit (use <=0 for all)")
    ap.add_argument("--audit_min_count", type=int, default=10, help="Min samples per variable to include in audit CSV")
    ap.add_argument("--audit_keep_first_n_vars", type=int, default=0, help="If >0, audit only first N variables encountered; default 0 means all")
    # Mask consistency & OOD options
    # Mask consistency (enabled by default; add --no_eval_mask_consistency to disable)
    ap.add_argument("--eval_mask_consistency", dest="eval_mask_consistency", action="store_true", help="Evaluate latent consistency across different measurement masks")
    ap.add_argument("--no_eval_mask_consistency", dest="eval_mask_consistency", action="store_false", help="Disable mask consistency evaluation")
    ap.set_defaults(eval_mask_consistency=True)
    ap.add_argument("--mask_ratios", type=float, nargs="*", default=[0.3, 0.5, 0.7], help="Mask ratios to test (fraction of tokens per subset)")
    ap.add_argument("--consistency_num_sets", type=int, default=200, help="#sets to sample for consistency eval")
    ap.add_argument("--consistency_repeats", type=int, default=3, help="Repeats per ratio per set")
    ap.add_argument("--consistency_disjoint", action="store_true", default=True, help="Use disjoint subset masks (else independent random masks)")
    # OOD scramble (enabled by default; add --no_eval_ood_scramble to disable)
    ap.add_argument("--eval_ood_scramble", dest="eval_ood_scramble", action="store_true", help="Evaluate outlier detection by scrambling values within sets")
    ap.add_argument("--no_eval_ood_scramble", dest="eval_ood_scramble", action="store_false", help="Disable OOD scramble evaluation")
    ap.set_defaults(eval_ood_scramble=True)
    ap.add_argument("--ood_num_sets", type=int, default=200, help="#sets to sample for OOD scramble eval")
    ap.add_argument(
        "--ood_scheme",
        type=str,
        choices=["sample_from_stats"],
        default="sample_from_stats",
        help="Scramble scheme",
    )
    # Per-set masking curves (consistency vs masked count)
    ap.add_argument(
        "--plot_per_set_mask_curves",
        action="store_true",
        default=True,
        help="Plot consistency curves for randomly sampled sets under progressive masking",
    )
    ap.add_argument(
        "--num_sets_for_curves",
        type=int,
        default=10,
        help="How many sets to plot curves for (default 10)",
    )
    ap.add_argument(
        "--min_set_len_for_curves",
        type=int,
        default=2,
        help="Minimum set length to include in per-set curves",
    )
    # Optional CLI overrides for dims (useful when config defaults mismatch checkpoint)
    ap.add_argument("--input_dim", type=int, default=None)
    ap.add_argument("--reduced_dim", type=int, default=None)
    ap.add_argument("--latent_dim", type=int, default=None)
    # Simplified evaluation mode
    ap.add_argument(
        "--simple",
        action="store_true",
        help="Simplified evaluation: minimal metrics, no heavy plots or extras",
    )
    args, _ = ap.parse_known_args()

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

    # If simplified mode, turn off heavy/expensive steps and shrink sample counts
    if getattr(args, "simple", False):
        # Reduce compute and disable optional evaluations/plots
        args.num_eval_batches = min(int(args.num_eval_batches), 30)
        args.num_vis_samples = 0
        args.eval_mask_consistency = False
        args.eval_ood_scramble = False
        args.plot_per_set_mask_curves = False
        args.audit_var_scale = False
        args.named_recon = "off"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state = _load_state_dict_pretrain(args.checkpoint)
    ckpt_type = _detect_ckpt_type(state, prefer=args.ckpt_type)
    # Infer dims from state, then allow CLI overrides to take precedence
    inferred = _infer_dims_from_state(state)
    dims = dict(inferred)
    if args.input_dim is not None:
        dims["input_dim"] = int(args.input_dim)
    if args.reduced_dim is not None:
        dims["reduced_dim"] = int(args.reduced_dim)
    if args.latent_dim is not None:
        dims["latent_dim"] = int(args.latent_dim)
    model = _build_model_pretrain(
        ckpt_type,
        lr=getattr(cfg, "lr", 3e-4),
        dims=dims,
        state=state,
    )
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

    print("Computing KL per-dimension stats across dataset ...")
    kl_dim_t = compute_kl_dataset(model, dl, num_batches=args.num_eval_batches)
    kl_dim = kl_dim_t.detach().cpu().numpy().astype(np.float32)
    total_kl = float(kl_dim.sum())
    active_ratio = float(np.mean(kl_dim > args.active_threshold)) if kl_dim.size > 0 else 0.0
    active_ratios_multi = compute_active_ratios(kl_dim, args.active_thresholds)
    sorted_vals = np.sort(kl_dim)[::-1]
    mass = np.cumsum(sorted_vals)
    denom = mass[-1] if mass.size > 0 else 1.0
    frac = mass / (denom + 1e-8)
    dim90 = int(np.sum(frac < 0.90) + 1) if frac.size > 0 else 0
    dim95 = int(np.sum(frac < 0.95) + 1) if frac.size > 0 else 0
    eff_dims = compute_effective_dimensions(kl_dim)

    # In simplified mode, skip plots, posterior moments, recon visualizations, and extras
    if getattr(args, "simple", False):
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        simple_report = {
            "mode": "simple",
            "checkpoint": args.checkpoint,
            "ckpt_type": ckpt_type,
            "split": args.split,
            "active_threshold": args.active_threshold,
            "latent_dim": int(getattr(model, "latent_dim", len(kl_dim))),
            "kl": {
                "total_kl": total_kl,
                "active_ratio": active_ratio,
                "dim90": dim90,
                "dim95": dim95,
                "effective_dimensions": eff_dims,
            },
            "collapse_flags": {
                "kl_active_ratio_below_0.1": active_ratio < 0.1,
                "total_kl_below_1e-2": total_kl < 1e-2,
                "risk": (active_ratio < 0.1) or (total_kl < 1e-2),
            },
        }
        out_json = os.path.join(args.output_dir, f"pretrain_eval_simple_{ts}.json")
        with open(out_json, "w") as f:
            json.dump(simple_report, f, indent=2)
        print(f"Saved simplified evaluation report to: {out_json}")
        return

    _plot_kl_hist_and_cdf(kl_dim, args.output_dir)
    _plot_bar_topk(kl_dim, args.topk_plot, title="Top-K KL per-dimension", out_file=os.path.join(args.output_dir, "kl_topk.png"), ylabel="KL (nats)")

    print("Computing posterior moments across dataset ...")
    moments = compute_posterior_moments_dataset(model, dl, num_batches=args.num_eval_batches)
    mu_abs_mean = moments["mu_abs_mean"]
    mu_mean = moments["mu_mean"]
    mu_std = moments["mu_std"]
    var_mean = moments["var_mean"]
    var_std = moments["var_std"]
    _plot_hist_generic(mu_abs_mean, "Mean |μ| per dimension", os.path.join(args.output_dir, "mu_abs_mean_hist.png"), "mean |μ|")
    _plot_hist_generic(var_mean, "Mean σ² per dimension", os.path.join(args.output_dir, "var_mean_hist.png"), "mean σ²")
    _plot_scatter_generic(mu_abs_mean, np.abs(var_mean - 1.0), "|μ| vs |σ²-1| per-dimension", os.path.join(args.output_dir, "muabs_vs_varminus1_scatter.png"), "mean |μ|", "|mean σ² - 1|")
    _plot_bar_topk(mu_abs_mean, args.topk_plot, title="Top-K mean |μ| per-dimension", out_file=os.path.join(args.output_dir, "mu_abs_topk.png"), ylabel="mean |μ|")

    collapsed_mask = (
        (kl_dim < args.collapse_kl_thresh)
        & (mu_abs_mean < args.collapse_mu_abs_thresh)
        & (np.abs(var_mean - 1.0) <= args.collapse_var_tol)
    )
    collapsed_indices = np.nonzero(collapsed_mask)[0].tolist()
    collapsed_count = int(len(collapsed_indices))

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
        mu_mat, var_mat = collect_mu_var_heatmaps(model, batch)
        _plot_heatmap(
            mu_mat,
            title=f"Sample {i} - μ across sets",
            out_file=os.path.join(args.output_dir, f"heatmap_mu_sample_{i}.png"),
            value_type="mean",
        )
        _plot_heatmap(
            var_mat,
            title=f"Sample {i} - σ² across sets",
            out_file=os.path.join(args.output_dir, f"heatmap_var_sample_{i}.png"),
            value_type="var",
        )
        orig, recon = collect_recon_events(model, batch, recon_from=getattr(args, "recon_from", "auto"))
        _plot_overlay_2d(
            orig,
            recon,
            out_file=os.path.join(
                args.output_dir, f"overlay_orig_recon_sample_{i}.png"
            ),
        )
        metrics = compute_recon_metrics(orig, recon)
        kd = compute_kl_dim_stats_for_batch(model, batch)
        kd_np = kd.detach().cpu().numpy().astype(np.float32)
        metrics["kl_total"] = float(kd_np.sum())
        metrics["active_ratio"] = float(np.mean(kd_np > args.active_threshold))
        metrics["sample_index"] = i
        recon_results.append(metrics)
        vis_count += 1

    def _agg(key: str) -> float:
        vals = [
            r[key]
            for r in recon_results
            if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])
        ]
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
        "kl_total_mean": _agg("kl_total"),
        "active_ratio_mean": _agg("active_ratio"),
    }

    def _corr(a_key: str, b_key: str) -> float:
        a = [
            m.get(a_key)
            for m in recon_results
            if isinstance(m.get(a_key), (int, float)) and math.isfinite(m.get(a_key))
        ]
        b = [
            m.get(b_key)
            for m in recon_results
            if isinstance(m.get(b_key), (int, float)) and math.isfinite(m.get(b_key))
        ]
        n = min(len(a), len(b))
        if n < 2:
            return float("nan")
        a = np.array(a[:n], dtype=np.float32)
        b = np.array(b[:n], dtype=np.float32)
        try:
            c = float(np.corrcoef(a, b)[0, 1])
        except Exception:
            c = float("nan")
        return c

    sample_correlations = {
        "kl_total_vs_nn_l2_mean": _corr("kl_total", "nn_l2_mean"),
        "kl_total_vs_chamfer_l2_mean": _corr("kl_total", "chamfer_l2_mean"),
        "kl_total_vs_dir_cosine_mean": _corr("kl_total", "dir_cosine_mean"),
        "kl_total_vs_mag_mae": _corr("kl_total", "mag_mae"),
        "kl_total_vs_mag_rmse": _corr("kl_total", "mag_rmse"),
    }

    # Optional: mask consistency and OOD-scramble evaluations
    extra_evals: Dict[str, object] = {}
    if getattr(args, "eval_mask_consistency", False):
        try:
            extra_evals["mask_consistency"] = evaluate_mask_consistency(
                model,
                dl_vis,
                mask_ratios=[float(r) for r in getattr(args, "mask_ratios", [0.3, 0.5, 0.7])],
                num_sets=int(getattr(args, "consistency_num_sets", 200)),
                repeats=int(getattr(args, "consistency_repeats", 3)),
                disjoint=bool(getattr(args, "consistency_disjoint", True)),
            )
            # Visualize
            try:
                visualize_mask_consistency(
                    model,
                    dl_vis,
                    out_dir=args.output_dir,
                    mask_ratios=[float(r) for r in getattr(args, "mask_ratios", [0.3, 0.5, 0.7])],
                    num_sets=int(getattr(args, "consistency_num_sets", 200)),
                    repeats=int(getattr(args, "consistency_repeats", 3)),
                    disjoint=bool(getattr(args, "consistency_disjoint", True)),
                )
            except Exception:
                pass
        except Exception as e:
            extra_evals["mask_consistency_error"] = str(e)
    if getattr(args, "eval_ood_scramble", False):
        try:
            # Optional resources for stats-based scramble
            vocab_event_names: Optional[List[str]] = None
            vocab_dirs_np: Optional[np.ndarray] = None
            stats_map: Dict[str, Tuple[float, float]] = {}
            try:
                if getattr(args, "event_emb_csv", "") and os.path.exists(args.event_emb_csv):
                    names, vecs = _load_global_vocab_embeddings(args.data_dir, args.event_emb_csv)
                    device = next(model.parameters()).device
                    vecs_t = torch.from_numpy(vecs).unsqueeze(0).to(device)
                    # Reduce if encoder has reducer
                    enc = _get_encoder_from_model(model)
                    if enc is not None and getattr(enc, "dim_reducer", None) is not None:
                        try:
                            vecs_red = enc.dim_reducer(vecs_t)
                        except Exception:
                            vecs_red = vecs_t
                        dirs_t = vecs_red / (torch.norm(vecs_red, p=2, dim=-1, keepdim=True) + 1e-8)
                    else:
                        dirs_t = vecs_t / (torch.norm(vecs_t, p=2, dim=-1, keepdim=True) + 1e-8)
                    vocab_dirs_np = dirs_t.squeeze(0).detach().cpu().numpy()
                    vocab_event_names = [str(n) for n in names]
                if getattr(args, "stats_csv", None) and os.path.exists(args.stats_csv):
                    df_stats = pd.read_csv(args.stats_csv)
                    key_col = _detect_key_column(df_stats)
                    # normalize mean/std naming
                    rename = {}
                    for c in df_stats.columns:
                        lc = c.lower()
                        if lc == "mean":
                            rename[c] = "mean"
                        if lc in ("std", "stdev", "stddev"):
                            rename[c] = "std"
                    df_stats = df_stats.rename(columns=rename)
                    if "mean" in df_stats.columns and "std" in df_stats.columns:
                        for _, r in df_stats.iterrows():
                            vname = str(r[key_col])
                            m = float(r["mean"]) if pd.notna(r["mean"]) else 0.0
                            s_raw = float(r["std"]) if pd.notna(r["std"]) else 1.0
                            s_eff = s_raw if abs(s_raw) > 1e-12 else 1.0
                            stats_map[vname] = (m, s_eff)
            except Exception:
                pass

            extra_evals["ood_scramble"] = evaluate_ood_scramble(
                model,
                dl_vis,
                num_sets=int(getattr(args, "ood_num_sets", 200)),
                scheme=str(getattr(args, "ood_scheme", "sample_from_stats")),
                stats_map=stats_map if len(stats_map) > 0 else None,
                vocab_event_names=vocab_event_names,
                vocab_dirs_np=vocab_dirs_np,
            )
            # Visualize
            try:
                visualize_ood_scramble(
                    model,
                    dl_vis,
                    out_dir=args.output_dir,
                    num_sets=int(getattr(args, "ood_num_sets", 200)),
                    scheme=str(getattr(args, "ood_scheme", "sample_from_stats")),
                )
            except Exception:
                pass
        except Exception as e:
            extra_evals["ood_scramble_error"] = str(e)

    # Optional per-set masking curves (10 random sets by default)
    if getattr(args, "plot_per_set_mask_curves", False):
        try:
            _visualize_per_set_masking_curves(
                model,
                dl_vis,
                out_dir=args.output_dir,
                num_sets=int(getattr(args, "num_sets_for_curves", 10)),
                min_len=int(getattr(args, "min_set_len_for_curves", 2)),
            )
        except Exception as e:
            print(f"[WARN] per-set masking curves plotting failed: {e}")

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "checkpoint": args.checkpoint,
        "ckpt_type": ckpt_type,
        "split": args.split,
        "active_threshold": args.active_threshold,
        "active_thresholds": args.active_thresholds,
        "latent_dim": int(getattr(model, "latent_dim", len(kl_dim))),
        "kl": {
            "mean_per_dim": kl_dim.tolist(),
            "total_kl": total_kl,
            "active_ratio": active_ratio,
            "active_ratios": active_ratios_multi,
            "dim90": dim90,
            "dim95": dim95,
            "effective_dimensions": eff_dims,
        },
        "posterior_moments": {
            "mu_abs_mean": mu_abs_mean.tolist(),
            "mu_mean": mu_mean.tolist(),
            "mu_std": mu_std.tolist(),
            "var_mean": var_mean.tolist(),
            "var_std": var_std.tolist(),
            "mu_abs_mean_stats": {
                "mean": float(np.mean(mu_abs_mean)) if mu_abs_mean.size > 0 else 0.0,
                "median": float(np.median(mu_abs_mean)) if mu_abs_mean.size > 0 else 0.0,
                "p95": float(np.percentile(mu_abs_mean, 95)) if mu_abs_mean.size > 0 else 0.0,
            },
            "var_mean_stats": {
                "mean": float(np.mean(var_mean)) if var_mean.size > 0 else 0.0,
                "median": float(np.median(var_mean)) if var_mean.size > 0 else 0.0,
                "p95_abs_var_minus_1": float(np.percentile(np.abs(var_mean - 1.0), 95)) if var_mean.size > 0 else 0.0,
            },
        },
        "reconstruction": {
            "summary": recon_summary,
            "details": recon_results,
        },
        "collapsed_dims": {
            "criteria": {
                "kl_thresh": args.collapse_kl_thresh,
                "mu_abs_thresh": args.collapse_mu_abs_thresh,
                "var_tol": args.collapse_var_tol,
            },
            "count": collapsed_count,
            "indices": collapsed_indices,
        },
        "sample_correlations": sample_correlations,
        "collapse_flags": {
            "kl_active_ratio_below_0.1": active_ratio < 0.1,
            "total_kl_below_1e-2": total_kl < 1e-2,
            "risk": (active_ratio < 0.1) or (total_kl < 1e-2),
        },
        "consistency_ood": extra_evals,
    }
    out_json = os.path.join(args.output_dir, f"pretrain_eval_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {out_json}")

    if getattr(args, "named_recon", "off") != "off":
        _run_named_recon_print(model, args)

    # Optional: per-variable scale audit
    if getattr(args, "audit_var_scale", False):
        try:
            _run_var_scale_audit(model, args)
        except Exception as e:
            print(f"[WARN] var-scale audit failed: {e}")


def _inject_ckpt_type_from_stage():
    # If user provided --stage, map to ckpt_type and inject into argv if not explicitly set
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", type=str, choices=["A", "B", "C"], required=False)
    args, _ = parser.parse_known_args()
    if not args.stage:
        return
    # Stage C handled separately below. For A/B only: inject ckpt_type
    for tok in sys.argv:
        if tok.startswith("--ckpt_type"):
            return
    if args.stage in {"A", "B"}:
        map_type = {"A": "setvae", "B": "poe"}[args.stage]
        sys.argv.append("--ckpt_type")
        sys.argv.append(map_type)
    # Remove --stage to avoid unknown-arg in _eval_PT
    new_argv = [sys.argv[0]]
    skip = 0
    for i, tok in enumerate(sys.argv[1:], start=1):
        if skip:
            skip -= 1
            continue
        if tok == "--stage":
            skip = 1
            continue
        if tok.startswith("--stage="):
            continue
        new_argv.append(tok)
    sys.argv = new_argv


def _run_discriminative_eval():
    import argparse as _ap
    ap = _ap.ArgumentParser(description="Stage C evaluation (classifier on PoE features)")
    ap.add_argument("--classifier_ckpt", required=True, type=str, help="Path to trained MortalityClassifier checkpoint (.ckpt)")
    ap.add_argument("--checkpoint", required=False, default=None, type=str, help="Optional: PoE checkpoint (.ckpt) to initialize backbone if classifier ckpt lacks PoE weights")
    ap.add_argument("--label_csv", required=True, type=str)
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 1))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--output_dir", type=str, default="./output")
    args, _ = ap.parse_known_args()

    # Build PoE backbone from provided ckpt (if any); weights from classifier ckpt will override if present
    poe_state = _load_state_dict(args.checkpoint) if args.checkpoint else {}
    poe = _build_poe_from_state(poe_state)
    dm = MortalityDataModule(
        saved_dir=args.data_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=False,
        smoke_batch_size=max(2, args.batch_size),
    )
    dm.setup()
    # Load trained classifier weights (and PoE if present) from checkpoint
    model = MortalityClassifier.load_from_checkpoint(
        args.classifier_ckpt,
        poe_model=poe,
        map_location="cpu",
    )
    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "discriminative_eval")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    trainer = pl.Trainer(logger=logger, callbacks=[TQDMProgressBar()], log_every_n_steps=10)
    # Run only test (no fit)
    trainer.test(model, dataloaders=dm.test_dataloader())


def main():
    # Peek stage
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stage", type=str, choices=["A", "B", "C"], required=False)
    s_args, _ = p.parse_known_args()
    if s_args.stage == "C":
        return _run_discriminative_eval()
    _inject_ckpt_type_from_stage()
    return _run_pretrain_eval()


if __name__ == "__main__":
    main()
