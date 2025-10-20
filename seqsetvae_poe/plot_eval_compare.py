#!/usr/bin/env python3
import json
import os
import glob
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OLD_DIR = ROOT / "eval"
NEW_DIR = ROOT / "eval_new"
FIG_DIR = ROOT / "figs"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_eval_json(dir_path: Path) -> Path:
    cands = sorted(dir_path.glob("pretrain_eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No pretrain_eval_*.json under {dir_path}")
    return cands[0]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    rec = d.get("reconstruction", {}).get("summary", {})
    kl = d.get("kl", {})
    eff = kl.get("effective_dimensions", {})
    post = d.get("posterior_moments", {})
    mu_stats = post.get("mu_abs_mean_stats", {})
    var_stats = post.get("var_mean_stats", {})

    metrics = {
        # Reconstruction
        "nn_l2_mean": rec.get("nn_l2_mean"),
        "nn_l2_median": rec.get("nn_l2_median"),
        "nn_l2_p95": rec.get("nn_l2_p95"),
        "dir_cosine_mean": rec.get("dir_cosine_mean"),
        "mag_mae": rec.get("mag_mae"),
        "mag_rmse": rec.get("mag_rmse"),
        "mag_corr": rec.get("mag_corr"),
        "chamfer_l2_mean": rec.get("chamfer_l2_mean"),
        "scale_ratio": rec.get("scale_ratio"),
        "kl_total_mean": rec.get("kl_total_mean"),
        # KL & dims
        "total_kl": kl.get("total_kl"),
        "dim90": kl.get("dim90"),
        "dim95": kl.get("dim95"),
        "participation_ratio": eff.get("participation_ratio"),
        "entropy_dim": eff.get("entropy_dim"),
        # Posterior moments (stats)
        "mu_abs_mean_mean": mu_stats.get("mean"),
        "mu_abs_mean_median": mu_stats.get("median"),
        "mu_abs_mean_p95": mu_stats.get("p95"),
        "var_mean_mean": var_stats.get("mean"),
        "var_mean_median": var_stats.get("median"),
        "var_mean_p95_abs_var_minus_1": var_stats.get("p95_abs_var_minus_1"),
    }
    return metrics


def get_arrays(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    post = d.get("posterior_moments", {})
    kl = d.get("kl", {})
    return {
        "kl_per_dim": np.array(kl.get("mean_per_dim", []), dtype=float),
        "mu_abs_mean": np.array(post.get("mu_abs_mean", []), dtype=float),
        "var_mean": np.array(post.get("var_mean", []), dtype=float),
    }


def bar_compare(metric_names: List[str], old_vals: List[float], new_vals: List[float], title: str, fname: str, ylog: bool=False):
    x = np.arange(len(metric_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(metric_names)*0.9), 5))
    ax.bar(x - width/2, old_vals, width, label='Old (eval)')
    ax.bar(x + width/2, new_vals, width, label='New (eval_new)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha='right')
    ax.legend()
    if ylog:
        ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)


def hist_overlay(data_old: np.ndarray, data_new: np.ndarray, title: str, xlabel: str, fname: str, bins: int=30, xlog: bool=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    common_kwargs = dict(alpha=0.5, density=True)
    if xlog:
        # Use log-spaced bins if all values positive
        min_val = max(min(data_old.min(), data_new.min()), 1e-8)
        max_val = max(data_old.max(), data_new.max())
        edges = np.geomspace(min_val, max_val, bins)
        ax.hist(data_old, bins=edges, label='Old (eval)', **common_kwargs)
        ax.hist(data_new, bins=edges, label='New (eval_new)', **common_kwargs)
        ax.set_xscale('log')
    else:
        ax.hist(data_old, bins=bins, label='Old (eval)', **common_kwargs)
        ax.hist(data_new, bins=bins, label='New (eval_new)', **common_kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)


def scatter_compare(x_old: np.ndarray, y_new: np.ndarray, title: str, xlabel: str, ylabel: str, fname: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x_old, y_new, s=18, alpha=0.6)
    # y=x reference
    min_v = float(min(x_old.min(), y_new.min()))
    max_v = float(max(x_old.max(), y_new.max()))
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_delta_table(metrics_old: Dict[str, float], metrics_new: Dict[str, float], fname: str, title: str = "Old vs New (delta)"):
    # Render a small table-like figure with metric -> old, new, delta, rel%
    rows = []
    for k in [
        'nn_l2_mean','nn_l2_median','nn_l2_p95','dir_cosine_mean','mag_mae','mag_rmse','mag_corr','chamfer_l2_mean','scale_ratio',
        'total_kl','kl_total_mean','dim90','dim95','participation_ratio','entropy_dim',
        'mu_abs_mean_mean','mu_abs_mean_median','mu_abs_mean_p95','var_mean_mean','var_mean_median','var_mean_p95_abs_var_minus_1',
    ]:
        if metrics_old.get(k) is None or metrics_new.get(k) is None:
            continue
        old = float(metrics_old[k])
        new = float(metrics_new[k])
        delta = new - old
        rel = (delta / old * 100.0) if old != 0 else float('inf')
        rows.append((k, old, new, delta, rel))
    if not rows:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, min(18, 1.2 + 0.35*len(rows))))
    ax.axis('off')
    col_labels = ['Metric', 'Old', 'New', 'Delta', 'Delta %']
    table_data = []
    for k, old, new, delta, rel in rows:
        table_data.append([k, f"{old:.6g}", f"{new:.6g}", f"{delta:+.6g}", f"{rel:+.2f}%"]) 
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200)
    plt.close(fig)


def main():
    old_json = find_eval_json(OLD_DIR)
    new_json = find_eval_json(NEW_DIR)
    d_old = load_json(old_json)
    d_new = load_json(new_json)

    m_old = extract_metrics(d_old)
    m_new = extract_metrics(d_new)

    arr_old = get_arrays(d_old)
    arr_new = get_arrays(d_new)

    # 1) Reconstruction error metrics (log scale)
    err_keys = ['nn_l2_mean','nn_l2_median','nn_l2_p95','chamfer_l2_mean','mag_mae','mag_rmse']
    old_vals = [m_old.get(k) for k in err_keys]
    new_vals = [m_new.get(k) for k in err_keys]
    bar_compare(err_keys, old_vals, new_vals, title='Reconstruction errors (lower is better, log scale)', fname='recon_errors_log.png', ylog=True)

    # 2) Reconstruction quality metrics
    qual_keys = ['dir_cosine_mean','mag_corr','scale_ratio']
    old_vals = [m_old.get(k) for k in qual_keys]
    new_vals = [m_new.get(k) for k in qual_keys]
    bar_compare(qual_keys, old_vals, new_vals, title='Reconstruction quality (higher/closer-to-1 is better)', fname='recon_quality.png', ylog=False)

    # 3) KL & dimensionality
    kld_keys = ['total_kl','dim90','dim95','participation_ratio','entropy_dim']
    old_vals = [m_old.get(k) for k in kld_keys]
    new_vals = [m_new.get(k) for k in kld_keys]
    bar_compare(kld_keys, old_vals, new_vals, title='KL and effective dimensions', fname='kl_and_dims.png', ylog=False)

    # 4) Posterior moment stats
    post_keys = ['mu_abs_mean_mean','mu_abs_mean_median','mu_abs_mean_p95','var_mean_mean','var_mean_median','var_mean_p95_abs_var_minus_1']
    old_vals = [m_old.get(k) for k in post_keys]
    new_vals = [m_new.get(k) for k in post_keys]
    bar_compare(post_keys, old_vals, new_vals, title='Posterior moment statistics', fname='posterior_stats.png', ylog=False)

    # 5) KL per-dimension histogram overlay (log x)
    if arr_old['kl_per_dim'].size > 0 and arr_new['kl_per_dim'].size > 0:
        hist_overlay(arr_old['kl_per_dim'], arr_new['kl_per_dim'], title='KL per-dimension (overlay)', xlabel='KL', fname='kl_per_dim_hist_overlay.png', bins=40, xlog=True)
        scatter_compare(arr_old['kl_per_dim'], arr_new['kl_per_dim'], title='KL per-dimension: New vs Old', xlabel='Old KL', ylabel='New KL', fname='kl_per_dim_scatter.png')

    # 6) mu_abs_mean histogram overlay
    if arr_old['mu_abs_mean'].size > 0 and arr_new['mu_abs_mean'].size > 0:
        hist_overlay(arr_old['mu_abs_mean'], arr_new['mu_abs_mean'], title='|mu| per-dimension (overlay)', xlabel='|mu|', fname='mu_abs_mean_hist_overlay.png', bins=40, xlog=False)

    # 7) var_mean histogram overlay
    if arr_old['var_mean'].size > 0 and arr_new['var_mean'].size > 0:
        hist_overlay(arr_old['var_mean'], arr_new['var_mean'], title='Posterior var per-dimension (overlay)', xlabel='var', fname='var_mean_hist_overlay.png', bins=40, xlog=False)

    # 8) Delta table figure
    save_delta_table(m_old, m_new, fname='metrics_delta_table.png', title='Old (eval) vs New (eval_new) metrics: delta and %')

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
