#!/usr/bin/env python3
"""
Visualize feasibility_results.json with heatmaps and scatter plot.

Outputs:
- Heatmaps for AUROC, AUPRC, recall@95spec across models x time windows
- Scatter plot AUROC vs AUPRC (marker style by family, hue by time window, size by recall)
- CSV summary table for easy ranking/filtering

Usage:
  python exp/visualize_feasibility.py \
    --input exp/feasibility_results.json \
    --outdir exp/feasibility_viz \
    --split test \
    --include-cal false
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize feasibility JSON results")
    parser.add_argument("--input", type=str, default="exp/feasibility_results.json", help="Path to feasibility_results.json")
    parser.add_argument("--outdir", type=str, default="exp/feasibility_viz", help="Directory to save figures and tables")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test", "both"], help="Which split to visualize")
    parser.add_argument("--include-cal", type=lambda v: str(v).lower() in {"1", "true", "yes", "y"}, default=False, help="Whether to include calibrated ('_cal') runs")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Figure file format")
    return parser.parse_args()


def parse_result_key(result_key: str) -> Dict[str, object]:
    """
    Parse keys like:
      mlp_w2_tw_72h_test
      linear_last_mu_logvar_tw_allh_valid_cal
      gaussian_mil_tw_24h_test

    Returns dict with fields: model, family, aggregator, time_window, split, calibrated
    """
    if "_tw_" not in result_key:
        # Fallback: attempt to parse best-effort
        parts = result_key.split("_")
        family = parts[0] if parts else "unknown"
        model = result_key
        return {
            "model": model,
            "family": family,
            "aggregator": "_".join(parts[1:]) if len(parts) > 1 else "",
            "time_window": "unknown",
            "split": "unknown",
            "calibrated": False,
        }

    left, right = result_key.split("_tw_", 1)

    right_parts = right.split("_")
    time_window = right_parts[0] if len(right_parts) >= 1 else "unknown"
    split = right_parts[1] if len(right_parts) >= 2 else "unknown"
    calibrated = any(token == "cal" for token in right_parts[2:])

    left_parts = left.split("_")
    family = left_parts[0] if len(left_parts) >= 1 else "unknown"
    aggregator = "_".join(left_parts[1:]) if len(left_parts) > 1 else ""
    model = left

    return {
        "model": model,
        "family": family,
        "aggregator": aggregator,
        "time_window": time_window,
        "split": split,
        "calibrated": calibrated,
    }


def load_results_as_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r") as f:
        results = json.load(f)

    rows: List[Dict[str, object]] = []
    for key, metrics in results.items():
        meta = parse_result_key(key)
        row = {
            **meta,
            # Keep metric names as-is; standardize an alias for recall for convenience
            "auroc": metrics.get("auroc", float("nan")),
            "auprc": metrics.get("auprc", float("nan")),
            "recall@95spec": metrics.get("recall@95spec", float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Create a readable model label with family prefix
    df["model_label"] = df.apply(
        lambda r: f"{r['family']}:{r['aggregator']}" if r["aggregator"] else str(r["family"]), axis=1
    )
    return df


def ensure_outdir(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def save_summary_table(df: pd.DataFrame, outdir: str) -> None:
    columns = [
        "model",
        "model_label",
        "family",
        "aggregator",
        "time_window",
        "split",
        "calibrated",
        "auroc",
        "auprc",
        "recall@95spec",
    ]
    path = os.path.join(outdir, "summary_table.csv")
    df.loc[:, columns].sort_values(["split", "auroc", "auprc", "recall@95spec"], ascending=[True, False, False, False]).to_csv(path, index=False)


def make_heatmap(
    df: pd.DataFrame,
    metric: str,
    split: str,
    outdir: str,
    file_format: str,
) -> None:
    # Pivot to models x time_window
    pivot = (
        df.pivot_table(index="model_label", columns="time_window", values=metric, aggfunc="mean")
        .sort_values(by=list(df["time_window"].unique()), ascending=False)
    )

    # Order rows by best value across time windows
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(8.0 + 1.2 * max(0, len(pivot.columns) - 3), max(5.5, 0.45 * len(pivot.index))))
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, cbar_kws={"label": metric})
    ax.set_title(f"{metric} by model and time window ({split})")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Model")
    plt.tight_layout()

    outfile = os.path.join(outdir, f"heatmap_{metric}_{split}.{file_format}")
    plt.savefig(outfile, dpi=200)
    plt.close()


def make_scatter(df: pd.DataFrame, split: str, outdir: str, file_format: str) -> None:
    # One point per (model_label, time_window)
    agg = (
        df.groupby(["model_label", "family", "time_window"], as_index=False)
        .agg({"auroc": "mean", "auprc": "mean", "recall@95spec": "mean"})
    )

    plt.figure(figsize=(9.5, 7.0))
    ax = sns.scatterplot(
        data=agg,
        x="auroc",
        y="auprc",
        hue="time_window",
        style="family",
        size="recall@95spec",
        sizes=(50, 350),
        palette="Set2",
    )
    ax.set_title(f"AUROC vs AUPRC ({split}); size ~ recall@95spec")
    ax.set_xlim(0.45, 0.85)
    ax.set_ylim(0.10, 0.45)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Optional light labeling for top points by AUPRC
    try:
        top = agg.sort_values(["auprc", "auroc"], ascending=[False, False]).head(8)
        for _, r in top.iterrows():
            ax.text(r["auroc"] + 0.002, r["auprc"] + 0.002, f"{r['model_label']}\n{r['time_window']}", fontsize=8, alpha=0.8)
    except Exception:
        pass

    plt.tight_layout()
    outfile = os.path.join(outdir, f"scatter_auroc_vs_auprc_{split}.{file_format}")
    plt.savefig(outfile, dpi=200)
    plt.close()


def run_visualization(args: argparse.Namespace) -> None:
    ensure_outdir(args.outdir)
    df_all = load_results_as_dataframe(args.input)

    target_splits: List[str]
    if args.split == "both":
        target_splits = sorted(df_all["split"].dropna().unique().tolist())
    else:
        target_splits = [args.split]

    for split in target_splits:
        df = df_all[df_all["split"] == split].copy()
        if not args.include_cal:
            df = df[~df["calibrated"]]

        if df.empty:
            print(f"[WARN] No rows for split='{split}', include_cal={args.include_cal}")
            continue

        # Save summary table for this split
        save_summary_table(df, args.outdir)

        # Heatmaps
        for metric in ["auroc", "auprc", "recall@95spec"]:
            make_heatmap(df, metric=metric, split=split, outdir=args.outdir, file_format=args.format)

        # Scatter
        make_scatter(df, split=split, outdir=args.outdir, file_format=args.format)

    print(f"Visualization complete. See '{args.outdir}' for outputs.")


def main() -> None:
    args = parse_args()
    # Consistent style
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
    run_visualization(args)


if __name__ == "__main__":
    main()

