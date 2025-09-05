#!/usr/bin/env python3
"""
LVCF: Expand per-patient raw EHR Parquet with Last-Value-Carry-Forward and enrich.

Pipeline per patient Parquet:
1) Read raw events with at least [variable, value, minute].
2) Clamp minute<0 to 0, sort by [minute, variable], assign set_index per unique minute.
3) Apply LVCF within a time window (minutes):
   - For variables missing at current minute, carry most recent value within window;
   - age = minutes since last observation (0 for raw events), is_carry in {0,1};
   - time = minute of the set.
4) Map event names to 768-d vectors from an offline CSV and materialize as columns v0..v767.
5) Normalize values using offline per-variable mean/std CSV and overwrite 'value' in place.

Input directory layout:
  input_dir/
    train/*.parquet  # one file per patient
    valid/*.parquet
    test/*.parquet

CSV expectations:
- event_emb_csv: contains a key column for event names and 768 numeric dims. Key column will be
  auto-detected among ['Key','variable','event','name','key'] or fallback to first column.
- value_stats_csv: contains columns [variable, mean, std]. Key column is detected same as above.

Outputs mirror the input split layout and include columns:
  [variable, value, time, set_index, age, is_carry, v0..v767]

Use --smoke to process and validate one sample (first file under train).
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _detect_key_column(df: pd.DataFrame) -> str:
    candidates = ["Key", "variable", "event", "name", "key"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: use the very first column
    return df.columns[0]


def _load_event_embeddings(csv_path: str) -> Tuple[pd.DataFrame, str, int]:
    cache = pd.read_csv(csv_path)
    key_col = _detect_key_column(cache)
    # Keep only numeric columns for embeddings
    numeric_cols = [c for c in cache.columns if c != key_col and pd.api.types.is_numeric_dtype(cache[c])]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric embedding columns found in event_emb_csv")
    # Ensure deterministic order for columns
    numeric_cols = sorted(numeric_cols, key=lambda x: (len(x), x))
    emb_dim = len(numeric_cols)
    # Rename to v0..v{D-1}
    rename_map = {c: f"v{i}" for i, c in enumerate(numeric_cols)}
    emb_df = cache[[key_col] + numeric_cols].rename(columns=rename_map)
    return emb_df, key_col, emb_dim


def _load_value_stats(csv_path: str) -> Tuple[pd.DataFrame, str]:
    stats = pd.read_csv(csv_path)
    key_col = _detect_key_column(stats)
    # Normalize column names for mean/std
    rename = {}
    for c in stats.columns:
        lc = c.lower()
        if lc == "mean":
            rename[c] = "mean"
        if lc == "std" or lc == "stdev" or lc == "stddev":
            rename[c] = "std"
    stats = stats.rename(columns=rename)
    if "mean" not in stats.columns or "std" not in stats.columns:
        raise ValueError("value_stats_csv must contain 'mean' and 'std' columns")
    return stats[[key_col, "mean", "std"]], key_col


def expand_patient(df: pd.DataFrame, lvcf_minutes: float) -> pd.DataFrame:
    df = df.copy()
    # Clamp minute
    df["minute"] = df["minute"].astype(float)
    df.loc[df["minute"] < 0, "minute"] = 0.0
    df = df.sort_values(["minute", "variable"]).reset_index(drop=True)

    # Assign set_index per unique minute
    unique_minutes: List[float] = sorted(df["minute"].unique().tolist())
    minute_to_setidx: Dict[float, int] = {m: i for i, m in enumerate(unique_minutes)}
    df["set_index"] = df["minute"].map(minute_to_setidx).astype(int)

    window = float(lvcf_minutes)
    last_seen_time: Dict[str, float] = {}
    last_seen_val: Dict[str, float] = {}
    variables: List[str] = sorted(df["variable"].unique().tolist())

    rows: List[Dict[str, float]] = []
    for m in unique_minutes:
        set_idx = minute_to_setidx[m]
        cur = df[df["minute"] == m]
        # Ensure each set has unique variables: keep the last observation for duplicates
        if len(cur) > 1:
            cur = cur.drop_duplicates(subset=["variable"], keep="last")
        present = set(cur["variable"].tolist())

        # Raw events
        for _, r in cur.iterrows():
            v = r["variable"]
            val = r["value"]
            rows.append({
                "variable": v,
                "value": val,  # temporary; will be normalized later
                "time": float(m),
                "set_index": int(set_idx),
                "age": 0.0,
                "is_carry": 0.0,
            })
            last_seen_time[v] = float(m)
            last_seen_val[v] = val

        # Carry-forward
        for v in variables:
            if v in present:
                continue
            if v in last_seen_time and (float(m) - last_seen_time[v]) <= window:
                dt = float(m) - last_seen_time[v]
                rows.append({
                    "variable": v,
                    "value": last_seen_val[v],  # temporary; will be normalized later
                    "time": float(m),
                    "set_index": int(set_idx),
                    "age": float(dt),
                    "is_carry": 1.0,
                })

    out = pd.DataFrame(rows).sort_values(["time", "variable"]).reset_index(drop=True)
    return out


def _apply_embeddings(df: pd.DataFrame, emb_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    # Merge on variable name
    merged = df.merge(emb_df, how="left", left_on="variable", right_on=key_col)
    # Ensure we always carry a single column named 'variable' after merge.
    # When key_col == 'variable', pandas will create 'variable_x'/'variable_y'.
    if "variable_x" in merged.columns:
        merged = merged.rename(columns={"variable_x": "variable"})
    if "variable_y" in merged.columns:
        merged = merged.drop(columns=["variable_y"]) 
    # If the right key column is present with a different name, drop it.
    if key_col in merged.columns and key_col != "variable":
        merged = merged.drop(columns=[key_col])
    # Fill missing embeddings with zeros
    vcols = [c for c in merged.columns if c.startswith("v") and c[1:].isdigit()]
    if len(vcols) == 0:
        raise RuntimeError("No embedding columns v0.. present after merge")
    for c in vcols:
        merged[c] = merged[c].astype(np.float32).fillna(0.0)
    return merged


def _apply_value_normalization(df: pd.DataFrame, stats_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    merged = df.merge(stats_df, how="left", left_on="variable", right_on=key_col)
    # Normalize duplicate key columns to a single 'variable'
    if "variable_x" in merged.columns:
        merged = merged.rename(columns={"variable_x": "variable"})
    if "variable_y" in merged.columns:
        merged = merged.drop(columns=["variable_y"]) 
    if key_col in merged.columns and key_col != "variable":
        merged = merged.drop(columns=[key_col])
    # If std is zero/NaN, use 1.0 to avoid division by zero
    std_safe = merged["std"].replace(0, np.nan).fillna(1.0).astype(np.float32)
    mean_safe = merged["mean"].fillna(0.0).astype(np.float32)
    merged["value"] = (merged["value"].astype(np.float32) - mean_safe) / std_safe
    return merged.drop(columns=["mean", "std"])


def _validate_output(df: pd.DataFrame, emb_dim: int) -> None:
    # Basic column presence
    expect = ["variable", "value", "time", "set_index", "age", "is_carry"]
    missing = [c for c in expect if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing expected columns: {missing}")
    # Embedding dims
    vcols = [f"v{i}" for i in range(emb_dim)]
    missing_v = [c for c in vcols if c not in df.columns]
    if missing_v:
        raise AssertionError(f"Missing embedding columns: {missing_v[:5]} ...")
    # Value normalization sanity: finite values
    if not np.isfinite(df["value"].to_numpy(dtype=np.float32)).all():
        raise AssertionError("Normalized 'value' contains non-finite entries")
    # Flags
    if not set(np.unique(df["is_carry"]).tolist()).issubset({0.0, 1.0}):
        raise AssertionError("'is_carry' must be 0/1")
    if (df["age"] < 0).any():
        raise AssertionError("'age' must be >= 0")


def _process_one_file(
    fp: str,
    lvcf_minutes: float,
    emb_df: pd.DataFrame,
    emb_key: str,
    stats_df: pd.DataFrame,
    stats_key: str,
) -> pd.DataFrame:
    raw = pd.read_parquet(fp, engine="pyarrow")
    # Drop irrelevant columns if present
    drop_cols = [c for c in ["TABLE", "abnormal"] if c in raw.columns]
    if drop_cols:
        raw = raw.drop(columns=drop_cols)
    if not {"variable", "value", "minute"}.issubset(set(raw.columns)):
        raise ValueError(f"Input parquet {fp} must contain columns ['variable','value','minute']")
    exp = expand_patient(raw, lvcf_minutes)
    exp = _apply_embeddings(exp, emb_df, emb_key)
    exp = _apply_value_normalization(exp, stats_df, stats_key)
    return exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Root directory with train/valid/test per-patient Parquets")
    ap.add_argument("--output_dir", required=True, help="Output root directory (mirrors train/valid/test)")
    ap.add_argument("--event_emb_csv", required=True, help="CSV mapping event name -> 768-d embedding")
    ap.add_argument("--value_stats_csv", required=True, help="CSV mapping event name -> value mean/std")
    ap.add_argument("--lvcf_minutes", type=float, default=60.0, help="LVCF window size in minutes")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--smoke", action="store_true", help="Process and validate a single sample (first train file)")
    args = ap.parse_args()

    emb_df, emb_key, emb_dim = _load_event_embeddings(args.event_emb_csv)
    stats_df, stats_key = _load_value_stats(args.value_stats_csv)

    os.makedirs(args.output_dir, exist_ok=True)
    for part in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)

    if args.smoke:
        train_files = sorted(glob.glob(os.path.join(args.input_dir, "train", "*.parquet")))
        if len(train_files) == 0:
            raise FileNotFoundError("No train parquet files found for smoke test")
        fp = train_files[0]
        out_df = _process_one_file(fp, args.lvcf_minutes, emb_df, emb_key, stats_df, stats_key)
        _validate_output(out_df, emb_dim)
        out_fp = os.path.join(args.output_dir, "train", os.path.basename(fp))
        out_df.to_parquet(out_fp, engine="pyarrow", index=False)
        print(f"[SMOKE] Success. Wrote: {out_fp} with shape {out_df.shape} and emb_dim={emb_dim}")
        return

    # Full processing
    for part in ["train", "valid", "test"]:
        files = sorted(glob.glob(os.path.join(args.input_dir, part, "*.parquet")))
        if len(files) == 0:
            print(f"[WARN] No files under {part}")
            continue
        for fp in tqdm(files, desc=f"{part}"):
            out_fp = os.path.join(args.output_dir, part, os.path.basename(fp))
            if (not args.overwrite) and os.path.exists(out_fp):
                continue
            out_df = _process_one_file(fp, args.lvcf_minutes, emb_df, emb_key, stats_df, stats_key)
            _validate_output(out_df, emb_dim)
            out_df.to_parquet(out_fp, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()

