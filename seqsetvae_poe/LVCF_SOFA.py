"""
LVCF-SOFA: Expand per-patient raw EHR Parquet but keep ONLY SOFA variables and
apply strict Last-Value-Carry-Forward (carry indefinitely until new value).

This mirrors the I/O and schema of seqsetvae_poe/LVCF.py, except:
- Variables are restricted to a SOFA list
- LVCF ignores any time window (always carry until a new observation)

Inputs and outputs follow the same directory layout:
  input_dir/
    train/*.parquet
    valid/*.parquet
    test/*.parquet

Outputs mirror the input split layout and include columns:
  [variable, value, time, set_index, age, is_carry, v0..v{D-1}]
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


# Default list of variables used in SOFA, matched to names present in the dataset
DEFAULT_SOFA_VARS: List[str] = [
    # Respiratory
    "PO2",  # PaO2
    "FiO2",
    "Intubated",  # indicator of respiratory support
    # Coagulation
    "Platelet Count",
    # Liver
    "Bilirubin (Total)",
    # Cardiovascular
    "MBP",  # Mean blood pressure (MAP)
    "Dopamine",
    "Norepinephrine",
    "Epinephrine",
    # Neurological
    "GCS_eye",
    "GCS_motor",
    "GCS_verbal",
    # Renal
    "Creatinine Blood",
    "Urine",
]


def _detect_key_column(df: pd.DataFrame) -> str:
    candidates = ["Key", "variable", "event", "name", "key"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]


def _load_event_embeddings(csv_path: str) -> Tuple[pd.DataFrame, str, int]:
    cache = pd.read_csv(csv_path)
    key_col = _detect_key_column(cache)
    numeric_cols = [
        c for c in cache.columns if c != key_col and pd.api.types.is_numeric_dtype(cache[c])
    ]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric embedding columns found in event_emb_csv")
    numeric_cols = sorted(numeric_cols, key=lambda x: (len(x), x))
    emb_dim = len(numeric_cols)
    rename_map = {c: f"v{i}" for i, c in enumerate(numeric_cols)}
    emb_df = cache[[key_col] + numeric_cols].rename(columns=rename_map)
    return emb_df, key_col, emb_dim


def _load_value_stats(csv_path: str) -> Tuple[pd.DataFrame, str]:
    stats = pd.read_csv(csv_path)
    key_col = _detect_key_column(stats)
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


def _parse_var_list(maybe_path_or_csv: Optional[str]) -> Optional[List[str]]:
    if not maybe_path_or_csv:
        return None
    if os.path.exists(maybe_path_or_csv):
        with open(maybe_path_or_csv, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return [s.strip() for s in maybe_path_or_csv.split(",") if s.strip()]


def expand_patient_strict_lvcf_sofa(
    df: pd.DataFrame,
    sofa_vars: Set[str],
) -> pd.DataFrame:
    """Expand events to per-set rows with strict LVCF for SOFA variables only.

    - set_index is defined by unique minutes across ALL events (not just SOFA).
    - Only SOFA variables contribute rows (raw or carry).
    - Carry is indefinite: once observed, a variable is carried at all later sets
      until a new observation arrives.
    - age is minutes since last observation of that variable.
    """
    df = df.copy()
    df["minute"] = df["minute"].astype(float)
    df.loc[df["minute"] < 0, "minute"] = 0.0
    df = df.sort_values(["minute", "variable"]).reset_index(drop=True)

    # Unique minutes from the full raw stream (so set timeline matches prior parquet)
    unique_minutes: List[float] = sorted(df["minute"].unique().tolist())
    minute_to_setidx: Dict[float, int] = {m: i for i, m in enumerate(unique_minutes)}

    # State per variable
    last_seen_time: Dict[str, float] = {}
    last_seen_val: Dict[str, float] = {}

    # We will only ever emit rows for SOFA variables actually observed
    sofa_vars_sorted: List[str] = sorted(list(sofa_vars))

    rows: List[Dict[str, float]] = []
    for m in unique_minutes:
        set_idx = minute_to_setidx[m]
        cur = df[df["minute"] == m]
        # De-duplicate within-minute observations per variable; keep last
        if len(cur) > 1:
            cur = cur.drop_duplicates(subset=["variable"], keep="last")

        # Record raw observations only for SOFA variables
        present_sofa: Set[str] = set()
        for _, r in cur.iterrows():
            v = r["variable"]
            if v not in sofa_vars:
                continue
            val = r["value"]
            present_sofa.add(v)
            rows.append(
                {
                    "variable": v,
                    "value": val,
                    "time": float(m),
                    "set_index": int(set_idx),
                    "age": 0.0,
                    "is_carry": 0.0,
                }
            )
            last_seen_time[v] = float(m)
            last_seen_val[v] = val

        # Strict LVCF: carry indefinitely for all previously seen SOFA variables
        for v in sofa_vars_sorted:
            if v in present_sofa:
                continue
            if v in last_seen_time:
                dt = float(m) - last_seen_time[v]
                rows.append(
                    {
                        "variable": v,
                        "value": last_seen_val[v],
                        "time": float(m),
                        "set_index": int(set_idx),
                        "age": float(dt),
                        "is_carry": 1.0,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["time", "variable"]).reset_index(drop=True)
    return out


def _apply_embeddings(df: pd.DataFrame, emb_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    merged = df.merge(emb_df, how="left", left_on="variable", right_on=key_col)
    if "variable_x" in merged.columns:
        merged = merged.rename(columns={"variable_x": "variable"})
    if "variable_y" in merged.columns:
        merged = merged.drop(columns=["variable_y"])
    if key_col in merged.columns and key_col != "variable":
        merged = merged.drop(columns=[key_col])
    vcols = [c for c in merged.columns if c.startswith("v") and c[1:].isdigit()]
    if len(vcols) == 0:
        raise RuntimeError("No embedding columns v0.. present after merge")
    for c in vcols:
        merged[c] = merged[c].astype(np.float32).fillna(0.0)
    return merged


def _apply_value_normalization(df: pd.DataFrame, stats_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    merged = df.merge(stats_df, how="left", left_on="variable", right_on=key_col)
    if "variable_x" in merged.columns:
        merged = merged.rename(columns={"variable_x": "variable"})
    if "variable_y" in merged.columns:
        merged = merged.drop(columns=["variable_y"])
    if key_col in merged.columns and key_col != "variable":
        merged = merged.drop(columns=[key_col])
    std_safe = merged["std"].replace(0, np.nan).fillna(1.0).astype(np.float32)
    mean_safe = merged["mean"].fillna(0.0).astype(np.float32)
    merged["value"] = (merged["value"].astype(np.float32) - mean_safe) / std_safe
    return merged.drop(columns=["mean", "std"])


def _validate_output(df: pd.DataFrame, emb_dim: int) -> None:
    expect = ["variable", "value", "time", "set_index", "age", "is_carry"]
    missing = [c for c in expect if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing expected columns: {missing}")
    vcols = [f"v{i}" for i in range(emb_dim)]
    missing_v = [c for c in vcols if c not in df.columns]
    if missing_v:
        raise AssertionError(f"Missing embedding columns: {missing_v[:5]} ...")
    if not np.isfinite(df["value"].to_numpy(dtype=np.float32)).all():
        raise AssertionError("Normalized 'value' contains non-finite entries")
    if not set(np.unique(df["is_carry"]).tolist()).issubset({0.0, 1.0}):
        raise AssertionError("'is_carry' must be 0/1")
    if (df["age"] < 0).any():
        raise AssertionError("'age' must be >= 0")


def _process_one_file(
    fp: str,
    emb_df: pd.DataFrame,
    emb_key: str,
    stats_df: pd.DataFrame,
    stats_key: str,
    sofa_vars: Set[str],
) -> pd.DataFrame:
    raw = pd.read_parquet(fp, engine="pyarrow")
    drop_cols = [c for c in ["TABLE", "abnormal"] if c in raw.columns]
    if drop_cols:
        raw = raw.drop(columns=drop_cols)
    if not {"variable", "value", "minute"}.issubset(set(raw.columns)):
        raise ValueError(
            f"Input parquet {fp} must contain columns ['variable','value','minute']"
        )
    # Expand using strict LVCF for SOFA variables only (timeline from all minutes)
    exp = expand_patient_strict_lvcf_sofa(raw, sofa_vars)
    exp = _apply_embeddings(exp, emb_df, emb_key)
    exp = _apply_value_normalization(exp, stats_df, stats_key)
    return exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Root directory with train/valid/test per-patient Parquets",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output root directory (mirrors train/valid/test)",
    )
    ap.add_argument(
        "--event_emb_csv",
        required=True,
        help="CSV mapping event name -> embedding dims (v0..)",
    )
    ap.add_argument(
        "--value_stats_csv",
        required=True,
        help="CSV mapping event name -> value mean/std",
    )
    ap.add_argument(
        "--sofa_vars",
        type=str,
        default=None,
        help=(
            "Optional: comma-separated list or a file (one per line) specifying SOFA variables. "
            "Defaults to a built-in set."
        ),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Process and validate a single sample (first train file)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    emb_df, emb_key, emb_dim = _load_event_embeddings(args.event_emb_csv)
    stats_df, stats_key = _load_value_stats(args.value_stats_csv)

    # Resolve SOFA vars
    sofa_vars_list = _parse_var_list(args.sofa_vars)
    if sofa_vars_list is None:
        sofa_vars_list = DEFAULT_SOFA_VARS
    sofa_vars: Set[str] = set([v for v in sofa_vars_list if v])
    if len(sofa_vars) == 0:
        raise ValueError("Resolved SOFA variable set is empty")

    os.makedirs(args.output_dir, exist_ok=True)
    for part in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)

    if args.smoke:
        train_files = sorted(glob.glob(os.path.join(args.input_dir, "train", "*.parquet")))
        if len(train_files) == 0:
            raise FileNotFoundError("No train parquet files found for smoke test")
        fp = train_files[0]
        out_df = _process_one_file(fp, emb_df, emb_key, stats_df, stats_key, sofa_vars)
        _validate_output(out_df, emb_dim)
        out_fp = os.path.join(args.output_dir, "train", os.path.basename(fp))
        out_df.to_parquet(out_fp, engine="pyarrow", index=False)
        print(
            f"[SMOKE] Success. Wrote: {out_fp} with shape {out_df.shape} and emb_dim={emb_dim}"
        )
        return

    # Full processing over all splits with a single tqdm
    all_files: List[Tuple[str, str]] = []
    for part in ["train", "valid", "test"]:
        files = sorted(glob.glob(os.path.join(args.input_dir, part, "*.parquet")))
        if len(files) == 0:
            print(f"[WARN] No files under {part}")
            continue
        for fp in files:
            all_files.append((part, fp))
    if len(all_files) == 0:
        print("[WARN] No parquet files found under input_dir")
        return

    with tqdm(total=len(all_files), desc="sofa_strict_lvcf") as pbar:
        for part, fp in all_files:
            out_fp = os.path.join(args.output_dir, part, os.path.basename(fp))
            if (not args.overwrite) and os.path.exists(out_fp):
                pbar.update(1)
                continue
            out_df = _process_one_file(fp, emb_df, emb_key, stats_df, stats_key, sofa_vars)
            _validate_output(out_df, emb_dim)
            out_df.to_parquet(out_fp, engine="pyarrow", index=False)
            pbar.update(1)


if __name__ == "__main__":
    main()

