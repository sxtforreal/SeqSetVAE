"""
Post-process previously generated LVCF Parquets to keep ONLY SOFA variables and
apply strict Last-Value-Carry-Forward (indefinite carry until a new value).

This script assumes the input Parquets were produced by seqsetvae_poe/LVCF.py
and already contain the following columns:
  [variable, value, time, set_index, age, is_carry, v0..]

Key behavior differences:
- set_index/time and the per-patient timeline are preserved EXACTLY.
- Only SOFA variables are retained.
- Rows are recomputed using strict LVCF (no time window). Output does NOT
  include 'age'. Values are the already-normalized values taken from raw
  observations and then carried forward.
- Embedding columns v0.. are copied from the first occurrence of the variable.

Directory layout mirrors input:
  input_dir/
    train/*.parquet
    valid/*.parquet
    test/*.parquet
  → output_dir/ with the same split structure and filenames.
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SOFA_VARS: List[str] = [
    # Respiratory
    "PO2",  # PaO2
    "FiO2",
    "Intubated",
    # Coagulation
    "Platelet Count",
    # Liver
    "Bilirubin (Total)",
    # Cardiovascular
    "MBP",  # Mean blood pressure
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


def _parse_var_list(maybe_path_or_csv: Optional[str]) -> Optional[List[str]]:
    if not maybe_path_or_csv:
        return None
    if os.path.exists(maybe_path_or_csv):
        with open(maybe_path_or_csv, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return [s.strip() for s in maybe_path_or_csv.split(",") if s.strip()]


def _detect_vcols(df: pd.DataFrame) -> List[str]:
    vcols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
    if not vcols:
        raise ValueError("Input parquet missing embedding columns v0..")
    # Ensure deterministic order
    vcols = sorted(vcols, key=lambda x: int(x[1:]))
    return vcols


def rebuild_strict_lvcf_for_sofa(
    df_in: pd.DataFrame,
    sofa_vars: Set[str],
) -> pd.DataFrame:
    """Rebuild rows using strict LVCF while preserving set_index/time timeline.

    Args:
        df_in: Parquet from prior pipeline with columns
               [variable, value, time, set_index, age, is_carry, v0..]
        sofa_vars: Variables to keep and rebuild.

    Returns:
        New DataFrame with the same schema but only SOFA variables and strict LVCF.
    """
    req = {"variable", "value", "time", "set_index", "age", "is_carry"}
    if not req.issubset(set(df_in.columns)):
        raise ValueError("Input parquet does not match expected schema from LVCF.py")

    df = df_in.copy()
    vcols = _detect_vcols(df)

    # Unique sets, keep original ordering by set_index (stable with time)
    sets = (
        df[["set_index", "time"]].drop_duplicates().sort_values(["set_index"]).to_numpy()
    )

    # Pre-compute variable -> embedding vector (take first seen row)
    var2vec: Dict[str, np.ndarray] = {}
    for v in sorted(sofa_vars):
        sub = df[df["variable"] == v]
        if len(sub) == 0:
            continue
        vec = sub.iloc[0][vcols].to_numpy(dtype=np.float32)
        var2vec[v] = vec

    # Extract raw observations (is_carry==0) for sofa vars
    raw = df[(df["variable"].isin(sofa_vars)) & (df["is_carry"] == 0.0)]
    raw = raw.drop_duplicates(subset=["set_index", "variable"], keep="last")
    # Map (set_index, variable) -> (value, time)
    raw_map: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for _, r in raw.iterrows():
        raw_map[(int(r["set_index"]), str(r["variable"]))] = (
            float(r["value"]),
            float(r["time"]),
        )

    rows: List[Dict[str, float]] = []
    last_time: Dict[str, float] = {}
    last_val: Dict[str, float] = {}

    # Iterate sets chronologically; rebuild per variable
    for set_idx, t in sets:
        set_idx = int(set_idx)
        t = float(t)
        for v in sorted(var2vec.keys()):
            key = (set_idx, v)
            if key in raw_map:
                val, t_obs = raw_map[key]
                rows.append(
                    {
                        "variable": v,
                        "value": float(val),
                        "time": t,
                        "set_index": set_idx,
                        "is_carry": 0.0,
                    }
                )
                last_time[v] = t_obs
                last_val[v] = float(val)
            elif v in last_time:
                rows.append(
                    {
                        "variable": v,
                        "value": float(last_val[v]),
                        "time": t,
                        "set_index": set_idx,
                        "is_carry": 1.0,
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        # No SOFA variables present for this patient; return empty with columns
        out = pd.DataFrame(columns=["variable", "value", "time", "set_index", "is_carry"] + vcols)
        return out

    # Attach embeddings using var2vec
    for c in vcols:
        out[c] = 0.0
    # Fill per variable
    for v, vec in var2vec.items():
        mask = out["variable"] == v
        if mask.any():
            out.loc[mask, vcols] = vec

    # Ensure types
    out["value"] = out["value"].astype(np.float32)
    out["time"] = out["time"].astype(float)
    out["set_index"] = out["set_index"].astype(int)
    out["is_carry"] = out["is_carry"].astype(np.float32)

    # Sort to match original convention
    out = out.sort_values(["time", "variable"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing prior LVCF outputs")
    ap.add_argument("--output_dir", required=True, help="Directory to write post-processed outputs")
    ap.add_argument(
        "--sofa_vars",
        type=str,
        default=None,
        help="Optional: file path or comma-separated SOFA variable names; defaults to built-in",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--smoke", action="store_true", help="Process a single sample from train")
    args = ap.parse_args()

    sofa_vars = set(_parse_var_list(args.sofa_vars) or DEFAULT_SOFA_VARS)
    if len(sofa_vars) == 0:
        raise ValueError("SOFA variable set is empty")

    os.makedirs(args.output_dir, exist_ok=True)
    for part in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)

    def _process(fp: str) -> pd.DataFrame:
        df = pd.read_parquet(fp, engine="pyarrow")
        return rebuild_strict_lvcf_for_sofa(df, sofa_vars)

    if args.smoke:
        files = sorted(glob.glob(os.path.join(args.input_dir, "train", "*.parquet")))
        if len(files) == 0:
            raise FileNotFoundError("No train parquet files found for smoke test")
        fp = files[0]
        out_df = _process(fp)
        out_fp = os.path.join(args.output_dir, "train", os.path.basename(fp))
        out_df.to_parquet(out_fp, engine="pyarrow", index=False)
        print(f"[SMOKE] Wrote {out_fp} shape={out_df.shape}")
        return

    all_files: List[Tuple[str, str]] = []
    for part in ["train", "valid", "test"]:
        files = sorted(glob.glob(os.path.join(args.input_dir, part, "*.parquet")))
        for fp in files:
            all_files.append((part, fp))
    if len(all_files) == 0:
        print("[WARN] No parquet files found under input_dir")
        return

    with tqdm(total=len(all_files), desc="post_sofa_strict_lvcf") as pbar:
        for part, fp in all_files:
            out_fp = os.path.join(args.output_dir, part, os.path.basename(fp))
            if (not args.overwrite) and os.path.exists(out_fp):
                pbar.update(1)
                continue
            out_df = _process(fp)
            out_df.to_parquet(out_fp, engine="pyarrow", index=False)
            pbar.update(1)


if __name__ == "__main__":
    main()

