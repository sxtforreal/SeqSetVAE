#!/usr/bin/env python3
"""
Utility to create a schema.csv for SetVAE probabilistic head.

It scans Parquet files under <data_dir>/train (optionally valid/test), collects unique
feature identifiers from the 'variable' column and inspects their 'value' distributions
(to guess types). It writes a CSV with columns:

- feature_id: global integer id (0..F-1)
- type: one of {'cont','bin','cat'}
- cardinality: positive integer for 'cat' (ignored for 'cont'/'bin')
- name: optional human-readable name (taken from 'variable' column when string)

Heuristics for type inference (can be overridden later by manual edits):
- If all observed values are in {0,1} -> 'bin'
- Else if values appear discrete with small support (<=20 unique integer values) and are contiguous integers
  (contiguous range allowed to start from any integer, not necessarily 0) -> 'cat' with K=unique_count
- Else -> 'cont'

Usage:
  python -u seqsetvae_poe/make_schema_csv.py \
    --data_dir /path/to/SeqSetVAE \
    --out_dir  /path/to/save \
    --include_valid --include_test \
    --max_files 200

Outputs: <out_dir>/schema.csv
"""
from __future__ import annotations
import os
import glob
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # fallback if tqdm not available
    def tqdm(x, **kwargs):  # type: ignore
        return x


def _list_parquets(root: str, include_valid: bool, include_test: bool, max_files: int) -> List[str]:
    parts = ["train"] + (["valid"] if include_valid else []) + (["test"] if include_test else [])
    files: List[str] = []
    for p in parts:
        base = os.path.join(root, p)
        files.extend(sorted(glob.glob(os.path.join(base, "*.parquet"))))
    if max_files > 0 and len(files) > max_files:
        files = files[:max_files]
    return files


def _infer_type(values: np.ndarray, max_cat_unique: int = 20) -> Tuple[str, int]:
    # values: 1-D numeric array (float) possibly with NaNs
    v = values[np.isfinite(values)]
    if v.size == 0:
        return "cont", 0
    # Binary
    uniq = np.unique(v)
    if uniq.size <= 2 and set(np.round(uniq).astype(int).tolist()).issubset({0, 1}):
        return "bin", 2
    # Small-integer categorical (allow non-zero start; require contiguity within integer range)
    uniq_int = np.unique(np.round(v).astype(int))
    if uniq_int.size <= max_cat_unique:
        # contiguous integers even if starting from non-zero (or negative)
        if int(uniq_int.max()) - int(uniq_int.min()) + 1 == int(uniq_int.size):
            return "cat", int(uniq_int.size)
    return "cont", 0


def build_schema(data_dir: str, include_valid: bool, include_test: bool, max_files: int) -> pd.DataFrame:
    files = _list_parquets(data_dir, include_valid, include_test, max_files)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}/train")

    # Accumulators
    # Map key -> sample of values and type of variable column
    # Key will be the raw 'variable' entry (string or integer)
    stats: Dict[Any, List[float]] = {}
    is_name_like: bool | None = None

    print(f"Scanning {len(files)} parquet files for schema...")
    for path in tqdm(files, desc="Scanning", unit="file"):
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except Exception:
            continue
        if "variable" not in df.columns or "value" not in df.columns:
            continue
        var_col = df["variable"]
        if is_name_like is None:
            is_name_like = (var_col.dtype == object)
        # Iterate rows (vectorized grouping could be used but keep simple and memory-friendly)
        for raw, val in zip(var_col.to_numpy(), df["value"].to_numpy(dtype=float)):
            if raw not in stats:
                stats[raw] = []
            if np.isfinite(val):
                lst = stats[raw]
                if len(lst) < 1000:  # cap per-feature sample for memory
                    lst.append(float(val))

    print("Assigning global feature ids and inferring types...")
    # Assign global ids deterministically (sorted by name for strings; by numeric order for ints)
    keys = list(stats.keys())
    if len(keys) == 0:
        raise RuntimeError("No features found via 'variable' column")
    if isinstance(keys[0], str) or (is_name_like is True):
        keys_sorted = sorted((str(k) for k in keys))
        name_to_id = {name: i for i, name in enumerate(keys_sorted)}
        records = []
        for name in keys_sorted:
            vals = np.asarray(stats[name], dtype=float)
            typ, card = _infer_type(vals)
            records.append({"feature_id": name_to_id[name], "type": typ, "cardinality": card, "name": name})
        df_out = pd.DataFrame.from_records(records)
    else:
        ids_sorted = sorted(int(k) for k in keys)
        records = []
        for fid in ids_sorted:
            vals = np.asarray(stats[fid], dtype=float)
            typ, card = _infer_type(vals)
            records.append({"feature_id": int(fid), "type": typ, "cardinality": card})
        df_out = pd.DataFrame.from_records(records)

    # Sanity: ensure feature_id are unique and in [0..F-1]
    df_out = df_out.sort_values(by=["feature_id"]).reset_index(drop=True)
    max_id = int(df_out["feature_id"].max())
    if set(df_out["feature_id"]) != set(range(max_id + 1)):
        # Reindex to contiguous ids
        remap = {old: new for new, old in enumerate(sorted(df_out["feature_id"].tolist()))}
        df_out["feature_id"] = df_out["feature_id"].map(remap)
    return df_out


def main():
    ap = argparse.ArgumentParser(description="Create schema.csv for SetVAE probabilistic head")
    ap.add_argument("--data_dir", type=str, required=True, help="Root data dir containing train/valid/test parquet folders")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write schema.csv")
    ap.add_argument("--include_valid", action="store_true", default=False)
    ap.add_argument("--include_test", action="store_true", default=False)
    ap.add_argument("--max_files", type=int, default=200, help="Max parquet files to scan")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    schema_df = build_schema(args.data_dir, args.include_valid, args.include_test, args.max_files)
    out_path = os.path.join(args.out_dir, "schema.csv")
    schema_df.to_csv(out_path, index=False)
    print(f"Wrote schema to: {out_path}")
    print("Summary:")
    print(schema_df["type"].value_counts())


if __name__ == "__main__":
    main()
