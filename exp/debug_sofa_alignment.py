#!/usr/bin/env python3
"""
Quick diagnostic: verify split ts_ids are covered by label CSV and SOFA CSV.

Usage:
  python -u exp/debug_sofa_alignment.py \
    --sofa_csv /path/to/sofa.csv \
    --label_csv /path/to/labels.csv \
    --split_dir /path/to/split_root

Assumptions after your changes:
  - Both SOFA CSV and label CSV use a unified identifier column named 'ts_id'.
  - Split patient files are named like <ts_id>.parquet under train/valid/test.
"""

from __future__ import annotations

import argparse
import os
import glob
from typing import Dict, List, Tuple

import pandas as pd


def _normalize_id(value) -> str:
    try:
        return str(int(value))
    except Exception:
        return str(value)


def scan_split_ts_ids(split_dir: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for part in ["train", "valid", "test"]:
        paths = sorted(glob.glob(os.path.join(split_dir, part, "*.parquet")))
        ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        out[part] = [_normalize_id(x) for x in ids]
    return out


def load_label_ids(label_csv: str) -> Tuple[Dict[str, int], set]:
    oc = pd.read_csv(label_csv)
    if "ts_id" not in oc.columns:
        raise ValueError("label_csv must contain 'ts_id'")
    id_col = "ts_id"
    lab_col: str | None = None
    for c in [
        "in_hospital_mortality",
        "mortality",
        "label",
        "outcome",
    ]:
        if c in oc.columns:
            lab_col = c
            break
    if lab_col is None:
        raise ValueError(
            f"Could not infer label column in label_csv. Columns: {list(oc.columns)}"
        )
    oc[id_col] = oc[id_col].apply(_normalize_id)
    labels = {str(r[id_col]): int(r[lab_col]) for _, r in oc.iterrows()}
    return labels, set(labels.keys())


def load_sofa_ids(sofa_csv: str) -> set:
    df = pd.read_csv(sofa_csv)
    if "ts_id" not in df.columns and "patient_id" in df.columns:
        # allow older files
        df["ts_id"] = df["patient_id"]
    if "ts_id" not in df.columns:
        raise ValueError("SOFA CSV must contain 'ts_id' or 'patient_id'")
    df["ts_id"] = df["ts_id"].apply(_normalize_id)
    # Only keep rows with non-null sofa_total, mirroring training script behavior
    if "sofa_total" in df.columns:
        df = df[df["sofa_total"].notna()].reset_index(drop=True)
    return set(df["ts_id"].unique().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sofa_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--split_dir", required=True)
    args = ap.parse_args()

    splits_ts = scan_split_ts_ids(args.split_dir)
    labels_map, label_ids = load_label_ids(args.label_csv)
    sofa_ids = load_sofa_ids(args.sofa_csv)

    print("[diag] Split counts (ts_id):", {k: len(v) for k, v in splits_ts.items()})
    print("[diag] Label rows:", len(label_ids), "| SOFA patient rows:", len(sofa_ids))

    for part in ["train", "valid", "test"]:
        ts_list = splits_ts.get(part, [])
        present_in_label = [t for t in ts_list if t in label_ids]
        present_in_sofa = [t for t in ts_list if t in sofa_ids]
        present_both = [t for t in ts_list if (t in label_ids and t in sofa_ids)]
        missing_label = [t for t in ts_list if t not in label_ids]
        missing_sofa = [t for t in ts_list if t not in sofa_ids]

        print(
            f"[diag] {part}: kept={len(present_both)}, "+
            f"dropped_missing_label={len(missing_label)}, "+
            f"dropped_missing_sofa={len(missing_sofa)}"
        )
        if len(present_both) == 0:
            # print a few examples to debug
            print(f"  examples missing label (up to 10): {missing_label[:10]}")
            print(f"  examples missing sofa (up to 10): {missing_sofa[:10]}")


if __name__ == "__main__":
    main()

