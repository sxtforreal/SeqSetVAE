#!/usr/bin/env python3
"""
Quick diagnostic: verify split patient ids are covered by label CSV and SOFA CSV.

Usage:
  python -u exp/debug_sofa_alignment.py \
    --sofa_csv /path/to/sofa.csv \
    --label_csv /path/to/labels.csv \
    --split_dir /path/to/split_root

Notes:
  - This script aligns by patient_id whenever available. If labels contain both
    'ts_id' and 'patient_id', it builds a ts_id -> patient_id mapping and uses
    patient_id for alignment. If only one id exists, it will fall back to that.
  - Split patient files are expected to be named like <patient_id>.parquet under
    train/valid/test.
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


def load_label_ids(label_csv: str) -> Tuple[Dict[str, int], set, Dict[str, str] | None]:
    """
    Load labels and return a mapping keyed by patient_id (preferred) and the set of
    labeled patient_ids.

    Fallbacks:
      - If only 'ts_id' exists, use it as the id (assume equivalence to patient_id).
      - If both exist, normalize both and key labels by patient_id.
    """
    oc = pd.read_csv(label_csv)

    # Infer label column
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

    has_ts = "ts_id" in oc.columns
    has_pid = "patient_id" in oc.columns

    if has_ts:
        oc["__ts__"] = oc["ts_id"].apply(_normalize_id)
    if has_pid:
        oc["__pid__"] = oc["patient_id"].apply(_normalize_id)

    if has_pid and has_ts:
        # Build ts_id -> patient_id mapping and key labels by patient_id
        ts_to_pid: Dict[str, str] = dict(zip(oc["__ts__"].tolist(), oc["__pid__"].tolist()))
        labels = {str(r["__pid__"]): int(r[lab_col]) for _, r in oc.iterrows()}
        return labels, set(labels.keys()), ts_to_pid
    if has_pid:
        labels = {str(r["__pid__"]): int(r[lab_col]) for _, r in oc.iterrows()}
        return labels, set(labels.keys()), None
    elif has_ts:
        labels = {str(r["__ts__"]): int(r[lab_col]) for _, r in oc.iterrows()}
        return labels, set(labels.keys()), None
    else:
        raise ValueError("label_csv must contain either 'patient_id' or 'ts_id'")


def load_sofa_ids(sofa_csv: str) -> set:
    """
    Return the set of patient identifiers present in SOFA CSV.

    Preference: use 'patient_id' when available; otherwise fall back to 'ts_id'.
    """
    df = pd.read_csv(sofa_csv)
    id_col: str | None = None
    if "patient_id" in df.columns:
        id_col = "patient_id"
    elif "ts_id" in df.columns:
        id_col = "ts_id"
    else:
        raise ValueError("SOFA CSV must contain an identifier column 'patient_id' or 'ts_id'")
    df[id_col] = df[id_col].apply(_normalize_id)
    # Only keep rows with non-null sofa_total, mirroring training script behavior
    if "sofa_total" in df.columns:
        df = df[df["sofa_total"].notna()].reset_index(drop=True)
    return set(df[id_col].unique().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sofa_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--split_dir", required=True)
    args = ap.parse_args()

    splits_ts = scan_split_ts_ids(args.split_dir)
    labels_map, label_ids, ts_to_pid = load_label_ids(args.label_csv)
    # If labels provide a ts->pid mapping, translate split file ids to patient_ids
    if ts_to_pid is not None:
        mapped: Dict[str, List[str]] = {}
        for part, ids in splits_ts.items():
            mapped[part] = [_normalize_id(ts_to_pid.get(i, i)) for i in ids]
        splits_ts = mapped
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

