#!/usr/bin/env python3
"""
Build split mapping from a split_dir structure into a JSON mapping:
{
  "train": [ts_id, ...],
  "valid": [ts_id, ...],
  "test":  [ts_id, ...]
}

Rules enforced:
- train and test must not overlap (remove any overlaps from train)
- valid contains all of test (ensure superset by unioning valid with test)

By default, scans files under:
  split_dir/{train,valid,test}/*.parquet

Usage:
  python exp/build_split_mapping.py --split_dir /path/to/splits [--ext parquet] [--output mapping.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _list_ids(split_dir: str, part: str, ext: str) -> List[str]:
    pattern = os.path.join(split_dir, part, f"*.{ext}")
    paths = sorted(glob.glob(pattern))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return _dedupe_preserve_order(ids)


def build_split_mapping(split_dir: str, ext: str = "parquet") -> Dict[str, List[str]]:
    parts = ["train", "valid", "test"]
    mapping: Dict[str, List[str]] = {p: [] for p in parts}

    # Gracefully handle missing subdirectories by treating them as empty
    for part in parts:
        part_dir = os.path.join(split_dir, part)
        if os.path.isdir(part_dir):
            mapping[part] = _list_ids(split_dir, part, ext)

    # Enforce constraints: train∩test=∅; valid ⊇ test
    test_set = set(mapping["test"]) if mapping["test"] else set()
    if mapping["train"] and test_set:
        mapping["train"] = [x for x in mapping["train"] if x not in test_set]

    if mapping["valid"] or test_set:
        # Ensure valid contains test; maintain order: keep original valid order, then append missing test ids
        valid_set = set(mapping["valid"]) if mapping["valid"] else set()
        missing_for_valid = [x for x in mapping["test"] if x not in valid_set]
        mapping["valid"] = _dedupe_preserve_order(mapping["valid"] + missing_for_valid)

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split mapping from split_dir")
    parser.add_argument("--split_dir", required=True, help="Root directory with train/valid/test subdirs")
    parser.add_argument("--ext", default="parquet", help="File extension to scan (default: parquet)")
    parser.add_argument("--output", default=None, help="Optional path to write JSON mapping")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    args = parser.parse_args()

    if not os.path.isdir(args.split_dir):
        print(f"Error: split_dir does not exist or is not a directory: {args.split_dir}", file=sys.stderr)
        sys.exit(1)

    mapping = build_split_mapping(args.split_dir, args.ext)

    # Basic sanity: ensure no overlap between train and test after enforcement
    train_set = set(mapping.get("train", []))
    test_set = set(mapping.get("test", []))
    if train_set & test_set:
        print("Warning: train and test still overlap after enforcement.", file=sys.stderr)

    # Ensure valid ⊇ test
    valid_set = set(mapping.get("valid", []))
    if not test_set.issubset(valid_set):
        print("Warning: valid does not fully contain test after enforcement.", file=sys.stderr)

    out_json = json.dumps(mapping, indent=args.indent, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json + "\n")
    else:
        print(out_json)


if __name__ == "__main__":
    main()

