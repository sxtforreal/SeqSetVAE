#!/usr/bin/env python3
"""
Build expanded per-patient Parquet with:
 - bag_raw[t]: raw events (variable,value) at minute t
 - bag_exp[t]: after LVCF up to N hours for best-knowledge values
 - age[t]
 - set_index (monotonic id per minute)
 - is_carry flag per row (1 if value is carried from previous observation)

Input: per-patient Parquet with at least columns [variable,value,minute,age].
Output: Parquet with the same schema plus is_carry and set_index, saved to a new directory.
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def expand_patient(df: pd.DataFrame, lvcf_hours: float) -> pd.DataFrame:
    df = df.sort_values(["minute", "variable"]).reset_index(drop=True)
    # define set_index per unique minute
    df["set_index"] = (df["minute"].diff().fillna(0) != 0).cumsum().astype(int)
    # Expand with last-value-carry-forward within a window
    window = lvcf_hours * 60.0
    # Track last seen value time per variable
    last_time = {}
    last_val = {}
    rows = []
    minutes = sorted(df["minute"].unique().tolist())
    variables = sorted(df["variable"].unique().tolist())
    for m in minutes:
        cur = df[df["minute"] == m]
        present = set(cur["variable"].tolist())
        # add raw rows
        for _, r in cur.iterrows():
            v = r["variable"]
            val = r["value"]
            age = r.get("age", np.nan)
            rows.append({"variable": v, "value": val, "minute": m, "age": age, "set_index": r["set_index"], "is_carry": 0.0})
            last_time[v] = m
            last_val[v] = val
        # fill carries for variables not present at this minute but seen recently
        for v in variables:
            if v in present:
                continue
            if v in last_time and m - last_time[v] <= window:
                rows.append({"variable": v, "value": last_val[v], "minute": m, "age": cur["age"].iloc[0] if len(cur) else np.nan, "set_index": cur["set_index"].iloc[0] if len(cur) else 0, "is_carry": 1.0})
    out = pd.DataFrame(rows).sort_values(["minute", "variable"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lvcf_hours", type=float, default=48)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for part in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)

    parts = ["train", "valid", "test"]
    for part in parts:
        files = sorted(glob.glob(os.path.join(args.input_dir, part, "*.parquet")))
        for fp in tqdm(files, desc=f"{part}"):
            df = pd.read_parquet(fp, engine="pyarrow")
            out = expand_patient(df, args.lvcf_hours)
            out_fp = os.path.join(args.output_dir, part, os.path.basename(fp))
            out.to_parquet(out_fp, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()

