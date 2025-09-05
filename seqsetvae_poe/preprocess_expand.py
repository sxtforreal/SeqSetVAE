#!/usr/bin/env python3
"""
按指定步骤构建扩展版 per-patient Parquet：
1) 从 /train,/valid,/test 读取每位病人的 EHR Parquet；
2) 所有 minute<0 的时间戳置 0；
3) 按 minute 打包为 raw sets（同一 set 内 minute 完全一致）；
4) 基于 LVCF 规则在窗口（默认 60 分钟）内进行扩展：
   - 查看距离当前 raw set 时间戳 ≤ window 的历史 sets；
   - 对“当前 raw set 不存在”的变量，取最近一次出现的数值，加入当前 set 形成 expanded set；
   - 为每条事件记录 age：若来自当前 raw set，则 age=0；若由历史 carry 而来，则 age=Δt（当前 set 时间戳 - 该变量最近出现的时间戳，单位：分钟）；
   - 生成二元变量 is_carry（raw=0，carry=1）；
   - 为每个 set 分配 set_index，并在每条事件上附带 time（即该 raw set 的时间戳）。

输入：每病人 Parquet 至少包含列 [variable, value, minute]；其余列将由本脚本生成。
输出：包含 [variable, value, minute, time, set_index, age, is_carry] 的 Parquet，保持 train/valid/test 目录结构。
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def expand_patient(df: pd.DataFrame, lvcf_minutes: float) -> pd.DataFrame:
    # 2) minute<0 置 0
    df = df.copy()
    df["minute"] = df["minute"].astype(float)
    df.loc[df["minute"] < 0, "minute"] = 0.0
    df = df.sort_values(["minute", "variable"]).reset_index(drop=True)

    # 3) raw sets：按 minute 划分；为每个 set 分配 set_index
    unique_minutes = sorted(df["minute"].unique().tolist())
    minute_to_setidx = {m: i for i, m in enumerate(unique_minutes)}
    df["set_index"] = df["minute"].map(minute_to_setidx).astype(int)

    # 4) LVCF 扩展（窗口默认 60 分钟）
    window = float(lvcf_minutes)
    last_seen_time: dict = {}
    last_seen_val: dict = {}
    variables = sorted(df["variable"].unique().tolist())

    rows = []
    for m in unique_minutes:
        set_idx = minute_to_setidx[m]
        cur = df[df["minute"] == m]
        present = set(cur["variable"].tolist())

        # 原始事件（raw）：age=0，is_carry=0
        for _, r in cur.iterrows():
            v = r["variable"]
            val = r["value"]
            rows.append({
                "variable": v,
                "value": val,
                "minute": float(m),
                "time": float(m),
                "set_index": int(set_idx),
                "age": 0.0,
                "is_carry": 0.0,
            })
            last_seen_time[v] = float(m)
            last_seen_val[v] = val

        # carry：仅对当前 set 缺失、且在 window 内出现过的变量进行补齐
        for v in variables:
            if v in present:
                continue
            if v in last_seen_time and (float(m) - last_seen_time[v]) <= window:
                dt = float(m) - last_seen_time[v]
                rows.append({
                    "variable": v,
                    "value": last_seen_val[v],
                    "minute": float(m),
                    "time": float(m),
                    "set_index": int(set_idx),
                    "age": float(dt),
                    "is_carry": 1.0,
                })

    out = pd.DataFrame(rows).sort_values(["minute", "variable"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lvcf_minutes", type=float, default=60.0, help="LVCF窗口（分钟）")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for part in ["train", "valid", "test"]:
        os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)

    parts = ["train", "valid", "test"]
    for part in parts:
        files = sorted(glob.glob(os.path.join(args.input_dir, part, "*.parquet")))
        for fp in tqdm(files, desc=f"{part}"):
            df = pd.read_parquet(fp, engine="pyarrow")
            out = expand_patient(df, args.lvcf_minutes)
            out_fp = os.path.join(args.output_dir, part, os.path.basename(fp))
            out.to_parquet(out_fp, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()

