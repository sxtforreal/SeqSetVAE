#!/usr/bin/env python3
"""
Real-time SOFA score calculator per patient and per set.

This utility reads a wide-format CSV time series where each row is a patient "set"
(a timestamped measurement bundle). Missing values are forward-filled per patient
to approximate real-time availability, then SOFA subscores are computed and summed.

Usage:
  python sofa_realtime.py --input input.csv --output sofa_scores.csv --mapping sofa_mapping.yaml

If no mapping is provided, the script expects default column names matching the
sample mapping file provided alongside this script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependency missing is handled via README
    yaml = None


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class ColumnMapping:
    """Container for input column names and units.

    All fields are optional; if a field is missing or None, that signal will be
    treated as unavailable and ignored in scoring for that row.
    """

    patient_id: str = "patient_id"
    set_id: Optional[str] = "set_id"
    time: str = "event_time"

    # Respiratory
    pao2_mmHg: Optional[str] = "pao2_mmHg"
    fio2_fraction: Optional[str] = "fio2_fraction"
    spo2_percent: Optional[str] = "spo2_percent"
    mech_vent: Optional[str] = "mech_vent"

    # Coagulation
    platelets_10e9_per_L: Optional[str] = "platelets_10e9_per_L"

    # Liver
    bilirubin_mg_dl: Optional[str] = "bilirubin_mg_dl"
    bilirubin_umol_L: Optional[str] = None

    # Cardiovascular
    map_mmHg: Optional[str] = "map_mmHg"
    norepinephrine_mcg_kg_min: Optional[str] = "norepinephrine_mcg_kg_min"
    epinephrine_mcg_kg_min: Optional[str] = "epinephrine_mcg_kg_min"
    dopamine_mcg_kg_min: Optional[str] = "dopamine_mcg_kg_min"
    dobutamine_mcg_kg_min: Optional[str] = "dobutamine_mcg_kg_min"

    # CNS
    gcs_total: Optional[str] = "gcs_total"

    # Renal
    creatinine_mg_dl: Optional[str] = "creatinine_mg_dl"
    creatinine_umol_L: Optional[str] = None
    urine_output_ml_24h: Optional[str] = "urine_output_ml_24h"


def load_mapping(path: Optional[str]) -> ColumnMapping:
    """Load a ColumnMapping from YAML file or return defaults when None.

    The YAML structure should be flat, using keys that match ColumnMapping fields.
    """

    if path is None:
        return ColumnMapping()

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load a mapping file. Install via requirements.txt"
        )

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Validate keys against ColumnMapping fields
    valid_keys = set(ColumnMapping.__annotations__.keys())
    unknown_keys = [k for k in data.keys() if k not in valid_keys]
    if unknown_keys:
        raise ValueError(
            f"Unknown mapping keys: {unknown_keys}. Valid keys: {sorted(valid_keys)}"
        )

    return ColumnMapping(**data)


# ------------------------------ Helper Utils ------------------------------- #


def to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float, returns None on failure or NaN.

    Handles strings, numbers, and pandas NA types gracefully.
    """

    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if pd.isna(f):
        return None
    return f


def get_value(row: pd.Series, col: Optional[str]) -> Optional[float]:
    if col is None:
        return None
    if col not in row:
        return None
    return to_float(row[col])


def convert_bilirubin_to_mg_dl(
    bilirubin_mg_dl: Optional[float], bilirubin_umol_L: Optional[float]
) -> Optional[float]:
    """Return bilirubin in mg/dL, converting from umol/L when needed.

    Conversion: 1 mg/dL = 17.104 umol/L
    """

    if bilirubin_mg_dl is not None:
        return bilirubin_mg_dl
    if bilirubin_umol_L is None:
        return None
    return bilirubin_umol_L / 17.104


def convert_creatinine_to_mg_dl(
    creatinine_mg_dl: Optional[float], creatinine_umol_L: Optional[float]
) -> Optional[float]:
    """Return creatinine in mg/dL, converting from umol/L when needed.

    Conversion: 1 mg/dL = 88.4 umol/L
    """

    if creatinine_mg_dl is not None:
        return creatinine_mg_dl
    if creatinine_umol_L is None:
        return None
    return creatinine_umol_L / 88.4


def compute_resp_subscore(row: pd.Series, m: ColumnMapping) -> Optional[int]:
    pao2 = get_value(row, m.pao2_mmHg)
    fio2 = get_value(row, m.fio2_fraction)
    spo2 = get_value(row, m.spo2_percent)
    mech_vent_flag = get_value(row, m.mech_vent)
    on_vent = bool(mech_vent_flag) if mech_vent_flag is not None else False

    pf_ratio: Optional[float] = None
    if pao2 is not None and fio2 is not None and fio2 > 0:
        pf_ratio = pao2 / fio2

    # If PF not available, attempt SF ratio approximation
    sf_ratio: Optional[float] = None
    if pf_ratio is None and spo2 is not None and fio2 is not None and fio2 > 0:
        # SpO2 is expected as percent (e.g., 95); FiO2 as fraction (e.g., 0.4)
        sf_ratio = spo2 / fio2

    score: Optional[int] = None
    # Prioritize PF ratio when available
    if pf_ratio is not None:
        if on_vent and pf_ratio < 100:
            score = 4
        elif on_vent and pf_ratio < 200:
            score = 3
        elif pf_ratio < 300:
            score = 2
        elif pf_ratio < 400:
            score = 1
        else:
            score = 0
        return score

    # Approximate using SF ratio if PF unavailable
    if sf_ratio is not None:
        # Approximate thresholds per literature: SF ~ 235 ~ PF 200; SF ~ 315 ~ PF 300
        if on_vent and sf_ratio < 150:
            return 4
        if on_vent and sf_ratio < 235:
            return 3
        if sf_ratio < 315:
            return 2
        if sf_ratio < 400:
            return 1
        return 0

    return None


def compute_coag_subscore(row: pd.Series, m: ColumnMapping) -> Optional[int]:
    platelets = get_value(row, m.platelets_10e9_per_L)
    if platelets is None:
        return None
    if platelets < 20:
        return 4
    if platelets < 50:
        return 3
    if platelets < 100:
        return 2
    if platelets < 150:
        return 1
    return 0


def compute_liver_subscore(row: pd.Series, m: ColumnMapping) -> Optional[int]:
    bilirubin = convert_bilirubin_to_mg_dl(
        get_value(row, m.bilirubin_mg_dl), get_value(row, m.bilirubin_umol_L)
    )
    if bilirubin is None:
        return None
    if bilirubin >= 12.0:
        return 4
    if bilirubin >= 6.0:
        return 3
    if bilirubin >= 2.0:
        return 2
    if bilirubin >= 1.2:
        return 1
    return 0


def compute_cardiovascular_subscore(row: pd.Series, m: ColumnMapping) -> Optional[int]:
    map_value = get_value(row, m.map_mmHg)
    norepi = get_value(row, m.norepinephrine_mcg_kg_min) or 0.0
    epi = get_value(row, m.epinephrine_mcg_kg_min) or 0.0
    dopa = get_value(row, m.dopamine_mcg_kg_min) or 0.0
    dobut = get_value(row, m.dobutamine_mcg_kg_min) or 0.0

    # Vasopressor-based grading has priority over MAP-only grading
    if norepi > 0.1 or epi > 0.1 or dopa > 15:
        return 4
    if (0.0 < norepi <= 0.1) or (0.0 < epi <= 0.1) or (5 < dopa <= 15):
        return 3
    if (0.0 < dopa <= 5) or (dobut > 0.0):
        return 2

    if map_value is None:
        return None
    if map_value < 70:
        return 1
    return 0


def compute_cns_subscore(row: pd.Series, m: ColumnMapping) -> Optional[int]:
    gcs = get_value(row, m.gcs_total)
    if gcs is None:
        return None
    if gcs < 6:
        return 4
    if gcs <= 9:
        return 3
    if gcs <= 12:
        return 2
    if gcs <= 14:
        return 1
    return 0


def compute_renal_subscore(row: pd.Series, m: ColumnMapping) -> Optional[int]:
    creat = convert_creatinine_to_mg_dl(
        get_value(row, m.creatinine_mg_dl), get_value(row, m.creatinine_umol_L)
    )
    urine_24h = get_value(row, m.urine_output_ml_24h)

    creat_score: Optional[int] = None
    if creat is not None:
        if creat >= 5.0:
            creat_score = 4
        elif creat >= 3.5:
            creat_score = 3
        elif creat >= 2.0:
            creat_score = 2
        elif creat >= 1.2:
            creat_score = 1
        else:
            creat_score = 0

    urine_score: Optional[int] = None
    if urine_24h is not None:
        if urine_24h < 200:
            urine_score = 4
        elif urine_24h < 500:
            urine_score = 3
        else:
            urine_score = 0

    if creat_score is None and urine_score is None:
        return None
    if creat_score is None:
        return urine_score
    if urine_score is None:
        return creat_score
    return max(creat_score, urine_score)


def compute_row_scores(row: pd.Series, m: ColumnMapping) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    return (
        compute_resp_subscore(row, m),
        compute_coag_subscore(row, m),
        compute_liver_subscore(row, m),
        compute_cardiovascular_subscore(row, m),
        compute_cns_subscore(row, m),
        compute_renal_subscore(row, m),
    )


def forward_fill_by_patient(df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    # Forward-fill all columns per patient to simulate real-time availability
    filled = (
        df.sort_values([patient_col, "__event_time__"])  # type: ignore[index]
        .groupby(patient_col, as_index=False, sort=False)
        .apply(lambda g: g.ffill())
        .reset_index(drop=True)
    )
    return filled


def compute_sofa(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    # Prepare time column
    if mapping.time not in df.columns:
        raise ValueError(f"Time column '{mapping.time}' not found in input data")
    df = df.copy()
    df["__event_time__"] = pd.to_datetime(df[mapping.time], errors="coerce")
    if df["__event_time__"].isna().any():
        raise ValueError("Some rows have invalid timestamps in time column")

    # Sort and forward-fill per patient
    df = df.sort_values([mapping.patient_id, "__event_time__"]).reset_index(drop=True)
    df = forward_fill_by_patient(df, mapping.patient_id)

    # Compute subscores
    subscores = df.apply(lambda row: compute_row_scores(row, mapping), axis=1, result_type="expand")
    subscores.columns = [
        "respiratory_score",
        "coagulation_score",
        "liver_score",
        "cardiovascular_score",
        "cns_score",
        "renal_score",
    ]

    out = pd.concat([df[[mapping.patient_id, mapping.time] + ([mapping.set_id] if mapping.set_id and mapping.set_id in df.columns else [])].rename(columns={mapping.time: "event_time"}), subscores], axis=1)

    # Total score: sum of available subscores
    out["sofa_total"] = out[[
        "respiratory_score",
        "coagulation_score",
        "liver_score",
        "cardiovascular_score",
        "cns_score",
        "renal_score",
    ]].sum(axis=1, min_count=1)

    # Ensure integer dtype where possible
    for col in [
        "respiratory_score",
        "coagulation_score",
        "liver_score",
        "cardiovascular_score",
        "cns_score",
        "renal_score",
        "sofa_total",
    ]:
        out[col] = out[col].astype("Int64")  # pandas nullable integer

    # Order columns
    base_cols = [mapping.patient_id]
    if mapping.set_id and mapping.set_id in out.columns:
        base_cols.append(mapping.set_id)
    base_cols.append("event_time")

    score_cols = [
        "respiratory_score",
        "coagulation_score",
        "liver_score",
        "cardiovascular_score",
        "cns_score",
        "renal_score",
        "sofa_total",
    ]

    out = out[base_cols + score_cols]
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute real-time SOFA score per patient and per set")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to write output CSV with SOFA scores")
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional YAML file mapping input column names to expected fields. See sofa_mapping.example.yaml",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default ',')",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding (default utf-8)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    mapping = load_mapping(args.mapping)

    try:
        df = pd.read_csv(args.input, delimiter=args.delimiter, encoding=args.encoding)
    except Exception as exc:
        print(f"Failed to read input CSV: {exc}", file=sys.stderr)
        return 2

    required_base_cols = [mapping.patient_id, mapping.time]
    for col in required_base_cols:
        if col not in df.columns:
            print(f"Missing required column '{col}' in input CSV", file=sys.stderr)
            return 3

    try:
        result = compute_sofa(df, mapping)
    except Exception as exc:
        print(f"SOFA computation failed: {exc}", file=sys.stderr)
        return 4

    try:
        result.to_csv(args.output, index=False)
    except Exception as exc:
        print(f"Failed to write output CSV: {exc}", file=sys.stderr)
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
