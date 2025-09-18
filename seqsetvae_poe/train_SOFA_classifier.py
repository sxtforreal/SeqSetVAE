#!/usr/bin/env python3
"""
Train a binary classifier on variable-length SOFA score sequences.

Inputs:
- A CSV produced by `seqsetvae_poe/SOFA.py` with columns including either
  an identifier column 'patient_id' or 'ts_id', plus:
  ['event_time', 'set_index'(optional), 'respiratory_score', 'coagulation_score',
   'liver_score', 'cardiovascular_score', 'cns_score', 'renal_score', 'sofa_total']
- An outcome CSV ("oc") with columns like ['ts_id', 'in_hospital_mortality'].
  If 'patient_id' is absent, it will be inferred from 'ts_id' (identity mapping).
- EITHER a JSON file specifying split ts_id lists, e.g. {"train": ["200001", ...],
  "valid": [...], "test": [...]}, OR a directory containing train/valid/test
  per-patient Parquets under split_dir/{train,valid,test}/*.parquet.

When --split_json is provided, splits are taken directly from that file.
Otherwise, the script derives splits by listing parquet basenames in
split_dir/{train,valid,test}.

Model:
- GRU-based sequence classifier using variable-length packing
- Features selectable: 'total', 'subscores', or 'all'
- When feature_set='total', the script selects only 'sofa_total' per patient and
  automatically drops rows where 'sofa_total' is NaN (no forward-fill or zero-fill).
  For 'subscores' or 'all', rows with any NaN among selected features are removed
  (no forward-fill or zero-fill). Additionally, a time-interval feature `dt_hours`
  (difference between consecutive `event_time` within each patient; first step = 0)
  is appended to the input features for all modes.

Usage example:
  python -u exp/train_sofa_classifier.py \
    --sofa_csv /path/to/sofa_scores.csv \
    --label_csv /path/to/oc.csv \
    --split_dir /path/to/patient_ehr \
    --feature_set total \
    --epochs 20 --batch_size 64 --bidirectional

Outputs:
- Best checkpoint by validation AUROC in --save_dir
- Summary metrics printed for train/valid/test
"""

from __future__ import annotations

import argparse
import os
import glob
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchmetrics.classification import AUROC, AveragePrecision, Accuracy


def _normalize_patient_id_to_str(value) -> str:
    """Robustly normalize patient identifiers to canonical string form.

    Accepts strings like "200003.0" or numeric types and returns "200003".
    """

    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def _infer_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Infer id and label column names in the oc dataframe.

    Returns a tuple (id_col, label_col). We will only use label_col in the
    current pipeline but keep the utility intact.
    """

    id_candidates = ["ts_id", "patient_id", "subject_id", "id"]
    label_candidates = [
        "in_hospital_mortality",
        "mortality",
        "label",
        "outcome",
    ]
    id_col = None
    lab_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    for c in label_candidates:
        if c in df.columns:
            lab_col = c
            break
    if id_col is None or lab_col is None:
        raise ValueError(
            f"Could not infer id/label columns in oc CSV. Columns present: {list(df.columns)}"
        )
    return id_col, lab_col


def _scan_split_ids(split_dir: str) -> Dict[str, List[str]]:
    """Return patient ids per split by scanning {train,valid,test} subdirs.

    Patient id is derived from parquet basename without extension, then normalized.
    """

    out: Dict[str, List[str]] = {}
    for part in ["train", "valid", "test"]:
        pattern = os.path.join(split_dir, part, "*.parquet")
        paths = sorted(glob.glob(pattern))
        ids = [
            _normalize_patient_id_to_str(os.path.splitext(os.path.basename(p))[0])
            for p in paths
        ]
        if len(ids) == 0:
            raise FileNotFoundError(
                f"No parquet files under {os.path.join(split_dir, part)}"
            )
        out[part] = ids
    return out


## Heuristic mapping removed in favor of strict ts_id -> patient_id via label CSV


def _load_split_ids_from_json(split_json: str) -> Dict[str, List[str]]:
    """Load split ts_id arrays from a JSON file.

    Expected structure:
    {
      "train": ["200001", "200006", ...],
      "valid": [ ... ],
      "test":  [ ... ]
    }

    Values may be strings or numbers; they will be normalized to canonical strings.
    Missing keys default to empty lists.
    """

    with open(split_json, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("split_json must contain a JSON object with keys train/valid/test")
    out: Dict[str, List[str]] = {}
    for part in ["train", "valid", "test"]:
        raw_list = obj.get(part, [])
        if raw_list is None:
            raw_list = []
        if not isinstance(raw_list, list):
            raise ValueError(f"split_json[{part}] must be a list")
        out[part] = [_normalize_patient_id_to_str(v) for v in raw_list]
    if len(out["train"]) == 0:
        raise ValueError("split_json must include a non-empty 'train' list")
    return out


def _build_feature_matrix(
    df: pd.DataFrame, feature_set: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Select and ensure numeric feature columns.

    Returns a view of df with selected feature columns and list of column names.
    """

    feature_set = feature_set.lower()
    subs = [
        "respiratory_score",
        "coagulation_score",
        "liver_score",
        "cardiovascular_score",
        "cns_score",
        "renal_score",
    ]
    if feature_set == "total":
        cols = ["sofa_total"]
    elif feature_set == "subscores":
        cols = subs
    elif feature_set in {"all", "subscores+total"}:
        cols = subs + ["sofa_total"]
    else:
        raise ValueError("feature_set must be one of: total, subscores, all")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"SOFA CSV missing columns: {missing}")

    return df[cols].astype(float), cols


@dataclass
class PatientSequence:
    patient_id: str
    features: np.ndarray  # [T, F] float32
    label: int


class SofaSequenceDataset(Dataset):
    """Dataset of per-patient variable-length SOFA feature sequences.

    The constructor expects the already-filtered list of patient ids for this split
    and a dictionary mapping patient id to PatientSequence.
    """

    def __init__(self, patient_ids: List[str], data_map: Dict[str, PatientSequence]):
        self.patient_ids = patient_ids
        self.data_map = data_map

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        pid = self.patient_ids[idx]
        item = self.data_map[pid]
        x = torch.from_numpy(item.features)  # [T, F]
        y = int(item.label)
        T = x.shape[0]
        return x, y, T, pid


def pad_collate(batch: List[Tuple[torch.Tensor, int, int, str]]):
    """Pad variable-length sequences to max length in batch and build mask.

    Returns dict with tensors:
      - x: [B, T_max, F]
      - lengths: [B]
      - y: [B]
      - padding_mask: [B, T_max] (True for padding)
      - patient_ids: list[str]
    """

    batch_size = len(batch)
    lengths = [b[2] for b in batch]
    max_len = int(max(lengths)) if lengths else 1
    feature_dim = int(batch[0][0].shape[1]) if batch else 1
    x = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    y = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    patient_ids: List[str] = []
    for i, (seq, label, T, pid) in enumerate(batch):
        if T > 0:
            x[i, :T] = seq
            padding_mask[i, :T] = False
        y[i] = int(label)
        patient_ids.append(pid)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    return {
        "x": x,
        "lengths": lengths_t,
        "y": y,
        "padding_mask": padding_mask,
        "patient_ids": patient_ids,
    }


class GRUSequenceClassifier(pl.LightningModule):
    """GRU-based binary sequence classifier with packing and metrics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = (
            torch.tensor([float(pos_weight)]) if pos_weight is not None else None
        )

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, max(32, out_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, out_dim // 2), 1),
        )

        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        if self.bidirectional:
            last_layer_out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_layer_out = h_n[-1]
        logits = self.head(last_layer_out).squeeze(1)
        return logits

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits, y.float(), pos_weight=self.pos_weight.to(logits.device)
            )
        return F.binary_cross_entropy_with_logits(logits, y.float())

    def _shared_step(self, batch, stage: str):
        x, lengths, y = batch["x"], batch["lengths"], batch["y"]
        logits = self(x, lengths)
        loss = self._compute_loss(logits, y)
        probs = torch.sigmoid(logits)
        if stage == "train":
            self.train_auc.update(probs, y)
            self.train_auprc.update(probs, y)
            self.train_acc.update(probs, y)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "val":
            self.val_auc.update(probs, y)
            self.val_auprc.update(probs, y)
            self.val_acc.update(probs, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.test_auc.update(probs, y)
            self.test_auprc.update(probs, y)
            self.test_acc.update(probs, y)
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train_auroc", self.train_auc.compute(), prog_bar=True)
        self.log("train_auprc", self.train_auprc.compute(), prog_bar=False)
        self.log("train_acc", self.train_acc.compute(), prog_bar=False)
        self.train_auc.reset()
        self.train_auprc.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auc.compute(), prog_bar=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=False)
        self.log("val_acc", self.val_acc.compute(), prog_bar=False)
        self.val_auc.reset()
        self.val_auprc.reset()
        self.val_acc.reset()

    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auc.compute(), prog_bar=True)
        self.log("test_auprc", self.test_auprc.compute(), prog_bar=False)
        self.log("test_acc", self.test_acc.compute(), prog_bar=False)
        self.test_auc.reset()
        self.test_auprc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return opt


def build_sequences(
    sofa_csv: str,
    label_csv: str,
    split_dir: Optional[str],
    feature_set: str,
    max_len: Optional[int] = None,
    normalize: bool = False,
    debug_alignment: bool = False,
    split_json: Optional[str] = None,
) -> Tuple[Dict[str, PatientSequence], Dict[str, List[str]], List[str]]:
    """Load data, align splits and build per-patient sequences.

    Returns:
      - seq_map: patient_id -> PatientSequence
      - splits: {'train': [...], 'valid': [...], 'test': [...]}
      - feature_names: list[str]
    """

    df = pd.read_csv(sofa_csv)
    if "event_time" not in df.columns:
        raise ValueError("SOFA CSV must contain 'event_time' column")
    # Ensure a canonical 'ts_id' column exists for alignment
    if "ts_id" not in df.columns:
        if "patient_id" in df.columns:
            df["ts_id"] = df["patient_id"]
        else:
            raise ValueError("SOFA CSV must contain an identifier column 'ts_id' or 'patient_id'")
    df["ts_id"] = df["ts_id"].apply(_normalize_patient_id_to_str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    if df["event_time"].isna().any():
        raise ValueError("Invalid timestamps present in SOFA CSV")

    # Always drop sets where 'sofa_total' is NaN, regardless of feature selection
    if "sofa_total" in df.columns:
        df = df[df["sofa_total"].notna()].reset_index(drop=True)

    features_df, feature_names = _build_feature_matrix(df, feature_set)

    df_feat = pd.concat(
        [
            df[["ts_id", "event_time"]].reset_index(drop=True),
            features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Handle missing values according to feature selection
    if feature_set.lower() == "total":
        # Use only sofa_total; drop rows with NaN (no fill)
        df_feat = df_feat.sort_values(["ts_id", "event_time"]).copy()
        df_feat = df_feat[df_feat["sofa_total"].notna()].reset_index(drop=True)
    else:
        # Multi-feature: drop rows that contain any NaN in selected features
        df_feat = df_feat.sort_values(["ts_id", "event_time"]).copy()
        df_feat = df_feat.dropna(subset=feature_names).reset_index(drop=True)

    # Append time-interval feature (hours) between consecutive sets per patient
    if len(df_feat) > 0:
        df_feat["dt_hours"] = (
            df_feat.groupby("ts_id")["event_time"]
            .diff()
            .dt.total_seconds()
            .div(3600.0)
        )
        # First step has NaN diff; set to 0. Also clamp negatives to 0 (guard malformed time order)
        df_feat["dt_hours"] = df_feat["dt_hours"].fillna(0.0).clip(lower=0.0)
        feature_names = feature_names + ["dt_hours"]

    # Load labels (must contain 'ts_id'; 'patient_id' optional)
    oc = pd.read_csv(label_csv)
    if "ts_id" not in oc.columns:
        if "patient_id" in oc.columns:
            oc["ts_id"] = oc["patient_id"]
        else:
            raise ValueError("label_csv must contain column 'ts_id' or 'patient_id'")
    # Determine label column
    _, lab_col = _infer_label_columns(oc)
    oc["__ts__"] = oc["ts_id"].apply(_normalize_patient_id_to_str)

    # Build overlap on ts_id between OC and SOFA
    df["__ts__"] = df["ts_id"].apply(_normalize_patient_id_to_str)
    ts_overlap: set[str] = set(oc["__ts__"].tolist()) & set(df["__ts__"].tolist())

    # Labels keyed by ts_id (restricted to overlap)
    label_map: Dict[str, int] = {
        ts: int(lab)
        for ts, lab in zip(oc["__ts__"].tolist(), oc[lab_col].tolist())
        if ts in ts_overlap
    }

    # Build split ts_id lists: prefer explicit JSON if provided, else scan directory
    if split_json is None and split_dir is None:
        raise ValueError("Provide either --split_json or --split_dir for building splits")
    if split_json is not None:
        raw_splits_ts = _load_split_ids_from_json(split_json)
    else:
        raw_splits_ts = _scan_split_ids(str(split_dir))
    if debug_alignment:
        print("[align] Raw split counts (ts_id):", {k: len(v) for k, v in raw_splits_ts.items()})
        present_ts = {k: sum(1 for s in v if s in ts_overlap) for k, v in raw_splits_ts.items()}
        print("[align] ts_id present in OC∩SOFA:", present_ts)
    # Enforce split assumptions: train∩test=∅; valid may contain test -> use valid\test
    train_ts = list(dict.fromkeys(raw_splits_ts.get("train", [])))
    valid_ts_raw = list(dict.fromkeys(raw_splits_ts.get("valid", [])))
    test_ts = list(dict.fromkeys(raw_splits_ts.get("test", [])))
    # Remove any accidental overlap between train and test
    test_set = set(test_ts)
    train_ts = [t for t in train_ts if t not in test_set]
    # valid := valid \ test
    valid_ts = [v for v in valid_ts_raw if v not in test_set]
    # Finally restrict each split to the OC∩SOFA overlap
    overlap = ts_overlap
    splits: Dict[str, List[str]] = {
        "train": [t for t in train_ts if t in overlap],
        "valid": [v for v in valid_ts if v in overlap],
        "test": [t for t in test_ts if t in overlap],
    }
    if debug_alignment:
        print("[align] Split counts after enforcing overlap & valid\\test:", {k: len(v) for k, v in splits.items()})

    # Optionally fit normalization on training set only
    mean_vec = None
    std_vec = None
    if normalize:
        train_df = df_feat[df_feat["ts_id"].isin(set(splits["train"]))]
        if len(train_df) == 0:
            raise RuntimeError(
                "No training rows found in SOFA CSV for the provided splits"
            )
        mean_vec = (
            train_df[feature_names]
            .astype(float)
            .mean(axis=0)
            .to_numpy(dtype=np.float32)
        )
        std_raw = (
            train_df[feature_names].astype(float).std(axis=0).to_numpy(dtype=np.float32)
        )
        std_vec = np.where(np.abs(std_raw) < 1e-6, 1.0, std_raw)

    # Build per-patient sequences
    seq_map: Dict[str, PatientSequence] = {}
    # Track which patients have at least one valid row after feature/NaN filtering
    available_ts_ids = set(df_feat["ts_id"].unique().tolist())
    for pid, g in df_feat.groupby("ts_id", sort=False):
        if pid not in label_map:
            continue
        g_sorted = g.sort_values("event_time")
        feats_np = g_sorted[feature_names].to_numpy(dtype=np.float32, copy=False)
        if feats_np.shape[0] == 0:
            # Skip patients whose sequence becomes empty after dropping NaNs
            continue
        if max_len is not None and len(feats_np) > max_len:
            feats_np = feats_np[:max_len]
        if normalize and mean_vec is not None and std_vec is not None:
            feats_np = (feats_np - mean_vec) / std_vec
        seq_map[pid] = PatientSequence(
            patient_id=str(pid), features=feats_np, label=int(label_map[pid])
        )

    # Filter split ids to those available in seq_map
    for part in ["train", "valid", "test"]:
        before = list(splits[part])
        splits[part] = [pid for pid in before if pid in seq_map]
        if debug_alignment:
            dropped_missing_label = [pid for pid in before if pid not in label_map]
            dropped_missing_sofa = [pid for pid in before if pid not in available_ts_ids]
            dropped_other = [pid for pid in before if pid not in seq_map and pid in label_map and pid in available_ts_ids]
            print(
                f"[align] {part}: kept={len(splits[part])}, dropped_missing_label={len(dropped_missing_label)}, dropped_missing_sofa={len(dropped_missing_sofa)}, dropped_other={len(dropped_other)}"
            )

    return seq_map, splits, feature_names


def compute_pos_weight(
    patient_ids: List[str], seq_map: Dict[str, PatientSequence]
) -> Optional[float]:
    labels = np.array([seq_map[pid].label for pid in patient_ids], dtype=np.int64)
    num_pos = int(labels.sum())
    num_neg = int((labels == 0).sum())
    if num_pos == 0 or num_neg == 0:
        return None
    return float(num_neg / max(1, num_pos))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train GRU classifier on SOFA sequences")
    ap.add_argument("--sofa_csv", required=True, help="CSV from seqsetvae_poe/SOFA.py")
    ap.add_argument(
        "--label_csv", required=True, help="Outcome CSV with mortality labels (oc)"
    )
    ap.add_argument(
        "--split_dir",
        default=None,
        help="Root directory containing train/valid/test per-patient Parquets",
    )
    ap.add_argument(
        "--split_json",
        default=None,
        help="JSON file specifying split ts_id lists: keys train/valid/test",
    )
    ap.add_argument(
        "--feature_set",
        default="total",
        choices=["total", "subscores", "all"],
        help="Which SOFA features to use",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Z-normalize features using training set statistics",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Optional cap on sequence length (truncate head)",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (epochs)"
    )
    ap.add_argument("--save_dir", type=str, default="/workspace/sofa_cls_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tqdm_refresh", type=int, default=1, help="TQDM refresh rate (steps)"
    )
    ap.add_argument(
        "--debug_alignment",
        action="store_true",
        help="Print detailed alignment diagnostics for splits/labels/SOFA",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    seq_map, splits, feature_names = build_sequences(
        sofa_csv=args.sofa_csv,
        label_csv=args.label_csv,
        split_dir=args.split_dir,
        feature_set=args.feature_set,
        max_len=args.max_len,
        normalize=args.normalize,
        debug_alignment=args.debug_alignment,
        split_json=args.split_json,
    )

    train_ids = splits["train"]
    val_ids = splits["valid"]
    test_ids = splits["test"]

    # Require only non-empty training set; validation/test optional
    if len(train_ids) == 0:
        raise RuntimeError(
            "Empty training split after alignment; ensure SOFA/labels and split mapping overlap"
        )

    train_ds = SofaSequenceDataset(train_ids, seq_map)
    val_ds = SofaSequenceDataset(val_ids, seq_map)
    test_ds = SofaSequenceDataset(test_ids, seq_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=pad_collate,
        )
        if len(val_ids) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=pad_collate,
        )
        if len(test_ids) > 0
        else None
    )

    # Diagnostics: batches per epoch and class balance
    try:
        num_train_batches = len(train_loader)
    except TypeError:
        num_train_batches = None
    train_labels = [seq_map[pid].label for pid in train_ids]
    num_train_pos = int(sum(train_labels))
    num_train_neg = int(len(train_labels) - num_train_pos)

    pos_w = compute_pos_weight(train_ids, seq_map)
    model = GRUSequenceClassifier(
        input_dim=len(feature_names),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_w,
    )

    monitor_metric = "val_auroc" if val_loader is not None else "train_auroc"
    ckpt_filename = (
        f"best-epoch{{epoch}}-{monitor_metric.upper()}{{{monitor_metric}:.4f}}"
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=ckpt_filename,
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(monitor=monitor_metric, mode="max", patience=args.patience)
    prog_cb = TQDMProgressBar(refresh_rate=max(1, int(args.tqdm_refresh)))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[ckpt_cb, es_cb, prog_cb],
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,
    )

    valid_count = len(val_ds) if val_loader is not None else 0
    test_count = len(test_ds) if test_loader is not None else 0
    print(
        f"Training on {len(train_ds)} patients; Valid {valid_count}; Test {test_count}"
    )
    if num_train_batches is not None:
        print(
            f"Train batches per epoch: {num_train_batches} (batch_size={args.batch_size})"
        )
        if num_train_batches == 1:
            print(
                "[Hint] Only one training batch this epoch. This usually means batch_size >= number of training patients, or very few patients remain after aligning SOFA/labels with split_dir. Try a smaller --batch_size or verify your splits."
            )
    print(f"Train label balance -> pos: {num_train_pos}, neg: {num_train_neg}")
    trainer.fit(model, train_loader, val_loader)

    print("\nEvaluating best checkpoint on test set...")
    best_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if best_path and os.path.exists(best_path):
        best_model = GRUSequenceClassifier.load_from_checkpoint(best_path)
    else:
        best_model = model
    if test_loader is not None:
        results = trainer.test(best_model, dataloaders=test_loader)
        # Explicitly print key test metrics (AUROC, AUPRC) after testing
        if (
            isinstance(results, list)
            and len(results) > 0
            and isinstance(results[0], dict)
        ):
            metrics = results[0]

            def _fmt(v: Optional[float]) -> str:
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "na"

            auroc_v = _fmt(metrics.get("test_auroc"))
            auprc_v = _fmt(metrics.get("test_auprc"))
            acc_v = _fmt(metrics.get("test_acc"))
            loss_v = _fmt(metrics.get("test_loss"))
            print(
                f"Test metrics -> AUROC: {auroc_v} | AUPRC: {auprc_v} | ACC: {acc_v} | loss: {loss_v}"
            )
    else:
        print("[WARN] Test split empty or used as training; skipping test evaluation.")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
