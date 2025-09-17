#!/usr/bin/env python3
"""
Train a binary classifier on variable-length SOFA score sequences (seqsetvae_poe variant).

This script mirrors exp/train_sofa_classifier.py but is co-located under seqsetvae_poe
for easier access when working with SOFA score generation code. It includes:
- patient_id preference for label alignment
- automatic mapping of split ids from ts_id to patient_id using oc.csv when needed
- explicit printing of test metrics after evaluation
"""

from __future__ import annotations

import argparse
import os
import glob
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
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def _infer_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
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


def _maybe_map_split_ids_to_patient(
    splits: Dict[str, List[str]], oc_df: pd.DataFrame
) -> Dict[str, List[str]]:
    def _norm(v):
        try:
            return str(int(float(v)))
        except Exception:
            return str(v)

    has_pid = "patient_id" in oc_df.columns
    has_tid = "ts_id" in oc_df.columns
    if not has_pid:
        return splits

    oc_pids = set(oc_df["patient_id"].map(_norm))
    oc_tids = set(oc_df["ts_id"].map(_norm)) if has_tid else set()
    ts2pid = (
        dict(zip(oc_df["ts_id"].map(_norm), oc_df["patient_id"].map(_norm)))
        if has_tid
        else {}
    )

    sampled: List[str] = []
    for part in ("train", "valid", "test"):
        ids = splits.get(part, [])
        if ids:
            sampled.extend(ids[: min(1000, len(ids))])
    in_pid = sum(1 for s in sampled if s in oc_pids)
    in_tid = sum(1 for s in sampled if s in oc_tids)

    if has_tid and in_tid > in_pid:
        mapped: Dict[str, List[str]] = {}
        for part, ids in splits.items():
            mapped_ids = [ts2pid[s] for s in ids if s in ts2pid]
            mapped[part] = mapped_ids
        return mapped
    return splits


@dataclass
class PatientSequence:
    patient_id: str
    features: np.ndarray
    label: int


class SofaSequenceDataset(Dataset):
    def __init__(self, patient_ids: List[str], data_map: Dict[str, PatientSequence]):
        self.patient_ids = patient_ids
        self.data_map = data_map

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        item = self.data_map[pid]
        x = torch.from_numpy(item.features)
        y = int(item.label)
        T = x.shape[0]
        return x, y, T, pid


def pad_collate(batch: List[Tuple[torch.Tensor, int, int, str]]):
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
        self.pos_weight = torch.tensor([float(pos_weight)]) if pos_weight is not None else None

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
        self.train_auc.reset(); self.train_auprc.reset(); self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auc.compute(), prog_bar=True)
        self.log("val_auprc", self.val_auprc.compute(), prog_bar=False)
        self.log("val_acc", self.val_acc.compute(), prog_bar=False)
        self.val_auc.reset(); self.val_auprc.reset(); self.val_acc.reset()

    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auc.compute(), prog_bar=True)
        self.log("test_auprc", self.test_auprc.compute(), prog_bar=False)
        self.log("test_acc", self.test_acc.compute(), prog_bar=False)
        self.test_auc.reset(); self.test_auprc.reset(); self.test_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt


def build_sequences(
    sofa_csv: str,
    label_csv: str,
    split_dir: str,
    feature_set: str,
    max_len: Optional[int] = None,
    normalize: bool = False,
) -> Tuple[Dict[str, PatientSequence], Dict[str, List[str]], List[str]]:
    df = pd.read_csv(sofa_csv)
    if "event_time" not in df.columns:
        raise ValueError("SOFA CSV must contain 'event_time' column")
    df["patient_id"] = df["patient_id"].apply(_normalize_patient_id_to_str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    if df["event_time"].isna().any():
        raise ValueError("Invalid timestamps present in SOFA CSV")

    if "sofa_total" in df.columns:
        df = df[df["sofa_total"].notna()].reset_index(drop=True)

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
    features_df = df[cols].astype(float)

    df_feat = pd.concat(
        [df[["patient_id", "event_time"]].reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1,
    )
    if feature_set == "total":
        df_feat = df_feat.sort_values(["patient_id", "event_time"]).copy()
        df_feat = df_feat[df_feat["sofa_total"].notna()].reset_index(drop=True)
    else:
        df_feat = df_feat.sort_values(["patient_id", "event_time"]).copy()
        df_feat = df_feat.dropna(subset=cols).reset_index(drop=True)

    if len(df_feat) > 0:
        df_feat["dt_hours"] = (
            df_feat.groupby("patient_id")["event_time"].diff().dt.total_seconds().div(3600.0)
        )
        df_feat["dt_hours"] = df_feat["dt_hours"].fillna(0.0).clip(lower=0.0)
        feature_names = cols + ["dt_hours"]
    else:
        feature_names = cols

    oc = pd.read_csv(label_csv)
    if "patient_id" in oc.columns:
        id_col = "patient_id"
        lab_col = _infer_label_columns(oc)[1]
    else:
        id_col, lab_col = _infer_label_columns(oc)
    oc["__pid__"] = oc[id_col].apply(_normalize_patient_id_to_str)
    label_map: Dict[str, int] = {
        str(pid): int(v) for pid, v in zip(oc["__pid__"].tolist(), oc[lab_col].tolist())
    }

    splits = _scan_split_ids(split_dir)
    splits = _maybe_map_split_ids_to_patient(splits, oc)

    seq_map: Dict[str, PatientSequence] = {}
    for pid, g in df_feat.groupby("patient_id", sort=False):
        if pid not in label_map:
            continue
        g_sorted = g.sort_values("event_time")
        feats_np = g_sorted[feature_names].to_numpy(dtype=np.float32, copy=False)
        if feats_np.shape[0] == 0:
            continue
        if max_len is not None and len(feats_np) > max_len:
            feats_np = feats_np[:max_len]
        seq_map[pid] = PatientSequence(patient_id=str(pid), features=feats_np, label=int(label_map[pid]))

    for part in ["train", "valid", "test"]:
        splits[part] = [pid for pid in splits[part] if pid in seq_map]

    return seq_map, splits, feature_names


def compute_pos_weight(patient_ids: List[str], seq_map: Dict[str, PatientSequence]) -> Optional[float]:
    labels = np.array([seq_map[pid].label for pid in patient_ids], dtype=np.int64)
    num_pos = int(labels.sum())
    num_neg = int((labels == 0).sum())
    if num_pos == 0 or num_neg == 0:
        return None
    return float(num_neg / max(1, num_pos))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train GRU classifier on SOFA sequences (seqsetvae_poe)")
    ap.add_argument("--sofa_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--feature_set", default="total", choices=["total", "subscores", "all"])
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--save_dir", type=str, default="/workspace/sofa_cls_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tqdm_refresh", type=int, default=1)
    ap.add_argument("--allow_partial_splits", action="store_true")
    ap.add_argument("--fallback_train_split", choices=["valid", "test", "auto"], default=None)
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
    )

    train_ids = splits["train"]
    val_ids = splits["valid"]
    test_ids = splits["test"]

    if (len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0) and (not args.allow_partial_splits):
        raise RuntimeError(
            "Empty split after alignment; ensure SOFA CSV and labels cover the split_dir patients"
        )

    if len(train_ids) == 0 and args.allow_partial_splits:
        fallback_choice = None
        if args.fallback_train_split == "valid" and len(val_ids) > 0:
            fallback_choice = "valid"
        elif args.fallback_train_split == "test" and len(test_ids) > 0:
            fallback_choice = "test"
        elif args.fallback_train_split == "auto":
            if len(val_ids) > 0:
                fallback_choice = "valid"
            elif len(test_ids) > 0:
                fallback_choice = "test"
        if fallback_choice is not None:
            if fallback_choice == "valid":
                train_ids = val_ids
                val_ids = []
            elif fallback_choice == "test":
                train_ids = test_ids
                test_ids = []
        if len(train_ids) == 0:
            raise RuntimeError(
                "Training split is empty after alignment and no usable fallback provided. "
                "Set --fallback_train_split (valid/test/auto) or fix inputs."
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
    ckpt_filename = f"best-epoch{epoch}-{monitor_metric.upper()}{{{monitor_metric}:.4f}}"
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
    print(f"Training on {len(train_ds)} patients; Valid {valid_count}; Test {test_count}")
    trainer.fit(model, train_loader, val_loader)

    print("\nEvaluating best checkpoint on test set...")
    best_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if best_path and os.path.exists(best_path):
        best_model = GRUSequenceClassifier.load_from_checkpoint(best_path)
    else:
        best_model = model
    if test_loader is not None:
        results = trainer.test(best_model, dataloaders=test_loader)
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            metrics = results[0]
            def _fmt(v: Optional[float]) -> str:
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "na"
            print(
                f"Test metrics -> AUROC: {_fmt(metrics.get('test_auroc'))} | "
                f"AUPRC: {_fmt(metrics.get('test_auprc'))} | "
                f"ACC: {_fmt(metrics.get('test_acc'))} | "
                f"loss: {_fmt(metrics.get('test_loss'))}"
            )
    else:
        print("[WARN] Test split empty or used as training; skipping test evaluation.")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

