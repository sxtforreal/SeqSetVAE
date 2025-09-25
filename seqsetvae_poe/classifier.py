#!/usr/bin/env python3
"""
Train a sequence-level mortality classifier on top of a frozen PoE backbone.

Features per time step (built from the PoE fusion pipeline):
  - State: posterior mean mu_post,t (projected), posterior logvar logσ²_post,t (projected)
  - Divergence: KL(q_post||p_prior)_t, Mahalanobis d2_t, |ΔH|_t (entropy difference), gate β_t
  - Structure/Time: set_size (log1p), has_change (0/1), log1p(Δt_t)

Sequence model: 2-layer GRU + attention pooling + MLP → mortality logit.

Usage example:
  python -u seqsetvae_poe/binary_classifier.py \
    --checkpoint /path/to/output/setvae-PT/version_X/checkpoints/poe_GRU_PT.ckpt \
    --label_csv /path/to/labels.csv \
    --data_dir /path/to/SeqSetVAE \
    --batch_size 4 --max_epochs 10 --num_workers 2 \
    --time_mode delta   # or none

label_csv must contain at least two columns: ts_id,in_hospital_mortality (0/1)
where ts_id equals parquet filename stem under {train,valid,test}.
"""

import os
import csv
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

# Optional metrics (fall back gracefully if unavailable)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None  # type: ignore
    average_precision_score = None  # type: ignore

# Local imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in os.sys.path:
    os.sys.path.append(THIS_DIR)

import config as cfg  # type: ignore
from dataset import PatientDataset, _collate_lvcf, _detect_vcols  # type: ignore
from model import PoESeqSetVAEPretrain  # type: ignore


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict")
    # Strip Lightning's 'model.' prefix if present
    if any(k.startswith("model.") for k in state.keys()):
        state = {
            k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")
        }
    return state

def _pid_variants(raw_id: Any) -> List[str]:
    """
    Generate robust id variants to match parquet stems and CSV ids.
    - "200001" <-> "200001.0" and numeric equivalents.
    """
    variants: List[str] = []
    try:
        s = str(raw_id).strip()
    except Exception:
        return variants
    if s == "":
        return variants
    variants.append(s)
    try:
        f = float(s)
        if math.isfinite(f):
            i = int(f)
            variants.append(str(i))
            variants.append(f"{i}.0")
    except Exception:
        pass
    seen = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _build_poe_from_state(state: Dict[str, torch.Tensor]) -> PoESeqSetVAEPretrain:
    # Instantiate PoE with config defaults; weights will be loaded next
    model = PoESeqSetVAEPretrain(
        input_dim=getattr(cfg, "input_dim", 768),
        reduced_dim=getattr(cfg, "reduced_dim", 256),
        latent_dim=getattr(cfg, "latent_dim", 128),
        levels=getattr(cfg, "levels", 2),
        heads=getattr(cfg, "heads", 2),
        m=getattr(cfg, "m", 16),
        beta=getattr(cfg, "beta", 0.1),
        lr=getattr(cfg, "lr", 2e-4),
        ff_dim=getattr(cfg, "ff_dim", 512),
        transformer_heads=getattr(cfg, "transformer_heads", 8),
        transformer_layers=getattr(cfg, "transformer_layers", 4),
        transformer_dropout=getattr(cfg, "transformer_dropout", 0.15),
        warmup_beta=getattr(cfg, "warmup_beta", True),
        max_beta=getattr(cfg, "max_beta", 0.05),
        beta_warmup_steps=getattr(cfg, "beta_warmup_steps", 8000),
        free_bits=getattr(cfg, "free_bits", 0.03),
        use_kl_capacity=getattr(cfg, "use_kl_capacity", True),
        capacity_per_dim_end=getattr(cfg, "capacity_per_dim_end", 0.03),
        capacity_warmup_steps=getattr(cfg, "capacity_warmup_steps", 20000),
        stale_dropout_p=getattr(cfg, "stale_dropout_p", 0.2),
        set_mae_ratio=getattr(cfg, "set_mae_ratio", 0.0),
        enable_next_change=True,
        next_change_weight=0.3,
        use_adaptive_poe=True,
        poe_beta_min=0.1,
        poe_beta_max=3.0,
        freeze_set_encoder=True,
        poe_mode="conditional",
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded PoE weights: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _read_label_map(label_csv: str) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    with open(label_csv, "r") as f:
        reader = csv.DictReader(f)
        assert (
            "ts_id" in reader.fieldnames
            and "in_hospital_mortality" in reader.fieldnames
        ), "label_csv must contain ts_id,in_hospital_mortality columns"
        for row in reader:
            pid_raw = row["ts_id"]
            y_raw = str(row["in_hospital_mortality"]).strip()
            if y_raw == "" or y_raw.lower() == "nan":
                lab = 0
            else:
                try:
                    lab = 1 if int(float(y_raw)) == 1 else 0
                except Exception:
                    lab = 0
            for pid in _pid_variants(pid_raw):
                if pid == "":
                    continue
                mp[pid] = lab
    return mp


class MortalityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        saved_dir: str,
        label_csv: str,
        batch_size: int = 4,
        num_workers: int = 2,
        pin_memory: bool = True,
        smoke: bool = False,
        smoke_batch_size: int = 8,
    ):
        super().__init__()
        self.saved_dir = saved_dir
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smoke = smoke
        self.smoke_batch_size = smoke_batch_size
        self.id_to_label: Dict[str, int] = {}
        self.vcols: Optional[List[str]] = None

    def setup(self, stage=None):
        self.id_to_label = _read_label_map(self.label_csv)
        self.train = PatientDataset("train", self.saved_dir)
        self.valid = PatientDataset("valid", self.saved_dir)
        self.test = PatientDataset("test", self.saved_dir)
        # Detect embedding columns
        sample_df, _ = self.train[0]
        self.vcols = _detect_vcols(sample_df)
        # Compute pos_weight from train labels
        tr_labels = [self.id_to_label.get(pid, 0) for pid in self.train.ids]
        n_pos = max(1, sum(tr_labels))
        n_neg = max(1, len(tr_labels) - n_pos)
        self.pos_weight = float(n_neg) / float(n_pos)
        # Smoke subset for quick test
        if self.smoke:
            n = len(self.train)
            k = min(self.smoke_batch_size, n)
            self._smoke_idx = list(range(k))
        else:
            self._smoke_idx = None

    def _collate_with_label(self, batch: List[Tuple[Any, str]]):
        assert self.vcols is not None
        out = _collate_lvcf(batch, self.vcols)
        labels_list: List[float] = []
        missing: List[str] = []
        for (_, pid) in batch:
            lab = self.id_to_label.get(pid, None)
            if lab is None:
                for v in _pid_variants(pid):
                    if v in self.id_to_label:
                        lab = self.id_to_label[v]
                        break
            if lab is None:
                missing.append(str(pid))
            else:
                labels_list.append(float(lab))
        if len(missing) > 0:
            raise ValueError(f"Missing labels for IDs (sample): {missing[:10]} (total {len(missing)})")
        out["y"] = torch.tensor(labels_list, dtype=torch.float32)
        return out

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_with_label,
        )

    def train_dataloader(self):
        if getattr(self, "_smoke_idx", None) is not None:
            subset = Subset(self.train, self._smoke_idx)
            return DataLoader(
                subset,
                batch_size=len(subset),
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate_with_label,
            )
        return self._loader(self.train, True)

    def val_dataloader(self):
        return self._loader(self.valid, False)

    def test_dataloader(self):
        return self._loader(self.test, False)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B,S,H], mask: [B,S] (True for valid)
        score = self.fc(h).squeeze(-1)  # [B,S]
        score = score.masked_fill(~mask, float("-inf"))
        att = torch.softmax(score, dim=-1)  # [B,S]
        att = att.unsqueeze(-1)  # [B,S,1]
        z = torch.sum(att * h, dim=1)  # [B,H]
        return z


class MortalityClassifier(pl.LightningModule):
    def __init__(
        self,
        poe_model: PoESeqSetVAEPretrain,
        latent_dim: int,
        mu_proj_dim: int = 64,
        logvar_proj_dim: int = 32,
        scalar_proj_dim: int = 16,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["poe_model"]
        )  # avoid serializing the big backbone
        self.poe = poe_model.eval()
        for p in self.poe.parameters():
            p.requires_grad = False
        self.latent_dim = latent_dim
        self.mu_proj = nn.Sequential(nn.Linear(latent_dim, mu_proj_dim), nn.GELU())
        self.logvar_proj = nn.Sequential(
            nn.Linear(latent_dim, logvar_proj_dim), nn.GELU()
        )
        # Divergence/structure/time scalars: [d2, kl, abs_delta_H, beta, log_set_size, has_change, log_dt1p] -> 7
        self.scalar_in_dim = 7
        self.scalar_proj = nn.Sequential(
            nn.Linear(self.scalar_in_dim, 32), nn.GELU(), nn.Linear(32, scalar_proj_dim)
        )
        feat_dim = mu_proj_dim + logvar_proj_dim + scalar_proj_dim
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.attn = AdditiveAttention(gru_hidden)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1),
        )
        self.lr = lr
        self.pos_weight = (
            None
            if pos_weight is None
            else torch.tensor([pos_weight], dtype=torch.float32)
        )
        # Buffers for epoch metrics
        self._val_logits: List[float] = []
        self._val_labels: List[int] = []
        self._test_logits: List[float] = []
        self._test_labels: List[int] = []

    @torch.no_grad()
    def _extract_features_single(
        self, sets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From a list of per-set dicts (as in PoE._split_sets), compute per-step features.
        Returns: (X, mask) where X: [S, F], mask: [S] (all True here; caller pads later)
        """
        device = next(self.parameters()).device
        S = len(sets)
        if S == 0:
            return torch.zeros(0, 1, device=device), torch.zeros(
                0, dtype=torch.bool, device=device
            )

        # Encode q_x(z|x) per set
        mu_qx_list: List[torch.Tensor] = []
        logvar_qx_list: List[torch.Tensor] = []
        enc_aggs: List[torch.Tensor] = []
        set_sizes: List[int] = []
        has_changes: List[float] = []
        set_times: List[torch.Tensor] = []
        for s in sets:
            var, val = s["var"], s["val"]  # [1,N,D], [1,N,1]
            z_list_enc, enc_tokens = self.poe.set_encoder.encode_from_var_val(var, val)
            _z, mu_qx, logvar_qx = z_list_enc[-1]
            mu_qx_list.append(mu_qx.squeeze(1))  # [1,D] → [D]
            logvar_qx_list.append(logvar_qx.squeeze(1))  # [1,D] → [D]
            if enc_tokens is not None and enc_tokens.numel() > 0:
                enc_aggs.append(enc_tokens.mean(dim=1))  # [1,N,D] → [1,D]
            else:
                enc_aggs.append(mu_qx)
            set_sizes.append(int(var.size(1)))
            has_changes.append(float(s.get("has_change", torch.tensor(0.0)).item()))
            set_times.append(s["set_time"])  # [1]

        mu_qx = torch.stack(mu_qx_list, dim=1).to(device)  # [1,S,D]
        logvar_qx = torch.stack(logvar_qx_list, dim=1).to(device)  # [1,S,D]
        set_aggs = torch.stack(enc_aggs, dim=1).to(device)  # [1,S,D]
        minutes = (
            torch.stack(set_times, dim=1).float().to(device)
        )  # [1,S,1]? set_time is [1]; keep [1,S,1]
        if minutes.dim() == 2:
            minutes = minutes.unsqueeze(-1)
        minutes = minutes.squeeze(-1)  # [1,S]

        # Time embedding and GRU rollout of prior
        time_emb = self.poe._relative_time_bucket_embedding(minutes)  # [1,S,D]
        B = 1
        h = torch.zeros(B, self.latent_dim, device=device)

        feat_rows: List[torch.Tensor] = []
        for t in range(S):
            # Prior from previous hidden
            prior_params = self.poe.prior_head(h)  # [1,2D]
            mu_p_t, logvar_p_t = prior_params.chunk(2, dim=-1)  # [1,D], [1,D]

            # Gating beta_t from divergence + time delta
            mu_qx_t = mu_qx[:, t, :]
            logvar_qx_t = logvar_qx[:, t, :]
            var_sum = (logvar_p_t.exp() + logvar_qx_t.exp()).clamp(min=1e-8)
            d2 = ((mu_qx_t - mu_p_t) ** 2 / var_sum).mean(dim=-1, keepdim=True)  # [1,1]
            delta_H = (logvar_p_t - logvar_qx_t).mean(dim=-1, keepdim=True)  # [1,1]
            dt = minutes[:, t : t + 1] - (
                minutes[:, t - 1 : t] if t > 0 else minutes[:, t : t + 1]
            )
            log_dt1p = torch.log1p(dt.clamp(min=0.0))  # [1,1]
            gate_inp = torch.cat([d2, delta_H, log_dt1p], dim=-1)
            beta_t = self.poe.poe_beta_min + (
                self.poe.poe_beta_max - self.poe.poe_beta_min
            ) * torch.sigmoid(
                self.poe.obs_gate(gate_inp)
            )  # [1,1]

            # Conditional PoE: likelihood natural params from set agg and prior
            set_agg_t = set_aggs[:, t, :]
            like_inp = torch.cat([set_agg_t, mu_p_t, logvar_p_t], dim=-1)
            like_out = self.poe.like_head(like_inp)
            log_prec_like_t, h_like_t = like_out.chunk(2, dim=-1)
            prec_like_t = F.softplus(log_prec_like_t) + 1e-4  # [1,D]
            Lambda_p_t = torch.exp(-logvar_p_t)
            prec_like_t = prec_like_t * beta_t
            h_like_t = h_like_t * beta_t
            Lambda_post_t = Lambda_p_t + prec_like_t
            h_post_t = Lambda_p_t * mu_p_t + h_like_t
            mu_post_t = h_post_t / (Lambda_post_t + 1e-8)
            logvar_post_t = -torch.log(Lambda_post_t + 1e-8)

            # KL(q_post||p_prior) per step (sum over dims)
            var_q = logvar_post_t.exp()
            var_p = logvar_p_t.exp()
            kld = 0.5 * (
                logvar_p_t
                - logvar_post_t
                + (var_q / (var_p + 1e-8))
                + ((mu_p_t - mu_post_t) ** 2) / (var_p + 1e-8)
                - 1.0
            )
            kl_scalar = kld.sum(dim=-1, keepdim=True)  # [1,1]

            # Build step feature
            mu_feat = self.mu_proj(mu_post_t)  # [1,Fm]
            logv_feat = self.logvar_proj(logvar_post_t)  # [1,Fv]
            scalar_feat = torch.cat(
                [
                    d2,
                    torch.abs(delta_H),
                    kl_scalar,
                    beta_t,
                    torch.log1p(torch.tensor([[float(set_sizes[t])]], device=device)),
                    torch.tensor([[has_changes[t]]], device=device),
                    log_dt1p,
                ],
                dim=-1,
            )  # [1,7]
            scalar_feat = self.scalar_proj(scalar_feat)  # [1,Fs]
            feat_t = torch.cat([mu_feat, logv_feat, scalar_feat], dim=-1).squeeze(
                0
            )  # [F]
            feat_rows.append(feat_t)

            # Update GRU hidden with posterior mean + time embedding (as in backbone)
            h = self.poe.gru_cell(mu_post_t + time_emb[:, t, :], h)

        X = torch.stack(feat_rows, dim=0)  # [S,F]
        mask = torch.ones(S, dtype=torch.bool, device=device)
        return X, mask

    @torch.no_grad()
    def _build_batch_features(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a collated batch from MortalityDataModule, produce padded features.
        Returns: (X_pad, mask, y)
          - X_pad: [B, S_max, F]
          - mask: [B, S_max] (True for valid steps)
          - y: [B]
        """
        var, val = batch["var"], batch["val"]
        minute, set_id = batch["minute"], batch["set_id"]
        padding_mask = batch.get("padding_mask", None)
        carry_mask = batch.get("carry_mask", None)
        B = var.size(0)
        # Split into per-patient sets using the backbone's utility
        all_sets = self.poe._split_sets(
            var, val, minute, set_id, padding_mask, carry_mask
        )
        feats: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for sets in all_sets:
            X_i, m_i = self._extract_features_single(sets)
            feats.append(X_i)
            masks.append(m_i)
        maxS = max((x.shape[0] for x in feats), default=0)
        feat_dim = feats[0].shape[1] if len(feats) > 0 and feats[0].ndim == 2 and feats[0].shape[0] > 0 else (self.mu_proj[0].out_features + self.logvar_proj[0].out_features + self.scalar_proj[-1].out_features)  # type: ignore
        X_pad = torch.zeros(B, maxS, feat_dim, device=var.device)
        M = torch.zeros(B, maxS, dtype=torch.bool, device=var.device)
        for b in range(B):
            s_len = feats[b].shape[0]
            if s_len > 0:
                X_pad[b, :s_len, :] = feats[b]
                M[b, :s_len] = masks[b]
        y = batch["y"].to(var.device)
        return X_pad, M, y

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        X, M, _ = self._build_batch_features(batch)
        lengths = M.sum(dim=1).long().clamp(min=1)
        packed = pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        H, _ = pad_packed_sequence(packed_out, batch_first=True)
        # Recreate mask size [B, S_out]
        S_out = H.size(1)
        M_out = torch.arange(S_out, device=H.device).unsqueeze(0) < lengths.unsqueeze(1)
        z = self.attn(H, M_out)
        logit = self.head(z).squeeze(-1)  # [B]
        return logit

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        logits = self.forward(batch)
        y = batch["y"].to(logits.dtype)
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.pos_weight.to(logits.device)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        logits = self.forward(batch)
        y = batch["y"].to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        # Accumulate for epoch metrics
        self._val_logits.extend(logits.detach().cpu().tolist())
        self._val_labels.extend(y.detach().cpu().tolist())
        return loss

    def on_validation_epoch_end(self) -> None:
        if len(self._val_labels) == 0:
            return
        try:
            y_true = np.array(self._val_labels, dtype=np.int32)
            y_prob = torch.sigmoid(torch.tensor(self._val_logits)).numpy()
            auroc = (
                float(roc_auc_score(y_true, y_prob))
                if roc_auc_score is not None
                else float("nan")
            )
            auprc = (
                float(average_precision_score(y_true, y_prob))
                if average_precision_score is not None
                else float("nan")
            )
        except Exception:
            auroc, auprc = float("nan"), float("nan")
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_auprc", auprc, prog_bar=True)
        self._val_logits.clear()
        self._val_labels.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        logits = self.forward(batch)
        y = batch["y"].to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self._test_logits.extend(logits.detach().cpu().tolist())
        self._test_labels.extend(y.detach().cpu().tolist())
        return loss

    def on_test_epoch_end(self) -> None:
        if len(self._test_labels) == 0:
            return
        try:
            y_true = np.array(self._test_labels, dtype=np.int32)
            y_prob = torch.sigmoid(torch.tensor(self._test_logits)).numpy()
            auroc = (
                float(roc_auc_score(y_true, y_prob))
                if roc_auc_score is not None
                else float("nan")
            )
            auprc = (
                float(average_precision_score(y_true, y_prob))
                if average_precision_score is not None
                else float("nan")
            )
        except Exception:
            auroc, auprc = float("nan"), float("nan")
        self.log("test_auroc", auroc, prog_bar=True)
        self.log("test_auprc", auprc, prog_bar=True)
        self._test_logits.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        return opt


def main():
    ap = argparse.ArgumentParser(
        description="Train mortality classifier on PoE features"
    )
    ap.add_argument(
        "--checkpoint", required=True, type=str, help="Path to PoE checkpoint (.ckpt)"
    )
    ap.add_argument(
        "--label_csv",
        required=True,
        type=str,
        help="CSV with columns: ts_id,in_hospital_mortality (0/1)",
    )
    ap.add_argument("--data_dir", type=str, default=getattr(cfg, "data_dir", ""))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 2))
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/home/sunx/data/aiiih/projects/sunx/projects/SSV/output",
        help="Root out_dir (saves under out_dir/mortality/version_X)",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="Quick smoke run on a few patients"
    )
    args = ap.parse_args()

    # Load PoE backbone (frozen)
    state = _load_state_dict(args.checkpoint)
    poe = _build_poe_from_state(state)

    # Data
    dm = MortalityDataModule(
        saved_dir=args.data_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        smoke=args.smoke,
        smoke_batch_size=max(2, args.batch_size),
    )
    dm.setup()

    # Model
    model = MortalityClassifier(
        poe_model=poe,
        latent_dim=getattr(cfg, "latent_dim", 128),
        mu_proj_dim=64,
        logvar_proj_dim=32,
        scalar_proj_dim=16,
        gru_hidden=128,
        gru_layers=2,
        dropout=args.dropout,
        lr=args.lr,
        pos_weight=getattr(dm, "pos_weight", None),
    )

    # Logging/checkpoints under a consistent layout
    out_root = args.output_dir if args.output_dir else "./output"
    project_dir = os.path.join(out_root, "classifier")
    os.makedirs(project_dir, exist_ok=True)
    try:
        logger = TensorBoardLogger(save_dir=project_dir, name="", sub_dir="logs")
    except TypeError:
        logger = TensorBoardLogger(save_dir=project_dir, name="")
    log_dir = getattr(logger, "log_dir", project_dir)
    if log_dir.endswith(os.sep + "logs") or log_dir.endswith("/logs"):
        version_dir = os.path.dirname(log_dir)
    else:
        version_dir = log_dir
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=1,
            monitor="val_auprc",
            mode="max",
            filename="mortality_cls",
        ),
        EarlyStopping(monitor="val_auprc", mode="max", patience=5),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.5,
    )
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    print("\nEvaluating best checkpoint on test split...")
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()
