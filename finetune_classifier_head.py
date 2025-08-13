#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics.classification import AUROC, AveragePrecision, Accuracy

# Local imports
from model import SeqSetVAE, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
import config


def build_weighted_sampler(train_dataset, label_map):
    """Create a WeightedRandomSampler based on class frequencies in training set.

    Args:
        train_dataset (SeqSetVAEDataset): training dataset (provides patient_ids)
        label_map (Dict[int,int]): mapping from patient id to label
    Returns:
        sampler, pos_count, neg_count
    """
    # Collect labels in train split order
    labels: List[int] = []
    for pid_str in train_dataset.patient_ids:
        try:
            pid = int(float(pid_str))
        except Exception:
            # Fallback if patient id is already int-like
            pid = int(pid_str)
        labels.append(int(label_map.get(pid, 0)))

    # Count classes
    pos_count = sum(1 for y in labels if y == 1)
    neg_count = len(labels) - pos_count
    pos_count = max(1, pos_count)
    neg_count = max(1, neg_count)

    # Inverse-frequency weights (classic imbalance handling)
    weight_pos = 1.0 / pos_count
    weight_neg = 1.0 / neg_count
    weights = torch.tensor([weight_pos if y == 1 else weight_neg for y in labels], dtype=torch.float)

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)
    return sampler, pos_count, neg_count


def compute_class_counts(train_dataset, label_map):
    """Compute positive and negative counts without building a sampler."""
    labels: List[int] = []
    for pid_str in train_dataset.patient_ids:
        try:
            pid = int(float(pid_str))
        except Exception:
            pid = int(pid_str)
        labels.append(int(label_map.get(pid, 0)))
    pos_count = sum(1 for y in labels if y == 1)
    neg_count = len(labels) - pos_count
    return max(1, pos_count), max(1, neg_count)


class ClassifierHeadFinetuner(SeqSetVAE):
    """Fine-tune only the classification head of SeqSetVAE.

    - Freezes encoder (SetVAE), transformer, decoder, and pooling/projection modules
    - Optimizes only `cls_head` parameters
    - Uses focal loss with alpha derived from class imbalance
    """

    def __init__(self, *args, head_lr: float = 1e-3, class_weights=None, focal_alpha: Optional[float] = None, focal_gamma: float = 2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lr = head_lr
        self.class_weights = class_weights  # Tensor of size [C] for CE if used
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Metrics
        task_type = "binary" if config.num_classes == 2 else "multiclass"
        self.val_auc = AUROC(task=task_type, num_classes=config.num_classes)
        self.val_auprc = AveragePrecision(task=task_type, num_classes=config.num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=config.num_classes)

        # Freeze everything except cls_head immediately
        self._freeze_backbone()

    def _freeze_backbone(self):
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
        # Unfreeze only classification head
        for p in self.cls_head.parameters():
            p.requires_grad = True
        # Put frozen modules in eval mode to disable dropout etc.
        if hasattr(self, 'setvae'):
            self.setvae.eval()
        if hasattr(self, 'transformer'):
            self.transformer.eval()
        if hasattr(self, 'decoder'):
            self.decoder.eval()
        if hasattr(self, 'feature_fusion'):
            self.feature_fusion.eval()
        if hasattr(self, 'feature_projection'):
            self.feature_projection.eval()
        if hasattr(self, 'post_transformer_norm'):
            self.post_transformer_norm.eval()

    def on_train_start(self):
        # Ensure modules stay frozen at the start of training
        self._freeze_backbone()

    # Lightweight classification-only forward: skip reconstruction and KL; run frozen backbone in inference_mode
    def forward(self, batch):
        var, val, time, set_ids = (
            batch["var"],
            batch["val"],
            batch["minute"],
            batch["set_id"],
        )
        padding_mask = batch.get("padding_mask", None)

        # Split into per-patient sets using parent's utility
        all_patient_sets = self._split_sets(var, val, time, set_ids, padding_mask)

        all_logits = []
        valid_patients = 0

        for patient_sets in all_patient_sets:
            if len(patient_sets) == 0:
                continue

            S = len(patient_sets)
            z_prims = []
            pos_list = []

            # Compute frozen backbone features without building autograd graph
            with torch.inference_mode():
                for s_dict in patient_sets:
                    var_t = s_dict["var"]
                    val_t = s_dict["val"]
                    time_t = s_dict["minute"]

                    assert time_t.unique().numel() == 1, "Time is not constant in this set"
                    minute_val = time_t.unique().float()
                    pos_list.append(minute_val)

                    if self.setvae.setvae.dim_reducer is not None:
                        reduced = self.setvae.setvae.dim_reducer(var_t)
                    else:
                        reduced = var_t

                    norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
                    reduced_normalized = reduced / (norms + 1e-8)
                    x = reduced_normalized * val_t

                    z_list, _ = self.setvae.setvae.encode(x)
                    z_sample, mu, logvar = z_list[-1]
                    z_prims.append(z_sample.squeeze(1))

                z_seq = torch.stack(z_prims, dim=1)  # [B, S, latent]
                pos_tensor = torch.stack(pos_list, dim=1)  # [B, S]

                # Positional/time encoding and transformer (frozen)
                z_seq = self._apply_positional_encoding(z_seq, pos_tensor, None)
                z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])

                attn_mask = self._create_causal_mask(S, z_seq.device, None)
                h_seq = self.transformer(z_seq, mask=attn_mask[0] if attn_mask.dim() == 3 else attn_mask)
                h_seq = self.post_transformer_norm(h_seq)

                enhanced_features = self._extract_enhanced_features(h_seq)
                if enhanced_features is None:
                    attn_weights = F.softmax(torch.sum(h_seq * z_seq, dim=-1), dim=1)
                    enhanced_features = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)

            # Classification head computes with gradients
            logits = self.cls_head(enhanced_features)
            all_logits.append(logits)
            valid_patients += 1

        if valid_patients == 0:
            batch_size = var.size(0)
            device = var.device
            return torch.zeros(batch_size, self.num_classes, device=device)

        return torch.cat(all_logits, dim=0)

    def training_step(self, batch, batch_idx):
        # Classification-only forward for head fine-tuning
        logits = self.forward(batch)
        labels = batch["label"]

        if getattr(config, 'use_focal_loss', True):
            from losses import FocalLoss
            alpha = self.focal_alpha if self.focal_alpha is not None else getattr(config, 'focal_alpha', 0.35)
            focal = FocalLoss(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
            loss = focal(logits, labels)
        else:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=weight, label_smoothing=0.0)

        self.log("train_class_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        labels = batch["label"]

        if getattr(config, 'use_focal_loss', True):
            from losses import FocalLoss
            alpha = self.focal_alpha if self.focal_alpha is not None else getattr(config, 'focal_alpha', 0.35)
            focal = FocalLoss(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
            loss = focal(logits, labels)
        else:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=weight, label_smoothing=0.0)

        # Metrics
        if config.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.softmax(logits, dim=1)
        self.val_auc.update(probs, labels)
        self.val_auprc.update(probs, labels)
        preds = logits.argmax(dim=1)
        self.val_acc.update(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        auprc = self.val_auprc.compute()
        acc = self.val_acc.compute()
        self.log_dict({"val_auc": auc, "val_auprc": auprc, "val_accuracy": acc}, prog_bar=True)
        self.val_auc.reset(); self.val_auprc.reset(); self.val_acc.reset()

    def configure_optimizers(self):
        # Use optimized parameters for classification head fine-tuning
        weight_decay = getattr(config, 'head_weight_decay', 0.001)
        betas = getattr(config, 'head_betas', (0.9, 0.999))
        eps = getattr(config, 'head_eps', 1e-8)
        
        optimizer = AdamW(
            self.cls_head.parameters(), 
            lr=self.head_lr, 
            weight_decay=weight_decay, 
            betas=betas, 
            eps=eps
        )
        
        # More responsive learning rate scheduler
        scheduler_factor = getattr(config, 'scheduler_factor', 0.7)
        scheduler_patience = getattr(config, 'scheduler_patience', 3)
        min_lr_factor = getattr(config, 'scheduler_min_lr_factor', 1e-3)
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=scheduler_factor, 
            patience=scheduler_patience, 
            verbose=True, 
            min_lr=self.head_lr * min_lr_factor
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"},
        }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune classification head only for SeqSetVAE")

    # Data
    parser.add_argument("--data_dir", type=str, default=config.saved_dir if hasattr(config, 'saved_dir') else "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr")
    parser.add_argument("--params_map_path", type=str, default=config.params_map_path if hasattr(config, 'params_map_path') else "/home/sunx/data/aiiih/data/mimic/processed/stats.csv")
    parser.add_argument("--label_path", type=str, default=config.label_path if hasattr(config, 'label_path') else "/home/sunx/data/aiiih/data/mimic/processed/oc.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model / training
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to initialize the whole SeqSetVAE (Lightning ckpt or state_dict)")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate for classification head")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=None)

    # Output
    parser.add_argument("--output_root_dir", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs")
    parser.add_argument("--output_dir", type=str, default=None, help="Alias of --output_root_dir; if provided, overrides output_root_dir")
    
    args = parser.parse_args()
    
    # Data module setup
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        use_dynamic_padding=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_module.setup()

    # Report class distribution (still used to derive focal alpha)
    pos_count, neg_count = compute_class_counts(data_module.train_dataset, data_module.label_map)
    total = pos_count + neg_count
    pos_ratio = pos_count / total
    print(f"Class distribution (train): pos={pos_count}, neg={neg_count} (pos_ratio={pos_ratio:.4f})")

    # Collate fn selection
    if args.batch_size == 1:
        collate_fn = data_module._collate_fn
    else:
        collate_fn = lambda batch: data_module._dynamic_collate_fn(batch)  # noqa: E731

    # Training loader: simplified to standard random shuffling
    train_loader = DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = data_module.val_dataloader()

    # Derive focal alpha from imbalance (clamped to [0.25, 0.7])
    focal_alpha = float(min(0.7, max(0.25, pos_ratio)))

    # Prepare model
    model = ClassifierHeadFinetuner(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        num_classes=config.num_classes,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        pretrained_ckpt=None,
        skip_pretrained_on_resume=True,
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
        use_focal_loss=True,
        focal_alpha=focal_alpha,
        focal_gamma=getattr(config, 'focal_gamma', 2.5),
        head_lr=args.head_lr,
    )

    # Optionally load a full SeqSetVAE checkpoint (weights only)
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading SeqSetVAE weights from checkpoint: {args.checkpoint}")
        state_dict = load_checkpoint_weights(args.checkpoint, device='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if isinstance(missing, list) and len(missing) > 0:
            print(f"Missing keys when loading: {len(missing)} (ok if only classifier differs)")
        if isinstance(unexpected, list) and len(unexpected) > 0:
            print(f"Unexpected keys when loading: {len(unexpected)} (ignored)")

    # Logging and unified directories: output_dir/config.model_name/{checkpoints,logs,analysis}
    checkpoint_name = "SeqSetVAE_cls_head_finetune"
    base_output_dir = (args.output_dir or args.output_root_dir).rstrip("/")
    model_name = getattr(config, 'model_name', getattr(config, 'name', 'SeqSetVAE-v3'))
    experiment_root = os.path.join(base_output_dir, model_name)
    checkpoints_root_dir = os.path.join(experiment_root, 'checkpoints')
    logs_root_dir = os.path.join(experiment_root, 'logs')
    analysis_root_dir = os.path.join(experiment_root, 'analysis')
    os.makedirs(os.path.join(checkpoints_root_dir, checkpoint_name), exist_ok=True)
    os.makedirs(os.path.join(logs_root_dir, checkpoint_name), exist_ok=True)
    os.makedirs(os.path.join(analysis_root_dir, checkpoint_name), exist_ok=True)

    callbacks = []
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(checkpoints_root_dir, checkpoint_name),
        filename="best",
        save_top_k=3,
        monitor="val_auc",
        mode="max",
        save_last=True,
    )
    callbacks.append(ckpt_cb)

    early_stop = EarlyStopping(
        monitor="val_auc", 
        mode="max", 
        patience=getattr(config, 'early_stopping_patience', 10), 
        min_delta=getattr(config, 'early_stopping_min_delta', 1e-4), 
        verbose=True
    )
    callbacks.append(early_stop)

    lr_mon = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_mon)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(save_dir=logs_root_dir, name="", version=timestamp, log_graph=False)

    # Devices autodetect
    devices = args.devices
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=getattr(config, 'gradient_clip_val', 0.05),  # Reduced for classification head
        val_check_interval=getattr(config, 'val_check_interval', 0.25),
        log_every_n_steps=getattr(config, 'log_every_n_steps', 25),
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    print(f"Starting classifier-head finetuning with head_lr={args.head_lr}, focal_alpha={focal_alpha:.3f}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()