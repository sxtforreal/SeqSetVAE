#!/usr/bin/env python3
import os
import argparse
from datetime import datetime
from typing import List

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


class ClassifierHeadFinetuner(SeqSetVAE):
    """Fine-tune only the classification head of SeqSetVAE.

    - Freezes encoder (SetVAE), transformer, decoder, and pooling/projection modules
    - Optimizes only `cls_head` parameters
    - Uses focal loss with alpha derived from class imbalance
    """

    def __init__(self, *args, head_lr: float = 1e-3, class_weights=None, focal_alpha: float | None = None, focal_gamma: float = 2.5, **kwargs):
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

    def training_step(self, batch, batch_idx):
        # Only classification loss for head fine-tuning
        logits, _, _ = self.forward(batch)
        labels = batch["label"]

        if getattr(config, 'use_focal_loss', True):
            # Focal loss with alpha derived from imbalance
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
        logits, _, _ = self.forward(batch)
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
        optimizer = AdamW(self.cls_head.parameters(), lr=self.head_lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=5, verbose=True, min_lr=self.head_lr * 1e-3)
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

    # Build weighted sampler for training (class imbalance handling)
    sampler, pos_count, neg_count = build_weighted_sampler(data_module.train_dataset, data_module.label_map)
    total = pos_count + neg_count
    pos_ratio = pos_count / total
    print(f"Class distribution (train): pos={pos_count}, neg={neg_count} (pos_ratio={pos_ratio:.4f})")

    # Collate fn selection
    if args.batch_size == 1:
        collate_fn = data_module._collate_fn
    else:
        collate_fn = lambda batch: data_module._dynamic_collate_fn(batch)  # noqa: E731

    # Custom loaders: use sampler for training, standard for val/test
    train_loader = DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
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
        ff_dim=getattr(config, 'ff_dim', config.enhanced_ff_dim if hasattr(config, 'enhanced_ff_dim') else 1024),
        transformer_heads=getattr(config, 'transformer_heads', config.enhanced_transformer_heads if hasattr(config, 'enhanced_transformer_heads') else 8),
        transformer_layers=getattr(config, 'transformer_layers', config.enhanced_transformer_layers if hasattr(config, 'enhanced_transformer_layers') else 4),
        pretrained_ckpt=config.pretrained_ckpt,
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

    # Logging and callbacks
    checkpoint_name = "SeqSetVAE_cls_head_finetune"
    base_output_dir = args.output_root_dir.rstrip("/")
    checkpoints_root_dir = os.path.join(base_output_dir, "checkpoints", checkpoint_name)
    logs_root_dir = os.path.join(base_output_dir, "logs")
    os.makedirs(checkpoints_root_dir, exist_ok=True)
    os.makedirs(logs_root_dir, exist_ok=True)

    callbacks = []
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoints_root_dir,
        filename="best",
        save_top_k=3,
        monitor="val_auc",
        mode="max",
        save_last=True,
    )
    callbacks.append(ckpt_cb)

    early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=8, min_delta=1e-4, verbose=True)
    callbacks.append(early_stop)

    lr_mon = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_mon)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(save_dir=logs_root_dir, name=checkpoint_name, version=timestamp, log_graph=False)

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
        gradient_clip_val=0.1,
        val_check_interval=0.15,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    print(f"Starting classifier-head finetuning with head_lr={args.head_lr}, focal_alpha={focal_alpha:.3f}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()