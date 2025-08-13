#!/usr/bin/env python3
"""
Optimized classifier head fine-tuning with improved architecture
"""

import os
import argparse
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics.classification import AUROC, AveragePrecision, Accuracy

# Local imports
from model import SeqSetVAE, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
from model_optimized_classifier import create_classifier_head, SimpleFeatureExtractor
import config


def build_weighted_sampler(train_dataset, label_map):
    """Create a WeightedRandomSampler based on class frequencies in training set."""
    labels: List[int] = []
    for pid_str in train_dataset.patient_ids:
        try:
            pid = int(float(pid_str))
        except Exception:
            pid = int(pid_str)
        labels.append(int(label_map.get(pid, 0)))

    pos_count = sum(1 for y in labels if y == 1)
    neg_count = len(labels) - pos_count
    pos_count = max(1, pos_count)
    neg_count = max(1, neg_count)

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


class OptimizedClassifierHeadFinetuner(SeqSetVAE):
    """
    Optimized fine-tuning of classification head with improved architecture
    """

    def __init__(self, *args, head_lr: float = 5e-4, class_weights=None, 
                 focal_alpha: Optional[float] = None, focal_gamma: float = 2.5,
                 classifier_type: str = "lightweight", **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lr = head_lr
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.classifier_type = classifier_type

        # Replace the original classification head with optimized version
        self._replace_classifier_head()
        
        # Replace feature extraction with simplified version
        self._replace_feature_extractor()

        # Metrics
        task_type = "binary" if config.num_classes == 2 else "multiclass"
        self.val_auc = AUROC(task=task_type, num_classes=config.num_classes)
        self.val_auprc = AveragePrecision(task=task_type, num_classes=config.num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=config.num_classes)

        # Freeze everything except the new classifier head
        self._freeze_backbone()

    def _replace_classifier_head(self):
        """Replace the original classifier head with optimized version"""
        # Remove the original classifier head
        if hasattr(self, 'cls_head'):
            del self.cls_head
        
        # Create optimized classifier head
        self.cls_head = create_classifier_head(
            self.classifier_type, 
            self.latent_dim, 
            self.num_classes
        )
        
        print(f"Replaced classifier head with {self.classifier_type} architecture")
        print(f"Classifier parameters: {sum(p.numel() for p in self.cls_head.parameters()):,}")

    def _replace_feature_extractor(self):
        """Replace complex feature fusion with simplified extractor"""
        # Remove complex feature fusion modules
        if hasattr(self, 'feature_fusion'):
            del self.feature_fusion
        if hasattr(self, 'feature_projection'):
            del self.feature_projection
        
        # Create simplified feature extractor
        self.simple_feature_extractor = SimpleFeatureExtractor(self.latent_dim)
        
        print("Replaced complex feature fusion with simplified attention pooling")

    def _freeze_backbone(self):
        """Freeze all parameters except the new classifier head"""
        for p in self.parameters():
            p.requires_grad = False
        
        # Unfreeze only the new classifier head and feature extractor
        for p in self.cls_head.parameters():
            p.requires_grad = True
        for p in self.simple_feature_extractor.parameters():
            p.requires_grad = True
        
        # Put frozen modules in eval mode
        if hasattr(self, 'setvae'):
            self.setvae.eval()
        if hasattr(self, 'transformer'):
            self.transformer.eval()
        if hasattr(self, 'decoder'):
            self.decoder.eval()
        if hasattr(self, 'post_transformer_norm'):
            self.post_transformer_norm.eval()

    def on_train_start(self):
        """Ensure modules stay frozen at the start of training"""
        self._freeze_backbone()

    def forward(self, batch):
        """Optimized forward pass with simplified feature extraction"""
        var, val, time, set_ids = (
            batch["var"],
            batch["val"],
            batch["minute"],
            batch["set_id"],
        )
        padding_mask = batch.get("padding_mask", None)

        # Split into per-patient sets
        all_patient_sets = self._split_sets(var, val, time, set_ids, padding_mask)

        all_logits = []
        valid_patients = 0

        for patient_sets in all_patient_sets:
            if len(patient_sets) == 0:
                continue

            S = len(patient_sets)
            z_prims = []
            pos_list = []

            # Compute frozen backbone features
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

                # Positional encoding and transformer (frozen)
                z_seq = self._apply_positional_encoding(z_seq, pos_tensor, None)
                z_seq = F.layer_norm(z_seq, [z_seq.size(-1)])

                attn_mask = self._create_causal_mask(S, z_seq.device, None)
                h_seq = self.transformer(z_seq, mask=attn_mask[0] if attn_mask.dim() == 3 else attn_mask)
                h_seq = self.post_transformer_norm(h_seq)

                # Use simplified feature extraction
                enhanced_features = self.simple_feature_extractor(h_seq)

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
        """Training step with optimized loss computation"""
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
        """Validation step"""
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
        """Compute validation metrics at epoch end"""
        auc = self.val_auc.compute()
        auprc = self.val_auprc.compute()
        acc = self.val_acc.compute()
        self.log_dict({"val_auc": auc, "val_auprc": auprc, "val_accuracy": acc}, prog_bar=True)
        self.val_auc.reset()
        self.val_auprc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Configure optimizer with improved parameters"""
        # Use optimized parameters for classification head fine-tuning
        weight_decay = getattr(config, 'head_weight_decay', 0.001)
        betas = getattr(config, 'head_betas', (0.9, 0.999))
        eps = getattr(config, 'head_eps', 1e-8)
        
        # Only optimize classifier head and feature extractor
        trainable_params = list(self.cls_head.parameters()) + list(self.simple_feature_extractor.parameters())
        
        optimizer = AdamW(
            trainable_params,
            lr=self.head_lr, 
            weight_decay=weight_decay, 
            betas=betas, 
            eps=eps
        )
        
        # Use cosine annealing scheduler for better convergence
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=getattr(config, 'max_epochs', 50),
            eta_min=self.head_lr * 1e-3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def main():
    """Main function for optimized classifier head fine-tuning"""
    parser = argparse.ArgumentParser(description="Optimized classifier head fine-tuning")

    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr")
    parser.add_argument("--params_map_path", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv")
    parser.add_argument("--label_path", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=None)
    
    # Classifier architecture
    parser.add_argument("--classifier_type", type=str, default="lightweight",
                       choices=["lightweight", "standard", "deep"],
                       help="Type of classifier head architecture")
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_root_dir", type=str, 
                       default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("OPTIMIZED CLASSIFIER HEAD FINE-TUNING")
    print("=" * 60)
    print(f"Classifier type: {args.classifier_type}")
    print(f"Learning rate: {args.head_lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 60)
    
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

    # Report class distribution
    pos_count, neg_count = compute_class_counts(data_module.train_dataset, data_module.label_map)
    total = pos_count + neg_count
    pos_ratio = pos_count / total
    print(f"Class distribution (train): pos={pos_count}, neg={neg_count} (pos_ratio={pos_ratio:.4f})")

    # Data loaders
    train_loader = DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_module._dynamic_collate_fn if args.batch_size > 1 else data_module._collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = data_module.val_dataloader()

    # Derive focal alpha from imbalance
    focal_alpha = float(min(0.7, max(0.25, pos_ratio)))

    # Create optimized model
    model = OptimizedClassifierHeadFinetuner(
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
        classifier_type=args.classifier_type,
    )

    # Load checkpoint if provided
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading SeqSetVAE weights from checkpoint: {args.checkpoint}")
        state_dict = load_checkpoint_weights(args.checkpoint, device='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if isinstance(missing, list) and len(missing) > 0:
            print(f"Missing keys when loading: {len(missing)} (ok if only classifier differs)")
        if isinstance(unexpected, list) and len(unexpected) > 0:
            print(f"Unexpected keys when loading: {len(unexpected)} (ignored)")

    # Setup callbacks and logging
    checkpoint_name = f"SeqSetVAE_optimized_{args.classifier_type}_finetune"
    base_output_dir = args.output_root_dir.rstrip("/")
    model_name = getattr(config, 'model_name', 'SeqSetVAE-v3')
    experiment_root = os.path.join(base_output_dir, model_name)
    checkpoints_root_dir = os.path.join(experiment_root, 'checkpoints')
    logs_root_dir = os.path.join(experiment_root, 'logs')
    
    os.makedirs(os.path.join(checkpoints_root_dir, checkpoint_name), exist_ok=True)
    os.makedirs(os.path.join(logs_root_dir, checkpoint_name), exist_ok=True)

    callbacks = []
    
    # Model checkpointing
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(checkpoints_root_dir, checkpoint_name),
        filename="best",
        save_top_k=3,
        monitor="val_auc",
        mode="max",
        save_last=True,
    )
    callbacks.append(ckpt_cb)

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_auc", 
        mode="max", 
        patience=10, 
        min_delta=1e-4, 
        verbose=True
    )
    callbacks.append(early_stop)

    # Learning rate monitoring
    lr_mon = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_mon)

    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(save_dir=logs_root_dir, name="", version=timestamp, log_graph=False)

    # Trainer setup
    devices = args.devices
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.05,
        val_check_interval=0.25,
        log_every_n_steps=25,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    print(f"Starting optimized classifier head fine-tuning...")
    print(f"Classifier type: {args.classifier_type}")
    print(f"Head learning rate: {args.head_lr}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()