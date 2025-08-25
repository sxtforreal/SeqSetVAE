#!/usr/bin/env python3
"""
Enhanced SeqSetVAE Finetuning Script with Advanced Optimizations
Designed to achieve AUC ~0.9 and AUPRC >0.5 on medical classification tasks
"""

import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything
from datetime import datetime

# Import modules
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorMetricsMonitor
import finetune_config as config


class AUCPRCMonitor(pl.Callback):
    """Enhanced monitoring callback for AUC and AUPRC optimization"""
    
    def __init__(self):
        self.best_auc = 0.0
        self.best_auprc = 0.0
        self.best_combined_score = 0.0
        
    def on_validation_epoch_end(self, trainer, pl_module):
        auc = pl_module.val_auc.compute()
        auprc = pl_module.val_auprc.compute()
        
        # Combined score prioritizing both AUC and AUPRC
        combined_score = 0.6 * auc + 0.4 * auprc  # Weight AUC slightly higher
        
        if combined_score > self.best_combined_score:
            self.best_combined_score = combined_score
            self.best_auc = auc
            self.best_auprc = auprc
            
        pl_module.log("best_auc", self.best_auc, prog_bar=True)
        pl_module.log("best_auprc", self.best_auprc, prog_bar=True)
        pl_module.log("combined_score", combined_score, prog_bar=True)
        
        # Print progress
        print(f"🎯 Current: AUC={auc:.4f}, AUPRC={auprc:.4f}, Combined={combined_score:.4f}")
        print(f"🏆 Best: AUC={self.best_auc:.4f}, AUPRC={self.best_auprc:.4f}, Combined={self.best_combined_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced SeqSetVAE Finetuning")
    
    # Basic parameters
    parser.add_argument("--pretrained_ckpt", type=str, required=True, 
                       help="Path to pretrained checkpoint")
    parser.add_argument("--data_dir", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr",
                       help="Data directory path")
    parser.add_argument("--params_map_path", type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
                       help="Parameter mapping file path")
    parser.add_argument("--label_path", type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv",
                       help="Label file path")
    parser.add_argument("--output_dir", type=str,
                       default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs_enhanced",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed, workers=True)
    
    print("🚀 Enhanced SeqSetVAE Finetuning Configuration:")
    print(f" - Pretrained checkpoint: {args.pretrained_ckpt}")
    print(f" - Max epochs: {config.max_epochs}")
    print(f" - Batch size: {config.batch_size}")
    print(f" - Classification head LR: {config.cls_head_lr}")
    print(f" - Focal loss: α={config.focal_alpha}, γ={config.focal_gamma}")
    print(f" - Early stopping patience: {config.early_stopping_patience}")
    
    # Setup directories
    experiment_root = os.path.join(args.output_dir, "SeqSetVAE_Enhanced")
    checkpoints_dir = os.path.join(experiment_root, 'checkpoints')
    logs_dir = os.path.join(experiment_root, 'logs')
    monitor_dir = os.path.join(experiment_root, 'monitor')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    
    print("📊 Setting up enhanced data module...")
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=config.batch_size,
        max_sequence_length=None,
        use_dynamic_padding=True,
        num_workers=6,
        pin_memory=True,
    )
    
    # Compute adaptive focal loss alpha
    print("🔧 Computing adaptive focal loss parameters...")
    adaptive_alpha = None
    try:
        data_module.setup()
        from collections import Counter
        label_counter = Counter()
        sample_batches = 0
        
        for batch in data_module.train_dataloader():
            labels = batch.get('label')
            if labels is not None:
                label_counter.update(labels.view(-1).tolist())
            sample_batches += 1
            if sample_batches >= 100:  # Sample more batches for better estimation
                break
                
        if len(label_counter) > 0:
            total = sum(label_counter.values())
            pos_freq = label_counter.get(1, 0) / max(1, total)
            neg_freq = label_counter.get(0, 0) / max(1, total)
            
            # Compute balanced alpha (weight for positive class)
            adaptive_alpha = neg_freq / (pos_freq + neg_freq) if (pos_freq + neg_freq) > 0 else 0.25
            adaptive_alpha = max(0.1, min(0.9, adaptive_alpha))  # Clamp to reasonable range
            
            print(f"📈 Data statistics: Pos={pos_freq:.3f}, Neg={neg_freq:.3f}")
            print(f"🎯 Adaptive focal alpha: {adaptive_alpha:.3f}")
    except Exception as e:
        print(f"⚠️ Failed to compute adaptive alpha: {e}")
        adaptive_alpha = config.focal_alpha
    
    print("🧠 Building enhanced model...")
    model = SeqSetVAE(
        input_dim=768,  # Medical embeddings dimension
        reduced_dim=256,
        latent_dim=256,
        levels=2,
        heads=2,
        m=16,
        beta=0.1,
        lr=config.lr,
        num_classes=2,
        ff_dim=512,
        transformer_heads=8,
        transformer_layers=4,
        pretrained_ckpt=args.pretrained_ckpt,
        w=3.0,
        free_bits=0.03,
        warmup_beta=True,
        max_beta=0.05,
        beta_warmup_steps=8000,
        kl_annealing=True,
        use_focal_loss=True,
        focal_alpha=adaptive_alpha or config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )
    
    # Enable enhanced finetuning mode
    model.enable_classification_only_mode(cls_head_lr=config.cls_head_lr)
    
    # Enhanced parameter freezing with selective unfreezing
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if name.startswith('cls_head'):
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"   ✅ Trainable: {name} ({param.numel():,} params)")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    # Initialize classifier with enhanced scheme
    model.init_classifier_head_xavier()
    
    print("🧊 Enhanced Finetuning Configuration:")
    print(f"   - Frozen parameters: {frozen_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Trainable ratio: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")
    print(f"   - Advanced classifier: Multi-head attention + residual connections")
    print(f"   - Auxiliary loss: 30% weight for better gradient flow")
    
    # Enhanced callbacks
    callbacks = []
    
    # Checkpoint callback optimized for AUC/AUPRC
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="enhanced_finetune_auc{val_auc:.4f}_auprc{val_auprc:.4f}",
        save_top_k=3,
        monitor="val_auc",
        mode="max",
        save_last=True,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Enhanced early stopping
    early_stopping = EarlyStopping(
        monitor="val_auc",
        patience=config.early_stopping_patience,
        mode="max",
        min_delta=0.001,  # Smaller delta for fine-grained improvements
        verbose=True,
        check_finite=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Custom AUC/AUPRC monitoring
    auc_monitor = AUCPRCMonitor()
    callbacks.append(auc_monitor)
    
    # Posterior collapse monitoring
    collapse_monitor = PosteriorMetricsMonitor(
        log_dir=monitor_dir,
        update_frequency=50,
        plot_frequency=200,
        window_size=100,
        verbose=False,
    )
    callbacks.append(collapse_monitor)
    
    # Enhanced logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="enhanced_finetune",
        version=f"auc_auprc_opt_{timestamp}",
        log_graph=True,
    )
    
    # Enhanced trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        val_check_interval=config.val_check_interval,
        limit_val_batches=config.limit_val_batches,
        log_every_n_steps=25,  # More frequent logging
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        deterministic=True,
        benchmark=False,  # Ensure reproducible results
        default_root_dir=experiment_root,
    )
    
    print("🚀 Starting enhanced training...")
    print("🎯 Target: AUC ≥ 0.90, AUPRC ≥ 0.50")
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Final results
    if hasattr(auc_monitor, 'best_auc'):
        print("\n" + "="*60)
        print("🏆 FINAL RESULTS:")
        print(f"   Best AUC: {auc_monitor.best_auc:.4f}")
        print(f"   Best AUPRC: {auc_monitor.best_auprc:.4f}")
        print(f"   Combined Score: {auc_monitor.best_combined_score:.4f}")
        
        if auc_monitor.best_auc >= 0.90:
            print("✅ AUC target achieved!")
        else:
            print(f"📈 AUC progress: {auc_monitor.best_auc:.4f}/0.90")
            
        if auc_monitor.best_auprc >= 0.50:
            print("✅ AUPRC target achieved!")
        else:
            print(f"📈 AUPRC progress: {auc_monitor.best_auprc:.4f}/0.50")
        print("="*60)
    
    print("✅ Enhanced training completed!")


if __name__ == "__main__":
    main()