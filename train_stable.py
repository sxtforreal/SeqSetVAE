#!/usr/bin/env python3
"""
Stable SeqSetVAE Training Script with Optimized Configuration
Designed to address high training loss issues
"""

import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch import seed_everything
from datetime import datetime

# Import local modules
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorMetricsMonitor
import config_optimized as config

def setup_training_environment():
    """Setup training environment with stability optimizations"""
    print("🔧 Setting up stable training environment...")
    
    # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)
    
    # Enable deterministic mode if requested
    if getattr(config, 'use_deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✅ Deterministic mode enabled")
    else:
        torch.backends.cudnn.benchmark = True
        print("✅ CUDNN benchmark enabled for speed")
    
    # Set precision
    if config.precision == "16-mixed" and getattr(config, 'use_amp', True):
        print("✅ Mixed precision training enabled")
    else:
        print("✅ Full precision training")
    
    print("✅ Training environment setup complete")

def get_stable_training_config(args):
    """Get stable training configuration"""
    device_config = getattr(config, 'device_config', {
        'cuda_available': torch.cuda.is_available(),
        'devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })
    
    if device_config['cuda_available']:
        if device_config['devices'] > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
            effective_batch_size = args.batch_size * getattr(config, 'gradient_accumulation_steps', 4) * device_config['devices']
        else:
            strategy = "auto"
            effective_batch_size = args.batch_size * getattr(config, 'gradient_accumulation_steps', 4)
        
        # Conservative worker settings for stability
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:
            num_workers = min(args.num_workers, 4)  # Reduced for stability
        elif gpu_memory >= 8:
            num_workers = min(args.num_workers, 3)  # Reduced for stability
        else:
            num_workers = min(args.num_workers, 2)  # Reduced for stability
            
        pin_memory = True
        compile_model = False  # Disable compile for stability
        
    else:
        strategy = "auto"
        effective_batch_size = args.batch_size * getattr(config, 'gradient_accumulation_steps', 4)
        
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(args.num_workers, max(1, cpu_count // 4))  # More conservative
        
        pin_memory = False
        compile_model = False
        
        if args.precision != "32":
            print("⚠️  CPU training detected, switching to 32-bit precision")
            args.precision = "32"
    
    return {
        'strategy': strategy,
        'effective_batch_size': effective_batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'compile_model': compile_model
    }

def main():
    parser = argparse.ArgumentParser(description="Stable SeqSetVAE Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--params_map_path", type=str, required=True, help="Parameters map path")
    parser.add_argument("--label_path", type=str, required=True, help="Label path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    
    args = parser.parse_args()
    
    # Setup training environment
    setup_training_environment()
    
    # Print configuration
    print("🚀 Stable SeqSetVAE Training Configuration:")
    print(f" - Batch size: {args.batch_size}")
    print(f" - Gradient accumulation steps: {getattr(config, 'gradient_accumulation_steps', 4)}")
    print(f" - Max epochs: {args.max_epochs}")
    print(f" - Devices: {args.devices}")
    print(f" - Strategy: {args.strategy}")
    print(f" - Precision: {args.precision}")
    print(f" - Data directory: {args.data_dir}")
    print(f" - Random seed: {args.seed}")
    print(f" - Deterministic: {args.deterministic}")
    
    print("🚀 STABLE TRAINING MODE ENABLED!")
    print("   - Lower learning rate: {config.lr}")
    print("   - Reduced KL weight: {config.beta}")
    print("   - Simplified architecture: {config.transformer_layers} layers, {config.transformer_heads} heads")
    print("   - Enhanced stability parameters")
    
    # Get stable training configuration
    stable_config = get_stable_training_config(args)
    
    # Set up data module
    print("📊 Setting up data module...")
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        max_sequence_length=None,
        use_dynamic_padding=True,
        num_workers=stable_config['num_workers'],
        pin_memory=stable_config['pin_memory'],
    )
    
    # Set up model with STABLE configuration
    print("🧠 Setting up STABLE model...")
    
    model = SeqSetVAE(
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
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
        use_focal_loss=getattr(config, 'use_focal_loss', True),
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )
    
    # Print model architecture details
    print(f"📊 STABLE Model Architecture:")
    print(f" - FF dimension: {config.ff_dim}")
    print(f" - Transformer heads: {config.transformer_heads}")
    print(f" - Transformer layers: {config.transformer_layers}")
    print(f" - Classification head layers: {config.cls_head_layers}")
    
    # Set up training configuration
    print("⚙️ Setting up STABLE training configuration...")
    
    # Use stable configuration
    checkpoint_name = "SeqSetVAE_stable"
    mode_suffix = "_stable"
    
    # Ensure subdirectories exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{checkpoint_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename=f"{checkpoint_name}_" + "{epoch:02d}_{val_loss:.4f}",
        monitor=config.monitor_metric,
        mode="min" if "loss" in config.monitor_metric else "max",
        save_top_k=config.save_top_k,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config.monitor_metric,
        mode="min" if "loss" in config.monitor_metric else "max",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Posterior collapse detection
    posterior_monitor = PosteriorMetricsMonitor()
    callbacks.append(posterior_monitor)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=f"{output_dir}/logs",
        name=checkpoint_name,
        version=timestamp,
        log_graph=False,
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=config.accelerator,
        devices=args.devices,
        strategy=stable_config['strategy'],
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        limit_val_batches=config.limit_val_batches,
        gradient_clip_val=config.gradient_clip_val,
        deterministic=args.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        use_distributed_sampler=True,
        sync_batchnorm=getattr(config, 'use_sync_batchnorm', False),
        accumulate_grad_batches=getattr(config, 'gradient_accumulation_steps', 4),
    )
    
    print("🚀 Starting STABLE training...")
    print(f"   - Output directory: {output_dir}")
    print(f"   - Checkpoint directory: {output_dir}/checkpoints")
    print(f"   - Log directory: {output_dir}/logs")
    
    # Start training
    trainer.fit(model, data_module)
    
    print("✅ Training completed!")
    print(f"   - Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"   - Best {config.monitor_metric}: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()