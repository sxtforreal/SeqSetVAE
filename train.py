#!/usr/bin/env python3
"""
Unified SeqSetVAE Training Script
Supports standard, AUC-optimized, and enhanced training modes
Combines functionality from train_optimized.py and train.py
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
import config

def get_adaptive_training_config(args):
    """
    Adaptively adjust training parameters based on device configuration
    """
    # Get device configuration
    device_config = getattr(config, 'device_config', {
        'cuda_available': torch.cuda.is_available(),
        'devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })
    
    # Adjust parameters based on device type
    if device_config['cuda_available']:
        # GPU training configuration
        if device_config['devices'] > 1:
            # Multi-GPU training
            strategy = DDPStrategy(find_unused_parameters=False)
            effective_batch_size = args.batch_size * args.gradient_accumulation_steps * device_config['devices']
        else:
            # Single GPU training
            strategy = "auto"
            effective_batch_size = args.batch_size * args.gradient_accumulation_steps
            
        # Adjust worker count based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:
            num_workers = min(args.num_workers, 8)
        elif gpu_memory >= 8:
            num_workers = min(args.num_workers, 6)
        else:
            num_workers = min(args.num_workers, 3)
            
        pin_memory = True
        compile_model = args.compile_model and hasattr(torch, 'compile')
        
    else:
        # CPU training configuration
        strategy = "auto"
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        
        # Reduce worker count for CPU training to avoid overload
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(args.num_workers, max(1, cpu_count // 2))
        
        pin_memory = False  # CPU training doesn't need pin_memory
        compile_model = False  # Disable compile for CPU training
        
        # Force 32-bit precision for CPU training
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

def get_auc_optimized_config():
    """Get AUC/AUPRC optimized configuration parameters"""
    return {
        'w': 3.0,  # Enhanced classification weight
        'free_bits': 0.03,  # Reduced KL divergence weight
        'max_beta': 0.03,  # Reduced maximum beta
        'beta_warmup_steps': 10000,  # Longer warmup
        'gradient_clip_val': 0.2,  # Stricter gradient clipping
        'focal_alpha': 0.35,  # Enhanced focal loss alpha
        'focal_gamma': 3.0,  # Enhanced focal loss gamma
        'lr': 5e-5,  # Lower learning rate for stability
        'val_check_interval': 0.15,  # More frequent validation
        'limit_val_batches': 0.6,  # Use more validation data
        'early_stopping_patience': 8,  # More patience
        'early_stopping_min_delta': 0.0005,  # Smaller improvement threshold
        'save_top_k': 5,  # Save more checkpoints
        'monitor_metric': 'val_auc',  # Monitor AUC
        'weight_decay': 0.03,  # Increased weight decay
        'scheduler_patience': 150,  # Longer scheduler patience
        'scheduler_factor': 0.6,  # Reduce LR by 40% on plateau
        'scheduler_min_lr': 1e-6,  # Minimum learning rate
    }

def setup_metrics_monitor(args, experiment_output_dir, auc_mode=False):
    """Set up posterior metrics monitoring."""
    if auc_mode:
        # AUC mode: more frequent monitoring and plotting
        update_frequency = 50
        plot_frequency = 200
        window_size = 100
        verbose = True
    else:
        # Standard mode: balanced monitoring
        update_frequency = 100
        plot_frequency = 500
        window_size = 200
        verbose = False
    
    monitor = PosteriorMetricsMonitor(
        log_dir=experiment_output_dir,
        update_frequency=update_frequency,
        plot_frequency=plot_frequency,
        window_size=window_size,
        verbose=verbose
    )
    
    return monitor

def main():
    parser = argparse.ArgumentParser(description="Train SeqSetVAE model with unified training script")
    
    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices (auto-detect if not specified)")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--compile_model", action="store_true", help="Enable model compilation")
    
    # Training modes
    parser.add_argument("--auc_mode", action="store_true", help="Enable AUC/AUPRC optimization mode")
    parser.add_argument("--enhanced_mode", action="store_true", help="Enable enhanced model architecture mode")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr", help="Data directory path")
    parser.add_argument("--params_map_path", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv", help="Parameter mapping file path")
    parser.add_argument("--label_path", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv", help="Label file path")
    
    # Advanced training parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    # Output root directory
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs",
        help="Root directory to save logs, checkpoints, and outputs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Alias of --output_root_dir; if provided, overrides output_root_dir"
    )
    
    args = parser.parse_args()
    
    # Normalize and prepare unified output directories: output_dir/config.model_name/{checkpoints,logs,analysis}
    base_output_dir = (args.output_dir or args.output_root_dir).rstrip("/")
    model_name = getattr(config, 'model_name', getattr(config, 'name', 'SeqSetVAE'))
    experiment_root = os.path.join(base_output_dir, model_name)
    checkpoints_root_dir = os.path.join(experiment_root, 'checkpoints')
    logs_root_dir = os.path.join(experiment_root, 'logs')
    analysis_root_dir = os.path.join(experiment_root, 'analysis')
    os.makedirs(checkpoints_root_dir, exist_ok=True)
    os.makedirs(logs_root_dir, exist_ok=True)
    os.makedirs(analysis_root_dir, exist_ok=True)
    outputs_root_dir = experiment_root
    
    # Set random seed for reproducibility
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Auto-detect devices if not specified
    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = torch.cuda.device_count()
        else:
            args.devices = 0
    
    # Print configuration
    print("🚀 Unified SeqSetVAE Training Configuration:")
    print(f" - Batch size: {args.batch_size}")
    print(f" - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f" - Max epochs: {args.max_epochs}")
    print(f" - Devices: {args.devices}")
    print(f" - Strategy: {args.strategy}")
    print(f" - Precision: {args.precision}")
    print(f" - Data directory: {args.data_dir}")
    print(f" - Random seed: {args.seed}")
    print(f" - Deterministic: {args.deterministic}")
    
    # All parameters now use enhanced values by default
    print("🚀 Enhanced model architecture mode enabled by default!")
    
    # Get adaptive training configuration
    adaptive_config = get_adaptive_training_config(args)
    
    # Set up data module
    print("📊 Setting up data module...")
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        max_sequence_length=None,  # No limit on sequence length
        use_dynamic_padding=True,  # Enable dynamic padding
        num_workers=adaptive_config['num_workers'],
        pin_memory=adaptive_config['pin_memory'],
    )
    
    # Get configuration parameters (all parameters now use enhanced values by default)
    model_w = config.w
    model_free_bits = config.free_bits
    model_max_beta = config.max_beta
    model_beta_warmup_steps = config.beta_warmup_steps
    model_gradient_clip_val = config.gradient_clip_val
    model_focal_alpha = config.focal_alpha
    model_focal_gamma = config.focal_gamma
    model_lr = config.lr
    model_weight_decay = config.weight_decay
    val_check_interval = config.val_check_interval
    limit_val_batches = config.limit_val_batches
    early_stopping_patience = config.early_stopping_patience
    early_stopping_min_delta = config.early_stopping_min_delta
    save_top_k = config.save_top_k
    monitor_metric = config.monitor_metric
    scheduler_patience = config.scheduler_patience
    scheduler_factor = config.scheduler_factor
    scheduler_min_lr = config.scheduler_min_lr
    print("✅ Enhanced configuration loaded successfully (default)")
    
    # Set up model with configuration based on mode
    print("🧠 Setting up model...")
    
    # Use enhanced configuration by default (all parameters are now enhanced)
    model_ff_dim = config.ff_dim
    model_transformer_heads = config.transformer_heads
    model_transformer_layers = config.transformer_layers
    print("🚀 Using enhanced model architecture (default)")
    
    model = SeqSetVAE(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=model_lr,
        num_classes=config.num_classes,
        ff_dim=model_ff_dim,
        transformer_heads=model_transformer_heads,
        transformer_layers=model_transformer_layers,
        pretrained_ckpt=None,
        w=model_w,
        free_bits=model_free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=model_max_beta,
        beta_warmup_steps=model_beta_warmup_steps,
        kl_annealing=config.kl_annealing,
        use_focal_loss=getattr(config, 'use_focal_loss', True),
        focal_alpha=model_focal_alpha,
        focal_gamma=model_focal_gamma,
    )
    
    # Print model architecture details
    print(f"📊 Model Architecture:")
    print(f" - FF dimension: {model_ff_dim}")
    print(f" - Transformer heads: {model_transformer_heads}")
    print(f" - Transformer layers: {model_transformer_layers}")
            print(f" - Classification head layers: {config.cls_head_layers}")
    
    # Set up training configuration
    print("⚙️ Setting up training configuration...")
    
    # All parameters now use enhanced values by default
    checkpoint_name = "SeqSetVAE_enhanced"
    mode_suffix = "_enhanced"
    
    # Ensure subdirectories exist at start
    os.makedirs(os.path.join(checkpoints_root_dir, checkpoint_name), exist_ok=True)
    os.makedirs(os.path.join(logs_root_dir, checkpoint_name), exist_ok=True)
    os.makedirs(os.path.join(analysis_root_dir, checkpoint_name), exist_ok=True)

    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoints_root_dir, checkpoint_name),
        filename=f"{checkpoint_name}_batch{args.batch_size}",
        save_top_k=save_top_k,
        monitor=monitor_metric,
        mode="max" if monitor_metric in ["val_auc", "val_auprc"] else "min",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=early_stopping_patience,
        mode="max" if monitor_metric in ["val_auc", "val_auprc"] else "min",
        min_delta=early_stopping_min_delta,
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Set up posterior metrics monitor (write to analysis directory)
    experiment_output_dir = os.path.join(analysis_root_dir, checkpoint_name)
    monitor = setup_metrics_monitor(args, experiment_output_dir, auc_mode=args.auc_mode)
    callbacks.append(monitor)
    
    # Set up logger (save under unified logs dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=os.path.join(logs_root_dir, checkpoint_name),
        name="",
        version=f"batch{args.batch_size}_{timestamp}",
        log_graph=True,
    )
    
    # Set up trainer
    print("🏃 Setting up trainer...")
    
    # Use adaptive strategy configuration
    strategy = adaptive_config['strategy']
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=model_gradient_clip_val,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=50,
        deterministic=args.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        use_distributed_sampler=True if args.devices > 1 else False,
        default_root_dir=experiment_root,
    )
    
    # Compile model if requested and supported
    if adaptive_config['compile_model']:
        print("🔧 Compiling model...")
        model = torch.compile(model)
    
    # Print training configuration
    print(f"🎯 Training Configuration:")
    print(f" - Mode: {mode_suffix}")
    print(f" - Learning rate: {model_lr}")
    print(f" - Weight decay: {model_weight_decay}")
    print(f" - Gradient clipping: {model_gradient_clip_val}")
    print(f" - Focal loss alpha: {model_focal_alpha}")
    print(f" - Focal loss gamma: {model_focal_gamma}")
    print(f" - Free bits: {model_free_bits}")
    print(f" - Max beta: {model_max_beta}")
    print(f" - Beta warmup steps: {model_beta_warmup_steps}")
    print(f" - Validation check interval: {val_check_interval}")
    print(f" - Early stopping patience: {early_stopping_patience}")
    print(f" - Monitor metric: {monitor_metric}")
    print(f" - Effective batch size: {adaptive_config['effective_batch_size']}")
    print(f" - Number of workers: {adaptive_config['num_workers']}")
    print(f" - Pin memory: {adaptive_config['pin_memory']}")
    
    # Start training
    print("🚀 Starting training...")
    try:
        trainer.fit(model, data_module)
        print("✅ Training completed successfully!")
        
        # Print monitoring summary
        if hasattr(monitor, 'steps') and len(monitor.steps) > 0:
            print(f"📊 Posterior metrics monitoring summary:")
            print(f" - Total steps monitored: {len(monitor.steps)}")
            print(f" - KL divergence range: {min(monitor.kl_divergences):.4f} - {max(monitor.kl_divergences):.4f}")
            print(f" - Latent variance range: {min(monitor.latent_variances):.4f} - {max(monitor.latent_variances):.4f}")
            print(f" - Active units ratio range: {min(monitor.active_units_ratios):.4f} - {max(monitor.active_units_ratios):.4f}")
            print(f" - Reconstruction loss range: {min(monitor.reconstruction_losses):.4f} - {max(monitor.reconstruction_losses):.4f}")
        else:
            print("📊 No posterior metrics available for summary")
            
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        
        # Save error model
        error_checkpoint_path = os.path.join(checkpoints_root_dir, checkpoint_name, f"error_{checkpoint_name}_batch{args.batch_size}.ckpt")
        os.makedirs(os.path.dirname(error_checkpoint_path), exist_ok=True)
        trainer.save_checkpoint(error_checkpoint_path)
        print(f"💾 Error model saved: {error_checkpoint_path}")
        
        raise
    
    print("🎉 Unified SeqSetVAE training completed!")

if __name__ == "__main__":
    main()
