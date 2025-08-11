#!/usr/bin/env python3
"""
Enhanced training script with both general optimization and AUC/AUPRC optimization modes
"""
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorMetricsMonitor
import config
import os
import argparse
from datetime import datetime
import torch

def get_adaptive_training_config(args):
    """
    Adaptively adjust training parameters based on device configuration
    """
    # Get device configuration
    device_config = config.device_config
    
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
            num_workers = min(args.num_workers, 8)  # Reduced from 12
        elif gpu_memory >= 8:
            num_workers = min(args.num_workers, 6)  # Reduced from 8
        else:
            num_workers = min(args.num_workers, 3)  # Reduced from 4
            
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
    """Set up posterior metrics monitor with optional AUC optimization"""
    
    # Set up metrics monitoring log directory based on config.name
    posterior_metrics_dir = os.path.join(experiment_output_dir, "posterior_metrics")
    
    if auc_mode:
        # AUC optimized monitoring parameters
        monitor = PosteriorMetricsMonitor(
            update_frequency=50,         # More frequent updates for AUC optimization
            plot_frequency=500,          # More frequent plotting
            window_size=100,             # Larger window for trend analysis
            
            log_dir=posterior_metrics_dir,
            verbose=True,                # Enable verbose for better monitoring
        )
        print(f"📊 AUC optimized monitor setup complete:")
    else:
        # Standard optimized monitoring parameters
        if args.fast_detection:
            # Fast monitoring mode - less frequent updates for better performance
            monitor = PosteriorMetricsMonitor(
                update_frequency=100,         # Reduced from 20 to 100
                plot_frequency=1000,          # Reduced from 200 to 1000
                window_size=50,               # Reduced from 100 to 50
                
                log_dir=posterior_metrics_dir,
                verbose=False,                # Disable verbose output for better performance
            )
        else:
            # Standard monitoring mode - optimized
            monitor = PosteriorMetricsMonitor(
                update_frequency=200,         # Reduced from 50 to 200
                plot_frequency=2000,          # Reduced from 500 to 2000
                window_size=50,               # Reduced from 100 to 50
                
                log_dir=posterior_metrics_dir,
                verbose=False,                # Disable verbose output for better performance
            )
        print(f"📊 Standard optimized monitor setup complete:")
    
    print(f"  - Update frequency: every {monitor.update_frequency} steps")
    print(f"  - Plot frequency: every {monitor.plot_frequency} steps")
    print(f"  - Log directory: {monitor.log_dir}")
    
    return monitor

def main():
    """Main training function with both optimization modes"""
    
    # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced SeqSetVAE Training with AUC Optimization")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--disable_metrics_monitoring", action="store_true", help="Disable metrics monitoring")
    parser.add_argument("--fast_detection", action="store_true", help="Enable fast detection mode")
    parser.add_argument("--compile_model", action="store_true", help="Enable model compilation")
    parser.add_argument("--auc_mode", action="store_true", help="Enable AUC/AUPRC optimization mode")
    parser.add_argument("--enhanced_mode", action="store_true", help="Enable enhanced model architecture mode")
    parser.add_argument("--data_dir", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr", help="Data directory path")
    parser.add_argument("--params_map_path", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv", help="Parameter mapping file path")
    parser.add_argument("--label_path", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv", help="Label file path")
    
    args = parser.parse_args()
    
    # Create experiment output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.enhanced_mode:
        mode_suffix = "_enhanced"
    elif args.auc_mode:
        mode_suffix = "_auc_optimized"
    else:
        mode_suffix = "_optimized"
    experiment_name = f"{config.name}{mode_suffix}_{timestamp}"
    experiment_output_dir = os.path.join("outputs", experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    print(f"🚀 Starting {experiment_name} training...")
    print(f"📁 Output directory: {experiment_output_dir}")
    if args.enhanced_mode:
        print("🚀 Enhanced model architecture mode enabled!")
    elif args.auc_mode:
        print("🎯 AUC/AUPRC optimization mode enabled!")
    
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
    
    # Get configuration parameters
    if args.enhanced_mode:
        # Enhanced mode: use enhanced architecture and training config
        model_w = config.enhanced_w
        model_free_bits = config.enhanced_free_bits
        model_max_beta = config.max_beta  # Keep from base config
        model_beta_warmup_steps = config.beta_warmup_steps  # Keep from base config
        model_gradient_clip_val = config.enhanced_gradient_clip_val
        model_focal_alpha = config.enhanced_focal_alpha
        model_focal_gamma = config.enhanced_focal_gamma
        model_lr = config.enhanced_lr
        model_weight_decay = config.enhanced_weight_decay
        val_check_interval = config.enhanced_val_check_interval
        limit_val_batches = config.enhanced_limit_val_batches
        early_stopping_patience = config.enhanced_early_stopping_patience
        early_stopping_min_delta = config.enhanced_early_stopping_min_delta
        save_top_k = config.enhanced_save_top_k
        monitor_metric = config.enhanced_monitor_metric
        scheduler_patience = config.enhanced_scheduler_patience
        scheduler_factor = config.enhanced_scheduler_factor
        scheduler_min_lr = config.enhanced_scheduler_min_lr
        print("✅ Enhanced configuration loaded successfully")
    elif args.auc_mode:
        auc_config = get_auc_optimized_config()
        # Override config parameters for AUC optimization
        model_w = auc_config['w']
        model_free_bits = auc_config['free_bits']
        model_max_beta = auc_config['max_beta']
        model_beta_warmup_steps = auc_config['beta_warmup_steps']
        model_gradient_clip_val = auc_config['gradient_clip_val']
        model_focal_alpha = auc_config['focal_alpha']
        model_focal_gamma = auc_config['focal_gamma']
        model_lr = auc_config['lr']
        model_weight_decay = auc_config['weight_decay']
        val_check_interval = auc_config['val_check_interval']
        limit_val_batches = auc_config['limit_val_batches']
        early_stopping_patience = auc_config['early_stopping_patience']
        early_stopping_min_delta = auc_config['early_stopping_min_delta']
        save_top_k = auc_config['save_top_k']
        monitor_metric = auc_config['monitor_metric']
        scheduler_patience = auc_config['scheduler_patience']
        scheduler_factor = auc_config['scheduler_factor']
        scheduler_min_lr = auc_config['scheduler_min_lr']
    else:
        # Use standard config parameters
        model_w = config.w
        model_free_bits = config.free_bits
        model_max_beta = config.max_beta
        model_beta_warmup_steps = config.beta_warmup_steps
        model_gradient_clip_val = config.gradient_clip_val
        model_focal_alpha = config.focal_alpha
        model_focal_gamma = config.focal_gamma
        model_lr = config.lr
        model_weight_decay = 0.01  # Default weight decay
        val_check_interval = 0.1
        limit_val_batches = 0.3
        early_stopping_patience = 3
        early_stopping_min_delta = 0.001
        save_top_k = 3
        monitor_metric = "val_loss"
        scheduler_patience = 100
        scheduler_factor = 0.7
        scheduler_min_lr = config.lr * 0.01
    
    # Set up model with configuration based on mode
    print("🧠 Setting up model...")
    
    # Choose configuration based on mode
    if args.enhanced_mode:
        # Enhanced mode: use enhanced architecture parameters
        model_ff_dim = config.enhanced_ff_dim
        model_transformer_heads = config.enhanced_transformer_heads
        model_transformer_layers = config.enhanced_transformer_layers
        print("🚀 Using enhanced model architecture")
    else:
        # Standard/AUC mode: use standard architecture parameters
        model_ff_dim = config.ff_dim
        model_transformer_heads = config.transformer_heads
        model_transformer_layers = config.transformer_layers
        print("📊 Using standard model architecture")
    
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
        pretrained_ckpt=config.pretrained_ckpt,
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
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=experiment_output_dir,
        name="tensorboard_logs",
        version=timestamp,
        log_graph=True,
    )
    
    # Set up callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="max" if monitor_metric == "val_auc" else "min",
        patience=early_stopping_patience,
        verbose=True,
        strict=True,
        min_delta=early_stopping_min_delta,
    )
    callbacks.append(early_stopping)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_output_dir, "checkpoints"),
        filename=f"{config.name}{mode_suffix}_batch{args.batch_size}_" + 
                ("{epoch:02d}_{val_auc:.4f}" if monitor_metric == "val_auc" else "{epoch:02d}_{val_loss:.4f}"),
        monitor=monitor_metric,
        mode="max" if monitor_metric == "val_auc" else "min",
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Set up posterior metrics monitor
    if not args.disable_metrics_monitoring:
        monitor = setup_metrics_monitor(args, experiment_output_dir, args.auc_mode)
        callbacks.append(monitor)
    else:
        print("⚠️  Posterior metrics monitoring disabled")
    
    # Create trainer with optimized configuration
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,  # Use adaptive configuration device count
        strategy=adaptive_config['strategy'],
        logger=logger,
        max_epochs=config.max_epochs,
        min_epochs=1,
        precision=config.precision,  # Use adaptive configuration precision
        callbacks=callbacks,
        profiler=None,  # Disable profiler for better performance
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=model_gradient_clip_val,
        gradient_clip_algorithm="norm",
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=args.gradient_accumulation_steps,  # Use gradient accumulation
        detect_anomaly=False,    # Turn off anomaly detection
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable model summary for faster startup
        sync_batchnorm=False,  # Disable sync batchnorm for better performance
        use_distributed_sampler=True,
    )
    
    # Print training configuration
    print("\n📋 Training configuration:")
    print(f"  - Mode: {'Enhanced Architecture' if args.enhanced_mode else 'AUC Optimized' if args.auc_mode else 'Standard Optimized'}")
    print(f"  - Max epochs: {config.max_epochs}")
    print(f"  - Accelerator: {config.accelerator}")
    print(f"  - Number of devices: {config.devices}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Effective batch size: {adaptive_config['effective_batch_size']}")
    print(f"  - Precision: {config.precision}")
    print(f"  - Learning rate: {model_lr}")
    print(f"  - Classification weight (w): {model_w}")
    print(f"  - Max beta: {model_max_beta}")
    print(f"  - Beta warmup steps: {model_beta_warmup_steps}")
    print(f"  - Gradient clip value: {model_gradient_clip_val}")
    print(f"  - Focal loss alpha: {model_focal_alpha}")
    print(f"  - Focal loss gamma: {model_focal_gamma}")
    print(f"  - Validation interval: {val_check_interval}")
    print(f"  - Validation batches: {limit_val_batches}")
    print(f"  - Early stopping patience: {early_stopping_patience}")
    print(f"  - Monitor metric: {monitor_metric}")
    print(f"  - Number of workers: {adaptive_config['num_workers']}")
    print(f"  - Pin memory: {adaptive_config['pin_memory']}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Model compilation: {'Enabled' if adaptive_config['compile_model'] else 'Disabled'}")
    print(f"  - Strategy: {adaptive_config['strategy']}")
    print(f"  - Metrics monitoring: {'Enabled' if not args.disable_metrics_monitoring else 'Disabled'}")
    
    # Print architecture information
    if args.enhanced_mode:
        print(f"  - Architecture: Enhanced (FF: {config.enhanced_ff_dim}, Heads: {config.enhanced_transformer_heads}, Layers: {config.enhanced_transformer_layers})")
    else:
        print(f"  - Architecture: Standard (FF: {config.ff_dim}, Heads: {config.transformer_heads}, Layers: {config.transformer_layers})")
    
    if args.resume_from_checkpoint:
        print(f"  - Resuming from checkpoint: {args.resume_from_checkpoint}")
    
    # Start training
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    try:
        # Start training with or without checkpoint resume
        if args.resume_from_checkpoint:
            trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, data_module)
        
        # Training completion summary
        print("\n✅ Training completed!")
        
        if not args.disable_metrics_monitoring:
            print("\n📊 Posterior metrics monitoring summary:")
            monitor = None
            for callback in trainer.callbacks:
                if isinstance(callback, PosteriorMetricsMonitor):
                    monitor = callback
                    break
                    
            if monitor:
                print(f"  - Total steps monitored: {len(monitor.steps)}")
                print(f"  - Update frequency: every {monitor.update_frequency} steps")
                print(f"  - Plot frequency: every {monitor.plot_frequency} steps")
                print(f"  - Log directory: {monitor.log_dir}")
        
        # Save final model
        final_model_path = os.path.join(experiment_output_dir, "checkpoints", 
                                      f"final_{config.name}{mode_suffix}_batch{args.batch_size}.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Print best metric achieved
        best_metric = checkpoint_callback.best_model_score
        if best_metric is not None:
            print(f"🏆 Best {monitor_metric} achieved: {best_metric:.4f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        
        # Save model when interrupted
        interrupt_model_path = os.path.join(experiment_output_dir, "checkpoints", 
                                         f"interrupted_{config.name}{mode_suffix}_batch{args.batch_size}.ckpt")
        trainer.save_checkpoint(interrupt_model_path)
        print(f"💾 Interrupted model saved: {interrupt_model_path}")
        
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        
        # Save model when error occurs
        error_model_path = os.path.join(experiment_output_dir, "checkpoints", 
                                      f"error_{config.name}{mode_suffix}_batch{args.batch_size}.ckpt")
        try:
            trainer.save_checkpoint(error_model_path)
            print(f"💾 Error model saved: {error_model_path}")
        except:
            print("❌ Unable to save error model")
        
        raise
    
    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()
