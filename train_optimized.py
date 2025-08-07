import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorMetricsMonitor
import config
import os
import argparse
from datetime import datetime
import torch

def setup_metrics_monitor(args, experiment_output_dir):
    """Set up posterior metrics monitor with optimized settings"""
    
    # Set up metrics monitoring log directory based on config.name
    posterior_metrics_dir = os.path.join(experiment_output_dir, "posterior_metrics")
    
    # Optimized monitoring parameters for better performance
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
    
    print(f"📊 Posterior metrics monitor setup complete:")
    print(f"  - Monitoring mode: {'Fast' if args.fast_detection else 'Standard'}")
    print(f"  - Update frequency: every {monitor.update_frequency} steps")
    print(f"  - Plot frequency: every {monitor.plot_frequency} steps")
    print(f"  - Log directory: {monitor.log_dir}")
    
    return monitor

class OptimizedEarlyStopping(EarlyStopping):
    """Optimized early stopping callback"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _should_stop_early(self, trainer, pl_module):
        return super()._should_stop_early(trainer, pl_module)

def main():
    parser = argparse.ArgumentParser(description='Train SeqSetVAE with optimized performance')
    
    # Basic training parameters
    parser.add_argument('--max_epochs', type=int, default=config.max_epochs, 
                       help='Maximum training epochs')
    parser.add_argument('--devices', type=int, default=config.devices,
                       help='Number of GPUs to use')
    
    # Checkpoint resume parameter
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    # Batch training parameters
    parser.add_argument('--batch_size', type=int, default=4,  # Increased default batch size
                       help='Batch size for training (1 for single-patient, >1 for multi-patient)')
    parser.add_argument('--max_sequence_length', type=int, default=1000,  # Added default limit
                       help='Maximum sequence length to truncate (None for no limit)')
    parser.add_argument('--use_dynamic_padding', action='store_true', default=True,
                       help='Use dynamic padding for batch training')
    
    # Performance optimization parameters
    parser.add_argument('--num_workers', type=int, default=8,  # Increased worker count
                       help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Use pin memory for faster data transfer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       help='Training precision (16-mixed, 32, bf16-mixed)')
    parser.add_argument('--compile_model', action='store_true', default=False,
                       help='Use torch.compile for model optimization')
    
    # Metrics monitoring parameters
    parser.add_argument('--fast_detection', action='store_true',
                       help='Enable fast monitoring mode (less frequent updates)')
    parser.add_argument('--disable_metrics_monitoring', action='store_true',
                       help='Disable posterior metrics monitoring')
    parser.add_argument('--log_dir', type=str, 
                       default=None,
                       help='Metrics monitoring log directory')
    
    # Data path parameters
    parser.add_argument('--data_dir', type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr",
                       help='Data directory path')
    parser.add_argument('--params_map_path', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
                       help='Parameter mapping file path')
    parser.add_argument('--label_path', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv", 
                       help='Label file path')
    
    # Output path parameters
    parser.add_argument('--output_dir', type=str,
                       default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs",
                       help='Output directory path')
    
    args = parser.parse_args()
    
    print("🚀 Starting optimized SeqSetVAE training")
    print("=" * 60)
    
    # Check if resuming from checkpoint
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from_checkpoint}")
        print(f"🔄 Resuming training from checkpoint: {args.resume_from_checkpoint}")
    
    # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)
    
    # Create output directory structure based on config.name
    experiment_output_dir = os.path.join(args.output_dir, config.name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_dir, "posterior_metrics"), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(experiment_output_dir, "analysis"), exist_ok=True)
    
    print(f"📁 Output directory structure created: {experiment_output_dir}")
    print(f"  - Checkpoints: {os.path.join(experiment_output_dir, 'checkpoints')}")
    print(f"  - Logs: {os.path.join(experiment_output_dir, 'logs')}")
    print(f"  - Posterior metrics: {os.path.join(experiment_output_dir, 'posterior_metrics')}")
    print(f"  - Visualizations: {os.path.join(experiment_output_dir, 'visualizations')}")
    print(f"  - Analysis: {os.path.join(experiment_output_dir, 'analysis')}")
    
    # Create data module with optimized settings
    print("📊 Creating optimized data module...")
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        use_dynamic_padding=args.use_dynamic_padding,
    )
    
    # Override data loader settings for better performance
    data_module.num_workers = args.num_workers
    data_module.pin_memory = args.pin_memory
    
    # Print data configuration
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max sequence length: {args.max_sequence_length or 'No limit'}")
    print(f"  - Dynamic padding: {args.use_dynamic_padding}")
    print(f"  - Number of workers: {args.num_workers}")
    print(f"  - Pin memory: {args.pin_memory}")
    
    # Set up logging
    logger = TensorBoardLogger(
        save_dir=experiment_output_dir,
        name=f"{config.name}_optimized_batch{args.batch_size}",
        version=None,
    )
    
    # Set up metrics monitoring log directory
    if args.log_dir is None:
        args.log_dir = os.path.join(experiment_output_dir, "posterior_metrics")
    
    # Create model
    print("🧠 Creating model...")
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
        freeze_ratio=0.0,
        pretrained_ckpt=config.pretrained_ckpt,
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
        skip_pretrained_on_resume=args.resume_from_checkpoint is not None,  # 如果从checkpoint恢复，跳过预训练加载
    )
    
    # Apply torch.compile if requested
    if args.compile_model and hasattr(torch, 'compile'):
        print("🔧 Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpointing - only save the best checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(experiment_output_dir, "checkpoints"),
        filename=f"best_{config.name}_optimized_batch{args.batch_size}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,  # Only save the best checkpoint
        save_last=False,  # Don't save the last checkpoint to save space
        verbose=True,
    )
    callbacks.append(checkpoint)
    
    # Set up metrics monitor (optional)
    if not args.disable_metrics_monitoring:
        metrics_monitor = setup_metrics_monitor(args, experiment_output_dir)
        callbacks.append(metrics_monitor)
        
        # Early stopping
        early_stopping = OptimizedEarlyStopping(
            monitor="val_auc",
            mode="max", 
            patience=3,  # Reduced patience for faster training
            verbose=True,
            strict=True,
        )
    else:
        print("⚠️  Posterior metrics monitoring disabled")
        early_stopping = OptimizedEarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            verbose=True,
            strict=True,
        )
    
    callbacks.append(early_stopping)
    
    # Create trainer with optimized settings
    print("⚡ Setting up optimized trainer...")
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",  # Disabled find_unused_parameters
        logger=logger,
        max_epochs=args.max_epochs,
        min_epochs=1,
        precision=args.precision,  # Use command line precision
        callbacks=callbacks,
        profiler=None,  # Disable profiler for better performance
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        val_check_interval=0.1,  # Reduced validation frequency
        limit_val_batches=0.1,   # Reduced validation batches
        accumulate_grad_batches=args.gradient_accumulation_steps,  # Use gradient accumulation
        detect_anomaly=False,    # Turn off anomaly detection
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable model summary for faster startup
        sync_batchnorm=False,  # Disable sync batchnorm for better performance
        use_distributed_sampler=True,
    )
    
    # Print training configuration
    print("\n📋 Optimized training configuration:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Number of devices: {args.devices}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.devices}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - Beta value: {config.beta}")
    print(f"  - Number of workers: {args.num_workers}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Model compilation: {'Enabled' if args.compile_model else 'Disabled'}")
    print(f"  - Metrics monitoring: {'Enabled' if not args.disable_metrics_monitoring else 'Disabled'}")
    if args.resume_from_checkpoint:
        print(f"  - Resuming from checkpoint: {args.resume_from_checkpoint}")
    
    # Start training
    print("\n🎯 Starting optimized training...")
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
                print(f"  - Total steps monitored: {len(monitor.steps_history)}")
                print(f"  - Update frequency: every {monitor.update_frequency} steps")
                print(f"  - Plot frequency: every {monitor.plot_frequency} steps")
                print(f"  - Log directory: {monitor.log_dir}")
        
        # Save final model
        final_model_path = os.path.join(experiment_output_dir, "checkpoints", f"final_{config.name}_optimized_batch{args.batch_size}.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        
        # Save model when interrupted
        interrupt_model_path = os.path.join(experiment_output_dir, "checkpoints", f"interrupted_{config.name}_optimized_batch{args.batch_size}.ckpt")
        trainer.save_checkpoint(interrupt_model_path)
        print(f"💾 Interrupted model saved: {interrupt_model_path}")
        
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        
        # Save model when error occurs
        error_model_path = os.path.join(experiment_output_dir, "checkpoints", f"error_{config.name}_optimized_batch{args.batch_size}.ckpt")
        try:
            trainer.save_checkpoint(error_model_path)
            print(f"💾 Error model saved: {error_model_path}")
        except:
            print("❌ Unable to save error model")
        
        raise
    
    print("\n🎉 Optimized training completed!")

if __name__ == "__main__":
    main()