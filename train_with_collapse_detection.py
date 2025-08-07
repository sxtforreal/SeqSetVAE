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

def setup_metrics_monitor(args):
    """Set up posterior metrics monitor"""
    
    # Adjust monitoring parameters based on data size
    if args.fast_detection:
        # Fast monitoring mode - more frequent updates
        monitor = PosteriorMetricsMonitor(
            update_frequency=20,          # Update every 20 steps
            plot_frequency=200,           # Save plot every 200 steps
            window_size=100,              # History window size
            
            log_dir=args.log_dir,
            verbose=True,
        )
    else:
        # Standard monitoring mode
        monitor = PosteriorMetricsMonitor(
            update_frequency=50,          # Update every 50 steps
            plot_frequency=500,           # Save plot every 500 steps
            window_size=100,              # History window size
            
            log_dir=args.log_dir,
            verbose=True,
        )
    
    print(f"📊 Posterior metrics monitor setup complete:")
    print(f"  - Monitoring mode: {'Fast' if args.fast_detection else 'Standard'}")
    print(f"  - Update frequency: every {monitor.update_frequency} steps")
    print(f"  - Plot frequency: every {monitor.plot_frequency} steps")
    print(f"  - Log directory: {monitor.log_dir}")
    
    return monitor

class CollapseAwareEarlyStopping(EarlyStopping):
    """Early stopping callback integrated with metrics monitoring"""
    
    def __init__(self, metrics_monitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_monitor = metrics_monitor
        
    def _should_stop_early(self, trainer, pl_module):
        # Use standard early stopping logic only
        return super()._should_stop_early(trainer, pl_module)

def main():
    parser = argparse.ArgumentParser(description='Train SeqSetVAE with posterior metrics monitoring')
    
    # Basic training parameters
    parser.add_argument('--max_epochs', type=int, default=config.max_epochs, 
                       help='Maximum training epochs')
    parser.add_argument('--devices', type=int, default=config.devices,
                       help='Number of GPUs to use')
    
    # Batch training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training (1 for single-patient, >1 for multi-patient)')
    parser.add_argument('--max_sequence_length', type=int, default=None,
                       help='Maximum sequence length to truncate (None for no limit)')
    parser.add_argument('--use_dynamic_padding', action='store_true', default=True,
                       help='Use dynamic padding for batch training')
    
    # Metrics monitoring parameters
    parser.add_argument('--fast_detection', action='store_true',
                       help='Enable fast monitoring mode (more frequent updates)')
    parser.add_argument('--disable_metrics_monitoring', action='store_true',
                       help='Disable posterior metrics monitoring')
    parser.add_argument('--log_dir', type=str, 
                       default=None,  # Will be set to match main log directory
                       help='Metrics monitoring log directory (default: same as main logs)')
    
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
    
    print("🚀 Starting SeqSetVAE training with posterior metrics monitoring")
    print("=" * 60)
    
    # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Create data module
    print("📊 Creating data module...")
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        use_dynamic_padding=args.use_dynamic_padding,
    )
    
    # Print data configuration
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max sequence length: {args.max_sequence_length or 'No limit'}")
    print(f"  - Dynamic padding: {args.use_dynamic_padding}")
    
    # Set up logging
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=f"{config.name}_with_metrics_monitoring_batch{args.batch_size}",
        version=None,  # Auto-increment version
    )
    
    # Set up metrics monitoring log directory
    if args.log_dir is None:
        args.log_dir = os.path.join(logger.log_dir, "posterior_metrics")
    
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
    )
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=f"best_{config.name}_metrics_monitoring_batch{args.batch_size}",
        monitor="val_auc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint)
    
    # Set up metrics monitor
    if not args.disable_metrics_monitoring:
        metrics_monitor = setup_metrics_monitor(args)
        callbacks.append(metrics_monitor)
        
        # Early stopping integrated with metrics monitoring
        early_stopping = CollapseAwareEarlyStopping(
            metrics_monitor=metrics_monitor, # Pass the monitor itself
            monitor="val_auc",
            mode="max", 
            patience=5,  # Increase patience since we have metrics monitoring
            verbose=True,
            strict=True,
        )
    else:
        print("⚠️  Posterior metrics monitoring disabled")
        early_stopping = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            verbose=True,
            strict=True,
        )
    
    callbacks.append(early_stopping)
    
    # Create trainer
    print("⚡ Setting up trainer...")
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto",
        logger=logger,
        max_epochs=args.max_epochs,
        min_epochs=1,
        precision=config.precision,
        callbacks=callbacks,
        profiler="simple",  # Use simple profiler to reduce overhead
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        val_check_interval=0.05,  # Slightly increase validation frequency
        limit_val_batches=0.2,   # Increase number of validation batches
        accumulate_grad_batches=1,
        detect_anomaly=False,    # Turn off anomaly detection for better performance
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Print training configuration
    print("\n📋 Training configuration:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Number of devices: {args.devices}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Precision: {config.precision}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - Beta value: {config.beta}")
    print(f"  - Metrics monitoring: {'Enabled' if not args.disable_metrics_monitoring else 'Disabled'}")
    
    # Start training
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    try:
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
        final_model_path = os.path.join(args.output_dir, "checkpoints", f"final_{config.name}_batch{args.batch_size}.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        
        # Save model when interrupted
        interrupt_model_path = os.path.join(args.output_dir, "checkpoints", f"interrupted_{config.name}_batch{args.batch_size}.ckpt")
        trainer.save_checkpoint(interrupt_model_path)
        print(f"💾 Interrupted model saved: {interrupt_model_path}")
        
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        
        # Save model when error occurs
        error_model_path = os.path.join(args.output_dir, "checkpoints", f"error_{config.name}_batch{args.batch_size}.ckpt")
        try:
            trainer.save_checkpoint(error_model_path)
            print(f"💾 Error model saved: {error_model_path}")
        except:
            print("❌ Unable to save error model")
        
        raise
    
    print("\n🎉 Program execution completed!")

if __name__ == "__main__":
    main()