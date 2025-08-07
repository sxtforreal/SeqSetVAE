import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorCollapseDetector
import config
import os
import argparse
from datetime import datetime

def setup_collapse_detector(args):
    """Set up posterior collapse detector"""
    
    # Adjust detection parameters based on data size
    if args.fast_detection:
        # Fast detection mode - more frequent checks, more sensitive thresholds
        detector = PosteriorCollapseDetector(
            kl_threshold=0.005,          # Stricter KL threshold
            var_threshold=0.05,          # Stricter variance threshold
            active_units_threshold=0.15, # Stricter active units threshold
            
            window_size=50,              # Smaller window, faster response
            check_frequency=20,          # Check every 20 steps
            
            early_stop_patience=100,     # Faster early stopping
            auto_save_on_collapse=True,
            
            log_dir=args.log_dir,
            plot_frequency=200,          # More frequent plotting
            verbose=True,
        )
    else:
        # Standard detection mode
        detector = PosteriorCollapseDetector(
            kl_threshold=0.01,
            var_threshold=0.1,
            active_units_threshold=0.1,
            
            window_size=100,
            check_frequency=50,
            
            early_stop_patience=200,
            auto_save_on_collapse=True,
            
            log_dir=args.log_dir,
            plot_frequency=500,
            verbose=True,
        )
    
    print(f"🔍 Collapse detector setup complete:")
    print(f"  - Detection mode: {'Fast' if args.fast_detection else 'Standard'}")
    print(f"  - KL threshold: {detector.kl_threshold}")
    print(f"  - Check frequency: every {detector.check_frequency} steps")
    print(f"  - Log directory: {detector.log_dir}")
    
    return detector

class CollapseAwareEarlyStopping(EarlyStopping):
    """Early stopping callback integrated with collapse detection"""
    
    def __init__(self, collapse_detector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collapse_detector = collapse_detector
        
    def _should_stop_early(self, trainer, pl_module):
        # Force early stopping if persistent collapse is detected
        if (self.collapse_detector.collapse_detected and 
            self.collapse_detector.collapse_consecutive_steps >= 50):
            
            print(f"\n🛑 Forced early stopping due to persistent posterior collapse!")
            return True
            
        # Otherwise use standard early stopping logic
        return super()._should_stop_early(trainer, pl_module)

def main():
    parser = argparse.ArgumentParser(description='Train SeqSetVAE with posterior collapse detection')
    
    # Basic training parameters
    parser.add_argument('--max_epochs', type=int, default=config.max_epochs, 
                       help='Maximum training epochs')
    parser.add_argument('--devices', type=int, default=config.devices,
                       help='Number of GPUs to use')
    
    # Collapse detection parameters
    parser.add_argument('--fast_detection', action='store_true',
                       help='Enable fast detection mode (more frequent checks, more sensitive thresholds)')
    parser.add_argument('--log_dir', type=str, 
                       default=None,  # Will be set to match main log directory
                       help='Collapse detection log directory (default: same as main logs)')
    parser.add_argument('--disable_collapse_detection', action='store_true',
                       help='Disable posterior collapse detection')
    
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
    
    print("🚀 Starting SeqSetVAE training with posterior collapse detection")
    print("=" * 60)
    
    # Set random seed
    seed_everything(0, workers=True)
    
    # Prepare data
    print("📊 Preparing data...")
    data_module = SeqSetVAEDataModule(args.data_dir, args.params_map_path, args.label_path)
    data_module.setup()
    print(f"  - Training data: {len(data_module.train_dataset)}")
    print(f"  - Validation data: {len(data_module.val_dataset)}")
    print(f"  - Test data: {len(data_module.test_dataset)}")
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name=f"{config.name}_with_collapse_detection",
    )
    
    # If log_dir not specified, use the same directory as main logs
    if args.log_dir is None:
        # Create a subdirectory for collapse logs within the main log directory
        args.log_dir = os.path.join(logger.log_dir, "collapse_detection")
    
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
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=f"best_{config.name}_collapse_aware",
        save_weights_only=True,
        save_last=True,  # Save last checkpoint
        every_n_train_steps=config.ckpt_every_n_steps,
        monitor="val_auc",
        mode="max",
        save_top_k=3,  # Save top 3 best models
        enable_version_counter=True,
    )
    callbacks.append(checkpoint)
    
    # Set up collapse detector
    if not args.disable_collapse_detection:
        collapse_detector = setup_collapse_detector(args)
        callbacks.append(collapse_detector)
        
        # Early stopping integrated with collapse detection
        early_stopping = CollapseAwareEarlyStopping(
            collapse_detector=collapse_detector,
            monitor="val_auc",
            mode="max", 
            patience=5,  # Increase patience since we have collapse detection
            verbose=True,
            strict=True,
        )
    else:
        print("⚠️  Posterior collapse detection disabled")
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
    print(f"  - Precision: {config.precision}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - Beta value: {config.beta}")
    print(f"  - Collapse detection: {'Enabled' if not args.disable_collapse_detection else 'Disabled'}")
    
    # Start training
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    try:
        trainer.fit(model, data_module)
        
        # Training completion summary
        print("\n✅ Training completed!")
        
        if not args.disable_collapse_detection:
            print("\n📊 Collapse detection summary:")
            detector = None
            for callback in trainer.callbacks:
                if isinstance(callback, PosteriorCollapseDetector):
                    detector = callback
                    break
                    
            if detector:
                print(f"  - Total checks: {detector.collapse_stats['total_checks']}")
                print(f"  - Warning count: {detector.collapse_stats['warnings_issued']}")
                print(f"  - Collapse detected: {'Yes' if detector.collapse_detected else 'No'}")
                if detector.collapse_step:
                    print(f"  - Collapse occurrence step: {detector.collapse_step}")
                print(f"  - Detailed log: {detector.log_file}")
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "checkpoints", f"final_{config.name}.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        
        # Save model when interrupted
        interrupt_model_path = os.path.join(args.output_dir, "checkpoints", f"interrupted_{config.name}.ckpt")
        trainer.save_checkpoint(interrupt_model_path)
        print(f"💾 Interrupted model saved: {interrupt_model_path}")
        
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        
        # Save model when error occurs
        error_model_path = os.path.join(args.output_dir, "checkpoints", f"error_{config.name}.ckpt")
        try:
            trainer.save_checkpoint(error_model_path)
            print(f"💾 Error model saved: {error_model_path}")
        except:
            print("❌ Unable to save error model")
        
        raise
    
    print("\n🎉 Program execution completed!")

if __name__ == "__main__":
    main()