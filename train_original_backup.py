"""
Original training script for SeqSetVAE model (backup version).

This script provides a simple training setup for the Sequential SetVAE model
without posterior collapse detection. It serves as a baseline implementation
for comparison with the enhanced version that includes collapse monitoring.
"""

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
import config

if __name__ == "__main__":
    # Set random seed for reproducibility across all random number generators
    seed_everything(0, workers=True)

    # Data paths configuration
    saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"        # Patient data directory
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"   # Normalization statistics
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"           # Patient outcome labels
    
    # Initialize data module for patient-level medical data
    data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
    data_module.setup()
    
    # Print dataset statistics
    print("Number of training data:", len(data_module.train_dataset))
    print("Number of validation data:", len(data_module.val_dataset))
    print("Number of test data:", len(data_module.test_dataset))

    # Set up TensorBoard logger for experiment tracking
    logger = TensorBoardLogger(
        save_dir="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/logs",
        name=f"{config.name}",  # Use experiment name from config
    )

    # Initialize the Sequential SetVAE model with configuration parameters
    model = SeqSetVAE(
        input_dim=config.input_dim,                    # Medical variable embedding dimension
        reduced_dim=config.reduced_dim,                # Dimension after reduction layer
        latent_dim=config.latent_dim,                  # VAE latent space dimension
        levels=config.levels,                          # Number of SetVAE encoder/decoder levels
        heads=config.heads,                            # Multi-head attention heads
        m=config.m,                                    # Inducing points in ISAB
        beta=config.beta,                              # KL divergence weight
        lr=config.lr,                                  # Learning rate
        num_classes=config.num_classes,                # Number of classification classes
        ff_dim=config.ff_dim,                          # Transformer feed-forward dimension
        transformer_heads=config.transformer_heads,    # Transformer attention heads
        transformer_layers=config.transformer_layers,  # Number of transformer layers
        freeze_ratio=0.0,                             # Don't freeze pretrained parameters, let model adapt
        pretrained_ckpt=config.pretrained_ckpt,       # Path to pretrained SetVAE weights
        w=config.w,                                   # Classification loss weight
        free_bits=config.free_bits,                   # Free bits for KL regularization
        warmup_beta=config.warmup_beta,               # Enable beta warmup
        max_beta=config.max_beta,                     # Maximum beta value
        beta_warmup_steps=config.beta_warmup_steps,   # Beta warmup duration
        kl_annealing=config.kl_annealing,             # Enable KL annealing
    )

    # Configure model checkpointing to save best model based on validation AUC
    checkpoint = ModelCheckpoint(
        dirpath="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints",
        filename=f"best_{config.name}",               # Checkpoint filename
        save_weights_only=True,                       # Save only model weights (not optimizer state)
        save_last=False,                              # Don't save last checkpoint
        every_n_train_steps=config.ckpt_every_n_steps, # Checkpoint frequency
        monitor="val_auc",                            # Metric to monitor for best model
        mode="max",                                   # Higher AUC is better
        save_top_k=1,                                 # Keep only the best checkpoint
        enable_version_counter=False,                 # Don't add version numbers to filenames
    )

    # Configure early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_auc",                            # Monitor validation AUC
        mode="max",                                   # Higher is better
        patience=3,                                   # Stop after 3 epochs without improvement
        verbose=True,                                 # Print early stopping messages
        strict=True,                                  # Crash if monitored metric is not available
    )

    # Initialize PyTorch Lightning trainer with distributed training support
    trainer = pl.Trainer(
        accelerator=config.accelerator,               # Use GPU acceleration
        devices=config.devices,                       # Number of devices to use
        strategy=DDPStrategy(find_unused_parameters=True), # Distributed data parallel strategy
        logger=logger,                                # TensorBoard logger
        max_epochs=config.max_epochs,                 # Maximum training epochs
        min_epochs=1,                                 # Minimum training epochs
        precision=config.precision,                   # Mixed precision training
        callbacks=[
            checkpoint,                               # Model checkpointing callback
            early_stopping,                           # Early stopping callback
        ],
        profiler="advanced",                          # Advanced profiler for performance analysis
        log_every_n_steps=config.log_every_n_steps, # Logging frequency
        gradient_clip_val=config.gradient_clip_val,  # Gradient clipping value from config
        gradient_clip_algorithm="norm",              # Use norm clipping instead of value clipping
        val_check_interval=0.04,                     # Validation check frequency (4% of training epoch)
        limit_val_batches=0.1,                       # Use 10% of validation data per check
        accumulate_grad_batches=1,                   # Gradient accumulation batches
        detect_anomaly=True,                         # Detect anomalies for debugging
    )

    # Start model training
    trainer.fit(model, data_module)
