#!/usr/bin/env python3
"""
SeqSetVAE Finetuning Script - Dedicated for finetuning only
"""

import os
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything
from datetime import datetime

# Import local modules
from model import SeqSetVAE, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorMetricsMonitor
import config
import finetune_config


def get_adaptive_training_config(args):
    """Adaptively adjust training parameters based on device configuration"""
    device_config = getattr(
        config,
        "device_config",
        {
            "cuda_available": torch.cuda.is_available(),
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    )
    if device_config["cuda_available"]:
        if device_config["devices"] > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
            effective_batch_size = (
                args.batch_size
                * args.gradient_accumulation_steps
                * device_config["devices"]
            )
        else:
            strategy = "auto"
            effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:
            num_workers = min(args.num_workers, 6)
        elif gpu_memory >= 8:
            num_workers = min(args.num_workers, 4)
        else:
            num_workers = min(args.num_workers, 2)
        pin_memory = True
        compile_model = args.compile_model and hasattr(torch, "compile")
    else:
        strategy = "auto"
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        num_workers = min(args.num_workers, max(1, cpu_count // 2))
        pin_memory = False
        compile_model = False
        if args.precision != "32":
            print("⚠️  CPU training detected, switching to 32-bit precision")
            args.precision = "32"
    return {
        "strategy": strategy,
        "effective_batch_size": effective_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "compile_model": compile_model,
    }


def setup_metrics_monitor(args, experiment_output_dir):
    """Set up posterior metrics monitoring."""
    monitor = PosteriorMetricsMonitor(
        log_dir=experiment_output_dir,
        update_frequency=100,
        plot_frequency=500,
        window_size=200,
        verbose=False,
    )
    return monitor


def remap_pretrain_to_finetune_keys(state_dict):
    """Remap keys from SeqSetVAEPretrain -> SeqSetVAE for compatible loading."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("set_encoder."):
            # Map set_encoder to setvae.setvae
            new_k = "setvae.setvae." + k[len("set_encoder.") :]
            new_state[new_k] = v
        elif k.startswith("transformer."):
            # Direct mapping for transformer
            new_state[k] = v
        elif k.startswith("post_transformer_norm."):
            # Direct mapping for post_transformer_norm
            new_state[k] = v
        elif k.startswith("decoder."):
            # Direct mapping for decoder
            new_state[k] = v
        # Skip other keys that don't have direct mappings
    return new_state


def main():
    parser = argparse.ArgumentParser(description="SeqSetVAE Finetuning")
    
    # Basic training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (uses config default if None)")
    parser.add_argument("--cls_head_lr", type=float, default=None, help="Classifier head learning rate")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--compile_model", action="store_true", help="Compile model for speed")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--params_map_path",
        type=str,
        default=None,
        help="Path to parameter mapping file"
    )
    parser.add_argument(
        "--label_path",
        type=str,
        required=True,
        help="Path to labels file (required for finetuning)"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to pretrained checkpoint (required for finetuning)"
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/outputs",
        help="Root directory to save logs, checkpoints, and outputs"
    )
    
    # Model arguments
    parser.add_argument(
        "--vae_fusion_method",
        type=str,
        choices=["simple_concat", "enhanced_concat"],
        default="simple_concat",
        help="VAE feature fusion method"
    )
    parser.add_argument(
        "--estimate_uncertainty",
        action="store_true",
        default=False,
        help="Enable uncertainty estimation"
    )
    
    # Training control
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic training")
    
    args = parser.parse_args()
    
    # Setup output directories
    base_output_dir = args.output_root_dir.rstrip("/")
    model_name = getattr(config, "model_name", getattr(config, "name", "SeqSetVAE"))
    experiment_root = os.path.join(base_output_dir, model_name, "finetune")
    checkpoints_root_dir = os.path.join(experiment_root, "checkpoints")
    logs_root_dir = os.path.join(experiment_root, "logs")
    analysis_root_dir = os.path.join(experiment_root, "analysis")
    monitor_root_dir = os.path.join(experiment_root, "monitor")
    
    # Create directories
    for dir_path in [checkpoints_root_dir, logs_root_dir, analysis_root_dir, monitor_root_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Seed/determinism
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Auto devices
    if args.devices is None:
        args.devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print("🚀 Finetuning Configuration:")
    print(f" - Batch size: {args.batch_size}")
    print(f" - Grad accumulation: {args.gradient_accumulation_steps}")
    print(f" - Max epochs: {args.max_epochs}")
    print(f" - Devices: {args.devices}")
    print(f" - Precision: {args.precision}")
    print(f" - Data dir: {args.data_dir}")
    print(f" - Label path: {args.label_path}")
    print(f" - Pretrained checkpoint: {args.pretrained_ckpt}")
    
    adaptive_config = get_adaptive_training_config(args)
    
    print("📊 Setting up data module...")
    # Use larger batch sizes for better GPU utilization in finetune
    dm_batch_size = max(args.batch_size, 4)  # Minimum batch size of 4 for finetune
    if args.batch_size < 4:
        print(f"⚠️  Increasing batch_size from {args.batch_size} to {dm_batch_size} for efficiency")
    
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=dm_batch_size,
        max_sequence_length=None,
        use_dynamic_padding=True,
        num_workers=adaptive_config["num_workers"],
        pin_memory=adaptive_config["pin_memory"],
    )
    
    # Use finetune config
    active_config = finetune_config
    print("📋 Using optimized finetune configuration")
    
    # Model hyperparams from finetune config
    model_lr = args.lr if args.lr is not None else active_config.lr
    model_free_bits = active_config.free_bits
    model_max_beta = active_config.max_beta
    model_beta_warmup_steps = active_config.beta_warmup_steps
    model_gradient_clip_val = active_config.gradient_clip_val
    
    ff_dim = active_config.ff_dim
    transformer_heads = active_config.transformer_heads
    transformer_layers = active_config.transformer_layers
    
    print("🧠 Building finetuning model...")
    model = SeqSetVAE(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=model_lr,
        ff_dim=ff_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        warmup_beta=config.warmup_beta,
        max_beta=model_max_beta,
        beta_warmup_steps=model_beta_warmup_steps,
        free_bits=model_free_bits,
        transformer_dropout=config.transformer_dropout,
        vae_fusion_method=args.vae_fusion_method,
        estimate_uncertainty=args.estimate_uncertainty,
        num_classes=getattr(config, "num_classes", 2),
    )
    
    # Load pretrained weights
    print(f"🔁 Loading pretrained weights from {args.pretrained_ckpt}")
    try:
        state = load_checkpoint_weights(args.pretrained_ckpt, device="cpu")
        # Remap keys for compatibility
        remapped_state = remap_pretrain_to_finetune_keys(state)
        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        print(f"   Missing keys: {len(missing)}")
        print(f"   Unexpected keys: {len(unexpected)}")
        if missing:
            print("   Missing keys (will be randomly initialized):")
            for key in missing[:10]:  # Show first 10
                print(f"     - {key}")
            if len(missing) > 10:
                print(f"     ... and {len(missing) - 10} more")
    except Exception as e:
        print(f"⚠️  Failed to load pretrained weights: {e}")
        raise
    
    # Freeze all parameters except classifier head
    print("🧊 Freezing backbone parameters...")
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if name.startswith("cls_head"):
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"   ✅ Trainable: {name} ({param.numel():,} params)")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    # Enable classification-only mode
    cls_head_lr = args.cls_head_lr or getattr(active_config, "cls_head_lr", None)
    model.enable_classification_only_mode(cls_head_lr=cls_head_lr)
    
    print("🧊 Finetune freeze applied and backbone set to eval; classification-only loss enabled.")
    print(f"   - Frozen parameters: {frozen_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Trainable ratio: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")
    
    print("📊 Model Architecture:")
    print(f" - FF dimension: {ff_dim}")
    print(f" - Transformer heads: {transformer_heads}")
    print(f" - Transformer layers: {transformer_layers}")
    print(f" - Learning rate: {model_lr}")
    print(f" - Classification head LR: {cls_head_lr or 'default'}")
    print(f" - VAE fusion method: {args.vae_fusion_method}")
    print(f" - Uncertainty estimation: {args.estimate_uncertainty}")
    
    print("⚙️ Setting up trainer...")
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_root_dir,
        filename="SeqSetVAE_finetune_batch{batch_size}".format(batch_size=args.batch_size),
        save_top_k=1,
        monitor="FT_val_auc",
        mode="max",
        save_last=True,
        save_on_train_epoch_end=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="FT_val_auc",
        patience=max(config.early_stopping_patience, 6),
        mode="max",
        min_delta=0.001,
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Posterior metrics monitor
    import shutil
    try:
        if os.path.isdir(monitor_root_dir):
            shutil.rmtree(monitor_root_dir)
    except Exception:
        pass
    os.makedirs(monitor_root_dir, exist_ok=True)
    callbacks.append(setup_metrics_monitor(args, monitor_root_dir))
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=logs_root_dir,
        name="",
        version=f"finetune_batch{args.batch_size}_{timestamp}",
        log_graph=True,
    )
    
    # Trainer
    strategy = adaptive_config["strategy"]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=model_gradient_clip_val,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=0.25,
        limit_val_batches=config.limit_val_batches,
        log_every_n_steps=25,
        deterministic=args.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        use_distributed_sampler=True if args.devices > 1 else False,
        default_root_dir=experiment_root,
    )
    
    # Compile model if requested
    if adaptive_config["compile_model"]:
        print("🔧 Compiling model for better performance...")
        model = torch.compile(model, mode="default")
    
    print("🚀 Starting finetuning...")
    trainer.fit(model, data_module)
    print("✅ Finetuning finished!")
    
    # Print checkpoint info and final metrics
    if checkpoint_callback.best_model_path:
        print(f"💾 Best checkpoint saved to: {checkpoint_callback.best_model_path}")
        print(f"🎯 Best validation AUC: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()