#!/usr/bin/env python3
"""
Unified SeqSetVAE Training Script
Modes: pretrain (reconstruction+KL only), finetune (freeze all except classifier head)
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
from model import SeqSetVAE, SeqSetVAEPretrain, load_checkpoint_weights
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
            num_workers = min(args.num_workers, 6)  # Optimized for better balance
        elif gpu_memory >= 8:
            num_workers = min(args.num_workers, 4)  # Reduced for stability
        else:
            num_workers = min(args.num_workers, 2)  # Conservative for low memory
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
            new_k = "setvae.setvae." + k[len("set_encoder.") :]
        elif k.startswith("transformer."):
            new_k = k
        elif k.startswith("post_transformer_norm."):
            new_k = k
        elif k.startswith("decoder."):
            new_k = k
        else:
            # Skip optimizer/scheduler/cls_head or unrelated keys
            continue
        new_state[new_k] = v
    return new_state


def main():
    parser = argparse.ArgumentParser(description="Train SeqSetVAE (pretrain/finetune)")
    # Basic training parameters
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
        help="Training mode",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Alias to set mode=finetune",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (optimized for finetune efficiency)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (optimized for larger batch sizes)",
    )
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of devices (auto-detect if not specified)",
    )
    parser.add_argument(
        "--strategy", type=str, default="auto", help="Training strategy"
    )
    parser.add_argument(
        "--precision", type=str, default="16-mixed", help="Training precision"
    )
    parser.add_argument(
        "--compile_model", action="store_true", help="Enable model compilation"
    )
    # Data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr",
        help="Data directory path",
    )
    parser.add_argument(
        "--params_map_path",
        type=str,
        default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
        help="Parameter mapping file path",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv",
        help="Label file path",
    )
    # Advanced training parameters
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Enable deterministic training"
    )
    # Pretrained checkpoint for finetune
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (from pretrain mode)",
    )
    # Output root directory
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/outputs",
        help="Root directory to save logs, checkpoints, and outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Alias of --output_root_dir; if provided, overrides output_root_dir",
    )

    args = parser.parse_args()

    # Alias handling: --finetune sets mode to finetune
    if getattr(args, "finetune", False):
        args.mode = "finetune"

    # Prepare unified output dirs
    base_output_dir = (args.output_dir or args.output_root_dir).rstrip("/")
    model_name = getattr(config, "model_name", getattr(config, "name", "SeqSetVAE"))
    experiment_root = os.path.join(base_output_dir, model_name)
    checkpoints_root_dir = os.path.join(experiment_root, "checkpoints")
    logs_root_dir = os.path.join(experiment_root, "logs")
    analysis_root_dir = os.path.join(experiment_root, "analysis")
    monitor_root_dir = os.path.join(experiment_root, "monitor")
    os.makedirs(checkpoints_root_dir, exist_ok=True)
    os.makedirs(logs_root_dir, exist_ok=True)
    os.makedirs(analysis_root_dir, exist_ok=True)
    os.makedirs(monitor_root_dir, exist_ok=True)

    # Seed/determinism
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Auto devices
    if args.devices is None:
        args.devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Print basic config
    print("🚀 Training Configuration:")
    print(f" - Mode: {args.mode}")
    print(f" - Batch size: {args.batch_size}")
    print(f" - Grad accumulation: {args.gradient_accumulation_steps}")
    print(f" - Max epochs: {args.max_epochs}")
    print(f" - Devices: {args.devices}")
    print(f" - Precision: {args.precision}")
    print(f" - Data dir: {args.data_dir}")

    adaptive_config = get_adaptive_training_config(args)

    print("📊 Setting up data module...")
    # Optimized batch policy: pretrain -> single patient; finetune -> larger batches for efficiency
    if args.mode == "pretrain":
        dm_batch_size = 1
    else:
        # Use larger batch sizes for better GPU utilization in finetune
        dm_batch_size = max(args.batch_size, 4)  # Minimum batch size of 4 for finetune
        if args.batch_size < 4:
            print(
                f"⚠️  Finetune efficiency: increasing batch_size from {args.batch_size} to {dm_batch_size}"
            )
    print(f" - DataModule batch_size: {dm_batch_size}")
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

    # Use optimized config for finetune mode
    if args.mode == "finetune":
        active_config = finetune_config
        print("📋 Using optimized finetune configuration")
    else:
        active_config = config
        print("📋 Using base configuration for pretrain")

    # Model hyperparams from active config
    model_lr = active_config.lr
    model_free_bits = active_config.free_bits
    model_max_beta = active_config.max_beta
    model_beta_warmup_steps = active_config.beta_warmup_steps
    model_gradient_clip_val = active_config.gradient_clip_val

    ff_dim = active_config.ff_dim
    transformer_heads = active_config.transformer_heads
    transformer_layers = active_config.transformer_layers

    print("🧠 Building model...")
    if args.mode == "pretrain":
        print(
            "📋 Using SeqSetVAEPretrain - maintains original design for representation learning"
        )
        model = SeqSetVAEPretrain(
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
        )
        # Optional: initialize from checkpoint (weights only)
        if args.pretrained_ckpt is not None:
            try:
                state = load_checkpoint_weights(args.pretrained_ckpt, device="cpu")
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(
                    f"🔁 Initialized pretrain model from ckpt (weights only). Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize pretrain model from checkpoint: {e}")
        checkpoint_name = "SeqSetVAE_pretrain"
        monitor_metric = "val_loss"
        monitor_mode = "min"
    else:
        print(
            "📋 Using SeqSetVAE - enhanced with modern VAE features and complete freezing for finetune"
        )
        model = SeqSetVAE(
            input_dim=active_config.input_dim,
            reduced_dim=active_config.reduced_dim,
            latent_dim=active_config.latent_dim,
            levels=active_config.levels,
            heads=active_config.heads,
            m=active_config.m,
            beta=active_config.beta,
            lr=model_lr,
            num_classes=active_config.num_classes,
            ff_dim=ff_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            pretrained_ckpt=None,
            w=active_config.w,
            free_bits=model_free_bits,
            warmup_beta=active_config.warmup_beta,
            max_beta=model_max_beta,
            beta_warmup_steps=model_beta_warmup_steps,
            kl_annealing=active_config.kl_annealing,
            use_focal_loss=getattr(active_config, "use_focal_loss", True),
            focal_alpha=active_config.focal_alpha,
            focal_gamma=active_config.focal_gamma,
        )
        checkpoint_name = "SeqSetVAE_finetune"
        monitor_metric = "val_auc"  # Monitor AUC for better finetune performance
        monitor_mode = "max"

        # Load pretrained weights if provided
        # Determine pretrained checkpoint path
        ckpt_path = args.pretrained_ckpt or getattr(active_config, "pretrained_ckpt", None)
        if ckpt_path is None:
            raise ValueError("Finetune requires a pretrained checkpoint. Provide via --pretrained_ckpt or config.pretrained_ckpt.")
        if ckpt_path is not None:
            try:
                state = load_checkpoint_weights(ckpt_path, device="cpu")
                remapped = remap_pretrain_to_finetune_keys(state)
                missing, unexpected = model.load_state_dict(remapped, strict=False)
                print(
                    f"🔁 Loaded pretrained weights with remap. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )
            except Exception as e:
                print(f"⚠️  Failed to load pretrained checkpoint: {e}")

        # Re-initialize classifier head with Xavier for finetune
        model.init_classifier_head_xavier()

        # Freeze everything except classifier head - COMPLETE FREEZE for better stability
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

        model.enable_classification_only_mode(
            cls_head_lr=getattr(active_config, "cls_head_lr", None)
        )
        print(
            "🧊 Finetune freeze applied and backbone set to eval; classification-only loss enabled."
        )
        print(f"   - Frozen parameters: {frozen_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(
            f"   - Trainable ratio: {trainable_params/(frozen_params+trainable_params)*100:.2f}%"
        )

    print("📊 Model Architecture:")
    print(f" - FF dimension: {ff_dim}")
    print(f" - Transformer heads: {transformer_heads}")
    print(f" - Transformer layers: {transformer_layers}")
    if args.mode == "finetune":
        print(
            f" - Classification head LR: {getattr(active_config, 'cls_head_lr', 'default')}"
        )
        print(f" - Using simplified architecture for better efficiency")
        print(" - Finetune computes focal loss only; recon/KL and their logs are disabled")

    print("⚙️ Trainer setup...")
    os.makedirs(checkpoints_root_dir, exist_ok=True)
    os.makedirs(logs_root_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_root_dir, checkpoint_name), exist_ok=True)

    # Callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_root_dir,
        filename=f"{checkpoint_name}_batch{args.batch_size}",
        save_top_k=1,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=max(
            getattr(active_config, "early_stopping_patience", config.early_stopping_patience), 6
        ),  # Ensure minimum patience for finetune
        mode=monitor_mode,
        min_delta=0.001,  # More lenient min_delta for AUC monitoring
        verbose=True,
    )
    callbacks.append(early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Posterior/metrics monitoring -> only enable during pretraining
    if args.mode == "pretrain":
        # Clean previous monitor directory to only keep latest run
        import shutil
        try:
            if os.path.isdir(monitor_root_dir):
                shutil.rmtree(monitor_root_dir)
        except Exception:
            pass
        os.makedirs(monitor_root_dir, exist_ok=True)
        callbacks.append(setup_metrics_monitor(args, monitor_root_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=logs_root_dir,
        name="",
        version=f"batch{args.batch_size}_{timestamp}",
        log_graph=True,
    )

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
        val_check_interval=getattr(active_config, "val_check_interval", 0.25),
        limit_val_batches=getattr(active_config, "limit_val_batches", config.limit_val_batches),
        log_every_n_steps=getattr(active_config, "log_every_n_steps", 25),
        deterministic=args.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        use_distributed_sampler=True if args.devices > 1 else False,
        default_root_dir=experiment_root,
    )

    if adaptive_config["compile_model"]:
        print("🔧 Compiling model for better performance...")
        model = torch.compile(model, mode="default")  # Use default mode for stability

    print("🚀 Start training...")
    trainer.fit(model, data_module)
    print("✅ Training finished!")


if __name__ == "__main__":
    main()
