#!/usr/bin/env python3
"""
Optimized SeqSetVAE Training Script - Performance Enhanced
解决训练速度慢和性能下降问题的优化版本
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

# Import optimized modules
from model_optimized import OptimizedSeqSetVAE
from model import SeqSetVAEPretrain, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
from posterior_collapse_detector import PosteriorMetricsMonitor
import finetune_config_optimized as config


def main():
    parser = argparse.ArgumentParser(description="Optimized SeqSetVAE Training")
    
    # 使用优化的默认参数
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune"], default="finetune", help="Training mode")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps, help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=config.max_epochs, help="Maximum epochs")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--compile_model", action="store_true", default=config.enable_torch_compile, help="Enable model compilation")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr", help="Data directory path")
    parser.add_argument("--params_map_path", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv", help="Parameter mapping file path")
    parser.add_argument("--label_path", type=str, default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv", help="Label file path")
    
    # Advanced parameters
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    
    # Pretrained checkpoint
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to pretrained checkpoint (REQUIRED for finetune)")
    
    # Output directory
    parser.add_argument("--output_root_dir", type=str, default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs", help="Root output directory")

    args = parser.parse_args()

    # 验证微调必需的预训练检查点
    if args.mode == "finetune" and not args.pretrained_ckpt:
        raise ValueError("❌ Finetune mode requires --pretrained_ckpt argument!")

    # 设置输出目录
    model_name = "OptimizedSeqSetVAE"
    experiment_root = os.path.join(args.output_root_dir, model_name)
    checkpoints_dir = os.path.join(experiment_root, 'checkpoints')
    logs_dir = os.path.join(experiment_root, 'logs')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 设置随机种子
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 自动检测设备
    if args.devices is None:
        args.devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    print("🚀 Optimized Training Configuration:")
    print(f" - Mode: {args.mode}")
    print(f" - Batch size: {args.batch_size}")
    print(f" - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f" - Max epochs: {args.max_epochs}")
    print(f" - Devices: {args.devices}")
    print(f" - Classification head LR: {config.cls_head_lr}")

    # 优化的数据模块配置
    print("📊 Setting up optimized data module...")
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        max_sequence_length=None,
        use_dynamic_padding=True,
        num_workers=args.num_workers,
        pin_memory=config.pin_memory,
    )

    # 创建优化的模型
    print("🧠 Building optimized model...")
    if args.mode == 'pretrain':
        # 使用原始预训练模型
        model = SeqSetVAEPretrain(
            input_dim=config.input_dim,
            reduced_dim=config.reduced_dim,
            latent_dim=config.latent_dim,
            levels=config.levels,
            heads=config.heads,
            m=config.m,
            beta=config.beta,
            lr=config.lr,
            ff_dim=config.ff_dim,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
            warmup_beta=config.warmup_beta,
            max_beta=config.max_beta,
            beta_warmup_steps=config.beta_warmup_steps,
            free_bits=config.free_bits,
            transformer_dropout=config.transformer_dropout,
        )
        checkpoint_name = "OptimizedSeqSetVAE_pretrain"
        monitor_metric = 'val_loss'
        monitor_mode = 'min'
    else:
        # 使用优化的微调模型
        model = OptimizedSeqSetVAE(
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
            pretrained_ckpt=args.pretrained_ckpt,
            w=config.w,
            free_bits=config.free_bits,
            warmup_beta=config.warmup_beta,
            max_beta=config.max_beta,
            beta_warmup_steps=config.beta_warmup_steps,
            kl_annealing=config.kl_annealing,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            cls_head_lr=config.cls_head_lr,
            simplified_cls_head=config.simplified_cls_head,
        )
        checkpoint_name = "OptimizedSeqSetVAE_finetune"
        monitor_metric = config.monitor_metric
        monitor_mode = config.monitor_mode

        # 初始化分类头
        model.init_classifier_head_xavier()

        # 应用优化的冻结策略
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
        
        model.enable_classification_only_mode(cls_head_lr=config.cls_head_lr)
        print("🧊 Optimized freeze applied:")
        print(f"   - Frozen: {frozen_params:,}")
        print(f"   - Trainable: {trainable_params:,}")
        print(f"   - Ratio: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")

    # 优化的回调函数
    callbacks = []
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=f"{checkpoint_name}_optimized",
        save_top_k=3,  # 保存更多检查点
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    callbacks.append(checkpoint_callback)

    # 早停回调
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=config.early_stopping_patience,
        mode=monitor_mode,
        min_delta=0.001,  # 更宽松的最小改进
        verbose=True,
    )
    callbacks.append(early_stopping)

    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="epoch")  # 改为epoch级别
    callbacks.append(lr_monitor)

    # TensorBoard日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="",
        version=f"optimized_{timestamp}",
        log_graph=False,  # 禁用图记录以提高速度
    )

    # 优化的训练器配置
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        strategy="auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=config.val_check_interval,
        limit_val_batches=config.limit_val_batches,
        log_every_n_steps=config.log_every_n_steps,
        deterministic=args.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        default_root_dir=experiment_root,
        # 性能优化设置
        sync_batchnorm=False,  # 单GPU时禁用
        benchmark=True,  # 启用cudnn benchmark
    )

    # 启用模型编译（如果支持）
    if args.compile_model and hasattr(torch, 'compile'):
        print("🔧 Compiling model for better performance...")
        model = torch.compile(model, mode='default')

    print("🚀 Starting optimized training...")
    trainer.fit(model, data_module)
    print("✅ Training finished!")

    # 输出最终结果
    if hasattr(trainer.callback_metrics, 'val_auc'):
        final_auc = trainer.callback_metrics.get('val_auc', 0.0)
        final_auprc = trainer.callback_metrics.get('val_auprc', 0.0)
        print(f"🎯 Final Results:")
        print(f"   - Val AUC: {final_auc:.4f}")
        print(f"   - Val AUPRC: {final_auprc:.4f}")


if __name__ == "__main__":
    main()