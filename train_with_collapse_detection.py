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
    """设置后验塌缩检测器"""
    
    # 根据数据大小调整检测参数
    if args.fast_detection:
        # 快速检测模式 - 更频繁的检查，更敏感的阈值
        detector = PosteriorCollapseDetector(
            kl_threshold=0.005,          # 更严格的KL阈值
            var_threshold=0.05,          # 更严格的方差阈值
            active_units_threshold=0.15, # 更严格的激活单元阈值
            
            window_size=50,              # 较小的窗口，更快响应
            check_frequency=20,          # 每20步检查一次
            
            early_stop_patience=100,     # 更快的早期停止
            auto_save_on_collapse=True,
            
            log_dir=args.log_dir,
            plot_frequency=200,          # 更频繁的绘图
            verbose=True,
        )
    else:
        # 标准检测模式
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
    
    print(f"🔍 塌缩检测器设置完成:")
    print(f"  - 检测模式: {'快速' if args.fast_detection else '标准'}")
    print(f"  - KL阈值: {detector.kl_threshold}")
    print(f"  - 检查频率: 每{detector.check_frequency}步")
    print(f"  - 日志目录: {detector.log_dir}")
    
    return detector

class CollapseAwareEarlyStopping(EarlyStopping):
    """集成塌缩检测的早期停止回调"""
    
    def __init__(self, collapse_detector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collapse_detector = collapse_detector
        
    def _should_stop_early(self, trainer, pl_module):
        # 如果检测到持续塌缩，强制早期停止
        if (self.collapse_detector.collapse_detected and 
            self.collapse_detector.collapse_consecutive_steps >= 50):
            
            print(f"\n🛑 由于检测到持续后验塌缩，强制早期停止训练！")
            return True
            
        # 否则使用标准早期停止逻辑
        return super()._should_stop_early(trainer, pl_module)

def main():
    parser = argparse.ArgumentParser(description='训练SeqSetVAE并检测后验塌缩')
    
    # 基本训练参数
    parser.add_argument('--max_epochs', type=int, default=config.max_epochs, 
                       help='最大训练轮数')
    parser.add_argument('--devices', type=int, default=config.devices,
                       help='使用的GPU数量')
    
    # 塌缩检测参数
    parser.add_argument('--fast_detection', action='store_true',
                       help='启用快速检测模式（更频繁检查，更敏感阈值）')
    parser.add_argument('--log_dir', type=str, 
                       default=f"./collapse_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='塌缩检测日志目录')
    parser.add_argument('--disable_collapse_detection', action='store_true',
                       help='禁用后验塌缩检测')
    
    # 数据路径参数
    parser.add_argument('--data_dir', type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr",
                       help='数据目录路径')
    parser.add_argument('--params_map_path', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
                       help='参数映射文件路径')
    parser.add_argument('--label_path', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv", 
                       help='标签文件路径')
    
    # 输出路径参数
    parser.add_argument('--output_dir', type=str,
                       default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs",
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    print("🚀 开始训练SeqSetVAE with 后验塌缩检测")
    print("=" * 60)
    
    # 设置随机种子
    seed_everything(0, workers=True)
    
    # 准备数据
    print("📊 准备数据...")
    data_module = SeqSetVAEDataModule(args.data_dir, args.params_map_path, args.label_path)
    data_module.setup()
    print(f"  - 训练数据: {len(data_module.train_dataset)}")
    print(f"  - 验证数据: {len(data_module.val_dataset)}")
    print(f"  - 测试数据: {len(data_module.test_dataset)}")
    
    # 设置日志记录器
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name=f"{config.name}_with_collapse_detection",
    )
    
    # 创建模型
    print("🧠 创建模型...")
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
    
    # 设置回调函数
    callbacks = []
    
    # 模型检查点
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=f"best_{config.name}_collapse_aware",
        save_weights_only=True,
        save_last=True,  # 保存最后一个检查点
        every_n_train_steps=config.ckpt_every_n_steps,
        monitor="val_auc",
        mode="max",
        save_top_k=3,  # 保存前3个最佳模型
        enable_version_counter=True,
    )
    callbacks.append(checkpoint)
    
    # 设置塌缩检测器
    if not args.disable_collapse_detection:
        collapse_detector = setup_collapse_detector(args)
        callbacks.append(collapse_detector)
        
        # 集成塌缩检测的早期停止
        early_stopping = CollapseAwareEarlyStopping(
            collapse_detector=collapse_detector,
            monitor="val_auc",
            mode="max", 
            patience=5,  # 增加耐心值，因为有塌缩检测
            verbose=True,
            strict=True,
        )
    else:
        print("⚠️  后验塌缩检测已禁用")
        early_stopping = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            verbose=True,
            strict=True,
        )
    
    callbacks.append(early_stopping)
    
    # 创建训练器
    print("⚡ 设置训练器...")
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto",
        logger=logger,
        max_epochs=args.max_epochs,
        min_epochs=1,
        precision=config.precision,
        callbacks=callbacks,
        profiler="simple",  # 使用简单profiler减少开销
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        val_check_interval=0.05,  # 稍微增加验证频率
        limit_val_batches=0.2,   # 增加验证批次数量
        accumulate_grad_batches=1,
        detect_anomaly=False,    # 关闭异常检测以提高性能
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 打印训练配置
    print("\n📋 训练配置:")
    print(f"  - 最大轮数: {args.max_epochs}")
    print(f"  - 设备数量: {args.devices}")
    print(f"  - 精度: {config.precision}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - Beta值: {config.beta}")
    print(f"  - 塌缩检测: {'启用' if not args.disable_collapse_detection else '禁用'}")
    
    # 开始训练
    print("\n🎯 开始训练...")
    print("=" * 60)
    
    try:
        trainer.fit(model, data_module)
        
        # 训练完成后的总结
        print("\n✅ 训练完成！")
        
        if not args.disable_collapse_detection:
            print("\n📊 塌缩检测总结:")
            detector = None
            for callback in trainer.callbacks:
                if isinstance(callback, PosteriorCollapseDetector):
                    detector = callback
                    break
                    
            if detector:
                print(f"  - 总检查次数: {detector.collapse_stats['total_checks']}")
                print(f"  - 警告次数: {detector.collapse_stats['warnings_issued']}")
                print(f"  - 检测到塌缩: {'是' if detector.collapse_detected else '否'}")
                if detector.collapse_step:
                    print(f"  - 塌缩发生步数: {detector.collapse_step}")
                print(f"  - 详细日志: {detector.log_file}")
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, "checkpoints", f"final_{config.name}.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"💾 最终模型已保存: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        
        # 保存中断时的模型
        interrupt_model_path = os.path.join(args.output_dir, "checkpoints", f"interrupted_{config.name}.ckpt")
        trainer.save_checkpoint(interrupt_model_path)
        print(f"💾 中断时模型已保存: {interrupt_model_path}")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        
        # 保存错误时的模型
        error_model_path = os.path.join(args.output_dir, "checkpoints", f"error_{config.name}.ckpt")
        try:
            trainer.save_checkpoint(error_model_path)
            print(f"💾 错误时模型已保存: {error_model_path}")
        except:
            print("❌ 无法保存错误时的模型")
        
        raise
    
    print("\n🎉 程序执行完成！")

if __name__ == "__main__":
    main()