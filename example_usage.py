#!/usr/bin/env python3
"""
改进的 Hierarchical SetVAE 使用示例
解决后验坍缩问题的完整方案
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
from posterior_collapse_diagnostics import PosteriorCollapseDiagnostics
import config

def main():
    print("=== 改进的 Hierarchical SetVAE 训练 ===")
    print("包含后验坍缩防护措施:")
    print("1. β退火策略 (Cyclical Annealing)")
    print("2. 谱归一化 (Spectral Normalization)")
    print("3. 改进的Free Bits策略")
    print("4. KL散度监控")
    print("5. 潜在空间诊断工具")
    print()

    # 设置随机种子
    pl.seed_everything(42, workers=True)

    # 数据模块
    saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    
    data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
    data_module.setup()
    
    print(f"训练数据: {len(data_module.train_dataset)}")
    print(f"验证数据: {len(data_module.val_dataset)}")
    print(f"测试数据: {len(data_module.test_dataset)}")

    # 创建改进的模型
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
        pretrained_ckpt=config.pretrained_ckpt,
        w=config.w,
        free_bits=config.free_bits,
        # 后验坍缩防护参数
        use_spectral_norm=config.use_spectral_norm,
        beta_strategy=config.beta_strategy,
        min_beta=config.min_beta,
        cycle_length=config.cycle_length,
        beta_warmup_steps=config.beta_warmup_steps,
        use_tc_decomposition=config.use_tc_decomposition,
        pc_threshold=config.pc_threshold,
    )

    print(f"\n模型配置:")
    print(f"- β策略: {config.beta_strategy}")
    print(f"- 谱归一化: {config.use_spectral_norm}")
    print(f"- Free bits: {config.free_bits}")
    print(f"- TC分解: {config.use_tc_decomposition}")

    # 日志记录器
    logger = TensorBoardLogger(
        save_dir="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/logs",
        name=config.name,
    )

    # 回调函数
    checkpoint = ModelCheckpoint(
        dirpath="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints",
        filename=f"best_{config.name}",
        save_weights_only=True,
        save_last=True,
        every_n_train_steps=config.ckpt_every_n_steps,
        monitor="val_auc",
        mode="max",
        save_top_k=3,  # 保存更多检查点以便分析
        enable_version_counter=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,  # 增加耐心值
        verbose=True,
        strict=True,
    )

    # 训练器
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        logger=logger,
        max_epochs=config.max_epochs,
        min_epochs=1,
        precision=config.precision,
        callbacks=[checkpoint, early_stopping],
        profiler="simple",
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",  # 使用L2范数裁剪
        val_check_interval=0.1,  # 更频繁的验证
        limit_val_batches=0.2,
        enable_model_summary=True,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.fit(model, data_module)

    # 训练后诊断
    print("\n=== 训练后诊断 ===")
    
    # 创建诊断工具
    diagnostics = PosteriorCollapseDiagnostics(save_dir="./diagnostics_results")
    
    # 生成诊断报告
    try:
        report = diagnostics.generate_training_report(
            model, 
            data_module.val_dataloader(),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("\n诊断完成！请查看 ./diagnostics_results/ 目录下的图表和报告。")
        
        # 给出改进建议
        summary = report['summary']
        print("\n=== 改进建议 ===")
        
        if summary.get('posterior_collapse_detected', False):
            print("⚠️  检测到后验坍缩！建议:")
            print("  - 降低β值或使用更激进的退火策略")
            print("  - 增加free_bits值")
            print("  - 考虑使用TC分解")
            print("  - 检查重构损失是否过小")
        else:
            print("✅ 未检测到严重的后验坍缩")
            
        if 'latent_utilization_ratio' in summary:
            ratio = summary['latent_utilization_ratio']
            if ratio < 0.3:
                print("⚠️  潜在空间利用率较低，建议:")
                print("  - 减少潜在维度")
                print("  - 增加模型复杂度")
                print("  - 调整正则化强度")
            elif ratio > 0.8:
                print("✅ 潜在空间利用率良好")
                
    except Exception as e:
        print(f"诊断过程中出现错误: {e}")
        print("请检查模型和数据加载器是否正确")

    print("\n训练和诊断完成！")

def analyze_existing_model():
    """分析已有模型的后验坍缩情况"""
    print("=== 分析现有模型 ===")
    
    # 加载现有模型
    ckpt_path = "/path/to/your/existing/checkpoint.ckpt"
    
    try:
        model = SeqSetVAE.load_from_checkpoint(ckpt_path)
        
        # 加载数据
        data_module = SeqSetVAEDataModule(
            "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr",
            "/home/sunx/data/aiiih/data/mimic/processed/stats.csv", 
            "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
        )
        data_module.setup()
        
        # 诊断
        diagnostics = PosteriorCollapseDiagnostics(save_dir="./existing_model_diagnostics")
        report = diagnostics.generate_training_report(
            model, 
            data_module.val_dataloader(),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("现有模型诊断完成！")
        
    except FileNotFoundError:
        print("未找到模型检查点文件，跳过现有模型分析")
    except Exception as e:
        print(f"分析现有模型时出错: {e}")

if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_existing_model()
    else:
        main()