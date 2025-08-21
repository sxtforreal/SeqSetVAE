"""
Optimized configuration for SeqSetVAE finetuning stage
专门针对分类头微调的优化配置
"""

import os
from config import *  # Import base config

# Override specific settings for better finetune performance

# ====== 关键改进设置 ======

# 1. 更保守的学习率设置
cls_head_lr = 2e-4  # 降低分类头学习率，避免过快收敛
lr = 5e-5  # 保持backbone学习率较低

# 2. 更好的正则化
cls_head_weight_decay = 0.02  # 增加分类头的权重衰减
dropout_rate = 0.3  # 增加dropout防止过拟合

# 3. 优化的训练策略
warmup_steps = 500  # 减少warmup步数，快速进入有效训练
max_epochs = 15  # 增加最大训练轮数
early_stopping_patience = 6  # 减少早停耐心，防止过拟合

# 4. 更稳定的损失函数设置
use_focal_loss = True
focal_alpha = 0.25  # 调整focal loss参数
focal_gamma = 2.0
label_smoothing = 0.05  # 添加标签平滑

# 5. 批次和验证设置
batch_size = 4  # 适中的批次大小
val_check_interval = 0.2  # 更频繁的验证
limit_val_batches = 1.0  # 使用全部验证数据

# 6. 特征提取优化
feature_extraction_mode = "stable"  # 使用稳定的特征提取模式

# 7. 梯度相关设置
gradient_clip_val = 0.5  # 适度的梯度裁剪
accumulate_grad_batches = 2  # 梯度累积

# 8. 学习率调度
scheduler_type = "cosine_with_restarts"
scheduler_patience = 3
scheduler_factor = 0.7
scheduler_min_lr = 1e-6

# ====== 监控和日志 ======
monitor_metric = "val_auc"
monitor_mode = "max"
log_every_n_steps = 50

# ====== 数据增强（如果适用）======
# 在分类微调阶段，通常不需要太多数据增强
data_augmentation = False
noise_std = 0.0  # 关闭噪声注入

print("🎯 Finetune config loaded with optimized settings for AUC/AUPRC performance")
print(f"   - Classification head LR: {cls_head_lr}")
print(f"   - Weight decay: {cls_head_weight_decay}")
print(f"   - Focal loss: alpha={focal_alpha}, gamma={focal_gamma}")
print(f"   - Early stopping patience: {early_stopping_patience}")