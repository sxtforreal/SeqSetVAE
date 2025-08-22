"""
Optimized configuration for SeqSetVAE finetuning stage - Performance Enhanced
解决训练速度慢和性能下降问题的优化配置
"""

import os
from config import *  # Import base config

# ====== Critical Performance Fixes ======

# 1. **FIXED** Learning Rate Configuration
cls_head_lr = 3e-4  # 增加分类头学习率，原来1e-4太低
lr = 1e-5  # 保持backbone LR较低（虽然会被冻结）

# 2. **FIXED** Batch Processing Efficiency  
batch_size = 8  # 增加batch size提高GPU利用率
gradient_accumulation_steps = 1  # 减少梯度累积，简化训练
num_workers = 6  # 优化数据加载并行度

# 3. **FIXED** Training Strategy
max_epochs = 20  # 增加最大训练轮数
early_stopping_patience = 8  # 增加早停耐心，避免过早停止
val_check_interval = 0.25  # 减少验证频率，提高训练速度

# 4. **FIXED** Loss Function Optimization
use_focal_loss = True
focal_alpha = 0.25  # 平衡正负样本
focal_gamma = 2.0  # 适中的难样本权重
label_smoothing = 0.0  # 移除标签平滑，避免过度正则化

# 5. **FIXED** Regularization Settings
cls_head_weight_decay = 0.005  # 减少权重衰减
dropout_rate = 0.15  # 减少dropout，保留更多信息

# 6. **FIXED** Scheduler Configuration
scheduler_type = "reduce_on_plateau"
scheduler_monitor = "val_auc"  # 监控AUC而不是loss
scheduler_mode = "max"  # AUC越大越好
scheduler_patience = 4  # 减少调度器耐心
scheduler_factor = 0.8  # 更温和的学习率衰减
scheduler_min_lr = 1e-6

# 7. **FIXED** Model Architecture Simplification
# 简化分类头架构，减少计算开销
simplified_cls_head = True
cls_head_layers = [128, 64]  # 简化为2层
cls_dropout = 0.1  # 减少dropout

# 8. **FIXED** Training Optimizations
pin_memory = True
persistent_workers = True
prefetch_factor = 2
drop_last = True  # 保持batch大小一致性

# 9. **FIXED** Monitoring Settings
monitor_metric = "val_auc"
monitor_mode = "max"
log_every_n_steps = 25  # 减少日志频率

# 10. **FIXED** Memory and Speed Optimizations
enable_torch_compile = True  # 启用PyTorch编译优化
mixed_precision = True
gradient_checkpointing = False  # 微调时不需要梯度检查点

print("🚀 Optimized Finetune Config Loaded - Performance Enhanced!")
print(f"   - Classification head LR: {cls_head_lr} (increased from 1e-4)")
print(f"   - Batch size: {batch_size} (increased for better GPU utilization)")
print(f"   - Scheduler monitors: {scheduler_monitor} (changed from val_loss)")
print(f"   - Simplified cls head: {simplified_cls_head}")
print(f"   - Early stopping patience: {early_stopping_patience}")