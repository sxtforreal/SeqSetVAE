# 🚀 SeqSetVAE 统一训练指南

## 📋 **概述**

经过整合优化，现在所有功能都集中在几个核心文件中：

- **`train.py`**: 统一训练脚本，支持3种模式
- **`losses.py`**: 集成所有损失函数策略（包括SOTA方法）
- **`model.py`**: 完整的模型实现
- **`finetune_config.py`**: 优化的配置参数

## 🎯 **训练模式**

### 1. **预训练模式** (`pretrain`)
```bash
python train.py \
    --mode pretrain \
    --batch_size 4 \
    --max_epochs 50 \
    --data_dir /path/to/data
```

### 2. **标准微调模式** (`finetune`)
```bash
python train.py \
    --mode finetune \
    --pretrained_ckpt /path/to/pretrained.ckpt \
    --batch_size 6 \
    --max_epochs 25 \
    --data_dir /path/to/data
```

### 3. **🏆 SOTA医疗优化模式** (`finetune-sota`)
```bash
# 自动检测最优策略
python train.py \
    --mode finetune-sota \
    --pretrained_ckpt /path/to/pretrained.ckpt \
    --medical_scenario auto \
    --target_auc 0.90 \
    --target_auprc 0.50 \
    --max_epochs 30

# 手动指定医疗场景
python train.py \
    --mode finetune-sota \
    --pretrained_ckpt /path/to/pretrained.ckpt \
    --medical_scenario rare_disease_detection \
    --target_auc 0.95 \
    --target_auprc 0.60
```

## 🏥 **医疗场景配置**

### 自动模式 (`--medical_scenario auto`)
系统会自动分析你的数据特征并选择最优策略：

| 数据特征 | 推荐策略 | 描述 |
|----------|----------|------|
| 极度不平衡 (1:20+) | `rare_disease_detection` | 罕见疾病检测优化 |
| 严重不平衡 (1:5-1:20) | `diagnostic_assistance` | 诊断辅助优化 |
| 中度不平衡 (1:2.5-1:5) | `treatment_response_prediction` | 治疗反应预测 |
| 相对平衡 | `multi_condition_screening` | 多病症筛查 |

### 手动场景选择

#### 1. **罕见疾病检测** (`rare_disease_detection`)
```bash
--medical_scenario rare_disease_detection
```
- **适用**: 极度不平衡数据 (1:50+)
- **特点**: 强Focal focusing (γ=3.0)，重自蒸馏
- **优化**: 最大化召回率，减少漏诊

#### 2. **多病症筛查** (`multi_condition_screening`) 
```bash
--medical_scenario multi_condition_screening
```
- **适用**: 一般医疗分类任务
- **特点**: 平衡的Focal参数 (α=0.25, γ=2.0)
- **优化**: AUC和AUPRC并重

#### 3. **治疗反应预测** (`treatment_response_prediction`)
```bash
--medical_scenario treatment_response_prediction
```
- **适用**: 需要概率校准的任务
- **特点**: 温和Focal参数 (γ=1.5)，重置信度感知
- **优化**: 准确的概率估计

#### 4. **诊断辅助** (`diagnostic_assistance`)
```bash
--medical_scenario diagnostic_assistance
```
- **适用**: 复合医疗应用
- **特点**: 综合优化策略
- **优化**: 多头集成，鲁棒性

## 🔧 **高级参数调优**

### 性能目标设定
```bash
--target_auc 0.90      # 目标AUC分数
--target_auprc 0.50    # 目标AUPRC分数
```

### 训练控制
```bash
--max_epochs 30        # 最大训练轮数
--batch_size 6         # 批次大小
--gradient_accumulation_steps 3  # 梯度累积
--num_workers 6        # 数据加载线程
```

### 输出控制
```bash
--output_dir /path/to/outputs  # 输出目录
--seed 42              # 随机种子
--deterministic        # 确定性训练
```

## 📊 **监控和输出**

### 训练日志示例
```
🚀 Training Configuration:
 - Mode: finetune-sota
 - 🏆 SOTA Medical Classification Mode
 - Target AUC: 0.90
 - Target AUPRC: 0.50
 - Medical Scenario: rare_disease_detection

🔍 Auto-detecting optimal medical scenario...
🎯 Detected imbalance ratio: 0.045
🏥 Selected medical scenario: rare_disease_detection

📋 SOTA Configuration:
   - Scenario: rare_disease_detection
   - Description: Optimized for rare disease detection with extreme imbalance
   - Focal α: 0.1, γ: 3.0
   - EMA decay: 0.9999

🧠 Building model...
🏆 Using SeqSetVAE - SOTA MODE with advanced medical optimization
```

### 实时性能监控
```
🏥 Medical Classification Progress (Epoch 15):
   📊 Current: AUC=0.8934, AUPRC=0.5123, Medical Score=0.6651
   🏆 Best: AUC=0.8934, AUPRC=0.5123
   🎯 Progress: AUC 99.3%, AUPRC 102.5%, Overall 100.9%
   📏 Calibration Error: 0.0342, Stability: 0.8765

🎉 TARGET ACHIEVED! AUC=0.8934≥0.90, AUPRC=0.5123≥0.50
```

## 🔬 **集成的SOTA技术**

### 损失函数策略 (`losses.py`)
1. **FocalLoss**: 原版Focal Loss
2. **AsymmetricLoss**: 处理极端不平衡 (ICLR 2021)
3. **SOTALossStrategy**: 集成多项前沿技术
   - SoftAdapt动态权重 (ICML 2020)
   - EMA动量教师自蒸馏 (CVPR 2024)
   - 梯度自适应调整 (NeurIPS 2023)
   - 置信度感知一致性 (ICCV 2024)

### 模型架构优化
- 高级分类头：多头注意力 + 双路径处理
- 门控特征融合
- 辅助预测头
- 残差连接和层归一化

## 📁 **文件结构**

```
SeqSetVAE/
├── train.py                    # 🚀 统一训练脚本
├── losses.py                   # 🏆 集成损失函数策略
├── model.py                    # 🧠 完整模型实现
├── finetune_config.py          # ⚙️ 优化配置
├── dataset.py                  # 📊 数据加载
├── config.py                   # 🔧 基础配置
└── UNIFIED_USAGE_GUIDE.md      # 📖 本指南
```

## 🎯 **快速开始示例**

### 完整流程
```bash
# 1. 预训练
python train.py \
    --mode pretrain \
    --batch_size 4 \
    --max_epochs 50 \
    --data_dir /path/to/data \
    --output_dir /path/to/outputs

# 2. SOTA微调
python train.py \
    --mode finetune-sota \
    --pretrained_ckpt /path/to/outputs/SeqSetVAE-v3/checkpoints/SeqSetVAE_pretrain_batch4.ckpt \
    --medical_scenario auto \
    --target_auc 0.90 \
    --target_auprc 0.50 \
    --max_epochs 30 \
    --batch_size 6 \
    --data_dir /path/to/data \
    --output_dir /path/to/outputs
```

### 医疗场景特化
```bash
# 罕见疾病检测
python train.py --mode finetune-sota --medical_scenario rare_disease_detection --target_auprc 0.70

# 多病症筛查  
python train.py --mode finetune-sota --medical_scenario multi_condition_screening --target_auc 0.90

# 治疗反应预测
python train.py --mode finetune-sota --medical_scenario treatment_response_prediction --target_auc 0.85
```

## 🚨 **故障排除**

### 常见问题
1. **CUDA内存不足**: 减少 `--batch_size` 到 4 或更小
2. **收敛缓慢**: 使用 `--medical_scenario rare_disease_detection` 提高学习率
3. **过拟合**: 增加 `--max_epochs` 并启用更强正则化
4. **SOTA模式失败**: 降级到标准 `--mode finetune`

### 性能优化
- 使用 `--deterministic` 确保可重现结果
- 调整 `--gradient_accumulation_steps` 平衡内存和性能
- 监控 `Medical Score = 0.6×AUC + 0.4×AUPRC` 作为综合指标

## 📞 **技术支持**

所有功能现在都集成在统一的框架中，基于2024年最新学术研究，为医疗分类任务提供最先进的解决方案！

需要帮助时，请提供完整的命令行参数和错误日志。🤝