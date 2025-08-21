# SeqSetVAE Quick Start Guide

## 🚀 快速开始

### **1. 预训练阶段**
```bash
python train.py --mode pretrain --batch_size 8 --max_epochs 100
```

### **2. 微调阶段**
```bash
python train.py --mode finetune \
    --pretrained_ckpt your_pretrain.ckpt \
    --batch_size 4 --max_epochs 15
```

### **3. 测试验证**
```bash
# 基础功能测试
python test_suite.py

# 性能分析测试
python test_suite.py \
    --checkpoint your_finetune.ckpt \
    --data_dir your_data \
    --params_map your_params.pkl \
    --label_file your_labels.csv
```

## 📋 关键文件说明

| 文件 | 用途 |
|------|------|
| `train.py` | 统一训练脚本（预训练+微调） |
| `model.py` | 包含SeqSetVAEPretrain和SeqSetVAE两个模型 |
| `test_suite.py` | 综合测试套件 |
| `finetune_config.py` | 微调专用配置 |
| `COMPLETE_GUIDE.md` | 完整技术文档 |

## ✅ 关键检查点

训练时确保看到：
- ✅ "✅ Loaded pretrained weights" - 权重加载成功
- ✅ "Trainable parameters: XXX" - 只有分类头参数可训练
- ✅ 训练过程稳定收敛

## 🎯 预期效果

预计AUC/AUPRC提升：
- 保守估计：+0.02-0.05
- 乐观估计：+0.05-0.10

主要改进：
1. 🔥 预训练权重正确加载
2. 🔥 完全参数冻结策略  
3. 🚀 现代VAE特征融合