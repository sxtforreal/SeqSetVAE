# SeqSetVAE Training Guide

由于原来的统一脚本存在日志冲突问题，现在已经分离成两个独立的脚本：

## 🚀 预训练 (Pretraining)

使用 `train_pretrain.py` 进行预训练：

```bash
python train_pretrain.py \
    --data_dir /path/to/your/data \
    --batch_size 1 \
    --max_epochs 1000 \
    --devices 1 \
    --precision 16-mixed
```

### 预训练参数：
- `--data_dir`: 数据目录路径 (必需)
- `--batch_size`: 批大小 (默认: 1)
- `--max_epochs`: 最大训练轮数 (默认: 1000)
- `--lr`: 学习率 (可选，使用config默认值)
- `--pretrained_ckpt`: 继续训练的检查点 (可选)
- `--output_root_dir`: 输出目录 (默认: outputs/)

预训练模型会保存在 `outputs/SeqSetVAE/pretrain/checkpoints/` 目录下。

## 🎯 微调 (Finetuning)

使用 `train_finetune.py` 进行微调：

```bash
python train_finetune.py \
    --data_dir /path/to/your/data \
    --label_path /path/to/your/labels.csv \
    --pretrained_ckpt /path/to/pretrain/checkpoint.ckpt \
    --batch_size 8 \
    --max_epochs 100 \
    --devices 1 \
    --precision 16-mixed
```

### 微调参数：
- `--data_dir`: 数据目录路径 (必需)
- `--label_path`: 标签文件路径 (必需)
- `--pretrained_ckpt`: 预训练检查点路径 (必需)
- `--batch_size`: 批大小 (默认: 8)
- `--max_epochs`: 最大训练轮数 (默认: 100)
- `--lr`: 基础学习率 (可选)
- `--cls_head_lr`: 分类头学习率 (可选)

微调模型会保存在 `outputs/SeqSetVAE/finetune/checkpoints/` 目录下。

## 📊 监控训练

两个脚本都支持：
- **TensorBoard 日志**: 在 `outputs/SeqSetVAE/{pretrain|finetune}/logs/` 目录
- **检查点保存**: 自动保存最佳模型
- **早停机制**: 防止过拟合
- **学习率监控**: 跟踪学习率变化

### 查看 TensorBoard:
```bash
tensorboard --logdir outputs/SeqSetVAE/pretrain/logs/  # 预训练
tensorboard --logdir outputs/SeqSetVAE/finetune/logs/  # 微调
```

## 🔄 完整训练流程

1. **预训练**:
```bash
python train_pretrain.py --data_dir /your/data --max_epochs 1000
```

2. **找到最佳预训练检查点**:
```bash
ls outputs/SeqSetVAE/pretrain/checkpoints/
```

3. **微调**:
```bash
python train_finetune.py \
    --data_dir /your/data \
    --label_path /your/labels.csv \
    --pretrained_ckpt outputs/SeqSetVAE/pretrain/checkpoints/best_checkpoint.ckpt \
    --max_epochs 100
```

## ✅ 优势

- **无日志冲突**: 每个脚本独立运行，避免日志重复问题
- **清晰分离**: 预训练和微调逻辑完全独立
- **完整监控**: 每个阶段都有完整的指标监控
- **简单易用**: 专门的参数设计，减少配置错误

## 📝 注意事项

1. 预训练不需要标签文件
2. 微调必须提供标签文件和预训练检查点
3. 输出目录会自动创建
4. 检查点自动保存最佳模型
5. 支持多GPU训练（设置 `--devices` 参数）