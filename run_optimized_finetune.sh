#!/bin/bash

# Optimized SeqSetVAE Finetune Script
# 使用修复后的配置和代码进行高效微调

echo "🚀 Starting optimized SeqSetVAE finetuning..."

# 检查预训练检查点是否存在
PRETRAINED_CKPT="${1:-/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt}"

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "❌ Error: Pretrained checkpoint not found at: $PRETRAINED_CKPT"
    echo "Please provide the correct path as the first argument:"
    echo "bash run_optimized_finetune.sh /path/to/your/pretrained/checkpoint.ckpt"
    exit 1
fi

echo "📦 Using pretrained checkpoint: $PRETRAINED_CKPT"

# 运行优化的微调训练
python train.py \
    --mode finetune \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_epochs 20 \
    --devices 1 \
    --precision "16-mixed" \
    --compile_model \
    --num_workers 4 \
    --seed 42

echo "✅ Optimized finetuning completed!"
echo "📊 Check the logs for val_auc and val_auprc improvements"