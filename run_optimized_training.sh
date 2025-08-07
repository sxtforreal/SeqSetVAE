#!/bin/bash

# 训练速度优化启动脚本
# 使用方法: ./run_optimized_training.sh [配置名称]

set -e

# 默认配置
DEFAULT_CONFIG="fast"

# 检查参数
if [ $# -eq 0 ]; then
    CONFIG=$DEFAULT_CONFIG
    echo "使用默认配置: $CONFIG"
else
    CONFIG=$1
fi

# 检查Python脚本是否存在
if [ ! -f "train_optimized.py" ]; then
    echo "错误: train_optimized.py 不存在"
    exit 1
fi

echo "🚀 启动优化训练 - 配置: $CONFIG"
echo "=================================="

case $CONFIG in
    "fast")
        echo "📊 快速训练配置 (推荐)"
        echo "   - 批处理大小: 4"
        echo "   - 工作进程数: 8"
        echo "   - 序列长度限制: 1000"
        echo "   - 梯度累积: 2"
        echo "   - 混合精度: 16-mixed"
        echo "   - 快速监控: 启用"
        echo "   - 模型编译: 禁用"
        
        python train_optimized.py \
            --batch_size 4 \
            --num_workers 8 \
            --pin_memory \
            --gradient_accumulation_steps 2 \
            --max_sequence_length 1000 \
            --precision 16-mixed \
            --fast_detection
        ;;
        
    "max_performance")
        echo "⚡ 最大性能配置"
        echo "   - 批处理大小: 4"
        echo "   - 工作进程数: 8"
        echo "   - 序列长度限制: 1000"
        echo "   - 梯度累积: 2"
        echo "   - 混合精度: 16-mixed"
        echo "   - 监控: 禁用"
        echo "   - 模型编译: 启用"
        
        python train_optimized.py \
            --batch_size 4 \
            --num_workers 8 \
            --pin_memory \
            --gradient_accumulation_steps 2 \
            --max_sequence_length 1000 \
            --precision 16-mixed \
            --disable_metrics_monitoring \
            --compile_model
        ;;
        
    "memory_efficient")
        echo "💾 内存高效配置"
        echo "   - 批处理大小: 2"
        echo "   - 工作进程数: 4"
        echo "   - 序列长度限制: 500"
        echo "   - 梯度累积: 2"
        echo "   - 混合精度: 16-mixed"
        echo "   - 快速监控: 启用"
        echo "   - 模型编译: 禁用"
        
        python train_optimized.py \
            --batch_size 2 \
            --num_workers 4 \
            --pin_memory \
            --gradient_accumulation_steps 2 \
            --max_sequence_length 500 \
            --precision 16-mixed \
            --fast_detection
        ;;
        
    "debug")
        echo "🐛 调试配置"
        echo "   - 批处理大小: 1"
        echo "   - 工作进程数: 2"
        echo "   - 序列长度限制: 100"
        echo "   - 梯度累积: 1"
        echo "   - 混合精度: 32"
        echo "   - 监控: 启用"
        echo "   - 模型编译: 禁用"
        
        python train_optimized.py \
            --batch_size 1 \
            --num_workers 2 \
            --max_sequence_length 100 \
            --precision 32 \
            --fast_detection
        ;;
        
    "custom")
        echo "🔧 自定义配置"
        echo "请手动运行: python train_optimized.py [参数]"
        echo ""
        echo "可用参数:"
        echo "  --batch_size INT              批处理大小 (默认: 4)"
        echo "  --num_workers INT             工作进程数 (默认: 8)"
        echo "  --max_sequence_length INT     最大序列长度 (默认: 1000)"
        echo "  --gradient_accumulation_steps INT 梯度累积步数 (默认: 2)"
        echo "  --precision STR               精度 (16-mixed, 32, bf16-mixed)"
        echo "  --fast_detection              快速监控模式"
        echo "  --disable_metrics_monitoring  禁用监控"
        echo "  --compile_model               启用模型编译"
        echo "  --pin_memory                  启用内存固定"
        exit 0
        ;;
        
    *)
        echo "❌ 未知配置: $CONFIG"
        echo ""
        echo "可用配置:"
        echo "  fast             快速训练配置 (推荐)"
        echo "  max_performance  最大性能配置"
        echo "  memory_efficient 内存高效配置"
        echo "  debug            调试配置"
        echo "  custom           自定义配置"
        exit 1
        ;;
esac

echo ""
echo "✅ 训练完成!"