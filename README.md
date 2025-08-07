# SeqSetVAE 训练优化项目

## 📋 项目概述

本项目是一个基于PyTorch Lightning的SeqSetVAE模型训练系统，专注于医疗时序数据的表示学习和分类任务。项目包含完整的训练流程、性能优化、后验塌缩检测和可视化分析功能。

## 🚀 主要功能

### 1. 训练优化
- **高性能训练**：支持混合精度训练、梯度累积、模型编译等优化
- **数据加载优化**：多进程数据加载、内存固定、动态填充
- **监控开销优化**：可配置的监控频率、可选的后验指标监控

### 2. 后验塌缩检测
- **实时监控**：训练过程中监控KL散度、潜在变量方差等关键指标
- **可视化分析**：自动生成监控图表和分析报告
- **预警机制**：及时发现后验塌缩问题

### 3. 可视化分析
- **训练曲线分析**：损失函数、性能指标、训练动态的可视化
- **模型可视化**：隐空间分析、重建质量评估
- **实时监控**：训练过程中的实时指标显示

## 📁 核心文件

### 训练相关
- `train_optimized.py` - **优化版本训练脚本**（推荐使用）
- `train_with_collapse_detection.py` - 带后验监控的训练脚本
- `model.py` - SeqSetVAE模型定义
- `dataset.py` - 数据加载器（已优化）
- `modules.py` - 模型组件
- `config.py` - 配置文件

### 监控和分析
- `posterior_collapse_detector.py` - 后验指标监控器
- `analyze_training_curves.py` - 训练曲线分析工具
- `visualize_model.py` - 模型可视化工具
- `collapse_visualizer.py` - 实时可视化工具

### 文档
- `TRAINING_OPTIMIZATION_GUIDE.md` - **训练速度优化完整指南**
- `VAE后验塌缩检测系统完整使用指南.md` - 后验塌缩检测使用指南

## 🎯 快速开始

### 1. 优化训练（推荐）

```bash
# 快速训练配置
python train_optimized.py \
    --batch_size 4 \
    --num_workers 8 \
    --pin_memory \
    --gradient_accumulation_steps 2 \
    --max_sequence_length 1000 \
    --precision 16-mixed \
    --fast_detection
```

### 2. 带监控的训练

```bash
# 带后验指标监控的训练
python train_with_collapse_detection.py \
    --batch_size 4 \
    --fast_detection
```

### 3. 性能测试

```bash
# 查看详细的优化指南
cat TRAINING_OPTIMIZATION_GUIDE.md
```

## 📊 性能提升

使用优化配置后，预期可以获得 **2-3倍** 的训练速度提升：

| 优化措施 | 预期速度提升 | 内存使用变化 |
|---------|-------------|-------------|
| 增加批处理大小 (1→4) | 30-50% | +50% |
| 增加工作进程数 (2→8) | 20-30% | +10% |
| 混合精度训练 | 20-40% | -30% |
| 梯度累积 | 10-20% | 无变化 |
| 模型编译 | 10-25% | 无变化 |
| 减少监控频率 | 5-15% | -10% |
| 限制序列长度 | 20-40% | -40% |

## 🔧 系统要求

- Python 3.8+
- PyTorch 2.0+（推荐，支持模型编译）
- PyTorch Lightning
- CUDA支持（推荐）
- 8GB+ GPU内存（推荐）

## 📝 使用说明

详细的优化指南请参考：[TRAINING_OPTIMIZATION_GUIDE.md](TRAINING_OPTIMIZATION_GUIDE.md)

后验塌缩检测使用说明请参考：[VAE后验塌缩检测系统完整使用指南.md](VAE后验塌缩检测系统完整使用指南.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。