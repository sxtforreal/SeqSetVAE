# 自适应设备选择功能

## 概述

本项目实现了智能的自适应设备选择功能，能够根据系统环境自动选择最优的训练设备配置。系统会自动检测是否有GPU可用，并根据GPU内存大小调整训练参数。

## 功能特性

### 🚀 自动设备检测
- **GPU检测**: 自动检测CUDA是否可用
- **GPU信息获取**: 获取GPU名称、内存大小、数量等信息
- **CPU检测**: 检测CPU核心数量

### ⚙️ 自适应配置
- **GPU训练**: 如果有GPU，自动使用GPU训练
- **CPU训练**: 如果没有GPU，自动切换到CPU训练
- **多GPU支持**: 支持多GPU训练（最多2个GPU）

### 🎯 智能参数调整
根据GPU内存大小自动调整训练参数：

| GPU内存 | 设备数量 | 精度 | 推荐批次大小 | 说明 |
|---------|----------|------|-------------|------|
| ≥16GB | 1-2 | 16-mixed | 8 | 高性能GPU，支持混合精度 |
| 8-16GB | 1 | 16-mixed | 4 | 中等GPU，平衡性能 |
| <8GB | 1 | 32 | 2 | 小内存GPU，使用32位精度 |
| CPU | 1 | 32 | 1 | CPU训练，32位精度 |

## 使用方法

### 1. 基本使用

训练脚本会自动使用自适应配置：

```bash
python train_optimized.py
```

### 2. 查看设备配置

运行测试脚本查看当前设备配置：

```bash
python test_config_logic.py
```

### 3. 手动指定参数

您仍然可以手动指定某些参数，系统会智能调整其他参数：

```bash
python train_optimized.py --batch_size 4 --precision 16-mixed
```

## 配置详情

### 配置文件 (`config.py`)

```python
# 自动设备检测
device_config = get_optimal_device_config()

# 设备配置
device = device_config['device']           # 'cuda' 或 'cpu'
accelerator = device_config['accelerator'] # 'gpu' 或 'cpu'
devices = device_config['devices']         # 设备数量
precision = device_config['precision']     # 训练精度
```

### 训练配置 (`train_optimized.py`)

```python
# 自适应训练配置
adaptive_config = get_adaptive_training_config(args)

# 训练器配置
trainer = pl.Trainer(
    accelerator=config.accelerator,        # 自动选择
    devices=config.devices,                # 自动选择
    strategy=adaptive_config['strategy'],  # 自动选择
    precision=config.precision,            # 自动选择
    # ... 其他参数
)
```

## 自适应策略

### GPU训练策略

1. **单GPU训练**:
   - 使用 `"auto"` 策略
   - 启用 `pin_memory=True`
   - 根据GPU内存调整worker数量

2. **多GPU训练**:
   - 使用 `DDPStrategy` 策略
   - 禁用 `find_unused_parameters=False`
   - 自动计算有效批次大小

### CPU训练策略

1. **自动调整**:
   - 强制使用32位精度
   - 禁用 `pin_memory`
   - 禁用模型编译
   - 减少worker数量避免过载

## 性能优化

### GPU优化
- **混合精度训练**: 16位精度加速训练
- **内存优化**: 根据GPU内存调整批次大小
- **数据加载优化**: 使用pin_memory加速数据传输

### CPU优化
- **内存友好**: 使用32位精度减少内存使用
- **负载均衡**: 根据CPU核心数调整worker数量
- **稳定性优先**: 禁用可能导致问题的优化功能

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 系统会自动降低批次大小
   - 切换到32位精度
   - 减少worker数量

2. **CPU过载**
   - 自动减少worker数量
   - 使用更保守的配置

3. **设备不可用**
   - 自动切换到CPU训练
   - 显示警告信息

### 调试信息

训练开始时会显示详细的设备配置信息：

```
🚀 GPU detected: NVIDIA GeForce RTX 3080 (10.0GB)
   - Using 1 GPU(s)
   - Precision: 16-mixed
   - Recommended batch size: 4

📋 Adaptive training configuration:
  - Accelerator: gpu
  - Number of devices: 1
  - Precision: 16-mixed
  - Strategy: auto
```

## 测试

运行测试脚本验证功能：

```bash
# 测试配置逻辑
python test_config_logic.py

# 测试完整功能（需要PyTorch环境）
python test_device_adaptive.py
```

## 扩展

### 添加新的设备类型

在 `config.py` 中的 `get_optimal_device_config()` 函数中添加新的设备检测逻辑。

### 自定义配置策略

在 `train_optimized.py` 中的 `get_adaptive_training_config()` 函数中修改配置策略。

## 注意事项

1. **环境依赖**: 需要正确安装PyTorch和CUDA
2. **内存监控**: 建议监控GPU内存使用情况
3. **性能测试**: 在不同设备上测试性能表现
4. **兼容性**: 确保与现有代码兼容

## 更新日志

- **v1.0**: 初始版本，支持基本的GPU/CPU自适应选择
- **v1.1**: 添加多GPU支持和智能参数调整
- **v1.2**: 优化CPU训练配置和性能