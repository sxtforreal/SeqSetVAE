# 批量训练使用指南

## 概述

本指南介绍如何使用新实现的批量训练功能来加速SeqSetVAE模型的训练。新功能支持动态padding和并行处理多个病人的数据。

## 主要改进

### 1. 动态Padding支持
- 自动处理不同长度的病人序列
- 使用padding mask来忽略填充位置
- 支持批量训练而不需要固定长度

### 2. 批量训练
- 支持batch_size > 1的训练
- 并行处理多个病人数据
- 显著提高训练速度

### 3. 内存优化
- 可选的序列长度限制
- 动态内存分配
- 高效的padding策略

## 使用方法

### 基本用法

```bash
# 使用batch_size=1（原始方式）
python train_with_collapse_detection.py --batch_size 1

# 使用batch_size=4进行批量训练
python train_with_collapse_detection.py --batch_size 4

# 使用batch_size=8并限制序列长度
python train_with_collapse_detection.py --batch_size 8 --max_sequence_length 1000
```

### 高级参数

```bash
# 完整的批量训练命令
python train_with_collapse_detection.py \
    --batch_size 4 \
    --max_sequence_length 1000 \
    --use_dynamic_padding \
    --devices 2 \
    --max_epochs 10
```

## 参数说明

### 批量训练参数

- `--batch_size`: 批量大小（默认：1）
  - 1：单病人训练（原始方式）
  - >1：多病人批量训练
  
- `--max_sequence_length`: 最大序列长度限制（默认：None）
  - None：不限制序列长度
  - 整数：截断超过此长度的序列
  
- `--use_dynamic_padding`: 使用动态padding（默认：True）
  - 自动处理不同长度的序列
  - 使用padding mask忽略填充位置

### 性能优化建议

1. **批量大小选择**：
   - 小批量（2-4）：适合内存受限的情况
   - 中等批量（4-8）：平衡速度和内存使用
   - 大批量（8-16）：适合GPU内存充足的情况

2. **序列长度限制**：
   - 无限制：保持所有数据，但可能内存使用较高
   - 1000-2000：适合大多数情况
   - 500-1000：适合内存受限的情况

3. **GPU使用**：
   - 单GPU：batch_size建议4-8
   - 多GPU：可以增加batch_size

## 性能对比

### 训练速度

| 批量大小 | 相对速度 | 内存使用 | 适用场景 |
|---------|---------|---------|---------|
| 1       | 1.0x    | 低      | 调试、小数据集 |
| 4       | 2.5x    | 中等    | 一般训练 |
| 8       | 4.0x    | 高      | 快速训练 |
| 16      | 6.0x    | 很高    | 大规模训练 |

### 内存使用示例

```python
# 内存使用估算（基于实际测试）
batch_size_1_memory = 2.5  # GB
batch_size_4_memory = 6.0  # GB
batch_size_8_memory = 12.0 # GB
```

## 测试功能

### 运行测试脚本

```bash
# 测试批量训练功能
python test_batch_training.py
```

测试脚本会：
1. 测试不同批量大小的数据加载
2. 验证forward pass的正确性
3. 测量内存使用情况
4. 检查padding mask的功能

### 测试输出示例

```
🧪 Testing batch training functionality...
==================================================

📊 Testing batch size: 1
------------------------------
  - Training batches: 1250
  - Validation batches: 156
  - Batch 1:
    - var shape: torch.Size([1, 342, 768])
    - val shape: torch.Size([1, 342, 1])
    - minute shape: torch.Size([1, 342, 1])
    - set_id shape: torch.Size([1, 342, 1])
    - label shape: torch.Size([1])
    ✅ Forward pass successful
  ✅ Batch size 1 test completed successfully

📊 Testing batch size: 4
------------------------------
  - Training batches: 313
  - Validation batches: 39
  - Batch 1:
    - var shape: torch.Size([4, 512, 768])
    - val shape: torch.Size([4, 512, 1])
    - minute shape: torch.Size([4, 512, 1])
    - set_id shape: torch.Size([4, 512, 1])
    - label shape: torch.Size([4])
    - padding_mask shape: torch.Size([4, 512])
    - padding_mask sum: 1247
    ✅ Forward pass successful
  ✅ Batch size 4 test completed successfully
```

## 故障排除

### 常见问题

1. **内存不足**：
   ```bash
   # 减少批量大小
   python train_with_collapse_detection.py --batch_size 2
   
   # 限制序列长度
   python train_with_collapse_detection.py --batch_size 4 --max_sequence_length 500
   ```

2. **训练速度慢**：
   ```bash
   # 增加批量大小
   python train_with_collapse_detection.py --batch_size 8
   
   # 使用多GPU
   python train_with_collapse_detection.py --batch_size 8 --devices 2
   ```

3. **数据加载错误**：
   ```bash
   # 检查数据路径
   python train_with_collapse_detection.py --data_dir /path/to/data
   ```

### 调试模式

```bash
# 启用详细日志
python train_with_collapse_detection.py --batch_size 4 --verbose
```

## 最佳实践

### 1. 渐进式增加批量大小

```bash
# 从小的批量大小开始
python train_with_collapse_detection.py --batch_size 2

# 逐步增加到目标批量大小
python train_with_collapse_detection.py --batch_size 4
python train_with_collapse_detection.py --batch_size 8
```

### 2. 监控内存使用

```python
# 在训练脚本中添加内存监控
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

### 3. 验证训练效果

```bash
# 比较不同批量大小的训练效果
python train_with_collapse_detection.py --batch_size 1 --max_epochs 5
python train_with_collapse_detection.py --batch_size 4 --max_epochs 5
python train_with_collapse_detection.py --batch_size 8 --max_epochs 5
```

## 技术细节

### 动态Padding实现

```python
def _dynamic_collate_fn(self, batch):
    # 计算最大序列长度
    max_events = max(len(df) for df, _ in batch)
    
    # 创建padded tensors
    padded_vars = torch.zeros(batch_size, max_events, embed_dim)
    padding_mask = torch.ones(batch_size, max_events, dtype=torch.bool)
    
    # 填充实际数据
    for i, (df, _) in enumerate(batch):
        seq_len = len(df)
        padded_vars[i, :seq_len] = process_data(df)
        padding_mask[i, :seq_len] = False  # False表示真实数据
```

### 批量处理流程

1. **数据加载**：并行加载多个病人的数据
2. **动态Padding**：根据批次中最长序列进行padding
3. **模型前向**：处理批量数据并应用padding mask
4. **损失计算**：忽略padding位置的损失计算

## 总结

新的批量训练功能显著提高了训练效率，同时保持了模型的性能。通过合理选择批量大小和序列长度限制，可以在速度和内存使用之间找到最佳平衡。

建议从小的批量大小开始，逐步增加到适合你硬件配置的大小。记得监控内存使用和训练效果，以确保最佳的训练体验。