# 后验指标监控使用示例

## 快速开始

### 1. 基本使用

```bash
# 启动带指标监控的训练
python train_with_collapse_detection.py --fast_detection

# 禁用指标监控（如果不需要）
python train_with_collapse_detection.py --disable_metrics_monitoring
```

### 2. 批量训练

```bash
# 使用batch_size=4进行批量训练
python train_with_collapse_detection.py --batch_size 4 --fast_detection

# 限制序列长度以节省内存
python train_with_collapse_detection.py --batch_size 8 --max_sequence_length 1000
```

## 监控参数说明

### 更新频率
- `update_frequency=20` (快速模式): 每20步更新一次指标
- `update_frequency=50` (标准模式): 每50步更新一次指标

### 图表保存频率
- `plot_frequency=200` (快速模式): 每200步保存一次图表
- `plot_frequency=500` (标准模式): 每500步保存一次图表

## 生成的图表

### 监控图表位置
```
./outputs/logs/SeqSetVAE_with_metrics_monitoring/version_X/posterior_metrics/
├── posterior_metrics_step_200_20231201_143022.png
├── posterior_metrics_step_400_20231201_143045.png
├── posterior_metrics_step_600_20231201_143108.png
└── ...
```

### 图表内容
每个图表包含四个子图：
1. **KL散度** - 显示KL散度随时间的变化
2. **潜在变量方差** - 显示潜在变量方差的变化
3. **激活单元比例** - 显示激活单元比例的变化
4. **重建损失** - 显示重建损失的变化

## 指标解读

### KL散度
- **正常范围**: 0.1-10
- **警告范围**: < 0.01 (可能发生后验塌缩)
- **过高范围**: > 100 (可能过度正则化)

### 潜在变量方差
- **正常范围**: > 0.1
- **警告范围**: < 0.1 (可能塌缩)

### 激活单元比例
- **正常范围**: > 0.1
- **警告范围**: < 0.1 (大量隐维度失活)

### 重建损失
- **正常趋势**: 持续下降
- **警告信号**: 停滞或上升

## 实际使用建议

### 1. 首次使用
```bash
# 建议从快速监控模式开始
python train_with_collapse_detection.py --fast_detection --batch_size 4
```

### 2. 生产环境
```bash
# 使用标准监控模式，减少开销
python train_with_collapse_detection.py --batch_size 8 --max_sequence_length 1000
```

### 3. 调试模式
```bash
# 禁用监控，专注于训练
python train_with_collapse_detection.py --disable_metrics_monitoring
```

## 故障排除

### 常见问题

1. **图表不生成**
   - 检查 `posterior_metrics` 目录是否存在
   - 确认训练步数是否达到 `plot_frequency`

2. **指标异常**
   - 检查模型输出格式
   - 确认潜在变量是否正确提取

3. **内存不足**
   - 减少 `batch_size`
   - 设置 `max_sequence_length`
   - 增加 `update_frequency`

## 示例输出

### 训练过程中的输出
```
📊 Posterior metrics monitor setup complete:
  - Monitoring mode: Fast
  - Update frequency: every 20 steps
  - Plot frequency: every 200 steps
  - Log directory: ./outputs/logs/SeqSetVAE_with_metrics_monitoring/version_0/posterior_metrics

📊 Step 20: KL=0.123456, Var=0.234567, Active=0.789, Recon=1.234567
📊 Step 40: KL=0.098765, Var=0.198765, Active=0.756, Recon=1.123456
...
📊 Posterior metrics plot saved: ./outputs/logs/SeqSetVAE_with_metrics_monitoring/version_0/posterior_metrics/posterior_metrics_step_200_20231201_143022.png
```

### 训练结束后的总结
```
📊 Posterior metrics monitoring summary:
===========================================
Total steps monitored: 1500
Update frequency: every 20 steps
Plot frequency: every 200 steps
Log directory: ./outputs/logs/SeqSetVAE_with_metrics_monitoring/version_0/posterior_metrics
```