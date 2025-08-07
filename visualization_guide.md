# 模型可视化与分析指南

本指南介绍如何使用提供的可视化脚本来分析您的VAE模型训练情况。

## 🔧 可用工具

### 1. **训练曲线分析** (`analyze_training_curves.py`)
从TensorBoard日志中提取并可视化训练指标。

```bash
# 基本用法
python analyze_training_curves.py --log_dir /path/to/tensorboard/logs --save_dir ./training_analysis

# 示例
python analyze_training_curves.py \
    --log_dir /home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/logs/SeqSetVAE_with_collapse_detection/version_0 \
    --save_dir ./my_training_analysis
```

**生成的文件：**
- `loss_curves.png` - 各种损失函数的训练曲线
- `performance_metrics.png` - AUC、AUPRC、准确率曲线
- `training_dynamics.png` - β退火、损失权重、KL-重建权衡
- `collapse_analysis.txt` - 后验塌缩分析总结

### 2. **模型隐空间可视化** (`visualize_model.py`)
分析训练好的模型的隐空间表示和重建质量。

```bash
# 基本用法
python visualize_model.py --checkpoint /path/to/model.ckpt --save_dir ./visualizations

# 完整示例
python visualize_model.py \
    --checkpoint /home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/last.ckpt \
    --data_dir /home/sunx/data/aiiih/data/mimic/processed/patient_ehr \
    --params_map_path /home/sunx/data/aiiih/data/mimic/processed/stats.csv \
    --label_path /home/sunx/data/aiiih/data/mimic/processed/oc.csv \
    --save_dir ./model_visualizations \
    --max_batches 20 \
    --batch_size 32
```

**生成的文件：**
- `posterior_collapse_analysis.png` - 后验塌缩详细分析
- `latent_space_umap.png` - 隐空间UMAP可视化
- `reconstruction_errors.png` - 重建误差分布
- `latent_correlation.png` - 隐变量相关性热图

## 📊 关键指标解读

### 损失函数指标
1. **KL散度 (`train_kl`/`val_kl`)**
   - 正常范围：0.1-10
   - < 0.01：可能发生后验塌缩
   - > 100：可能过度正则化

2. **重建损失 (`train_recon`/`val_recon`)**
   - 应该持续下降
   - 停滞可能表示模型容量不足

3. **预测损失 (`train_pred`/`val_pred`)**
   - 反映下游任务性能
   - 与AUC等指标相关

### 性能指标
1. **AUC (Area Under ROC Curve)**
   - 0.5：随机猜测
   - 0.7-0.8：良好
   - > 0.9：优秀

2. **AUPRC (Area Under PR Curve)**
   - 对不平衡数据集更敏感
   - 通常低于AUC

### 训练动态
1. **β退火曲线**
   - 应该从0逐渐增加到max_beta
   - 过快增加可能导致训练不稳定

2. **损失权重分布**
   - 显示各损失项的相对重要性
   - 帮助理解模型优化重点

## 🚨 问题诊断

### 1. 后验塌缩
**症状：**
- KL散度 < 0.01
- 大量隐维度失活
- 性能停滞

**解决方案：**
- 降低β值
- 使用free bits
- 增加warmup步数

### 2. 过拟合
**症状：**
- 训练损失下降，验证损失上升
- 训练/验证性能差距大

**解决方案：**
- 增加dropout
- 减少模型容量
- 数据增强

### 3. 欠拟合
**症状：**
- 训练和验证损失都很高
- 性能指标低

**解决方案：**
- 增加模型容量
- 调整学习率
- 检查数据预处理

## 💡 使用建议

1. **定期监控**：每隔几个epoch运行一次分析
2. **对比实验**：保存不同配置的分析结果进行对比
3. **早期诊断**：在训练初期就检查是否有异常模式

## 📝 示例工作流

```bash
# 1. 开始训练
python train_with_collapse_detection.py --fast_detection

# 2. 训练过程中，分析训练曲线
python analyze_training_curves.py \
    --log_dir ./outputs/logs/SeqSetVAE_with_collapse_detection/version_0

# 3. 训练完成后，分析模型
python visualize_model.py \
    --checkpoint ./outputs/checkpoints/last.ckpt \
    --save_dir ./final_analysis

# 4. 如果发现问题，调整配置重新训练
```

## 🔗 相关文件
- `train_config.py` - 模型配置
- `posterior_collapse_detector.py` - 实时塌缩检测
- `collapse_visualizer.py` - 实时监控面板