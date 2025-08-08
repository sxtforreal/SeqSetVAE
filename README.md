# SeqSetVAE Training and Utilities (简体中文)

本仓库包含用于 SeqSetVAE 的训练脚本、数据加载、可视化与训练分析工具。

## 依赖
- Python 3.10+
- PyTorch / Lightning / TorchMetrics / Transformers / Pandas / PyArrow / Tqdm / Seaborn / Matplotlib / Scikit-learn / UMAP

## 数据准备
- 患者级数据目录：`patient_ehr/{train,valid,test}/*.parquet`
- 变量统计：`stats.csv`（包含每个 variable 的 mean/std）
- 标签文件：`oc.csv`（列：`ts_id`, `in_hospital_mortality`）
- 变量嵌入：`../cached.csv`（用于构建变量嵌入向量）

## 训练
使用 `train_optimized.py`：

```bash
python train_optimized.py \
  --data_dir /path/to/patient_ehr \
  --params_map_path /path/to/stats.csv \
  --label_path /path/to/oc.csv \
  --output_dir /path/to/outputs \
  --batch_size 4 \
  --max_epochs 2
```

要点：
- 默认监控 `val_auc` 做早停与保存最优模型。
- 验证集采样比例已设为 `limit_val_batches=0.3`，避免小切片内单一类别导致 AUPRC 为 NaN。
- 若仍出现 AUPRC 为 NaN，可将 `limit_val_batches` 提升至 `1.0` 或确保验证集分层分布。

## 可视化（单样本静态分析 / 实时监控）
使用 `comprehensive_visualizer.py`：

- 静态分析单个样本：
```bash
python comprehensive_visualizer.py \
  --mode static \
  --checkpoint /path/to/checkpoint.ckpt \
  --data_dir /path/to/patient_ehr \
  --params_map_path /path/to/stats.csv \
  --label_path /path/to/oc.csv \
  --save_dir ./visualizations \
  --sample_idx 0
```

- 实时监控（基于 `posterior_collapse_detector.py` 生成的 JSON 日志）：
```bash
python comprehensive_visualizer.py --mode realtime --log_dir ./outputs/posterior_metrics
```

## 训练后分析（TensorBoard 或回调日志）
使用 `comprehensive_analyzer.py`：
```bash
python comprehensive_analyzer.py \
  --mode tensorboard \
  --log_dir /path/to/outputs/logs \
  --save_dir ./analysis_results
```

## 主要脚本/模块
- `train_optimized.py`：训练入口，包含自适应设备配置与回调。
- `model.py`：`SeqSetVAE` 模型，含重建与分类、多项监控指标（AUC/AUPRC/Accuracy）。
- `dataset.py`：`SeqSetVAEDataModule` 与相关数据集/拼接逻辑。
- `posterior_collapse_detector.py`：后验坍塌监控回调（训练时可选）。
- `comprehensive_visualizer.py`：单样本静态可视化与实时监控面板。
- `comprehensive_analyzer.py`：训练后综合分析（从 TensorBoard 事件或监控日志）。

## 常见问题
- 验证集 AUPRC 显示 NaN：通常由验证切片中只含单一类别引起。增大验证覆盖（将 `limit_val_batches` 提高或设为 1.0），或使用分层验证集。

## 许可证
详见 `LICENSE`。