实验计划脚本（Stage 1–3）

数据期望格式：单个 `.npz` 文件包含键：`mu[B,T,D]`，`logvar[B,T,D]`，`dt[B,T,1]`（或 `[B,T]`），`mask[B,T]`（0/1），`y[B]`。

快速开始：

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 运行 Stage 1：线性探针与顺序/不确定性检验

```bash
python stage1.py --data_path /path/to/data.npz --out_dir runs/stage1
```

3) 运行 Stage 2：候选路线最小实现对比（从 A–F 中选择）

```bash
python stage2.py --data_path /path/to/data.npz --out_dir runs/stage2 --routes A,B,C
```

4) 运行 Stage 3：诊断性消融（近期 vs 远期、顺序敏感、不确定性贡献/宽度）

```bash
python stage3.py --data_path /path/to/data.npz --out_dir runs/stage3 --best_routes A,C
```

备注
- 主指标为 AUPRC（Val 上早停）。同时报告 AUROC、Recall@Precision≥{0.6,0.7}、Precision@Top-k 和 ECE（可选）。
- 所有脚本均支持多次随机种子重复（默认 3 次）并输出表格汇总。
 - 数据键名可通过简单修改 `exp/utils.py` 与 `exp/data.py` 适配。
# SeqSetVAE

Unified training interface with two modes and **advanced VAE feature fusion** (2024 update):

## Training Modes

- Pretrain (reconstruction+KL only, strict causal):
  ```bash
  python3 train.py --mode pretrain \
    --data_dir /path/to/patient_ehr \
    --params_map_path /path/to/stats.csv \
    --label_path /path/to/oc.csv
  ```
  - Batch: single patient per batch
  - Monitor: val_loss
  - Checkpoints: best + last, validated every 10% epoch

- Finetune (classification only with **uncertainty-aware VAE fusion**):
  ```bash
  python3 train.py --mode finetune \
    --pretrained_ckpt /path/to/pretrain.ckpt \
    --batch_size 8 \
    --data_dir /path/to/patient_ehr \
    --params_map_path /path/to/stats.csv \
    --label_path /path/to/oc.csv \
    --vae_fusion_method enhanced_concat \
    --estimate_uncertainty
  ```
  - Batch: multi-patient
  - Backbone frozen and set to eval; only `cls_head` is trained with higher LR
  - Monitor: val_auc
  - Checkpoints: best + last, validated every 10% epoch

## 🎯 Simple & Effective VAE Feature Fusion

Based on empirical evidence that **simple approaches often work best** in medical domains, we provide two optimized methods:

### Available Fusion Methods

1. **Simple Concatenation** (`simple_concat`) - **Recommended for stability**
   - Basic mean + std concatenation
   - Proven effective, minimal complexity
   - Best for medical data where robustness > complexity

2. **Enhanced Concatenation** (`enhanced_concat`) - **Try if you need more**
   - Adds 2 key uncertainty features: total variance + mean magnitude
   - Minimal overhead, targeted improvements
   - Only use if simple version is not sufficient

### Usage Examples

#### Recommended: Simple & Robust
```bash
python3 train.py --mode finetune \
  --pretrained_ckpt model.ckpt \
  --vae_fusion_method simple_concat
```

#### Optional: Enhanced with Minimal Uncertainty
```bash
python3 train.py --mode finetune \
  --pretrained_ckpt model.ckpt \
  --vae_fusion_method enhanced_concat \
  --estimate_uncertainty
```

### 📊 Design Philosophy

- **Simplicity First**: Avoid over-engineering for medical data
- **Proven Methods**: Focus on well-established techniques
- **Minimal Overhead**: Add complexity only when clearly beneficial
- **Medical-Friendly**: Prioritize interpretability and robustness

### 🔧 Technical Details

- **Feature Dimensionality**: 
  - Simple: 2×latent_dim (mean + std)
  - Enhanced: 2×latent_dim + 2 (+ total variance + mean magnitude)
- **Uncertainty**: Optional light dropout (0.1) for basic regularization
- **Temperature Scaling**: Single parameter for calibration (if uncertainty enabled)

Notes:
- Time/position encoding is unified across modes with robust design: sinusoidal index + relative time buckets + ALiBi time bias, strict causal mask.
- For dataset structure and collate details, see `dataset.py`.
- **Focal loss** remains the only classification loss for handling class imbalance.