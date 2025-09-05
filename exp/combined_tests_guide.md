# Combined Feasibility + Gaussian-MIL Ablation Tests

This run evaluates only the full history (time window = all). No 24h/72h truncation is used.

## Datasets and Inputs
- Cached per-set posteriors produced by `exp/cache_features.py` under `train/valid/test`.
- Each sample contains tensors: `mu [S,D]`, `logvar [S,D]`, `minutes [S]`, `label`, and a padding mask is derived at collate time.

## Metrics
- AUROC, AUPRC, and Recall at 95% specificity are reported.
- We also perform temperature calibration on the validation split and report calibrated metrics.

## Part A — Feasibility Suite (single pass, all-time)
1. Linear probe (last mu):
   - Selects the last non-padded time step's latent mean per sequence and fits a single linear layer.
2. Linear probe (last [mu||logvar]):
   - Concatenates last-step `mu` and `logvar` and trains a linear head.
3. Pooled MLPs over per-set posteriors: mean / Product-of-Experts (PoE) / W2:
   - Pools `mu` and `logvar` across sets via mean, precision-weighted PoE, or Wasserstein-2 style average; then feeds `[mu_pool||logvar_pool]` to a small MLP.
4. Closed-form Logistic–Gaussian on PoE aggregate:
   - Applies a MacKay-style scaling for a logistic head directly on PoE `(mu, logvar)` aggregates.
5. Baseline Gaussian-MIL head:
   - Uses the project’s `model.GaussianMILHead` with time as an input, trained with BCE-with-logits.
6. Light Monte-Carlo (MC) sampling + mean pooling + MLP:
   - Draws K samples from each per-set Gaussian, mean-pools sampled `z`, and feeds to an MLP; logits are averaged over K.

## Part B — Gaussian-MIL Ablations (all-time)
Each ablation modifies the Gaussian-MIL variant head while keeping the dataset and training loop fixed. Results are averaged across multiple seeds.

- baseline_mean_or: Mean aggregation mixed with a Noisy-OR branch.
- lse_or: LogSumExp aggregation mixed with Noisy-OR.
- pmean_or: Learnable power-mean aggregation mixed with Noisy-OR.
- topk2_mean_or / topk3_mean_or: Keep only the top-2/top-3 instances according to the gate, then normalize weights and aggregate with mean, mixed with Noisy-OR.
- precision_attention: Interpolates gate weights with precision-based attention (`mu^2/var`).
- snr_features: Augments gate inputs with three SNR-like features: `||mu||/sqrt(D)`, `mean(logvar)`, and `mean(mu^2/var)`.
- group_lasso: Adds a group lasso penalty on the first gate layer’s input groups (controlled by `--group_lasso_lambda`).
- deepsets_ctx: Adds a DeepSets context vector computed from per-instance `[mu||logvar]`.
- focal_bce_hinge: Optionally swaps BCE for focal loss (controlled by flags) and adds a small hinge penalty nudging negatives below a batch-wise specificity threshold.

## Outputs
- JSON: `combined_results.json` aggregates all metrics for both Part A and Part B.
- This guide summarizes what was tested.