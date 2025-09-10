### SeqSetVAE-PoE (Pretraining)

An updated pretraining repo that keeps the existing data format (per-patient Parquet) and the SetVAE + causal Transformer backbone, while adding:

- Product-of-Experts (PoE): combine set posterior q_x(z_t|x_t) with a causal prior p(z_t|z_{<t}, \u0394t) predicted by the Transformer.
- Stale-dropout: dropout only on carry-forward values (LVCF) to avoid over-relying on imputation.
- Set-MAE masking: optional random masking of a subset of events inside each set during training.
- Age and carry indicators added as two extra channels to the variable embedding; data remains in Parquet for fast I/O.

This repo intentionally removes multi-type decoders and random training windows. It uses the same decoder as before.

#### Layout

- `config.py`: Hyperparameters and paths.
- `dataset.py`: DataModule that reads per-patient Parquets and emits tensors including `carry_mask` and augmented embeddings.
- `modules.py`: SetVAE components copied from the original codebase.
- `model.py`: `PoESeqSetVAEPretrain` implementing PoE + stale-dropout.
- `train_pretrain.py`: CLI to run pretraining.
- `LVCF.py`: Build expanded Parquet with LVCF (bag_exp), `age`, `is_carry`, `set_index`, map event embeddings, and normalize values.

#### Quickstart

1) Prepare expanded Parquet (optional if you already have it):

```bash
python LVCF.py \
  --input_dir /path/to/patient_ehr \
  --output_dir /path/to/patient_expanded \
  --event_emb_csv /path/to/event_name_to_768d.csv \
  --value_stats_csv /path/to/event_value_mean_std.csv \
  --lvcf_minutes 2880 \
  --smoke
```

2) Pretrain with PoE:

```bash
python train_pretrain.py \
  --data_dir /path/to/patient_expanded \
  --params_map_path /path/to/stats.csv \
  --batch_size 4 --max_epochs 50
```

Parquet remains the storage format for fastest dataloading. The SetVAE, causal Transformer, and decoder are preserved.

### SetVAE-only Pretraining

This trains only the SetVAE on LVCF-expanded sets with a truthful, task-agnostic objective and small, mechanism-aligned perturbations.

Objective:
- Reconstruction: permutation-invariant Chamfer loss on x_target = normalize(var) * val
- KL: q(z|x) vs N(0,I) with free-bits and beta warmup

Training-time perturbations (all sets):
- Value dropout: stronger on carried tokens (p_stale≈0.5), light on live tokens (p_live≈0.05)
- Token masking (Set-MAE): for N<=5, mask 1 token with prob 0.4; for larger sets, mask ceil(0.15*N), capped at 2
- Additive value noise: Normal(0, 0.07)
- Directional jitter on normalized variable vectors: Normal(0, 0.01) then renormalize
- Decoder noise during training: 0.3 (eval: 0–0.05)

Run:
```bash
python train_setvae.py \
  --data_dir /path/to/patient_expanded \
  --batch_size 4 --max_epochs 50 \
  --p_stale 0.5 --p_live 0.05 \
  --set_mae_ratio 0.15 --small_set_mask_prob 0.4 --small_set_threshold 5 --max_masks_per_set 2 \
  --val_noise_std 0.07 --dir_noise_std 0.01 \
  --warmup_beta --max_beta 0.2 --free_bits 0.05
```

Outputs:
- TensorBoard logs under `./outputs/SetVAE-Only-PT/` (default)
- Best checkpoint `setvae_pretrain.ckpt`

Usage after pretraining:
- Load the SetVAE encoder/decoder and freeze them for likelihood estimation. Use per-set NELBO (Recon + beta*KL) or per-step KL between q_x(z|x) and a separate prior to trigger anomaly alarms when the distributional gap is large.

