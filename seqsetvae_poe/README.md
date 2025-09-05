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
- `preprocess_expand.py`: Build expanded Parquet with LVCF (bag_exp), `age`, `is_carry`, `set_index` from raw per-patient Parquet.

#### Quickstart

1) Prepare expanded Parquet (optional if you already have it):

```bash
python preprocess_expand.py \
  --input_dir /path/to/patient_ehr \
  --output_dir /path/to/patient_expanded \
  --lvcf_hours 48
```

2) Pretrain with PoE:

```bash
python train_pretrain.py \
  --data_dir /path/to/patient_expanded \
  --params_map_path /path/to/stats.csv \
  --batch_size 4 --max_epochs 50
```

Parquet remains the storage format for fastest dataloading. The SetVAE, causal Transformer, and decoder are preserved.

