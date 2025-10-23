#!/usr/bin/env bash
set -euo pipefail

# Usage: bash exp/run_transformer_baseline.sh \
#   /path/to/setvae.ckpt \
#   /path/to/data_dir \
#   /path/to/labels.csv \
#   /path/to/output_dir
#
# Notes:
# - data_dir should contain train/ valid/ test/ parquet files with LVCF schema
# - labels.csv must contain columns: ts_id,in_hospital_mortality

STAGEA_CKPT=${1:-""}
DATA_DIR=${2:-""}
LABELS=${3:-""}
OUT_DIR=${4:-"./output"}

if [[ -z "${STAGEA_CKPT}" || -z "${DATA_DIR}" || -z "${LABELS}" ]]; then
  echo "Usage: $0 STAGEA_CKPT DATA_DIR LABELS_CSV [OUT_DIR]" >&2
  exit 1
fi

python -u seqsetvae_poe/train.py \
  --branch discriminative \
  --mode transformer \
  --setvae_ckpt "${STAGEA_CKPT}" \
  --data_dir "${DATA_DIR}" \
  --label_csv "${LABELS}" \
  --batch_size 64 \
  --num_workers 2 \
  --max_epochs 20 \
  --lr 3e-4 \
  --dropout 0.2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 3 \
  --output_dir "${OUT_DIR}" \
  --precision 16-mixed
