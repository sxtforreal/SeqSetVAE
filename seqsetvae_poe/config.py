import os

# Core model dims (align with existing SetVAE/Transformer)
input_dim = 768
reduced_dim = 256
latent_dim = 256
levels = 2
heads = 2
m = 16

# Transformer
ff_dim = 512
transformer_heads = 8
transformer_layers = 4
transformer_dropout = 0.15

# Training
lr = 1e-4
beta = 0.1
free_bits = 0.03
warmup_beta = True
max_beta = 0.05
beta_warmup_steps = 8000
gradient_clip_val = 0.2

# Data
batch_size = 4
num_workers = 4

# PoE / Regularization switches
use_poe = True
stale_dropout_p = 0.2  # only applied to carry-forwarded values
set_mae_ratio = 0.0    # 0 to disable

# I/O
data_dir = os.environ.get("SEQSET_PTN_DATA", "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr_expanded")
params_map_path = os.environ.get("SEQSET_STATS", "/home/sunx/data/aiiih/data/mimic/processed/stats.csv")

name = "SeqSetVAE-PoE-PT"
