import torch

# Set device cuda for GPU if it is available, otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
min_epochs = 1
max_epochs = 2
input_dim = 768
reduced_dim = 256
latent_dim = 256
levels = 2
heads = 2
m = 16
beta = 0.5
lr = 1e-4
num_classes = 2
ff_dim = 256
transformer_heads = 2
transformer_layers = 2
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt"
w = 0.5
free_bits = 0.2

# 后验坍缩防护参数
use_spectral_norm = True
beta_strategy = "cyclical"  # "linear", "cyclical", "sigmoid"
min_beta = 0.0
cycle_length = 5000
beta_warmup_steps = 1000
use_tc_decomposition = False
pc_threshold = 0.1

# Logger
name = "SeqSetVAE-v3-improved"
log_every_n_steps = 200
ckpt_every_n_steps = 200

# Compute related
accelerator = "gpu"
devices = 1
precision = "16-mixed"
