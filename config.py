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
beta = 0.1  # 降低KL权重，从0.5改为0.1
lr = 1e-4
num_classes = 2
ff_dim = 256
transformer_heads = 2
transformer_layers = 2
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt"
w = 1.0  # 增加分类损失权重，从0.5改为1.0
free_bits = 0.1  # 降低free_bits，从0.2改为0.1
warmup_beta = True  # 添加beta warmup
max_beta = 0.1  # beta的最大值
beta_warmup_steps = 5000  # beta warmup步数
kl_annealing = True  # 添加KL annealing
gradient_clip_val = 0.5  # 降低梯度裁剪值

# Logger
name = "SeqSetVAE-v2"
log_every_n_steps = 200
ckpt_every_n_steps = 200

# Compute related
accelerator = "gpu"
devices = 1
precision = "16-mixed"
