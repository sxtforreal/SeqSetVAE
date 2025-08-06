import torch

# Device Configuration
# Set device to CUDA for GPU if available, otherwise run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Training Hyperparameters
min_epochs = 1                    # Minimum number of training epochs
max_epochs = 2                    # Maximum number of training epochs
input_dim = 768                   # Input embedding dimension (medical variable embeddings)
reduced_dim = 256                 # Reduced dimension after dimension reduction layer
latent_dim = 256                  # Latent space dimension for VAE
levels = 2                        # Number of encoder/decoder levels in SetVAE
heads = 2                         # Number of attention heads in multi-head attention
m = 16                           # Number of inducing points in ISAB (Induced Set Attention Block)
beta = 0.1                       # KL divergence weight (reduced from 0.5 to prevent posterior collapse)
lr = 1e-4                        # Learning rate for optimizer
num_classes = 2                  # Number of output classes for classification (binary classification)
ff_dim = 256                     # Feed-forward network dimension in transformer
transformer_heads = 2            # Number of attention heads in transformer encoder
transformer_layers = 2           # Number of transformer encoder layers

# Model Checkpoint and Pretrained Weights
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt"

# Loss Function Weights
w = 1.0                          # Classification loss weight (increased from 0.5 to emphasize classification)
free_bits = 0.1                  # Free bits for KL divergence (reduced from 0.2 to prevent collapse)

# Training Regularization and Optimization
warmup_beta = True               # Enable beta warmup for KL annealing
max_beta = 0.1                   # Maximum value of beta during warmup
beta_warmup_steps = 5000         # Number of steps for beta warmup
kl_annealing = True              # Enable KL annealing schedule
gradient_clip_val = 0.5          # Gradient clipping value (reduced to prevent exploding gradients)

# Logging Configuration
name = "SeqSetVAE-v2"            # Experiment name for logging
log_every_n_steps = 200          # Log metrics every N training steps
ckpt_every_n_steps = 200         # Save checkpoint every N training steps

# Compute Configuration
accelerator = "gpu"              # Training accelerator type (gpu/cpu/tpu)
devices = 1                      # Number of devices to use for training
precision = "16-mixed"           # Mixed precision training (16-bit float + 32-bit float)
