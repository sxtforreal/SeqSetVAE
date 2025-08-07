import torch
import os

def get_optimal_device_config():
    """
    智能检测并返回最优的设备配置
    自适应选择：如果有GPU就使用GPU，否则使用CPU
    """
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        # 获取GPU信息
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
        
        print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 根据GPU内存调整配置
        if gpu_memory >= 16:  # 16GB+ GPU
            devices = min(gpu_count, 2)  # 最多使用2个GPU
            precision = "16-mixed"
            batch_size_recommendation = 8
        elif gpu_memory >= 8:  # 8-16GB GPU
            devices = 1
            precision = "16-mixed"
            batch_size_recommendation = 4
        else:  # 小于8GB GPU
            devices = 1
            precision = "32"  # 使用32位精度避免内存不足
            batch_size_recommendation = 2
            
        accelerator = "gpu"
        device = torch.device("cuda")
        
        print(f"   - Using {devices} GPU(s)")
        print(f"   - Precision: {precision}")
        print(f"   - Recommended batch size: {batch_size_recommendation}")
        
    else:
        # CPU配置
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        print(f"💻 CPU detected: {cpu_count} cores")
        
        devices = 1
        accelerator = "cpu"
        precision = "32"  # CPU训练使用32位精度
        device = torch.device("cpu")
        batch_size_recommendation = 1
        
        print(f"   - Using CPU training")
        print(f"   - Precision: {precision}")
        print(f"   - Recommended batch size: {batch_size_recommendation}")
    
    return {
        'device': device,
        'accelerator': accelerator,
        'devices': devices,
        'precision': precision,
        'batch_size_recommendation': batch_size_recommendation,
        'cuda_available': cuda_available
    }

# 获取最优设备配置
device_config = get_optimal_device_config()

# Device Configuration
device = device_config['device']
accelerator = device_config['accelerator']
devices = device_config['devices']
precision = device_config['precision']

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
seed = 0

# Compute Configuration
accelerator = "gpu"              # Training accelerator type (gpu/cpu/tpu)
devices = 1                      # Number of devices to use for training
precision = "16-mixed"           # Mixed precision training (16-bit float + 32-bit float)
