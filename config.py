import os

# Lazy initialization of device configuration
_device_config = None

def get_optimal_device_config(verbose: bool = False):
    """
    Intelligently detect and return optimal device configuration
    Adaptive selection: use GPU if available, otherwise use CPU
    """
    try:
        import torch
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
            
            if verbose:
                print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Adjust configuration based on GPU memory
            if gpu_memory >= 16:  # 16GB+ GPU
                devices = min(gpu_count, 2)  # Use at most 2 GPUs
                precision = "16-mixed"
                batch_size_recommendation = 8
            elif gpu_memory >= 8:  # 8-16GB GPU
                devices = 1
                precision = "16-mixed"
                batch_size_recommendation = 4
            else:  # Less than 8GB GPU
                devices = 1
                precision = "32"  # Use 32-bit precision to avoid memory issues
                batch_size_recommendation = 2
                
            accelerator = "gpu"
            device = torch.device("cuda")
            
            if verbose:
                print(f"   - Using {devices} GPU(s)")
                print(f"   - Precision: {precision}")
                print(f"   - Recommended batch size: {batch_size_recommendation}")
            
        else:
            # CPU configuration
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            
            if verbose:
                print(f"💻 CPU detected: {cpu_count} cores")
            
            devices = 1
            accelerator = "cpu"
            precision = "32"  # Use 32-bit precision for CPU training
            device = torch.device("cpu")
            batch_size_recommendation = 1
            
            if verbose:
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
        
    except ImportError:
        # If torch is not available, return default CPU configuration
        if verbose:
            print("⚠️  PyTorch not available, using default CPU configuration")
        return {
            'device': 'cpu',
            'accelerator': 'cpu',
            'devices': 1,
            'precision': '32',
            'batch_size_recommendation': 1,
            'cuda_available': False
        }

def get_device_config():
    """Get device configuration with lazy initialization"""
    global _device_config
    if _device_config is None:
        _device_config = get_optimal_device_config(verbose=False)
    return _device_config

# Device configuration attributes
def get_device_config_attr():
    """Get device configuration with lazy initialization"""
    return get_device_config()

# Set device configuration as module attributes
device_config = get_device_config_attr()

# Extract commonly used attributes from device configuration
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

# Focal Loss Hyperparameters
use_focal_loss = True
# alpha for positive class in binary classification (set None to disable balancing)
focal_alpha = 0.25
focal_gamma = 2.0

if __name__ == "__main__":
    cfg = get_optimal_device_config(verbose=True)
    print("\nDevice configuration summary:")
    print(cfg)
