import os

# Lazy initialization of device configuration
_device_config = None


def get_optimal_device_config():
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
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_count > 0
                else 0
            )

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

            print(f"   - Using {devices} GPU(s)")
            print(f"   - Precision: {precision}")
            print(f"   - Recommended batch size: {batch_size_recommendation}")

        else:
            # CPU configuration
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()

            print(f"💻 CPU detected: {cpu_count} cores")

            devices = 1
            accelerator = "cpu"
            precision = "32"  # Use 32-bit precision for CPU training
            device = torch.device("cpu")
            batch_size_recommendation = 1

            print(f"   - Using CPU training")
            print(f"   - Precision: {precision}")
            print(f"   - Recommended batch size: {batch_size_recommendation}")

        return {
            "device": device,
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
            "batch_size_recommendation": batch_size_recommendation,
            "cuda_available": cuda_available,
        }

    except ImportError:
        # If torch is not available, return default CPU configuration
        print("⚠️  PyTorch not available, using default CPU configuration")
        return {
            "device": "cpu",
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32",
            "batch_size_recommendation": 1,
            "cuda_available": False,
        }


def get_device_config():
    """Get device configuration with lazy initialization"""
    global _device_config
    if _device_config is None:
        _device_config = get_optimal_device_config()
    return _device_config


# Device configuration attributes
def get_device_config_attr():
    """Get device configuration with lazy initialization"""
    return get_device_config()


# Set device configuration as module attributes
device_config = get_device_config_attr()

# Extract commonly used attributes from device configuration
device = device_config["device"]
accelerator = device_config["accelerator"]
devices = device_config["devices"]
precision = device_config["precision"]

# Model Training Hyperparameters
min_epochs = 1  # Minimum number of training epochs
max_epochs = 2  # Maximum number of training epochs
input_dim = 768  # Input embedding dimension (medical variable embeddings)
reduced_dim = 256  # Reduced dimension after dimension reduction layer
latent_dim = 256  # Latent space dimension for VAE
levels = 2  # Number of encoder/decoder levels in SetVAE
heads = 2  # Number of attention heads in multi-head attention
m = 16  # Number of inducing points in ISAB (Induced Set Attention Block)
beta = 0.1  # KL divergence weight (reduced from 0.5 to prevent posterior collapse)
lr = 1e-4  # Learning rate for optimizer
num_classes = 2  # Number of output classes for classification (binary classification)

# Enhanced model architecture parameters (for enhanced mode)
ff_dim = 256  # Feed-forward network dimension in transformer
transformer_heads = 2  # Number of attention heads in transformer encoder
transformer_layers = 2  # Number of transformer encoder layers

# Enhanced mode overrides (when --enhanced_mode is used)
enhanced_ff_dim = 512  # Enhanced feed-forward network dimension
enhanced_transformer_heads = 8  # Enhanced number of attention heads
enhanced_transformer_layers = 4  # Enhanced number of transformer layers
enhanced_cls_head_layers = [256, 128, 64]  # Enhanced classification head layers
enhanced_cls_dropout = 0.2  # Enhanced classification dropout
enhanced_transformer_dropout = 0.15  # Enhanced transformer dropout
enhanced_post_norm = True  # Enable post-transformer normalization
enhanced_feature_fusion = True  # Enable multi-scale feature fusion

# Use enhanced mode by default to match checkpoint
ff_dim = enhanced_ff_dim  # Use enhanced feed-forward dimension by default
transformer_heads = enhanced_transformer_heads  # Use enhanced transformer heads by default
transformer_layers = enhanced_transformer_layers  # Use enhanced transformer layers by default

# Model Checkpoint and Pretrained Weights
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt"

# Loss Function Weights
w = 2.0  # Classification loss weight (increased from 1.0 to emphasize classification)
free_bits = 0.05  # Free bits for KL divergence (reduced from 0.1 to prevent collapse)

# Enhanced mode loss weights (when --enhanced_mode is used)
enhanced_w = 3.0  # Enhanced classification loss weight
enhanced_free_bits = 0.03  # Enhanced KL divergence weight
enhanced_focal_alpha = 0.35  # Enhanced focal loss alpha
enhanced_focal_gamma = 3.0  # Enhanced focal loss gamma

# Use enhanced loss weights by default to match checkpoint
w = enhanced_w  # Use enhanced classification loss weight by default
free_bits = enhanced_free_bits  # Use enhanced KL divergence weight by default
focal_alpha = enhanced_focal_alpha  # Use enhanced focal loss alpha by default
focal_gamma = enhanced_focal_gamma  # Use enhanced focal loss gamma by default

# Training Regularization and Optimization
warmup_beta = True  # Enable beta warmup for KL annealing
max_beta = 0.05  # Maximum value of beta during warmup (reduced from 0.1)
beta_warmup_steps = 8000  # Number of steps for beta warmup (increased from 5000)
kl_annealing = True  # Enable KL annealing schedule
gradient_clip_val = 0.3  # Gradient clipping value (reduced from 0.5 to prevent exploding gradients)

# Enhanced mode training parameters
enhanced_gradient_clip_val = 0.2  # Enhanced gradient clipping
enhanced_weight_decay = 0.03  # Enhanced weight decay
enhanced_lr = 5e-5  # Enhanced learning rate
enhanced_scheduler_patience = 150  # Enhanced scheduler patience
enhanced_scheduler_factor = 0.6  # Enhanced scheduler factor
enhanced_scheduler_min_lr = 1e-6  # Enhanced scheduler min learning rate
enhanced_early_stopping_patience = 8  # Enhanced early stopping patience
enhanced_early_stopping_min_delta = 0.0005  # Enhanced early stopping min delta
enhanced_val_check_interval = 0.15  # Enhanced validation interval
enhanced_limit_val_batches = 0.6  # Enhanced validation batches
enhanced_save_top_k = 5  # Enhanced checkpoint saving
enhanced_monitor_metric = "val_auc"  # Enhanced monitoring metric

# Use enhanced training parameters by default to match checkpoint
lr = enhanced_lr  # Use enhanced learning rate by default
gradient_clip_val = enhanced_gradient_clip_val  # Use enhanced gradient clipping by default

# Logging Configuration
name = "SeqSetVAE-v3"  # Experiment/model name for unified output directories
model_name = name
log_every_n_steps = 200  # Log metrics every N training steps
ckpt_every_n_steps = 200  # Save checkpoint every N training steps
seed = 0

# Compute Configuration
accelerator = "gpu"  # Training accelerator type (gpu/cpu/tpu)
devices = 1  # Number of devices to use for training
precision = "16-mixed"  # Mixed precision training (16-bit float + 32-bit float)

# Focal Loss Hyperparameters
use_focal_loss = True
# alpha for positive class in binary classification (set None to disable balancing)
focal_alpha = enhanced_focal_alpha  # Use enhanced focal loss alpha by default
focal_gamma = enhanced_focal_gamma  # Use enhanced focal loss gamma by default
