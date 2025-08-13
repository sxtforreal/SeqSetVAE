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

# ============================================================================
# OPTIMIZED TRAINING CONFIGURATION FOR STABLE TRAINING
# ============================================================================

# Model Training Hyperparameters - OPTIMIZED FOR STABILITY
min_epochs = 1
max_epochs = 2
input_dim = 768
reduced_dim = 256
latent_dim = 256
levels = 2
heads = 2
m = 16
beta = 0.01  # REDUCED: Lower KL weight for more stable training
lr = 1e-5    # REDUCED: Much lower learning rate for stability
num_classes = 2

# Model architecture parameters - SIMPLIFIED FOR STABILITY
ff_dim = 256        # REDUCED: Smaller feed-forward dimension
transformer_heads = 4  # REDUCED: Fewer attention heads
transformer_layers = 2  # REDUCED: Fewer transformer layers
cls_head_layers = [128, 64]  # REDUCED: Simpler classification head
cls_dropout = 0.1   # REDUCED: Lower dropout
transformer_dropout = 0.1  # REDUCED: Lower dropout
post_norm = True
feature_fusion = False  # DISABLED: Reduce complexity

# Model Checkpoint and Pretrained Weights
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt"

# Loss Function Weights - OPTIMIZED FOR STABILITY
w = 1.0           # REDUCED: Lower classification weight
free_bits = 0.1   # INCREASED: More free bits to prevent posterior collapse
focal_alpha = 0.5  # BALANCED: Neutral alpha for focal loss
focal_gamma = 2.0  # REDUCED: Lower gamma for less aggressive focusing

# Training Regularization and Optimization - OPTIMIZED FOR STABILITY
warmup_beta = True
max_beta = 0.01    # REDUCED: Lower maximum beta
beta_warmup_steps = 15000  # INCREASED: Longer warmup for stability
kl_annealing = True
gradient_clip_val = 1.0    # INCREASED: Less aggressive gradient clipping
weight_decay = 0.01        # REDUCED: Lower weight decay
scheduler_patience = 200   # INCREASED: More patience
scheduler_factor = 0.8     # INCREASED: Gentler learning rate reduction
scheduler_min_lr = 1e-7    # REDUCED: Lower minimum learning rate
early_stopping_patience = 15  # INCREASED: More patience
early_stopping_min_delta = 0.001  # INCREASED: Larger improvement threshold
val_check_interval = 0.25  # REDUCED: Less frequent validation
limit_val_batches = 0.5    # REDUCED: Use less validation data
save_top_k = 3             # REDUCED: Save fewer checkpoints
monitor_metric = "val_loss"  # CHANGED: Monitor validation loss instead of AUC

# Logging Configuration
name = "SeqSetVAE-optimized-stable"
model_name = name
log_every_n_steps = 100    # REDUCED: More frequent logging
ckpt_every_n_steps = 100   # REDUCED: More frequent checkpointing
seed = 42                  # CHANGED: Different seed for reproducibility

# Compute Configuration
accelerator = "gpu"
devices = 1
precision = "16-mixed"

# Focal Loss Hyperparameters
use_focal_loss = True

# Additional stability parameters
use_gradient_accumulation = True
gradient_accumulation_steps = 4  # NEW: Gradient accumulation for stability
use_amp = True                   # NEW: Automatic mixed precision
use_deterministic = False        # NEW: Disable deterministic mode for speed
use_sync_batchnorm = False       # NEW: Disable sync batch norm for single GPU