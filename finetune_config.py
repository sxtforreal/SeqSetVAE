"""
Optimized configuration for SeqSetVAE finetuning stage
Specialized configuration optimized for classification head finetuning
"""

import os
from config import *  # Import base config

# Override specific settings for better finetune performance

# ====== Key Improvement Settings ======

# 1. Optimized learning rate settings (for fully frozen backbone)
cls_head_lr = 3e-4  # Increased classification head LR for better convergence
lr = 1e-4  # Backbone LR (will be frozen, but used for initialization)

# 2. Better regularization (compatible with VAE feature fusion)
cls_head_weight_decay = (
    0.01  # Moderate weight decay, avoid over-regularizing VAE features
)
dropout_rate = 0.2  # Moderate dropout, allows VAE uncertainty information to flow

# 3. Optimized training strategy
warmup_steps = 500  # Reduced warmup steps for faster effective training
max_epochs = 15  # Increased maximum training epochs
early_stopping_patience = 6  # Reduced early stopping patience to prevent overfitting

# 4. Focal loss settings (always enabled)
focal_alpha = 0.25  # Adjusted focal loss parameters
focal_gamma = 2.0
label_smoothing = 0.05  # Added label smoothing

# 5. Optimized batch and validation settings
batch_size = 8  # Increased batch size for better GPU utilization
val_check_interval = 0.25  # Less frequent validation for speed
limit_val_batches = 1.0  # Use all validation data

# 6. Feature extraction optimization
feature_extraction_mode = "stable"  # Use stable feature extraction mode

# 7. Gradient-related settings
gradient_clip_val = 0.5  # Moderate gradient clipping
accumulate_grad_batches = 2  # Gradient accumulation

# 8. Learning rate scheduling
scheduler_type = "cosine_with_restarts"
scheduler_patience = 3
scheduler_factor = 0.7
scheduler_min_lr = 1e-6

# ====== Monitoring and Logging ======
monitor_metric = "val_auc"
monitor_mode = "max"
log_every_n_steps = 50

# ====== Data Augmentation (if applicable) ======
# Usually don't need much data augmentation in classification finetuning stage
data_augmentation = False
noise_std = 0.0  # Disable noise injection

print("🎯 Finetune config loaded with optimized settings for AUC/AUPRC performance")
print(f"   - Classification head LR: {cls_head_lr}")
print(f"   - Weight decay: {cls_head_weight_decay}")
print(f"   - Focal loss: alpha={focal_alpha}, gamma={focal_gamma}")
print(f"   - Early stopping patience: {early_stopping_patience}")
