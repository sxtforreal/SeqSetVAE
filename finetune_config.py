"""
Optimized configuration for SeqSetVAE finetuning stage
Specialized configuration optimized for classification head finetuning
"""

import os
from config import *  # Import base config

# Override specific settings for better finetune performance

# ====== Key Improvement Settings ======

# 1. ENHANCED learning rate settings for advanced classifier
cls_head_lr = 8e-4  # Higher LR for complex classifier architecture
lr = 2e-4  # Slightly higher backbone LR for initialization

# 2. Optimized regularization for advanced architecture
cls_head_weight_decay = 0.005  # Reduced for complex architecture
dropout_rate = 0.15  # Optimized dropout for attention mechanisms

# 3. Enhanced training strategy for better convergence
warmup_steps = 300  # Faster warmup for advanced architecture
max_epochs = 25  # More epochs for complex model
early_stopping_patience = 8  # Increased patience for better convergence

# 4. IMPROVED Focal loss settings for class imbalance
focal_alpha = 0.20  # Optimized for better precision-recall balance
focal_gamma = 2.5  # Higher gamma for harder negative mining
label_smoothing = 0.03  # Reduced label smoothing

# 5. Enhanced batch and validation settings
batch_size = 6  # Slightly reduced for complex architecture
val_check_interval = 0.2  # More frequent validation for monitoring
limit_val_batches = 1.0  # Use all validation data

# 6. Feature extraction optimization
feature_extraction_mode = "stable"  # Use stable feature extraction mode

# 7. Enhanced gradient-related settings
gradient_clip_val = 0.3  # Tighter clipping for complex architecture
accumulate_grad_batches = 3  # Increased accumulation for stability

# 8. Advanced learning rate scheduling
scheduler_type = "cosine_with_restarts"
scheduler_patience = 2  # Faster adaptation
scheduler_factor = 0.5  # More aggressive decay
scheduler_min_lr = 5e-7  # Lower minimum LR

# 9. NEW: Advanced optimization settings
use_swa = True  # Stochastic Weight Averaging for better generalization
swa_start_epoch = 10  # Start SWA after initial convergence
use_lookahead = False  # Disable for medical data stability

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
