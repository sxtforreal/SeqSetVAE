#!/usr/bin/env python3
"""
Configuration file specifically optimized for classifier head fine-tuning
"""

# Model architecture parameters (inherited from main config)
input_dim = 768
reduced_dim = 256
latent_dim = 256
levels = 2
heads = 2
m = 16
beta = 0.1
num_classes = 2

# Transformer architecture
ff_dim = 512
transformer_heads = 8
transformer_layers = 4
cls_head_layers = [256, 128, 64]
cls_dropout = 0.2
transformer_dropout = 0.15
post_norm = True
feature_fusion = True

# Classification head fine-tuning specific parameters
head_lr = 5e-4  # Reduced from 1e-3 for more stable training
head_weight_decay = 0.001  # Reduced from 0.01 for classification head
head_betas = (0.9, 0.999)
head_eps = 1e-8

# Learning rate scheduler for classification head
scheduler_factor = 0.7  # Reduce LR by 30% on plateau
scheduler_patience = 3  # Reduced from 5 for more responsive adjustment
scheduler_min_lr_factor = 1e-3  # Minimum LR = head_lr * 1e-3

# Focal loss parameters
use_focal_loss = True
focal_alpha = 0.35
focal_gamma = 2.5  # Reduced from 3.0 for more stable training

# Training parameters
max_epochs = 50  # Increased for better convergence
batch_size = 8
precision = "16-mixed"
gradient_clip_val = 0.05  # Reduced for classification head

# Early stopping
early_stopping_patience = 10
early_stopping_min_delta = 1e-4

# Validation and logging
val_check_interval = 0.25
log_every_n_steps = 25

# Model checkpoint path (update this to your actual checkpoint)
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SetVAE.ckpt"

# Output configuration
name = "SeqSetVAE-ClassifierHead"
model_name = name