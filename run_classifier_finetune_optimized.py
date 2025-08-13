#!/usr/bin/env python3
"""
Optimized classifier head fine-tuning script with improved hyperparameters
"""

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the optimized configuration
import config_classifier_finetune as config

# Import the fine-tuning script
from finetune_classifier_head import main as finetune_main

def main():
    """Run classifier head fine-tuning with optimized parameters"""
    
    # Set up argument parser with optimized defaults
    parser = argparse.ArgumentParser(description="Optimized classifier head fine-tuning")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr")
    parser.add_argument("--params_map_path", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv")
    parser.add_argument("--label_path", type=str, 
                       default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv")
    
    # Training arguments with optimized defaults
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--head_lr", type=float, default=config.head_lr, 
                       help=f"Learning rate for classification head (default: {config.head_lr})")
    parser.add_argument("--max_epochs", type=int, default=config.max_epochs)
    parser.add_argument("--precision", type=str, default=config.precision)
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, default=config.pretrained_ckpt,
                       help="Path to pretrained SeqSetVAE checkpoint")
    
    # Output arguments
    parser.add_argument("--output_root_dir", type=str, 
                       default="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs")
    
    args = parser.parse_args()
    
    # Print configuration summary
    print("=" * 60)
    print("OPTIMIZED CLASSIFIER HEAD FINE-TUNING CONFIGURATION")
    print("=" * 60)
    print(f"Learning rate: {args.head_lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Precision: {args.precision}")
    print(f"Gradient clip: {config.gradient_clip_val}")
    print(f"LR scheduler factor: {config.scheduler_factor}")
    print(f"LR scheduler patience: {config.scheduler_patience}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print(f"Focal loss gamma: {config.focal_gamma}")
    print("=" * 60)
    
    # Set environment variables for the fine-tuning script
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Run the fine-tuning with our arguments
    sys.argv = [
        'finetune_classifier_head.py',
        '--data_dir', args.data_dir,
        '--params_map_path', args.params_map_path,
        '--label_path', args.label_path,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--head_lr', str(args.head_lr),
        '--max_epochs', str(args.max_epochs),
        '--precision', args.precision,
        '--checkpoint', args.checkpoint,
        '--output_root_dir', args.output_root_dir,
    ]
    
    # Run the main fine-tuning function
    finetune_main()

if __name__ == "__main__":
    main()