#!/usr/bin/env python3
"""
Utility script to identify and clean up checkpoints that were saved with save_weights_only=True.
These checkpoints cannot be used to resume training with optimizer state.
"""

import os
import torch
import argparse
from pathlib import Path

def check_checkpoint(checkpoint_path):
    """Check if a checkpoint contains optimizer states."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        has_optimizer = 'optimizer_states' in checkpoint
        has_scheduler = 'lr_schedulers' in checkpoint
        has_epoch = 'epoch' in checkpoint
        has_global_step = 'global_step' in checkpoint
        
        return {
            'path': checkpoint_path,
            'has_optimizer': has_optimizer,
            'has_scheduler': has_scheduler,
            'has_epoch': has_epoch,
            'has_global_step': has_global_step,
            'is_valid_for_resume': has_optimizer and has_scheduler
        }
    except Exception as e:
        return {
            'path': checkpoint_path,
            'error': str(e),
            'is_valid_for_resume': False
        }

def find_checkpoints(directory):
    """Find all checkpoint files in a directory."""
    checkpoint_extensions = ['.ckpt', '.pth', '.pt']
    checkpoints = []
    
    for ext in checkpoint_extensions:
        checkpoints.extend(Path(directory).rglob(f'*{ext}'))
    
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description='Check and clean up invalid checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints to check')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove checkpoints that cannot be used for resuming training')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ Directory not found: {args.checkpoint_dir}")
        return
    
    print(f"🔍 Checking checkpoints in: {args.checkpoint_dir}")
    print("=" * 60)
    
    checkpoints = find_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print("No checkpoint files found.")
        return
    
    valid_checkpoints = []
    invalid_checkpoints = []
    error_checkpoints = []
    
    for checkpoint_path in checkpoints:
        result = check_checkpoint(checkpoint_path)
        
        if 'error' in result:
            error_checkpoints.append(result)
            print(f"❌ {checkpoint_path.name}: Error - {result['error']}")
        elif result['is_valid_for_resume']:
            valid_checkpoints.append(result)
            print(f"✅ {checkpoint_path.name}: Valid for resuming training")
        else:
            invalid_checkpoints.append(result)
            print(f"⚠️  {checkpoint_path.name}: Invalid for resuming training (weights only)")
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"  - Total checkpoints found: {len(checkpoints)}")
    print(f"  - Valid for resuming: {len(valid_checkpoints)}")
    print(f"  - Invalid (weights only): {len(invalid_checkpoints)}")
    print(f"  - Error loading: {len(error_checkpoints)}")
    
    if invalid_checkpoints:
        print(f"\n⚠️  Found {len(invalid_checkpoints)} checkpoints that cannot be used for resuming training:")
        for cp in invalid_checkpoints:
            print(f"    - {cp['path']}")
        
        if args.cleanup:
            print(f"\n🗑️  {'DRY RUN: Would remove' if args.dry_run else 'Removing'} invalid checkpoints...")
            for cp in invalid_checkpoints:
                if args.dry_run:
                    print(f"    Would remove: {cp['path']}")
                else:
                    try:
                        os.remove(cp['path'])
                        print(f"    Removed: {cp['path']}")
                    except Exception as e:
                        print(f"    Failed to remove {cp['path']}: {e}")
        elif not args.dry_run:
            print("\n💡 To remove invalid checkpoints, run with --cleanup flag")
    
    if valid_checkpoints:
        print(f"\n✅ Valid checkpoints for resuming training:")
        for cp in valid_checkpoints:
            print(f"    - {cp['path']}")

if __name__ == "__main__":
    main()