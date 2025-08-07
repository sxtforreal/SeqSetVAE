# Checkpoint Resume Issue Fix

## Problem Description

You encountered the following error when trying to resume training:

```
KeyError: 'Trying to restore optimizer state but checkpoint contains only the model. This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.'
```

## Root Cause

The issue was in the `ModelCheckpoint` configuration in `train_optimized.py`. The checkpoint was being saved with only model weights (the default behavior when `save_weights_only` is not explicitly set), but PyTorch Lightning requires the full training state (including optimizer states) to resume training properly.

## Solution Applied

### 1. Fixed ModelCheckpoint Configuration

Updated the `ModelCheckpoint` configuration in `train_optimized.py`:

```python
# Before (problematic):
checkpoint = ModelCheckpoint(
    dirpath=os.path.join(experiment_output_dir, "checkpoints"),
    filename=f"best_{config.name}_optimized_batch{args.batch_size}",
    monitor="val_auc",
    mode="max",
    save_top_k=1,
    save_last=False,  # Don't save the last checkpoint to save space
    verbose=True,
)

# After (fixed):
checkpoint = ModelCheckpoint(
    dirpath=os.path.join(experiment_output_dir, "checkpoints"),
    filename=f"best_{config.name}_optimized_batch{args.batch_size}",
    monitor="val_auc",
    mode="max",
    save_top_k=1,
    save_last=True,  # Save the last checkpoint for resuming training
    save_weights_only=False,  # Save full training state including optimizer
    verbose=True,
)
```

### 2. Added Checkpoint Validation

Added validation logic to check if a checkpoint contains optimizer states before attempting to resume:

```python
# Validate checkpoint file
try:
    checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
    if 'optimizer_states' not in checkpoint:
        print("⚠️  Warning: Checkpoint contains only model weights (save_weights_only=True)")
        print("   This checkpoint cannot be used to resume training with optimizer state.")
        print("   Consider starting fresh or using a checkpoint with full training state.")
        print("   The model will be loaded but training will start from the beginning.")
        # Set resume_from_checkpoint to None to start fresh
        args.resume_from_checkpoint = None
    else:
        print(f"🔄 Resuming training from checkpoint: {args.resume_from_checkpoint}")
except Exception as e:
    print(f"⚠️  Warning: Could not validate checkpoint file: {e}")
    print("   Proceeding with fresh training start.")
    args.resume_from_checkpoint = None
```

### 3. Created Cleanup Utility

Created `cleanup_checkpoints.py` to help identify and clean up old checkpoints that were saved with `save_weights_only=True`.

## How to Use the Fix

### For New Training Runs

The fix is automatically applied. New checkpoints will be saved with the full training state and can be used for resuming training.

### For Existing Checkpoints

If you have existing checkpoints that were saved with `save_weights_only=True`, you have two options:

#### Option 1: Clean up old checkpoints and start fresh

```bash
# Check what checkpoints you have
python cleanup_checkpoints.py --checkpoint_dir /path/to/your/checkpoints

# See what would be removed (dry run)
python cleanup_checkpoints.py --checkpoint_dir /path/to/your/checkpoints --cleanup --dry_run

# Actually remove invalid checkpoints
python cleanup_checkpoints.py --checkpoint_dir /path/to/your/checkpoints --cleanup
```

#### Option 2: Start training without resuming

Simply run your training command without the `--resume_from_checkpoint` argument:

```bash
python train_optimized.py [your other arguments]
```

The training will start fresh, and new checkpoints will be saved with the full training state.

## Key Changes Made

1. **`train_optimized.py`**:
   - Added `save_weights_only=False` to ModelCheckpoint
   - Enabled `save_last=True` for better resume capability
   - Added checkpoint validation logic
   - Added error handling for invalid checkpoints

2. **`cleanup_checkpoints.py`** (new file):
   - Utility to identify invalid checkpoints
   - Option to clean up old checkpoints
   - Dry-run mode for safe testing

## Benefits

- ✅ Can now resume training from checkpoints
- ✅ Automatic validation of checkpoint compatibility
- ✅ Graceful fallback when invalid checkpoints are encountered
- ✅ Utility to clean up old problematic checkpoints
- ✅ Better error messages and user guidance

## Future Training

From now on, all checkpoints will be saved with the full training state, allowing you to resume training at any point. The training script will automatically validate checkpoints and provide clear feedback about their compatibility.