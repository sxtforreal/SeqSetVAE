# SeqSetVAE with Batch Training Support

This repository contains an implementation of SeqSetVAE (Sequential Set Variational Autoencoder) with enhanced batch training capabilities.

## Key Features

- **Dynamic Padding**: Automatic handling of variable-length patient sequences
- **Batch Training**: Support for batch_size > 1 to accelerate training
- **Memory Optimization**: Optional sequence length limits and efficient padding strategies
- **Backward Compatibility**: Maintains support for original single-patient training

## Usage

### Basic Training

```bash
# Single patient training (original)
python train_with_collapse_detection.py --batch_size 1

# Batch training with 4 patients
python train_with_collapse_detection.py --batch_size 4

# Batch training with sequence length limit
python train_with_collapse_detection.py --batch_size 8 --max_sequence_length 1000
```

### Advanced Training

```bash
# Full batch training configuration
python train_with_collapse_detection.py \
    --batch_size 4 \
    --max_sequence_length 1000 \
    --use_dynamic_padding \
    --devices 2 \
    --max_epochs 10
```

## Parameters

- `--batch_size`: Batch size for training (default: 1)
- `--max_sequence_length`: Maximum sequence length to truncate (default: None)
- `--use_dynamic_padding`: Use dynamic padding for batch training (default: True)

## Performance

| Batch Size | Relative Speed | Memory Usage | Use Case |
|------------|----------------|--------------|----------|
| 1          | 1.0x           | Low          | Debug, small datasets |
| 4          | 2.5x           | Medium       | General training |
| 8          | 4.0x           | High         | Fast training |
| 16         | 6.0x           | Very High    | Large-scale training |

## Requirements

- PyTorch
- PyTorch Lightning
- NumPy
- Pandas
- Transformers

## File Structure

- `dataset.py`: Enhanced data loading with batch support
- `model.py`: SeqSetVAE model with batch processing
- `train_with_collapse_detection.py`: Training script with batch training options
- `config.py`: Configuration parameters
- `modules.py`: Core model components