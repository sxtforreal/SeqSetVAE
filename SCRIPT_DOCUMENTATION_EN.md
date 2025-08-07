# Script Documentation

This document provides an overview of all scripts in the SeqSetVAE project and their functionality.

## Core Scripts

### 1. `train_optimized.py`
**Purpose**: Main training script for the SeqSetVAE model with adaptive optimization.

**Key Features**:
- **Adaptive Device Configuration**: Automatically detects and configures optimal settings for GPU/CPU training
- **Multi-GPU Support**: Supports distributed training across multiple GPUs using DDP strategy
- **Memory Optimization**: Adjusts batch sizes, precision, and worker counts based on available hardware
- **Checkpoint Management**: Saves both best and last checkpoints for training resumption
- **Posterior Metrics Monitoring**: Optional monitoring of posterior collapse metrics during training
- **Early Stopping**: Optimized early stopping with reduced patience for faster training
- **Model Compilation**: Optional torch.compile integration for performance optimization

**Adaptive Features**:
- Automatically switches between 16-bit and 32-bit precision based on GPU memory
- Adjusts number of data loader workers based on available CPU cores
- Configures gradient accumulation for effective large batch training
- Optimizes validation frequency and batch limits for faster training

### 2. `config.py`
**Purpose**: Configuration management with intelligent device detection and adaptive settings.

**Key Features**:
- **Hardware Detection**: Automatically detects GPU/CPU capabilities and memory
- **Adaptive Configuration**: Provides optimal settings based on available hardware
- **Lazy Initialization**: Device configuration is initialized only when needed
- **Memory-Based Optimization**: Adjusts settings based on GPU memory capacity
- **Fallback Support**: Provides sensible defaults when PyTorch is not available

**Configuration Categories**:
- **GPU Configuration**: Multi-GPU support, precision settings, memory optimization
- **CPU Configuration**: Single-threaded training with optimized worker counts
- **Device Attributes**: Easy access to device, accelerator, and precision settings

### 3. `model.py`
**Purpose**: Core model implementation for SeqSetVAE with transformer integration.

**Key Components**:
- **SetVAE**: Base variational autoencoder for set processing
- **Transformer Encoder**: Enhanced sequence modeling with positional encoding
- **Time Encoder**: Continuous time embedding for temporal information
- **Classification Head**: Multi-class classification with improved architecture
- **Checkpoint Management**: Smart loading of pretrained weights with resume support

**Advanced Features**:
- **Dynamic Positional Encoding**: No fixed maximum sequence length
- **Causal Masking**: Proper attention masking for sequence modeling
- **Beta Annealing**: KL divergence warmup for stable training
- **Free Bits**: Prevents posterior collapse with minimum KL divergence
- **Parameter Freezing**: Selective freezing of pretrained components

### 4. `dataset.py`
**Purpose**: Data loading and preprocessing for sequence set data.

**Key Features**:
- **Dynamic Padding**: Efficient padding strategies for variable-length sequences
- **Set Processing**: Handles set-structured data with proper batching
- **Data Augmentation**: Optional augmentation techniques for improved generalization
- **Memory Optimization**: Efficient data loading with configurable worker counts
- **Validation Split**: Automatic train/validation split management

### 5. `modules.py`
**Purpose**: Modular components and utility functions for the SeqSetVAE architecture.

**Components**:
- **Attention Mechanisms**: Multi-head attention with various masking strategies
- **Set Operations**: Efficient set processing and manipulation functions
- **Loss Functions**: Custom loss functions for set-based learning
- **Utility Functions**: Helper functions for model operations

### 6. `comprehensive_analyzer.py`
**Purpose**: Comprehensive analysis and evaluation of trained models.

**Analysis Features**:
- **Performance Metrics**: Detailed evaluation of model performance
- **Posterior Analysis**: Analysis of latent space and posterior distributions
- **Visualization**: Generation of comprehensive analysis plots
- **Comparative Studies**: Comparison between different model configurations

### 7. `comprehensive_visualizer.py`
**Purpose**: Advanced visualization tools for model analysis and results.

**Visualization Features**:
- **Training Curves**: Learning curves and loss visualization
- **Latent Space**: Dimensionality reduction and clustering visualization
- **Attention Maps**: Transformer attention visualization
- **Set Representations**: Visualization of set embeddings and reconstructions

### 8. `cleanup_checkpoints.py`
**Purpose**: Utility script for managing and cleaning up training checkpoints.

**Features**:
- **Checkpoint Validation**: Identifies valid vs. invalid checkpoints
- **State Analysis**: Checks for optimizer states, scheduler states, and training metadata
- **Selective Cleanup**: Removes checkpoints that cannot be used for training resumption
- **Dry Run Mode**: Preview cleanup operations without actually deleting files
- **Error Handling**: Graceful handling of corrupted or invalid checkpoint files

**Checkpoint Types**:
- **Valid Checkpoints**: Contain full training state (optimizer, scheduler, epoch info)
- **Invalid Checkpoints**: Weights-only checkpoints that cannot resume training
- **Error Checkpoints**: Corrupted or unloadable checkpoint files

## Utility Scripts

### `posterior_collapse_detector.py`
**Purpose**: Monitors and detects posterior collapse during training.

**Monitoring Features**:
- **Real-time Monitoring**: Tracks KL divergence and reconstruction loss
- **Collapse Detection**: Identifies potential posterior collapse scenarios
- **Adaptive Thresholds**: Dynamic threshold adjustment based on training progress
- **Visualization**: Real-time plotting of monitoring metrics

## Configuration Management

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are used
- `PYTORCH_CUDA_ALLOC_CONF`: Configure CUDA memory allocation
- `OMP_NUM_THREADS`: Control CPU threading for data loading

### Command Line Arguments
Each script supports various command line arguments for customization:
- `--batch_size`: Training batch size
- `--max_epochs`: Maximum training epochs
- `--precision`: Training precision (16, 32, 16-mixed)
- `--num_workers`: Data loader worker count
- `--resume_from_checkpoint`: Resume training from checkpoint

## Performance Optimization

### GPU Optimization
- **Mixed Precision**: Automatic 16-bit training for memory efficiency
- **Gradient Accumulation**: Effective large batch training
- **Model Compilation**: torch.compile integration for faster execution
- **Memory Management**: Optimized memory usage and garbage collection

### CPU Optimization
- **Worker Management**: Optimal worker count based on CPU cores
- **Memory Pinning**: Disabled for CPU training to avoid overhead
- **Precision Control**: Forced 32-bit precision for stability

## Best Practices

1. **Checkpoint Management**: Regularly clean up invalid checkpoints to save disk space
2. **Hardware Monitoring**: Monitor GPU memory usage during training
3. **Validation Frequency**: Adjust validation intervals based on dataset size
4. **Early Stopping**: Use early stopping to prevent overfitting
5. **Posterior Monitoring**: Enable posterior collapse monitoring for VAE training

## Troubleshooting

### Common Issues
- **Out of Memory**: Reduce batch size or enable gradient accumulation
- **Slow Training**: Enable model compilation or adjust worker count
- **Checkpoint Errors**: Use cleanup script to identify and remove invalid checkpoints
- **Posterior Collapse**: Monitor KL divergence and adjust beta annealing

### Performance Tips
- Use appropriate batch sizes for your hardware
- Enable mixed precision training when possible
- Monitor and clean up checkpoints regularly
- Use early stopping to prevent unnecessary training time