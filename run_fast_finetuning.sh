#!/bin/bash

# Fast Classifier Head Fine-tuning Script
# This script provides easy execution of the optimized fine-tuning process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_CHECKPOINT=""
DEFAULT_DATA_DIR="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
DEFAULT_HEAD_LR="5e-4"
DEFAULT_BATCH_SIZE="16"
DEFAULT_MAX_EPOCHS="30"
DEFAULT_NUM_WORKERS="8"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Fast Classifier Head Fine-tuning Script"
    echo ""
    echo "Required Options:"
    echo "  -c, --checkpoint PATH    Path to pretrained SeqSetVAE checkpoint (REQUIRED)"
    echo ""
    echo "Optional Options:"
    echo "  -d, --data-dir PATH      Data directory path (default: $DEFAULT_DATA_DIR)"
    echo "  -l, --head-lr LR         Learning rate for classifier head (default: $DEFAULT_HEAD_LR)"
    echo "  -b, --batch-size SIZE    Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  -e, --max-epochs EPOCHS  Maximum training epochs (default: $DEFAULT_MAX_EPOCHS)"
    echo "  -w, --num-workers WORKERS Number of data loading workers (default: $DEFAULT_NUM_WORKERS)"
    echo "  -o, --output-dir PATH    Output directory (default: config default)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -c /path/to/checkpoint.ckpt"
    echo "  $0 -c /path/to/checkpoint.ckpt -b 32 -e 50"
    echo "  $0 --checkpoint /path/to/checkpoint.ckpt --head-lr 3e-4"
}

# Parse command line arguments
CHECKPOINT=""
DATA_DIR="$DEFAULT_DATA_DIR"
HEAD_LR="$DEFAULT_HEAD_LR"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
MAX_EPOCHS="$DEFAULT_MAX_EPOCHS"
NUM_WORKERS="$DEFAULT_NUM_WORKERS"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -l|--head-lr)
            HEAD_LR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        -w|--num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CHECKPOINT" ]]; then
    print_error "Checkpoint path is required!"
    show_usage
    exit 1
fi

# Check if checkpoint file exists
if [[ ! -f "$CHECKPOINT" ]]; then
    print_error "Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

# Print configuration
echo "============================================================"
echo "🚀 FAST CLASSIFIER HEAD FINE-TUNING CONFIGURATION"
echo "============================================================"
echo "Checkpoint:     $CHECKPOINT"
echo "Data Directory: $DATA_DIR"
echo "Learning Rate:  $HEAD_LR"
echo "Batch Size:     $BATCH_SIZE"
echo "Max Epochs:     $MAX_EPOCHS"
echo "Workers:        $NUM_WORKERS"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output Dir:     $OUTPUT_DIR"
fi
echo "============================================================"

# Check Python environment
print_info "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
print_info "Checking required files..."
REQUIRED_FILES=(
    "finetune_classifier_head_fast.py"
    "model.py"
    "dataset.py"
    "config.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "Required file not found: $file"
        exit 1
    fi
done

print_success "All required files found"

# Check GPU availability
print_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_info "GPU detected: $GPU_INFO"
else
    print_warning "No GPU detected, will use CPU (training will be slower)"
fi

# Build command
CMD="python3 finetune_classifier_head_fast.py"
CMD="$CMD --checkpoint \"$CHECKPOINT\""
CMD="$CMD --data_dir \"$DATA_DIR\""
CMD="$CMD --head_lr $HEAD_LR"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_epochs $MAX_EPOCHS"
CMD="$CMD --num_workers $NUM_WORKERS"

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output_root_dir \"$OUTPUT_DIR\""
fi

# Print final command
echo ""
print_info "Executing command:"
echo "$CMD"
echo ""

# Confirm execution
read -p "Do you want to proceed with fine-tuning? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Fine-tuning cancelled by user"
    exit 0
fi

# Execute fine-tuning
print_info "Starting fast classifier head fine-tuning..."
echo ""

# Execute the command
eval $CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    print_success "Fine-tuning completed successfully!"
    print_info "Check the outputs directory for results and checkpoints"
else
    print_error "Fine-tuning failed with exit code $?"
    exit 1
fi