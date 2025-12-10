#!/bin/bash

# Size-Aware Training Script (Stage 3)
# This trains the Size Encoder and Size Controller while freezing the base model

# CONFIGURE THESE:
DATA_DIR="${VITON_PATH:-/path/to/VITON-HD}"  # Set VITON_PATH environment variable or change this
OUTPUT_DIR="./results/size_aware_stage3"
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"

# Training hyperparameters
BATCH_SIZE=4                    # Good for 1× A100-80GB
GRAD_ACCUM=2                    # Effective batch size = 4 × 2 = 8
NUM_EPOCHS=50                   # Full training (reduce to 5 for testing)
LEARNING_RATE=5e-5
CHECKPOINTING_EPOCHS=10         # Save every 10 epochs

# GPU settings
export CUDA_VISIBLE_DEVICES=0   # Use first GPU

echo "========================================"
echo "Size-Aware VTON Training (Stage 3)"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "========================================"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please set VITON_PATH environment variable or edit this script"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with accelerate
accelerate launch --mixed_precision="fp16" train_size_aware.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --data_dir="$DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --num_train_epochs=$NUM_EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --checkpointing_epoch=$CHECKPOINTING_EPOCHS \
    --gradient_checkpointing \
    --use_8bit_adam \
    --snr_gamma=5.0 \
    --logging_steps=100 \
    --height=512 \
    --width=384 \
    --enable_size_conditioning \
    --size_augmentation \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "========================================"
