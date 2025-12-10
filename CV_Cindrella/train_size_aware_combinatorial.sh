#!/bin/bash

# ============================================================
# Size-Aware IDM-VTON Training Script
# Full Combinatorial Training with Optional IP-Adapter Fine-tuning
# ============================================================

# Configuration
export CUDA_VISIBLE_DEVICES=0

# Paths - MODIFY THESE
DATA_DIR="/path/to/VITON-HD"  # CHANGE THIS to your VITON-HD dataset path
OUTPUT_DIR="./results/size_aware_combinatorial"

# Optional: Path to size annotations JSON file
# SIZE_ANNOTATION_FILE="./size_annotations.json"

# Training hyperparameters
BATCH_SIZE=4
EPOCHS=100
LEARNING_RATE=1e-5
SIZE_EMBEDDER_LR=1e-4  # Higher LR for size embedder
SIZE_DROPOUT=0.1

# IP-Adapter fine-tuning options
TRAIN_IP_ADAPTER=false       # Set to true to fine-tune IP-Adapter
USE_IP_ADAPTER_LORA=true     # Use LoRA (recommended) vs full fine-tuning
IP_ADAPTER_LORA_RANK=16      # LoRA rank (8-32 typical)
IP_ADAPTER_LR=1e-6           # Very low LR for IP-Adapter

# Model paths
PRETRAINED_MODEL="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
GARMENTNET_PATH="stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_PATH="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin"
IMAGE_ENCODER_PATH="ckpt/image_encoder"

# ============================================================
# Check prerequisites
# ============================================================

echo "============================================================"
echo "Size-Aware IDM-VTON Training - Full Combinatorial"
echo "============================================================"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATA_DIR"
    echo "Please set DATA_DIR to your VITON-HD dataset path"
    exit 1
fi

if [ ! -f "$IP_ADAPTER_PATH" ]; then
    echo "ERROR: IP-Adapter weights not found: $IP_ADAPTER_PATH"
    echo "Please download the IP-Adapter weights"
    exit 1
fi

echo "Dataset: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Size dropout: $SIZE_DROPOUT"
echo "Train IP-Adapter: $TRAIN_IP_ADAPTER"
if [ "$TRAIN_IP_ADAPTER" = true ]; then
    echo "  - Use LoRA: $USE_IP_ADAPTER_LORA"
    echo "  - LoRA rank: $IP_ADAPTER_LORA_RANK"
    echo "  - IP-Adapter LR: $IP_ADAPTER_LR"
fi
echo "============================================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# ============================================================
# Build training command
# ============================================================

CMD="python train_xl_size_aware.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --pretrained_garmentnet_path $GARMENTNET_PATH \
    --pretrained_ip_adapter_path $IP_ADAPTER_PATH \
    --image_encoder_path $IMAGE_ENCODER_PATH \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --size_embedder_lr $SIZE_EMBEDDER_LR \
    --size_dropout_prob $SIZE_DROPOUT \
    --size_injection_method added_cond \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision fp16 \
    --checkpointing_epochs 10 \
    --logging_steps 500 \
    --seed 42"

# Add IP-Adapter options if enabled
if [ "$TRAIN_IP_ADAPTER" = true ]; then
    CMD="$CMD --train_ip_adapter --ip_adapter_lr $IP_ADAPTER_LR"
    if [ "$USE_IP_ADAPTER_LORA" = true ]; then
        CMD="$CMD --use_ip_adapter_lora --ip_adapter_lora_rank $IP_ADAPTER_LORA_RANK"
    fi
fi

# ============================================================
# Run training
# ============================================================

echo "Running: $CMD"
eval $CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

echo "============================================================"
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "============================================================"
