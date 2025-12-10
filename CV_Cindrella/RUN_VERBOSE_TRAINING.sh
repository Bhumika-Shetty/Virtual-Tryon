#!/bin/bash

# RUN SMALL VERBOSE TRAINING
# This will train on 100 samples for 3 epochs and save all size calculations

echo "🚀 Starting Small Verbose Training..."
echo ""

# Try to activate environment
ENV_ACTIVATED=0

# Option 1: Use idm environment directly
if [ -d "/scratch/bds9746/envs/idm" ] && [ -f "/scratch/bds9746/envs/idm/bin/python" ]; then
    echo "  Found idm environment at /scratch/bds9746/envs/idm"
    export PATH="/scratch/bds9746/envs/idm/bin:$PATH"
    export CONDA_PREFIX="/scratch/bds9746/envs/idm"
    export CONDA_DEFAULT_ENV="idm"
    ENV_ACTIVATED=1
fi

echo "Current Python: $(which python 2>/dev/null || echo 'not found')"
echo ""

# Check if PyTorch is available
if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "✅ PyTorch found: $PYTORCH_VERSION"
    echo ""

    # Navigate to directory
    cd /scratch/bds9746/CV_Vton/CV_Cindrella

    # Run training and save output
    echo "Running training and saving output to training_verbose_log.txt..."
    echo ""

    python train_small_verbose.py \
        --data_dir /scratch/bds9746/datasets/VITON-HD \
        --num_samples 100 \
        --num_epochs 3 \
        --batch_size 2 \
        --log_every 10 \
        2>&1 | tee training_verbose_log.txt

    echo ""
    echo "✅ Training complete!"
    echo "📄 Output saved to: /scratch/bds9746/CV_Vton/CV_Cindrella/training_verbose_log.txt"
    echo ""
    echo "You can now share this file with your data team to show them how size is calculated!"

else
    echo "❌ PyTorch not found!"
    echo ""
    echo "Please activate your Python environment first:"
    echo ""
    echo "  Option 1: Direct activation"
    echo "    export PATH=\"/scratch/bds9746/envs/idm/bin:\$PATH\""
    echo ""
    echo "  Option 2: Conda activation"
    echo "    conda activate idm"
    echo ""
    echo "Then run this script again:"
    echo "    bash RUN_VERBOSE_TRAINING.sh"
fi
