#!/bin/bash

# SIMPLE TEST RUNNER
# Just run: bash RUN_ME_TO_TEST.sh

echo "🧪 Starting Pipeline Test..."
echo ""

# Try to find and activate Python environment
echo "Step 1: Setting up Python environment..."

ENV_ACTIVATED=0

# Option 1: Try to use the idm environment directly (if it's a standalone conda env)
if [ -d "/scratch/bds9746/envs/idm" ] && [ -f "/scratch/bds9746/envs/idm/bin/python" ]; then
    echo "  Found idm environment at /scratch/bds9746/envs/idm"
    export PATH="/scratch/bds9746/envs/idm/bin:$PATH"
    export CONDA_PREFIX="/scratch/bds9746/envs/idm"
    export CONDA_DEFAULT_ENV="idm"
    ENV_ACTIVATED=1
fi

# Option 2: Try conda initialization (look in common locations)
if [ $ENV_ACTIVATED -eq 0 ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        echo "  Found conda at $HOME/miniconda3"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate idm 2>/dev/null && ENV_ACTIVATED=1
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        echo "  Found conda at $HOME/anaconda3"
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        conda activate idm 2>/dev/null && ENV_ACTIVATED=1
    elif [ -f "/scratch/bds9746/conda_cache/conda-24.11.3-py311h06a4308_0/etc/profile.d/conda.sh" ]; then
        echo "  Found conda in conda_cache"
        # Try to find conda.sh in conda_cache
        CONDA_SH=$(find /scratch/bds9746/conda_cache -name "conda.sh" -path "*/etc/profile.d/conda.sh" 2>/dev/null | head -1)
        if [ -n "$CONDA_SH" ]; then
            source "$CONDA_SH"
            conda activate idm 2>/dev/null && ENV_ACTIVATED=1
        fi
    fi
fi

# Option 3: Module system (as fallback)
if [ $ENV_ACTIVATED -eq 0 ] && command -v module &> /dev/null; then
    echo "  Trying module system..."
    module load python/intel/3.8.6 2>/dev/null || module load python 2>/dev/null
    module load cuda/11.6.2 2>/dev/null || module load cuda 2>/dev/null
fi

echo ""
echo "Current Python: $(which python 2>/dev/null || echo 'not found')"
echo ""

# Check if PyTorch is available
echo "Step 2: Checking for PyTorch..."
if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    echo "  ✅ PyTorch found: $PYTORCH_VERSION"
    echo "  ✅ CUDA available: $CUDA_AVAILABLE"
    echo ""

    # Run the test
    echo "Step 3: Running pipeline test..."
    echo ""
    cd /scratch/bds9746/CV_Vton/CV_Cindrella
    python test_pipeline.py

else
    echo "  ❌ PyTorch not found or not working"
    echo ""
    echo "  You need to activate your PyTorch environment first."
    echo ""
    echo "  Try one of these methods:"
    echo ""
    echo "  Method 1: Use the idm environment directly"
    echo "    export PATH=\"/scratch/bds9746/envs/idm/bin:\$PATH\""
    echo "    export CONDA_PREFIX=\"/scratch/bds9746/envs/idm\""
    echo ""
    echo "  Method 2: Initialize conda and activate"
    echo "    # Find conda.sh first:"
    echo "    find /scratch/bds9746/conda_cache -name conda.sh"
    echo "    # Then source it and activate:"
    echo "    source /path/to/conda.sh"
    echo "    conda activate idm"
    echo ""
    echo "  Method 3: Use module system"
    echo "    module load python/intel/3.8.6"
    echo "    module load cuda/11.6.2"
    echo ""
    echo "  After activating, run this script again:"
    echo "    bash RUN_ME_TO_TEST.sh"
    echo ""
    echo "  OR run the test directly:"
    echo "    python test_pipeline.py"
fi
