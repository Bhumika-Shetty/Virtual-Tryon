#!/bin/bash
# Helper script to activate the idm environment
# Usage: source activate_idm.sh

echo "Activating idm environment..."

# Method 1: Try to use the environment directly
if [ -d "/scratch/bds9746/envs/idm" ]; then
    export PATH="/scratch/bds9746/envs/idm/bin:$PATH"
    export CONDA_PREFIX="/scratch/bds9746/envs/idm"
    export CONDA_DEFAULT_ENV="idm"
    export LD_LIBRARY_PATH="/scratch/bds9746/envs/idm/lib:$LD_LIBRARY_PATH"
    
    # Try to find and add conda's library paths
    if [ -d "/scratch/bds9746/envs/idm/lib" ]; then
        export LD_LIBRARY_PATH="/scratch/bds9746/envs/idm/lib:$LD_LIBRARY_PATH"
    fi
    
    echo "✅ Environment paths set"
    echo "   Python: $(which python 2>/dev/null || echo 'not found')"
    
    # Test PyTorch
    if python -c "import torch" 2>/dev/null; then
        PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        echo "✅ PyTorch $PYTORCH_VERSION - CUDA: $CUDA_AVAILABLE"
    else
        echo "⚠️  PyTorch import failed - you may need to reinstall PyTorch"
        echo "   Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    fi
else
    echo "❌ idm environment not found at /scratch/bds9746/envs/idm"
    echo ""
    echo "Try creating it with:"
    echo "  conda env create -f environment.yaml"
fi

