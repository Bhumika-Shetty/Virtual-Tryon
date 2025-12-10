#!/bin/bash

# Test Pipeline Runner
# This script activates the environment and runs the test

echo "========================================"
echo "Size-Aware Pipeline Test Runner"
echo "========================================"
echo ""

# Find conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "⚠️  Warning: Could not find conda"
    echo "Trying to run with system Python..."
fi

# Activate environment (try common names)
if conda activate idm 2>/dev/null; then
    echo "✅ Activated 'idm' environment"
elif conda activate vton 2>/dev/null; then
    echo "✅ Activated 'vton' environment"
elif conda activate base 2>/dev/null; then
    echo "✅ Activated 'base' environment"
else
    echo "⚠️  Could not activate conda environment"
    echo "Continuing with current Python..."
fi

echo ""
echo "Python: $(which python)"
echo "PyTorch available: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""

# Run the test
cd /scratch/bds9746/CV_Vton/CV_Cindrella
python test_pipeline.py

echo ""
echo "========================================"
