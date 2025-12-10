# Environment Setup Guide

## Current Situation

Your `idm` conda environment is located at `/scratch/bds9746/envs/idm`, but PyTorch has a library dependency issue (`undefined symbol: iJIT_NotifyEvent`).

## Quick Fix Options

### Option 1: Reinstall PyTorch (Recommended)

The PyTorch installation in your environment seems corrupted. Try reinstalling it:

```bash
# Activate the environment
export PATH="/scratch/bds9746/envs/idm/bin:$PATH"
export CONDA_PREFIX="/scratch/bds9746/envs/idm"

# Reinstall PyTorch with CUDA 11.8 (as specified in environment.yaml)
pip install --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Or use conda if available
conda install pytorch=2.0.1 pytorch-cuda=11.8 torchvision=0.15.2 torchaudio=2.0.2 -c pytorch -c nvidia
```

### Option 2: Use the Helper Script

Try using the activation helper:

```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella
source activate_idm.sh
python test_pipeline.py
```

### Option 3: Recreate the Environment

If the above doesn't work, recreate the environment from scratch:

```bash
# If you have conda initialized
conda env remove -n idm
conda env create -f environment.yaml

# Then activate
conda activate idm
```

### Option 4: Use Module System (Fallback)

If conda environments don't work, try using the module system:

```bash
module load python/intel/3.8.6
module load cuda/11.6.2

# Install PyTorch in user space
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then run tests
cd /scratch/bds9746/CV_Vton/CV_Cindrella
python test_pipeline.py
```

## Testing Your Setup

After setting up, verify PyTorch works:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

You should see something like:
```
PyTorch 2.0.1 - CUDA: True
```

## Running the Test

Once PyTorch is working:

```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella
python test_pipeline.py
```

Or use the automated script:

```bash
bash RUN_ME_TO_TEST.sh
```

