# IDM-VTON Setup Guide for Local Machine

This guide will help you set up IDM-VTON on your local machine after transferring the folder.

## Step 1: Extract the Archive

After transferring `IDM-VTON.tar.gz` to your local machine, extract it:

```bash
cd /path/to/destination
tar -xzf IDM-VTON.tar.gz
cd IDM-VTON
```

## Step 2: Set Up Conda Environment

The project requires a specific conda environment. Create it using the provided `environment.yaml`:

**Option 1: Default location (creates in conda's default envs directory)**
```bash
conda env create -f environment.yaml
conda activate idm
```

**Option 2: Custom location (use `-p` flag to specify path)**
If you want to create the environment in a specific directory (e.g., scratch directory):
```bash
conda env create -f environment.yaml -p /path/to/scratch/envs/idm
conda activate /path/to/scratch/envs/idm
```

Or if you want it relative to the project directory:
```bash
conda env create -f environment.yaml -p ./conda_envs/idm
conda activate ./conda_envs/idm
```

**Note:** This will install:
- Python 3.10.0
- PyTorch 2.0.1 with CUDA 11.8
- All required dependencies (accelerate, transformers, diffusers, gradio, etc.)

## Step 3: Verify Checkpoints

The following checkpoints are already included in the `ckpt/` folder:
- ✅ `ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin` - IP-Adapter model
- ✅ `ckpt/image_encoder/` - Image encoder (config.json, model.safetensors)
- ✅ `ckpt/densepose/model_final_162be9.pkl` - DensePose model
- ✅ `ckpt/humanparsing/parsing_atr.onnx` and `parsing_lip.onnx` - Human parsing models
- ✅ `ckpt/openpose/ckpts/body_pose_model.pth` - OpenPose model

**No additional downloads needed for these!**

## Step 4: Choose Your Use Case

### Option A: Run Gradio Demo (Easiest - No dataset needed)

The Gradio demo allows you to try on clothes interactively without needing a full dataset.

**Requirements:**
- All checkpoints are already in place ✅
- GPU recommended (but can run on CPU, slower)

**Run:**
```bash
conda activate idm
cd IDM-VTON
python gradio_demo/app.py
```

This will start a web interface (usually at `http://localhost:7860`) where you can upload images and try on clothes.

### Option B: Run Inference on VITON-HD Dataset

**Requirements:**
1. Download VITON-HD dataset from: https://github.com/shadow2496/VITON-HD
2. Organize the dataset structure:

```
your_dataset_path/
├── train/
│   ├── image/
│   ├── image-densepose/
│   ├── agnostic-mask/
│   ├── cloth/
│   └── vitonhd_train_tagged.json
└── test/
    ├── image/
    ├── image-densepose/
    ├── agnostic-mask/
    ├── cloth/
    └── vitonhd_test_tagged.json
```

**Note:** The JSON files (`vitonhd_test_tagged.json` and `vitonhd_train_tagged.json`) are already in the IDM-VTON folder - copy them to your dataset folders.

**Run inference:**
```bash
conda activate idm
cd IDM-VTON

# For paired setting
accelerate launch inference.py \
    --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --data_dir "/path/to/your/dataset" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0

# For unpaired setting (add --unpaired flag)
accelerate launch inference.py \
    --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "/path/to/your/dataset" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0
```

### Option C: Run Inference on DressCode Dataset

**Requirements:**
1. Download DressCode dataset from: https://github.com/aimagelab/dress-code
2. Download pre-computed densepose images and captions from: https://kaistackr-my.sharepoint.com/:u:/g/personal/cpis7_kaist_ac_kr/EaIPRG-aiRRIopz9i002FOwBDa-0-BHUKVZ7Ia5yAVVG3A?e=YxkAip
3. Organize the dataset structure:

```
DressCode/
├── dresses/
│   ├── images/
│   ├── image-densepose/
│   └── dc_caption.txt
├── lower_body/
│   ├── images/
│   ├── image-densepose/
│   └── dc_caption.txt
└── upper_body/
    ├── images/
    ├── image-densepose/
    └── dc_caption.txt
```

**Run inference:**
```bash
conda activate idm
cd IDM-VTON

# For upper_body category
accelerate launch inference_dc.py \
    --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "/path/to/DressCode" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0 \
    --category "upper_body"

# For lower_body or dresses, change --category accordingly
```

## Step 5: Training (Optional)

If you want to train the model, you'll need:

1. **IP-Adapter models** (already included ✅):
   - `ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin` ✅
   - `ckpt/image_encoder/` ✅

2. **Training dataset** (VITON-HD or DressCode)

3. **Run training:**
```bash
conda activate idm
cd IDM-VTON

accelerate launch train_xl.py \
    --gradient_checkpointing --use_8bit_adam \
    --output_dir=result --train_batch_size=6 \
    --data_dir=/path/to/your/dataset
```

## Summary of External Dependencies Needed

### Already Included ✅
- All model checkpoints (ip_adapter, image_encoder, densepose, humanparsing, openpose)
- All code and configuration files
- Environment specification

### You Need to Provide:
1. **Conda** - to create the environment
2. **GPU** (recommended) - for faster inference/training
3. **Dataset** (only if running inference/training):
   - VITON-HD dataset OR
   - DressCode dataset
4. **Internet connection** - for downloading the pretrained model from HuggingFace (`yisol/IDM-VTON`) during first run

## Quick Start (Gradio Demo - No Dataset Needed)

If you just want to try it out quickly:

```bash
# 1. Extract archive
tar -xzf IDM-VTON.tar.gz
cd IDM-VTON

# 2. Create environment (use -p flag for custom path if needed)
conda env create -f environment.yaml -p /path/to/scratch/envs/idm
conda activate /path/to/scratch/envs/idm

# Or use default location:
# conda env create -f environment.yaml
# conda activate idm

# 3. Run demo
python gradio_demo/app.py
```

Then open your browser to the URL shown (usually `http://localhost:7860`).

## Troubleshooting

1. **CUDA out of memory**: Reduce `test_batch_size` in inference commands
2. **Model download issues**: The model will be downloaded from HuggingFace on first run - ensure internet connection
3. **Missing dependencies**: Make sure conda environment is activated (`conda activate idm`)
4. **Gradio not starting**: Check if port 7860 is available, or modify `app.py` to use a different port

## References

- GitHub Repository: https://github.com/yisol/IDM-VTON
- Paper: https://arxiv.org/abs/2403.05139
- HuggingFace Model: https://huggingface.co/yisol/IDM-VTON
- Demo: https://huggingface.co/spaces/yisol/IDM-VTON

