# Training Quick Start Guide

**GPU Available:** 1× NVIDIA A100-SXM4-80GB ✅
**Dataset Status:** Need to download VITON-HD
**Current Status:** Ready to train!

---

## Option 1: Train with VITON-HD Dataset (Recommended)

### Step 1: Download VITON-HD Dataset

You need the VITON-HD dataset. Download from:
- **Official:** https://github.com/shadow2496/VITON-HD
- **Google Drive:** Usually ~10-15GB

Expected structure after download:
```
/path/to/VITON-HD/
├── train/
│   ├── image/          # Person images
│   ├── cloth/          # Garment images
│   ├── image-densepose/
│   ├── agnostic-mask/
│   └── vitonhd_train_tagged.json
└── test/
    └── ...
```

**OR** if you already have it somewhere, find it:
```bash
find /scratch /data /home -name "VITON-HD" -type d 2>/dev/null
```

### Step 2: Run Training

Once you have the dataset path, run:

```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella

# Set your dataset path
export VITON_PATH="/path/to/VITON-HD"  # CHANGE THIS!

# Run training
bash train_size_aware.sh
```

---

## Option 2: Quick Test with Minimal Data (For Testing)

If you just want to test the training pipeline works:

```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella

# This will use a small subset for testing
bash train_size_aware_test.sh
```

This creates a minimal training run to verify everything works.

---

## What the Training Script Does

**Stage 3: Size Module Training**
- Trains: Size Encoder + Size Controller
- Freezes: Base IDM-VTON (UNet, GarmentNet, IP-Adapter)
- Epochs: 50 (you can reduce for testing)
- Batch Size: 4 (good for 1× A100-80GB)
- Expected Time: ~6-8 hours for full training

**Output:**
- Checkpoints saved to: `./results/size_aware_stage3/`
- Logs: `./results/size_aware_stage3/logs/`
- Sample images generated every N steps

---

## Monitoring Training

Watch the output for:
- ✅ Loss decreasing
- ✅ Size accuracy improving (should reach >70% minimum)
- ✅ No CUDA out of memory errors
- ✅ Sample images look reasonable

---

## Troubleshooting

### "Cannot find VITON-HD dataset"
→ Set the correct path in the training script or use `--data_dir` flag

### "CUDA out of memory"
→ Reduce batch size in the script: `--train_batch_size=2`

### "ModuleNotFoundError: size_modules"
→ Run from the CV_Cindrella directory

---

## After Training

Once training finishes:
1. Check `./results/size_aware_stage3/checkpoint-XXXX/`
2. Run inference to test: `python inference_size_aware.py`
3. Evaluate metrics: `python evaluate_size_aware.py`

---

**Ready? Let's start with downloading the dataset!**
