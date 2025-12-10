# 🚀 START TRAINING - STEP BY STEP GUIDE

**You're here because you want to train the size-aware model!**

---

## ⚡ FASTEST WAY TO START

### 1. **Find or Download VITON-HD Dataset** (Most Important!)

```bash
# Option A: Search for existing dataset on your system
find /scratch /data /home -name "VITON-HD" -type d 2>/dev/null

# Option B: If not found, you need to download it
# Go to: https://github.com/shadow2496/VITON-HD
# Or ask your advisor where the dataset is stored
```

### 2. **Quick Setup**

```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella

# Set your data path (CHANGE THIS!)
export VITON_PATH="/path/to/VITON-HD"

# Verify path is correct
ls $VITON_PATH/train/image/ | head -5
```

### 3. **Option 1: Simple Training (RECOMMENDED TO START)**

Since we need to first create `train_size_aware.py`, here's the **simplest approach**:

```bash
# Copy and modify existing training script
cp train_xl.py train_size_aware_simple.py
```

Then make these minimal changes to `train_size_aware_simple.py`:

**Line ~31-40 (Replace VitonHDDataset):**
```python
# BEFORE:
# from train_xl.py line ~31
class VitonHDDataset(data.Dataset):
    ...

# AFTER: Replace with
from size_aware_dataset import SizeAwareVitonHDDataset as VitonHDDataset
```

That's it! This gives you size-aware dataset with minimal changes.

### 4. **Run Training**

```bash
# Test with small number of epochs first
accelerate launch --mixed_precision="fp16" train_size_aware_simple.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --data_dir="$VITON_PATH" \
    --output_dir="./results/test_run" \
    --train_batch_size=2 \
    --num_train_epochs=2 \
    --learning_rate=1e-5 \
    --gradient_checkpointing
```

**This will:**
- ✅ Load size-aware dataset
- ✅ Compute size ratios on-the-fly
- ✅ Train for 2 epochs (quick test)
- ✅ Save checkpoints

---

## Option 2: Full Implementation (After Basic Works)

Once the simple version works, you can add:

1. **Size Encoder & Controller**:
```python
from size_modules import SizeEncoder, SimpleSizeController

size_encoder = SizeEncoder().to(device)
size_controller = SimpleSizeController().to(device)
```

2. **In Training Loop**:
```python
# Get size ratios from batch
size_ratios = batch['size_ratios'].to(device)

# Encode
size_embedding = size_encoder(size_ratios)
size_map = size_controller(size_embedding)

# TODO: Pass to UNet (requires UNet modification)
```

---

## 📊 What To Expect

### First Run (Simple):
- **Time:** ~1-2 hours for 2 epochs
- **GPU Memory:** ~40-50GB (you have 80GB ✅)
- **Output:** Model checkpoints in `./results/test_run/`
- **Check:** Dataset returns size info correctly

### Full Training (50 epochs):
- **Time:** 6-8 hours
- **Result:** Size-aware model ready for inference

---

## 🔍 Monitoring Progress

Watch for these in the terminal:
```
Step 1/XXX | Loss: 0.XXXX
Step 100/XXX | Loss: 0.XXXX  # Should decrease
...
Epoch 1/2 complete
```

---

## ❓ Common Issues

### "Cannot find dataset"
```bash
# Check your path
echo $VITON_PATH
ls $VITON_PATH/train/
```

### "CUDA out of memory"
```bash
# Reduce batch size
--train_batch_size=1
```

### "accelerate: command not found"
```bash
# Activate your conda environment first!
conda activate idm  # or your environment name
```

---

## 🎯 Next Steps After Training Works

1. ✅ Basic training works → Add size modules
2. ✅ Size modules integrated → Train full 50 epochs
3. ✅ Training complete → Run evaluation
4. ✅ Good results → Write report!

---

## 💡 Pro Tips

1. **Start Small:** 2 epochs, small batch size
2. **Check Samples:** Look at generated images every few checkpoints
3. **Monitor GPU:** Run `watch -n 1 nvidia-smi` in another terminal
4. **Save Often:** Use `--checkpointing_epoch=5` to save frequently

---

## 🆘 Need Help?

Check these files:
- **Training basics:** `NEXT_STEPS.md`
- **Architecture details:** `SIZE_AWARE_IMPLEMENTATION_SUMMARY.md`
- **Module docs:** `size_modules/README.md`

---

**READY? Let's start with the simple version first!**

```bash
# Step 1: Find dataset
find /scratch -name "*VITON*" -type d 2>/dev/null

# Step 2: Set path
export VITON_PATH="/the/path/you/found"

# Step 3: Copy training script
cd /scratch/bds9746/CV_Vton/CV_Cindrella
cp train_xl.py train_size_aware_simple.py

# Step 4: Edit line ~31 in train_size_aware_simple.py to use SizeAwareVitonHDDataset

# Step 5: Run!
accelerate launch train_size_aware_simple.py \
    --data_dir="$VITON_PATH" \
    --output_dir="./results/test" \
    --train_batch_size=2 \
    --num_train_epochs=2
```

**That's it! 🎉**
