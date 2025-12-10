# Next Steps: Training Integration Guide

**Date:** 2025-11-30
**Current Status:** Core modules complete ✅ | Ready for training integration 🚀

---

## 🎯 Immediate Next Steps (Priority Order)

### 1. **Test the Modules** (< 1 hour)

First, ensure all modules work correctly:

```bash
# Activate conda environment
conda activate idm  # or whatever your environment is named

# Test each module
cd /scratch/bds9746/CV_Vton/CV_Cindrella

python size_modules/size_annotation.py
python size_modules/size_encoder.py
python size_modules/size_controller.py
```

**Expected Output:** All tests should pass with shape information printed.

---

### 2. **Create Size-Aware Training Script** (2-3 hours)

Create `train_size_aware.py` based on `train_xl.py`:

**Key Changes:**

```python
# 1. Replace dataset
from size_aware_dataset import SizeAwareVitonHDDataset

train_dataset = SizeAwareVitonHDDataset(
    dataroot_path=args.data_dir,
    phase="train",
    size=(512, 384),
    size_augmentation=True,  # Enable size augmentation
    enable_size_conditioning=True
)

# 2. Initialize size modules
from size_modules import SizeEncoder, SimpleSizeController

size_encoder = SizeEncoder(
    input_dim=3,
    hidden_dim=256,
    output_dim=768
).to(accelerator.device)

size_controller = SimpleSizeController(
    size_embedding_dim=768,
    output_size=(128, 96)  # Match latent size
).to(accelerator.device)

# 3. In training loop
for batch in train_dataloader:
    # Get size information
    size_ratios = batch['size_ratios'].to(accelerator.device)

    # Encode size
    size_embedding = size_encoder(size_ratios)
    size_map = size_controller(size_embedding)

    # TODO: Pass to UNet (requires UNet modification)
    # For now, just compute size loss

# 4. Add size loss
# L_size = F.mse_loss(predicted_size, target_size)
# total_loss = reconstruction_loss + 0.5 * L_size
```

**File to create:** `train_size_aware.py`

---

### 3. **Modify UNet Forward Pass** (3-4 hours)

Update `src/unet_hacked_tryon.py` to accept size conditioning:

```python
# In UNet2DConditionModel.forward()

def forward(
    self,
    sample,
    timestep,
    encoder_hidden_states,
    # NEW: Add size parameters
    size_embedding=None,
    size_map=None,
    **kwargs
):
    # Store size info in kwargs for attention layers
    if size_embedding is not None:
        kwargs['size_embedding'] = size_embedding
    if size_map is not None:
        kwargs['size_map'] = size_map

    # Rest of forward pass...
    # Attention layers will receive size info via kwargs
```

**Files to modify:**
- `src/unet_hacked_tryon.py`
- `src/attentionhacked_tryon.py` (to use size info)

**Minimal Implementation:**
For now, just pass the size info through. Full attention modification can come later.

---

### 4. **Run Initial Training** (4-6 hours)

**Stage 3: Size Module Training**

```bash
# Activate environment
conda activate idm

# Run training (freeze base model, train size modules)
accelerate launch train_size_aware.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --data_dir="/path/to/VITON-HD" \
    --output_dir="./checkpoints/stage3_size_training" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=50 \
    --learning_rate=5e-5 \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --checkpointing_epoch=10
```

**What should happen:**
- Size encoder and controller train
- Base model remains frozen
- Should see size-related metrics improving

---

### 5. **Implement Evaluation** (2-3 hours)

Create `evaluate_size_aware.py`:

```python
def evaluate_size_accuracy(predictions, ground_truth):
    """
    Measure how accurately the model predicts garment fit.
    """
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truth):
        pred_label = get_size_label(pred)
        gt_label = get_size_label(gt)

        if pred_label == gt_label:
            correct += 1

    return correct / total

def compute_gfd(generated_img, garment_mask, body_mask, size_ratios):
    """
    Geometric Fit Deviation (GFD) metric.
    Measures spatial accuracy of fit.
    """
    # Extract garment boundary from generated image
    # Compare with expected boundary based on size ratios
    # Return deviation score
    pass
```

**Metrics to track:**
- Size Accuracy (> 85% target)
- LPIPS (< 0.10 target)
- SSIM (> 0.90 target)
- FID (< 6.0 target)

---

## 📝 Detailed Task Breakdown

### Task 1: Test Modules ✅
**Time:** 30 min
**Files:** None to create, just run tests
**Commands:**
```bash
conda activate idm
python size_modules/size_annotation.py
python size_modules/size_encoder.py
python size_modules/size_controller.py
```

### Task 2: Create Training Script 🔄
**Time:** 2-3 hours
**File to create:** `train_size_aware.py`
**Base:** Copy from `train_xl.py` and modify
**Key sections:**
1. Import size modules
2. Initialize size encoder + controller
3. Update dataset to SizeAwareVitonHDDataset
4. Add size encoding in training loop
5. Add size-specific losses

### Task 3: Modify UNet 🔄
**Time:** 3-4 hours
**Files to modify:**
- `src/unet_hacked_tryon.py` (add size parameters to forward)
- `src/attentionhacked_tryon.py` (use size in attention)
**Strategy:** Start minimal, add complexity gradually

### Task 4: Initial Training ⏳
**Time:** 4-6 hours (actual GPU time)
**Prerequisites:** Tasks 1-3 complete
**Dataset:** VITON-HD (needs to be available)
**GPU:** Ideally 4× A100 80GB (but can start with less)

### Task 5: Evaluation ⏳
**Time:** 2-3 hours
**File to create:** `evaluate_size_aware.py`
**Metrics:** Size Accuracy, GFD, LPIPS, SSIM, FID

---

## 🔧 Troubleshooting Guide

### Issue: "ModuleNotFoundError: No module named 'size_modules'"

**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:/scratch/bds9746/CV_Vton/CV_Cindrella"
```

Or add to training script:
```python
import sys
sys.path.insert(0, '/scratch/bds9746/CV_Vton/CV_Cindrella')
```

### Issue: "Size extraction fails for some images"

**Solution:** The dataset loader has fallback to default values.
Check warnings in console. Most likely: missing OpenPose data.

**Temporary fix:** Use heuristic keypoints (already implemented in dataset)

### Issue: "Out of memory during training"

**Solutions:**
1. Reduce batch size: `--train_batch_size=2`
2. Increase gradient accumulation: `--gradient_accumulation_steps=4`
3. Use gradient checkpointing: `--gradient_checkpointing`
4. Use simple controller instead of full controller

### Issue: "Training loss not decreasing"

**Check:**
1. Are size modules actually being trained? (check `requires_grad`)
2. Is learning rate appropriate? (try 1e-5 to 5e-5)
3. Are losses weighted correctly?
4. Is data augmentation too aggressive?

---

## 📊 Expected Timeline

**Optimistic (everything works):** 2-3 days
**Realistic (debugging needed):** 5-7 days
**Conservative (significant issues):** 1-2 weeks

### Day 1: Setup & Testing
- ✅ Test modules (DONE)
- ✅ Create training script
- ✅ Run first training experiment

### Day 2-3: Training
- Run Stage 3 training (50 epochs)
- Monitor metrics
- Debug issues

### Day 4: Evaluation
- Implement metrics
- Evaluate trained model
- Compare with baseline

### Day 5-7: Iteration
- Fix issues found in evaluation
- Re-train if needed
- Prepare for Stage 4 (joint fine-tuning)

---

## 🎓 Training Tips

### 1. Start Small
- Use small batch size first (2-4)
- Test on subset of data (100-500 samples)
- Ensure no errors before full training

### 2. Monitor Closely
- Watch size accuracy metric
- Check generated samples every few epochs
- Visualize size maps to ensure they make sense

### 3. Save Often
- Checkpoint every 5-10 epochs
- Save size modules separately
- Keep best checkpoint based on validation

### 4. Ablation Studies
Try training with/without:
- Size augmentation
- Size Controller (use random maps)
- Hybrid encoder vs simple encoder

---

## 📦 Required Resources

### Data:
- [ ] VITON-HD dataset (11,647 pairs)
- [ ] Preprocessed features (DensePose, masks, etc.)
- [ ] (Optional) OpenPose keypoints JSON files

### Compute:
- [ ] GPU access (ideally 4× A100 80GB)
- [ ] Disk space (~100GB for checkpoints)
- [ ] Conda environment with dependencies

### Checkpoints:
- [ ] Pretrained SDXL weights
- [ ] Pretrained IDM-VTON (if available)
- [ ] IP-Adapter weights (if separate)

---

## ✅ Checklist Before Training

- [ ] All modules tested and working
- [ ] Dataset loader returns size info correctly
- [ ] Training script created and debugged
- [ ] UNet modified to accept size parameters
- [ ] GPU resources allocated
- [ ] VITON-HD dataset available and preprocessed
- [ ] Output directory created
- [ ] Logging configured
- [ ] Accelerate config set up

---

## 💾 File Checklist

**Already Created:**
- [x] `size_modules/__init__.py`
- [x] `size_modules/size_annotation.py`
- [x] `size_modules/size_encoder.py`
- [x] `size_modules/size_controller.py`
- [x] `size_modules/README.md`
- [x] `size_aware_dataset.py`
- [x] `IMPLEMENTATION_LOG.md`
- [x] `SIZE_AWARE_IMPLEMENTATION_SUMMARY.md`
- [x] `NEXT_STEPS.md` (this file)

**To Create:**
- [ ] `train_size_aware.py` (Stage 3 training)
- [ ] `train_joint.py` (Stage 4 training)
- [ ] `evaluate_size_aware.py` (Evaluation)
- [ ] `inference_size_aware.py` (Size-controllable inference)

**To Modify:**
- [ ] `src/unet_hacked_tryon.py`
- [ ] `src/attentionhacked_tryon.py`
- [ ] `gradio_demo/app.py` (add size slider)

---

## 🚀 Quick Start Command

Once everything is ready:

```bash
# Activate environment
conda activate idm

# Test modules first
python size_modules/size_encoder.py

# Run training
accelerate launch train_size_aware.py \
    --data_dir="/path/to/VITON-HD" \
    --output_dir="./results/size_aware_stage3" \
    --train_batch_size=4 \
    --num_train_epochs=50 \
    --learning_rate=5e-5 \
    --gradient_checkpointing \
    --use_8bit_adam
```

---

## 📞 Questions to Answer Before Training

1. **Where is VITON-HD dataset?** → Locate path
2. **Do we have pretrained checkpoints?** → Check ckpt/
3. **What GPU resources available?** → Check allocation
4. **How long can we train?** → Plan schedule
5. **Where to save checkpoints?** → Set output_dir

---

**Good luck with training! 🎉**

**Last Updated:** 2025-11-30
