# How to Test the Pipeline

The test script is ready, but you need to activate your Python environment first.

## ✅ **Quick Test (Recommended)**

### Step 1: Activate Your Environment

You need to activate the conda environment with PyTorch. Try these commands:

```bash
# Option A: Use the helper script (Recommended)
source activate_idm.sh

# Option B: If you have conda initialized
conda activate idm

# Option C: Manual activation (if conda not available)
export PATH="/scratch/bds9746/envs/idm/bin:$PATH"
export CONDA_PREFIX="/scratch/bds9746/envs/idm"
export LD_LIBRARY_PATH="/scratch/bds9746/envs/idm/lib:$LD_LIBRARY_PATH"

# Option D: If using module system
module load python/intel/3.8.6
module load cuda/11.6.2
```

### Step 2: Verify PyTorch is Available

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

You should see something like:
```
PyTorch 2.0.1 - CUDA: True
```

### Step 3: Run the Test

```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella
python test_pipeline.py
```

---

## 📋 **What the Test Does**

The test script (`test_pipeline.py`) checks:

1. ✅ **Module Imports** - Can import all size modules
2. ✅ **Size Annotation** - Extracts dimensions and computes ratios
3. ✅ **Size Encoder** - Converts ratios to 768-dim embeddings
4. ✅ **Size Controller** - Generates spatial size maps
5. ✅ **Dataset Loader** - Loads size-aware data
6. ✅ **End-to-End Flow** - Complete pipeline works
7. ✅ **GPU Check** - Verifies CUDA availability

---

## 🎯 **Expected Output**

If everything works, you should see:

```
============================================================
Size-Aware Pipeline Test
============================================================

Test 1: Checking module imports...
✅ size_modules imports successful
✅ size_aware_dataset import successful

Test 2: Testing size annotation...
  Body dimensions: {...}
  Garment dimensions: {...}
  Size ratios: width=1.200, length=1.000, shoulder=1.200
  Size label: loose
✅ Size annotation working

Test 3: Testing size encoder...
  Using device: cuda
  Input shape: torch.Size([4, 3])
  Output shape: torch.Size([4, 768])
✅ Size encoder working

Test 4: Testing size controller...
  Output shape: torch.Size([4, 1, 128, 96])
✅ Size controller working

... (more tests)

============================================================
✅ Pipeline test complete!
============================================================
```

---

## 🐛 **Troubleshooting**

### "ModuleNotFoundError: No module named 'torch'"

**Solution:** Activate your conda environment first
```bash
conda activate idm  # or your environment name
```

### "ModuleNotFoundError: No module named 'size_modules'"

**Solution:** You're not in the right directory
```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella
python test_pipeline.py
```

### "CUDA not available"

**Solution:** That's OK for testing! The test will run on CPU (slower but works)

---

## 🚀 **After Testing Passes**

Once the test passes, you're ready to:

1. **Download VITON-HD** (if not already done)
2. **Run training** with `train_size_aware.sh`
3. **Evaluate results**

---

## 💡 **Manual Test (If Automated Fails)**

You can also test each component manually:

```bash
# Activate environment first!
conda activate idm

# Test size encoder
python size_modules/size_encoder.py

# Test size controller
python size_modules/size_controller.py

# Test size annotation
python size_modules/size_annotation.py
```

Each module has built-in tests that run when executed directly.

---

**Need Help?**

Check these files:
- Implementation details: `SIZE_AWARE_IMPLEMENTATION_SUMMARY.md`
- Training guide: `NEXT_STEPS.md`
- Quick start: `START_TRAINING_HERE.md`
