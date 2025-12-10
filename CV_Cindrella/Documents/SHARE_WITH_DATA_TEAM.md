# 📦 Files to Share with Your Data Team

---

## 🎯 Main Document for Data Collection

### **📄 DATA_COLLECTION_GUIDE.md** ← START HERE
**Purpose:** Complete guide for collecting custom dataset

**What it covers:**
- ✅ What images to collect (models + garments)
- ✅ How many samples needed (500-1000)
- ✅ Size distribution targets (15% tight, 35% fitted, 30% loose, 20% oversized)
- ✅ Photography requirements (resolution, lighting, poses)
- ✅ File organization structure
- ✅ Metadata JSON format
- ✅ Quality checklist
- ✅ Step-by-step plan

**Action for data team:** Read this first, follow the guide

---

## 📊 Supporting Documents

### **1. DATA_PREPROCESSING_GUIDE.md**
**Purpose:** Explains HOW size calculations work

**What it covers:**
- Size ratio formulas
- OpenPose keypoint extraction
- Garment dimension measurement
- Classification rules (tight/fitted/loose/oversized)
- Complete preprocessing pipeline

**Use this for:** Understanding the technical details behind size calculation

---

### **2. SIZE_CALCULATION_FIX_SUMMARY.md**
**Purpose:** Explains why we need custom dataset

**What it covers:**
- Bug we discovered (95% oversized in VITON-HD)
- How we fixed it (warped masks vs flat cloth)
- Why VITON-HD isn't ideal (naturally oversized)
- Why balanced custom dataset is needed

**Use this for:** Context on why data collection is important

---

### **3. training_verbose_log.txt** (To be generated)
**Purpose:** Real examples of size calculations

**How to generate:**
```bash
cd /scratch/bds9746/CV_Vton/CV_Cindrella

python train_small_verbose.py \
    --data_dir /scratch/bds9746/datasets/VITON-HD \
    --num_samples 100 \
    --num_epochs 3 \
    --batch_size 2 \
    --log_every 10 \
    2>&1 | tee training_verbose_log.txt
```

**What it shows:**
- Real size calculations from VITON-HD
- Width/length/shoulder ratios
- Size classifications
- Data shapes and preprocessing

**Use this for:** Showing data team what the output looks like

---

## 🎬 Quick Start for Data Team

### **Phase 1: Understanding (30 minutes)**
1. Read **DATA_COLLECTION_GUIDE.md** fully
2. Skim **DATA_PREPROCESSING_GUIDE.md** (technical details)
3. Look at **training_verbose_log.txt** examples

### **Phase 2: Planning (1-2 days)**
1. Plan photography session (location, equipment, models)
2. Select garments to photograph (aim for variety)
3. Create shooting schedule
4. Set up file organization system

### **Phase 3: Pilot Collection (2-3 days)**
1. Collect **30 sample pairs** (10 garments × 3 body sizes)
2. Follow DATA_COLLECTION_GUIDE.md structure
3. Create metadata JSON
4. Send to us for verification

### **Phase 4: Full Collection (1-2 weeks)**
1. Based on feedback, collect full dataset (500+ pairs)
2. Ensure balanced size distribution
3. Send in batches (don't wait for everything)

### **Phase 5: Preprocessing (1 day per batch)**
1. We run DensePose, OpenPose, segmentation
2. Verify size calculations
3. Prepare for training

---

## 📋 What Data Team Needs to Provide

### **Minimum Requirements:**

1. **Model Images** (`train/image/`)
   - 500+ images of models wearing garments
   - Multiple models with different body sizes
   - Full-body frontal view
   - 1024×768 resolution
   - Clean background

2. **Garment Images** (`train/cloth/`)
   - Flat lay photos of garments
   - 768×1024 resolution
   - White background
   - Same garments as worn by models

3. **Metadata JSON** (`vitonhd_train_tagged.json`)
   - Tags for each image (sleeve length, neck type, fit, etc.)
   - Garment size (XS/S/M/L/XL)
   - Model size (XS/S/M/L/XL)

4. **Size Distribution Report**
   - How many tight/fitted/loose/oversized samples
   - Target: balanced across all categories

---

## 💡 Key Points to Emphasize

### **1. Size Diversity is CRITICAL**
```
❌ BAD:  All fitted garments (like VITON-HD)
✅ GOOD: Mix of tight, fitted, loose, oversized

Example approach:
- Take 1 garment (size M)
- Photograph on 3 models:
  * Small model → garment looks oversized
  * Medium model → garment looks fitted
  * Large model → garment looks tight
```

### **2. Consistency Matters**
- Same camera position for all shots
- Same lighting setup
- Same background
- Same pose style

### **3. Quality Over Quantity**
- Better to have 300 high-quality diverse samples
- Than 1000 low-quality or similar samples

---

## 🔍 Example Size Combinations

To make it crystal clear, here are specific examples:

### **Example 1: T-Shirt**
| Garment | Model Body Size | Result | Ratio | Label |
|---------|----------------|--------|-------|-------|
| Size S | Small (XS-S) | Perfect fit | ~1.0 | Fitted |
| Size S | Medium (M) | Too tight | ~0.8 | Tight |
| Size M | Small (XS-S) | Oversized | ~1.4 | Oversized |
| Size M | Medium (M) | Good fit | ~1.05 | Fitted |
| Size L | Medium (M) | Loose fit | ~1.2 | Loose |

### **Example 2: Sweater**
| Garment | Model Body Size | Result | Ratio | Label |
|---------|----------------|--------|-------|-------|
| Size XS | Medium (M) | Very tight | ~0.7 | Tight |
| Size S | Medium (M) | Fitted | ~0.95 | Fitted |
| Size M | Medium (M) | Normal | ~1.08 | Fitted |
| Size L | Medium (M) | Loose | ~1.25 | Loose |
| Size XL | Small (XS-S) | Very loose | ~1.6 | Oversized |

**Strategy:** Same garment on different body sizes = natural size variation!

---

## ❓ FAQ for Data Team

**Q: Can we use existing fashion photography?**
A: Only if you have rights and they meet requirements (full body, clean background, consistent)

**Q: Do models need to be professional?**
A: No! Regular people are perfect. We need body size diversity, not modeling skills.

**Q: How exact do measurements need to be?**
A: Don't stress exact measurements. Our algorithm auto-calculates from images.

**Q: What if we can't get all size combinations?**
A: Focus on common combinations first. Send what you have in batches.

**Q: Can we shoot over multiple days?**
A: Yes! Just maintain consistency in setup.

---

## 📞 Contact & Questions

**When data team has questions, they should ask about:**
- Photography setup
- File organization
- Metadata format
- Size distribution targets
- Quality requirements

**We can help with:**
- Preprocessing (DensePose, OpenPose, etc.)
- Verification of collected data
- Technical questions about size calculation
- Feedback on pilot batch

---

## ✅ Before Data Collection Starts

**Data team should confirm:**
- [ ] Read and understood DATA_COLLECTION_GUIDE.md
- [ ] Have access to models with different body sizes
- [ ] Have photography equipment ready
- [ ] Understand file naming convention
- [ ] Know target size distribution (15/35/30/20)
- [ ] Ready to start with pilot batch (30 samples)

---

## 🚀 Timeline

### **Pilot Batch (30 samples)**
- Collection: 2-3 days
- Verification by us: 1 day
- Feedback & adjustments: 1 day
- **Total: ~1 week**

### **Full Dataset (500 samples)**
- Collection: 1-2 weeks (can batch)
- Preprocessing: 1 day per 100 samples
- **Total: 2-3 weeks**

### **Training & Results**
- Training: 2-3 days
- Evaluation: 1 day
- **Total: 3-4 days**

**Grand Total: 3-4 weeks** from start to trained model

---

## 📦 Deliverables from Data Team

1. **Raw Images**
   - Organized in folder structure
   - Named according to convention

2. **Metadata JSON**
   - Complete tags for all images

3. **Distribution Report**
   - Count of tight/fitted/loose/oversized samples
   - Any issues or concerns

4. **Sample Preview**
   - 2-3 examples of each size category
   - For quick verification

---

## 🎯 Success Criteria

Data collection is successful when:
- ✅ 500+ high-quality image pairs
- ✅ Balanced size distribution (not 90% one category)
- ✅ Variety in garment types
- ✅ Consistent quality across all images
- ✅ Complete metadata for all samples
- ✅ Our size calculation algorithm works on the data

---

**Good luck with data collection! 🎉**

**Remember:** Quality and diversity matter more than quantity!
