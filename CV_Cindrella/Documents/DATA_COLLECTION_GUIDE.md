# Custom Dataset Collection Guide for Size-Aware VTON

**Purpose:** This guide tells your data team exactly what data to collect and how to prepare it for size-aware virtual try-on training.

---

## 📊 Overview: What We Need

We need **paired images** of:
1. **Models wearing garments** (in various sizes)
2. **The same garments laid flat** (product photos)

**Key Requirement:** The same garment should be photographed on models with **different body types** to create size diversity (tight/fitted/loose/oversized fit).

---

## 🎯 Size Distribution Target

For a balanced dataset, aim for this distribution:

| Size Category | Target % | Description | Example |
|--------------|----------|-------------|---------|
| **Tight** | 15-20% | Garment smaller than body | Size S on size L model |
| **Fitted** | 30-40% | Garment matches body | Size M on size M model |
| **Loose** | 25-35% | Garment larger than body | Size L on size M model |
| **Oversized** | 10-20% | Garment much larger | Size XL on size S model |

**Total samples needed:** Minimum 500 pairs (ideally 1000+)

---

## 📸 Data Collection Requirements

### **A. Model Images (Person wearing garment)**

#### Required Shots:
1. **Full-body frontal view**
   - Model standing straight, facing camera
   - Arms slightly away from body
   - Neutral pose (similar to VITON-HD)
   - Clear view of entire garment

#### Image Specifications:
- **Resolution:** 1024×768 pixels minimum (portrait orientation)
- **Format:** JPG or PNG
- **Background:** Solid, neutral color (white, gray, or light blue)
- **Lighting:** Even, no harsh shadows
- **Camera angle:** Straight-on, not from above/below

#### Size Variations - CRITICAL:
For each garment, photograph it on **at least 3 different body sizes**:
- **Small model + Small garment** → Fitted
- **Small model + Large garment** → Oversized
- **Medium model + Medium garment** → Fitted
- **Medium model + Small garment** → Tight
- **Medium model + Large garment** → Loose
- **Large model + Small garment** → Very tight
- **Large model + Large garment** → Fitted

This creates natural size variation!

---

### **B. Garment Images (Flat lay product photos)**

#### Required Shots:
1. **Garment laid flat on solid background**
   - Spread out naturally (not folded)
   - All details visible
   - Centered in frame

#### Image Specifications:
- **Resolution:** 768×1024 pixels minimum
- **Format:** JPG or PNG
- **Background:** Pure white (#FFFFFF) or neutral
- **No shadows, wrinkles, or distortions**

---

## 📁 Dataset Structure

Organize your collected data following this structure:

```
your_custom_dataset/
├── train/
│   ├── image/                    # Models wearing garments
│   │   ├── 00000_00.jpg
│   │   ├── 00001_00.jpg
│   │   └── ...
│   │
│   ├── cloth/                    # Flat garment photos
│   │   ├── 00000_00.jpg
│   │   ├── 00001_00.jpg
│   │   └── ...
│   │
│   ├── image-densepose/          # [Generate after collection]
│   ├── openpose_json/            # [Generate after collection]
│   ├── cloth-mask/               # [Generate after collection]
│   └── vitonhd_train_tagged.json # [Create after collection]
│
└── test/
    ├── image/
    ├── cloth/
    └── ... (same structure as train)
```

---

## 🔤 File Naming Convention

Use this naming format: `{ID}_{SIZE}.jpg`

Examples:
- `00001_00.jpg` - Model 1, Size fitted
- `00001_01.jpg` - Model 1, Size loose (different garment size)
- `00002_00.jpg` - Model 2, Size tight

**Naming Guide:**
- `{ID}` = 5-digit identifier (00001, 00002, etc.)
- `{SIZE}` = 2-digit size variation (00, 01, 02, etc.)

**Important:** The same garment on different models should have **different IDs** but can reuse cloth images.

---

## 📝 Required Metadata File

Create `vitonhd_train_tagged.json` with this structure:

```json
{
  "upper_body": [
    {
      "file_name": "00001_00.jpg",
      "tag_info": [
        {"tag_name": "item", "tag_category": "upper_body"},
        {"tag_name": "sleeveLength", "tag_category": "long_sleeve"},
        {"tag_name": "neckLine", "tag_category": "round_neck"},
        {"tag_name": "fit", "tag_category": "fitted"},
        {"tag_name": "garment_size", "tag_category": "M"},
        {"tag_name": "model_size", "tag_category": "M"}
      ]
    },
    {
      "file_name": "00002_00.jpg",
      "tag_info": [
        {"tag_name": "item", "tag_category": "upper_body"},
        {"tag_name": "sleeveLength", "tag_category": "short_sleeve"},
        {"tag_name": "neckLine", "tag_category": "v_neck"},
        {"tag_name": "fit", "tag_category": "oversized"},
        {"tag_name": "garment_size", "tag_category": "XL"},
        {"tag_name": "model_size", "tag_category": "S"}
      ]
    }
  ]
}
```

**Required Tags:**
- `item`: Always "upper_body"
- `sleeveLength`: long_sleeve, short_sleeve, sleeveless
- `neckLine`: round_neck, v_neck, collar, etc.
- `fit`: tight, fitted, loose, oversized (your manual label)
- `garment_size`: XS, S, M, L, XL, XXL
- `model_size`: XS, S, M, L, XL, XXL

---

## 🎨 Garment Type Requirements

### **For Balanced Dataset, Include:**

| Garment Type | Quantity | Examples |
|--------------|----------|----------|
| T-shirts | 30% | Plain tees, graphic tees, polo |
| Shirts | 25% | Button-downs, blouses |
| Sweaters | 20% | Crew neck, hoodies, cardigans |
| Jackets | 15% | Blazers, denim jackets |
| Tank tops | 10% | Sleeveless, athletic |

### **Style Variations:**
- Different sleeve lengths (long, short, sleeveless)
- Different neck styles (round, V-neck, collar, turtleneck)
- Different fits (slim-fit, regular, relaxed, oversized)
- Different fabrics (cotton, knit, denim, etc.)

---

## 🔧 Preprocessing Steps (After Data Collection)

After collecting raw images, you need to generate these files:

### **1. DensePose Maps** (`image-densepose/`)
```bash
# Use DensePose to generate body part segmentation
python generate_densepose.py --input train/image --output train/image-densepose
```

### **2. OpenPose JSON** (`openpose_json/`)
```bash
# Use OpenPose to extract body keypoints
python generate_openpose.py --input train/image --output train/openpose_json
```

### **3. Cloth Masks** (`cloth-mask/`)
```bash
# Use segmentation model to extract garment masks
python generate_cloth_masks.py --input train/cloth --output train/cloth-mask
```

### **4. Warped Garment Masks** (`gt_cloth_warped_mask/`)
```bash
# Warp flat garment to match body pose (for size calculation)
python warp_cloth_to_body.py --cloth train/cloth --body train/image --output train/gt_cloth_warped_mask
```

**Note:** We can help you set up these preprocessing scripts once you have the raw images.

---

## ✅ Data Quality Checklist

Before submitting data, verify:

- [ ] **Image Quality**
  - [ ] All images are sharp, not blurry
  - [ ] Consistent lighting across all shots
  - [ ] Clean background (no clutter)

- [ ] **Size Diversity**
  - [ ] At least 15% tight fits
  - [ ] At least 30% fitted
  - [ ] At least 25% loose
  - [ ] At least 10% oversized

- [ ] **File Organization**
  - [ ] All files follow naming convention
  - [ ] Image/cloth pairs match (same filename)
  - [ ] JSON metadata complete for all images

- [ ] **Coverage**
  - [ ] Multiple garment types represented
  - [ ] Multiple sleeve lengths
  - [ ] Multiple neck styles
  - [ ] Multiple model body types

---

## 📊 Size Calculation Method (For Reference)

Our system will automatically compute size ratios using:

```
Width Ratio = garment_shoulder_width / body_shoulder_width
Length Ratio = garment_length / body_torso_length
Shoulder Ratio = garment_shoulder_width / body_shoulder_width
```

**Classification:**
- **Tight:** ratio < 0.9
- **Fitted:** 0.9 ≤ ratio < 1.1
- **Loose:** 1.1 ≤ ratio < 1.3
- **Oversized:** ratio ≥ 1.3

**Example:**
- Small garment (40cm width) on Large body (50cm shoulders) → ratio = 0.8 → **Tight**
- Medium garment (45cm) on Medium body (45cm) → ratio = 1.0 → **Fitted**
- Large garment (55cm) on Medium body (45cm) → ratio = 1.22 → **Loose**
- XL garment (65cm) on Small body (40cm) → ratio = 1.625 → **Oversized**

---

## 🎯 Quick Start Plan

### **Phase 1: Pilot (100 samples)**
1. Select 10 garments
2. Photograph each on 3 different body sizes
3. Take flat lay photos of each garment
4. Organize files following structure above
5. Send to us for preprocessing and verification

### **Phase 2: Full Collection (500-1000 samples)**
1. Expand to 50+ garments
2. Multiple shots per garment-model combo
3. Ensure balanced size distribution
4. Complete metadata JSON
5. Full preprocessing pipeline

### **Phase 3: Validation**
1. We verify size calculations
2. Check distribution matches targets
3. Run test training
4. Adjust collection if needed

---

## 📸 Photography Setup Recommendations

### **Equipment:**
- Camera: DSLR or high-quality smartphone (12MP+)
- Tripod: For consistent framing
- Lighting: 2-3 softbox lights or natural window light
- Background: White/gray seamless paper or wall

### **Model Requirements:**
- At least 3 models with different body types:
  - **Small:** Generally size XS-S, ~160-165cm height
  - **Medium:** Generally size M, ~165-172cm height
  - **Large:** Generally size L-XL, ~172-180cm height

### **Shooting Process:**
1. Set up consistent camera position (mark floor)
2. Model stands in same spot for all shots
3. Take photo of model wearing garment
4. Change garment size, repeat
5. After model session: Flat lay photos of all garments used

---

## 💡 Pro Tips for Data Collection

1. **Consistency is key:** Same lighting, background, and camera angle for all shots
2. **Size variety:** Don't just photograph garments that fit - intentionally mismatch sizes!
3. **Natural poses:** Models should stand naturally, not stiff
4. **Clear garment details:** Make sure garment edges, seams, and details are visible
5. **Batch shooting:** Do all shots with one model before switching
6. **Label immediately:** Mark file names and metadata right after shooting (easy to forget later!)

---

## 📞 What to Send Us

Once you have collected data, send:

1. **Raw images** organized in folder structure above
2. **Metadata JSON** with all annotations
3. **Size distribution report** (how many tight/fitted/loose/oversized)
4. **Sample images** (2-3 examples of each size category for verification)

We will then:
- Run preprocessing to generate DensePose, OpenPose, masks
- Verify size calculations match expected distribution
- Run test training
- Give you feedback for next batch

---

## 📚 Example Dataset Structure

```
custom_dataset/
├── train/
│   ├── image/                    # 500 model images
│   │   ├── 00001_00.jpg         # Model 1, garment fits well
│   │   ├── 00001_01.jpg         # Model 1, same garment size up (loose)
│   │   ├── 00002_00.jpg         # Model 2 (different size), tight fit
│   │   └── ...
│   │
│   ├── cloth/                    # 100 unique garments
│   │   ├── 00001_00.jpg         # Garment A laid flat
│   │   ├── 00002_00.jpg         # Garment B laid flat
│   │   └── ...
│   │
│   └── vitonhd_train_tagged.json
│
└── test/                         # 100 pairs for testing
    └── ... (same structure)
```

---

## ❓ FAQ for Data Team

**Q: How many images do we need minimum?**
A: At least 500 train + 100 test pairs. More is better (1000+ ideal).

**Q: Can we use stock photos?**
A: Yes, IF you have rights and they meet quality requirements. Make sure to get variety in sizes.

**Q: Do we need professional models?**
A: No! Regular people are fine, just need variety in body sizes.

**Q: What if we can't get certain size combinations?**
A: Focus on the common combinations first (fitted, loose, oversized). Tight fits are hardest but most valuable.

**Q: Can we reuse garments across models?**
A: YES! That's encouraged. Same garment on different body sizes = perfect size variation.

**Q: How long will preprocessing take?**
A: For 500 images: ~2-3 hours on our H100 GPU. We'll handle this part.

---

## 🚀 Next Steps

1. **Read this guide thoroughly**
2. **Plan your shooting schedule** (models, garments, location)
3. **Start with pilot batch** (10 garments × 3 models = 30 pairs)
4. **Send pilot to us for verification**
5. **Get feedback, then proceed with full collection**
6. **Send batches as you complete them** (don't wait for everything)

---

**Questions?** Share this document with us and we'll clarify anything!

**Timeline estimate:**
- Pilot batch (30 pairs): 1-2 days shooting
- Full collection (500 pairs): 1-2 weeks shooting
- Preprocessing: 1 day per batch
- Training: 2-3 days for full model

---

**Document Version:** 1.0
**Last Updated:** 2025-11-30
**Contact:** Share questions in project chat
