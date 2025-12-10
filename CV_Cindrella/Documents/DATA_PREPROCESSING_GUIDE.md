# Data Preprocessing & Size Calculation Guide

**For Data Team: How to Prepare Dataset for Size-Aware VTON Training**

---

## 📊 **Overview: What Happens During Training**

During training, for each image pair, we automatically compute:
1. Body dimensions from pose keypoints
2. Garment dimensions from segmentation masks
3. Size ratios (garment/body)
4. Size labels (tight/fitted/loose/oversized)
5. Spatial size guidance maps

**NO MANUAL SIZE LABELING REQUIRED!**

---

## 🔍 **Detailed Size Calculation Pipeline**

### **Step 1: Extract Body Dimensions from OpenPose**

**Input:** OpenPose keypoints JSON file (18 keypoints)

**Process:**
```python
# From keypoints, extract:
keypoints = {
    'neck': [x, y, confidence],           # keypoint 1
    'left_shoulder': [x, y, confidence],  # keypoint 5
    'right_shoulder': [x, y, confidence], # keypoint 2
    'left_hip': [x, y, confidence],       # keypoint 11
    'right_hip': [x, y, confidence],      # keypoint 8
}

# Calculate body dimensions:
shoulder_width = distance(left_shoulder, right_shoulder)
torso_length = distance(neck, midpoint(left_hip, right_hip))
body_width_at_waist = distance(left_hip, right_hip)
```

**Output:**
```python
body_dims = {
    'shoulder_width': 95.3,    # pixels
    'torso_length': 187.6,     # pixels
    'body_width_at_waist': 81.2  # pixels
}
```

---

### **Step 2: Extract Garment Dimensions from Mask**

**Input:** Garment segmentation mask (binary image, H×W)

**Process:**
```python
# Find garment contours in mask
contours = cv2.findContours(garment_mask, ...)
main_contour = max(contours, key=cv2.contourArea)

# Get bounding box
x, y, w, h = cv2.boundingRect(main_contour)

garment_length = h  # height of bounding box
garment_width = w   # width of bounding box

# Measure shoulder width at top 20% of garment
shoulder_region = mask[y:y+int(h*0.2), :]
garment_shoulder_width = width_of_nonzero_pixels(shoulder_region)
```

**Output:**
```python
garment_dims = {
    'garment_length': 195.0,    # pixels
    'garment_width': 114.5,     # pixels
    'garment_shoulder_width': 110.2  # pixels
}
```

---

### **Step 3: Compute Size Ratios**

**Formula:**
```python
width_ratio = garment_width / body_shoulder_width
length_ratio = garment_length / body_torso_length
shoulder_ratio = garment_shoulder_width / body_shoulder_width
```

**Example:**
```python
width_ratio = 114.5 / 95.3 = 1.201
length_ratio = 195.0 / 187.6 = 1.039
shoulder_ratio = 110.2 / 95.3 = 1.156
```

---

### **Step 4: Classify Size Label**

**Classification Rules:**
```python
if width_ratio < 0.9:
    size_label = "tight"      # XS garment on L body
elif 0.9 <= width_ratio < 1.1:
    size_label = "fitted"     # M garment on M body
elif 1.1 <= width_ratio < 1.3:
    size_label = "loose"      # L garment on S body
else:  # >= 1.3
    size_label = "oversized"  # XXL garment on XS body
```

**Example:** width_ratio = 1.201 → **"loose"**

---

### **Step 5: Create Spatial Size Map**

**Process:**
```python
# Convert label to continuous value
label_to_value = {
    'tight': 0.0,
    'fitted': 0.33,
    'loose': 0.66,
    'oversized': 1.0
}

value = label_to_value[size_label]  # 0.66 for "loose"

# Create spatial map (all pixels have same value for now)
size_map = np.ones((H//4, W//4)) * value  # Shape: (128, 96)
```

**Output:** Spatial guidance map with values [0, 1]

---

## 📦 **Required Dataset Structure**

For each training sample, you need:

```
dataset/
├── train/
│   ├── image/                  # Person images (JPG, 1024×768)
│   ├── cloth/                  # Garment images (JPG)
│   ├── image-densepose/        # DensePose visualizations (PNG/JPG)
│   ├── agnostic-mask/          # Inpainting masks (PNG, binary)
│   ├── openpose_json/          # OpenPose keypoints (JSON) ← CRITICAL
│   └── vitonhd_train_tagged.json  # Garment annotations
└── test/
    └── (same structure)
```

---

## 🔑 **Critical Files Explained**

### **1. OpenPose JSON (MOST IMPORTANT for size calculation)**

**Format:**
```json
{
  "version": 1.3,
  "people": [
    {
      "pose_keypoints_2d": [
        x0, y0, conf0,  // nose
        x1, y1, conf1,  // neck
        x2, y2, conf2,  // right shoulder
        x3, y3, conf3,  // right elbow
        x4, y4, conf4,  // right wrist
        x5, y5, conf5,  // left shoulder
        x6, y6, conf6,  // left elbow
        x7, y7, conf7,  // left wrist
        x8, y8, conf8,  // right hip
        x9, y9, conf9,  // right knee
        x10, y10, conf10,  // right ankle
        x11, y11, conf11,  // left hip
        // ... 18 keypoints total
      ]
    }
  ]
}
```

**Keypoints we use:**
- 1: Neck (for torso length)
- 2, 5: Right/Left shoulders (for shoulder width)
- 8, 11: Right/Left hips (for torso length, waist width)

---

### **2. Garment Segmentation Mask**

**Requirements:**
- Binary mask (0 = background, 255 = garment)
- Same dimensions as garment image
- Clean segmentation (no holes)

**Format:** PNG or JPG, single channel or RGB with same values

---

### **3. DensePose**

**Purpose:** Pose guidance for try-on generation

**Format:** RGB visualization image (IUV format)

---

### **4. Agnostic Mask**

**Purpose:** Mask out original garment from person image

**Requirements:**
- Binary mask showing areas to inpaint
- Preserves arms, skin tone, body shape
- Removes only the garment

---

## 🔄 **Data Preprocessing During Training**

### **Full Pipeline (What Happens Per Sample):**

```python
# 1. Load images
person_image = load_image("train/image/00001_00.jpg")
garment_image = load_image("train/cloth/00001_00.jpg")
densepose = load_image("train/image-densepose/00001_00.jpg")
mask = load_image("train/agnostic-mask/00001_00_mask.png")

# 2. Load OpenPose keypoints
keypoints = load_json("train/openpose_json/00001_00_keypoints.json")

# 3. Extract body dimensions
body_dims = extract_body_dimensions(keypoints)
# → shoulder_width, torso_length, body_width_at_waist

# 4. Extract garment dimensions
garment_mask = segment_garment(garment_image)  # or load pre-computed
garment_dims = extract_garment_dimensions(garment_mask)
# → garment_width, garment_length, garment_shoulder_width

# 5. Compute size ratios
width_ratio = garment_dims['garment_width'] / body_dims['shoulder_width']
length_ratio = garment_dims['garment_length'] / body_dims['torso_length']
shoulder_ratio = garment_dims['garment_shoulder_width'] / body_dims['shoulder_width']

# 6. Get size label
size_label = get_size_label(width_ratio)  # tight/fitted/loose/oversized
size_label_id = {'tight': 0, 'fitted': 1, 'loose': 2, 'oversized': 3}[size_label]

# 7. Create size map
size_map = create_size_map(size_label, height=128, width=96)

# 8. Standard augmentations
if training:
    # Random horizontal flip
    if random() > 0.5:
        person_image = flip(person_image)
        garment_image = flip(garment_image)
        mask = flip(mask)
        densepose = flip(densepose)

    # Color jitter
    if random() > 0.5:
        person_image = adjust_brightness/contrast/hue/saturation(person_image)
        garment_image = adjust_brightness/contrast/hue/saturation(garment_image)

    # Random scale (0.8-1.2x)
    if random() > 0.5:
        person_image = scale(person_image, factor=random(0.8, 1.2))
        mask = scale(mask, factor=same)
        densepose = scale(densepose, factor=same)

    # Random shift
    if random() > 0.5:
        person_image = shift(person_image, dx=random(-0.2, 0.2), dy=random(-0.2, 0.2))
        mask = shift(mask, dx=same, dy=same)
        densepose = shift(densepose, dx=same, dy=same)

    # SIZE AUGMENTATION (key for size diversity!)
    if random() > 0.5:
        garment_scale = random(0.7, 1.5)  # Scale garment to create size variations
        garment_dims['garment_width'] *= garment_scale
        garment_dims['garment_length'] *= garment_scale
        garment_dims['garment_shoulder_width'] *= garment_scale
        # Recompute ratios with scaled garment
        width_ratio = garment_dims['garment_width'] / body_dims['shoulder_width']
        # This creates synthetic tight/loose examples!

# 9. Normalize and encode
person_latent = vae.encode(person_image)
garment_latent = vae.encode(garment_image)
densepose_latent = vae.encode(densepose)

# 10. CLIP encoding for garment
clip_features = clip_processor(garment_image)

# 11. Text prompts
text_prompt = f"model is wearing {garment_annotation}"
cloth_prompt = f"a photo of {garment_annotation}"

# 12. Return batch
return {
    'image': person_latent,
    'cloth': clip_features,
    'cloth_pure': garment_latent,
    'inpaint_mask': mask,
    'pose_img': densepose_latent,
    'size_ratios': [width_ratio, length_ratio, shoulder_ratio],  # ← SIZE INFO
    'size_label': size_label,
    'size_label_id': size_label_id,
    'size_map': size_map,
    'caption': text_prompt,
    'caption_cloth': cloth_prompt
}
```

---

## 📝 **For Your Data Team: What to Prepare**

### **Minimum Requirements:**

1. **Person Images** (JPG, 1024×768 or 768×1024)
2. **Garment Images** (JPG, any size - will be resized)
3. **OpenPose Keypoints** (JSON, 18-point COCO format) ← **CRITICAL**
4. **DensePose Maps** (RGB visualization)
5. **Agnostic Masks** (Binary PNG)

### **Optional but Helpful:**

6. **Garment Masks** (for better dimension extraction)
7. **Garment Annotations** (JSON with category/style info)

### **Tools to Generate:**

- **OpenPose:** https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **DensePose:** Detectron2 with DensePose configs
- **Garment Segmentation:** SAM, U2-Net, or Graphonomy
- **Human Parsing:** SCHP, Graphonomy, or CIHP

---

## 💾 **Expected Data Volumes**

- **Training Set:** 10,000-15,000 image pairs
- **Test Set:** 2,000-3,000 image pairs
- **Storage:** ~50-70GB total (with all preprocessed features)

---

## ✅ **Quality Checks**

Before providing dataset, verify:

1. ✅ All OpenPose JSONs have valid keypoints (confidence > 0.3)
2. ✅ All masks are binary (0 and 255 only)
3. ✅ Image dimensions match (person and masks same size)
4. ✅ No missing files (every image has corresponding mask, DensePose, etc.)
5. ✅ JSON format is correct (test loading with Python)

---

## 🎯 **Example Calculation Walkthrough**

**Sample:** Person wearing a loose shirt

**Inputs:**
- Person image: 1024×768 pixels
- OpenPose keypoints show:
  - Shoulder width: 100 pixels
  - Torso length: 200 pixels
- Garment mask bounding box:
  - Width: 120 pixels
  - Length: 210 pixels

**Calculations:**
```
width_ratio = 120 / 100 = 1.20
length_ratio = 210 / 200 = 1.05
shoulder_ratio = 115 / 100 = 1.15

Classification: width_ratio = 1.20 → "loose" (1.1 ≤ 1.20 < 1.3)
```

**Result:** Model learns that ratio ~1.2 should produce loose-fitting appearance

---

**Questions? Contact the ML team!**

**Last Updated:** 2025-11-30
