# Size-Aware Modules for Cinderella Virtual Try-On

This package implements size-conditioning modules for the Cinderella size-aware virtual try-on system.

## Overview

The size-awareness capability is achieved through three core modules:

1. **Size Annotation** (`size_annotation.py`): Extracts garment and body dimensions to compute size ratios
2. **Size Encoder** (`size_encoder.py`): Maps size ratios to high-dimensional embeddings
3. **Size Controller** (`size_controller.py`): Generates spatial size guidance maps

## Module Descriptions

### 1. Size Annotation (`size_annotation.py`)

**Purpose:** Extract dimensional information from images to compute garment-to-body size ratios.

**Key Classes:**
- `SizeAnnotator`: Main class for dimension extraction
  - `extract_body_dimensions()`: Extract shoulder width, torso length from OpenPose keypoints
  - `extract_garment_dimensions()`: Extract garment width, length from segmentation masks

**Key Functions:**
- `compute_size_ratio(body_dims, garment_dims)`: Returns (width_ratio, length_ratio, shoulder_ratio)
- `get_size_label(width_ratio)`: Classifies into {tight, fitted, loose, oversized}
- `create_size_map(size_label, height, width)`: Generates spatial size guidance

**Size Classification:**
```python
tight:     ratio < 0.9
fitted:    0.9 ≤ ratio < 1.1
loose:     1.1 ≤ ratio < 1.3
oversized: ratio ≥ 1.3
```

**Usage Example:**
```python
from size_modules import SizeAnnotator, compute_size_ratio, get_size_label

annotator = SizeAnnotator()

# Extract dimensions
body_dims = annotator.extract_body_dimensions(openpose_keypoints)
garment_dims = annotator.extract_garment_dimensions(garment_mask)

# Compute ratios
width_ratio, length_ratio, shoulder_ratio = compute_size_ratio(body_dims, garment_dims)

# Get label
size_label = get_size_label(width_ratio)
```

---

### 2. Size Encoder (`size_encoder.py`)

**Purpose:** Encode size ratios into high-dimensional embeddings compatible with the diffusion model.

**Architecture:**
```
Input (3-dim) → Linear(256) → LayerNorm → GELU → Dropout
              → Linear(256) → LayerNorm → GELU → Dropout
              → Linear(768) → LayerNorm → Output (768-dim)
```

**Key Classes:**
- `SizeEncoder`: MLP-based continuous ratio encoder (~198K parameters)
- `SizeEmbeddingLayer`: Learnable embeddings for discrete size classes
- `HybridSizeEncoder`: Combines continuous and discrete encodings

**Usage Example:**
```python
from size_modules import SizeEncoder
import torch

encoder = SizeEncoder(
    input_dim=3,
    hidden_dim=256,
    output_dim=768,
    num_layers=3
)

# Encode size ratios
size_ratios = torch.tensor([[1.0, 1.05, 1.0]])  # fitted
size_embedding = encoder(size_ratios)  # (1, 768)
```

**Features:**
- Xavier initialization for stable training
- Input normalization to [-2, 2] range
- Layer normalization for better convergence
- Dropout for regularization

---

### 3. Size Controller (`size_controller.py`)

**Purpose:** Generate spatial size guidance maps from fused features and size embeddings.

**Architecture:**
- **Full Controller**: U-Net style with encoder-decoder (~1.2M parameters)
  - Residual blocks with skip connections
  - Spatial self-attention at bottleneck
  - Outputs (B, 1, H, W) size map

- **Simple Controller**: Lightweight MLP-based (~400K parameters)
  - Direct mapping from size embedding to spatial map
  - Faster for prototyping

**Key Classes:**
- `SizeController`: Full CNN-based controller with attention
- `SimpleSizeController`: Lightweight MLP-based alternative
- `ResidualBlock`: Building block with optional up/downsampling
- `SpatialAttention`: Self-attention for spatial features

**Usage Example:**
```python
from size_modules import SizeController, SimpleSizeController
import torch

# Full controller
controller = SizeController(
    in_channels=512,
    hidden_channels=256,
    output_size=(128, 96)
)

fused_features = torch.randn(2, 512, 128, 96)
size_embedding = torch.randn(2, 768)

size_map = controller(fused_features, size_embedding)  # (2, 1, 128, 96)

# Or use simple controller
simple_controller = SimpleSizeController(output_size=(128, 96))
size_map = simple_controller(size_embedding)
```

---

## Integration with Training Pipeline

### Step 1: Dataset Integration

Use the size-aware dataset loader:

```python
from size_aware_dataset import SizeAwareVitonHDDataset

dataset = SizeAwareVitonHDDataset(
    dataroot_path="/path/to/VITON-HD",
    phase="train",
    size_augmentation=True,  # Enable size augmentation
    enable_size_conditioning=True
)

# Each sample now includes:
# - size_ratios: (3,) tensor
# - size_label: str ('tight', 'fitted', 'loose', 'oversized')
# - size_label_id: int (0, 1, 2, 3)
# - size_map: (1, H/4, W/4) spatial guidance map
```

### Step 2: Model Integration

Initialize size modules:

```python
from size_modules import SizeEncoder, SizeController

size_encoder = SizeEncoder(
    input_dim=3,
    hidden_dim=256,
    output_dim=768
)

size_controller = SimpleSizeController(
    size_embedding_dim=768,
    output_size=(128, 96)  # Match latent dimensions
)
```

### Step 3: Training Loop

```python
# In training loop
for batch in dataloader:
    size_ratios = batch['size_ratios']  # (B, 3)

    # Encode size
    size_embedding = size_encoder(size_ratios)  # (B, 768)

    # Generate size map
    size_map = size_controller(size_embedding)  # (B, 1, H, W)

    # Pass to UNet
    noise_pred = unet(
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        size_embedding=size_embedding,  # Inject into cross-attention
        size_map=size_map,  # Modulate self-attention
        **kwargs
    )
```

---

## Training Strategy

### Stage 3: Size Module Training (50 epochs)

**Objective:** Train size encoder and controller while freezing base IDM-VTON

**Frozen:**
- TryonNet (UNet)
- GarmentNet (UNet Encoder)
- IP-Adapter

**Trainable:**
- Size Encoder
- Size Controller

**Loss Function:**
```python
L_total = L_rec + 0.5 * L_size_consistency + 0.3 * L_spatial

where:
- L_rec: Reconstruction loss (ensure visual quality)
- L_size_consistency: Size prediction consistency
- L_spatial: Spatial alignment of size maps
```

**Data Augmentation:**
- Garment scaling: 0.7× to 1.5× to create size variations
- This synthetically generates tight/loose examples from fitted garments

### Stage 4: Joint Fine-tuning (30 epochs)

**Objective:** End-to-end optimization of all components

**Trainable:**
- TryonNet decoder (keep encoder frozen for stability)
- IP-Adapter projection
- Size Encoder
- Size Controller

**Loss Function:**
```python
L_total = 0.3 * L_idm + 0.25 * L_ip + 0.25 * L_size + 0.15 * L_detail + 0.05 * L_human
```

---

## File Structure

```
size_modules/
├── __init__.py              # Package initialization
├── README.md                # This file
├── size_annotation.py       # Size ratio extraction (352 lines)
├── size_encoder.py          # Size embedding encoder (275 lines)
├── size_controller.py       # Spatial guidance generator (320 lines)
└── size_aware_attention.py  # Attention modifications (optional)
```

---

## Parameters Summary

| Module | Parameters | Input | Output |
|--------|-----------|-------|--------|
| Size Encoder | ~198K | (B, 3) ratios | (B, 768) embedding |
| Size Controller (Full) | ~1.2M | (B, 512, H, W) + (B, 768) | (B, 1, H, W) map |
| Size Controller (Simple) | ~400K | (B, 768) | (B, 1, H, W) map |

---

## Testing

Each module includes built-in tests. To run:

```bash
# Activate conda environment first
conda activate idm

# Test individual modules
python size_modules/size_annotation.py
python size_modules/size_encoder.py
python size_modules/size_controller.py
```

---

## Notes

- **Quick Approach**: Currently uses OpenPose keypoints for body dimensions (fast but less precise)
- **Future Improvement**: Train dedicated 10-point landmark detector on DeepFashion2
- **Fallback**: If size extraction fails, defaults to 'fitted' (ratio = 1.0)
- **Compatibility**: Designed to be backward compatible with original IDM-VTON

---

## References

- IDM-VTON (Choi et al., ECCV 2024): Base architecture
- Size Does Matter (Chen et al., ICCV 2023): Size-aware VTON inspiration
- DeepFashion2: Landmark annotations for future improvement

---

**Author:** Cinderella Team
**Date:** 2025-11-30
**Version:** 1.0
