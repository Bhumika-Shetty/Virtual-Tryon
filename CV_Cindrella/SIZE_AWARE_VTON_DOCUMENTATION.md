# Size-Aware Virtual Try-On: Complete Technical Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset Specification](#3-dataset-specification)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Core Modules](#5-core-modules)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Configuration Reference](#8-configuration-reference)
9. [API Reference](#9-api-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### 1.1 Project Description

Size-Aware Virtual Try-On extends the IDM-VTON (Image-based Diffusion Model for Virtual Try-ON) system to understand and generate clothing fit based on body-clothing size relationships. The system learns to generate realistic try-on images that respect physical constraints - showing tight fits when clothing is smaller than body size, and loose/draped fits when clothing is larger.

### 1.2 Key Innovation

Traditional VTON models treat all body-clothing combinations identically. Our approach introduces:

- **Full Combinatorial Training**: Explicit modeling of 9 body×cloth size combinations
- **SizeEmbedder Module**: Learnable embeddings for size conditioning
- **Relative Fit Encoding**: Continuous signal capturing tightness/looseness
- **Optional IP-Adapter LoRA**: Efficient fine-tuning of garment encoder

### 1.3 Directory Structure

```
CV_Cindrella/
├── src/
│   ├── size_embedder.py          # Size conditioning module
│   ├── size_aware_tryon_net.py   # UNet wrapper with size injection
│   ├── unet_hacked_tryon.py      # Modified UNet for try-on
│   └── unet_block_hacked_tryon.py
├── data_size_aware/              # Preprocessed dataset
│   ├── train/
│   │   ├── image/                # Person images
│   │   ├── cloth/                # Garment images
│   │   ├── cloth-mask/           # Garment segmentation
│   │   ├── image-densepose/      # Body pose maps
│   │   └── agnostic-mask/        # Inpainting masks
│   ├── size_annotations.json
│   ├── train_pairs.txt
│   └── vitonhd_train_tagged.json
├── ckpt/                         # Pretrained checkpoints
│   ├── ip_adapter/
│   └── image_encoder/
├── train_xl_size_aware.py        # Main training script
├── train_size_aware_combinatorial.sh  # Training launcher
├── inference_size_aware.py       # Inference script
├── preprocess_size_dataset.py    # Dataset preprocessing
├── test_training_pipeline.py     # Validation tests
└── environment.yaml              # Conda environment
```

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SIZE-AWARE IDM-VTON                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Person Image │    │ Garment Image│    │ Size Indices             │   │
│  │  (1024×768)  │    │  (1024×768)  │    │ body_size, cloth_size    │   │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘   │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   VAE        │    │ IP-Adapter   │    │     SizeEmbedder         │   │
│  │  Encoder     │    │ (CLIP ViT-H) │    │  (Learnable Embeddings)  │   │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘   │
│         │                   │                        │                   │
│         │            ┌──────┴───────┐                │                   │
│         │            │ Image Proj   │                │                   │
│         │            │    Model     │                │                   │
│         │            └──────┬───────┘                │                   │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    SizeAwareTryonNet                            │    │
│  │  ┌─────────────────────────────────────────────────────────┐   │    │
│  │  │                  Modified UNet                           │   │    │
│  │  │  - Garment features injected via cross-attention        │   │    │
│  │  │  - Size embedding injected via added_cond_kwargs        │   │    │
│  │  │  - Text conditioning from CLIP text encoders            │   │    │
│  │  └─────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│                          ┌──────────────┐                               │
│                          │ VAE Decoder  │                               │
│                          └──────┬───────┘                               │
│                                 │                                        │
│                                 ▼                                        │
│                          ┌──────────────┐                               │
│                          │ Output Image │                               │
│                          │  (1024×768)  │                               │
│                          └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Descriptions

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| VAE Encoder | Compress images to latent space | RGB image (1024×768) | Latent (128×96×4) |
| IP-Adapter | Extract garment visual features | Garment image | Feature embeddings |
| SizeEmbedder | Encode size relationship | Size indices | 1280-dim vector |
| UNet | Diffusion denoising | Noisy latent + conditions | Denoised latent |
| VAE Decoder | Reconstruct image from latent | Latent (128×96×4) | RGB image (1024×768) |

### 2.3 Size Injection Methods

The system supports three methods for injecting size conditioning:

#### Method 1: added_cond (Recommended)
```python
# Size embedding added to SDXL's additional conditioning
added_cond_kwargs['size_emb'] = size_embedding  # (B, 1280)
```

#### Method 2: cross_attention
```python
# Size tokens concatenated to cross-attention sequence
encoder_hidden_states = torch.cat([text_embeds, size_tokens], dim=1)
```

#### Method 3: timestep_add
```python
# Size embedding added to timestep embedding
time_emb = time_emb + size_time_emb
```

---

## 3. Dataset Specification

### 3.1 Size Encoding

#### Body Size Classes
| Size | Index | Description |
|------|-------|-------------|
| Small (S) | 0 | Slim/petite body type |
| Medium (M) | 1 | Average body type |
| Large (L) | 2 | Larger body type |

#### Cloth Size Classes
| Size | Index | Description |
|------|-------|-------------|
| Small (S) | 0 | Fitted cut |
| Medium (M) | 1 | Regular cut |
| Large (L) | 2 | Relaxed cut |
| Extra Large (XL) | 3 | Oversized cut |

#### Relative Fit Calculation
```python
relative_fit = cloth_size_index - body_size_index
# Range: -2 (very tight) to +3 (very loose)
# Examples:
#   L body + S cloth = 2 - 0 = +2 (very tight)
#   S body + L cloth = 0 - 2 = -2 (very loose)
#   M body + M cloth = 1 - 1 = 0  (fitted)
```

### 3.2 Full Combinatorial Matrix

```
                    Cloth Size
                 S       M       L
            ┌───────┬───────┬───────┐
Body    S   │ 0,0   │ 0,1   │ 0,2   │  Fitted → Loose
Size    M   │ 1,0   │ 1,1   │ 1,2   │  Tight → Loose
        L   │ 2,0   │ 2,1   │ 2,2   │  Very Tight → Fitted
            └───────┴───────┴───────┘
              Tight   Fitted  Loose
```

### 3.3 Required Files

#### 3.3.1 Image Files

| Directory | Format | Resolution | Description |
|-----------|--------|------------|-------------|
| `train/image/` | JPG/PNG | 768×1024 | Full person images |
| `train/cloth/` | JPG/PNG | 768×1024 | Flat garment images |
| `train/cloth-mask/` | PNG | 768×1024 | Binary garment mask |
| `train/image-densepose/` | JPG/PNG | 768×1024 | DensePose visualization |
| `train/agnostic-mask/` | PNG | 768×1024 | Clothing region mask |

#### 3.3.2 size_annotations.json

```json
{
  "annotations": [
    {
      "id": "S_M",
      "image": "S_M.jpg",
      "cloth": "tshirt.jpg",
      "body_size": "small",
      "body_size_code": "S",
      "cloth_size": "medium",
      "cloth_size_code": "M",
      "original_file": "small-medium.png"
    },
    {
      "id": "L_S",
      "image": "L_S.jpg",
      "cloth": "tshirt.jpg",
      "body_size": "large",
      "body_size_code": "L",
      "cloth_size": "small",
      "cloth_size_code": "S",
      "original_file": "large-small.png"
    }
  ]
}
```

#### 3.3.3 train_pairs.txt

```
S_S.jpg tshirt.jpg
S_M.jpg tshirt.jpg
S_L.jpg tshirt.jpg
M_S.jpg tshirt.jpg
M_M.jpg tshirt.jpg
M_L.jpg tshirt.jpg
L_S.jpg tshirt.jpg
L_M.jpg tshirt.jpg
L_L.jpg tshirt.jpg
```

#### 3.3.4 vitonhd_train_tagged.json (Optional)

```json
[
  {
    "file_name": "S_M.jpg",
    "caption": "a woman wearing a casual t-shirt",
    "body_size": 0,
    "cloth_size": 1
  }
]
```

### 3.4 Naming Convention

**Person Images**: `{body_size}_{cloth_size}.jpg`
- `S_S.jpg` = Small body, Small clothes
- `M_L.jpg` = Medium body, Large clothes
- `L_S.jpg` = Large body, Small clothes

**Mask Files**: `{body_size}_{cloth_size}_mask.png`

---

## 4. Preprocessing Pipeline

### 4.1 Preprocessing Script

**File**: `preprocess_size_dataset.py`

```python
"""
Dataset Preprocessing for Size-Aware Training

Converts raw images with {body}-{cloth}.png naming to VITON-HD format.

Usage:
    python preprocess_size_dataset.py \
        --input_dir /path/to/raw/images \
        --output_dir /path/to/data_size_aware
"""
```

### 4.2 Preprocessing Steps

#### Step 1: Parse File Names
```python
def parse_size_from_filename(filename):
    """
    Parse body and cloth size from filename.

    Examples:
        'small-medium.png' -> ('small', 'S', 'medium', 'M')
        'large-large.png'  -> ('large', 'L', 'large', 'L')
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('-')
    body_size = parts[0]   # 'small', 'medium', 'large'
    cloth_size = parts[1]  # 'small', 'medium', 'large'

    size_map = {'small': 'S', 'medium': 'M', 'large': 'L'}
    return body_size, size_map[body_size], cloth_size, size_map[cloth_size]
```

#### Step 2: Resize and Convert Images
```python
def process_image(input_path, output_path, target_size=(768, 1024)):
    """Resize image to VITON-HD dimensions."""
    img = Image.open(input_path)
    img = img.convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img.save(output_path, 'JPEG', quality=95)
```

#### Step 3: Generate Placeholder Masks
```python
def generate_agnostic_mask(image_path, output_path):
    """
    Generate clothing region mask.

    For production: Use segmentation model (e.g., Graphonomy)
    For testing: Generate placeholder mask covering upper body
    """
    img = Image.open(image_path)
    width, height = img.size

    # Placeholder: mask upper body region
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Upper body bounding box (approximate)
    left = int(width * 0.2)
    right = int(width * 0.8)
    top = int(height * 0.15)
    bottom = int(height * 0.6)

    draw.rectangle([left, top, right, bottom], fill=255)
    mask.save(output_path)
```

#### Step 4: Generate DensePose (Placeholder)
```python
def generate_densepose_placeholder(image_path, output_path):
    """
    Generate DensePose visualization.

    For production: Use detectron2 DensePose model
    For testing: Generate colored placeholder
    """
    img = Image.open(image_path)
    # Create colorized body part visualization
    # ... (see full implementation in preprocess_size_dataset.py)
```

#### Step 5: Create Annotation Files
```python
def create_annotations(processed_files, output_dir):
    """Create size_annotations.json and train_pairs.txt"""

    annotations = {"annotations": []}
    pairs = []

    for f in processed_files:
        body_size, body_code, cloth_size, cloth_code = parse_size_from_filename(f)

        annotation = {
            "id": f"{body_code}_{cloth_code}",
            "image": f"{body_code}_{cloth_code}.jpg",
            "cloth": "tshirt.jpg",
            "body_size": body_size,
            "body_size_code": body_code,
            "cloth_size": cloth_size,
            "cloth_size_code": cloth_code,
            "original_file": f
        }
        annotations["annotations"].append(annotation)
        pairs.append(f"{body_code}_{cloth_code}.jpg tshirt.jpg")

    # Save files
    with open(os.path.join(output_dir, "size_annotations.json"), 'w') as f:
        json.dump(annotations, f, indent=2)

    with open(os.path.join(output_dir, "train_pairs.txt"), 'w') as f:
        f.write('\n'.join(pairs))
```

### 4.3 Running Preprocessing

```bash
# Full preprocessing
python preprocess_size_dataset.py \
    --input_dir /path/to/raw/dataset \
    --output_dir /path/to/data_size_aware \
    --garment_file tshirt.png

# Verify output
ls -la data_size_aware/train/
# image/  cloth/  cloth-mask/  image-densepose/  agnostic-mask/

cat data_size_aware/train_pairs.txt
# S_S.jpg tshirt.jpg
# S_M.jpg tshirt.jpg
# ...
```

---

## 5. Core Modules

### 5.1 SizeEmbedder

**File**: `src/size_embedder.py`

#### 5.1.1 Class Definition

```python
class SizeEmbedder(nn.Module):
    """
    Embeds body size and clothing size information.

    Full Combinatorial Training approach:
    - Learns separate embeddings for body size (3 classes: S/M/L)
    - Learns separate embeddings for cloth size (4 classes: S/M/L/XL)
    - Optionally incorporates relative fit as continuous signal

    Output embedding is compatible with SDXL's conditioning dimension (1280).
    """

    def __init__(
        self,
        num_body_sizes: int = 3,      # S, M, L
        num_cloth_sizes: int = 4,     # S, M, L, XL
        embedding_dim: int = 1280,    # SDXL compatible
        use_relative_fit: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_body_sizes = num_body_sizes
        self.num_cloth_sizes = num_cloth_sizes
        self.embedding_dim = embedding_dim
        self.use_relative_fit = use_relative_fit

        # Learnable embeddings for categorical sizes
        self.body_size_embed = nn.Embedding(num_body_sizes, embedding_dim // 2)
        self.cloth_size_embed = nn.Embedding(num_cloth_sizes, embedding_dim // 2)

        # Projection network
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        if use_relative_fit:
            # Continuous embedding for relative fit
            self.fit_embed = nn.Sequential(
                nn.Linear(1, embedding_dim // 4),
                nn.SiLU(),
                nn.Linear(embedding_dim // 4, embedding_dim)
            )

            # Combine categorical + continuous
            self.combine = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
```

#### 5.1.2 Forward Pass

```python
def forward(
    self,
    body_size: torch.Tensor,      # (B,) indices 0-2
    cloth_size: torch.Tensor,     # (B,) indices 0-3
    relative_fit: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        body_size: (B,) tensor of body size indices (0=S, 1=M, 2=L)
        cloth_size: (B,) tensor of cloth size indices (0=S, 1=M, 2=L, 3=XL)
        relative_fit: (B,) optional tensor of relative fit values

    Returns:
        size_embedding: (B, embedding_dim) tensor
    """
    # Get categorical embeddings
    body_emb = self.body_size_embed(body_size)    # (B, dim/2)
    cloth_emb = self.cloth_size_embed(cloth_size)  # (B, dim/2)

    # Concatenate and project
    combined = torch.cat([body_emb, cloth_emb], dim=-1)  # (B, dim)
    size_emb = self.project(combined)

    # Add relative fit encoding if enabled
    if self.use_relative_fit:
        if relative_fit is None:
            relative_fit = cloth_size.float() - body_size.float()

        fit_emb = self.fit_embed(relative_fit.unsqueeze(-1).float())
        size_emb = self.combine(torch.cat([size_emb, fit_emb], dim=-1))

    return size_emb
```

#### 5.1.3 Null Embedding (for Classifier-Free Guidance)

```python
def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Get null embedding for classifier-free guidance dropout.

    Returns:
        Zero embedding of shape (batch_size, embedding_dim)
    """
    return torch.zeros(batch_size, self.embedding_dim, device=device)
```

#### 5.1.4 Alternative Embedders

**SinusoidalSizeEmbedder**: For continuous size interpolation
```python
class SinusoidalSizeEmbedder(nn.Module):
    """
    Sinusoidal positional encoding for sizes.
    Allows generating intermediate sizes (e.g., between M and L).
    """
    def forward(self, body_size, cloth_size):
        body_norm = body_size.float() / 2.0   # 0,1,2 -> 0, 0.5, 1
        cloth_norm = cloth_size.float() / 3.0  # 0,1,2,3 -> 0, 0.33, 0.67, 1
        # ... sinusoidal encoding
```

**RelativeFitEmbedder**: Focus on fit relationship only
```python
class RelativeFitEmbedder(nn.Module):
    """
    Embeds only the relative fit (tight/fitted/loose).
    Simpler alternative when absolute sizes don't matter.
    """
    def forward(self, body_size, cloth_size):
        relative_fit = cloth_size - body_size
        fit_class = torch.clamp(relative_fit + 2, 0, 4).long()
        return self.fit_embed(fit_class)
```

### 5.2 SizeAwareTryonNet

**File**: `src/size_aware_tryon_net.py`

#### 5.2.1 Class Definition

```python
class SizeAwareTryonNet(nn.Module):
    """
    Modified TryonNet with size conditioning injection.
    Wraps the base UNet and injects size embeddings during forward pass.
    """

    def __init__(
        self,
        base_unet: nn.Module,
        size_embedder: nn.Module,
        injection_method: str = 'added_cond',  # 'added_cond', 'cross_attention', 'timestep_add'
        cross_attention_dim: int = 2048,
    ):
        super().__init__()

        self.unet = base_unet
        self.size_embedder = size_embedder
        self.injection_method = injection_method
        self.cross_attention_dim = cross_attention_dim

        # For cross-attention injection
        if injection_method == 'cross_attention':
            self.size_to_sequence = nn.Linear(
                size_embedder.embedding_dim,
                cross_attention_dim
            )

        # For timestep addition
        if injection_method == 'timestep_add':
            time_embed_dim = getattr(base_unet.config, 'time_embed_dim', 1280)
            self.size_to_timestep = nn.Linear(
                size_embedder.embedding_dim,
                time_embed_dim
            )
```

#### 5.2.2 Forward Pass

```python
def forward(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    body_size: Optional[torch.Tensor] = None,
    cloth_size: Optional[torch.Tensor] = None,
    relative_fit: Optional[torch.Tensor] = None,
    size_embedding: Optional[torch.Tensor] = None,
    added_cond_kwargs: Optional[Dict[str, Any]] = None,
    garment_features: Optional[List[torch.Tensor]] = None,
    **kwargs
):
    """
    Forward pass with size conditioning.

    Args:
        sample: Noisy latent input (B, C, H, W)
        timestep: Diffusion timestep
        encoder_hidden_states: Text embeddings (B, seq_len, dim)
        body_size: (B,) body size indices
        cloth_size: (B,) cloth size indices
        relative_fit: (B,) optional relative fit values
        size_embedding: Pre-computed size embedding (overrides body/cloth_size)
        added_cond_kwargs: SDXL additional conditioning
        garment_features: Garment features from IP-Adapter

    Returns:
        UNet output with size conditioning applied
    """
    # Get size embedding
    if size_embedding is None and body_size is not None and cloth_size is not None:
        size_emb = self.size_embedder(body_size, cloth_size, relative_fit)
    elif size_embedding is not None:
        size_emb = size_embedding
    else:
        # No size conditioning - use null embedding
        batch_size = sample.shape[0]
        size_emb = self.size_embedder.get_null_embedding(batch_size, sample.device)

    # Initialize added_cond_kwargs if not provided
    if added_cond_kwargs is None:
        added_cond_kwargs = {}

    # Inject size conditioning based on method
    if self.injection_method == 'added_cond':
        added_cond_kwargs = dict(added_cond_kwargs)
        added_cond_kwargs['size_emb'] = size_emb

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            garment_features=garment_features,
            **kwargs
        )

    elif self.injection_method == 'cross_attention':
        size_tokens = self.size_to_sequence(size_emb).unsqueeze(1)
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, size_tokens],
            dim=1
        )

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            garment_features=garment_features,
            **kwargs
        )

    elif self.injection_method == 'timestep_add':
        size_time = self.size_to_timestep(size_emb)
        added_cond_kwargs = dict(added_cond_kwargs)
        added_cond_kwargs['size_time_emb'] = size_time

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            garment_features=garment_features,
            **kwargs
        )
```

### 5.3 LoRA Implementation

**File**: `train_xl_size_aware.py` (lines 59-132)

#### 5.3.1 LoRALinear Class

```python
class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.

    Mathematical formulation:
        output = W*x + (B*A)*x * (alpha/rank)

    Where:
        W: Original frozen weights
        A: Low-rank matrix (rank × in_features)
        B: Low-rank matrix (out_features × rank)
        alpha: Scaling factor
        rank: Low-rank dimension
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros
        # This ensures LoRA contribution starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original layer
        original_layer.requires_grad_(False)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward (frozen)
        result = self.original_layer(x)

        # Add LoRA contribution (trainable)
        lora_result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_result
```

#### 5.3.2 Adding LoRA to Model

```python
def add_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 1.0
) -> List[nn.Parameter]:
    """
    Add LoRA adapters to all linear layers in a model.

    Args:
        model: Model to add LoRA to
        rank: LoRA rank (typically 8-32)
        alpha: LoRA scaling factor (typically = rank)

    Returns:
        List of LoRA parameters for optimizer
    """
    lora_params = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRA layer
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)

            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    return lora_params
```

### 5.4 Dataset Class

**File**: `train_xl_size_aware.py` (lines 135-280)

```python
class SizeAwareCombinatorialDataset(data.Dataset):
    """
    Dataset for Full Combinatorial Training.

    Supports explicit body_size and cloth_size labels for each sample.
    """

    def __init__(
        self,
        dataroot_path: str,
        phase: str = "train",
        order: str = "paired",
        size: Tuple[int, int] = (512, 384),
        size_annotation_file: Optional[str] = None,
    ):
        self.dataroot = dataroot_path
        self.phase = phase
        self.size = size

        # Size mapping
        self.body_size_map = {'S': 0, 'M': 1, 'L': 2}
        self.cloth_size_map = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}

        # Load pairs
        pairs_file = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        with open(pairs_file, 'r') as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

        # Load size annotations
        self.size_annotations = {}
        ann_file = size_annotation_file or os.path.join(dataroot_path, "size_annotations.json")
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                data = json.load(f)
                for ann in data['annotations']:
                    self.size_annotations[ann['image']] = ann

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, cloth_name = self.pairs[idx]

        # Load images
        img_path = os.path.join(self.dataroot, self.phase, "image", img_name)
        cloth_path = os.path.join(self.dataroot, self.phase, "cloth", cloth_name)
        mask_path = os.path.join(self.dataroot, self.phase, "agnostic-mask",
                                  img_name.replace('.jpg', '_mask.png'))
        densepose_path = os.path.join(self.dataroot, self.phase, "image-densepose", img_name)

        image = Image.open(img_path).convert('RGB')
        cloth = Image.open(cloth_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        densepose = Image.open(densepose_path).convert('RGB')

        # Get size labels
        if img_name in self.size_annotations:
            ann = self.size_annotations[img_name]
            body_size = self.body_size_map[ann['body_size_code']]
            cloth_size = self.cloth_size_map[ann['cloth_size_code']]
        else:
            # Estimate from filename if no annotation
            body_size, cloth_size = self._estimate_size_from_filename(img_name)

        # Apply transforms
        image = self.transform(image)
        cloth = self.transform(cloth)
        mask = transforms.ToTensor()(mask)
        densepose = self.transform(densepose)

        return {
            'image': image,
            'cloth': cloth,
            'mask': mask,
            'densepose': densepose,
            'body_size': body_size,
            'cloth_size': cloth_size,
            'image_name': img_name,
        }

    def _estimate_size_from_filename(self, filename):
        """Fallback: estimate size from filename pattern."""
        name = filename.replace('.jpg', '').replace('.png', '')
        parts = name.split('_')
        if len(parts) >= 2:
            body_code = parts[0]  # S, M, L
            cloth_code = parts[1]  # S, M, L
            return self.body_size_map.get(body_code, 1), self.cloth_size_map.get(cloth_code, 1)
        return 1, 1  # Default to medium
```

---

## 6. Training Pipeline

### 6.1 Training Script Overview

**File**: `train_xl_size_aware.py`

```python
"""
Size-Aware IDM-VTON Training Script

Features:
- Full Combinatorial Training with size conditioning
- Optional IP-Adapter fine-tuning with LoRA
- Gradient checkpointing and mixed precision
- Multi-GPU support via Accelerate
"""
```

### 6.2 Training Loop

```python
def train():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Load models
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    # ... load other components

    # Create size embedder
    size_embedder = SizeEmbedder(
        num_body_sizes=3,
        num_cloth_sizes=4,
        embedding_dim=1280,
        use_relative_fit=True,
    )

    # Wrap UNet with size-aware net
    size_aware_unet = SizeAwareTryonNet(
        base_unet=unet,
        size_embedder=size_embedder,
        injection_method=args.size_injection_method,
    )

    # Setup optimizer with separate learning rates
    param_groups = [
        {"params": list(size_embedder.parameters()), "lr": args.size_embedder_lr},
    ]

    if not args.freeze_unet:
        unet_params = [p for p in unet.parameters() if p.requires_grad]
        param_groups.append({"params": unet_params, "lr": args.learning_rate})

    if args.train_ip_adapter:
        if args.use_ip_adapter_lora:
            ip_adapter_lora_params = add_lora_to_model(
                image_proj_model,
                rank=args.ip_adapter_lora_rank,
            )
            param_groups.append({"params": ip_adapter_lora_params, "lr": args.ip_adapter_lr})

    optimizer = torch.optim.AdamW(param_groups)

    # Training loop
    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(size_aware_unet):
                # Get inputs
                latents = vae.encode(batch['image']).latent_dist.sample()
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (batch_size,))
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Get size conditioning (with dropout for CFG)
                body_size = batch['body_size']
                cloth_size = batch['cloth_size']

                if random.random() < args.size_dropout_prob:
                    # Null conditioning for classifier-free guidance
                    size_emb = size_embedder.get_null_embedding(batch_size, device)
                else:
                    size_emb = size_embedder(body_size, cloth_size)

                # Forward pass
                model_pred = size_aware_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    size_embedding=size_emb,
                    added_cond_kwargs=added_cond_kwargs,
                    garment_features=garment_features,
                ).sample

                # Compute loss
                loss = F.mse_loss(model_pred, noise)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
```

### 6.3 Training Configurations

#### Configuration 1: Size Embedder Only (Fast)

```bash
python train_xl_size_aware.py \
    --data_dir ./data_size_aware \
    --output_dir ./results/size_only \
    --freeze_unet \
    --size_embedder_lr 1e-4 \
    --num_train_epochs 50
```

**Use case**: Quick experimentation, limited GPU memory

#### Configuration 2: Size Embedder + UNet Decoder

```bash
python train_xl_size_aware.py \
    --data_dir ./data_size_aware \
    --output_dir ./results/decoder_finetune \
    --train_unet_decoder_only \
    --learning_rate 1e-5 \
    --size_embedder_lr 1e-4 \
    --num_train_epochs 100
```

**Use case**: Better quality with reasonable training time

#### Configuration 3: Full Training + IP-Adapter LoRA

```bash
python train_xl_size_aware.py \
    --data_dir ./data_size_aware \
    --output_dir ./results/full_training \
    --learning_rate 1e-5 \
    --size_embedder_lr 1e-4 \
    --train_ip_adapter \
    --use_ip_adapter_lora \
    --ip_adapter_lora_rank 16 \
    --ip_adapter_lr 1e-6 \
    --num_train_epochs 100
```

**Use case**: Highest quality, production model

### 6.4 Training Shell Script

**File**: `train_size_aware_combinatorial.sh`

```bash
#!/bin/bash

# ============================================================
# Size-Aware IDM-VTON Training Script
# ============================================================

# Configuration
export CUDA_VISIBLE_DEVICES=0

# Paths
DATA_DIR="/path/to/VITON-HD"
OUTPUT_DIR="./results/size_aware_combinatorial"

# Training hyperparameters
BATCH_SIZE=4
EPOCHS=100
LEARNING_RATE=1e-5
SIZE_EMBEDDER_LR=1e-4
SIZE_DROPOUT=0.1

# IP-Adapter options
TRAIN_IP_ADAPTER=false
USE_IP_ADAPTER_LORA=true
IP_ADAPTER_LORA_RANK=16
IP_ADAPTER_LR=1e-6

# Model paths
PRETRAINED_MODEL="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
IP_ADAPTER_PATH="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin"
IMAGE_ENCODER_PATH="ckpt/image_encoder"

# Build command
CMD="python train_xl_size_aware.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --pretrained_ip_adapter_path $IP_ADAPTER_PATH \
    --image_encoder_path $IMAGE_ENCODER_PATH \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --size_embedder_lr $SIZE_EMBEDDER_LR \
    --size_dropout_prob $SIZE_DROPOUT \
    --size_injection_method added_cond \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --seed 42"

# Add IP-Adapter options if enabled
if [ "$TRAIN_IP_ADAPTER" = true ]; then
    CMD="$CMD --train_ip_adapter --ip_adapter_lr $IP_ADAPTER_LR"
    if [ "$USE_IP_ADAPTER_LORA" = true ]; then
        CMD="$CMD --use_ip_adapter_lora --ip_adapter_lora_rank $IP_ADAPTER_LORA_RANK"
    fi
fi

# Run training
echo "Running: $CMD"
eval $CMD 2>&1 | tee "$OUTPUT_DIR/training.log"
```

### 6.5 Checkpointing

```python
# Save checkpoint
def save_checkpoint(epoch, output_dir):
    checkpoint = {
        'epoch': epoch,
        'size_embedder': size_embedder.state_dict(),
        'unet': unet.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if args.train_ip_adapter and args.use_ip_adapter_lora:
        # Save LoRA weights separately
        lora_state = {}
        for name, module in image_proj_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[f"{name}.lora_A"] = module.lora_A
                lora_state[f"{name}.lora_B"] = module.lora_B
        checkpoint['ip_adapter_lora'] = lora_state

    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint-{epoch}.pt"))

# Load checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    size_embedder.load_state_dict(checkpoint['size_embedder'])
    unet.load_state_dict(checkpoint['unet'])
    # ...
```

---

## 7. Inference Pipeline

### 7.1 Inference Script

**File**: `inference_size_aware.py`

```python
"""
Size-Aware Virtual Try-On Inference

Usage:
    python inference_size_aware.py \
        --person_image person.jpg \
        --garment_image garment.jpg \
        --body_size 1 \
        --cloth_size 2 \
        --output_path result.jpg
"""

import torch
from PIL import Image
from src.size_embedder import SizeEmbedder
from src.size_aware_tryon_net import SizeAwareTryonNet

class SizeAwareTryOnPipeline:
    def __init__(
        self,
        unet_path: str,
        size_embedder_path: str,
        vae_path: str,
        device: str = "cuda"
    ):
        self.device = device

        # Load models
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)

        # Load size embedder
        self.size_embedder = SizeEmbedder(
            num_body_sizes=3,
            num_cloth_sizes=4,
            embedding_dim=1280
        )
        self.size_embedder.load_state_dict(torch.load(size_embedder_path))

        # Create size-aware wrapper
        self.size_aware_unet = SizeAwareTryonNet(
            base_unet=self.unet,
            size_embedder=self.size_embedder,
            injection_method='added_cond'
        )

        # Move to device
        self.vae.to(device)
        self.size_aware_unet.to(device)

        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(unet_path, subfolder="scheduler")

    @torch.no_grad()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        body_size: int,
        cloth_size: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        size_guidance_scale: float = 1.5,
    ) -> Image.Image:
        """
        Generate try-on result with size conditioning.

        Args:
            person_image: Input person image
            garment_image: Garment to try on
            body_size: Body size index (0=S, 1=M, 2=L)
            cloth_size: Cloth size index (0=S, 1=M, 2=L, 3=XL)
            num_inference_steps: Denoising steps
            guidance_scale: Text guidance scale
            size_guidance_scale: Size conditioning guidance scale

        Returns:
            Generated try-on image
        """
        # Preprocess images
        person_tensor = self.preprocess(person_image)
        garment_tensor = self.preprocess(garment_image)

        # Create size tensors
        body_size_t = torch.tensor([body_size], device=self.device)
        cloth_size_t = torch.tensor([cloth_size], device=self.device)

        # Get size embedding
        size_emb = self.size_embedder(body_size_t, cloth_size_t)
        null_size_emb = self.size_embedder.get_null_embedding(1, self.device)

        # Encode to latent space
        latents = torch.randn((1, 4, 128, 96), device=self.device)

        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # Classifier-free guidance on size
            # Predict with size conditioning
            noise_pred_cond = self.size_aware_unet(
                latents,
                t,
                encoder_hidden_states=text_embeds,
                size_embedding=size_emb,
                garment_features=garment_features,
            ).sample

            # Predict without size conditioning
            noise_pred_uncond = self.size_aware_unet(
                latents,
                t,
                encoder_hidden_states=text_embeds,
                size_embedding=null_size_emb,
                garment_features=garment_features,
            ).sample

            # Apply guidance
            noise_pred = noise_pred_uncond + size_guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        image = self.vae.decode(latents).sample
        image = self.postprocess(image)

        return image
```

### 7.2 Inference Examples

```python
# Load pipeline
pipeline = SizeAwareTryOnPipeline(
    unet_path="./results/size_aware/unet",
    size_embedder_path="./results/size_aware/size_embedder.pt",
    vae_path="./results/size_aware/vae",
)

# Generate results for different size combinations
person = Image.open("person.jpg")
garment = Image.open("tshirt.jpg")

# Fitted look (same sizes)
result_fitted = pipeline(
    person_image=person,
    garment_image=garment,
    body_size=1,  # Medium
    cloth_size=1, # Medium
)
result_fitted.save("result_fitted.jpg")

# Loose/oversized look
result_loose = pipeline(
    person_image=person,
    garment_image=garment,
    body_size=0,  # Small
    cloth_size=2, # Large
)
result_loose.save("result_loose.jpg")

# Tight fit
result_tight = pipeline(
    person_image=person,
    garment_image=garment,
    body_size=2,  # Large
    cloth_size=0, # Small
)
result_tight.save("result_tight.jpg")
```

---

## 8. Configuration Reference

### 8.1 All Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | Required | Path to VITON-HD dataset |
| `--output_dir` | str | "output" | Output directory for checkpoints |
| `--pretrained_model_name_or_path` | str | Required | Path to pretrained SDXL model |
| `--pretrained_ip_adapter_path` | str | Required | Path to IP-Adapter weights |
| `--image_encoder_path` | str | Required | Path to CLIP image encoder |

#### Size Conditioning
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size_injection_method` | str | "added_cond" | How to inject size: added_cond, cross_attention, timestep_add |
| `--size_embedder_lr` | float | 1e-4 | Learning rate for size embedder |
| `--size_dropout_prob` | float | 0.1 | Dropout probability for CFG |
| `--num_body_sizes` | int | 3 | Number of body size classes |
| `--num_cloth_sizes` | int | 4 | Number of cloth size classes |

#### UNet Training
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--freeze_unet` | flag | False | Freeze UNet, train only size embedder |
| `--train_unet_decoder_only` | flag | False | Train only UNet decoder |
| `--learning_rate` | float | 1e-5 | UNet learning rate |

#### IP-Adapter
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_ip_adapter` | flag | False | Fine-tune IP-Adapter |
| `--use_ip_adapter_lora` | flag | False | Use LoRA for IP-Adapter |
| `--ip_adapter_lora_rank` | int | 16 | LoRA rank |
| `--ip_adapter_lr` | float | 1e-6 | IP-Adapter learning rate |

#### Training
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_batch_size` | int | 4 | Batch size per GPU |
| `--num_train_epochs` | int | 100 | Total training epochs |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation |
| `--gradient_checkpointing` | flag | False | Enable gradient checkpointing |
| `--mixed_precision` | str | "fp16" | Mixed precision: no, fp16, bf16 |
| `--use_8bit_adam` | flag | False | Use 8-bit Adam optimizer |

#### Optimizer
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--adam_beta1` | float | 0.9 | Adam beta1 |
| `--adam_beta2` | float | 0.999 | Adam beta2 |
| `--adam_weight_decay` | float | 0.01 | Weight decay |
| `--adam_epsilon` | float | 1e-8 | Adam epsilon |

#### Logging
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--logging_steps` | int | 500 | Log every N steps |
| `--checkpointing_epochs` | int | 10 | Save checkpoint every N epochs |
| `--seed` | int | 42 | Random seed |

### 8.2 Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Accelerate distributed training
export ACCELERATE_MIXED_PRECISION=fp16
```

### 8.3 Hardware Requirements

| Configuration | GPU VRAM | System RAM | Training Time* |
|--------------|----------|------------|----------------|
| Size embedder only | 16 GB | 32 GB | ~2 hours |
| + UNet decoder | 24 GB | 48 GB | ~8 hours |
| Full + IP-Adapter LoRA | 32 GB | 64 GB | ~24 hours |

*Estimated for 100 epochs on 1000 samples with batch size 4

---

## 9. API Reference

### 9.1 SizeEmbedder API

```python
class SizeEmbedder(nn.Module):
    """
    Parameters:
        num_body_sizes (int): Number of body size classes. Default: 3
        num_cloth_sizes (int): Number of cloth size classes. Default: 4
        embedding_dim (int): Output embedding dimension. Default: 1280
        use_relative_fit (bool): Include relative fit encoding. Default: True
        dropout (float): Dropout probability. Default: 0.1

    Methods:
        forward(body_size, cloth_size, relative_fit=None) -> Tensor
            Compute size embedding.

        get_null_embedding(batch_size, device) -> Tensor
            Get null embedding for classifier-free guidance.

    Example:
        >>> embedder = SizeEmbedder()
        >>> body_size = torch.tensor([0, 1, 2])  # S, M, L
        >>> cloth_size = torch.tensor([1, 1, 1])  # All M
        >>> embedding = embedder(body_size, cloth_size)
        >>> print(embedding.shape)  # torch.Size([3, 1280])
    """
```

### 9.2 SizeAwareTryonNet API

```python
class SizeAwareTryonNet(nn.Module):
    """
    Parameters:
        base_unet (nn.Module): Base IDM-VTON UNet model
        size_embedder (nn.Module): Size embedding module
        injection_method (str): 'added_cond', 'cross_attention', or 'timestep_add'
        cross_attention_dim (int): Cross-attention dimension. Default: 2048

    Methods:
        forward(sample, timestep, encoder_hidden_states,
                body_size=None, cloth_size=None, size_embedding=None,
                added_cond_kwargs=None, garment_features=None, **kwargs)
            Forward pass with size conditioning.

        enable_gradient_checkpointing()
            Enable gradient checkpointing for memory efficiency.

        enable_xformers_memory_efficient_attention()
            Enable xformers attention.

    Properties:
        config: Returns base UNet config
    """
```

### 9.3 LoRALinear API

```python
class LoRALinear(nn.Module):
    """
    Parameters:
        original_layer (nn.Linear): Original linear layer to wrap
        rank (int): LoRA rank. Default: 16
        alpha (float): LoRA scaling factor. Default: 1.0

    Attributes:
        lora_A (Parameter): Shape (rank, in_features)
        lora_B (Parameter): Shape (out_features, rank)
        scaling (float): alpha / rank

    Methods:
        forward(x) -> Tensor
            Compute output = original(x) + lora_contribution(x)
    """
```

### 9.4 Dataset API

```python
class SizeAwareCombinatorialDataset(torch.utils.data.Dataset):
    """
    Parameters:
        dataroot_path (str): Path to dataset root
        phase (str): 'train' or 'test'. Default: 'train'
        order (str): 'paired' or 'unpaired'. Default: 'paired'
        size (tuple): Image size (H, W). Default: (512, 384)
        size_annotation_file (str): Path to size annotations JSON

    Returns (dict):
        'image': Tensor (3, H, W) - Person image
        'cloth': Tensor (3, H, W) - Garment image
        'mask': Tensor (1, H, W) - Agnostic mask
        'densepose': Tensor (3, H, W) - DensePose map
        'body_size': int - Body size index (0-2)
        'cloth_size': int - Cloth size index (0-3)
        'image_name': str - Image filename
    """
```

---

## 10. Troubleshooting

### 10.1 Common Errors

#### ImportError: cannot import name 'cached_download'
```
Cause: Version mismatch between diffusers and huggingface_hub

Solution:
pip install diffusers==0.25.0 huggingface_hub==0.23.5
```

#### CUDA out of memory
```
Cause: Insufficient GPU memory

Solutions:
1. Enable gradient checkpointing: --gradient_checkpointing
2. Reduce batch size: --train_batch_size 2
3. Use mixed precision: --mixed_precision fp16
4. Freeze UNet: --freeze_unet
```

#### Size embedding shape mismatch
```
Cause: Mismatched embedding dimensions

Solution: Ensure embedding_dim matches UNet's expected dimension (1280 for SDXL)
```

#### Dataset loading errors
```
Cause: Missing files or incorrect paths

Solution:
1. Verify folder structure matches VITON-HD format
2. Check size_annotations.json exists and is valid JSON
3. Verify all images referenced in train_pairs.txt exist
```

### 10.2 Performance Optimization

#### Memory Optimization
```python
# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Use mixed precision
accelerator = Accelerator(mixed_precision="fp16")

# Use 8-bit Adam
pip install bitsandbytes
# Then use --use_8bit_adam
```

#### Speed Optimization
```python
# Enable xformers
pip install xformers
model.enable_xformers_memory_efficient_attention()

# Use multiple workers
DataLoader(..., num_workers=4, pin_memory=True)
```

### 10.3 Validation

Run the test script to verify installation:

```bash
python test_training_pipeline.py

# Expected output:
# ============================================================
# Size-Aware Training Pipeline Tests
# ============================================================
# Testing imports...
#   ✓ size_embedder imports OK
#   ✓ size_aware_tryon_net imports OK
# Testing SizeEmbedder...
#   ✓ SizeEmbedder test passed
# Testing SizeAwareTryonNet...
#   ✓ SizeAwareTryonNet test passed
# Testing Dataset...
#   ✓ Dataset test passed
# Testing LoRA...
#   ✓ LoRA test passed
#
# Total: 5 passed, 0 failed
# 🎉 All tests passed!
```

---

## Appendix A: File Checksums

For verification:

```bash
# Generate checksums
md5sum src/size_embedder.py
md5sum src/size_aware_tryon_net.py
md5sum train_xl_size_aware.py
```

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-08 | Initial release with Full Combinatorial Training |

## Appendix C: References

1. IDM-VTON: Improving Diffusion Models for Virtual Try-on
2. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
3. LoRA: Low-Rank Adaptation of Large Language Models
4. IP-Adapter: Text Compatible Image Prompt Adapter

---

*Documentation for Cinderella Project - Size-Aware Virtual Try-On*
*Generated: December 2025*
