# Size-Aware IDM-VTON: Training Guide and Architecture

## Overview

This guide explains how to add size awareness to IDM-VTON (Improving Diffusion Models for Virtual Try-on) using your dataset of different body sizes (small, medium, large) wearing various clothing sizes. We'll cover two main approaches:

1. **Full Combinatorial Training** - Using your dataset with all body-size × clothing-size combinations
2. **Paired Data Only Approach** - Alternatives that require less data complexity

---

## Understanding IDM-VTON Architecture

IDM-VTON consists of three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        IDM-VTON Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  IP-Adapter  │     │   TryonNet   │     │  GarmentNet  │    │
│  │ (High-level  │────▶│  (Main UNet) │◀────│ (Low-level   │    │
│  │  semantics)  │     │              │     │  features)   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│        │                     │                     │            │
│        │              Cross-Attention        Self-Attention     │
│        └──────────────────────────────────────────┘            │
│                                                                  │
│  Inputs: Person Image + DensePose + Agnostic Mask + Garment     │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **TryonNet**: Main UNet processing person image (noised latents + mask + densepose)
- **IP-Adapter**: Encodes high-level garment semantics → fused via cross-attention
- **GarmentNet**: Encodes low-level garment features → fused via self-attention

---

## Approach 1: Training with Your Combinatorial Dataset

### Dataset Structure

Your dataset has 9 combinations:
```
Body Size × Clothing Size Matrix:
                 Small Clothes | Medium Clothes | Large Clothes
Small Body       ✓ (tight fit) | ✓ (normal fit) | ✓ (loose fit)
Medium Body      ✓ (tight fit) | ✓ (normal fit) | ✓ (loose fit)  
Large Body       ✓ (tight fit) | ✓ (normal fit) | ✓ (loose fit)
```

### Method A: Size Embedding Conditioning

Add size embeddings to the model that encode both body size and clothing size.

#### Step 1: Modify the Dataset Class

```python
# src/dataset_size_aware.py

import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os

class SizeAwareVitonDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load annotations with size info
        with open(os.path.join(data_dir, 'size_annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        
        # Size mappings
        self.body_size_map = {'small': 0, 'medium': 1, 'large': 2}
        self.cloth_size_map = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load images
        image = Image.open(os.path.join(self.data_dir, 'image', item['image']))
        cloth = Image.open(os.path.join(self.data_dir, 'cloth', item['cloth']))
        densepose = Image.open(os.path.join(self.data_dir, 'image-densepose', item['densepose']))
        mask = Image.open(os.path.join(self.data_dir, 'agnostic-mask', item['mask']))
        
        # Get size labels
        body_size = self.body_size_map[item['body_size']]
        cloth_size = self.cloth_size_map[item['cloth_size']]
        
        # Optional: Compute relative fit (how the cloth fits on this body)
        # -1 = tight, 0 = normal, 1 = loose
        relative_fit = cloth_size - body_size  # Simplified; adjust based on your labeling
        
        if self.transform:
            image = self.transform(image)
            cloth = self.transform(cloth)
            densepose = self.transform(densepose)
            mask = self.transform(mask)
        
        return {
            'image': image,
            'cloth': cloth,
            'densepose': densepose,
            'mask': mask,
            'body_size': torch.tensor(body_size),
            'cloth_size': torch.tensor(cloth_size),
            'relative_fit': torch.tensor(relative_fit),
            'caption': item.get('caption', 'a person wearing clothes')
        }
```

#### Step 2: Create Size Embedding Module

```python
# src/size_embedder.py

import torch
import torch.nn as nn
import math

class SizeEmbedder(nn.Module):
    """
    Embeds body size and clothing size information.
    Can be used with different injection methods.
    """
    def __init__(
        self, 
        num_body_sizes=3,      # small, medium, large
        num_cloth_sizes=4,     # S, M, L, XL
        embedding_dim=1280,    # Match SDXL's embedding dimension
        use_relative_fit=True
    ):
        super().__init__()
        
        self.use_relative_fit = use_relative_fit
        
        # Learned embeddings for categorical sizes
        self.body_size_embed = nn.Embedding(num_body_sizes, embedding_dim // 2)
        self.cloth_size_embed = nn.Embedding(num_cloth_sizes, embedding_dim // 2)
        
        # Project to final dimension
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        if use_relative_fit:
            # Continuous embedding for relative fit (-2 to +2 range typically)
            self.fit_embed = nn.Sequential(
                nn.Linear(1, embedding_dim // 4),
                nn.SiLU(),
                nn.Linear(embedding_dim // 4, embedding_dim)
            )
            
            # Combine both
            self.combine = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, body_size, cloth_size, relative_fit=None):
        """
        Args:
            body_size: (B,) tensor of body size indices
            cloth_size: (B,) tensor of clothing size indices
            relative_fit: (B,) optional tensor of relative fit values
        Returns:
            size_embedding: (B, embedding_dim) tensor
        """
        body_emb = self.body_size_embed(body_size)  # (B, dim/2)
        cloth_emb = self.cloth_size_embed(cloth_size)  # (B, dim/2)
        
        combined = torch.cat([body_emb, cloth_emb], dim=-1)  # (B, dim)
        size_emb = self.project(combined)
        
        if self.use_relative_fit and relative_fit is not None:
            fit_emb = self.fit_embed(relative_fit.unsqueeze(-1).float())
            size_emb = self.combine(torch.cat([size_emb, fit_emb], dim=-1))
        
        return size_emb


class SinusoidalSizeEmbedder(nn.Module):
    """
    Alternative: Sinusoidal positional encoding for sizes.
    Better for continuous interpolation between sizes.
    """
    def __init__(self, embedding_dim=1280, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
        # MLP to process sinusoidal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def sinusoidal_embed(self, x, dim):
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device) / half
        )
        args = x[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, body_size, cloth_size):
        """
        Args:
            body_size: (B,) normalized body size (0-1 range)
            cloth_size: (B,) normalized cloth size (0-1 range)
        """
        body_emb = self.sinusoidal_embed(body_size.float(), self.embedding_dim)
        cloth_emb = self.sinusoidal_embed(cloth_size.float(), self.embedding_dim)
        
        combined = torch.cat([body_emb, cloth_emb], dim=-1)
        return self.mlp(combined)
```

#### Step 3: Modify TryonNet to Accept Size Conditioning

```python
# src/tryon_pipeline_size_aware.py

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from .size_embedder import SizeEmbedder

class SizeAwareTryonNet(nn.Module):
    """
    Modified TryonNet with size conditioning injection.
    """
    def __init__(self, base_unet, size_embedder, injection_method='add_to_timestep'):
        super().__init__()
        self.unet = base_unet
        self.size_embedder = size_embedder
        self.injection_method = injection_method
        
        # For cross-attention injection
        if injection_method == 'cross_attention':
            self.size_to_sequence = nn.Linear(1280, 1280)
    
    def forward(
        self, 
        sample, 
        timestep, 
        encoder_hidden_states,
        body_size,
        cloth_size,
        relative_fit=None,
        **kwargs
    ):
        # Get size embedding
        size_emb = self.size_embedder(body_size, cloth_size, relative_fit)
        
        if self.injection_method == 'add_to_timestep':
            # Method 1: Add to timestep embedding (similar to SDXL size conditioning)
            # This is the recommended approach
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs={
                    'size_emb': size_emb,
                    **kwargs.get('added_cond_kwargs', {})
                },
                **{k: v for k, v in kwargs.items() if k != 'added_cond_kwargs'}
            )
            
        elif self.injection_method == 'cross_attention':
            # Method 2: Concatenate to cross-attention sequence
            size_tokens = self.size_to_sequence(size_emb).unsqueeze(1)  # (B, 1, dim)
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, size_tokens], 
                dim=1
            )
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                **kwargs
            )
            
        elif self.injection_method == 'channel_concat':
            # Method 3: Concatenate to input channels
            # Requires modifying UNet's first conv layer
            B, C, H, W = sample.shape
            size_spatial = size_emb[:, :, None, None].expand(B, -1, H, W)
            sample = torch.cat([sample, size_spatial[:, :4]], dim=1)  # Take first 4 channels
            return self.unet(sample, timestep, encoder_hidden_states, **kwargs)
```

#### Step 4: Modified Training Script

```python
# train_xl_size_aware.py

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import argparse

from src.dataset_size_aware import SizeAwareVitonDataset
from src.size_embedder import SizeEmbedder
from src.tryon_pipeline_size_aware import SizeAwareTryonNet

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    
    # Load base models
    # ... (load UNet, VAE, text encoders as in original IDM-VTON)
    
    # Initialize size embedder
    size_embedder = SizeEmbedder(
        num_body_sizes=3,
        num_cloth_sizes=4,
        embedding_dim=1280,
        use_relative_fit=True
    )
    
    # Create size-aware TryonNet
    tryon_net = SizeAwareTryonNet(
        base_unet=unet,
        size_embedder=size_embedder,
        injection_method='add_to_timestep'
    )
    
    # Dataset
    dataset = SizeAwareVitonDataset(
        data_dir=args.data_dir,
        transform=train_transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    # Optimizer - train size embedder and optionally fine-tune UNet
    trainable_params = list(size_embedder.parameters())
    if args.train_unet:
        trainable_params += list(tryon_net.unet.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # Prepare with accelerator
    tryon_net, optimizer, dataloader = accelerator.prepare(
        tryon_net, optimizer, dataloader
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        for batch in tqdm(dataloader):
            with accelerator.accumulate(tryon_net):
                # Get inputs
                images = batch['image']
                cloth = batch['cloth']
                densepose = batch['densepose']
                mask = batch['mask']
                body_size = batch['body_size']
                cloth_size = batch['cloth_size']
                relative_fit = batch['relative_fit']
                
                # Encode images to latent space
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch['caption'])
                
                # Prepare conditioning (densepose + mask + cloth features)
                # ... (as in original IDM-VTON)
                
                # Classifier-free guidance dropout
                if torch.rand(1) < args.size_dropout_prob:
                    # Drop size conditioning
                    body_size = torch.zeros_like(body_size)
                    cloth_size = torch.zeros_like(cloth_size)
                
                # Forward pass with size conditioning
                noise_pred = tryon_net(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    body_size=body_size,
                    cloth_size=cloth_size,
                    relative_fit=relative_fit,
                )
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            accelerator.save_state(f"{args.output_dir}/checkpoint-{epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--size_dropout_prob", type=float, default=0.1)
    parser.add_argument("--train_unet", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    
    train(args)
```

#### Step 5: Inference with Size Control

```python
# inference_size_aware.py

import torch
from diffusers import DDIMScheduler

def inference_with_size(
    pipeline,
    person_image,
    cloth_image,
    target_body_size,    # 'small', 'medium', 'large'
    target_cloth_size,   # 'S', 'M', 'L', 'XL'
    num_inference_steps=30,
    guidance_scale=2.0,
    size_guidance_scale=1.5,  # How strongly to follow size conditioning
):
    body_size_map = {'small': 0, 'medium': 1, 'large': 2}
    cloth_size_map = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
    
    body_size = torch.tensor([body_size_map[target_body_size]])
    cloth_size = torch.tensor([cloth_size_map[target_cloth_size]])
    relative_fit = cloth_size - body_size
    
    # Run inference with classifier-free guidance for both text and size
    # ... (implement denoising loop with dual guidance)
    
    return generated_image
```

---

## Approach 2: Paired Data Only (Less Complex Alternatives)

If you want to avoid collecting the full combinatorial dataset, here are alternatives:

### Option A: DensePose-Based Size Inference

**Concept**: The model learns to infer body size from DensePose and adjusts clothing accordingly.

**Advantages**:
- Only need paired data (person + clothing they're wearing)
- DensePose already encodes body shape information
- No explicit size labels needed

**Implementation**:

```python
class DensePoseBodySizeEstimator(nn.Module):
    """
    Estimates body size category from DensePose representation.
    Can be pre-trained or learned end-to-end.
    """
    def __init__(self, densepose_channels=25, hidden_dim=256, num_sizes=3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(densepose_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sizes)
        )
    
    def forward(self, densepose):
        features = self.encoder(densepose)
        size_logits = self.classifier(features)
        return size_logits  # Can use soft labels during training
```

**Training approach**:
1. Pre-train the body size estimator on a small labeled subset
2. Use predicted body size during main training
3. Model learns the relationship between DensePose shape and clothing fit

### Option B: Implicit Learning Through Data Augmentation

**Concept**: Augment training data to simulate different size relationships.

```python
class SizeAugmentedDataset(Dataset):
    """
    Augments data to create pseudo size variations.
    """
    def __init__(self, base_dataset, augment_prob=0.3):
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if torch.rand(1) < self.augment_prob:
            # Apply size-related augmentations
            scale_factor = torch.empty(1).uniform_(0.85, 1.15).item()
            
            # Scale the person (simulates different body sizes)
            item['image'] = self.scale_and_pad(item['image'], scale_factor)
            item['densepose'] = self.scale_and_pad(item['densepose'], scale_factor)
            item['mask'] = self.scale_and_pad(item['mask'], scale_factor)
            
            # Cloth stays the same size (creates tight/loose fit illusion)
            
        return item
    
    def scale_and_pad(self, img, scale):
        # Scale image and pad to original size
        # Implementation details...
        pass
```

### Option C: Text-Based Size Conditioning (Zero-Shot)

**Concept**: Leverage the text encoder to understand size descriptions.

**Advantages**:
- No additional training needed
- Uses existing text-conditioning capabilities
- Flexible size descriptions

**Implementation**:

```python
# At inference time, modify the prompt
def get_size_aware_prompt(base_prompt, body_size, cloth_size, fit_description):
    """
    Creates prompts that describe the size relationship.
    """
    fit_descriptions = {
        (-2, -1): "very tight fitting",
        (0,): "perfectly fitting",
        (1, 2): "loose and oversized"
    }
    
    # Find appropriate fit description
    diff = cloth_size - body_size
    for range_keys, desc in fit_descriptions.items():
        if diff in range_keys:
            fit_desc = desc
            break
    
    return f"{base_prompt}, {fit_desc} clothes on a {body_size} person"

# Example usage:
prompt = get_size_aware_prompt(
    "a woman wearing a blue shirt",
    body_size="petite",
    cloth_size="large",
    fit_description=None
)
# Output: "a woman wearing a blue shirt, loose and oversized clothes on a petite person"
```

### Option D: LoRA Fine-tuning for Specific Size Combinations

**Concept**: Train separate LoRA adapters for different fit types.

```python
# Train 3 LoRA adapters:
# 1. lora_tight_fit - for when cloth_size < body_size
# 2. lora_normal_fit - for when cloth_size ≈ body_size  
# 3. lora_loose_fit - for when cloth_size > body_size

# At inference, blend LoRAs based on desired fit:
def blend_loras(tight_lora, normal_lora, loose_lora, target_fit):
    """
    target_fit: -1 (tight) to 1 (loose)
    """
    if target_fit < 0:
        return interpolate_loras(tight_lora, normal_lora, 1 + target_fit)
    else:
        return interpolate_loras(normal_lora, loose_lora, target_fit)
```

---

## Recommended Approach Based on Your Situation

### If you have the full combinatorial dataset (9 combinations):

**Use Method A (Size Embedding Conditioning)** with:
- `injection_method='add_to_timestep'` (most stable)
- Train size embedder + fine-tune decoder layers of UNet
- Use classifier-free guidance for size conditioning (10% dropout)

### If you want to reduce data collection:

**Use Option A (DensePose-Based)** because:
- IDM-VTON already uses DensePose
- Body size information is implicitly encoded
- Only need to add a lightweight estimator module

### If you want minimal code changes:

**Use Option C (Text-Based)** because:
- No architectural changes needed
- Works with pretrained model
- Trade-off: less precise control

---

## Dataset Annotation Format

```json
{
  "annotations": [
    {
      "id": "001",
      "image": "image/001.jpg",
      "cloth": "cloth/shirt_blue_M.jpg",
      "densepose": "image-densepose/001.jpg",
      "mask": "agnostic-mask/001.jpg",
      "body_size": "medium",
      "cloth_size": "M",
      "body_measurements": {
        "height_cm": 165,
        "bust_cm": 88,
        "waist_cm": 68,
        "hip_cm": 94
      },
      "cloth_measurements": {
        "size_label": "M",
        "bust_cm": 92,
        "length_cm": 65
      },
      "caption": "a woman wearing a blue cotton shirt"
    }
  ]
}
```

---

## Training Tips

1. **Balanced Sampling**: Ensure equal representation of all size combinations
   ```python
   sampler = WeightedRandomSampler(
       weights=compute_sample_weights(dataset),
       num_samples=len(dataset),
       replacement=True
   )
   ```

2. **Progressive Training**: Start with size embedder frozen, then unfreeze
   ```python
   # Phase 1: Train only UNet with random size embeddings
   # Phase 2: Train size embedder with frozen UNet
   # Phase 3: Fine-tune both together
   ```

3. **Evaluation Metrics**: Add size-specific metrics
   - FID per size combination
   - Size consistency score (does generated fit match conditioning?)
   - User study on perceived fit accuracy

4. **Data Collection Tips**:
   - Use mannequins or dress forms for consistent poses
   - Standardize lighting and background
   - Capture multiple angles if possible
   - Include measurement data for precise size annotations

---

## Summary

| Approach | Data Requirement | Implementation Complexity | Control Precision |
|----------|-----------------|---------------------------|-------------------|
| Size Embedding | Full combinatorial | High | Very High |
| DensePose-Based | Paired only | Medium | High |
| Data Augmentation | Paired only | Low | Medium |
| Text-Based | None (pretrained) | Very Low | Low-Medium |
| LoRA per Fit | Few examples per fit | Medium | High |

Choose based on your available data and desired level of control over the size-fitting behavior.
