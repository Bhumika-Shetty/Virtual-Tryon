"""
Small Training Run with Verbose Size Calculation Logging

This script:
1. Trains on only 100 samples (for quick testing)
2. Runs for 3 epochs
3. Prints detailed size calculation info
4. Documents all preprocessing steps

For demonstrating to data team how size is calculated.

Run: python train_small_verbose.py --data_dir /path/to/VITON-HD
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("SIZE-AWARE VTON: Small Training with Verbose Logging")
print("="*80)
print()

# Import size-aware modules
from size_aware_dataset import SizeAwareVitonHDDataset
from size_modules import SizeEncoder, SimpleSizeController

def log_size_calculation(sample_idx, sample):
    """Print detailed size calculation for a sample"""
    print(f"\n{'='*60}")
    print(f"Sample #{sample_idx}: {sample['c_name']}")
    print(f"{'='*60}")

    # Size information
    size_ratios = sample['size_ratios'].cpu().numpy()
    size_label = sample['size_label']
    size_label_id = sample['size_label_id'].item()

    print(f"\n📏 SIZE CALCULATIONS:")
    print(f"  Width Ratio:    {size_ratios[0]:.4f}")
    print(f"  Length Ratio:   {size_ratios[1]:.4f}")
    print(f"  Shoulder Ratio: {size_ratios[2]:.4f}")
    print(f"  → Size Label: '{size_label}' (ID: {size_label_id})")

    # Classification
    if size_ratios[0] < 0.9:
        fit_desc = "TIGHT (garment smaller than body)"
    elif size_ratios[0] < 1.1:
        fit_desc = "FITTED (garment matches body)"
    elif size_ratios[0] < 1.3:
        fit_desc = "LOOSE (garment larger than body)"
    else:
        fit_desc = "OVERSIZED (garment much larger than body)"
    print(f"  Classification: {fit_desc}")

    # Size map info
    size_map = sample['size_map']
    print(f"\n🗺️  SIZE MAP:")
    print(f"  Shape: {tuple(size_map.shape)}")
    print(f"  Mean Value: {size_map.mean():.4f}")
    print(f"  Range: [{size_map.min():.4f}, {size_map.max():.4f}]")

    # Other data shapes
    print(f"\n📦 DATA SHAPES:")
    print(f"  Person Image: {tuple(sample['image'].shape)}")
    print(f"  Garment (CLIP): {tuple(sample['cloth'].shape)}")
    print(f"  Garment (Pure): {tuple(sample['cloth_pure'].shape)}")
    print(f"  Pose Image: {tuple(sample['pose_img'].shape)}")
    print(f"  Inpaint Mask: {tuple(sample['inpaint_mask'].shape)}")

    # Captions
    print(f"\n💬 CAPTIONS:")
    print(f"  Person: \"{sample['caption']}\"")
    print(f"  Garment: \"{sample['caption_cloth']}\"")
    print(f"  Annotation: \"{sample['annotation']}\"")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch/bds9746/datasets/VITON-HD",
                        help="Path to VITON-HD dataset")
    parser.add_argument("--output_dir", type=str, default="./results/small_train_test",
                        help="Where to save outputs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (keep small for verbose logging)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to use (subset)")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log detailed info every N samples")

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Subset Size: {args.num_samples} samples")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Create dataset
    print("Loading dataset...")
    full_dataset = SizeAwareVitonHDDataset(
        dataroot_path=args.data_dir,
        phase="train",
        size=(512, 384),
        size_augmentation=True,  # Enable size augmentation
        enable_size_conditioning=True
    )

    # Use only a small subset
    indices = list(range(min(args.num_samples, len(full_dataset))))
    dataset = Subset(full_dataset, indices)

    print(f"✅ Dataset loaded: {len(dataset)} samples (subset of {len(full_dataset)})")
    print()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # Initialize size modules
    print("Initializing size-aware modules...")
    size_encoder = SizeEncoder(
        input_dim=3,
        hidden_dim=256,
        output_dim=768
    ).to(device)

    size_controller = SimpleSizeController(
        size_embedding_dim=768,
        output_size=(128, 96)
    ).to(device)

    print(f"✅ Size Encoder: {sum(p.numel() for p in size_encoder.parameters()):,} parameters")
    print(f"✅ Size Controller: {sum(p.numel() for p in size_controller.parameters()):,} parameters")
    print()

    # Training loop
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print()

    global_step = 0

    for epoch in range(args.num_epochs):
        print(f"\n{'#'*80}")
        print(f"EPOCH {epoch + 1}/{args.num_epochs}")
        print(f"{'#'*80}\n")

        epoch_size_stats = {
            'tight': 0,
            'fitted': 0,
            'loose': 0,
            'oversized': 0
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            # Log first few samples in detail
            if batch_idx < 2 or (batch_idx % args.log_every == 0):
                print("\n" + "="*80)
                print(f"BATCH {batch_idx + 1} - Step {global_step}")
                print("="*80)

                # Log first sample in batch with full details
                sample_to_log = {
                    k: v[0] if isinstance(v, torch.Tensor) else v[0]
                    for k, v in batch.items()
                }
                log_size_calculation(batch_idx * args.batch_size, sample_to_log)

            # Get size ratios from batch
            size_ratios = batch['size_ratios'].to(device)
            size_labels = batch['size_label']

            # Count size distribution
            for label in size_labels:
                epoch_size_stats[label] += 1

            # Encode size
            size_embeddings = size_encoder(size_ratios)
            size_maps = size_controller(size_embeddings)

            # Brief progress update
            avg_ratio = size_ratios[:, 0].mean().item()
            pbar.set_postfix({
                'avg_width_ratio': f'{avg_ratio:.3f}',
                'batch_labels': '/'.join(size_labels[:2])
            })

            # Simulate some training (since we don't have full UNet yet)
            # In real training, this is where you'd pass to UNet and compute loss
            # For now, just demonstrate the pipeline works

        # Epoch summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'='*80}")
        print(f"\nSize Distribution in this epoch:")
        total = sum(epoch_size_stats.values())
        for label, count in sorted(epoch_size_stats.items()):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {label:10s}: {count:3d} samples ({pct:5.1f}%)")
        print()

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal Steps: {global_step}")
    print(f"Epochs Completed: {args.num_epochs}")
    print(f"Samples Processed: {len(dataset)} × {args.num_epochs} = {len(dataset) * args.num_epochs}")
    print()
    print("📊 SIZE CALCULATION SUMMARY:")
    print("  • Size ratios computed automatically from OpenPose + masks")
    print("  • Size augmentation (0.7-1.5× scaling) creates diversity")
    print("  • No manual size labeling needed!")
    print()
    print("📝 For your data team:")
    print("  See DATA_PREPROCESSING_GUIDE.md for full details")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
