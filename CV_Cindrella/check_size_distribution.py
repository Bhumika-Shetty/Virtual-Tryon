"""
Quick check of size distribution in dataset
This helps verify if size calculations are correct
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset
from size_aware_dataset import SizeAwareVitonHDDataset

print("Checking size distribution WITHOUT augmentation...")
print("This shows the natural distribution in VITON-HD\n")

# Create dataset WITHOUT size augmentation
dataset = SizeAwareVitonHDDataset(
    dataroot_path="/scratch/bds9746/datasets/VITON-HD",
    phase="train",
    size=(512, 384),
    size_augmentation=False,  # ← DISABLED augmentation
    enable_size_conditioning=True
)

# Use small subset
subset = Subset(dataset, list(range(100)))

# Count size labels
size_counts = {'tight': 0, 'fitted': 0, 'loose': 0, 'oversized': 0}
size_ratios_list = []

print("Analyzing 100 samples (no augmentation)...\n")

for i in range(100):
    try:
        sample = subset[i]
        label = sample['size_label']
        ratios = sample['size_ratios'].numpy()

        size_counts[label] += 1
        size_ratios_list.append(ratios)

        if i < 10:  # Print first 10
            print(f"Sample {i:3d}: width={ratios[0]:.3f}, length={ratios[1]:.3f}, shoulder={ratios[2]:.3f} → {label}")
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        continue

print("\n" + "="*60)
print("SIZE DISTRIBUTION (NO AUGMENTATION)")
print("="*60)
total = sum(size_counts.values())
for label, count in sorted(size_counts.items()):
    pct = (count / total * 100) if total > 0 else 0
    print(f"{label:10s}: {count:3d} samples ({pct:5.1f}%)")

print("\n" + "="*60)
print("RATIO STATISTICS")
print("="*60)
if size_ratios_list:
    import numpy as np
    ratios_array = np.array(size_ratios_list)
    print(f"Width Ratio:    mean={ratios_array[:, 0].mean():.3f}, std={ratios_array[:, 0].std():.3f}, range=[{ratios_array[:, 0].min():.3f}, {ratios_array[:, 0].max():.3f}]")
    print(f"Length Ratio:   mean={ratios_array[:, 1].mean():.3f}, std={ratios_array[:, 1].std():.3f}, range=[{ratios_array[:, 1].min():.3f}, {ratios_array[:, 1].max():.3f}]")
    print(f"Shoulder Ratio: mean={ratios_array[:, 2].mean():.3f}, std={ratios_array[:, 2].std():.3f}, range=[{ratios_array[:, 2].min():.3f}, {ratios_array[:, 2].max():.3f}]")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if total > 0:
    oversized_pct = (size_counts['oversized'] / total * 100)
    if oversized_pct > 70:
        print("⚠️  HIGH oversized percentage!")
        print("   This could mean:")
        print("   1. VITON-HD naturally has oversized garments")
        print("   2. Size calculation might need adjustment")
        print("   3. Classification thresholds might need tuning")
    else:
        print("✅ Size distribution looks reasonable")
print("="*60)
