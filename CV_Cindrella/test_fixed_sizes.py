"""
Test the fixed size calculation using warped masks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset
from size_aware_dataset import SizeAwareVitonHDDataset
import numpy as np

print("="*70)
print("TESTING FIXED SIZE CALCULATION")
print("="*70)
print("\nNow using WARPED garment masks instead of flat cloth images")
print("This should give more realistic size distributions\n")

# Create dataset WITHOUT size augmentation to see natural distribution
dataset = SizeAwareVitonHDDataset(
    dataroot_path="/scratch/bds9746/datasets/VITON-HD",
    phase="train",
    size=(512, 384),
    size_augmentation=False,  # No augmentation
    enable_size_conditioning=True
)

# Use small subset
subset = Subset(dataset, list(range(100)))

# Count size labels
size_counts = {'tight': 0, 'fitted': 0, 'loose': 0, 'oversized': 0}
size_ratios_list = []

print("Analyzing 100 samples with WARPED masks...\n")

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

print("\n" + "="*70)
print("SIZE DISTRIBUTION (FIXED - USING WARPED MASKS)")
print("="*70)
total = sum(size_counts.values())
for label, count in sorted(size_counts.items()):
    pct = (count / total * 100) if total > 0 else 0
    bar = "█" * int(pct / 2)  # Simple bar chart
    print(f"{label:10s}: {count:3d} samples ({pct:5.1f}%) {bar}")

print("\n" + "="*70)
print("RATIO STATISTICS")
print("="*70)
if size_ratios_list:
    ratios_array = np.array(size_ratios_list)
    print(f"Width Ratio:    mean={ratios_array[:, 0].mean():.3f}, std={ratios_array[:, 0].std():.3f}, range=[{ratios_array[:, 0].min():.3f}, {ratios_array[:, 0].max():.3f}]")
    print(f"Length Ratio:   mean={ratios_array[:, 1].mean():.3f}, std={ratios_array[:, 1].std():.3f}, range=[{ratios_array[:, 1].min():.3f}, {ratios_array[:, 1].max():.3f}]")
    print(f"Shoulder Ratio: mean={ratios_array[:, 2].mean():.3f}, std={ratios_array[:, 2].std():.3f}, range=[{ratios_array[:, 2].min():.3f}, {ratios_array[:, 2].max():.3f}]")

print("\n" + "="*70)
print("COMPARISON WITH PREVIOUS")
print("="*70)
print("BEFORE (using flat cloth):")
print("  Width ratio mean: 2.274 (garment 2.27× wider than body!)")
print("  Distribution: 95% oversized, 3% tight, 1% fitted, 1% loose")
print()
print("AFTER (using warped masks):")
if total > 0:
    avg_width = ratios_array[:, 0].mean()
    oversized_pct = (size_counts['oversized'] / total * 100)
    print(f"  Width ratio mean: {avg_width:.3f}")
    print(f"  Distribution: {size_counts['oversized']} oversized ({oversized_pct:.0f}%), "
          f"{size_counts['tight']} tight, {size_counts['fitted']} fitted, {size_counts['loose']} loose")
    print()

    if avg_width < 1.5 and oversized_pct < 60:
        print("✅ FIXED! Size distribution looks much more realistic now")
    elif avg_width < 2.0:
        print("⚠️  Better, but still slightly high. May need further tuning.")
    else:
        print("❌ Still too high. Additional fixes needed.")

print("="*70)
