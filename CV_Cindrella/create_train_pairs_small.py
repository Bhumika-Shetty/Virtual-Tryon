"""
Create train_pairs.txt for small subset training (100 samples)

Format: person_image.jpg cloth_image.jpg
For simplicity, we'll pair each person with their corresponding garment
"""

import os
import random

# Paths
dataset_dir = "/scratch/bds9746/datasets/VITON-HD/train"
image_dir = os.path.join(dataset_dir, "image")
output_file = "/scratch/bds9746/CV_Vton/CV_Cindrella/train_pairs_small.txt"

print("Creating train_pairs_small.txt...")

# Get all image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
print(f"Found {len(image_files)} total images")

# Take first 100 for our small subset
subset_size = 100
subset_images = image_files[:subset_size]

print(f"Selected {len(subset_images)} images for small training")

# Create pairs - for simplicity, pair each person with their own garment
# In real training, you might want to do cross-pairing
pairs = []
for img_file in subset_images:
    # Format: person_image.jpg cloth_image.jpg
    pairs.append(f"{img_file} {img_file}\n")

# Write to file
with open(output_file, 'w') as f:
    f.writelines(pairs)

print(f"✅ Created {output_file}")
print(f"   Total pairs: {len(pairs)}")
print(f"\nFirst 5 pairs:")
for pair in pairs[:5]:
    print(f"  {pair.strip()}")
