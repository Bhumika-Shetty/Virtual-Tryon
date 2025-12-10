"""
Preprocess Size-Aware Dataset

Converts the raw dataset (body-cloth naming convention) to VITON-HD format
with size annotations for Full Combinatorial Training.

Author: Cinderella Team
Date: 2025-12-08
"""

import os
import json
import shutil
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple

# Configuration
SOURCE_DIR = "/scratch/bds9746/CV_Vton_backup/dataset"
OUTPUT_DIR = "/scratch/bds9746/CV_Vton_backup/CV_Cindrella/data_size_aware"
TARGET_SIZE = (768, 1024)  # (width, height) - VITON-HD standard

# Size mappings
BODY_SIZE_MAP = {'small': 'S', 'medium': 'M', 'large': 'L'}
CLOTH_SIZE_MAP = {'small': 'S', 'medium': 'M', 'large': 'L'}


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract body size and cloth size.

    Args:
        filename: e.g., 'small-large.png'

    Returns:
        (body_size, cloth_size)
    """
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = name.split('-')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def resize_image(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize image while maintaining aspect ratio and padding."""
    target_w, target_h = target_size
    orig_w, orig_h = img.size

    # Calculate scaling factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create padded image (white background)
    padded = Image.new('RGB', target_size, (255, 255, 255))

    # Center the resized image
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(img_resized, (paste_x, paste_y))

    return padded


def create_simple_mask(img: Image.Image) -> Image.Image:
    """
    Create a simple segmentation mask for the upper body region.
    This is a placeholder - ideally use a proper segmentation model.
    """
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # Create a simple rectangular mask for upper body (rough approximation)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Upper body region (roughly center of image, top 60%)
    top = int(h * 0.15)
    bottom = int(h * 0.65)
    left = int(w * 0.25)
    right = int(w * 0.75)

    mask[top:bottom, left:right] = 255

    # Apply some blur to soften edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask)


def create_simple_densepose(img: Image.Image) -> Image.Image:
    """
    Create a simple placeholder DensePose visualization.
    This is a placeholder - ideally use DensePose model.
    """
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # Create a simple colored representation
    densepose = np.zeros((h, w, 3), dtype=np.uint8)

    # Torso region (blue)
    top = int(h * 0.15)
    bottom = int(h * 0.55)
    left = int(w * 0.35)
    right = int(w * 0.65)
    densepose[top:bottom, left:right] = [100, 100, 200]  # Blue-ish for torso

    # Head region (skin color)
    head_top = int(h * 0.02)
    head_bottom = int(h * 0.15)
    head_left = int(w * 0.40)
    head_right = int(w * 0.60)
    densepose[head_top:head_bottom, head_left:head_right] = [200, 150, 150]

    # Arms (different colors)
    # Left arm
    densepose[int(h*0.15):int(h*0.45), int(w*0.15):int(w*0.35)] = [150, 200, 150]
    # Right arm
    densepose[int(h*0.15):int(h*0.45), int(w*0.65):int(w*0.85)] = [150, 200, 150]

    # Legs (lower body)
    densepose[int(h*0.55):int(h*0.95), int(w*0.30):int(w*0.70)] = [200, 100, 100]

    return Image.fromarray(densepose)


def create_cloth_mask(cloth_img: Image.Image) -> Image.Image:
    """Create a mask for the garment (remove white background)."""
    img_np = np.array(cloth_img)

    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # Threshold to find non-white areas (garment)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("Size-Aware Dataset Preprocessing")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}")
    print("=" * 60)

    # Create directories
    for phase in ['train', 'test']:
        for subdir in ['image', 'cloth', 'cloth-mask', 'image-densepose', 'agnostic-mask']:
            os.makedirs(os.path.join(OUTPUT_DIR, phase, subdir), exist_ok=True)

    # Find all person images
    person_images = []
    garment_image = None

    for f in os.listdir(SOURCE_DIR):
        if f.startswith('.'):
            continue
        if f == 'tshirt.png':
            garment_image = f
        elif f.endswith('.png') or f.endswith('.jpg'):
            body_size, cloth_size = parse_filename(f)
            if body_size and cloth_size:
                person_images.append({
                    'filename': f,
                    'body_size': body_size,
                    'cloth_size': cloth_size
                })

    print(f"Found {len(person_images)} person images")
    print(f"Found garment: {garment_image}")

    if not garment_image:
        print("ERROR: No garment image (tshirt.png) found!")
        return

    # Process garment image
    print("\nProcessing garment...")
    garment_path = os.path.join(SOURCE_DIR, garment_image)
    garment_img = Image.open(garment_path).convert('RGB')
    garment_resized = resize_image(garment_img, TARGET_SIZE)

    # Save garment for both train and test
    for phase in ['train', 'test']:
        garment_resized.save(os.path.join(OUTPUT_DIR, phase, 'cloth', 'tshirt.jpg'))

        # Create and save cloth mask
        cloth_mask = create_cloth_mask(garment_resized)
        cloth_mask.save(os.path.join(OUTPUT_DIR, phase, 'cloth-mask', 'tshirt.jpg'))

    print("  Garment processed and saved")

    # Process person images
    print("\nProcessing person images...")
    annotations = []
    train_pairs = []

    for i, item in enumerate(person_images):
        print(f"  [{i+1}/{len(person_images)}] {item['filename']}")

        # Load and resize image
        img_path = os.path.join(SOURCE_DIR, item['filename'])
        img = Image.open(img_path).convert('RGB')
        img_resized = resize_image(img, TARGET_SIZE)

        # Generate output filename
        body_code = BODY_SIZE_MAP[item['body_size']]
        cloth_code = CLOTH_SIZE_MAP[item['cloth_size']]
        out_name = f"{body_code}_{cloth_code}.jpg"

        # Decide train/test split (use all for train in this small dataset)
        phase = 'train'

        # Save person image
        img_resized.save(os.path.join(OUTPUT_DIR, phase, 'image', out_name))

        # Create and save mask
        mask = create_simple_mask(img_resized)
        mask.save(os.path.join(OUTPUT_DIR, phase, 'agnostic-mask', out_name.replace('.jpg', '_mask.png')))

        # Create and save DensePose
        densepose = create_simple_densepose(img_resized)
        densepose.save(os.path.join(OUTPUT_DIR, phase, 'image-densepose', out_name))

        # Add to annotations
        annotations.append({
            'id': out_name.replace('.jpg', ''),
            'image': out_name,
            'cloth': 'tshirt.jpg',
            'body_size': item['body_size'],
            'body_size_code': body_code,
            'cloth_size': item['cloth_size'],
            'cloth_size_code': cloth_code,
            'original_file': item['filename']
        })

        # Add to train pairs
        train_pairs.append(f"{out_name} tshirt.jpg")

    # Save size annotations JSON
    annotations_path = os.path.join(OUTPUT_DIR, 'size_annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump({'annotations': annotations}, f, indent=2)
    print(f"\nSaved size annotations to {annotations_path}")

    # Save train pairs
    train_pairs_path = os.path.join(OUTPUT_DIR, 'train_pairs.txt')
    with open(train_pairs_path, 'w') as f:
        f.write('\n'.join(train_pairs))
    print(f"Saved train pairs to {train_pairs_path}")

    # Also save to train subdirectory (some scripts expect it there)
    shutil.copy(train_pairs_path, os.path.join(OUTPUT_DIR, 'train', 'train_pairs.txt'))

    # Create a minimal vitonhd_train_tagged.json (required by dataset loader)
    tagged_json = {}
    for ann in annotations:
        tagged_json[ann['image']] = [{
            'file_name': ann['image'],
            'tag_info': [
                {'tag_name': 'item', 'tag_category': 'shirt'},
                {'tag_name': 'sleeveLength', 'tag_category': 'short sleeve'},
                {'tag_name': 'neckLine', 'tag_category': 'round neck'}
            ]
        }]

    tagged_path = os.path.join(OUTPUT_DIR, 'train', 'vitonhd_train_tagged.json')
    with open(tagged_path, 'w') as f:
        json.dump(tagged_json, f, indent=2)
    print(f"Saved tagged JSON to {tagged_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(annotations)}")
    print(f"\nDataset structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    train/")
    print(f"      image/          ({len(annotations)} files)")
    print(f"      cloth/          (1 file)")
    print(f"      cloth-mask/     (1 file)")
    print(f"      image-densepose/ ({len(annotations)} files)")
    print(f"      agnostic-mask/  ({len(annotations)} files)")
    print(f"    size_annotations.json")
    print(f"    train_pairs.txt")
    print("\nSize distribution:")
    for ann in annotations:
        print(f"  {ann['id']}: body={ann['body_size_code']}, cloth={ann['cloth_size_code']}")

    print("\n" + "=" * 60)
    print("NOTE: The DensePose and masks are PLACEHOLDERS!")
    print("For production, generate proper DensePose using detectron2")
    print("and masks using a segmentation model.")
    print("=" * 60)


if __name__ == '__main__':
    main()
