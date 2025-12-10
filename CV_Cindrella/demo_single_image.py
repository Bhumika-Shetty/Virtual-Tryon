#!/usr/bin/env python3
"""
DEMO: Size Calculation for a Single Image
==========================================
NEW APPROACH: Excess Fabric Analysis

Instead of comparing widths/ratios, we measure how much of the garment
extends BEYOND the body silhouette. This directly captures "excess fabric"
which is what defines visual fit.

- Fitted: Garment closely follows body contours (minimal excess)
- Loose/Oversized: Garment extends beyond body (more excess)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import cv2
from PIL import Image

# Paths - can be changed via command line
BASE_PATH = "/scratch/bds9746/datasets/VITON-HD/train"
IMAGE_NAME = "01681_00.jpg"

# Allow command line argument
if len(sys.argv) > 1:
    IMAGE_NAME = os.path.basename(sys.argv[1])

person_image_path = f"{BASE_PATH}/image/{IMAGE_NAME}"
openpose_json_path = f"{BASE_PATH}/openpose_json/{IMAGE_NAME.replace('.jpg', '_keypoints.json')}"
warped_mask_path = f"{BASE_PATH}/gt_cloth_warped_mask/{IMAGE_NAME}"
cloth_path = f"{BASE_PATH}/cloth/{IMAGE_NAME}"
densepose_path = f"{BASE_PATH}/image-densepose/{IMAGE_NAME}"

print("="*70)
print(f"SIZE CALCULATION DEMO FOR: {IMAGE_NAME}")
print("="*70)

# ============================================================
# STEP 1: Check Available Files
# ============================================================
print("\n📁 STEP 1: Checking Available Files")
print("-"*50)

files_to_check = [
    ("Person Image", person_image_path),
    ("OpenPose JSON", openpose_json_path),
    ("Warped Garment Mask", warped_mask_path),
    ("Flat Cloth Image", cloth_path),
    ("DensePose Image", densepose_path),
]

for name, path in files_to_check:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"  {exists} {name}: {path}")

# ============================================================
# STEP 2: Load DensePose and create body torso mask
# ============================================================
print(f"\n{'='*70}")
print("🔬 STEP 2: Extract Body Torso from DensePose")
print("-"*50)

densepose_img = np.array(Image.open(densepose_path))
print(f"  DensePose image size: {densepose_img.shape}")

# In DensePose visualization:
# - Blue shades = torso
# - Yellow/Green = arms
# - Other colors = legs, etc.
# We want the TORSO region as the body reference

# Extract blue channel dominant pixels (torso)
# Torso pixels have high blue, lower red/green
blue = densepose_img[:, :, 2].astype(float)
red = densepose_img[:, :, 0].astype(float)
green = densepose_img[:, :, 1].astype(float)

# Torso mask: blue is dominant and significant
torso_mask = ((blue > 80) & (blue > red * 1.2) & (blue > green * 1.2)).astype(np.uint8) * 255

# Clean up with morphological operations
kernel = np.ones((5, 5), np.uint8)
torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel)
torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_OPEN, kernel)

torso_pixels = np.sum(torso_mask > 0)
print(f"  Torso pixels detected: {torso_pixels}")

# ============================================================
# STEP 3: Load Warped Garment Mask
# ============================================================
print(f"\n{'='*70}")
print("👕 STEP 3: Load Warped Garment Mask")
print("-"*50)

warped_mask = np.array(Image.open(warped_mask_path).convert('L'))
binary_garment = (warped_mask > 127).astype(np.uint8) * 255

garment_pixels = np.sum(binary_garment > 0)
print(f"  Garment mask size: {warped_mask.shape}")
print(f"  Garment pixels: {garment_pixels}")

# ============================================================
# STEP 4: EXCESS FABRIC ANALYSIS
# ============================================================
print(f"\n{'='*70}")
print("📏 STEP 4: EXCESS FABRIC ANALYSIS")
print("-"*50)
print("  NEW APPROACH: Measure garment pixels OUTSIDE the body torso")
print("  More excess = looser fit")

# Get garment bounding box to focus analysis
contours, _ = cv2.findContours(binary_garment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    main_contour = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(main_contour)
    print(f"\n  Garment bounding box: x={gx}, y={gy}, w={gw}, h={gh}")

# Calculate overlap and excess
# - Garment ON body = garment AND torso
# - Garment OUTSIDE body = garment AND NOT torso

garment_on_body = cv2.bitwise_and(binary_garment, torso_mask)
garment_outside_body = cv2.bitwise_and(binary_garment, cv2.bitwise_not(torso_mask))

pixels_on_body = np.sum(garment_on_body > 0)
pixels_outside_body = np.sum(garment_outside_body > 0)
total_garment_pixels = pixels_on_body + pixels_outside_body

print(f"\n  Garment pixels ON body (overlapping torso): {pixels_on_body}")
print(f"  Garment pixels OUTSIDE body: {pixels_outside_body}")
print(f"  Total garment pixels: {total_garment_pixels}")

# IMPROVED: Measure excess only in TORSO VERTICAL ZONE
# This excludes sleeves that extend below the torso from the excess calculation
# Long sleeves going down arms shouldn't count as "loose/oversized"
torso_y_coords = np.where(torso_mask > 0)[0]
if len(torso_y_coords) > 0:
    torso_top_y = torso_y_coords.min()
    torso_bottom_y = torso_y_coords.max()

    # Create mask for torso vertical zone only
    torso_zone_mask = np.zeros_like(binary_garment)
    torso_zone_mask[torso_top_y:torso_bottom_y, :] = 255

    # Garment in torso zone
    garment_in_zone = cv2.bitwise_and(binary_garment, torso_zone_mask)
    garment_on_body_in_zone = cv2.bitwise_and(garment_in_zone, torso_mask)
    garment_outside_in_zone = cv2.bitwise_and(garment_in_zone, cv2.bitwise_not(torso_mask))

    pixels_on_body_zone = np.sum(garment_on_body_in_zone > 0)
    pixels_outside_zone = np.sum(garment_outside_in_zone > 0)
    total_in_zone = pixels_on_body_zone + pixels_outside_zone

    print(f"\n  TORSO ZONE ONLY (excludes sleeves below torso):")
    print(f"     Torso vertical range: Y={torso_top_y} to Y={torso_bottom_y}")
    print(f"     Garment ON body (in zone): {pixels_on_body_zone}")
    print(f"     Garment OUTSIDE body (in zone): {pixels_outside_zone}")
else:
    total_in_zone = total_garment_pixels
    pixels_outside_zone = pixels_outside_body
    pixels_on_body_zone = pixels_on_body

# Calculate EXCESS RATIO - USE ZONE-BASED for more accurate fit assessment
# This excludes sleeves that extend below torso from the calculation
if total_in_zone > 0:
    excess_ratio = pixels_outside_zone / total_in_zone
    excess_ratio_total = pixels_outside_body / total_garment_pixels if total_garment_pixels > 0 else 0
    coverage_ratio = pixels_on_body / torso_pixels if torso_pixels > 0 else 0
else:
    excess_ratio = 0
    excess_ratio_total = 0
    coverage_ratio = 0

print(f"\n  📊 EXCESS RATIO (torso zone): {excess_ratio:.4f} ({excess_ratio*100:.1f}%)")
print(f"     This measures garment extension in the TORSO region only")
print(f"     (excludes sleeves that go down arms)")
print(f"\n  📊 EXCESS RATIO (total): {excess_ratio_total:.4f} ({excess_ratio_total*100:.1f}%)")
print(f"     (For reference - includes sleeves)")
print(f"\n  📊 COVERAGE RATIO: {coverage_ratio:.4f}")
print(f"     How much of the torso the garment covers")

# ============================================================
# STEP 5: Analyze excess by region (left/right sides)
# ============================================================
print(f"\n{'='*70}")
print("📐 STEP 5: Regional Excess Analysis")
print("-"*50)

# Find torso center X
torso_coords = np.where(torso_mask > 0)
if len(torso_coords[1]) > 0:
    torso_center_x = int(np.mean(torso_coords[1]))
    torso_left = torso_coords[1].min()
    torso_right = torso_coords[1].max()
    print(f"  Torso X range: {torso_left} to {torso_right} (center: {torso_center_x})")
else:
    torso_center_x = densepose_img.shape[1] // 2
    torso_left = torso_center_x - 50
    torso_right = torso_center_x + 50

# Measure garment extension on LEFT side of body
garment_coords = np.where(binary_garment > 0)
if len(garment_coords[1]) > 0:
    garment_left = garment_coords[1].min()
    garment_right = garment_coords[1].max()

    left_extension = max(0, torso_left - garment_left)  # How far garment extends left of body
    right_extension = max(0, garment_right - torso_right)  # How far garment extends right of body

    print(f"  Garment X range: {garment_left} to {garment_right}")
    print(f"  Left extension (beyond body): {left_extension} px")
    print(f"  Right extension (beyond body): {right_extension} px")
    print(f"  Total side extension: {left_extension + right_extension} px")

    # SIDE EXTENSION RATIO: How much the garment extends beyond body sides
    # relative to body width
    body_width = torso_right - torso_left
    if body_width > 0:
        side_extension_ratio = (left_extension + right_extension) / body_width
        print(f"\n  📊 SIDE EXTENSION RATIO: {side_extension_ratio:.4f}")
        print(f"     (total side extension / body width)")
    else:
        side_extension_ratio = 0

# ============================================================
# STEP 6: Classification based on EXCESS FABRIC
# ============================================================
print(f"\n{'='*70}")
print("🏷️  STEP 6: Size Classification (Excess Fabric Method)")
print("-"*50)

# Classification thresholds - HYBRID APPROACH (REFINED v2)
# Observed from test images:
# - 00001 (fitted): excess=22.1%, side_ext=0.155
# - 00002 (fitted long-sleeve): excess=36.9%, side_ext=0.046 (sleeves inflate excess!)
# - 00007 (loose):  excess=30.3%, side_ext=0.144
# - 00086 (oversized): excess=32.4%, side_ext=0.071 (was incorrectly FITTED!)
#
# REFINED: Tighter thresholds to avoid false positives
# - side_ext < 0.05 (was 0.10) - only truly minimal extension = fitted
# - excess < 0.32 (was 0.36) for LOOSE boundary

print(f"\n  Classification Rules (HYBRID - Refined v2):")
print(f"     side_ext < 0.05                → FITTED (truly minimal extension)")
print(f"     excess < 0.26                  → FITTED (low overall excess)")
print(f"     excess < 0.32                  → LOOSE")
print(f"     otherwise                      → OVERSIZED")

# Hybrid classification using both metrics
# FITTED if: VERY low side extension (< 0.05) OR low excess ratio (< 0.26)
if side_extension_ratio < 0.05:
    # Truly minimal horizontal extension = fitted (handles long-sleeve fitted only)
    label = "FITTED"
    meaning = "Garment closely follows body contours (minimal side extension)"
elif excess_ratio < 0.26:
    label = "FITTED"
    meaning = "Garment closely follows body contours"
elif excess_ratio < 0.32:
    label = "LOOSE"
    meaning = "Garment has moderate drape beyond body"
else:
    label = "OVERSIZED"
    meaning = "Garment extends significantly beyond body"

print(f"\n  For {IMAGE_NAME}:")
print(f"     Excess Ratio: {excess_ratio:.4f}")
print(f"     Side Extension Ratio: {side_extension_ratio:.4f}")
print(f"     → Classification: {label}")
print(f"     → {meaning}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"📋 FINAL SUMMARY FOR {IMAGE_NAME}")
print("="*70)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  EXCESS FABRIC ANALYSIS                                         │
├─────────────────────────────────────────────────────────────────┤
│  Garment pixels total:      {total_garment_pixels:>8}                           │
│  Pixels ON body:            {pixels_on_body:>8}                           │
│  Pixels OUTSIDE body:       {pixels_outside_body:>8}                           │
├─────────────────────────────────────────────────────────────────┤
│  EXCESS RATIO:              {excess_ratio:>8.4f}  ({excess_ratio*100:.1f}% outside body)     │
│  SIDE EXTENSION RATIO:      {side_extension_ratio:>8.4f}                           │
├─────────────────────────────────────────────────────────────────┤
│  CLASSIFICATION:            {label:>8}                           │
│  {meaning:<62}│
└─────────────────────────────────────────────────────────────────┘
""")

print("="*70)
print("✅ Size calculation complete with EXCESS FABRIC method!")
print("="*70)
