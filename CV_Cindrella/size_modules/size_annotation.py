"""
Size Annotation Module - Excess Fabric Analysis Approach

This module extracts size information using EXCESS FABRIC ANALYSIS.
Instead of comparing width ratios, we measure how much of the garment
extends BEYOND the body silhouette using DensePose.

Key Insight:
- Fitted clothes: minimal excess fabric beyond body
- Loose/Oversized clothes: significant excess fabric extending beyond body

This approach handles long sleeves correctly (they don't inflate "looseness")
and works regardless of arm positions or poses.

Key Functions:
- extract_body_dimensions(): Get body measurements from OpenPose keypoints
- extract_garment_dimensions(): Get garment measurements from segmentation mask
- extract_excess_fabric_metrics(): NEW - calculate excess using DensePose
- compute_size_ratio(): Calculate garment/body ratios (for compatibility)
- get_size_label(): Classify into tight/fitted/loose/oversized

Author: Cinderella Team
Date: 2025-11-30
Updated: 2025-12-04 - Excess Fabric Analysis approach
"""

import numpy as np
import json
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional, List
import torch


# OpenPose keypoint indices (18-point COCO format)
OPENPOSE_KEYPOINTS = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'right_hip': 8,
    'right_knee': 9,
    'right_ankle': 10,
    'left_hip': 11,
    'left_knee': 12,
    'left_ankle': 13,
    'right_eye': 14,
    'left_eye': 15,
    'right_ear': 16,
    'left_ear': 17,
}


class SizeAnnotator:
    """
    Quick size annotation using OpenPose keypoints and garment masks.

    This is the "quick approach" that uses existing preprocessing outputs
    without requiring additional landmark detection models.
    """

    def __init__(self):
        """Initialize the size annotator."""
        self.keypoint_indices = OPENPOSE_KEYPOINTS

    def extract_body_dimensions(
        self,
        keypoints: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Extract body dimensions from OpenPose keypoints.

        Args:
            keypoints: OpenPose keypoints array of shape (18, 3) where each row is [x, y, confidence]
            confidence_threshold: Minimum confidence to consider a keypoint valid

        Returns:
            Dictionary with body measurements:
            - shoulder_width: Distance between shoulders (pixels)
            - torso_length: Distance from neck to hips (pixels)
            - body_width_at_waist: Estimated waist width (pixels)
        """
        dimensions = {}

        # Extract key points
        neck = keypoints[self.keypoint_indices['neck']]
        left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
        right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
        left_hip = keypoints[self.keypoint_indices['left_hip']]
        right_hip = keypoints[self.keypoint_indices['right_hip']]

        # Shoulder width
        if left_shoulder[2] > confidence_threshold and right_shoulder[2] > confidence_threshold:
            dimensions['shoulder_width'] = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        else:
            dimensions['shoulder_width'] = 0.0

        # Torso length (neck to midpoint of hips)
        if neck[2] > confidence_threshold and left_hip[2] > confidence_threshold and right_hip[2] > confidence_threshold:
            hip_midpoint = (left_hip[:2] + right_hip[:2]) / 2
            dimensions['torso_length'] = np.linalg.norm(neck[:2] - hip_midpoint)
        else:
            dimensions['torso_length'] = 0.0

        # Hip width (proxy for body width)
        if left_hip[2] > confidence_threshold and right_hip[2] > confidence_threshold:
            dimensions['body_width_at_waist'] = np.linalg.norm(left_hip[:2] - right_hip[:2])
        else:
            dimensions['body_width_at_waist'] = dimensions['shoulder_width'] * 0.85  # Estimate

        return dimensions

    def extract_garment_dimensions(
        self,
        garment_mask: np.ndarray,
        garment_image: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract garment dimensions from segmentation mask.

        IMPORTANT: Measures at BODY LEVEL (shoulder/chest), not full bounding box.
        This avoids including sleeves which extend beyond the torso.

        Args:
            garment_mask: Binary mask of the garment (H, W) or (H, W, 1)
            garment_image: Optional RGB image for better segmentation
            keypoints: Optional OpenPose keypoints to measure at body level

        Returns:
            Dictionary with garment measurements:
            - garment_width: Width at chest level (not including sleeves)
            - garment_length: Height of garment
            - garment_shoulder_width: Width at shoulder level
        """
        dimensions = {}

        # Ensure mask is 2D
        if len(garment_mask.shape) == 3:
            garment_mask = garment_mask[:, :, 0]

        # Binarize mask
        binary_mask = (garment_mask > 127).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # No garment found, return zeros
            return {'garment_width': 0.0, 'garment_length': 0.0, 'garment_shoulder_width': 0.0}

        # Get largest contour (main garment)
        main_contour = max(contours, key=cv2.contourArea)

        # Bounding box (for reference and length)
        x, y, w, h = cv2.boundingRect(main_contour)
        dimensions['garment_length'] = float(h)

        # If keypoints provided, measure at body level
        if keypoints is not None:
            left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
            right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]

            # Only use if keypoints are valid
            if left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1:
                # Get shoulder Y coordinate
                shoulder_y = int((left_shoulder[1] + right_shoulder[1]) / 2)

                # Measure garment at shoulder level (±15 pixels band)
                band_top = max(0, shoulder_y - 15)
                band_bottom = min(binary_mask.shape[0], shoulder_y + 15)
                shoulder_band = binary_mask[band_top:band_bottom, :]

                if shoulder_band.sum() > 0:
                    shoulder_pixels = np.where(shoulder_band > 0)
                    garment_width_at_shoulder = float(shoulder_pixels[1].max() - shoulder_pixels[1].min())
                    dimensions['garment_shoulder_width'] = garment_width_at_shoulder
                else:
                    dimensions['garment_shoulder_width'] = float(w) * 0.7  # Fallback with correction

                # Measure at CHEST level - where arms are typically AWAY from body
                # This allows detecting oversized while excluding disconnected sleeves

                left_hip = keypoints[self.keypoint_indices['left_hip']]
                right_hip = keypoints[self.keypoint_indices['right_hip']]

                # Calculate chest Y (midpoint between shoulders and hips)
                if left_hip[2] > 0.1 and right_hip[2] > 0.1:
                    hip_y = (left_hip[1] + right_hip[1]) / 2
                    chest_y = int((shoulder_y + hip_y) / 2)
                else:
                    # Fallback: use garment center Y
                    chest_y = y + h // 2

                # Find horizontal center of garment (not body!)
                garment_cols = np.where(binary_mask.sum(axis=0) > 0)[0]
                if len(garment_cols) > 0:
                    center_x = (garment_cols.min() + garment_cols.max()) // 2
                else:
                    center_x = binary_mask.shape[1] // 2

                # Get the row at chest level
                if 0 <= chest_y < binary_mask.shape[0]:
                    row = binary_mask[chest_y, :]

                    # Find central contiguous segment (excludes disconnected sleeves)
                    if row[center_x] > 0:
                        left = center_x
                        while left > 0 and row[left - 1] > 0:
                            left -= 1
                        right = center_x
                        while right < len(row) - 1 and row[right + 1] > 0:
                            right += 1

                        segment_width = float(right - left)
                        dimensions['garment_width'] = segment_width
                    else:
                        # Center not on garment, fallback to shoulder width
                        dimensions['garment_width'] = dimensions['garment_shoulder_width']
                else:
                    dimensions['garment_width'] = dimensions['garment_shoulder_width']

                return dimensions

        # Fallback: No keypoints, use corrected bounding box method
        # Measure at vertical center of garment (avoids sleeves)
        center_y = y + h // 2
        center_band = binary_mask[center_y - 15:center_y + 15, :]

        if center_band.sum() > 0:
            center_pixels = np.where(center_band > 0)
            garment_width_center = float(center_pixels[1].max() - center_pixels[1].min())
            dimensions['garment_width'] = garment_width_center
        else:
            # Last resort: use bounding box with correction factor
            dimensions['garment_width'] = float(w) * 0.7  # Sleeves typically add ~30%

        # Shoulder width: measure at top 15% of garment
        shoulder_region_height = int(y + h * 0.15)
        shoulder_mask = binary_mask[y:shoulder_region_height, :]

        if shoulder_mask.sum() > 0:
            shoulder_points = np.where(shoulder_mask > 0)
            if len(shoulder_points[1]) > 0:
                shoulder_width = shoulder_points[1].max() - shoulder_points[1].min()
                dimensions['garment_shoulder_width'] = float(shoulder_width)
            else:
                dimensions['garment_shoulder_width'] = dimensions['garment_width']
        else:
            dimensions['garment_shoulder_width'] = dimensions['garment_width']

        return dimensions

    def load_openpose_keypoints(self, json_path: str) -> np.ndarray:
        """
        Load OpenPose keypoints from JSON file.

        Args:
            json_path: Path to OpenPose JSON output

        Returns:
            Keypoints array of shape (18, 3)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # OpenPose format: people[0]['pose_keypoints_2d'] is a flat list of [x, y, conf] * 18
        if 'people' in data and len(data['people']) > 0:
            keypoints_flat = data['people'][0]['pose_keypoints_2d']
            keypoints = np.array(keypoints_flat).reshape(-1, 3)
            return keypoints
        else:
            # No person detected, return zeros
            return np.zeros((18, 3))

    def extract_excess_fabric_metrics(
        self,
        garment_mask: np.ndarray,
        densepose_image: np.ndarray,
    ) -> Dict[str, float]:
        """
        Extract excess fabric metrics using DensePose body segmentation.

        This is the NEW approach that measures how much of the garment extends
        BEYOND the body silhouette. This directly captures visual "fit".

        Args:
            garment_mask: Binary mask of the warped garment (H, W)
            densepose_image: RGB DensePose visualization image (H, W, 3)

        Returns:
            Dictionary with:
            - excess_ratio: Fraction of garment outside body (0-1)
            - side_extension_ratio: Horizontal extension relative to body width
            - pixels_on_body: Garment pixels overlapping torso
            - pixels_outside_body: Garment pixels outside torso
        """
        # Ensure garment mask is 2D and binary
        if len(garment_mask.shape) == 3:
            garment_mask = garment_mask[:, :, 0]
        binary_garment = (garment_mask > 127).astype(np.uint8) * 255

        # Extract body torso from DensePose
        # In DensePose visualization: blue shades = torso
        blue = densepose_image[:, :, 2].astype(float)
        red = densepose_image[:, :, 0].astype(float)
        green = densepose_image[:, :, 1].astype(float)

        # Torso mask: blue is dominant and significant
        torso_mask = ((blue > 80) & (blue > red * 1.2) & (blue > green * 1.2)).astype(np.uint8) * 255

        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_OPEN, kernel)

        # Calculate overlap and excess
        garment_on_body = cv2.bitwise_and(binary_garment, torso_mask)
        garment_outside_body = cv2.bitwise_and(binary_garment, cv2.bitwise_not(torso_mask))

        pixels_on_body = np.sum(garment_on_body > 0)
        pixels_outside_body = np.sum(garment_outside_body > 0)
        total_garment_pixels = pixels_on_body + pixels_outside_body

        # Measure in torso vertical zone only (excludes sleeves below torso)
        torso_y_coords = np.where(torso_mask > 0)[0]
        if len(torso_y_coords) > 0:
            torso_top_y = torso_y_coords.min()
            torso_bottom_y = torso_y_coords.max()

            # Create mask for torso vertical zone
            torso_zone_mask = np.zeros_like(binary_garment)
            torso_zone_mask[torso_top_y:torso_bottom_y, :] = 255

            # Garment metrics in torso zone only
            garment_in_zone = cv2.bitwise_and(binary_garment, torso_zone_mask)
            garment_on_body_zone = cv2.bitwise_and(garment_in_zone, torso_mask)
            garment_outside_zone = cv2.bitwise_and(garment_in_zone, cv2.bitwise_not(torso_mask))

            pixels_on_body_zone = np.sum(garment_on_body_zone > 0)
            pixels_outside_zone = np.sum(garment_outside_zone > 0)
            total_in_zone = pixels_on_body_zone + pixels_outside_zone
        else:
            total_in_zone = total_garment_pixels
            pixels_outside_zone = pixels_outside_body

        # Calculate EXCESS RATIO (zone-based for accuracy)
        if total_in_zone > 0:
            excess_ratio = pixels_outside_zone / total_in_zone
        else:
            excess_ratio = 0.0

        # Calculate SIDE EXTENSION RATIO
        torso_coords = np.where(torso_mask > 0)
        garment_coords = np.where(binary_garment > 0)

        if len(torso_coords[1]) > 0 and len(garment_coords[1]) > 0:
            torso_left = torso_coords[1].min()
            torso_right = torso_coords[1].max()
            garment_left = garment_coords[1].min()
            garment_right = garment_coords[1].max()

            left_extension = max(0, torso_left - garment_left)
            right_extension = max(0, garment_right - torso_right)
            body_width = torso_right - torso_left

            if body_width > 0:
                side_extension_ratio = (left_extension + right_extension) / body_width
            else:
                side_extension_ratio = 0.0
        else:
            side_extension_ratio = 0.0

        return {
            'excess_ratio': excess_ratio,
            'side_extension_ratio': side_extension_ratio,
            'pixels_on_body': pixels_on_body,
            'pixels_outside_body': pixels_outside_body,
            'total_garment_pixels': total_garment_pixels,
        }


def compute_size_ratio(
    body_dims: Dict[str, float],
    garment_dims: Dict[str, float]
) -> Tuple[float, float, float]:
    """
    Compute size ratios from body and garment dimensions.

    Args:
        body_dims: Dictionary with body measurements
        garment_dims: Dictionary with garment measurements

    Returns:
        Tuple of (width_ratio, length_ratio, shoulder_ratio)
        where ratio = garment_dimension / body_dimension
    """
    # Width ratio: garment width / body shoulder width
    if body_dims['shoulder_width'] > 0:
        width_ratio = garment_dims['garment_width'] / body_dims['shoulder_width']
    else:
        width_ratio = 1.0  # Default to fitted

    # Length ratio: garment length / torso length
    if body_dims['torso_length'] > 0:
        length_ratio = garment_dims['garment_length'] / body_dims['torso_length']
    else:
        length_ratio = 1.0

    # Shoulder ratio: garment shoulder width / body shoulder width
    if body_dims['shoulder_width'] > 0:
        shoulder_ratio = garment_dims['garment_shoulder_width'] / body_dims['shoulder_width']
    else:
        shoulder_ratio = 1.0

    return width_ratio, length_ratio, shoulder_ratio


def get_size_label(
    width_ratio: float = None,
    length_ratio: float = None,
    shoulder_ratio: float = None,
    excess_ratio: float = None,
    side_extension_ratio: float = None,
) -> str:
    """
    Classify size into discrete labels using HYBRID approach.

    NEW APPROACH (2025-12-04): Uses excess fabric metrics for better accuracy.
    Falls back to ratio-based classification if excess metrics not provided.

    Classification logic:
    1. If excess metrics provided (preferred):
       - side_extension < 0.10 → FITTED (minimal horizontal extension)
       - excess_ratio < 0.26 → FITTED (low overall excess)
       - excess_ratio < 0.36 → LOOSE
       - otherwise → OVERSIZED

    2. Fallback (ratio-based):
       - r < 0.9 → tight
       - 0.9 ≤ r < 1.1 → fitted
       - 1.1 ≤ r < 1.3 → loose
       - r ≥ 1.3 → oversized

    Args:
        width_ratio: Garment width / body width ratio (legacy, for compatibility)
        length_ratio: Optional length ratio for additional info
        shoulder_ratio: Optional shoulder ratio (fallback)
        excess_ratio: NEW - fraction of garment outside body (0-1)
        side_extension_ratio: NEW - horizontal extension relative to body

    Returns:
        Size label: 'tight', 'fitted', 'loose', or 'oversized'
    """
    # NEW: Use excess fabric metrics if available (more accurate)
    # REFINED v2: Tighter thresholds based on testing
    if excess_ratio is not None:
        # Hybrid classification using both metrics
        if side_extension_ratio is not None and side_extension_ratio < 0.05:
            # Truly minimal horizontal extension = fitted (long-sleeve fitted only)
            return 'fitted'
        elif excess_ratio < 0.26:
            return 'fitted'
        elif excess_ratio < 0.32:
            return 'loose'
        else:
            return 'oversized'

    # FALLBACK: Use ratio-based classification
    ratio = shoulder_ratio if shoulder_ratio is not None else width_ratio
    if ratio is None:
        return 'fitted'  # Default

    if ratio < 0.9:
        return 'tight'
    elif 0.9 <= ratio < 1.1:
        return 'fitted'
    elif 1.1 <= ratio < 1.3:
        return 'loose'
    else:  # >= 1.3
        return 'oversized'


def get_size_label_id(size_label: str) -> int:
    """Convert size label to integer ID for model training."""
    label_map = {
        'tight': 0,
        'fitted': 1,
        'loose': 2,
        'oversized': 3
    }
    return label_map.get(size_label, 1)  # Default to fitted


def create_size_map(
    size_label: str,
    height: int,
    width: int,
    garment_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create a spatial size guidance map.

    This map will be used by the Size Controller to guide the diffusion process.
    For now, we create a simple uniform map, but this can be refined to have
    spatial variations (e.g., tighter at shoulders, looser at torso).

    Args:
        size_label: Size classification ('tight', 'fitted', 'loose', 'oversized')
        height: Output height
        width: Output width
        garment_mask: Optional garment mask for spatial variations

    Returns:
        Size map of shape (height, width) with values in [0, 1]
        0 = tight, 1 = very loose
    """
    label_to_value = {
        'tight': 0.0,
        'fitted': 0.33,
        'loose': 0.66,
        'oversized': 1.0
    }

    value = label_to_value.get(size_label, 0.33)
    size_map = np.ones((height, width), dtype=np.float32) * value

    # TODO: Add spatial variations based on garment_mask if needed

    return size_map


# Convenience function for batch processing
def annotate_sample(
    image_path: str,
    garment_path: str,
    openpose_json_path: Optional[str] = None,
    garment_mask_path: Optional[str] = None,
    densepose_path: Optional[str] = None,
    image_keypoints: Optional[np.ndarray] = None,
) -> Dict:
    """
    Annotate a single image-garment pair with size information.

    NEW (2025-12-04): Supports DensePose-based excess fabric analysis for
    more accurate size classification.

    Args:
        image_path: Path to person image
        garment_path: Path to garment image
        openpose_json_path: Path to OpenPose JSON (if available)
        garment_mask_path: Path to garment mask (warped mask preferred)
        densepose_path: Path to DensePose image (for excess fabric analysis)
        image_keypoints: Pre-loaded keypoints (alternative to JSON)

    Returns:
        Dictionary with size annotation:
        - width_ratio, length_ratio, shoulder_ratio (legacy)
        - excess_ratio, side_extension_ratio (NEW - more accurate)
        - size_label
        - size_label_id
        - body_dims, garment_dims (for debugging)
    """
    annotator = SizeAnnotator()

    # Load keypoints
    if image_keypoints is not None:
        keypoints = image_keypoints
    elif openpose_json_path is not None:
        keypoints = annotator.load_openpose_keypoints(openpose_json_path)
    else:
        keypoints = None  # Will use fallback methods

    # Extract body dimensions (for legacy compatibility)
    if keypoints is not None:
        body_dims = annotator.extract_body_dimensions(keypoints)
    else:
        body_dims = {'shoulder_width': 0, 'torso_length': 0, 'body_width_at_waist': 0}

    # Load and process garment
    if garment_mask_path is not None:
        garment_mask = np.array(Image.open(garment_mask_path))
    else:
        # Use full garment image and threshold
        garment_img = np.array(Image.open(garment_path))
        # Simple background removal: assume white/light background
        gray = cv2.cvtColor(garment_img, cv2.COLOR_RGB2GRAY)
        _, garment_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Extract garment dimensions (legacy)
    garment_dims = annotator.extract_garment_dimensions(garment_mask, keypoints=keypoints)

    # Compute legacy ratios
    width_ratio, length_ratio, shoulder_ratio = compute_size_ratio(body_dims, garment_dims)

    # NEW: Extract excess fabric metrics if DensePose available (more accurate)
    excess_ratio = None
    side_extension_ratio = None

    if densepose_path is not None:
        try:
            densepose_img = np.array(Image.open(densepose_path))
            excess_metrics = annotator.extract_excess_fabric_metrics(garment_mask, densepose_img)
            excess_ratio = excess_metrics['excess_ratio']
            side_extension_ratio = excess_metrics['side_extension_ratio']
        except Exception as e:
            # Fall back to ratio-based if DensePose fails
            pass

    # Get label using hybrid approach (excess metrics if available, else ratios)
    size_label = get_size_label(
        width_ratio=width_ratio,
        length_ratio=length_ratio,
        shoulder_ratio=shoulder_ratio,
        excess_ratio=excess_ratio,
        side_extension_ratio=side_extension_ratio,
    )
    size_label_id = get_size_label_id(size_label)

    result = {
        'width_ratio': width_ratio,
        'length_ratio': length_ratio,
        'shoulder_ratio': shoulder_ratio,
        'size_label': size_label,
        'size_label_id': size_label_id,
        'body_dims': body_dims,
        'garment_dims': garment_dims,
    }

    # Add excess metrics if available
    if excess_ratio is not None:
        result['excess_ratio'] = excess_ratio
        result['side_extension_ratio'] = side_extension_ratio

    return result


if __name__ == '__main__':
    # Test the annotator
    print("Size Annotation Module - Test Mode")
    print("=" * 50)

    # Example usage
    annotator = SizeAnnotator()

    # Create dummy keypoints for testing
    test_keypoints = np.zeros((18, 3))
    test_keypoints[:, 2] = 1.0  # Set all confidences to 1.0

    # Simulate a person with shoulder width ~100px, torso length ~200px
    test_keypoints[OPENPOSE_KEYPOINTS['neck']] = [100, 50, 1.0]
    test_keypoints[OPENPOSE_KEYPOINTS['left_shoulder']] = [50, 50, 1.0]
    test_keypoints[OPENPOSE_KEYPOINTS['right_shoulder']] = [150, 50, 1.0]
    test_keypoints[OPENPOSE_KEYPOINTS['left_hip']] = [75, 250, 1.0]
    test_keypoints[OPENPOSE_KEYPOINTS['right_hip']] = [125, 250, 1.0]

    body_dims = annotator.extract_body_dimensions(test_keypoints)
    print("\nBody Dimensions:")
    for k, v in body_dims.items():
        print(f"  {k}: {v:.2f}px")

    # Test garment dimensions with dummy mask
    test_garment_mask = np.zeros((300, 200), dtype=np.uint8)
    test_garment_mask[50:250, 40:160] = 255  # 200px tall, 120px wide

    garment_dims = annotator.extract_garment_dimensions(test_garment_mask)
    print("\nGarment Dimensions:")
    for k, v in garment_dims.items():
        print(f"  {k}: {v:.2f}px")

    # Compute ratios
    width_ratio, length_ratio, shoulder_ratio = compute_size_ratio(body_dims, garment_dims)
    print("\nSize Ratios:")
    print(f"  Width ratio: {width_ratio:.3f}")
    print(f"  Length ratio: {length_ratio:.3f}")
    print(f"  Shoulder ratio: {shoulder_ratio:.3f}")

    # Get label
    size_label = get_size_label(width_ratio, length_ratio, shoulder_ratio)
    print(f"\nSize Label: {size_label}")
    print(f"Size Label ID: {get_size_label_id(size_label)}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
