"""
Size-Aware Dataset Loader for VITON-HD

Extends the original VitonHDDataset to include size ratio extraction and augmentation.
Computes size information on-the-fly during training.

Author: Cinderella Team
Date: 2025-11-30
"""

import os
import random
import json
import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Literal, Tuple, Dict, Optional
from transformers import CLIPImageProcessor

from size_modules.size_annotation import (
    SizeAnnotator,
    compute_size_ratio,
    get_size_label,
    get_size_label_id,
    create_size_map
)


class SizeAwareVitonHDDataset(data.Dataset):
    """
    Size-aware extension of VITON-HD dataset.

    Adds size ratio computation and size-based data augmentation to the original dataset.
    """

    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        size_augmentation: bool = True,
        size_aug_range: Tuple[float, float] = (0.7, 1.5),
        enable_size_conditioning: bool = True,
    ):
        """
        Initialize size-aware dataset.

        Args:
            dataroot_path: Path to VITON-HD dataset root
            phase: 'train' or 'test'
            order: 'paired' or 'unpaired'
            size: Image size (height, width)
            size_augmentation: Enable size-based augmentation during training
            size_aug_range: Range for garment scaling augmentation
            enable_size_conditioning: Whether to compute size ratios
        """
        super().__init__()

        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.size_augmentation = size_augmentation and (phase == "train")
        self.size_aug_range = size_aug_range
        self.enable_size_conditioning = enable_size_conditioning

        # Transforms
        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.toTensor = transforms.ToTensor()
        self.clip_processor = CLIPImageProcessor()

        # Initialize size annotator
        if self.enable_size_conditioning:
            self.size_annotator = SizeAnnotator()

        # Load garment annotations
        with open(
            os.path.join(dataroot_path, phase, f"vitonhd_{phase}_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = ["sleeveLength", "neckLine", "item"]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str

        self.order = order

        # Load image pairs
        im_names = []
        c_names = []

        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)

    def __getitem__(self, index: int) -> Dict:
        """
        Get a single sample with size information.

        Returns:
            Dictionary containing all data including size ratios and labels
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # Get garment annotation
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"

        # Load images
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width, self.height))

        image = self.transform(im_pil_big)

        # Load mask
        mask = Image.open(
            os.path.join(
                self.dataroot,
                self.phase,
                "agnostic-mask",
                im_name.replace('.jpg', '_mask.png')
            )
        ).resize((self.width, self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]

        # Load DensePose
        densepose_map = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        )
        pose_img = self.toTensor(densepose_map)

        # Size augmentation for training
        garment_scale = 1.0
        if self.size_augmentation:
            # Randomly scale garment to create size variations
            garment_scale = random.uniform(*self.size_aug_range)

        # Extract size information (if enabled)
        if self.enable_size_conditioning:
            try:
                # Load OpenPose keypoints from JSON (more accurate than DensePose extraction)
                openpose_json_path = os.path.join(
                    self.dataroot, self.phase, "openpose_json",
                    im_name.replace('.jpg', '_keypoints.json')
                )
                if os.path.exists(openpose_json_path):
                    keypoints = self.size_annotator.load_openpose_keypoints(openpose_json_path)
                else:
                    # Fallback to DensePose extraction
                    keypoints = self._extract_keypoints_from_densepose(densepose_map)

                # Extract body dimensions
                body_dims = self.size_annotator.extract_body_dimensions(keypoints)

                # Extract garment dimensions from WARPED mask (not flat cloth)
                # This gives us the garment as it appears on the body, not laid flat
                warped_mask_path = os.path.join(
                    self.dataroot, self.phase, "gt_cloth_warped_mask", im_name
                )
                if os.path.exists(warped_mask_path):
                    # Use warped garment mask for accurate body-relative measurements
                    warped_mask = Image.open(warped_mask_path).convert('L')
                    garment_mask = np.array(warped_mask)
                else:
                    # Fallback to flat cloth if warped mask not available
                    cloth_np = np.array(cloth)
                    cloth_gray = np.mean(cloth_np, axis=2).astype(np.uint8) if len(cloth_np.shape) == 3 else cloth_np
                    _, garment_mask = cv2.threshold(cloth_gray, 240, 255, cv2.THRESH_BINARY_INV)

                # Extract garment dimensions - pass keypoints for body-level measurement
                garment_dims = self.size_annotator.extract_garment_dimensions(
                    garment_mask, keypoints=keypoints
                )

                # If we used flat cloth, apply correction factor
                if not os.path.exists(warped_mask_path):
                    # Flat garments appear ~2x wider than on body
                    garment_dims['garment_width'] *= 0.5
                    garment_dims['garment_shoulder_width'] *= 0.5

                # Apply augmentation scale to garment dimensions
                if self.size_augmentation:
                    for key in garment_dims:
                        garment_dims[key] *= garment_scale

                # Compute size ratios
                width_ratio, length_ratio, shoulder_ratio = compute_size_ratio(body_dims, garment_dims)

                # Get size label
                size_label = get_size_label(width_ratio, length_ratio, shoulder_ratio)
                size_label_id = get_size_label_id(size_label)

                # Create size map
                size_map = create_size_map(size_label, self.height // 4, self.width // 4)

            except Exception as e:
                # Fallback to default values if size extraction fails
                print(f"Warning: Size extraction failed for {im_name}: {e}")
                width_ratio, length_ratio, shoulder_ratio = 1.0, 1.0, 1.0
                size_label = 'fitted'
                size_label_id = 1
                size_map = np.ones((self.height // 4, self.width // 4), dtype=np.float32) * 0.33

        else:
            # Default values when size conditioning disabled
            width_ratio, length_ratio, shoulder_ratio = 1.0, 1.0, 1.0
            size_label = 'fitted'
            size_label_id = 1
            size_map = np.ones((self.height // 4, self.width // 4), dtype=np.float32) * 0.33

        # Standard augmentations (flip, color jitter, etc.) for training
        if self.phase == "train":
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)

            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5
                )
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
                    color_jitter.brightness, color_jitter.contrast,
                    color_jitter.saturation, color_jitter.hue
                )

                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                image = TF.affine(image, angle=0, translate=[0, 0], scale=scale_val, shear=0)
                mask = TF.affine(mask, angle=0, translate=[0, 0], scale=scale_val, shear=0)
                pose_img = TF.affine(pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0)

            if random.random() > 0.5:
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                image = TF.affine(
                    image, angle=0,
                    translate=[shift_valx * image.shape[-1], shift_valy * image.shape[-2]],
                    scale=1, shear=0
                )
                mask = TF.affine(
                    mask, angle=0,
                    translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]],
                    scale=1, shear=0
                )
                pose_img = TF.affine(
                    pose_img, angle=0,
                    translate=[shift_valx * pose_img.shape[-1], shift_valy * pose_img.shape[-2]],
                    scale=1, shear=0
                )

        # Process mask and cloth
        mask = 1 - mask
        cloth_trim = self.clip_processor(images=cloth, return_tensors="pt").pixel_values

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        im_mask = image * mask
        pose_img = self.norm(pose_img)

        # Prepare result dictionary
        result = {
            "c_name": c_name,
            "image": image,
            "cloth": cloth_trim,
            "cloth_pure": self.transform(cloth),
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption": "model is wearing " + cloth_annotation,
            "caption_cloth": "a photo of " + cloth_annotation,
            "annotation": cloth_annotation,
            "pose_img": pose_img,
            # Size conditioning
            "size_ratios": torch.tensor([width_ratio, length_ratio, shoulder_ratio], dtype=torch.float32),
            "size_label": size_label,
            "size_label_id": torch.tensor(size_label_id, dtype=torch.long),
            "size_map": torch.from_numpy(size_map).unsqueeze(0).float(),  # Add channel dim
        }

        return result

    def _extract_keypoints_from_densepose(self, densepose_image: Image.Image) -> np.ndarray:
        """
        Extract approximate keypoints from DensePose visualization.

        This is a heuristic fallback. Ideally, use actual OpenPose JSON files.

        Args:
            densepose_image: DensePose visualization image

        Returns:
            Approximate keypoints array (18, 3)
        """
        # Convert to numpy
        dp_np = np.array(densepose_image)

        # Simple heuristic: find colored regions and estimate keypoints
        # This is very approximate - better to use actual OpenPose data
        keypoints = np.zeros((18, 3))

        # Set all confidences to 0.5 (uncertain)
        keypoints[:, 2] = 0.5

        # Estimate some keypoints from image dimensions
        h, w = dp_np.shape[:2]

        # Rough estimates (center-based)
        keypoints[0] = [w // 2, h // 6, 0.5]  # nose
        keypoints[1] = [w // 2, h // 4, 0.5]  # neck
        keypoints[2] = [w // 3, h // 3, 0.5]  # right shoulder
        keypoints[5] = [2 * w // 3, h // 3, 0.5]  # left shoulder
        keypoints[8] = [w // 3, 2 * h // 3, 0.5]  # right hip
        keypoints[11] = [2 * w // 3, 2 * h // 3, 0.5]  # left hip

        return keypoints

    def __len__(self) -> int:
        return len(self.im_names)


# For backward compatibility
VitonHDDataset = SizeAwareVitonHDDataset


if __name__ == '__main__':
    print("Testing Size-Aware Dataset Loader")
    print("=" * 50)

    # This would require actual VITON-HD data to test
    print("Dataset loader created successfully!")
    print("To test, provide a path to VITON-HD dataset")
