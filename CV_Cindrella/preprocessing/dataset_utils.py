"""
Dataset utilities for VITON-HD and DressCode datasets.

Provides:
- Dataset loading and management
- Feature access and caching
- Batch creation for training/inference
- Data validation and organization
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetItem:
    """Single item from dataset with all features"""

    image_id: str
    human_image: Image.Image
    garment_image: Image.Image
    pose_image: Image.Image
    mask: Image.Image
    keypoints: Dict
    category: str
    dataset_type: str

    # Optional tensors (loaded on demand)
    garment_tensor: Optional[torch.Tensor] = None
    pose_tensor: Optional[torch.Tensor] = None


class VITONHDDataset(Dataset):
    """
    VITON-HD Dataset loader.

    Directory structure:
        dataset_root/
        ├── humans/
        │   ├── keypoints/
        │   │   └── *.json
        │   ├── parsing/
        │   │   └── *.npy
        │   ├── masks/
        │   │   └── *.jpg
        │   └── poses/
        │       └── *.jpg
        └── garments/
            ├── *.jpg
            └── tensors/
                └── *.pt
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        load_tensors: bool = False,
        cache_images: bool = False
    ):
        """
        Initialize VITON-HD dataset.

        Args:
            root_dir: Root directory with preprocessed features
            load_tensors: Load precomputed tensors from disk
            cache_images: Cache images in memory (faster but memory-intensive)
        """
        self.root_dir = Path(root_dir)
        self.load_tensors = load_tensors
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None

        # Collect all image IDs
        keypoint_dir = self.root_dir / 'humans' / 'keypoints'
        if not keypoint_dir.exists():
            raise FileNotFoundError(f"Keypoints directory not found: {keypoint_dir}")

        self.image_ids = sorted([
            f.stem for f in keypoint_dir.glob('*.json')
        ])

        if not self.image_ids:
            raise ValueError("No images found in dataset")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Load single dataset item"""
        image_id = self.image_ids[idx]
        return self._load_item(image_id)

    def _load_item(self, image_id: str) -> DatasetItem:
        """Load all features for a single image"""
        cache_key = f"viton-hd_{image_id}"

        if self.image_cache is not None and cache_key in self.image_cache:
            return self.image_cache[cache_key]

        # Load images
        human_img = Image.open(
            self.root_dir / 'humans' / f"{image_id}.jpg"
        ).convert("RGB")
        garment_img = Image.open(
            self.root_dir / 'garments' / f"{image_id}.jpg"
        ).convert("RGB")
        pose_img = Image.open(
            self.root_dir / 'humans' / 'poses' / f"{image_id}.jpg"
        ).convert("RGB")
        mask_img = Image.open(
            self.root_dir / 'humans' / 'masks' / f"{image_id}.jpg"
        ).convert("L")

        # Load keypoints
        with open(self.root_dir / 'humans' / 'keypoints' / f"{image_id}.json") as f:
            keypoints = json.load(f)

        # Load tensors if requested
        garment_tensor = None
        if self.load_tensors:
            tensor_path = self.root_dir / 'garments' / 'tensors' / f"{image_id}.pt"
            if tensor_path.exists():
                garment_tensor = torch.load(tensor_path)

        item = DatasetItem(
            image_id=image_id,
            human_image=human_img,
            garment_image=garment_img,
            pose_image=pose_img,
            mask=mask_img,
            keypoints=keypoints,
            category='upper_body',
            dataset_type='viton-hd',
            garment_tensor=garment_tensor
        )

        if self.image_cache is not None:
            self.image_cache[cache_key] = item

        return item

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Get batch of items as tensors for model input.

        Args:
            indices: List of dataset indices

        Returns:
            Dictionary with batched tensors
        """
        from torchvision import transforms

        items = [self[idx] for idx in indices]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        human_batch = []
        garment_batch = []
        pose_batch = []
        mask_batch = []

        for item in items:
            human_batch.append(transform(item.human_image))
            garment_batch.append(transform(item.garment_image))
            pose_batch.append(transform(item.pose_image))
            mask_batch.append(torch.from_numpy(np.array(item.mask)).float() / 255.0)

        return {
            'human': torch.stack(human_batch),
            'garment': torch.stack(garment_batch),
            'pose': torch.stack(pose_batch),
            'mask': torch.stack(mask_batch),
            'image_ids': [items[i].image_id for i in range(len(items))]
        }


class DressCodeDataset(Dataset):
    """
    DressCode Dataset loader.

    Directory structure per category:
        dataset_root/{category}/
        ├── humans/
        │   ├── keypoints/
        │   ├── parsing/
        │   ├── masks/
        │   └── poses/
        └── garments/
            ├── *.jpg
            └── tensors/
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        category: str = 'upper_body',
        load_tensors: bool = False,
        cache_images: bool = False
    ):
        """
        Initialize DressCode dataset.

        Args:
            root_dir: Root directory with preprocessed features
            category: 'upper_body', 'lower_body', or 'dresses'
            load_tensors: Load precomputed tensors
            cache_images: Cache images in memory
        """
        self.root_dir = Path(root_dir) / category
        self.category = category
        self.load_tensors = load_tensors
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None

        # Collect image IDs
        keypoint_dir = self.root_dir / 'humans' / 'keypoints'
        if not keypoint_dir.exists():
            raise FileNotFoundError(f"Keypoints directory not found: {keypoint_dir}")

        self.image_ids = sorted([
            f.stem for f in keypoint_dir.glob('*.json')
        ])

        if not self.image_ids:
            raise ValueError(f"No images found in {category}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Load single dataset item"""
        image_id = self.image_ids[idx]
        return self._load_item(image_id)

    def _load_item(self, image_id: str) -> DatasetItem:
        """Load all features for a single image"""
        cache_key = f"dresscode_{self.category}_{image_id}"

        if self.image_cache is not None and cache_key in self.image_cache:
            return self.image_cache[cache_key]

        # Load images
        human_img = Image.open(
            self.root_dir / 'humans' / f"{image_id}.jpg"
        ).convert("RGB")
        garment_img = Image.open(
            self.root_dir / 'garments' / f"{image_id}.jpg"
        ).convert("RGB")
        pose_img = Image.open(
            self.root_dir / 'humans' / 'poses' / f"{image_id}.jpg"
        ).convert("RGB")
        mask_img = Image.open(
            self.root_dir / 'humans' / 'masks' / f"{image_id}.jpg"
        ).convert("L")

        # Load keypoints
        with open(self.root_dir / 'humans' / 'keypoints' / f"{image_id}.json") as f:
            keypoints = json.load(f)

        # Load tensors if requested
        garment_tensor = None
        if self.load_tensors:
            tensor_path = self.root_dir / 'garments' / 'tensors' / f"{image_id}.pt"
            if tensor_path.exists():
                garment_tensor = torch.load(tensor_path)

        item = DatasetItem(
            image_id=image_id,
            human_image=human_img,
            garment_image=garment_img,
            pose_image=pose_img,
            mask=mask_img,
            keypoints=keypoints,
            category=self.category,
            dataset_type='dresscode',
            garment_tensor=garment_tensor
        )

        if self.image_cache is not None:
            self.image_cache[cache_key] = item

        return item

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get batch of items as tensors"""
        from torchvision import transforms

        items = [self[idx] for idx in indices]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        human_batch = []
        garment_batch = []
        pose_batch = []
        mask_batch = []

        for item in items:
            human_batch.append(transform(item.human_image))
            garment_batch.append(transform(item.garment_image))
            pose_batch.append(transform(item.pose_image))
            mask_batch.append(torch.from_numpy(np.array(item.mask)).float() / 255.0)

        return {
            'human': torch.stack(human_batch),
            'garment': torch.stack(garment_batch),
            'pose': torch.stack(pose_batch),
            'mask': torch.stack(mask_batch),
            'image_ids': [items[i].image_id for i in range(len(items))]
        }


class DatasetValidator:
    """Validate dataset structure and completeness"""

    @staticmethod
    def validate_viton_hd(root_dir: Union[str, Path]) -> Dict[str, any]:
        """
        Validate VITON-HD dataset structure.

        Returns:
            Dictionary with validation results
        """
        root_dir = Path(root_dir)
        results = {
            'valid': True,
            'missing_dirs': [],
            'missing_files': [],
            'total_images': 0
        }

        required_dirs = [
            'humans/keypoints',
            'humans/parsing',
            'humans/masks',
            'humans/poses',
            'garments/tensors'
        ]

        for dir_path in required_dirs:
            full_path = root_dir / dir_path
            if not full_path.exists():
                results['missing_dirs'].append(dir_path)
                results['valid'] = False

        # Check for matching files
        keypoint_dir = root_dir / 'humans' / 'keypoints'
        if keypoint_dir.exists():
            keypoint_files = set(f.stem for f in keypoint_dir.glob('*.json'))
            results['total_images'] = len(keypoint_files)

            # Check other directories
            for dir_path, extension in [
                ('humans/masks', '*.jpg'),
                ('humans/poses', '*.jpg'),
                ('garments', '*.jpg')
            ]:
                dir_full_path = root_dir / dir_path
                if dir_full_path.exists():
                    found_files = set(f.stem for f in dir_full_path.glob(extension))
                    missing = keypoint_files - found_files
                    if missing:
                        results['missing_files'].append({
                            'dir': dir_path,
                            'count': len(missing),
                            'examples': list(missing)[:5]
                        })
                        results['valid'] = False

        return results

    @staticmethod
    def validate_dresscode(root_dir: Union[str, Path]) -> Dict[str, any]:
        """Validate DressCode dataset structure"""
        root_dir = Path(root_dir)
        results = {
            'valid': True,
            'categories': {}
        }

        for category in ['upper_body', 'lower_body', 'dresses']:
            category_path = root_dir / category
            if not category_path.exists():
                continue

            cat_results = {
                'valid': True,
                'total_images': 0,
                'missing_dirs': []
            }

            required_dirs = [
                'humans/keypoints',
                'humans/masks',
                'humans/poses',
                'garments'
            ]

            for dir_path in required_dirs:
                if not (category_path / dir_path).exists():
                    cat_results['missing_dirs'].append(dir_path)
                    cat_results['valid'] = False

            # Count images
            keypoint_dir = category_path / 'humans' / 'keypoints'
            if keypoint_dir.exists():
                cat_results['total_images'] = len(list(keypoint_dir.glob('*.json')))

            results['categories'][category] = cat_results
            results['valid'] = results['valid'] and cat_results['valid']

        return results


def get_dataset_loader(
    dataset_type: str,
    root_dir: str,
    batch_size: int = 4,
    category: Optional[str] = None,
    load_tensors: bool = False,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for specified dataset.

    Args:
        dataset_type: 'viton-hd' or 'dresscode'
        root_dir: Dataset root directory
        batch_size: Batch size
        category: Category for DressCode (required if dresscode)
        load_tensors: Load precomputed tensors
        shuffle: Shuffle dataset
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    if dataset_type == 'viton-hd':
        dataset = VITONHDDataset(
            root_dir,
            load_tensors=load_tensors,
            cache_images=False
        )
    elif dataset_type == 'dresscode':
        if category is None:
            raise ValueError("category required for dresscode")
        dataset = DressCodeDataset(
            root_dir,
            category=category,
            load_tensors=load_tensors,
            cache_images=False
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_batch
    )


def _collate_batch(batch: List[DatasetItem]) -> Dict[str, any]:
    """Collate function for DataLoader"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    human_batch = []
    garment_batch = []
    pose_batch = []
    mask_batch = []
    image_ids = []

    for item in batch:
        human_batch.append(transform(item.human_image))
        garment_batch.append(transform(item.garment_image))
        pose_batch.append(transform(item.pose_image))
        mask_batch.append(torch.from_numpy(np.array(item.mask)).float() / 255.0)
        image_ids.append(item.image_id)

    return {
        'human': torch.stack(human_batch),
        'garment': torch.stack(garment_batch),
        'pose': torch.stack(pose_batch),
        'mask': torch.stack(mask_batch),
        'image_ids': image_ids
    }


if __name__ == '__main__':
    # Example usage
    print("Dataset utilities for VITON-HD and DressCode")
    print("\nValidate VITON-HD dataset:")
    print("  validator = DatasetValidator()")
    print("  results = validator.validate_viton_hd('/path/to/viton-hd')")

    print("\nLoad DressCode dataset:")
    print("  dataset = DressCodeDataset('/path/to/dresscode', category='upper_body')")
    print("  loader = DataLoader(dataset, batch_size=4)")

    print("\nUse helper function:")
    print("  loader = get_dataset_loader('dresscode', '/path', category='upper_body')")
