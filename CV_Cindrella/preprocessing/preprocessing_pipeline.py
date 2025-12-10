"""
IDM-VTON Preprocessing Pipeline

This module provides a unified preprocessing pipeline for virtual try-on tasks.
It extracts features from human and garment images for the IDM-VTON model.

Features extracted:
- Human keypoints (OpenPose)
- Human parsing (clothing segmentation)
- Inpainting mask
- Pose visualization
- Garment features
"""




import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any
import numpy as np
from PIL import Image
import torch
import json
from dataclasses import dataclass, asdict

# Add parent directories to path for accessing local packages
PROJECT_ROOT = Path(__file__).absolute().parent.parent
GRADIO_DEMO = PROJECT_ROOT / 'gradio_demo'

# Add IDM-VTON root to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GRADIO_DEMO))

# Import from local detectron2 and densepose
try:
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
except ImportError:
    # Fallback if detectron2 not in standard location
    try:
        import sys
        sys.path.insert(0, str(GRADIO_DEMO / 'detectron2'))
        from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    except ImportError as e:
        print(f"Warning: Could not import detectron2 utilities: {e}")



@dataclass
class PreprocessingOutput:
    """Container for preprocessed features from a single image pair"""

    # Human image features
    human_image: Image.Image
    human_keypoints: Dict[str, Any]
    human_parsing: Image.Image
    inpaint_mask: Image.Image
    pose_image: Image.Image

    # Garment image features
    garment_image: Image.Image
    garment_tensor: Optional[torch.Tensor] = None

    # Metadata
    category: Optional[str] = None
    input_size: Tuple[int, int] = (768, 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization, excluding images and tensors)"""
        result = {}
        for key, value in asdict(self).items():
            if key not in ['human_image', 'garment_image', 'human_parsing', 'inpaint_mask', 'pose_image', 'garment_tensor']:
                result[key] = value
        return result


class FeatureExtractor:
    """
    Unified feature extraction for virtual try-on preprocessing.

    Handles:
    - Keypoint detection (OpenPose)
    - Human parsing (clothing segmentation)
    - Mask generation
    - Pose visualization
    """

    def __init__(self, gpu_id: int = 0, device: str = 'cuda:0'):
        """
        Initialize the feature extractor.

        Args:
            gpu_id: GPU device ID
            device: Device string ('cuda:0' or 'cpu')
        """
        self.gpu_id = gpu_id
        self.device = device

        # Initialize preprocessing models
        try:
            # Import from IDM-VTON preprocess modules
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose

            self.parsing_model = Parsing(gpu_id)
            self.openpose_model = OpenPose(gpu_id)

            # Import detectron2 utilities (with fallback)
            try:
                from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
            except ImportError:
                sys.path.insert(0, str(GRADIO_DEMO / 'detectron2'))
                from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

            self.convert_PIL_to_numpy = convert_PIL_to_numpy
            self._apply_exif_orientation = _apply_exif_orientation

        except Exception as e:
            print(f"Warning: Failed to initialize some preprocessing models: {e}")
            print("Make sure you are running from IDM-VTON root directory")
            raise RuntimeError(f"Failed to initialize preprocessing models: {e}")

        # Initialize DensePose
        self._init_densepose()

        # Tensor transform
        from torchvision import transforms
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _init_densepose(self):
        """Initialize DensePose for pose visualization"""
        try:
            import apply_net
            self.densepose_parser = apply_net
        except Exception as e:
            print(f"Warning: DensePose initialization failed: {e}")
            self.densepose_parser = None

    def extract_keypoints(self, human_image: Image.Image) -> Dict[str, Any]:
        """
        Extract OpenPose keypoints from human image.

        Args:
            human_image: PIL Image of human

        Returns:
            Dictionary with pose keypoints
        """
        human_img_resized = human_image.resize((384, 512))
        keypoints = self.openpose_model(human_img_resized)
        return keypoints

    def extract_parsing(self, human_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Extract human parsing segmentation.

        Args:
            human_image: PIL Image of human

        Returns:
            Tuple of (parsing_image, face_mask)
        """
        human_img_resized = human_image.resize((384, 512))
        parsing_image, face_mask = self.parsing_model(human_img_resized)
        return parsing_image, face_mask

    def generate_mask(
        self,
        human_image: Image.Image,
        parsing_image: Image.Image,
        keypoints: Dict[str, Any],
        category: str = 'upper_body',
        model_type: str = 'hd'
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate inpainting mask for the specified garment category.

        Args:
            human_image: PIL Image of human (768x1024)
            parsing_image: Parsing segmentation image
            keypoints: OpenPose keypoints dictionary
            category: 'upper_body', 'lower_body', or 'dresses'
            model_type: 'hd' or 'dc' (affects arm width)

        Returns:
            Tuple of (inpaint_mask, mask_gray)
        """
        from utils_mask import get_mask_location

        # Ensure images are correct size
        human_resized = human_image.resize((768, 1024))
        parsing_resized = parsing_image.resize((768, 1024))

        mask, mask_gray = get_mask_location(
            model_type, category, parsing_resized, keypoints
        )

        return mask, mask_gray

    def extract_pose_image(
        self,
        human_image: Image.Image,
        device: str = 'cuda'
    ) -> Image.Image:
        """
        Extract DensePose visualization.

        Args:
            human_image: PIL Image of human (768x1024)
            device: Device to use ('cuda' or 'cpu')

        Returns:
            Pose visualization image
        """
        if self.densepose_parser is None:
            raise RuntimeError("DensePose not available")

        # Prepare image for DensePose
        human_img_small = human_image.resize((384, 512))
        human_img_np = self._apply_exif_orientation(human_img_small)
        human_img_np = self.convert_PIL_to_numpy(human_img_np, format="BGR")

        # Extract DensePose
        try:
            args = self.densepose_parser.create_argument_parser().parse_args(
                (
                    'show',
                    './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                    './ckpt/densepose/model_final_162be9.pkl',
                    'dp_segm',
                    '-v',
                    '--opts',
                    'MODEL.DEVICE',
                    device
                )
            )
            pose_img = args.func(args, human_img_np)
            pose_img = pose_img[:, :, ::-1]  # BGR to RGB
            pose_img = Image.fromarray(pose_img).resize((768, 1024))

        except Exception as e:
            print(f"Warning: DensePose extraction failed: {e}")
            pose_img = human_image.resize((768, 1024))

        return pose_img

    def prepare_garment_tensor(
        self,
        garment_image: Image.Image,
        device: str = 'cuda:0'
    ) -> torch.Tensor:
        """
        Prepare garment image as tensor for model input.

        Args:
            garment_image: PIL Image of garment
            device: Device for tensor

        Returns:
            Tensor of shape (1, 3, 1024, 768)
        """
        garment_resized = garment_image.convert("RGB").resize((768, 1024))
        garment_tensor = self.tensor_transform(garment_resized).unsqueeze(0)
        garment_tensor = garment_tensor.to(device, torch.float16)

        return garment_tensor

    def prepare_pose_tensor(
        self,
        pose_image: Image.Image,
        device: str = 'cuda:0'
    ) -> torch.Tensor:
        """
        Prepare pose image as tensor for model input.

        Args:
            pose_image: PIL Image of pose visualization
            device: Device for tensor

        Returns:
            Tensor of shape (1, 3, 1024, 768)
        """
        pose_tensor = self.tensor_transform(pose_image).unsqueeze(0)
        pose_tensor = pose_tensor.to(device, torch.float16)

        return pose_tensor

    def process(
        self,
        human_image: Union[str, Image.Image],
        garment_image: Union[str, Image.Image],
        category: str = 'upper_body',
        use_auto_mask: bool = True,
        mask_image: Optional[Image.Image] = None,
        extract_tensors: bool = False,
        device: str = 'cuda:0'
    ) -> PreprocessingOutput:
        """
        Complete preprocessing pipeline for human-garment pair.

        Args:
            human_image: Path or PIL Image of human
            garment_image: Path or PIL Image of garment
            category: 'upper_body', 'lower_body', or 'dresses'
            use_auto_mask: If True, auto-generate mask; else use provided mask_image
            mask_image: Optional manual mask (PIL Image)
            extract_tensors: If True, return tensors for model input
            device: Device for tensor operations

        Returns:
            PreprocessingOutput with all extracted features

        Example:
            ```python
            extractor = FeatureExtractor(gpu_id=0)
            output = extractor.process(
                'person.jpg',
                'garment.jpg',
                category='upper_body',
                extract_tensors=True
            )
            ```
        """
        # Load images
        if isinstance(human_image, str):
            human_img = Image.open(human_image).convert("RGB")
        else:
            human_img = human_image.convert("RGB")

        if isinstance(garment_image, str):
            garment_img = Image.open(garment_image).convert("RGB")
        else:
            garment_img = garment_image.convert("RGB")

        # Resize to standard size
        human_img = human_img.resize((768, 1024))
        garment_img = garment_img.resize((768, 1024))

        # Extract features
        keypoints = self.extract_keypoints(human_img)
        parsing_image, _ = self.extract_parsing(human_img)

        # Generate mask
        if use_auto_mask:
            inpaint_mask, mask_gray = self.generate_mask(
                human_img, parsing_image, keypoints, category=category
            )
        else:
            if mask_image is None:
                raise ValueError("mask_image must be provided when use_auto_mask=False")
            inpaint_mask = mask_image.resize((768, 1024))
            from torchvision import transforms
            mask_gray = (1 - transforms.ToTensor()(inpaint_mask)) * self.tensor_transform(human_img)
            from torchvision.transforms.functional import to_pil_image
            mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        # Extract pose
        pose_image = self.extract_pose_image(human_img, device=device.split(':')[0])

        # Prepare tensors if requested
        garment_tensor = None
        if extract_tensors:
            garment_tensor = self.prepare_garment_tensor(garment_img, device=device)

        return PreprocessingOutput(
            human_image=human_img,
            human_keypoints=keypoints,
            human_parsing=parsing_image,
            inpaint_mask=inpaint_mask,
            pose_image=pose_image,
            garment_image=garment_img,
            garment_tensor=garment_tensor,
            category=category,
            input_size=(768, 1024)
        )


class DatasetPreprocessor:
    """
    Batch preprocessing for VITON-HD and DressCode datasets.

    Handles directory structures:

    VITON-HD:
        |-- image/
        |-- image-densepose/
        |-- agnostic-mask/
        |-- cloth/

    DressCode:
        |-- {category}/
            |-- images/
            |-- image-densepose/
            |-- dc_caption.txt
    """

    def __init__(self, gpu_id: int = 0, device: str = 'cuda:0'):
        """Initialize dataset preprocessor"""
        self.extractor = FeatureExtractor(gpu_id=gpu_id, device=device)
        self.device = device

    def preprocess_viton_hd(
        self,
        dataset_root: str,
        output_root: str,
        categories: list = None
    ):
        """
        Preprocess VITON-HD dataset.

        Args:
            dataset_root: Path to VITON-HD directory
            output_root: Path to save preprocessed features
            categories: List of categories to process (None = all)

        Example structure created:
            output_root/
            ├── humans/
            │   ├── keypoints/
            │   ├── parsing/
            │   ├── masks/
            │   └── poses/
            └── garments/
                └── tensors/
        """
        dataset_root = Path(dataset_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Create output structure
        (output_root / 'humans' / 'keypoints').mkdir(parents=True, exist_ok=True)
        (output_root / 'humans' / 'parsing').mkdir(parents=True, exist_ok=True)
        (output_root / 'humans' / 'masks').mkdir(parents=True, exist_ok=True)
        (output_root / 'humans' / 'poses').mkdir(parents=True, exist_ok=True)
        (output_root / 'garments' / 'tensors').mkdir(parents=True, exist_ok=True)

        # Process images
        image_dir = dataset_root / 'image'
        cloth_dir = dataset_root / 'cloth'

        if not image_dir.exists() or not cloth_dir.exists():
            raise FileNotFoundError("VITON-HD structure not found")

        for human_img_path in sorted(image_dir.glob('*.jpg')):
            image_id = human_img_path.stem
            cloth_img_path = cloth_dir / f"{image_id}.jpg"

            if not cloth_img_path.exists():
                print(f"Skipping {image_id}: cloth image not found")
                continue

            try:
                output = self.extractor.process(
                    human_img_path,
                    cloth_img_path,
                    category='upper_body',
                    extract_tensors=True,
                    device=self.device
                )

                # Save features
                self._save_features(output, image_id, output_root, 'viton-hd')
                print(f"Processed: {image_id}")

            except Exception as e:
                print(f"Error processing {image_id}: {e}")

    def preprocess_dresscode(
        self,
        dataset_root: str,
        output_root: str,
        categories: list = None
    ):
        """
        Preprocess DressCode dataset.

        Args:
            dataset_root: Path to DressCode directory
            output_root: Path to save preprocessed features
            categories: List of categories ('upper_body', 'lower_body', 'dresses')
        """
        if categories is None:
            categories = ['upper_body', 'lower_body', 'dresses']

        dataset_root = Path(dataset_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for category in categories:
            category_root = dataset_root / category
            if not category_root.exists():
                print(f"Category not found: {category}")
                continue

            # Create output structure
            (output_root / category / 'humans' / 'keypoints').mkdir(parents=True, exist_ok=True)
            (output_root / category / 'humans' / 'parsing').mkdir(parents=True, exist_ok=True)
            (output_root / category / 'humans' / 'masks').mkdir(parents=True, exist_ok=True)
            (output_root / category / 'humans' / 'poses').mkdir(parents=True, exist_ok=True)
            (output_root / category / 'garments' / 'tensors').mkdir(parents=True, exist_ok=True)

            image_dir = category_root / 'images'
            if not image_dir.exists():
                print(f"Images directory not found for {category}")
                continue

            for human_img_path in sorted(image_dir.glob('*.jpg')):
                image_id = human_img_path.stem

                try:
                    output = self.extractor.process(
                        human_img_path,
                        human_img_path,  # Use same image for feature extraction
                        category=category,
                        extract_tensors=True,
                        device=self.device
                    )

                    self._save_features(output, image_id, output_root / category, 'dresscode')
                    print(f"Processed {category}/{image_id}")

                except Exception as e:
                    print(f"Error processing {category}/{image_id}: {e}")

    def _save_features(
        self,
        output: PreprocessingOutput,
        image_id: str,
        output_root: Path,
        dataset_type: str
    ):
        """Save preprocessed features to disk"""
        output_root = Path(output_root)

        # Save images
        output.human_image.save(output_root / 'humans' / f"{image_id}.jpg")
        output.pose_image.save(output_root / 'humans' / 'poses' / f"{image_id}.jpg")
        output.inpaint_mask.save(output_root / 'humans' / 'masks' / f"{image_id}.jpg")
        output.garment_image.save(output_root / 'garments' / f"{image_id}.jpg")

        # Save keypoints as JSON
        with open(output_root / 'humans' / 'keypoints' / f"{image_id}.json", 'w') as f:
            json.dump(output.human_keypoints, f, indent=2)

        # Save parsing as numpy
        np.save(
            output_root / 'humans' / 'parsing' / f"{image_id}.npy",
            np.array(output.human_parsing)
        )

        # Save tensors if available
        if output.garment_tensor is not None:
            torch.save(
                output.garment_tensor,
                output_root / 'garments' / 'tensors' / f"{image_id}.pt"
            )


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess virtual try-on datasets")
    parser.add_argument('--dataset', type=str, choices=['viton-hd', 'dresscode'])
    parser.add_argument('--input', type=str, required=True, help='Input dataset root')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device string')

    args = parser.parse_args()

    preprocessor = DatasetPreprocessor(gpu_id=args.gpu, device=args.device)

    if args.dataset == 'viton-hd':
        preprocessor.preprocess_viton_hd(args.input, args.output)
    elif args.dataset == 'dresscode':
        preprocessor.preprocess_dresscode(args.input, args.output)


if __name__ == '__main__':
    main()
