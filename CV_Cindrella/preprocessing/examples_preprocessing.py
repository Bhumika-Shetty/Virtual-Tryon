"""
Examples for using the preprocessing pipeline.

Shows:
1. Single image preprocessing
2. Dataset preprocessing
3. Feature extraction and visualization
4. Custom pipeline workflows

IMPORTANT: Run these examples from IDM-VTON root directory:
    cd /path/to/IDM-VTON
    python preprocessing/examples_preprocessing.py
"""

import sys
from pathlib import Path
import torch
from PIL import Image

# Add parent directory to path if running directly
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from preprocessing module
from preprocessing.preprocessing_pipeline import FeatureExtractor, DatasetPreprocessor
from preprocessing.dataset_utils import (
    VITONHDDataset, DressCodeDataset, DatasetValidator,
    get_dataset_loader
)


# =============================================================================
# Example 1: Single Image Preprocessing
# =============================================================================

def example_single_image():
    """Process a single human-garment pair"""

    # Initialize feature extractor
    extractor = FeatureExtractor(gpu_id=0, device='cuda:0')

    # Process images
    output = extractor.process(
        human_image='path/to/human.jpg',
        garment_image='path/to/garment.jpg',
        category='upper_body',
        use_auto_mask=True,
        extract_tensors=True,
        device='cuda:0'
    )

    # Access extracted features
    print(f"Human image shape: {output.human_image.size}")
    print(f"Keypoints: {output.human_keypoints}")
    print(f"Mask shape: {output.inpaint_mask.size}")
    print(f"Pose image shape: {output.pose_image.size}")
    print(f"Garment tensor shape: {output.garment_tensor.shape}")

    # Save outputs
    output.human_image.save('output_human.jpg')
    output.pose_image.save('output_pose.jpg')
    output.inpaint_mask.save('output_mask.jpg')
    output.garment_image.save('output_garment.jpg')


# =============================================================================
# Example 2: Batch Dataset Preprocessing
# =============================================================================

def example_dataset_preprocessing():
    """Preprocess entire VITON-HD or DressCode dataset"""

    preprocessor = DatasetPreprocessor(gpu_id=0, device='cuda:0')

    # Option A: VITON-HD
    print("Preprocessing VITON-HD dataset...")
    preprocessor.preprocess_viton_hd(
        dataset_root='/path/to/VITON-HD',
        output_root='/path/to/preprocessed/VITON-HD'
    )

    # Option B: DressCode
    print("Preprocessing DressCode dataset...")
    preprocessor.preprocess_dresscode(
        dataset_root='/path/to/DressCode',
        output_root='/path/to/preprocessed/DressCode',
        categories=['upper_body', 'lower_body', 'dresses']
    )


# =============================================================================
# Example 3: Dataset Validation
# =============================================================================

def example_validation():
    """Validate dataset structure"""

    validator = DatasetValidator()

    # Validate VITON-HD
    print("Validating VITON-HD...")
    viton_results = validator.validate_viton_hd('/path/to/preprocessed/VITON-HD')
    print(f"Valid: {viton_results['valid']}")
    print(f"Total images: {viton_results['total_images']}")
    if viton_results['missing_dirs']:
        print(f"Missing directories: {viton_results['missing_dirs']}")

    # Validate DressCode
    print("\nValidating DressCode...")
    dresscode_results = validator.validate_dresscode('/path/to/preprocessed/DressCode')
    for category, cat_info in dresscode_results['categories'].items():
        print(f"{category}: {cat_info['total_images']} images")


# =============================================================================
# Example 4: Loading and Iterating Through Dataset
# =============================================================================

def example_dataset_loading():
    """Load preprocessed dataset and iterate through batches"""

    # Load VITON-HD dataset
    print("Loading VITON-HD dataset...")
    viton_dataset = VITONHDDataset(
        root_dir='/path/to/preprocessed/VITON-HD',
        load_tensors=True,
        cache_images=False
    )

    print(f"Dataset size: {len(viton_dataset)}")

    # Access single item
    item = viton_dataset[0]
    print(f"\nItem 0:")
    print(f"  Image ID: {item.image_id}")
    print(f"  Human shape: {item.human_image.size}")
    print(f"  Garment shape: {item.garment_image.size}")
    print(f"  Keypoints: {item.keypoints.keys()}")

    # Get batch
    batch = viton_dataset.get_batch([0, 1, 2, 3])
    print(f"\nBatch shapes:")
    print(f"  Human: {batch['human'].shape}")
    print(f"  Garment: {batch['garment'].shape}")
    print(f"  Pose: {batch['pose'].shape}")
    print(f"  Mask: {batch['mask'].shape}")


# =============================================================================
# Example 5: Using DataLoader for Training
# =============================================================================

def example_dataloader():
    """Use DataLoader for training/inference"""

    # Create loader with helper function
    loader = get_dataset_loader(
        dataset_type='dresscode',
        root_dir='/path/to/preprocessed/DressCode',
        category='upper_body',
        batch_size=4,
        load_tensors=True,
        shuffle=True,
        num_workers=4
    )

    # Iterate through batches
    for batch_idx, batch in enumerate(loader):
        human = batch['human']  # Shape: (batch, 3, 1024, 768)
        garment = batch['garment']  # Shape: (batch, 3, 1024, 768)
        pose = batch['pose']  # Shape: (batch, 3, 1024, 768)
        mask = batch['mask']  # Shape: (batch, 1, 1024, 768)
        image_ids = batch['image_ids']

        print(f"Batch {batch_idx}:")
        print(f"  Shapes - Human: {human.shape}, Garment: {garment.shape}")
        print(f"  Image IDs: {image_ids}")

        # Example: Use batch for model inference
        # output = model(human, garment, pose, mask)

        if batch_idx >= 2:  # Just show first 3 batches
            break


# =============================================================================
# Example 6: Custom Preprocessing Workflow
# =============================================================================

def example_custom_workflow():
    """Create a custom preprocessing workflow"""

    extractor = FeatureExtractor(gpu_id=0, device='cuda:0')

    # Load raw images
    human_img = Image.open('path/to/human.jpg')
    garment_img = Image.open('path/to/garment.jpg')

    # Step 1: Extract features separately
    print("Extracting features...")
    keypoints = extractor.extract_keypoints(human_img)
    parsing, face_mask = extractor.extract_parsing(human_img)

    # Step 2: Generate mask for different categories
    categories = ['upper_body', 'lower_body', 'dresses']
    masks = {}

    for category in categories:
        try:
            mask, mask_gray = extractor.generate_mask(
                human_img, parsing, keypoints, category=category
            )
            masks[category] = (mask, mask_gray)
            print(f"  {category}: ✓")
        except Exception as e:
            print(f"  {category}: ✗ ({e})")

    # Step 3: Extract pose visualization
    pose_img = extractor.extract_pose_image(human_img, device='cuda')

    # Step 4: Prepare tensors
    garment_tensor = extractor.prepare_garment_tensor(garment_img)
    pose_tensor = extractor.prepare_pose_tensor(pose_img)

    print(f"\nExtracted tensors:")
    print(f"  Garment: {garment_tensor.shape}")
    print(f"  Pose: {pose_tensor.shape}")


# =============================================================================
# Example 7: Batch Processing with Custom Output
# =============================================================================

def example_batch_processing():
    """Process multiple images with custom handling"""

    from preprocessing.preprocessing_pipeline import DatasetPreprocessor
    from pathlib import Path
    import json

    preprocessor = DatasetPreprocessor(gpu_id=0, device='cuda:0')
    extractor = preprocessor.extractor

    # List of image pairs
    image_pairs = [
        ('human1.jpg', 'garment1.jpg'),
        ('human2.jpg', 'garment2.jpg'),
        ('human3.jpg', 'garment3.jpg'),
    ]

    # Process with custom handling
    results = []

    for human_path, garment_path in image_pairs:
        try:
            output = extractor.process(
                human_image=human_path,
                garment_image=garment_path,
                category='upper_body',
                extract_tensors=True,
                device='cuda:0'
            )

            # Custom handling
            result = {
                'image_pair': (human_path, garment_path),
                'keypoints_count': len(output.human_keypoints.get('pose_keypoints_2d', [])),
                'mask_size': output.inpaint_mask.size,
                'status': 'success'
            }
            results.append(result)

            # Save outputs
            output_dir = Path('output') / output.human_keypoints.get('image_id', 'unknown')
            output_dir.mkdir(parents=True, exist_ok=True)
            output.human_image.save(output_dir / 'human.jpg')
            output.pose_image.save(output_dir / 'pose.jpg')

        except Exception as e:
            result = {
                'image_pair': (human_path, garment_path),
                'status': 'error',
                'error': str(e)
            }
            results.append(result)

    # Save summary
    with open('processing_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(image_pairs)} pairs")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")


if __name__ == '__main__':
    print("=" * 70)
    print("IDM-VTON Preprocessing Pipeline Examples")
    print("=" * 70)

    print("\nUncomment the example you want to run:")
    print("1. example_single_image() - Single pair preprocessing")
    print("2. example_dataset_preprocessing() - Batch dataset preprocessing")
    print("3. example_validation() - Validate dataset structure")
    print("4. example_dataset_loading() - Load and iterate dataset")
    print("5. example_dataloader() - Use PyTorch DataLoader")
    print("6. example_custom_workflow() - Custom preprocessing steps")
    print("7. example_batch_processing() - Batch with custom handling")

    # Run example (uncomment one):
    # example_single_image()
    # example_dataset_preprocessing()
    # example_validation()
    # example_dataset_loading()
    # example_dataloader()
    # example_custom_workflow()
    # example_batch_processing()
