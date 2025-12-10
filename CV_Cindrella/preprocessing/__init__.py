"""
IDM-VTON Preprocessing Module

This module provides unified preprocessing pipeline for virtual try-on tasks.
It extracts features from human and garment images for the IDM-VTON model.

Usage:
    from preprocessing import FeatureExtractor, DatasetPreprocessor
    from preprocessing import VITONHDDataset, DressCodeDataset, get_dataset_loader

Example:
    extractor = FeatureExtractor(gpu_id=0)
    output = extractor.process('human.jpg', 'garment.jpg')
"""

import sys
from pathlib import Path

# Add parent directories to path for accessing local packages and modules
__file_path__ = Path(__file__).absolute()
PROJECT_ROOT = __file_path__.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'gradio_demo'))

# Import main classes
from .preprocessing_pipeline import FeatureExtractor, DatasetPreprocessor, PreprocessingOutput
from .dataset_utils import (
    VITONHDDataset, DressCodeDataset, DatasetValidator,
    DatasetItem, get_dataset_loader
)

__version__ = '1.0.0'
__all__ = [
    'FeatureExtractor',
    'DatasetPreprocessor',
    'PreprocessingOutput',
    'VITONHDDataset',
    'DressCodeDataset',
    'DatasetValidator',
    'DatasetItem',
    'get_dataset_loader',
]
