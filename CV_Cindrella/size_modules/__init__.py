"""
Size-Aware Modules for Cinderella Virtual Try-On
Author: Cinderella Team
Date: 2025-11-30

This package contains the size-conditioning modules for size-aware virtual try-on:
- size_annotation.py: Size ratio extraction from OpenPose keypoints
- size_encoder.py: MLP encoder mapping size ratios to embeddings
- size_controller.py: CNN controller generating spatial size guidance maps
"""

from .size_annotation import SizeAnnotator, compute_size_ratio, get_size_label, get_size_label_id
from .size_encoder import SizeEncoder
from .size_controller import SizeController, SimpleSizeController

__all__ = [
    'SizeAnnotator',
    'compute_size_ratio',
    'get_size_label',
    'get_size_label_id',
    'SizeEncoder',
    'SizeController',
    'SimpleSizeController',
]
