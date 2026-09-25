"""Modern, reproducible utilities for digital histopathology workflows."""

from .augmentation import random_transform, transform_patch
from .config import PatchExtractionConfig
from .mask import TissueMask, iter_tissue_coordinates, select_tissue_coordinates
from .sampling import allocate_by_weights, reservoir_sample, train_test_split_indices

__all__ = [
    "PatchExtractionConfig",
    "TissueMask",
    "allocate_by_weights",
    "iter_tissue_coordinates",
    "random_transform",
    "reservoir_sample",
    "select_tissue_coordinates",
    "train_test_split_indices",
    "transform_patch",
]
