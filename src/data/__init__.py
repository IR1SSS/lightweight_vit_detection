"""Data module for Lightweight ViT Detection System.

This module provides data loading functionality including:
- COCO dataset implementation
- Data transforms for training and validation
- DataLoader construction utilities
- Automatic COCO dataset download
"""

from .datasets import COCODataset, DetectionDataset, build_dataset
from .transforms import (
    TrainTransform,
    ValTransform,
    Compose,
    Resize,
    RandomHorizontalFlip,
    Normalize,
    ToTensor
)
from .dataloader import build_dataloader, collate_fn
from .download import download_coco_dataset, ensure_coco_dataset, check_coco_dataset

__all__ = [
    'COCODataset',
    'DetectionDataset',
    'build_dataset',
    'TrainTransform',
    'ValTransform',
    'Compose',
    'Resize',
    'RandomHorizontalFlip',
    'Normalize',
    'ToTensor',
    'build_dataloader',
    'collate_fn',
    'download_coco_dataset',
    'ensure_coco_dataset',
    'check_coco_dataset',
]
