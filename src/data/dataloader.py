"""
DataLoader utilities for object detection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .coco_dataset import COCODataset
from .transforms import (
    Compose,
    Resize,
    RandomFlip,
    RandomHSV,
    Normalize,
    Letterbox,
)


def collate_fn(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Dictionary of batched tensors
    """
    images = []
    targets = []
    
    for image, target in batch:
        # Convert image to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        images.append(image)
        
        # Prepare target
        target_dict = {
            "boxes": torch.from_numpy(target["boxes"]) if isinstance(target["boxes"], np.ndarray) else target["boxes"],
            "labels": torch.from_numpy(target["labels"]) if isinstance(target["labels"], np.ndarray) else target["labels"],
        }
        if "image_id" in target:
            target_dict["image_id"] = target["image_id"]
        targets.append(target_dict)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return {
        "images": images,
        "targets": targets,
    }


class InfiniteSampler(Sampler):
    """
    Infinite sampler for continuous training.
    """
    
    def __init__(self, data_size: int, shuffle: bool = True):
        """
        Initialize infinite sampler.
        
        Args:
            data_size: Size of the dataset
            shuffle: Whether to shuffle
        """
        self.data_size = data_size
        self.shuffle = shuffle
    
    def __iter__(self):
        """Generate infinite indices."""
        indices = list(range(self.data_size))
        while True:
            if self.shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                yield idx
    
    def __len__(self):
        return self.data_size


class BatchSampler:
    """
    Batch sampler that yields batches from infinite sampler.
    """
    
    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = True,
    ):
        """
        Initialize batch sampler.
        
        Args:
            sampler: Base sampler
            batch_size: Batch size
            drop_last: Drop last incomplete batch
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        """Generate batches."""
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
    
    def __len__(self):
        return len(self.sampler) // self.batch_size


def build_transforms(
    image_size: int = 640,
    is_train: bool = True,
    use_mosaic: bool = True,
    use_mixup: bool = True,
) -> Compose:
    """
    Build data transforms.
    
    Args:
        image_size: Target image size
        is_train: Whether for training
        use_mosaic: Use mosaic augmentation
        use_mixup: Use mixup augmentation
        
    Returns:
        Composed transforms
    """
    transforms = []
    
    if is_train:
        transforms.extend([
            RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4),
            RandomFlip(prob=0.5, direction="horizontal"),
        ])
    
    transforms.extend([
        Letterbox(target_size=image_size),
        Normalize(),
    ])
    
    return Compose(transforms)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    use_infinite: bool = False,
) -> DataLoader:
    """
    Build DataLoader for object detection.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        use_infinite: Use infinite sampler
        
    Returns:
        DataLoader instance
    """
    if use_infinite:
        sampler = InfiniteSampler(len(dataset), shuffle=shuffle)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


def create_dataloaders(
    train_root: str,
    train_ann: str,
    val_root: str,
    val_ann: str,
    image_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_root: Training images root directory
        train_ann: Training annotation file path
        val_root: Validation images root directory
        val_ann: Validation annotation file path
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_transforms = build_transforms(image_size, is_train=True)
    val_transforms = build_transforms(image_size, is_train=False)
    
    train_dataset = COCODataset(
        root=train_root,
        annotation_file=train_ann,
        transforms=train_transforms,
    )
    
    val_dataset = COCODataset(
        root=val_root,
        annotation_file=val_ann,
        transforms=val_transforms,
    )
    
    # Create dataloaders
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    
    return train_loader, val_loader
