"""
DataLoader Utilities for Lightweight ViT Detection System.

This module provides data loader construction and collation functions
for object detection training and evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for object detection.
    
    Handles variable number of objects per image by keeping
    targets as a list of dictionaries.
    
    Args:
        batch: List of (image, target) tuples.
        
    Returns:
        Tuple of (batched images, list of targets).
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
        
    # Stack images into batch
    images = torch.stack(images, dim=0)
    
    return images, targets


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
    distributed: bool = False,
    drop_last: bool = False,
    pin_memory: bool = False
) -> DataLoader:
    """
    Build DataLoader for object detection.
    
    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle data.
        distributed: Whether to use distributed sampler.
        drop_last: Whether to drop the last incomplete batch.
        pin_memory: Whether to pin memory for faster GPU transfer.
        
    Returns:
        DataLoader instance.
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        # When using distributed sampler, shuffle should be False for DataLoader
        dataloader_shuffle = False
    else:
        sampler = None
        dataloader_shuffle = shuffle
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
    
    return dataloader


def build_train_dataloader(
    dataset: Dataset,
    config: Dict[str, Any],
    distributed: bool = False
) -> DataLoader:
    """
    Build training DataLoader from configuration.
    
    Args:
        dataset: Training dataset.
        config: Training configuration dictionary.
        distributed: Whether to use distributed training.
        
    Returns:
        Training DataLoader.
    """
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 0)
    
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        distributed=distributed,
        drop_last=True,
        pin_memory=False
    )


def build_val_dataloader(
    dataset: Dataset,
    config: Dict[str, Any],
    distributed: bool = False
) -> DataLoader:
    """
    Build validation DataLoader from configuration.
    
    Args:
        dataset: Validation dataset.
        config: Training configuration dictionary.
        distributed: Whether to use distributed training.
        
    Returns:
        Validation DataLoader.
    """
    # Use smaller batch size for validation to avoid OOM
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 0)
    
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        distributed=distributed,
        drop_last=False,
        pin_memory=False
    )


class InfiniteDataLoader:
    """
    DataLoader wrapper that yields batches infinitely.
    
    Useful for iteration-based training instead of epoch-based.
    """
    
    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initialize infinite data loader.
        
        Args:
            dataloader: Base DataLoader to wrap.
        """
        self.dataloader = dataloader
        self._iterator = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
            
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
            
        return batch
        
    def __len__(self):
        return len(self.dataloader)


class PrefetchLoader:
    """
    DataLoader wrapper with CUDA prefetching for faster data loading.
    
    Transfers data to GPU asynchronously while model is computing.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device = torch.device('cuda')
    ) -> None:
        """
        Initialize prefetch loader.
        
        Args:
            dataloader: Base DataLoader to wrap.
            device: Device to prefetch data to.
        """
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self):
        return self._prefetch_generator()
        
    def _prefetch_generator(self):
        """Generator that prefetches batches to GPU."""
        loader_iter = iter(self.dataloader)
        
        # Load first batch
        try:
            batch = next(loader_iter)
        except StopIteration:
            return
            
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                batch = self._to_device(batch)
                
        while True:
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
                
            current_batch = batch
            
            # Try to load next batch
            try:
                batch = next(loader_iter)
            except StopIteration:
                yield current_batch
                return
                
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    batch = self._to_device(batch)
            else:
                batch = self._to_device(batch)
                
            yield current_batch
            
    def _to_device(
        self, 
        batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Transfer batch to device."""
        images, targets = batch
        images = images.to(self.device, non_blocking=True)
        targets = [
            {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
        return images, targets
        
    def __len__(self):
        return len(self.dataloader)
