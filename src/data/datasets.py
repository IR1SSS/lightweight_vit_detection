"""
Dataset Implementations for Lightweight ViT Detection System.

This module provides dataset classes for object detection,
including COCO format dataset support.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    """
    Base class for object detection datasets.
    
    Provides common functionality for loading images and annotations.
    """
    
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None
    ) -> None:
        """
        Initialize detection dataset.
        
        Args:
            root: Root directory of the dataset.
            transforms: Optional transforms to apply to images.
            target_transforms: Optional transforms to apply to targets.
        """
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        
        self.images: List[str] = []
        self.annotations: List[Dict] = []
        
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError("Subclasses must implement __getitem__")
        
    def _load_image(self, path: str) -> Image.Image:
        """Load image from path."""
        return Image.open(path).convert('RGB')


class COCODataset(DetectionDataset):
    """
    COCO format dataset for object detection.
    
    Loads images and annotations in COCO format, supporting
    bounding box detection tasks.
    
    Expected directory structure:
        root/
        ├── images/
        │   ├── image1.jpg
        │   └── ...
        └── annotations.json
    
    Annotation format (COCO):
        {
            "images": [...],
            "annotations": [...],
            "categories": [...]
        }
    """
    
    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        min_area: float = 0.0,
        remove_empty: bool = True,
        max_samples: int = 0
    ) -> None:
        """
        Initialize COCO dataset.
        
        Args:
            root: Root directory containing images.
            ann_file: Path to COCO format annotation file.
            transforms: Optional transforms to apply.
            min_area: Minimum box area to include.
            remove_empty: Whether to remove images without annotations.
            max_samples: Maximum samples to load (0 = no limit).
        """
        super().__init__(root, transforms)
        
        self.ann_file = ann_file
        self.min_area = min_area
        self.remove_empty = remove_empty
        self.max_samples = max_samples
        
        # Load annotations
        self._load_annotations()
        
    def _load_annotations(self) -> None:
        """Load COCO format annotations."""
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
            
        # Build image id to filename mapping
        self.img_info: Dict[int, Dict] = {}
        for img in coco_data['images']:
            self.img_info[img['id']] = {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }
            
        # Build category id to index mapping
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.categories.keys())}
        self.num_classes = len(self.categories)
        
        # Group annotations by image
        img_to_anns: Dict[int, List] = {img_id: [] for img_id in self.img_info}
        
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in img_to_anns:
                # Filter by area
                if ann.get('area', float('inf')) >= self.min_area:
                    img_to_anns[img_id].append(ann)
                    
        # Build final lists
        for img_id, anns in img_to_anns.items():
            if self.remove_empty and len(anns) == 0:
                continue
                
            self.images.append(img_id)
            self.annotations.append(anns)
            
            # Limit samples for quick testing
            if self.max_samples > 0 and len(self.images) >= self.max_samples:
                break
            
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get image and target at index.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (image tensor, target dictionary).
            Target contains:
                - boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
                - labels: Class labels [N]
                - image_id: Original image ID
                - area: Box areas [N]
                - iscrowd: Crowd flags [N]
        """
        img_id = self.images[idx]
        img_info = self.img_info[img_id]
        anns = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.root, img_info['file_name'])
        image = self._load_image(img_path)
        
        # Parse annotations
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO format: [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
            
        # Convert to tensors
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']])
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target
        
    def get_category_name(self, idx: int) -> str:
        """Get category name from index."""
        for cat_id, cat_idx in self.cat_id_to_idx.items():
            if cat_idx == idx:
                return self.categories[cat_id]
        return "unknown"
        
    @property
    def class_names(self) -> List[str]:
        """Get list of class names in index order."""
        names = [""] * self.num_classes
        for cat_id, idx in self.cat_id_to_idx.items():
            names[idx] = self.categories[cat_id]
        return names


class CocoDetectionMock(DetectionDataset):
    """
    Mock COCO dataset for testing without actual data.
    
    Generates random images and annotations.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (640, 640),
        num_classes: int = 80,
        max_objects: int = 10,
        transforms: Optional[Callable] = None
    ) -> None:
        """
        Initialize mock dataset.
        
        Args:
            num_samples: Number of samples to generate.
            image_size: Size of generated images (H, W).
            num_classes: Number of object classes.
            max_objects: Maximum objects per image.
            transforms: Optional transforms.
        """
        super().__init__("", transforms)
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        self.images = list(range(num_samples))
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate random sample."""
        # Random image
        image = Image.fromarray(
            np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)
        )
        
        # Random annotations
        num_objects = np.random.randint(1, self.max_objects + 1)
        
        boxes = []
        for _ in range(num_objects):
            x1 = np.random.randint(0, self.image_size[1] - 50)
            y1 = np.random.randint(0, self.image_size[0] - 50)
            x2 = x1 + np.random.randint(20, 100)
            y2 = y1 + np.random.randint(20, 100)
            x2 = min(x2, self.image_size[1])
            y2 = min(y2, self.image_size[0])
            boxes.append([x1, y1, x2, y2])
            
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.randint(0, self.num_classes, (num_objects,)),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros(num_objects, dtype=torch.int64)
        }
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target


def build_dataset(config: Dict[str, Any], split: str = 'train') -> DetectionDataset:
    """
    Build dataset from configuration.
    
    Args:
        config: Data configuration dictionary.
        split: Dataset split ('train', 'val', 'test').
        
    Returns:
        Dataset instance.
    """
    import os
    from .transforms import TrainTransform, ValTransform
    from .download import ensure_coco_dataset
    
    dataset_type = config.get('dataset', 'coco')
    data_dir = config.get('data_dir', 'data/coco')
    
    if split == 'train':
        root = config.get('train_path', os.path.join(data_dir, 'train2017'))
        ann_file = config.get('ann_train', os.path.join(data_dir, 'annotations/instances_train2017.json'))
        transforms = TrainTransform()
    else:
        root = config.get('val_path', os.path.join(data_dir, 'val2017'))
        ann_file = config.get('ann_val', os.path.join(data_dir, 'annotations/instances_val2017.json'))
        transforms = ValTransform()
        
    if dataset_type == 'coco':
        # Check if dataset files exist, auto-download if not
        need_download = False
        
        # Check annotation file
        if not os.path.exists(ann_file):
            print(f"COCO annotation file not found: {ann_file}")
            need_download = True
            
        # Check image directory
        if not os.path.exists(root):
            print(f"COCO image directory not found: {root}")
            need_download = True
        elif os.path.isdir(root):
            # Check if directory has images
            num_images = len([f for f in os.listdir(root) if f.endswith(('.jpg', '.jpeg', '.png'))])
            expected_min = 100000 if split == 'train' else 4000
            if num_images < expected_min:
                print(f"COCO {split} images incomplete: found {num_images}, expected ~{expected_min}+")
                need_download = True
                
        if need_download:
            print("Checking and downloading COCO dataset...")
            
            # Determine what needs to be downloaded
            require_train = (split == 'train')
            require_val = (split == 'val' or split == 'test')
            
            # Auto download dataset
            ensure_coco_dataset(
                data_dir=data_dir,
                require_train=require_train,
                require_val=require_val,
                auto_download=True
            )
            
        return COCODataset(
            root=root,
            ann_file=ann_file,
            transforms=transforms,
            max_samples=config.get('max_samples', 0)  # Use full dataset for training
        )
    elif dataset_type == 'mock':
        num_samples = config.get('num_samples', 1000 if split == 'train' else 200)
        return CocoDetectionMock(num_samples=num_samples, transforms=transforms)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
