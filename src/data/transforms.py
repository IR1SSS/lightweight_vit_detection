"""
Data Transforms for Lightweight ViT Detection System.

This module provides image and target transforms for
object detection training and validation.
"""

import random
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List) -> None:
        """
        Initialize compose transform.
        
        Args:
            transforms: List of transforms to compose.
        """
        self.transforms = transforms
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply all transforms sequentially."""
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
        
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n    " + repr(t)
        format_string += "\n)"
        return format_string


class ToTensor:
    """Convert PIL Image to tensor."""
    
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert image to tensor."""
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image tensor with mean and std."""
    
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> None:
        """
        Initialize normalize transform.
        
        Args:
            mean: Channel means for normalization.
            std: Channel standard deviations for normalization.
        """
        self.mean = mean
        self.std = std
        
    def __call__(
        self, 
        image: torch.Tensor, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Normalize image tensor."""
        image = F.normalize(image, self.mean, self.std)
        return image, target


class Resize:
    """Resize image and adjust bounding boxes."""
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        max_size: Optional[int] = None
    ) -> None:
        """
        Initialize resize transform.
        
        Args:
            size: Target size. If int, resize shorter edge to this size.
                  If tuple, resize to exact (height, width).
            max_size: Maximum size of longer edge.
        """
        self.size = size
        self.max_size = max_size
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Resize image and adjust boxes."""
        orig_w, orig_h = image.size
        
        # Calculate new size
        if isinstance(self.size, int):
            # Resize shorter edge
            if orig_w < orig_h:
                new_w = self.size
                new_h = int(orig_h * self.size / orig_w)
            else:
                new_h = self.size
                new_w = int(orig_w * self.size / orig_h)
                
            # Apply max_size constraint
            if self.max_size is not None:
                if new_h > self.max_size:
                    new_w = int(new_w * self.max_size / new_h)
                    new_h = self.max_size
                if new_w > self.max_size:
                    new_h = int(new_h * self.max_size / new_w)
                    new_w = self.max_size
        else:
            new_h, new_w = self.size
            
        # Resize image
        image = F.resize(image, (new_h, new_w))
        
        # Adjust bounding boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
            
        # Update size info
        target['size'] = torch.tensor([new_h, new_w])
        
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and boxes horizontally."""
    
    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize random horizontal flip.
        
        Args:
            p: Probability of flipping.
        """
        self.p = p
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply random horizontal flip."""
        if random.random() < self.p:
            image = F.hflip(image)
            
            if 'boxes' in target and len(target['boxes']) > 0:
                w = image.size[0]
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
                
        return image, target


class RandomCrop:
    """Randomly crop image and adjust boxes."""
    
    def __init__(self, size: Tuple[int, int]) -> None:
        """
        Initialize random crop.
        
        Args:
            size: Crop size (height, width).
        """
        self.size = size
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply random crop."""
        w, h = image.size
        th, tw = self.size
        
        if w == tw and h == th:
            return image, target
            
        # Random crop position
        top = random.randint(0, max(0, h - th))
        left = random.randint(0, max(0, w - tw))
        
        # Crop image
        image = F.crop(image, top, left, th, tw)
        
        # Adjust boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            
            # Clip to crop region
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, tw)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, th)
            
            # Remove boxes with zero area
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            target['boxes'] = boxes[keep]
            target['labels'] = target['labels'][keep]
            if 'area' in target:
                target['area'] = target['area'][keep]
            if 'iscrowd' in target:
                target['iscrowd'] = target['iscrowd'][keep]
                
        return image, target


class ColorJitter:
    """Randomly change brightness, contrast, saturation, and hue."""
    
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1
    ) -> None:
        """
        Initialize color jitter.
        
        Args:
            brightness: How much to jitter brightness.
            contrast: How much to jitter contrast.
            saturation: How much to jitter saturation.
            hue: How much to jitter hue.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply color jitter."""
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = F.adjust_brightness(image, factor)
            
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            image = F.adjust_contrast(image, factor)
            
        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            image = F.adjust_saturation(image, factor)
            
        if self.hue > 0:
            factor = random.uniform(-self.hue, self.hue)
            image = F.adjust_hue(image, factor)
            
        return image, target


class RandomScale:
    """Randomly scale image and boxes."""
    
    def __init__(
        self,
        min_scale: float = 0.5,
        max_scale: float = 1.5
    ) -> None:
        """
        Initialize random scale.
        
        Args:
            min_scale: Minimum scale factor.
            max_scale: Maximum scale factor.
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply random scale."""
        scale = random.uniform(self.min_scale, self.max_scale)
        
        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = F.resize(image, (new_h, new_w))
        
        if 'boxes' in target and len(target['boxes']) > 0:
            target['boxes'] = target['boxes'] * scale
            
        return image, target


class TrainTransform:
    """
    Training transforms for object detection.
    
    Includes:
    - Random horizontal flip
    - Color jitter
    - Resize
    - ToTensor
    - Normalize
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = (640, 640),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> None:
        """
        Initialize training transforms.
        
        Args:
            size: Target image size.
            mean: Normalization mean.
            std: Normalization std.
        """
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            Resize(size),
            ToTensor(),
            Normalize(mean, std)
        ])
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply training transforms."""
        return self.transforms(image, target)


class ValTransform:
    """
    Validation transforms for object detection.
    
    Includes:
    - Resize
    - ToTensor
    - Normalize
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = (640, 640),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> None:
        """
        Initialize validation transforms.
        
        Args:
            size: Target image size.
            mean: Normalization mean.
            std: Normalization std.
        """
        self.transforms = Compose([
            Resize(size),
            ToTensor(),
            Normalize(mean, std)
        ])
        
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply validation transforms."""
        return self.transforms(image, target)
