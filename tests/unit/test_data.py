"""
Unit tests for data components.

Tests cover:
- Data transforms
- Dataset classes
- DataLoader utilities
"""

import pytest
import torch
import numpy as np
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.transforms import (
    Compose, ToTensor, Normalize, Resize,
    RandomHorizontalFlip, TrainTransform, ValTransform
)
from src.data.datasets import CocoDetectionMock
from src.data.dataloader import collate_fn, build_dataloader


class TestTransforms:
    """Tests for data transforms."""
    
    def test_to_tensor(self):
        """Test ToTensor transform."""
        transform = ToTensor()
        
        image = Image.new('RGB', (100, 100), color='red')
        target = {'boxes': torch.tensor([[10, 10, 50, 50]])}
        
        image_tensor, target_out = transform(image, target)
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 100, 100)
        assert image_tensor.dtype == torch.float32
        
    def test_normalize(self):
        """Test Normalize transform."""
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        image = torch.rand(3, 100, 100)
        target = {'boxes': torch.tensor([[10, 10, 50, 50]])}
        
        image_out, target_out = transform(image, target)
        
        assert image_out.shape == image.shape
        # Normalized values should have different statistics
        
    def test_resize(self):
        """Test Resize transform."""
        transform = Resize((320, 320))
        
        image = Image.new('RGB', (640, 480))
        target = {
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        }
        
        image_out, target_out = transform(image, target)
        
        assert image_out.size == (320, 320)
        # Boxes should be scaled
        assert target_out['boxes'][0, 0] == 100 * (320 / 640)
        
    def test_random_horizontal_flip(self):
        """Test RandomHorizontalFlip transform."""
        transform = RandomHorizontalFlip(p=1.0)  # Always flip
        
        image = Image.new('RGB', (100, 100))
        target = {
            'boxes': torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        }
        
        image_out, target_out = transform(image, target)
        
        # Box should be flipped: x1 = width - x2, x2 = width - x1
        assert target_out['boxes'][0, 0] == 70  # 100 - 30
        assert target_out['boxes'][0, 2] == 90  # 100 - 10
        
    def test_compose(self):
        """Test Compose transform."""
        transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.new('RGB', (640, 480))
        target = {'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)}
        
        image_out, target_out = transforms(image, target)
        
        assert isinstance(image_out, torch.Tensor)
        assert image_out.shape == (3, 224, 224)
        
    def test_train_transform(self):
        """Test TrainTransform."""
        transform = TrainTransform(size=(320, 320))
        
        image = Image.new('RGB', (640, 480))
        target = {'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)}
        
        image_out, target_out = transform(image, target)
        
        assert isinstance(image_out, torch.Tensor)
        assert image_out.shape == (3, 320, 320)
        
    def test_val_transform(self):
        """Test ValTransform."""
        transform = ValTransform(size=(320, 320))
        
        image = Image.new('RGB', (640, 480))
        target = {'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)}
        
        image_out, target_out = transform(image, target)
        
        assert isinstance(image_out, torch.Tensor)
        assert image_out.shape == (3, 320, 320)


class TestMockDataset:
    """Tests for mock dataset."""
    
    def test_dataset_length(self):
        """Test dataset length."""
        dataset = CocoDetectionMock(num_samples=50)
        
        assert len(dataset) == 50
        
    def test_getitem(self):
        """Test getting item from dataset."""
        dataset = CocoDetectionMock(
            num_samples=10,
            image_size=(320, 320),
            num_classes=80
        )
        
        image, target = dataset[0]
        
        assert isinstance(image, (Image.Image, torch.Tensor))
        assert 'boxes' in target
        assert 'labels' in target
        
    def test_with_transforms(self):
        """Test dataset with transforms."""
        transform = ValTransform(size=(224, 224))
        dataset = CocoDetectionMock(
            num_samples=10,
            transforms=transform
        )
        
        image, target = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)


class TestCollateFn:
    """Tests for collate function."""
    
    def test_collate_batch(self):
        """Test collating a batch."""
        batch = [
            (torch.randn(3, 224, 224), {'boxes': torch.tensor([[1, 1, 2, 2]])}),
            (torch.randn(3, 224, 224), {'boxes': torch.tensor([[3, 3, 4, 4]])})
        ]
        
        images, targets = collate_fn(batch)
        
        assert images.shape == (2, 3, 224, 224)
        assert len(targets) == 2
        
    def test_variable_num_boxes(self):
        """Test collating with variable number of boxes."""
        batch = [
            (torch.randn(3, 224, 224), {'boxes': torch.tensor([[1, 1, 2, 2]])}),
            (torch.randn(3, 224, 224), {'boxes': torch.tensor([[1, 1, 2, 2], [3, 3, 4, 4]])})
        ]
        
        images, targets = collate_fn(batch)
        
        assert images.shape[0] == 2
        assert len(targets[0]['boxes']) == 1
        assert len(targets[1]['boxes']) == 2


class TestBuildDataloader:
    """Tests for dataloader building."""
    
    def test_build_dataloader(self):
        """Test building dataloader."""
        dataset = CocoDetectionMock(num_samples=32, transforms=ValTransform())
        
        dataloader = build_dataloader(
            dataset,
            batch_size=4,
            num_workers=0,
            shuffle=False
        )
        
        assert len(dataloader) == 8  # 32 / 4
        
        # Get first batch
        images, targets = next(iter(dataloader))
        
        assert images.shape[0] == 4
        assert len(targets) == 4
