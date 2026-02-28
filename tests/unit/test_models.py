"""
Unit tests for model components.

Tests cover:
- MobileViT backbone
- Detection head
- Full detection model
"""

import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.mobilevit import MobileViT, MobileViTBlock, MobileViTBackbone
from src.models.detection_head import SimpleFPN, RetinaHead
from src.models.backbone import build_backbone


class TestMobileViTBlock:
    """Tests for MobileViTBlock."""
    
    def test_forward_shape(self):
        """Test output shape of MobileViTBlock."""
        block = MobileViTBlock(
            in_channels=64,
            out_channels=64,
            dim=96,
            depth=2,
            patch_size=(2, 2)
        )
        
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        
        assert output.shape == (2, 64, 32, 32), f"Expected (2, 64, 32, 32), got {output.shape}"
        
    def test_different_input_sizes(self):
        """Test with different input spatial sizes."""
        block = MobileViTBlock(
            in_channels=32,
            out_channels=32,
            dim=64,
            depth=2
        )
        
        for size in [16, 32, 64]:
            x = torch.randn(1, 32, size, size)
            output = block(x)
            assert output.shape == (1, 32, size, size)


class TestMobileViTBackbone:
    """Tests for MobileViTBackbone."""
    
    def test_forward_output_count(self):
        """Test number of output feature maps."""
        backbone = MobileViTBackbone()
        
        x = torch.randn(2, 3, 224, 224)
        outputs = backbone(x)
        
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"
        
    def test_output_channels(self):
        """Test output channel dimensions."""
        backbone = MobileViTBackbone(
            channels=[16, 32, 48, 64, 80, 96, 384]
        )
        
        x = torch.randn(1, 3, 256, 256)
        outputs = backbone(x)
        
        expected_channels = backbone.out_channels
        for i, (output, expected) in enumerate(zip(outputs, expected_channels)):
            assert output.shape[1] == expected, f"Stage {i}: expected {expected} channels, got {output.shape[1]}"


class TestSimpleFPN:
    """Tests for SimpleFPN."""
    
    def test_forward_shape(self):
        """Test FPN output shapes."""
        fpn = SimpleFPN(
            in_channels=[48, 64, 80, 384],
            out_channels=256,
            num_outs=5
        )
        
        inputs = [
            torch.randn(1, 48, 64, 64),
            torch.randn(1, 64, 32, 32),
            torch.randn(1, 80, 16, 16),
            torch.randn(1, 384, 8, 8)
        ]
        
        outputs = fpn(inputs)
        
        assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"
        
        for output in outputs:
            assert output.shape[1] == 256, f"Expected 256 channels, got {output.shape[1]}"


class TestRetinaHead:
    """Tests for RetinaHead."""
    
    def test_forward_shape(self):
        """Test detection head output shapes."""
        head = RetinaHead(
            num_classes=80,
            in_channels=256,
            num_anchors=9
        )
        
        features = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 256, 20, 20),
            torch.randn(2, 256, 10, 10),
            torch.randn(2, 256, 5, 5)
        ]
        
        cls_scores, bbox_preds = head(features)
        
        assert len(cls_scores) == 5
        assert len(bbox_preds) == 5
        
        # Check classification output
        assert cls_scores[0].shape[1] == 80 * 9  # num_classes * num_anchors
        
        # Check regression output
        assert bbox_preds[0].shape[1] == 4 * 9  # 4 coords * num_anchors


class TestMobileViTModel:
    """Tests for full MobileViT detection model."""
    
    def test_model_creation(self, sample_config):
        """Test model can be created from config."""
        model = MobileViT(sample_config)
        
        assert model is not None
        assert model.backbone is not None
        assert model.neck is not None
        assert model.head is not None
        
    def test_forward_inference(self, sample_config, sample_image):
        """Test forward pass in inference mode."""
        model = MobileViT(sample_config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_image)
            
        assert 'cls_scores' in outputs
        assert 'bbox_preds' in outputs
        
    def test_parameter_count(self, sample_config):
        """Test parameter count is within expected range."""
        model = MobileViT(sample_config)
        
        num_params = sum(p.numel() for p in model.parameters())
        num_params_m = num_params / 1e6
        
        # Should be lightweight (< 50M parameters typically)
        assert num_params_m < 100, f"Model has {num_params_m:.2f}M parameters, expected < 100M"
        
    def test_get_complexity(self, sample_config):
        """Test complexity metrics computation."""
        model = MobileViT(sample_config)
        
        complexity = model.get_complexity()
        
        assert 'params' in complexity
        assert 'model_size' in complexity
        assert complexity['params'] > 0


class TestBuildBackbone:
    """Tests for backbone building function."""
    
    def test_build_mobilevit_backbone(self):
        """Test building MobileViT backbone."""
        config = {
            'name': 'mobilevit_s',
            'mobilevit_block': {
                'dims': [96, 128, 160],
                'depths': [2, 4, 3]
            },
            'mv2_block': {
                'channels': [16, 32, 48, 64, 80, 96]
            }
        }
        
        backbone = build_backbone(config)
        
        assert backbone is not None
        assert hasattr(backbone, 'out_channels')
