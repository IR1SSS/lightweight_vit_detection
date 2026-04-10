"""
Unit tests for model modules.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn


class TestAttention:
    """Test attention modules."""
    
    def test_multihead_attention(self):
        """Test standard multi-head attention."""
        from src.models.backbone.attention import MultiHeadAttention
        
        attn = MultiHeadAttention(dim=128, num_heads=4)
        x = torch.randn(2, 196, 128)
        
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_linear_attention(self):
        """Test linear attention."""
        from src.models.backbone.attention import LinearAttention
        
        attn = LinearAttention(dim=128, num_heads=4)
        x = torch.randn(2, 196, 128)
        
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_pool_attention(self):
        """Test pooling attention."""
        from src.models.backbone.attention import PoolAttention
        
        attn = PoolAttention(dim=128, num_heads=4)
        x = torch.randn(2, 196, 128)
        
        output = attn(x, H=14, W=14)
        
        assert output.shape == x.shape


class TestBackbone:
    """Test backbone modules."""
    
    def test_mobilevit_small(self):
        """Test MobileViT small model."""
        from src.models.backbone import MobileViT
        
        model = MobileViT(model_size="small")
        x = torch.randn(2, 3, 320, 320)
        
        features = model(x)
        
        assert isinstance(features, list)
        assert len(features) >= 3  # At least 3 feature levels
    
    def test_efficientformer(self):
        """Test EfficientFormer model."""
        from src.models.backbone import EfficientFormerV2
        
        model = EfficientFormerV2(
            layers=[3, 3, 9, 6],  # S1 config
            embed_dims=[32, 48, 120, 224],
            downsamples=[True, True, True, True],
            vit_num=2,
            fork_feat=True,
        )
        x = torch.randn(2, 3, 320, 320)
        
        features = model(x)
        
        assert isinstance(features, list)
        assert len(features) == 4  # Four stages


class TestNeck:
    """Test neck modules."""
    
    def test_fpn(self):
        """Test FPN."""
        from src.models.neck import FPN
        
        in_channels = [64, 128, 256]
        fpn = FPN(in_channels, out_channels=128)
        
        # Create fake feature maps
        features = [
            torch.randn(2, 64, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 256, 10, 10),
        ]
        
        outputs = fpn(features)
        
        assert len(outputs) == 3
        for out in outputs:
            assert out.shape[1] == 128
    
    def test_pafpn(self):
        """Test PAFPN."""
        from src.models.neck import PAFPN
        
        in_channels = [64, 128, 256]
        pafpn = PAFPN(in_channels, out_channels=128)
        
        features = [
            torch.randn(2, 64, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 256, 10, 10),
        ]
        
        outputs = pafpn(features)
        
        assert len(outputs) == 3


class TestHead:
    """Test detection head modules."""
    
    def test_detection_head(self):
        """Test detection head."""
        from src.models.head import DetectionHead
        
        head = DetectionHead(
            in_channels=128,
            num_classes=80,
            num_anchors=3,
        )
        
        features = [
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]
        
        cls_preds, reg_preds, obj_preds = head(features)
        
        assert len(cls_preds) == 3
        assert len(reg_preds) == 3
        assert len(obj_preds) == 3


class TestDetector:
    """Test complete detector."""
    
    def test_vit_detector_forward(self):
        """Test ViT detector forward pass."""
        from src.models.detector import ViTDetector
        
        model = ViTDetector(
            backbone_name="mobilevit_small",
            num_classes=80,
        )
        model.eval()
        
        x = torch.randn(1, 3, 320, 320)
        
        with torch.no_grad():
            cls_pred, reg_pred, obj_pred = model(x)
        
        # Check outputs exist
        assert cls_pred is not None
        assert reg_pred is not None
        assert obj_pred is not None
    
    def test_detector_output_shape(self):
        """Test detector output shapes."""
        from src.models.detector import ViTDetector
        
        model = ViTDetector(
            backbone_name="mobilevit_small",
            num_classes=80,
        )
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 320, 320)
        
        with torch.no_grad():
            cls_pred, reg_pred, obj_pred = model(x)
        
        # Check batch dimension
        assert cls_pred.shape[0] == batch_size


class TestModelUtils:
    """Test model utility functions."""
    
    def test_param_count(self):
        """Test parameter counting."""
        from src.models.backbone import MobileViT
        
        model = MobileViT(model_size="small")
        num_params = sum(p.numel() for p in model.parameters())
        
        # MobileViT-S should have around 5-6M parameters
        assert 3_000_000 < num_params < 10_000_000
    
    def test_model_device(self):
        """Test model device placement."""
        from src.models.backbone import MobileViT
        
        model = MobileViT(model_size="small")
        
        # Should be on CPU initially
        assert next(model.parameters()).device.type == "cpu"
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
