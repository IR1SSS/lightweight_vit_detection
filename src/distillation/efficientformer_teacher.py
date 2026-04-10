"""
EfficientFormerV2 Teacher Model Wrapper for Knowledge Distillation.

This module provides a wrapper for the EfficientFormerV2-S1 model
to be used as a teacher in knowledge distillation training.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from pathlib import Path


class EfficientFormerV2Teacher(nn.Module):
    """
    EfficientFormerV2-S1 Teacher Model for Knowledge Distillation.
    
    Loads pretrained weights and extracts multi-scale features
    for distillation to student models.
    """
    
    # EfficientFormerV2-S1 configuration
    S1_CONFIG = {
        'layers': [3, 3, 9, 6],  # EfficientFormer_depth['S1']
        'embed_dims': [32, 48, 120, 224],  # EfficientFormer_width['S1']
        'downsamples': [True, True, True, True],
        'vit_num': 2,
    }
    
    def __init__(
        self,
        weights_path: str = "./weights/eformer_s1_450.pth",
        resolution: int = 224,
        freeze: bool = True,
    ):
        """
        Initialize EfficientFormerV2-S1 Teacher.
        
        Args:
            weights_path: Path to pretrained weights
            resolution: Input resolution for the model
            freeze: Whether to freeze the teacher model
        """
        super().__init__()
        
        # Import from backbone (now uses EfficientFormerV2)
        from ..models.backbone.efficientformer import (
            EfficientFormerV2,
            EfficientFormer_depth,
            EfficientFormer_width,
            expansion_ratios_S1,
        )
        
        # Create model in feature extraction mode
        self.backbone = EfficientFormerV2(
            layers=EfficientFormer_depth['S1'],
            embed_dims=EfficientFormer_width['S1'],
            downsamples=[True, True, True, True],
            vit_num=2,
            fork_feat=True,  # Feature extraction mode
            resolution=resolution,
            e_ratios=expansion_ratios_S1,
        )
        
        # Output channels for each feature level
        self.out_channels = EfficientFormer_width['S1']
        
        # Load pretrained weights
        self._load_weights(weights_path)
        
        # Freeze if requested
        if freeze:
            self._freeze()
    
    def _load_weights(self, weights_path: str):
        """Load pretrained weights from checkpoint."""
        if weights_path is None:
            print("No weights path provided, using random initialization")
            return
            
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Teacher weights not found: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out head weights (not needed for feature extraction)
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('head') and not k.startswith('dist_head')
        }
        
        # Load weights (strict=False because we exclude heads)
        result = self.backbone.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Loaded EfficientFormerV2-S1 teacher weights from: {weights_path}")
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    
    def _freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass to extract multi-scale features.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        # EfficientFormerV2 expects specific input size
        # If input size differs, resize for feature extraction
        B, C, H, W = x.shape
        
        # Resize if needed (teacher trained on 224x224)
        if H != 224 or W != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Extract features (already in eval mode if frozen)
        with torch.no_grad():
            features = self.backbone(x)
        
        return features
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get named features for distillation.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps
        """
        features = self.forward(x)
        return {
            f'layer{i}': feat for i, feat in enumerate(features)
        }


def create_efficientformer_teacher(
    weights_path: str = "./weights/eformer_s1_450.pth",
    resolution: int = 224,
    freeze: bool = True,
) -> EfficientFormerV2Teacher:
    """
    Factory function to create EfficientFormerV2-S1 teacher.
    
    Args:
        weights_path: Path to pretrained weights
        resolution: Input resolution
        freeze: Whether to freeze the model
        
    Returns:
        EfficientFormerV2Teacher instance
    """
    return EfficientFormerV2Teacher(
        weights_path=weights_path,
        resolution=resolution,
        freeze=freeze,
    )
