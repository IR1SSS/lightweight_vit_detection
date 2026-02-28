"""
Backbone Network Interface for Lightweight ViT Detection System.

This module provides a unified interface for building backbone networks,
supporting both MobileViT and traditional CNN backbones like MobileNetV2.
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 Backbone for feature extraction.
    
    A lightweight CNN backbone that can be used as a baseline
    or combined with ViT components.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: List[int] = [3, 6, 13, 17],
        frozen_stages: int = -1
    ) -> None:
        """
        Initialize MobileNetV2 backbone.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights.
            out_indices: Indices of layers to output features from.
            frozen_stages: Number of stages to freeze (-1 means no freezing).
        """
        super().__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
            
        self.features = mobilenet.features
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Get output channels for each stage
        self.out_channels = []
        for idx in out_indices:
            self.out_channels.append(self.features[idx][0].out_channels 
                                    if hasattr(self.features[idx][0], 'out_channels')
                                    else self.features[idx].out_channels)
            
        # Freeze stages if specified
        self._freeze_stages()
        
    def _freeze_stages(self) -> None:
        """Freeze early stages of the backbone."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                for param in self.features[i].parameters():
                    param.requires_grad = False
                    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            List of feature tensors at different scales.
        """
        outputs = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
                
        return outputs


class BackboneWrapper(nn.Module):
    """
    Generic wrapper for backbone networks.
    
    Provides a unified interface for different backbone architectures.
    """
    
    def __init__(
        self,
        backbone_type: str,
        pretrained: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize backbone wrapper.
        
        Args:
            backbone_type: Type of backbone ('mobilenetv2', 'mobilevit', etc.).
            pretrained: Whether to use pretrained weights.
            **kwargs: Additional arguments for specific backbones.
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        
        if backbone_type == 'mobilenetv2':
            self.backbone = MobileNetV2Backbone(pretrained=pretrained, **kwargs)
            self.out_channels = self.backbone.out_channels
        elif backbone_type in ['mobilevit', 'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs']:
            from .mobilevit import MobileViTBackbone
            self.backbone = MobileViTBackbone(**kwargs)
            self.out_channels = self.backbone.out_channels
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale features."""
        return self.backbone(x)


def build_backbone(config: Dict[str, Any]) -> nn.Module:
    """
    Build backbone network from configuration.
    
    Args:
        config: Backbone configuration dictionary with keys:
            - name: Backbone type ('mobilenetv2', 'mobilevit', etc.)
            - pretrained: Whether to use pretrained weights
            - Additional backbone-specific parameters
            
    Returns:
        Backbone network module.
        
    Example:
        >>> config = {
        ...     'name': 'mobilevit_s',
        ...     'pretrained': True,
        ...     'mobilevit_block': {'dims': [96, 128, 160]}
        ... }
        >>> backbone = build_backbone(config)
    """
    backbone_name = config.get('name', 'mobilevit_s')
    pretrained = config.get('pretrained', True)
    
    if backbone_name == 'mobilenetv2':
        return MobileNetV2Backbone(
            pretrained=pretrained,
            out_indices=config.get('out_indices', [3, 6, 13, 17]),
            frozen_stages=config.get('frozen_stages', -1)
        )
    elif backbone_name in ['mobilevit', 'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs']:
        from .mobilevit import MobileViTBackbone
        
        mvit_cfg = config.get('mobilevit_block', {})
        mv2_cfg = config.get('mv2_block', {})
        
        return MobileViTBackbone(
            dims=mvit_cfg.get('dims', [96, 128, 160]),
            channels=mv2_cfg.get('channels', [16, 32, 48, 64, 80, 96]) + [384],
            depths=mvit_cfg.get('depths', [2, 4, 3]),
            expansion=mv2_cfg.get('expansion', 4),
            patch_size=(mvit_cfg.get('patch_size', 2), mvit_cfg.get('patch_size', 2))
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


# Registry for backbone networks
BACKBONE_REGISTRY = {
    'mobilenetv2': MobileNetV2Backbone,
    'mobilevit': 'mobilevit.MobileViTBackbone',
    'mobilevit_s': 'mobilevit.MobileViTBackbone',
    'mobilevit_xs': 'mobilevit.MobileViTBackbone',
    'mobilevit_xxs': 'mobilevit.MobileViTBackbone',
}


def get_available_backbones() -> List[str]:
    """Get list of available backbone names."""
    return list(BACKBONE_REGISTRY.keys())
