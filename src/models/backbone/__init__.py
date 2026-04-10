"""
Backbone modules for the Lightweight ViT Detection System.
"""

from .attention import (
    LinearAttention,
    PoolAttention,
    EfficientAttention,
    MultiHeadAttention,
)
from .mobilevit import MobileViT, MobileViTBlock, MV2Block
from .efficientformer import (
    EfficientFormerV2,
    EfficientFormer_width,
    EfficientFormer_depth,
    expansion_ratios_S0,
    expansion_ratios_S1,
    expansion_ratios_S2,
    expansion_ratios_L,
    efficientformerv2_s0,
    efficientformerv2_s1,
    efficientformerv2_s2,
    efficientformerv2_l,
)

# 保持向后兼容的别名
EfficientFormer = EfficientFormerV2

__all__ = [
    # Attention modules
    "LinearAttention",
    "PoolAttention",
    "EfficientAttention",
    "MultiHeadAttention",
    # Backbone modules
    "MobileViT",
    "MobileViTBlock",
    "MV2Block",
    # EfficientFormerV2
    "EfficientFormerV2",
    "EfficientFormer",
    "EfficientFormer_width",
    "EfficientFormer_depth",
    "efficientformerv2_s0",
    "efficientformerv2_s1",
    "efficientformerv2_s2",
    "efficientformerv2_l",
]
