"""
MobileViT backbone implementation.
Light-weight, General-purpose, and Mobile-friendly Vision Transformer.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, LinearAttention


def _make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Make value divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: nn.Module = nn.ReLU,
    ):
        """
        Initialize ConvBNReLU.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            groups: Groups for depthwise convolution
            activation: Activation function
        """
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            layers.append(activation(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """Inverted residual block from MobileNetV2."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
    ):
        """
        Initialize inverted residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride (1 or 2)
            expand_ratio: Expansion ratio
        """
        super().__init__()
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        
        # Depthwise convolution
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                      padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class TransformerBlock(nn.Module):
    """Transformer block for MobileViT."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_type: str = "linear",
    ):
        """
        Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_type: Type of attention ("linear", "standard")
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention
        if attention_type == "linear":
            self.attn = LinearAttention(dim, num_heads=num_heads, dropout=dropout)
        else:
            self.attn = MultiHeadAttention(dim, num_heads=num_heads, attn_drop=dropout)
        
        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT block combining local (conv) and global (transformer) processing.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        transformer_blocks: int = 2,
        patch_size: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        stride: int = 2,
    ):
        """
        Initialize MobileViT block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            transformer_dim: Transformer hidden dimension
            transformer_blocks: Number of transformer blocks
            patch_size: Patch size for unfolding
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio
            dropout: Dropout rate
            stride: Stride for the block
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim
        
        # Local feature extraction
        self.local_conv = ConvBNReLU(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels
        )
        
        # Projection to transformer dimension
        self.proj_in = ConvBNReLU(in_channels, transformer_dim, kernel_size=1)
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(transformer_dim, num_heads, mlp_ratio, dropout)
              for _ in range(transformer_blocks)]
        )
        
        # Projection back
        self.proj_out = ConvBNReLU(transformer_dim, in_channels, kernel_size=1)
        
        # Fusion convolution with stride
        self.fusion = ConvBNReLU(
            2 * in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor
        """
        B, C, H, W = x.shape
        residual = x
        
        # Local features
        local_feat = self.local_conv(x)
        
        # Project to transformer dimension
        feat = self.proj_in(local_feat)
        
        # Simple reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        B, C_dim, H_feat, W_feat = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply transformer
        feat = self.transformer(feat)  # (B, H*W, C)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        feat = feat.transpose(1, 2).reshape(B, C_dim, H_feat, W_feat)
        
        # Project back to original channels
        feat = self.proj_out(feat)
        
        # Fuse with residual
        fused = torch.cat([residual, feat], dim=1)
        out = self.fusion(fused)
        
        return out


class MV2Block(nn.Module):
    """MobileNetV2 block wrapper."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
    ):
        """
        Initialize MV2 block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride
            expand_ratio: Expansion ratio
        """
        super().__init__()
        self.block = InvertedResidual(in_channels, out_channels, stride, expand_ratio)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class MobileViT(nn.Module):
    """
    MobileViT backbone for object detection.
    
    A light-weight vision transformer that combines the strengths of CNNs and Transformers
    for mobile-friendly inference.
    """
    
    # Configuration for different model sizes
    CONFIGS = {
        "small": {
            "width_mult": 0.5,
            "blocks": [
                # (type, out_channels, stride, transformer_dim, transformer_blocks)
                ("mv2", 32, 1, None, None),
                ("mv2", 64, 2, None, None),
                ("mv2", 96, 2, None, None),
                ("mobilevit", 128, 2, 144, 2),
                ("mobilevit", 160, 2, 192, 4),
                ("mobilevit", 320, 2, 240, 3),
            ],
        },
        "base": {
            "width_mult": 1.0,
            "blocks": [
                ("mv2", 32, 1, None, None),
                ("mv2", 64, 2, None, None),
                ("mv2", 128, 2, None, None),
                ("mobilevit", 160, 2, 192, 2),
                ("mobilevit", 224, 2, 240, 4),
                ("mobilevit", 288, 2, 336, 3),
            ],
        },
    }
    
    def __init__(
        self,
        model_size: str = "small",
        in_channels: int = 3,
        width_mult: Optional[float] = None,
        attention_type: str = "linear",
        dropout: float = 0.0,
    ):
        """
        Initialize MobileViT backbone.
        
        Args:
            model_size: Model size ("small" or "base")
            in_channels: Number of input channels
            width_mult: Width multiplier (overrides default)
            attention_type: Type of attention ("linear" or "standard")
            dropout: Dropout rate
        """
        super().__init__()
        
        config = self.CONFIGS[model_size]
        width_mult = width_mult or config["width_mult"]
        blocks_config = config["blocks"]
        
        # Stem convolution
        self.stem = ConvBNReLU(
            in_channels, _make_divisible(16 * width_mult), kernel_size=3, stride=2, padding=1
        )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        self.out_channels = []
        in_ch = _make_divisible(16 * width_mult)
        
        for block_type, out_ch, stride, trans_dim, trans_blocks in blocks_config:
            out_ch = _make_divisible(out_ch * width_mult)
            
            if block_type == "mv2":
                block = MV2Block(in_ch, out_ch, stride=stride)
            else:  # mobilevit
                trans_dim = _make_divisible(trans_dim * width_mult)
                block = MobileViTBlock(
                    in_ch, out_ch, trans_dim, trans_blocks,
                    patch_size=2, stride=stride, dropout=dropout
                )
            
            self.blocks.append(block)
            in_ch = out_ch
            
            # Record output channels for detection (after stride-2 blocks)
            if stride == 2 and block_type == "mobilevit":
                self.out_channels.append(out_ch)
        
        # Final convolution
        self.final_conv = ConvBNReLU(in_ch, _make_divisible(640 * width_mult), kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
            # Collect features for FPN
            if hasattr(block, 'fusion'):  # MobileViT blocks
                features.append(x)
        
        x = self.final_conv(x)
        features.append(x)
        
        # Return last 3 features for FPN
        return features[-3:]
    
    def get_output_channels(self) -> List[int]:
        """
        Get output channel dimensions for each feature level.
        
        Returns:
            List of channel dimensions
        """
        # Return actual output channels from forward pass
        # For small model: [80, 160, 320]
        return [80, 160, 320]  # Will be dynamically determined by forward pass
