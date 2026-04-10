"""
Feature Pyramid Network (FPN) and Path Aggregation FPN implementations.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_bn: bool = True,
        use_act: bool = True,
    ):
        """
        Initialize convolution block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            groups: Groups for depthwise convolution
            use_bn: Use batch normalization
            use_act: Use activation
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=not use_bn)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if use_act:
            layers.append(nn.SiLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class DepthwiseConvBlock(nn.Module):
    """Depthwise separable convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Initialize depthwise convolution block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
        """
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    
    Builds a feature pyramid from multi-scale features for object detection.
    Uses top-down pathway with lateral connections.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        num_outs: int = 3,
        use_depthwise: bool = False,
    ):
        """
        Initialize FPN.
        
        Args:
            in_channels_list: List of input channel sizes for each level
            out_channels: Output channel size for all levels
            num_outs: Number of output feature levels
            use_depthwise: Use depthwise separable convolutions
        """
        super().__init__()
        
        self.num_ins = len(in_channels_list)
        self.num_outs = num_outs
        self.out_channels = out_channels
        
        # Lateral convolutions (1x1 to unify channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            if use_depthwise:
                lateral_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                lateral_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            self.lateral_convs.append(lateral_conv)
        
        # Output convolutions (3x3 to reduce aliasing)
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_outs):
            if use_depthwise:
                self.fpn_convs.append(
                    DepthwiseConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
                )
            else:
                self.fpn_convs.append(
                    ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
                )
        
        # Extra convolutions for additional output levels
        self.extra_convs = None
        if num_outs > len(in_channels_list):
            self.extra_convs = nn.ModuleList()
            for _ in range(num_outs - len(in_channels_list)):
                self.extra_convs.append(
                    ConvBlock(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
        
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
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: List of feature maps from backbone (low-res to high-res)
            
        Returns:
            List of FPN feature maps
        """
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        
        # Apply output convolutions
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # Add extra levels
        if self.extra_convs is not None:
            for extra_conv in self.extra_convs:
                outs.append(extra_conv(outs[-1]))
        
        return outs


class PAFPN(nn.Module):
    """
    Path Aggregation FPN (PAFPN) from YOLOv4.
    
    Adds a bottom-up pathway to FPN for better feature aggregation.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        num_outs: int = 3,
        use_depthwise: bool = False,
    ):
        """
        Initialize PAFPN.
        
        Args:
            in_channels_list: List of input channel sizes
            out_channels: Output channel size
            num_outs: Number of output feature levels
            use_depthwise: Use depthwise separable convolutions
        """
        super().__init__()
        
        self.num_ins = len(in_channels_list)
        self.num_outs = num_outs
        self.out_channels = out_channels
        
        # Top-down pathway
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            )
        
        for _ in range(len(in_channels_list)):
            if use_depthwise:
                self.fpn_convs.append(DepthwiseConvBlock(out_channels, out_channels))
            else:
                self.fpn_convs.append(ConvBlock(out_channels, out_channels))
        
        # Bottom-up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        
        for _ in range(len(in_channels_list) - 1):
            self.downsample_convs.append(
                ConvBlock(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            if use_depthwise:
                self.pafpn_convs.append(DepthwiseConvBlock(out_channels, out_channels))
            else:
                self.pafpn_convs.append(ConvBlock(out_channels, out_channels))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: List of feature maps from backbone
            
        Returns:
            List of PAFPN feature maps
        """
        # Top-down pathway
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        
        # Apply FPN convolutions
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # Bottom-up pathway
        pafpn_outs = [fpn_outs[0]]
        
        for i in range(len(fpn_outs) - 1):
            downsampled = self.downsample_convs[i](pafpn_outs[-1])
            fused = downsampled + fpn_outs[i + 1]
            pafpn_outs.append(self.pafpn_convs[i](fused))
        
        return pafpn_outs
