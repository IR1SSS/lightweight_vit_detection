"""
MobileViT Model Implementation for Lightweight ViT Detection System.

This module implements the MobileViT architecture which combines the strengths
of CNNs and Vision Transformers for efficient mobile vision applications.

Reference:
    MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
    https://arxiv.org/abs/2110.02178
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_model import BaseModel


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that the number of channels is divisible by 8.
    
    Args:
        v: Original channel value.
        divisor: Divisor for channel alignment.
        min_value: Minimum channel value.
        
    Returns:
        Adjusted channel value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        use_act: bool = True,
        act_layer: nn.Module = nn.SiLU
    ) -> None:
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups,
            dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer() if use_act else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.
    
    Consists of:
    1. 1x1 expansion convolution
    2. 3x3 depthwise convolution
    3. 1x1 projection convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 4.0,
        act_layer: nn.Module = nn.SiLU
    ) -> None:
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expansion)
        
        layers = []
        
        # Expansion
        if expansion != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, 1, act_layer=act_layer))
            
        # Depthwise
        layers.append(ConvBNAct(
            hidden_dim, hidden_dim, 3, stride=stride,
            groups=hidden_dim, act_layer=act_layer
        ))
        
        # Projection (no activation)
        layers.append(ConvBNAct(hidden_dim, out_channels, 1, use_act=False))
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head Self-Attention module."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU
    ) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT Block combining local and global representations.
    
    The block consists of:
    1. Local representation using convolutions
    2. Global representation using transformers
    3. Fusion of local and global features
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        depth: int = 2,
        kernel_size: int = 3,
        patch_size: Tuple[int, int] = (2, 2),
        mlp_ratio: float = 4.0,
        num_heads: int = 4,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ) -> None:
        super().__init__()
        
        self.patch_h, self.patch_w = patch_size
        
        # Local representation (before transformer)
        self.conv1 = ConvBNAct(in_channels, in_channels, kernel_size)
        self.conv2 = ConvBNAct(in_channels, dim, 1)
        
        # Global representation (transformer)
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop
            ) for _ in range(depth)
        ])
        
        # Fusion
        self.conv3 = ConvBNAct(dim, in_channels, 1)
        self.conv4 = ConvBNAct(in_channels * 2, out_channels, kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local representation
        local_rep = self.conv1(x)
        local_rep = self.conv2(local_rep)
        
        B, C, H, W = local_rep.shape
        
        # Unfold to patches
        patch_h, patch_w = self.patch_h, self.patch_w
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        
        # Pad if necessary
        if new_h != H or new_w != W:
            local_rep = F.pad(local_rep, (0, new_w - W, 0, new_h - H))
            
        # Reshape for transformer: (B, C, H, W) -> (B * num_patches, patch_h * patch_w, C)
        x_unfolded = rearrange(
            local_rep,
            'b c (h ph) (w pw) -> (b h w) (ph pw) c',
            ph=patch_h, pw=patch_w
        )
        
        # Global representation
        global_rep = self.transformer(x_unfolded)
        
        # Fold back
        global_rep = rearrange(
            global_rep,
            '(b h w) (ph pw) c -> b c (h ph) (w pw)',
            h=new_h // patch_h, w=new_w // patch_w,
            ph=patch_h, pw=patch_w
        )
        
        # Remove padding
        if new_h != H or new_w != W:
            global_rep = global_rep[:, :, :H, :W]
            
        # Fusion
        global_rep = self.conv3(global_rep)
        fusion = torch.cat([x, global_rep], dim=1)
        output = self.conv4(fusion)
        
        return output


class MobileViTBackbone(nn.Module):
    """
    MobileViT Backbone Network.
    
    A hybrid backbone combining MobileNetV2 inverted residuals with
    MobileViT blocks for efficient feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        dims: List[int] = [96, 128, 160],
        channels: List[int] = [16, 32, 48, 64, 80, 96, 384],
        depths: List[int] = [2, 4, 3],
        expansion: float = 4.0,
        patch_size: Tuple[int, int] = (2, 2),
        num_heads: int = 4
    ) -> None:
        super().__init__()
        
        self.out_channels = []
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, channels[0], 3, stride=2),
            InvertedResidual(channels[0], channels[1], stride=1, expansion=expansion)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expansion=expansion),
            InvertedResidual(channels[2], channels[2], stride=1, expansion=expansion),
            InvertedResidual(channels[2], channels[2], stride=1, expansion=expansion)
        )
        self.out_channels.append(channels[2])
        
        # Stage 2
        self.stage2 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expansion=expansion),
            MobileViTBlock(
                channels[3], channels[3], dims[0], depth=depths[0],
                patch_size=patch_size, num_heads=num_heads
            )
        )
        self.out_channels.append(channels[3])
        
        # Stage 3
        self.stage3 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expansion=expansion),
            MobileViTBlock(
                channels[4], channels[4], dims[1], depth=depths[1],
                patch_size=patch_size, num_heads=num_heads
            )
        )
        self.out_channels.append(channels[4])
        
        # Stage 4
        self.stage4 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expansion=expansion),
            MobileViTBlock(
                channels[5], channels[5], dims[2], depth=depths[2],
                patch_size=patch_size, num_heads=num_heads
            ),
            ConvBNAct(channels[5], channels[6], 1)
        )
        self.out_channels.append(channels[6])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            List of feature tensors at different scales.
        """
        features = []
        
        x = self.stem(x)
        
        x = self.stage1(x)
        features.append(x)
        
        x = self.stage2(x)
        features.append(x)
        
        x = self.stage3(x)
        features.append(x)
        
        x = self.stage4(x)
        features.append(x)
        
        return features


class MobileViT(BaseModel):
    """
    MobileViT Detection Model.
    
    Combines MobileViT backbone with FPN neck and detection head
    for efficient object detection on mobile devices.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize MobileViT detection model.
        
        Args:
            config: Model configuration dictionary.
        """
        # Store config before calling parent init
        self._config = config
        super().__init__(config)
        
    def _build_backbone(self) -> nn.Module:
        """Build MobileViT backbone."""
        backbone_cfg = self._config.get('model', {}).get('backbone', {})
        mvit_cfg = backbone_cfg.get('mobilevit_block', {})
        mv2_cfg = backbone_cfg.get('mv2_block', {})
        
        return MobileViTBackbone(
            in_channels=self._config.get('model', {}).get('input', {}).get('channels', 3),
            dims=mvit_cfg.get('dims', [96, 128, 160]),
            channels=mv2_cfg.get('channels', [16, 32, 48, 64, 80, 96]) + [384],
            depths=mvit_cfg.get('depths', [2, 4, 3]),
            expansion=mv2_cfg.get('expansion', 4),
            patch_size=(mvit_cfg.get('patch_size', 2), mvit_cfg.get('patch_size', 2))
        )
        
    def _build_neck(self) -> Optional[nn.Module]:
        """Build FPN neck for multi-scale feature fusion."""
        from .detection_head import SimpleFPN
        
        neck_cfg = self._config.get('model', {}).get('neck', {})
        
        return SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_cfg.get('out_channels', 256),
            num_outs=neck_cfg.get('num_outs', 5)
        )
        
    def _build_head(self) -> nn.Module:
        """Build detection head."""
        from .detection_head import RetinaHead
        
        head_cfg = self._config.get('model', {}).get('head', {})
        
        return RetinaHead(
            num_classes=head_cfg.get('num_classes', 80),
            in_channels=self._config.get('model', {}).get('neck', {}).get('out_channels', 256),
            feat_channels=head_cfg.get('feat_channels', 256),
            stacked_convs=head_cfg.get('stacked_convs', 4),
            num_anchors=len(head_cfg.get('anchor', {}).get('ratios', [0.5, 1.0, 2.0]))
        )
        
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MobileViT detection model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            targets: Optional list of target dictionaries.
            
        Returns:
            Dictionary with detection outputs or losses.
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply FPN
        if self.neck is not None:
            features = self.neck(features)
            
        # Detection head
        cls_scores, bbox_preds = self.head(features)
        
        if self.training and targets is not None:
            # Calculate losses (placeholder - implement in actual training)
            return self.head.compute_loss(cls_scores, bbox_preds, targets)
        else:
            # Return predictions
            return {
                'cls_scores': cls_scores,
                'bbox_preds': bbox_preds
            }
            
    def get_complexity(self) -> Dict[str, float]:
        """Calculate model complexity metrics."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        params_mb = total_params / 1e6
        
        # Estimate model size (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # FLOPs estimation (simplified)
        # For accurate FLOPs, use thop or similar library
        input_size = self._config.get('model', {}).get('input', {}).get('image_size', [640, 640])
        flops_estimate = total_params * input_size[0] * input_size[1] / 1e9
        
        return {
            'params': params_mb,
            'flops': flops_estimate,
            'model_size': model_size_mb
        }


def build_mobilevit(config: Dict[str, Any]) -> MobileViT:
    """
    Build MobileViT detection model from config.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        MobileViT detection model.
    """
    return MobileViT(config)
