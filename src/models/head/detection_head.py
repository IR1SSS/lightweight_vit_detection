"""
Detection heads for object detection.
Implements anchor-based and anchor-free detection heads.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_depthwise: bool = False,
    ):
        """
        Initialize convolution block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            use_depthwise: Use depthwise separable convolution
        """
        super().__init__()
        
        if use_depthwise:
            self.conv = nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                # Pointwise
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class DetectionHead(nn.Module):
    """
    Detection head for object detection.
    
    Implements a shared feature processing with separate classification
    and regression branches.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        num_anchors: int = 3,
        num_convs: int = 4,
        use_depthwise: bool = True,
    ):
        """
        Initialize detection head.
        
        Args:
            in_channels: Input channel dimension
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
            num_convs: Number of shared convolution layers
            use_depthwise: Use depthwise separable convolutions
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolution layers
        shared_convs = []
        for i in range(num_convs):
            shared_convs.append(
                ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise)
            )
        self.shared_convs = nn.Sequential(*shared_convs)
        
        # Classification branch
        self.cls_convs = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise),
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        
        # Regression branch
        self.reg_convs = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise),
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise),
        )
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
        # Objectness branch (for quality estimation)
        self.obj_pred = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize classification bias for better initial training
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
    
    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: List of feature maps from FPN
            
        Returns:
            Tuple of (cls_preds, reg_preds, obj_preds) for each level
        """
        cls_preds = []
        reg_preds = []
        obj_preds = []
        
        for feat in features:
            # Shared features
            shared = self.shared_convs(feat)
            
            # Classification
            cls_feat = self.cls_convs(shared)
            cls_pred = self.cls_pred(cls_feat)
            cls_preds.append(cls_pred)
            
            # Regression
            reg_feat = self.reg_convs(shared)
            reg_pred = self.reg_pred(reg_feat)
            reg_preds.append(reg_pred)
            
            # Objectness
            obj_pred = self.obj_pred(reg_feat)
            obj_preds.append(obj_pred)
        
        return cls_preds, reg_preds, obj_preds


class AnchorHead(nn.Module):
    """
    Anchor-based detection head.
    
    Uses predefined anchor boxes for detection.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        num_anchors: int = 3,
        anchor_sizes: Optional[List[Tuple[int, int]]] = None,
        strides: Optional[List[int]] = None,
    ):
        """
        Initialize anchor head.
        
        Args:
            in_channels: Input channel dimension
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
            anchor_sizes: List of anchor sizes for each level
            strides: Strides for each feature level
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Default anchor sizes
        if anchor_sizes is None:
            anchor_sizes = [
                [(10, 13), (16, 30), (33, 23)],  # Small
                [(30, 61), (62, 45), (59, 119)],  # Medium
                [(116, 90), (156, 198), (373, 326)],  # Large
            ]
        
        # Default strides
        if strides is None:
            strides = [8, 16, 32]
        
        self.anchor_sizes = anchor_sizes
        self.strides = strides
        
        # Prediction layers
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_pred = nn.Conv2d(in_channels, num_anchors, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in [self.cls_pred, self.reg_pred, self.obj_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: List of feature maps
            
        Returns:
            Tuple of predictions for each level
        """
        cls_preds = []
        reg_preds = []
        obj_preds = []
        
        for feat in features:
            cls_preds.append(self.cls_pred(feat))
            reg_preds.append(self.reg_pred(feat))
            obj_preds.append(self.obj_pred(feat))
        
        return cls_preds, reg_preds, obj_preds
    
    def generate_anchors(
        self,
        feature_sizes: List[Tuple[int, int]],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Generate anchors for each feature level.
        
        Args:
            feature_sizes: List of (H, W) for each feature level
            device: Device to create anchors on
            
        Returns:
            List of anchor tensors for each level
        """
        all_anchors = []
        
        for level_idx, (h, w) in enumerate(feature_sizes):
            stride = self.strides[level_idx]
            anchors = self.anchor_sizes[level_idx]
            
            # Generate grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij"
            )
            
            # Base positions
            cx = (grid_x + 0.5) * stride
            cy = (grid_y + 0.5) * stride
            
            # Create anchors
            level_anchors = []
            for (anchor_w, anchor_h) in anchors:
                # Expand to match grid
                anchor_cx = cx.float()
                anchor_cy = cy.float()
                anchor_w_tensor = torch.full_like(anchor_cx, anchor_w)
                anchor_h_tensor = torch.full_like(anchor_cy, anchor_h)
                
                # Stack: (4, H, W) -> (H, W, 4)
                anchor = torch.stack([
                    anchor_cx, anchor_cy, anchor_w_tensor, anchor_h_tensor
                ], dim=-1)
                level_anchors.append(anchor)
            
            # Stack all anchors: (num_anchors, H, W, 4)
            level_anchors = torch.stack(level_anchors, dim=0)
            all_anchors.append(level_anchors)
        
        return all_anchors


class AnchorFreeHead(nn.Module):
    """
    Anchor-free detection head.
    
    Predicts objects without predefined anchors, using center-based detection.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        num_convs: int = 4,
        head_channels: int = 256,
    ):
        """
        Initialize anchor-free head.
        
        Args:
            in_channels: Input channel dimension
            num_classes: Number of object classes
            num_convs: Number of convolution layers
            head_channels: Channel dimension for head layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Heatmap prediction
        cls_convs = []
        for i in range(num_convs):
            cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(head_channels if i > 0 else in_channels, head_channels, 3, padding=1),
                    nn.BatchNorm2d(head_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_pred = nn.Conv2d(head_channels, num_classes, 3, padding=1)
        
        # Box regression
        reg_convs = []
        for i in range(num_convs):
            reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(head_channels if i > 0 else in_channels, head_channels, 3, padding=1),
                    nn.BatchNorm2d(head_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_pred = nn.Conv2d(head_channels, 4, 3, padding=1)
        
        # Center-ness prediction
        self.ctr_pred = nn.Conv2d(head_channels, 1, 3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize heatmap bias
        bias = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_pred.bias, bias)
    
    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: List of feature maps
            
        Returns:
            Tuple of (cls_heatmaps, box_preds, ctr_preds) for each level
        """
        cls_heatmaps = []
        box_preds = []
        ctr_preds = []
        
        for feat in features:
            cls_feat = self.cls_convs(feat)
            cls_heatmaps.append(self.cls_pred(cls_feat))
            
            reg_feat = self.reg_convs(feat)
            box_preds.append(self.reg_pred(reg_feat))
            ctr_preds.append(self.ctr_pred(reg_feat))
        
        return cls_heatmaps, box_preds, ctr_preds
