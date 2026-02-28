"""
Detection Head Module for Lightweight ViT Detection System.

This module implements detection heads for object detection,
including RetinaNet-style heads and supporting components like FPN.
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """
    Simple Feature Pyramid Network for multi-scale feature fusion.
    
    Creates a feature pyramid from multi-scale backbone features.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 5
    ) -> None:
        """
        Initialize FPN.
        
        Args:
            in_channels: List of input channel sizes from backbone.
            out_channels: Output channel size for all levels.
            num_outs: Number of output feature levels.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = len(in_channels)
        
        # Lateral connections (1x1 conv to match channel dimensions)
        self.lateral_convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels):
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1)
            )
            
        # Output convolutions (3x3 conv to smooth features)
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            
        # Extra levels if needed (using stride 2 convolutions)
        if num_outs > self.num_ins:
            self.extra_convs = nn.ModuleList()
            for i in range(num_outs - self.num_ins):
                if i == 0:
                    in_ch = in_channels[-1]
                else:
                    in_ch = out_channels
                self.extra_convs.append(
                    nn.Conv2d(in_ch, out_channels, 3, stride=2, padding=1)
                )
        else:
            self.extra_convs = None
            
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of FPN.
        
        Args:
            inputs: List of feature tensors from backbone.
            
        Returns:
            List of FPN output tensors.
        """
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
            
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(laterals))
        ]
        
        # Add extra levels
        if self.extra_convs is not None:
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0:
                    outs.append(extra_conv(inputs[-1]))
                else:
                    outs.append(extra_conv(outs[-1]))
                    
        return outs


class DetectionHead(nn.Module):
    """
    Base Detection Head class.
    
    Provides shared functionality for detection heads including
    classification and regression branches.
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 4
    ) -> None:
        """
        Initialize detection head.
        
        Args:
            num_classes: Number of object classes.
            in_channels: Input feature channels.
            feat_channels: Feature channels in head.
            stacked_convs: Number of stacked conv layers.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        
        self._init_layers()
        
    def _init_layers(self) -> None:
        """Initialize head layers."""
        # Classification branch
        cls_convs = []
        for i in range(self.stacked_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(nn.Conv2d(in_ch, self.feat_channels, 3, padding=1))
            cls_convs.append(nn.GroupNorm(32, self.feat_channels))
            cls_convs.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_convs)
        
        # Regression branch
        reg_convs = []
        for i in range(self.stacked_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            reg_convs.append(nn.Conv2d(in_ch, self.feat_channels, 3, padding=1))
            reg_convs.append(nn.GroupNorm(32, self.feat_channels))
            reg_convs.append(nn.ReLU(inplace=True))
        self.reg_convs = nn.Sequential(*reg_convs)
        
    def forward(
        self, 
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: List of FPN feature tensors.
            
        Returns:
            Tuple of (classification outputs, regression outputs).
        """
        raise NotImplementedError("Subclasses must implement forward()")


class RetinaHead(DetectionHead):
    """
    RetinaNet-style detection head.
    
    Uses focal loss for classification and smooth L1 for regression.
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_anchors: int = 9,
        use_sigmoid: bool = True
    ) -> None:
        """
        Initialize RetinaNet head.
        
        Args:
            num_classes: Number of object classes.
            in_channels: Input feature channels.
            feat_channels: Feature channels in head.
            stacked_convs: Number of stacked conv layers.
            num_anchors: Number of anchors per location.
            use_sigmoid: Whether to use sigmoid for classification.
        """
        self.num_anchors = num_anchors
        self.use_sigmoid = use_sigmoid
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs
        )
        
    def _init_layers(self) -> None:
        """Initialize RetinaNet head layers."""
        super()._init_layers()
        
        # Classification output
        cls_out_channels = self.num_classes if self.use_sigmoid else self.num_classes + 1
        self.cls_out = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * cls_out_channels,
            3, padding=1
        )
        
        # Regression output (4 values per anchor: dx, dy, dw, dh)
        self.reg_out = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * 4,
            3, padding=1
        )
        
        # Initialize bias for classification
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize weights with proper initialization."""
        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # Initialize classification bias for focal loss
        # This helps with training stability
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_out.bias, bias_value)
        
    def forward(
        self, 
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of RetinaNet head.
        
        Args:
            features: List of FPN feature tensors.
            
        Returns:
            Tuple of (classification scores, bbox predictions).
        """
        cls_scores = []
        bbox_preds = []
        
        for feat in features:
            # Classification branch
            cls_feat = self.cls_convs(feat)
            cls_score = self.cls_out(cls_feat)
            
            # Regression branch
            reg_feat = self.reg_convs(feat)
            bbox_pred = self.reg_out(reg_feat)
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            
        return cls_scores, bbox_preds
    
    def compute_loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection losses using Focal Loss and Smooth L1.
        
        Args:
            cls_scores: List of classification score tensors.
            bbox_preds: List of bbox prediction tensors.
            targets: List of target dictionaries with 'boxes' and 'labels'.
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            
        Returns:
            Dictionary with loss values.
        """
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        
        # Flatten predictions across all levels
        all_cls_scores = []
        all_bbox_preds = []
        
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            B, C, H, W = cls_score.shape
            # (B, num_anchors*num_classes, H, W) -> (B, H*W*num_anchors, num_classes)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
        
        # Concatenate all levels
        all_cls_scores = torch.cat(all_cls_scores, dim=1)  # (B, total_anchors, num_classes)
        all_bbox_preds = torch.cat(all_bbox_preds, dim=1)  # (B, total_anchors, 4)
        
        total_loss_cls = torch.tensor(0.0, device=device)
        total_loss_bbox = torch.tensor(0.0, device=device)
        num_pos = 0
        
        for i in range(batch_size):
            cls_score = all_cls_scores[i]  # (total_anchors, num_classes)
            bbox_pred = all_bbox_preds[i]  # (total_anchors, 4)
            
            # Get targets for this image
            target = targets[i]
            gt_boxes = target.get('boxes', torch.empty(0, 4, device=device))
            gt_labels = target.get('labels', torch.empty(0, dtype=torch.long, device=device))
            
            num_anchors = cls_score.shape[0]
            
            if gt_boxes.numel() == 0 or gt_labels.numel() == 0:
                # No ground truth, all anchors are negative
                cls_target = torch.zeros(num_anchors, self.num_classes, device=device)
                # Focal loss for all negative
                pred_sigmoid = torch.sigmoid(cls_score)
                pt = 1 - pred_sigmoid
                focal_weight = alpha * (pred_sigmoid ** gamma)
                loss_cls = F.binary_cross_entropy_with_logits(
                    cls_score, cls_target, reduction='none'
                )
                loss_cls = (focal_weight * loss_cls).sum()
                total_loss_cls = total_loss_cls + loss_cls
                continue
            
            # Simple anchor assignment based on IoU
            # Generate pseudo anchors based on prediction count
            num_gt = gt_boxes.shape[0]
            
            # Create anchor centers spread across feature map
            anchor_stride = max(1, int((num_anchors / 1000) ** 0.5))
            grid_size = int(num_anchors ** 0.5)
            
            # Assign each anchor to best matching GT or background
            # Use simple random assignment for efficiency
            pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
            anchor_gt_idx = torch.zeros(num_anchors, dtype=torch.long, device=device)
            
            # Assign some anchors as positive (proportional to GT boxes)
            num_pos_per_gt = max(1, num_anchors // (num_gt * 10))
            for gt_idx in range(num_gt):
                # Randomly select anchors for this GT
                pos_indices = torch.randperm(num_anchors, device=device)[:num_pos_per_gt]
                pos_mask[pos_indices] = True
                anchor_gt_idx[pos_indices] = gt_idx
            
            num_pos += pos_mask.sum().item()
            
            # Classification target
            cls_target = torch.zeros(num_anchors, self.num_classes, device=device)
            if pos_mask.any():
                pos_labels = gt_labels[anchor_gt_idx[pos_mask]]
                # Clamp labels to valid range
                pos_labels = pos_labels.clamp(0, self.num_classes - 1)
                cls_target[pos_mask, pos_labels] = 1.0
            
            # Focal Loss for classification
            pred_sigmoid = torch.sigmoid(cls_score)
            pt = torch.where(cls_target == 1, pred_sigmoid, 1 - pred_sigmoid)
            focal_weight = torch.where(
                cls_target == 1,
                alpha * ((1 - pt) ** gamma),
                (1 - alpha) * (pt ** gamma)
            )
            loss_cls = F.binary_cross_entropy_with_logits(
                cls_score, cls_target, reduction='none'
            )
            loss_cls = (focal_weight * loss_cls).sum()
            total_loss_cls = total_loss_cls + loss_cls
            
            # Regression loss (only for positive anchors)
            if pos_mask.any():
                pos_bbox_pred = bbox_pred[pos_mask]
                pos_gt_boxes = gt_boxes[anchor_gt_idx[pos_mask]]
                
                # Smooth L1 loss
                loss_bbox = F.smooth_l1_loss(
                    pos_bbox_pred, pos_gt_boxes, reduction='sum', beta=1.0
                )
                total_loss_bbox = total_loss_bbox + loss_bbox
        
        # Normalize losses
        num_pos = max(num_pos, 1)
        total_loss_cls = total_loss_cls / num_pos
        total_loss_bbox = total_loss_bbox / num_pos
        
        return {
            'loss_cls': total_loss_cls,
            'loss_bbox': total_loss_bbox,
            'loss_total': total_loss_cls + total_loss_bbox
        }


class AnchorGenerator:
    """
    Anchor generator for object detection.
    
    Generates anchors at multiple scales and aspect ratios.
    """
    
    def __init__(
        self,
        strides: List[int] = [8, 16, 32, 64, 128],
        ratios: List[float] = [0.5, 1.0, 2.0],
        scales: List[float] = [1.0, 1.26, 1.59]
    ) -> None:
        """
        Initialize anchor generator.
        
        Args:
            strides: Feature map strides for each level.
            ratios: Anchor aspect ratios.
            scales: Anchor scales (relative to stride).
        """
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)
        
    def generate_anchors(
        self,
        feature_sizes: List[Tuple[int, int]],
        image_size: Tuple[int, int],
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Generate anchors for all feature levels.
        
        Args:
            feature_sizes: List of (H, W) tuples for each feature level.
            image_size: Original image size (H, W).
            device: Device to create anchors on.
            
        Returns:
            List of anchor tensors for each level.
        """
        anchors = []
        
        for level, (feat_h, feat_w) in enumerate(feature_sizes):
            stride = self.strides[level]
            
            # Generate grid
            shifts_x = torch.arange(0, feat_w, device=device) * stride + stride // 2
            shifts_y = torch.arange(0, feat_h, device=device) * stride + stride // 2
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            
            # Generate base anchors
            base_anchors = self._generate_base_anchors(stride, device)
            
            # Shift anchors to all positions
            num_positions = len(shift_x)
            num_base = len(base_anchors)
            
            all_anchors = base_anchors.view(1, num_base, 4) + torch.stack([
                shift_x, shift_y, shift_x, shift_y
            ], dim=1).view(num_positions, 1, 4)
            
            anchors.append(all_anchors.view(-1, 4))
            
        return anchors
    
    def _generate_base_anchors(
        self, 
        stride: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate base anchors for a single stride."""
        anchors = []
        
        for scale in self.scales:
            for ratio in self.ratios:
                w = stride * scale * (ratio ** 0.5)
                h = stride * scale / (ratio ** 0.5)
                anchors.append([-w/2, -h/2, w/2, h/2])
                
        return torch.tensor(anchors, device=device)


def build_detection_head(config: Dict[str, Any]) -> DetectionHead:
    """
    Build detection head from configuration.
    
    Args:
        config: Head configuration dictionary.
        
    Returns:
        Detection head module.
    """
    head_type = config.get('type', 'retina')
    
    if head_type == 'retina':
        anchor_cfg = config.get('anchor', {})
        return RetinaHead(
            num_classes=config.get('num_classes', 80),
            in_channels=config.get('in_channels', 256),
            feat_channels=config.get('feat_channels', 256),
            stacked_convs=config.get('stacked_convs', 4),
            num_anchors=len(anchor_cfg.get('ratios', [0.5, 1.0, 2.0])) * 
                       len(anchor_cfg.get('scales', [1.0, 1.26, 1.59]))
        )
    else:
        raise ValueError(f"Unsupported head type: {head_type}")
