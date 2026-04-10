"""
Complete ViT Detector combining backbone, neck, and head.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileViT
from .neck import FPN, PAFPN
from .head import DetectionHead, AnchorHead
from ..utils.config import Config


class ViTDetector(nn.Module):
    """
    Complete Vision Transformer Detector.
    
    Combines a lightweight ViT backbone with FPN neck and detection head
    for real-time object detection.
    """
    
    def __init__(
        self,
        backbone_name: str = "mobilevit_small",
        neck_name: str = "fpn",
        num_classes: int = 80,
        num_anchors: int = 3,
        use_depthwise: bool = True,
        pretrained_backbone: bool = False,
    ):
        """
        Initialize ViT Detector.
        
        Args:
            backbone_name: Name of the backbone ("mobilevit_small", "mobilevit_base", "efficientformer_l1")
            neck_name: Name of the neck ("fpn", "pafpn")
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
            use_depthwise: Use depthwise separable convolutions
            pretrained_backbone: Use pretrained backbone weights
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Build backbone
        if backbone_name.startswith("mobilevit"):
            model_size = "small" if "small" in backbone_name else "base"
            self.backbone = MobileViT(model_size=model_size)
            backbone_channels = self.backbone.get_output_channels()
        elif backbone_name.startswith("efficientformer"):
            # EfficientFormerV2 (S0, S1, S2, L)
            from .backbone import (
                EfficientFormerV2,
                EfficientFormer_width,
                EfficientFormer_depth,
                expansion_ratios_S0,
                expansion_ratios_S1,
                expansion_ratios_S2,
                expansion_ratios_L,
            )
            
            # Parse model size from name (e.g., "efficientformerv2_s1" -> "S1")
            size_map = {"s0": "S0", "s1": "S1", "s2": "S2", "l": "L", "l1": "S1"}
            if "v2" in backbone_name.lower():
                size_key = backbone_name.lower().split("_")[-1]
                model_size = size_map.get(size_key, "S1")
            else:
                model_size = "S1"  # Default to S1
            
            # Get expansion ratios
            expansion_map = {
                "S0": expansion_ratios_S0,
                "S1": expansion_ratios_S1,
                "S2": expansion_ratios_S2,
                "L": expansion_ratios_L,
            }
            
            self.backbone = EfficientFormerV2(
                layers=EfficientFormer_depth[model_size],
                embed_dims=EfficientFormer_width[model_size],
                downsamples=[True, True, True, True],
                vit_num=2 if model_size in ["S1", "S0"] else (4 if model_size == "S2" else 6),
                fork_feat=True,  # Feature extraction mode
                e_ratios=expansion_map[model_size],
            )
            # Use last 3 stages for FPN (skip first stage with smallest features)
            backbone_channels = EfficientFormer_width[model_size][1:]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Build neck
        # Use appropriate FPN channels based on backbone
        if "efficientformer" in backbone_name:
            fpn_channels = 128
        else:
            fpn_channels = 128 if "small" in backbone_name else 192
        if neck_name == "fpn":
            self.neck = FPN(backbone_channels, fpn_channels, use_depthwise=use_depthwise)
        elif neck_name == "pafpn":
            self.neck = PAFPN(backbone_channels, fpn_channels, use_depthwise=use_depthwise)
        else:
            raise ValueError(f"Unknown neck: {neck_name}")
        
        # Build head
        self.head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            use_depthwise=use_depthwise,
        )
        
        # Strides for each feature level
        self.strides = [8, 16, 32]
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Tuple of (cls_preds, reg_preds, obj_preds) concatenated from all levels
        """
        # Backbone features
        features = self.backbone(x)
        
        # For EfficientFormerV2, use last 3 stages (skip first)
        if hasattr(self, 'backbone_name') and 'efficientformer' in self.backbone_name:
            features = features[1:]  # Skip first stage
        
        # Neck features
        fpn_features = self.neck(features)
        
        # Head predictions
        cls_preds, reg_preds, obj_preds = self.head(fpn_features)
        
        # Concatenate predictions from all levels
        cls_pred = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
        reg_pred = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
        obj_pred = torch.cat([o.flatten(2) for o in obj_preds], dim=2)
        
        # Reshape for output
        B = x.shape[0]
        cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, -1).permute(0, 1, 3, 2)
        reg_pred = reg_pred.view(B, self.num_anchors, 4, -1).permute(0, 1, 3, 2)
        obj_pred = obj_pred.view(B, self.num_anchors, 1, -1).permute(0, 1, 3, 2)
        
        return cls_pred, reg_pred, obj_pred
    
    def get_predictions(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        obj_pred: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw predictions to final detections.
        
        Args:
            cls_pred: Classification predictions (B, A, N, C)
            reg_pred: Box regression predictions (B, A, N, 4)
            obj_pred: Objectness predictions (B, A, N, 1)
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
            
        Returns:
            Dictionary with boxes, scores, labels
        """
        B, A, N, C = cls_pred.shape
        
        # Flatten predictions
        cls_pred = cls_pred.reshape(B, -1, C)
        reg_pred = reg_pred.reshape(B, -1, 4)
        obj_pred = obj_pred.reshape(B, -1, 1)
        
        # Apply sigmoid to get probabilities
        cls_scores = cls_pred.sigmoid()
        obj_scores = obj_pred.sigmoid()
        scores = cls_scores * obj_scores
        
        # Generate anchor points
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for b in range(B):
            batch_scores = scores[b]
            batch_regs = reg_pred[b]
            
            # Get top predictions
            max_scores, labels = batch_scores.max(dim=1)
            
            # Filter by confidence
            mask = max_scores > conf_threshold
            filtered_scores = max_scores[mask]
            filtered_labels = labels[mask]
            filtered_regs = batch_regs[mask]
            
            if len(filtered_scores) == 0:
                all_boxes.append(torch.zeros(0, 4, device=cls_pred.device))
                all_scores.append(torch.zeros(0, device=cls_pred.device))
                all_labels.append(torch.zeros(0, dtype=torch.long, device=cls_pred.device))
                continue
            
            # Decode boxes (assuming ltrb format)
            # This is simplified - actual implementation needs proper anchor decoding
            boxes = filtered_regs
            
            # Apply NMS
            keep = self._nms(boxes, filtered_scores, nms_threshold)
            
            # Keep top detections
            keep = keep[:max_detections]
            
            all_boxes.append(boxes[keep])
            all_scores.append(filtered_scores[keep])
            all_labels.append(filtered_labels[keep])
        
        return {
            "boxes": torch.stack(all_boxes),
            "scores": torch.stack(all_scores),
            "labels": torch.stack(all_labels),
        }
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        """
        Non-maximum suppression.
        
        Args:
            boxes: Boxes (N, 4)
            scores: Scores (N,)
            iou_threshold: IoU threshold
            
        Returns:
            Indices of kept boxes
        """
        # Sort by score
        order = scores.argsort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU
            ious = self._compute_iou(boxes[i:i+1], boxes[order[1:]])
            
            # Keep boxes with IoU below threshold
            mask = ious.squeeze(0) < iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, device=boxes.device)
    
    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: First set (N, 4)
            boxes2: Second set (M, 4)
            
        Returns:
            IoU matrix (N, M)
        """
        # Ensure xyxy format
        x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)
        
        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
        
        return intersection / (union + 1e-6)
    
    def get_param_groups(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.05,
        bias_lr_factor: float = 1.0,
        bias_weight_decay: float = 0.0,
    ) -> List[Dict]:
        """
        Get parameter groups for optimizer.
        
        Args:
            lr: Base learning rate
            weight_decay: Weight decay for weights
            bias_lr_factor: Learning rate factor for biases
            bias_weight_decay: Weight decay for biases
            
        Returns:
            List of parameter group dictionaries
        """
        params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            param_lr = lr
            param_wd = weight_decay
            
            if "bias" in name:
                param_lr = lr * bias_lr_factor
                param_wd = bias_weight_decay
            
            params.append({
                "params": [param],
                "lr": param_lr,
                "weight_decay": param_wd,
            })
        
        return params


def build_detector(config: Config) -> ViTDetector:
    """
    Build detector from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        ViTDetector instance
    """
    model_config = config.model
    
    detector = ViTDetector(
        backbone_name=model_config.backbone.name,
        neck_name=model_config.neck.name,
        num_classes=model_config.head.num_classes,
        num_anchors=model_config.head.num_anchors,
        use_depthwise=model_config.head.get("use_depthwise", True),
    )
    
    return detector
