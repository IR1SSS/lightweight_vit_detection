"""
Distillation losses for knowledge distillation.
Implements response, feature, and relation distillation losses.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseDistillationLoss(nn.Module):
    """
    Response-based distillation loss.
    
    Distills knowledge through the output logits using KL divergence
    with temperature scaling.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        reduction: str = "mean",
    ):
        """
        Initialize response distillation loss.
        
        Args:
            temperature: Temperature for softmax softening
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute response distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits (soft labels)
            
        Returns:
            Distillation loss
        """
        # Soften logits with temperature
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        
        # Scale by temperature squared
        loss = loss * (self.temperature ** 2)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss.
    
    Distills knowledge through intermediate feature maps.
    Supports MSE, L1, and cosine similarity losses.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True,
    ):
        """
        Initialize feature distillation loss.
        
        Args:
            loss_type: Loss type ("mse", "l1", "cosine")
            normalize: Whether to normalize features
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: Student feature maps
            teacher_features: Teacher feature maps
            
        Returns:
            Distillation loss
        """
        # Handle different spatial dimensions
        if student_features.shape != teacher_features.shape:
            # Resize student features to match teacher
            student_features = F.interpolate(
                student_features,
                size=teacher_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        # Handle different channel dimensions
        if student_features.shape[1] != teacher_features.shape[1]:
            # Adapt student channels
            student_features = self._adapt_channels(
                student_features, teacher_features.shape[1]
            )
        
        # Normalize if needed
        if self.normalize:
            student_features = F.normalize(student_features, dim=1)
            teacher_features = F.normalize(teacher_features, dim=1)
        
        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(student_features, teacher_features)
        elif self.loss_type == "l1":
            loss = F.l1_loss(student_features, teacher_features)
        elif self.loss_type == "cosine":
            # Flatten for cosine similarity
            s_flat = student_features.flatten(1)
            t_flat = teacher_features.flatten(1)
            loss = 1 - F.cosine_similarity(s_flat, t_flat, dim=1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _adapt_channels(
        self,
        features: torch.Tensor,
        target_channels: int,
    ) -> torch.Tensor:
        """Adapt feature channels using 1x1 convolution."""
        current_channels = features.shape[1]
        if current_channels == target_channels:
            return features
        
        # Use adaptive average pooling for channel adaptation
        if target_channels > current_channels:
            # Pad channels
            pad_size = target_channels - current_channels
            features = F.pad(features, (0, 0, 0, 0, 0, pad_size))
        else:
            # Average pool channels
            features = features[:, :target_channels]
        
        return features


class RelationDistillationLoss(nn.Module):
    """
    Relation-based distillation loss.
    
    Distills knowledge through the relationships between samples
    or features (e.g., attention maps, similarity matrices).
    """
    
    def __init__(
        self,
        loss_type: str = "spatial_attention",
    ):
        """
        Initialize relation distillation loss.
        
        Args:
            loss_type: Type of relation ("spatial_attention", "channel_attention", "similarity")
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relation distillation loss.
        
        Args:
            student_features: Student feature maps (B, C, H, W)
            teacher_features: Teacher feature maps (B, C, H, W)
            
        Returns:
            Distillation loss
        """
        if self.loss_type == "spatial_attention":
            return self._spatial_attention_loss(student_features, teacher_features)
        elif self.loss_type == "channel_attention":
            return self._channel_attention_loss(student_features, teacher_features)
        elif self.loss_type == "similarity":
            return self._similarity_loss(student_features, teacher_features)
        else:
            raise ValueError(f"Unknown relation type: {self.loss_type}")
    
    def _spatial_attention_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial attention-based loss."""
        # Compute spatial attention maps
        student_attention = self._compute_spatial_attention(student_features)
        teacher_attention = self._compute_spatial_attention(teacher_features)
        
        # Resize if needed
        if student_attention.shape != teacher_attention.shape:
            student_attention = F.interpolate(
                student_attention.unsqueeze(1),
                size=teacher_attention.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        
        return F.mse_loss(student_attention, teacher_attention)
    
    def _channel_attention_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute channel attention-based loss."""
        # Compute channel attention vectors
        student_attention = self._compute_channel_attention(student_features)
        teacher_attention = self._compute_channel_attention(teacher_features)
        
        # Handle different channel dimensions
        min_channels = min(student_attention.shape[1], teacher_attention.shape[1])
        student_attention = student_attention[:, :min_channels]
        teacher_attention = teacher_attention[:, :min_channels]
        
        return F.mse_loss(student_attention, teacher_attention)
    
    def _similarity_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity matrix-based loss."""
        # Flatten features
        B, C, H, W = student_features.shape
        student_flat = student_features.view(B, C, -1)  # (B, C, H*W)
        teacher_flat = teacher_features.view(B, C, -1)
        
        # Compute similarity matrices
        student_sim = torch.bmm(student_flat.transpose(1, 2), student_flat)
        teacher_sim = torch.bmm(teacher_flat.transpose(1, 2), teacher_flat)
        
        # Normalize
        student_sim = F.normalize(student_sim, dim=-1)
        teacher_sim = F.normalize(teacher_sim, dim=-1)
        
        return F.mse_loss(student_sim, teacher_sim)
    
    def _compute_spatial_attention(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial attention map."""
        # Average and max pooling along channel dimension
        avg_pool = features.mean(dim=1)
        max_pool = features.max(dim=1)[0]
        
        # Concatenate and compute attention
        attention = torch.stack([avg_pool, max_pool], dim=1)
        attention = attention.mean(dim=1)
        
        return attention
    
    def _compute_channel_attention(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute channel attention vector."""
        # Global average and max pooling
        avg_pool = features.mean(dim=[2, 3], keepdim=True)
        max_pool = features.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        
        # Concatenate
        attention = torch.cat([avg_pool, max_pool], dim=1)
        
        return attention


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.
    
    Combines response, feature, and relation distillation losses
    with configurable weights.
    """
    
    def __init__(
        self,
        response_weight: float = 1.0,
        feature_weight: float = 0.5,
        relation_weight: float = 0.3,
        temperature: float = 4.0,
        feature_loss_type: str = "mse",
        relation_loss_type: str = "spatial_attention",
    ):
        """
        Initialize combined distillation loss.
        
        Args:
            response_weight: Weight for response distillation
            feature_weight: Weight for feature distillation
            relation_weight: Weight for relation distillation
            temperature: Temperature for response distillation
            feature_loss_type: Loss type for feature distillation
            relation_loss_type: Loss type for relation distillation
        """
        super().__init__()
        
        self.response_weight = response_weight
        self.feature_weight = feature_weight
        self.relation_weight = relation_weight
        
        # Initialize loss modules
        self.response_loss = ResponseDistillationLoss(temperature=temperature)
        self.feature_loss = FeatureDistillationLoss(loss_type=feature_loss_type)
        self.relation_loss = RelationDistillationLoss(loss_type=relation_loss_type)
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Response distillation
        if self.response_weight > 0:
            student_logits = student_outputs.get("cls_pred", student_outputs.get("logits"))
            teacher_logits = teacher_outputs.get("cls_pred", teacher_outputs.get("logits"))
            
            if student_logits is not None and teacher_logits is not None:
                response_loss = self.response_loss(student_logits, teacher_logits)
                losses["response_loss"] = response_loss
                total_loss += self.response_weight * response_loss
        
        # Feature distillation
        if self.feature_weight > 0 and student_features and teacher_features:
            feature_loss = 0.0
            num_layers = 0
            
            for layer_name, s_feat in student_features.items():
                if layer_name in teacher_features:
                    t_feat = teacher_features[layer_name]
                    feature_loss += self.feature_loss(s_feat, t_feat)
                    num_layers += 1
            
            if num_layers > 0:
                feature_loss /= num_layers
                losses["feature_loss"] = feature_loss
                total_loss += self.feature_weight * feature_loss
        
        # Relation distillation
        if self.relation_weight > 0 and student_features and teacher_features:
            relation_loss = 0.0
            num_layers = 0
            
            for layer_name, s_feat in student_features.items():
                if layer_name in teacher_features:
                    t_feat = teacher_features[layer_name]
                    relation_loss += self.relation_loss(s_feat, t_feat)
                    num_layers += 1
            
            if num_layers > 0:
                relation_loss /= num_layers
                losses["relation_loss"] = relation_loss
                total_loss += self.relation_weight * relation_loss
        
        losses["total_distill_loss"] = total_loss
        
        return losses
