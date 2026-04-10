"""
Feature Adapter for Cross-Architecture Knowledge Distillation.

Handles feature alignment between teacher and student models
with different channel dimensions and spatial resolutions.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """
    Single-level feature adapter for channel and spatial alignment.
    
    Aligns student features to match teacher features using:
    - 1x1 convolution for channel alignment
    - Bilinear interpolation for spatial alignment
    """
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        mode: str = "conv",
    ):
        """
        Initialize feature adapter.
        
        Args:
            student_channels: Number of student feature channels
            teacher_channels: Number of teacher feature channels
            mode: Alignment mode ("conv", "linear", "pad")
        """
        super().__init__()
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.mode = mode
        
        if mode == "conv":
            # Learnable 1x1 convolution for channel alignment
            self.channel_align = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, bias=False
            )
            # Initialize with Xavier
            nn.init.xavier_uniform_(self.channel_align.weight)
        elif mode == "linear":
            # Linear projection (flatten -> project -> reshape)
            self.channel_align = nn.Linear(student_channels, teacher_channels, bias=False)
        else:
            self.channel_align = None
    
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align student features to match teacher features.
        
        Args:
            student_feat: Student feature tensor (B, C_s, H_s, W_s)
            teacher_feat: Teacher feature tensor (B, C_t, H_t, W_t)
            
        Returns:
            Tuple of (aligned student features, teacher features)
        """
        # Channel alignment
        if self.student_channels != self.teacher_channels:
            student_feat = self._align_channels(student_feat)
        
        # Spatial alignment
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        
        return student_feat, teacher_feat
    
    def _align_channels(self, feat: torch.Tensor) -> torch.Tensor:
        """Align feature channels based on mode."""
        if self.mode == "conv":
            return self.channel_align(feat)
        
        if self.mode == "linear":
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            feat = self.channel_align(feat)
            return feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Simple padding/truncation
        if self.teacher_channels > self.student_channels:
            pad = self.teacher_channels - self.student_channels
            return F.pad(feat, (0, 0, 0, 0, 0, pad))
        return feat[:, :self.teacher_channels]


class MultiLevelFeatureAdapter(nn.Module):
    """
    Multi-level feature adapter for hierarchical distillation.
    
    Handles feature alignment at multiple scales with learnable
    projection layers for each level.
    """
    
    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        mode: str = "conv",
    ):
        """
        Initialize multi-level feature adapter.
        
        Args:
            student_channels: List of student channel dimensions
            teacher_channels: List of teacher channel dimensions
            mode: Alignment mode for each level
        """
        super().__init__()
        
        self.num_levels = min(len(student_channels), len(teacher_channels))
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        
        # Create adapter for each level
        self.adapters = nn.ModuleList([
            FeatureAdapter(s_ch, t_ch, mode)
            for s_ch, t_ch in zip(student_channels[:self.num_levels], 
                                   teacher_channels[:self.num_levels])
        ])
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Align multi-level student features to teacher features.
        
        Args:
            student_features: List of student feature tensors
            teacher_features: List of teacher feature tensors
            
        Returns:
            Tuple of (aligned student features, teacher features)
        """
        aligned_student = []
        aligned_teacher = []
        
        for i, adapter in enumerate(self.adapters):
            s_aligned, t_feat = adapter(student_features[i], teacher_features[i])
            aligned_student.append(s_aligned)
            aligned_teacher.append(t_feat)
        
        return aligned_student, aligned_teacher


class FeatureProjectionHead(nn.Module):
    """
    Projection head for feature distillation.
    
    Projects features to a common embedding space for better
    knowledge transfer during distillation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 2,
    ):
        """
        Initialize projection head.
        
        Args:
            in_channels: Input channel dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of projection layers
        """
        super().__init__()
        
        layers: list = []
        current_ch = in_channels
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(current_ch, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
            current_ch = hidden_channels
        
        layers.append(nn.Conv2d(current_ch, out_channels, 1))
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space."""
        return self.projection(x)


def compute_feature_similarity(
    features: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute feature similarity matrix for relation distillation.
    
    Args:
        features: Feature tensor (B, C, H, W)
        normalize: Whether to normalize features
        
    Returns:
        Similarity matrix (B, H*W, H*W)
    """
    B, C, H, W = features.shape
    
    # Reshape to (B, C, H*W)
    feat = features.flatten(2)
    
    if normalize:
        feat = F.normalize(feat, dim=1)
    
    # Compute similarity: (B, H*W, H*W)
    similarity = torch.bmm(feat.transpose(1, 2), feat)
    
    return similarity


class FeatureDistillationWithAdapter(nn.Module):
    """
    Complete feature distillation module with adapters.
    
    Combines feature alignment and distillation loss computation
    for cross-architecture knowledge transfer.
    """
    
    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        projection_dim: int = 256,
        loss_type: str = "mse",
        use_projection: bool = True,
    ):
        """
        Initialize feature distillation with adapter.
        
        Args:
            student_channels: Student feature channel dimensions
            teacher_channels: Teacher feature channel dimensions
            projection_dim: Dimension for projection space
            loss_type: Type of distillation loss
            use_projection: Whether to use projection head
        """
        super().__init__()
        
        self.adapter = MultiLevelFeatureAdapter(
            student_channels, teacher_channels, mode="conv"
        )
        self.loss_type = loss_type
        self.use_projection = use_projection
        
        if use_projection:
            # Create projection heads for each level
            self.student_projections = nn.ModuleList([
                FeatureProjectionHead(t_ch, projection_dim, projection_dim)
                for t_ch in teacher_channels
            ])
            self.teacher_projections = nn.ModuleList([
                FeatureProjectionHead(t_ch, projection_dim, projection_dim)
                for t_ch in teacher_channels
            ])
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: Student multi-level features
            teacher_features: Teacher multi-level features
            
        Returns:
            Total distillation loss
        """
        aligned_student, aligned_teacher = self.adapter(
            student_features, teacher_features
        )
        
        total_loss = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(aligned_student, aligned_teacher)):
            total_loss = total_loss + self._compute_level_loss(i, s_feat, t_feat)

        return total_loss / len(aligned_student)
    
    def _compute_level_loss(
        self,
        level: int,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss for a single level."""
        if self.use_projection:
            s_proj = F.normalize(self.student_projections[level](s_feat), dim=1)
            t_proj = F.normalize(self.teacher_projections[level](t_feat.detach()), dim=1)
            return F.mse_loss(s_proj, t_proj)
        
        s_norm = F.normalize(s_feat, dim=1)
        t_norm = F.normalize(t_feat, dim=1)
        
        if self.loss_type == "cosine":
            return 1 - F.cosine_similarity(
                s_norm.flatten(1), t_norm.flatten(1), dim=1
            ).mean()
        if self.loss_type == "l1":
            return F.l1_loss(s_norm, t_norm)
        return F.mse_loss(s_norm, t_norm)
