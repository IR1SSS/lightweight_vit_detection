"""
Distillation module for Lightweight ViT Detection System.

This module provides knowledge distillation functionality including:
- Base distiller class
- Various distillation loss functions
"""

from .base_distiller import BaseDistiller, DistillationTrainer
from .losses import (
    ResponseDistillationLoss,
    FeatureDistillationLoss,
    RelationDistillationLoss,
    CombinedDistillationLoss
)

__all__ = [
    'BaseDistiller',
    'DistillationTrainer',
    'ResponseDistillationLoss',
    'FeatureDistillationLoss',
    'RelationDistillationLoss',
    'CombinedDistillationLoss',
]
