"""
Training module for Lightweight ViT Detection System.

This module provides training functionality including:
- Trainer class for training loop management
- Optimizer construction utilities
- Metrics computation
"""

from .trainer import Trainer, TrainerConfig
from .optimizer import build_optimizer, build_scheduler
from .metrics import COCOEvaluator, compute_map

__all__ = [
    'Trainer',
    'TrainerConfig',
    'build_optimizer',
    'build_scheduler',
    'COCOEvaluator',
    'compute_map',
]
