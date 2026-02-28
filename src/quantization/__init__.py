"""
Quantization module for Lightweight ViT Detection System.

This module provides quantization functionality including:
- Post-training quantization
- Quantization-aware training (QAT)
"""

from .qat_trainer import QATTrainer, prepare_qat_model

__all__ = [
    'QATTrainer',
    'prepare_qat_model',
]
