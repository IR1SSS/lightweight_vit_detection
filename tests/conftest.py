"""
Pytest configuration for Lightweight ViT Detection System.

This module provides shared fixtures and configuration for tests.
"""

import os
import sys
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_config():
    """Get sample model configuration."""
    return {
        'model': {
            'name': 'mobilevit_detection',
            'type': 'mobilevit',
            'input': {
                'image_size': [640, 640],
                'channels': 3,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'backbone': {
                'name': 'mobilevit_s',
                'mobilevit_block': {
                    'dims': [96, 128, 160],
                    'depths': [2, 4, 3],
                    'expansion': 4,
                    'patch_size': 2
                },
                'mv2_block': {
                    'channels': [16, 32, 48, 64, 80, 96],
                    'expansion': 4
                }
            },
            'neck': {
                'type': 'fpn',
                'out_channels': 256,
                'num_outs': 5
            },
            'head': {
                'type': 'retina',
                'num_classes': 80,
                'in_channels': 256,
                'feat_channels': 256,
                'stacked_convs': 4,
                'anchor': {
                    'ratios': [0.5, 1.0, 2.0],
                    'scales': [1.0]
                }
            }
        }
    }


@pytest.fixture
def sample_image():
    """Get sample input image tensor."""
    return torch.randn(1, 3, 640, 640)


@pytest.fixture
def sample_batch():
    """Get sample batch of images."""
    return torch.randn(4, 3, 640, 640)


@pytest.fixture
def sample_targets():
    """Get sample target annotations."""
    return [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'image_id': torch.tensor([1]),
            'area': torch.tensor([10000, 10000], dtype=torch.float32),
            'iscrowd': torch.tensor([0, 0], dtype=torch.int64)
        }
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Get temporary directory for test outputs."""
    return str(tmp_path)
