"""Lightweight ViT Detection System.

A lightweight Vision Transformer-based object detection system
optimized for mobile and edge device deployment.

Main modules:
- models: Model architectures (MobileViT, EfficientFormer, etc.)
- distillation: Knowledge distillation framework
- data: Data loading and transforms
- training: Training utilities and metrics
- quantization: Model quantization for deployment
- deployment: ONNX export and inference engines
- utils: Configuration, logging, and visualization
- applications: Demo applications
- gui: Desktop GUI application (requires PySide6)
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import models
from . import distillation
from . import data
from . import training
from . import quantization
from . import deployment
from . import utils
from . import applications

# GUI module is optional (requires PySide6)
try:
    from . import gui
    _has_gui = True
except ImportError:
    _has_gui = False

__all__ = [
    "models",
    "distillation",
    "data",
    "training",
    "quantization",
    "deployment",
    "utils",
    "applications",
]

if _has_gui:
    __all__.append("gui")
