#!/usr/bin/env python
"""
Lightweight ViT Detection System - Desktop Application Entry Point.

This script launches the GUI application for INFERENCE ONLY.
It loads pre-trained/distilled model weights and performs real-time object detection.

PREREQUISITES (should be completed before running this app):
    1. Model training: python scripts/train.py --config configs/model/mobilevit.yaml
    2. Knowledge distillation: python scripts/distill.py --config configs/training/distillation.yaml
    3. Model export (optional): python scripts/inference.py --export-onnx
    
The trained model weights should be placed in:
    - models/mobilevit_distilled.pth (recommended)
    - models/mobilevit_best.pth
    - outputs/best_model.pth

Usage:
    python run_app.py
    
For packaging as executable:
    python scripts/build_exe.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """Launch the GUI application."""
    # Import and create main window
    from src.gui.main_window_tk import MainWindow
    
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
