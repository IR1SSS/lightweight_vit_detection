"""
ONNX Export Script for Lightweight ViT Detection System.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config_loader import load_config
from src.models.mobilevit import build_mobilevit
from src.deployment.onnx_exporter import export_to_onnx


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], help='Input size [H, W]')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Build model
    print("Building model...")
    model = build_mobilevit(config)
    model.eval()
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Export config
    export_config = {
        'opset_version': 13,
        'simplify': False,
        'validate': False
    }
    
    # Input shape
    input_shape = (1, 3, args.input_size[0], args.input_size[1])
    print(f"Input shape: {input_shape}")
    
    # Export
    print(f"Exporting to ONNX: {args.output}")
    export_to_onnx(model, args.output, input_shape, export_config)
    print("Export complete!")


if __name__ == '__main__':
    main()
