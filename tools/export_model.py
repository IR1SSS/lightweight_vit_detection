#!/usr/bin/env python
"""
Model export script for the Lightweight ViT Detection System.

Usage:
    python tools/export_model.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --format onnx
    python tools/export_model.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --format torchscript
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.models.detector import build_detector
from src.optimization.export import (
    export_onnx,
    export_torchscript,
    export_tensorrt,
    optimize_onnx,
    verify_onnx,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["onnx"],
        choices=["onnx", "torchscript", "tensorrt"],
        help="Export format(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exports",
        help="Output directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Input image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported model",
    )
    parser.add_argument(
        "--tensorrt-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="TensorRT precision",
    )
    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name="export")
    logger.info("Starting model export...")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    config = load_config(args.config)
    model = build_detector(config)
    
    # Load weights (weights_only=False for PyTorch 2.6+ compatibility)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    logger.info(f"Loaded weights from: {args.weights}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to each format
    exported_files = {}
    
    if "onnx" in args.format:
        logger.info("Exporting to ONNX...")
        onnx_path = output_dir / "model.onnx"
        
        export_onnx(
            model=model,
            output_path=str(onnx_path),
            input_shape=(3, args.image_size, args.image_size),
            batch_size=args.batch_size,
            simplify=args.simplify,
        )
        
        exported_files["onnx"] = str(onnx_path)
        logger.info(f"ONNX model saved to: {onnx_path}")
        
        # Verify if requested
        if args.verify:
            logger.info("Verifying ONNX model...")
            success = verify_onnx(
                str(onnx_path),
                model,
                input_shape=(3, args.image_size, args.image_size),
                batch_size=args.batch_size,
            )
            logger.info(f"Verification: {'PASSED' if success else 'FAILED'}")
    
    if "torchscript" in args.format:
        logger.info("Exporting to TorchScript...")
        ts_path = output_dir / "model.pt"
        
        export_torchscript(
            model=model,
            output_path=str(ts_path),
            input_shape=(3, args.image_size, args.image_size),
            batch_size=args.batch_size,
            method="trace",
        )
        
        exported_files["torchscript"] = str(ts_path)
        logger.info(f"TorchScript model saved to: {ts_path}")
    
    if "tensorrt" in args.format:
        if "onnx" not in exported_files:
            logger.error("ONNX export required for TensorRT. Exporting ONNX first...")
            onnx_path = output_dir / "model.onnx"
            export_onnx(
                model=model,
                output_path=str(onnx_path),
                input_shape=(3, args.image_size, args.image_size),
                batch_size=args.batch_size,
            )
            exported_files["onnx"] = str(onnx_path)
        
        logger.info("Exporting to TensorRT...")
        trt_path = output_dir / "model.engine"
        
        try:
            export_tensorrt(
                onnx_path=exported_files["onnx"],
                output_path=str(trt_path),
                precision=args.tensorrt_precision,
                max_batch_size=args.batch_size,
            )
            exported_files["tensorrt"] = str(trt_path)
            logger.info(f"TensorRT engine saved to: {trt_path}")
        except ImportError as e:
            logger.warning(f"TensorRT export failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Export Summary")
    logger.info("=" * 50)
    for fmt, path in exported_files.items():
        logger.info(f"{fmt.upper()}: {path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
