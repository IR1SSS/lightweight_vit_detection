#!/usr/bin/env python
"""
Model quantization script for the Lightweight ViT Detection System.

Supports:
- Dynamic quantization (post-training)
- Static quantization with calibration (post-training)
- Quantization-aware training (QAT)

Usage:
    # Dynamic quantization (fastest, no calibration needed)
    python tools/quantize.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --method dynamic

    # Static quantization with calibration
    python tools/quantize.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --method static --calibration-data ./data/coco/val2017

    # Quantization-aware training
    python tools/quantize.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --method qat --epochs 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.models.detector import build_detector
from src.optimization.quantization import (
    QuantizationAwareTraining,
    quantize_model_dynamic,
    quantize_model_static,
    calibrate_model,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantize detection model")
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
        "--method",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "qat"],
        help="Quantization method",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/quantized_model.pth",
        help="Output path for quantized model",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fbgemm",
        choices=["fbgemm", "qnnpack"],
        help="Quantization backend (fbgemm for x86, qnnpack for ARM)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        choices=["qint8", "fp16"],
        help="Target data type for quantization",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration data (required for static quantization)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=500,
        help="Number of calibration samples",
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
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (quantization typically runs on CPU)",
    )
    # QAT specific arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of QAT fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for QAT fine-tuning",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data for QAT",
    )
    return parser.parse_args()


def create_calibration_dataloader(data_path: str, batch_size: int, num_samples: int, image_size: int):
    """
    Create a calibration data loader.
    
    Args:
        data_path: Path to calibration images
        batch_size: Batch size
        num_samples: Number of samples to use
        image_size: Image size
        
    Returns:
        DataLoader for calibration
    """
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import glob
    import numpy as np
    
    class CalibrationDataset(Dataset):
        def __init__(self, image_paths, image_size):
            self.image_paths = image_paths[:num_samples]
            self.image_size = image_size
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img).transpose(2, 0, 1) / 255.0
            return torch.from_numpy(img).float()
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(Path(data_path) / '**' / ext), recursive=True))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_path}")
    
    print(f"Found {len(image_paths)} images, using {min(num_samples, len(image_paths))} for calibration")
    
    dataset = CalibrationDataset(image_paths, image_size, num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def dynamic_quantization(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """
    Apply dynamic quantization.
    
    Args:
        model: Model to quantize
        dtype: Target data type
        
    Returns:
        Quantized model
    """
    print("Applying dynamic quantization...")
    
    # Dynamic quantization works best on linear layers
    # For ViT models, we quantize both Linear and Conv2d
    quantized_model = quantize_model_dynamic(model, dtype=dtype)
    
    return quantized_model


def static_quantization(
    model: nn.Module,
    calibration_loader,
    backend: str,
) -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Args:
        model: Model to quantize
        calibration_loader: Calibration data loader
        backend: Quantization backend
        
    Returns:
        Quantized model
    """
    print("Applying static quantization with calibration...")
    
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Apply static quantization
    quantized_model = quantize_model_static(model, calibration_loader, backend=backend)
    
    return quantized_model


def quantization_aware_training(
    model: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    backend: str,
    device: str,
) -> nn.Module:
    """
    Apply quantization-aware training.
    
    Args:
        model: Model to quantize
        train_loader: Training data loader
        epochs: Number of epochs
        lr: Learning rate
        backend: Quantization backend
        device: Device to use
        
    Returns:
        Quantized model
    """
    print("Starting quantization-aware training...")
    
    # Initialize QAT
    qat = QuantizationAwareTraining(model, backend=backend)
    
    # Prepare model for QAT
    prepared_model = qat.prepare()
    prepared_model.to(device)
    prepared_model.train()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(prepared_model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                images = batch.get("images", batch.get("image"))
            else:
                images = batch[0]
            
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = prepared_model(images)
            
            # Compute loss (simplified - in practice use detection loss)
            if isinstance(outputs, tuple):
                loss = sum(o.mean() for o in outputs if o.requires_grad)
            else:
                loss = outputs.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")
    
    # Convert to quantized model
    quantized_model = qat.convert()
    
    return quantized_model


def compare_model_size(original: nn.Module, quantized: nn.Module) -> None:
    """
    Compare model sizes.
    
    Args:
        original: Original model
        quantized: Quantized model
    """
    import os
    import tempfile
    
    # Save models temporarily to get sizes
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f1:
        torch.save(original.state_dict(), f1.name)
        original_size = os.path.getsize(f1.name) / (1024 * 1024)
        os.unlink(f1.name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f2:
        torch.save(quantized.state_dict(), f2.name)
        quantized_size = os.path.getsize(f2.name) / (1024 * 1024)
        os.unlink(f2.name)
    
    print(f"\n{'='*50}")
    print("Model Size Comparison")
    print(f"{'='*50}")
    print(f"Original model:  {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Compression:     {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"{'='*50}")


def main():
    """Main quantization function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name="quantize")
    logger.info("Starting model quantization...")
    logger.info(f"Method: {args.method}")
    logger.info(f"Backend: {args.backend}")
    
    # Set device (quantization typically runs on CPU)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    config = load_config(args.config)
    model = build_detector(config)
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    logger.info(f"Loaded weights from: {args.weights}")
    
    # Store original model for comparison
    original_model = model
    
    # Apply quantization based on method
    if args.method == "dynamic":
        dtype = torch.qint8 if args.dtype == "qint8" else torch.float16
        quantized_model = dynamic_quantization(model, dtype)
        
    elif args.method == "static":
        if args.calibration_data is None:
            logger.error("--calibration-data is required for static quantization")
            sys.exit(1)
        
        calibration_loader = create_calibration_dataloader(
            args.calibration_data,
            args.batch_size,
            args.calibration_samples,
            args.image_size,
        )
        quantized_model = static_quantization(model, calibration_loader, args.backend)
        
    elif args.method == "qat":
        if args.train_data is None:
            logger.error("--train-data is required for QAT")
            sys.exit(1)
        
        train_loader = create_calibration_dataloader(
            args.train_data,
            args.batch_size,
            args.calibration_samples,
            args.image_size,
        )
        quantized_model = quantization_aware_training(
            model, train_loader, args.epochs, args.lr, args.backend, args.device
        )
    
    # Compare sizes
    try:
        compare_model_size(original_model, quantized_model)
    except Exception as e:
        logger.warning(f"Could not compare model sizes: {e}")
    
    # Save quantized model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    save_dict = {
        "model_state_dict": quantized_model.state_dict(),
        "config": config,
        "quantization_method": args.method,
        "backend": args.backend,
    }
    torch.save(save_dict, output_path)
    logger.info(f"Quantized model saved to: {output_path}")
    
    logger.info("\nQuantization completed successfully!")


if __name__ == "__main__":
    main()
