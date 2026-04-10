#!/usr/bin/env python
"""
Model pruning script for the Lightweight ViT Detection System.

Supports:
- L1 unstructured pruning (weights level)
- L1 structured pruning (channel/filter level)
- Iterative pruning with fine-tuning

Usage:
    # L1 unstructured pruning (30% sparsity)
    python tools/prune.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --method l1_unstructured --ratio 0.3

    # L1 structured pruning (channel pruning)
    python tools/prune.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --method l1_structured --ratio 0.2

    # Iterative pruning with fine-tuning
    python tools/prune.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --method iterative --ratio 0.5 --iterations 5 --finetune-epochs 5
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
from src.optimization.pruning import (
    L1UnstructuredPruner,
    L1StructuredPruner,
    ChannelPruner,
    iterative_pruning,
    get_pruning_statistics,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prune detection model")
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
        default="l1_unstructured",
        choices=["l1_unstructured", "l1_structured", "channel", "iterative"],
        help="Pruning method",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Pruning ratio (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/pruned_model.pth",
        help="Output path for pruned model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=None,
        help="Specific layers to prune (default: all Conv2d and Linear)",
    )
    # Iterative pruning arguments
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of pruning iterations for iterative method",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="Fine-tuning epochs per iteration",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data for fine-tuning",
    )
    # Export options
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export pruned model to ONNX",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Input image size for ONNX export",
    )
    return parser.parse_args()


def print_model_info(model: nn.Module, title: str = "Model Info") -> None:
    """
    Print model information.
    
    Args:
        model: Model to analyze
        title: Title for the info block
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB (fp32)")
    print(f"{'='*50}")


def print_pruning_statistics(model: nn.Module) -> None:
    """
    Print pruning statistics.
    
    Args:
        model: Pruned model
    """
    stats = get_pruning_statistics(model)
    
    print(f"\n{'='*50}")
    print("Pruning Statistics")
    print(f"{'='*50}")
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Pruned parameters: {stats['pruned_params']:,}")
    print(f"Global sparsity: {stats['global_sparsity']:.2%}")
    print(f"{'='*50}")
    
    # Print per-layer statistics (top 10 sparsest layers)
    if stats['layer_stats']:
        print("\nTop 10 Sparsest Layers:")
        sorted_layers = sorted(
            stats['layer_stats'].items(),
            key=lambda x: x[1]['sparsity'],
            reverse=True
        )[:10]
        
        for name, layer_stats in sorted_layers:
            print(f"  {name}: {layer_stats['sparsity']:.2%} ({layer_stats['zeros']}/{layer_stats['total']})")


def create_train_loader(data_path: str, batch_size: int, image_size: int):
    """
    Create a training data loader for fine-tuning.
    
    Args:
        data_path: Path to training data
        batch_size: Batch size
        image_size: Image size
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import glob
    import numpy as np
    
    class SimpleDataset(Dataset):
        def __init__(self, image_paths, image_size):
            self.image_paths = image_paths
            self.image_size = image_size
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img).transpose(2, 0, 1) / 255.0
            return {"image": torch.from_numpy(img).float()}
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(Path(data_path) / '**' / ext), recursive=True))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_path}")
    
    print(f"Found {len(image_paths)} images for fine-tuning")
    
    dataset = SimpleDataset(image_paths, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def finetune_model(
    model: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    """
    Fine-tune pruned model.
    
    Args:
        model: Pruned model
        train_loader: Training data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        
    Returns:
        Fine-tuned model
    """
    model.train()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss (simplified)
            if isinstance(outputs, tuple):
                loss = sum(o.mean() for o in outputs if isinstance(o, torch.Tensor) and o.requires_grad)
            else:
                loss = outputs.mean() if isinstance(outputs, torch.Tensor) else torch.tensor(0.0, device=device, requires_grad=True)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def main():
    """Main pruning function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name="prune")
    logger.info("Starting model pruning...")
    logger.info(f"Method: {args.method}")
    logger.info(f"Target ratio: {args.ratio}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    
    # Print original model info
    print_model_info(model, "Original Model Info")
    
    # Apply pruning based on method
    if args.method == "l1_unstructured":
        logger.info("Applying L1 unstructured pruning...")
        pruner = L1UnstructuredPruner(
            model,
            pruning_ratio=args.ratio,
            prune_layers=args.layers,
        )
        model = pruner.prune()
        
    elif args.method == "l1_structured":
        logger.info("Applying L1 structured pruning...")
        pruner = L1StructuredPruner(
            model,
            pruning_ratio=args.ratio,
            prune_layers=args.layers,
        )
        model = pruner.prune()
        
    elif args.method == "channel":
        logger.info("Applying channel pruning...")
        pruner = ChannelPruner(
            model,
            pruning_ratio=args.ratio,
            input_shape=(3, args.image_size, args.image_size),
        )
        model = pruner.prune()
        
    elif args.method == "iterative":
        if args.train_data is None:
            logger.error("--train-data is required for iterative pruning")
            sys.exit(1)
        
        logger.info("Applying iterative pruning with fine-tuning...")
        
        train_loader = create_train_loader(
            args.train_data,
            batch_size=8,
            image_size=args.image_size,
        )
        
        def finetune_fn(m):
            return finetune_model(
                m, train_loader, args.finetune_epochs,
                args.finetune_lr, str(device)
            )
        
        model = iterative_pruning(
            model,
            target_sparsity=args.ratio,
            num_iterations=args.iterations,
            train_fn=finetune_fn,
            pruner_class=L1UnstructuredPruner,
        )
    
    # Print pruning statistics
    print_pruning_statistics(model)
    
    # Print pruned model info
    print_model_info(model, "Pruned Model Info")
    
    # Remove pruning masks and make pruning permanent
    logger.info("Making pruning permanent...")
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            try:
                import torch.nn.utils.prune as prune
                prune.remove(module, 'weight')
            except ValueError:
                pass  # Already removed
    
    # Export to ONNX if requested
    if args.export_onnx:
        logger.info("Exporting pruned model to ONNX...")
        from src.optimization.export import export_onnx
        
        onnx_path = Path(args.output).with_suffix('.onnx')
        model.cpu()
        export_onnx(
            model=model,
            output_path=str(onnx_path),
            input_shape=(3, args.image_size, args.image_size),
            simplify=True,
        )
        logger.info(f"ONNX model saved to: {onnx_path}")
    
    # Save pruned model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "pruning_method": args.method,
        "pruning_ratio": args.ratio,
    }
    torch.save(save_dict, output_path)
    logger.info(f"Pruned model saved to: {output_path}")
    
    logger.info("\nPruning completed successfully!")


if __name__ == "__main__":
    main()
