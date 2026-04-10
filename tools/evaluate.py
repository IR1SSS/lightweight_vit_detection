#!/usr/bin/env python
"""
Evaluation script for the Lightweight ViT Detection System.

Usage:
    python tools/evaluate.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import DetectionMetrics
from src.models.detector import build_detector
from src.data.coco_dataset import COCODataset
from src.data.transforms import Compose, Letterbox, Normalize


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate detection model")
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
        "--data-root",
        type=str,
        default="./data/coco/val2017",
        help="Path to validation images",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="./data/coco/annotations/instances_val2017.json",
        help="Path to annotation file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results",
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name="evaluate")
    logger.info("Starting evaluation...")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    config = load_config(args.config)
    model = build_detector(config)
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded weights from: {args.weights}")
    
    # Setup dataset
    logger.info("Loading dataset...")
    transforms = Compose([
        Letterbox(target_size=args.image_size),
        Normalize(),
    ])
    
    dataset = COCODataset(
        root=args.data_root,
        annotation_file=args.annotation,
        transforms=transforms,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Metrics
    metrics = DetectionMetrics(num_classes=80)
    
    # Run evaluation
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            image, target = dataset[idx]
            
            # Prepare input
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Forward pass
            cls_pred, reg_pred, obj_pred = model(input_tensor)
            
            # Get predictions
            predictions = decode_predictions(
                cls_pred, reg_pred, obj_pred,
                conf_threshold=0.05,
                nms_threshold=0.5,
            )
            
            # Update metrics
            metrics.update(
                {
                    "boxes": predictions["boxes"],
                    "scores": predictions["scores"],
                    "labels": predictions["labels"],
                },
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
            )
    
    # Compute final metrics
    results = metrics.compute()
    
    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    logger.info(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    logger.info("=" * 50)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {args.output}")


def decode_predictions(cls_pred, reg_pred, obj_pred, conf_threshold=0.05, nms_threshold=0.5):
    """Decode model predictions to detections."""
    import numpy as np
    
    # Convert to numpy
    cls_pred = cls_pred[0].cpu().numpy()
    reg_pred = reg_pred[0].cpu().numpy()
    obj_pred = obj_pred[0].cpu().numpy()
    
    # Get scores
    scores = cls_pred.sigmoid() * obj_pred.sigmoid()
    max_scores = scores.max(axis=-1)
    labels = scores.argmax(axis=-1)
    
    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes = reg_pred[mask]
    scores = max_scores[mask]
    labels = labels[mask]
    
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


if __name__ == "__main__":
    main()
