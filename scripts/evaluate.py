#!/usr/bin/env python
"""
Evaluation Script for Lightweight ViT Detection System.

This script evaluates trained models on COCO validation set
and reports mAP metrics.

Usage:
    python evaluate.py --model outputs/best_model.pth --config configs/model/mobilevit.yaml
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.models.mobilevit import build_mobilevit
from src.data.datasets import build_dataset
from src.data.dataloader import build_val_dataloader
from src.training.metrics import COCOEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--data-config', type=str, default=None,
        help='Path to data configuration (if different from model config)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run evaluation on'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Evaluation batch size'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--save-predictions', action='store_true',
        help='Save predictions to JSON file'
    )
    
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, evaluator, device, save_predictions=False, output_dir=None):
    """
    Run evaluation on dataset.
    
    Args:
        model: Detection model.
        dataloader: Validation data loader.
        evaluator: COCO evaluator instance.
        device: Device to run on.
        save_predictions: Whether to save predictions.
        output_dir: Output directory for predictions.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    coco_results = []
    
    print("Running evaluation...")
    
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        
        # Run inference
        outputs = model(images)
        
        # Collect predictions
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # Extract predictions for this image
            pred = {
                'boxes': outputs.get('boxes', [torch.tensor([])])[i] if isinstance(outputs.get('boxes'), list) else outputs.get('boxes', torch.tensor([]))[i] if outputs.get('boxes') is not None else torch.tensor([]),
                'scores': outputs.get('scores', [torch.tensor([])])[i] if isinstance(outputs.get('scores'), list) else outputs.get('scores', torch.tensor([]))[i] if outputs.get('scores') is not None else torch.tensor([]),
                'labels': outputs.get('labels', [torch.tensor([])])[i] if isinstance(outputs.get('labels'), list) else outputs.get('labels', torch.tensor([]))[i] if outputs.get('labels') is not None else torch.tensor([])
            }
            
            all_predictions.append(pred)
            all_targets.append(targets[i])
            
            # Convert to COCO format for saving
            if save_predictions:
                image_id = targets[i]['image_id'].item()
                
                if 'boxes' in pred and len(pred['boxes']) > 0:
                    boxes = pred['boxes'].cpu().numpy()
                    scores = pred['scores'].cpu().numpy() if 'scores' in pred else [1.0] * len(boxes)
                    labels = pred['labels'].cpu().numpy() if 'labels' in pred else [0] * len(boxes)
                    
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        coco_results.append({
                            'image_id': int(image_id),
                            'category_id': int(label) + 1,
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'score': float(score)
                        })
                        
    # Save predictions
    if save_predictions and output_dir:
        predictions_path = os.path.join(output_dir, 'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(coco_results, f)
        print(f"Predictions saved to: {predictions_path}")
        
    # Compute metrics
    metrics = evaluator(all_predictions, all_targets)
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger(
        'evaluate',
        log_dir=args.output_dir,
        log_file=f'evaluate_{timestamp}.log'
    )
    
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    if args.data_config:
        data_config = load_config(args.data_config)
    else:
        data_config = config.get('data', {})
        
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building model...")
    model = build_mobilevit(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"Loaded checkpoint from epoch: {epoch}")
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Build dataset and dataloader
    logger.info("Building validation dataset...")
    val_dataset = build_dataset(data_config, split='val')
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    val_loader = build_val_dataloader(
        val_dataset,
        {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    )
    
    # Build evaluator
    ann_file = data_config.get('ann_val', '')
    evaluator = COCOEvaluator(ann_file=ann_file if os.path.exists(ann_file) else None)
    
    # Run evaluation
    metrics = evaluate(
        model, val_loader, evaluator, device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info("=" * 60)
    
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
        
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    # Check against targets
    target_map = 0.75
    achieved_map = metrics.get('mAP50', 0)
    
    if achieved_map >= target_map:
        logger.info(f"Target mAP@0.5 >= {target_map} ACHIEVED!")
    else:
        logger.info(f"Target mAP@0.5 >= {target_map} not yet achieved. Current: {achieved_map:.4f}")


if __name__ == '__main__':
    main()
