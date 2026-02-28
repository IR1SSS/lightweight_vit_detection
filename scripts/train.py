#!/usr/bin/env python
"""
Training Script for Lightweight ViT Detection System.

This script provides the main entry point for training detection models.

Usage:
    python train.py --config configs/model/mobilevit.yaml
    python train.py --config configs/model/mobilevit.yaml --resume outputs/checkpoint.pth
"""

import os
import sys
import argparse
from datetime import datetime

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import setup_logger, get_logger
from src.models.mobilevit import build_mobilevit
from src.data.datasets import build_dataset
from src.data.dataloader import build_train_dataloader, build_val_dataloader
from src.training.trainer import Trainer, TrainerConfig
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.metrics import COCOEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Lightweight ViT Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to model configuration file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--training-config', type=str, default=None,
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to train on (cuda/cpu)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (overrides config)'
    )
    
    # Distributed training
    parser.add_argument(
        '--distributed', action='store_true',
        help='Use distributed training'
    )
    parser.add_argument(
        '--local-rank', type=int, default=0,
        help='Local rank for distributed training'
    )
    
    # Debugging
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()


def setup_distributed(args):
    """Setup distributed training."""
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return True
    return False


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logger(
        'train',
        log_dir=log_dir,
        log_file=f'train_{timestamp}.log'
    )
    
    logger.info("=" * 60)
    logger.info("Lightweight ViT Detection - Training")
    logger.info("=" * 60)
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    if args.training_config:
        training_config = load_config(args.training_config)
        config = merge_configs(config, training_config)
        
    # Override config with command line args
    if args.epochs is not None:
        config['training'] = config.get('training', {})
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training'] = config.get('training', {})
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training'] = config.get('training', {})
        config['training']['optimizer'] = config['training'].get('optimizer', {})
        config['training']['optimizer']['lr'] = args.lr
        
    # Setup distributed training
    distributed = setup_distributed(args)
    is_main_process = args.local_rank == 0
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda', args.local_rank if distributed else 0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
        
    # Build model
    logger.info("Building model...")
    model = build_mobilevit(config)
    model = model.to(device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Build datasets
    logger.info("Building datasets...")
    data_config = config.get('data', {})
    
    train_dataset = build_dataset(data_config, split='train')
    val_dataset = build_dataset(data_config, split='val')
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Build data loaders
    training_config = config.get('training', {})
    
    train_loader = build_train_dataloader(
        train_dataset, training_config, distributed
    )
    val_loader = build_val_dataloader(
        val_dataset, training_config, distributed
    )
    
    # Build optimizer and scheduler
    logger.info("Building optimizer...")
    optimizer = build_optimizer(
        model, training_config.get('optimizer', {})
    )
    
    scheduler = build_scheduler(
        optimizer,
        training_config.get('scheduler', {}),
        steps_per_epoch=len(train_loader)
    )
    
    # Build evaluator
    evaluator = COCOEvaluator(
        ann_file=data_config.get('ann_val')
    )
    
    # Create trainer config
    trainer_config = TrainerConfig(
        epochs=training_config.get('epochs', 5),
        log_interval=training_config.get('log_interval', 50),
        eval_interval=training_config.get('eval_interval', 5),
        save_interval=training_config.get('save_interval', 10),
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
        device=str(device)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=trainer_config,
        evaluator=evaluator
    )
    
    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Start training
    logger.info("Starting training...")
    metrics = trainer.train()
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best mAP: {metrics.get('best_mAP', 0):.4f}")
    logger.info("=" * 60)
    
    # Cleanup
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
