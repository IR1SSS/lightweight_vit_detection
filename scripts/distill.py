#!/usr/bin/env python
"""
Knowledge Distillation Training Script.

This script trains a student model using knowledge distillation
from a pre-trained teacher model.

Usage:
    python distill.py --config configs/training/distillation.yaml
"""

import os
import sys
import argparse
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.models.mobilevit import build_mobilevit
from src.data.datasets import build_dataset
from src.data.dataloader import build_train_dataloader, build_val_dataloader
from src.distillation.base_distiller import build_distiller, DistillationTrainer
from src.training.optimizer import build_optimizer, build_scheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to distillation configuration file'
    )
    parser.add_argument(
        '--teacher-model', type=str, default=None,
        help='Path to teacher model checkpoint'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/distillation',
        help='Output directory'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to train on'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from checkpoint'
    )
    
    return parser.parse_args()


def load_teacher_model(config: dict, device: torch.device) -> torch.nn.Module:
    """
    Load pre-trained teacher model.
    
    Args:
        config: Distillation configuration.
        device: Device to load model on.
        
    Returns:
        Loaded teacher model.
    """
    teacher_cfg = config.get('distillation', {}).get('teacher', {})
    model_type = teacher_cfg.get('model_type', 'detr')
    model_path = teacher_cfg.get('model_path', '')
    
    # Load teacher model based on type
    # This is a placeholder - actual implementation would load specific models
    
    if model_type == 'detr':
        # Load DETR model
        try:
            from torchvision.models.detection import detr_resnet50
            teacher = detr_resnet50(pretrained=True)
        except ImportError:
            print("Warning: DETR not available. Using placeholder teacher.")
            teacher = build_mobilevit({})
    else:
        # Default: use MobileViT as teacher
        teacher = build_mobilevit({})
        
    # Load checkpoint if provided
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['model_state_dict'])
        else:
            teacher.load_state_dict(checkpoint)
            
    return teacher.to(device)


def main():
    """Main distillation training function."""
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(
        'distill',
        log_dir=log_dir,
        log_file=f'distill_{timestamp}.log'
    )
    
    logger.info("=" * 60)
    logger.info("Knowledge Distillation Training")
    logger.info("=" * 60)
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override teacher model path if provided
    if args.teacher_model:
        config['distillation']['teacher']['model_path'] = args.teacher_model
        
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load teacher model
    logger.info("Loading teacher model...")
    teacher = load_teacher_model(config, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    logger.info("Teacher model loaded and frozen")
    
    # Build student model
    logger.info("Building student model...")
    student_config_path = config.get('distillation', {}).get('student', {}).get('config_path', '')
    
    if os.path.exists(student_config_path):
        student_config = load_config(student_config_path)
    else:
        student_config = config
        
    student = build_mobilevit(student_config)
    student = student.to(device)
    
    num_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student parameters: {num_params / 1e6:.2f}M")
    
    # Build distiller
    logger.info("Building distiller...")
    distiller = build_distiller(teacher, student, config.get('distillation', {}))
    
    # Build datasets - use val dataset for quick validation
    logger.info("Building datasets...")
    data_config = config.get('data', {})
    
    # Use validation set for quick testing
    val_dataset = build_dataset(data_config, split='val')
    
    # Build data loaders
    training_config = config.get('training', {})
    train_loader = build_train_dataloader(val_dataset, training_config)
    val_loader = build_val_dataloader(val_dataset, training_config)
    
    # Build optimizer
    optimizer = build_optimizer(student, training_config.get('optimizer', {}))
    
    scheduler = build_scheduler(
        optimizer,
        training_config.get('scheduler', {}),
        steps_per_epoch=len(train_loader)
    )
    
    # Create distillation trainer
    distill_trainer = DistillationTrainer(
        distiller=distiller,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device)
    )
    
    # Training loop
    epochs = training_config.get('epochs', 5)
    max_batches = 0  # Full dataset for training
    logger.info(f"Starting distillation training for {epochs} epochs...")
    
    best_metric = 0.0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train epoch
        losses = distill_trainer.train_epoch(train_loader, max_batches=max_batches)
        
        loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
        logger.info(f"  Train: {loss_str}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.output_dir, 'checkpoints', f'distill_epoch_{epoch+1}.pth'
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
            
    logger.info("=" * 60)
    logger.info("Distillation training complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
