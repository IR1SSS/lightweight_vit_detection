#!/usr/bin/env python
"""
Training script for the Lightweight ViT Detection System.

Usage:
    python tools/train.py --config configs/train/distillation.yaml
    python tools/train.py --config configs/train/distillation.yaml --resume checkpoints/last.pth
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Filter repeated model registration warnings from timm
# This happens when efficientformer.py is imported multiple times in multi-worker dataloaders
warnings.filterwarnings(
    "ignore",
    message="Overwriting .* in registry",
    category=UserWarning,
)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger, get_logger
from src.utils.checkpoint import CheckpointManager
from src.models.detector import build_detector
from src.data.dataloader import create_dataloaders
from src.distillation.trainer import DistillationTrainer, DetectionLoss
from src.distillation.efficientformer_teacher import EfficientFormerV2Teacher


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # For distillation config, load student model config
    if hasattr(config, 'student') and hasattr(config.student, 'config'):
        student_config_path = config.student.config
        student_cfg = load_config(student_config_path)
        # Merge student model config into main config
        config = merge_configs(config, student_cfg)
    
    # Setup output directory
    output_dir = args.output_dir or config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        name="vit_detection",
        log_dir=output_dir,
        log_file="train.log",
    )
    logger.info(f"Configuration loaded from: {args.config}")
    if hasattr(config, 'model'):
        logger.info(f"Model config: {config.model.name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    if hasattr(config.experiment, "seed"):
        torch.manual_seed(config.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.experiment.seed)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    # Convert relative paths to absolute paths
    train_root_path = Path(config.data.train_root)
    val_root_path = Path(config.data.val_root)
    train_ann_path = Path(config.data.train_ann)
    val_ann_path = Path(config.data.val_ann)
    
    if not train_root_path.is_absolute():
        train_root_path = ROOT / train_root_path
    if not val_root_path.is_absolute():
        val_root_path = ROOT / val_root_path
    if not train_ann_path.is_absolute():
        train_ann_path = ROOT / train_ann_path
    if not val_ann_path.is_absolute():
        val_ann_path = ROOT / val_ann_path
    
    train_loader, val_loader = create_dataloaders(
        train_root=str(train_root_path),
        train_ann=str(train_ann_path),
        val_root=str(val_root_path),
        val_ann=str(val_ann_path),
        image_size=config.data.image_size,
        batch_size=args.batch_size or config.data.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Build student model
    logger.info("Building student model...")
    student_model = build_detector(config)
    logger.info(f"Student model: {student_model.backbone_name}")
    logger.info(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Build teacher model (for distillation)
    teacher_model = None
    use_distillation = config.distillation.enabled
    
    if use_distillation:
        # Check for EfficientFormerV2-S1 teacher (special handling)
        if config.teacher.get('type') == 'efficientformerv2_s1':
            logger.info("Building EfficientFormerV2-S1 teacher model...")
            teacher_model = EfficientFormerV2Teacher(
                weights_path=config.teacher.weights,
                resolution=config.teacher.get('resolution', 224),
                freeze=True,
            )
            logger.info(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
        else:
            # Standard teacher model loading
            logger.info("Building teacher model...")
            teacher_config = load_config(config.teacher.config)
            teacher_model = build_detector(teacher_config)
            
            # Load teacher weights
            if config.teacher.weights:
                checkpoint = torch.load(config.teacher.weights, map_location=device)
                if "model_state_dict" in checkpoint:
                    teacher_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    teacher_model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded teacher weights from: {config.teacher.weights}")
            
            teacher_model.eval()
            logger.info(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    
    # Move models to device
    student_model.to(device)
    if teacher_model:
        teacher_model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(
        student_model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=tuple(config.training.optimizer.betas),
    )
    
    # Setup scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=config.training.scheduler.min_lr,
    )
    
    # Setup trainer
    if teacher_model and config.distillation.enabled:
        # Get feature channels for adapter
        teacher_channels = None
        student_channels = None
        
        if config.teacher.get('type') == 'efficientformerv2_s1':
            # EfficientFormerV2-S1 channels: [32, 48, 120, 224]
            teacher_channels = [32, 48, 120, 224]
            logger.info(f"Teacher channels (EfficientFormerV2-S1): {teacher_channels}")
        
        # Get student channels from model
        if hasattr(student_model, 'backbone') and hasattr(student_model.backbone, 'get_output_channels'):
            student_channels = student_model.backbone.get_output_channels()
            logger.info(f"Student channels: {student_channels}")
        
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            distill_config=config.distillation.strategies,
            teacher_channels=teacher_channels,
            student_channels=student_channels,
            gradient_clip=getattr(config.training, 'gradient_clip', 1.0),
        )
    else:
        # Standard training (without distillation)
        from src.distillation.trainer import DetectionLoss
        
        criterion = DetectionLoss()
        trainer = StandardTrainer(
            model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            output_dir=output_dir,
        )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from: {args.resume}")
    
    # Train
    logger.info("Starting training...")
    results = trainer.train(
        num_epochs=config.training.epochs,
        eval_interval=config.evaluation.eval_interval,
        log_interval=config.logging.log_interval,
    )
    
    logger.info(f"Training completed. Best mAP: {results.get('best_mAP', 0):.4f}")


class StandardTrainer:
    """Standard trainer without distillation."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        criterion=None,
        device="cuda",
        output_dir="./outputs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or DetectionLoss()
        self.device = device
        self.output_dir = output_dir
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=output_dir,
            max_checkpoints=5,
            save_best=True,
        )
        self.logger = get_logger(__name__)
        self.current_epoch = 0
        self.best_mAP = 0.0
    
    def train(self, num_epochs, eval_interval=10, log_interval=100):
        """Run training."""
        import time
        from tqdm import tqdm
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                images = batch["images"].to(self.device)
                targets = self._move_targets(batch["targets"])
                
                # Forward
                outputs = self.model(images)
                
                # Loss
                loss = self.criterion(outputs, targets)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Evaluate
            if (epoch + 1) % eval_interval == 0:
                metrics = self.evaluate()
                self.checkpoint_manager.save(
                    self.model, epoch, metrics,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
        
        return {"best_mAP": self.best_mAP}
    
    def evaluate(self):
        """Evaluate model."""
        # Simplified evaluation
        return {"mAP": 0.0}
    
    def _move_targets(self, targets):
        """Move targets to device."""
        return [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
    
    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1


if __name__ == "__main__":
    main()
