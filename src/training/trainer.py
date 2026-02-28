"""
Trainer Class for Lightweight ViT Detection System.

This module implements the training loop with support for:
- Mixed precision training
- Gradient accumulation
- Checkpoint saving and loading
- Logging and metrics tracking
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ..utils.logger import get_logger


@dataclass
class TrainerConfig:
    """Configuration for Trainer."""
    
    # Training parameters
    epochs: int = 5
    log_interval: int = 50
    eval_interval: int = 5
    save_interval: int = 10
    max_batches_per_epoch: int = 0  # Full dataset for training
    
    # Optimization
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    use_amp: bool = True
    
    # Directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    
    # Device
    device: str = "cuda"


class Trainer:
    """
    Trainer class for object detection model training.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[TrainerConfig] = None,
        evaluator: Optional[Callable] = None
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: Detection model to train.
            optimizer: Optimizer instance.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            scheduler: Optional learning rate scheduler.
            config: Trainer configuration.
            evaluator: Optional evaluation function.
        """
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        
        # Mixed precision - only enable if CUDA is available
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Logging
        self.logger = get_logger(__name__)
        
        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of final metrics.
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}"
            )
            
            # Validation
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                # Clear GPU memory before validation
                torch.cuda.empty_cache()
                val_metrics = self.validate()
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Val mAP: {val_metrics.get('mAP', 0):.4f}"
                )
                
                # Save best model
                if val_metrics.get('mAP', 0) > self.best_metric:
                    self.best_metric = val_metrics['mAP']
                    self.save_checkpoint('best_model.pth')
                    
            # Regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
        return {'best_mAP': self.best_metric}
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        accumulation_counter = 0
        
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, targets)
                    loss = outputs.get('loss_total', outputs.get('loss', torch.tensor(0.0)))
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = self.model(images, targets)
                loss = outputs.get('loss_total', outputs.get('loss', torch.tensor(0.0)))
                loss = loss / self.config.accumulation_steps
                
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulation_counter += 1
            
            # Optimizer step
            if accumulation_counter >= self.config.accumulation_steps:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                accumulation_counter = 0
                self.global_step += 1
                
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"  Batch {batch_idx+1}/{len(self.train_loader)} - "
                    f"Loss: {total_loss/num_batches:.4f} - "
                    f"Time: {elapsed:.2f}s"
                )
            
            # Early stop for quick validation
            if self.config.max_batches_per_epoch > 0 and num_batches >= self.config.max_batches_per_epoch:
                break
                
        return {
            'loss': total_loss / num_batches,
            'time': time.time() - start_time
        }
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        for images, targets in self.val_loader:
            images = images.to(self.device)
            
            outputs = self.model(images)
            
            # Move outputs to CPU to save GPU memory
            outputs_cpu = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    outputs_cpu[k] = v.cpu()
                elif isinstance(v, list):
                    outputs_cpu[k] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in v]
                else:
                    outputs_cpu[k] = v
            
            all_predictions.append(outputs_cpu)
            all_targets.extend(targets)
            
            # Clear intermediate results
            del images, outputs
            torch.cuda.empty_cache()
            
        # Compute metrics
        if self.evaluator is not None:
            metrics = self.evaluator(all_predictions, all_targets)
        else:
            metrics = {'mAP': 0.0}  # Placeholder
            
        return metrics
        
    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")


class DistributedTrainer(Trainer):
    """
    Trainer with distributed data parallel support.
    
    Extends base Trainer for multi-GPU training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[TrainerConfig] = None,
        evaluator: Optional[Callable] = None,
        local_rank: int = 0
    ) -> None:
        """
        Initialize distributed trainer.
        
        Args:
            local_rank: Local process rank for distributed training.
        """
        super().__init__(
            model, optimizer, train_loader, val_loader,
            scheduler, config, evaluator
        )
        
        self.local_rank = local_rank
        self.is_main_process = local_rank == 0
        
        # Wrap model with DDP
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank
            )
            
    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint only on main process."""
        if self.is_main_process:
            super().save_checkpoint(filename)
