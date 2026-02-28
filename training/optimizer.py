"""
Optimizer and Scheduler Construction for Lightweight ViT Detection System.

This module provides utilities for building optimizers and learning rate
schedulers from configuration.
"""

from typing import Dict, Any, Optional, List, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> Optimizer:
    """
    Build optimizer from configuration.
    
    Args:
        model: Model to optimize.
        config: Optimizer configuration with keys:
            - type: Optimizer type ('adamw', 'sgd', 'adam')
            - lr: Learning rate
            - weight_decay: Weight decay factor
            - Additional optimizer-specific parameters
            
    Returns:
        Configured optimizer instance.
    """
    opt_type = config.get('type', 'adamw').lower()
    lr = config.get('lr', 0.0001)
    weight_decay = config.get('weight_decay', 0.05)
    
    # Separate parameters for weight decay
    params = get_param_groups(model, weight_decay)
    
    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(config.get('betas', [0.9, 0.999])),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=tuple(config.get('betas', [0.9, 0.999])),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', True)
        )
    elif opt_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
        
    return optimizer


def get_param_groups(
    model: nn.Module,
    weight_decay: float
) -> List[Dict[str, Any]]:
    """
    Get parameter groups with weight decay handling.
    
    Excludes bias and normalization layer parameters from weight decay.
    
    Args:
        model: Model to get parameters from.
        weight_decay: Weight decay factor for applicable parameters.
        
    Returns:
        List of parameter group dictionaries.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Skip weight decay for bias and normalization parameters
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def build_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    steps_per_epoch: Optional[int] = None
) -> _LRScheduler:
    """
    Build learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule.
        config: Scheduler configuration with keys:
            - type: Scheduler type ('cosine', 'step', 'multistep', 'exponential')
            - warmup_epochs: Number of warmup epochs
            - Additional scheduler-specific parameters
        steps_per_epoch: Number of steps per epoch (for warmup).
        
    Returns:
        Configured scheduler instance.
    """
    sched_type = config.get('type', 'cosine').lower()
    warmup_epochs = config.get('warmup_epochs', 5)
    
    if sched_type == 'cosine':
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=config.get('epochs', 300),
            min_lr=config.get('min_lr', 1e-6),
            steps_per_epoch=steps_per_epoch
        )
    elif sched_type == 'step':
        scheduler = WarmupStepLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1),
            steps_per_epoch=steps_per_epoch
        )
    elif sched_type == 'multistep':
        scheduler = WarmupMultiStepLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            milestones=config.get('milestones', [100, 200]),
            gamma=config.get('gamma', 0.1),
            steps_per_epoch=steps_per_epoch
        )
    elif sched_type == 'exponential':
        scheduler = WarmupExponentialLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            gamma=config.get('gamma', 0.95),
            steps_per_epoch=steps_per_epoch
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
        
    return scheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Learning rate increases linearly during warmup, then follows
    cosine annealing to minimum learning rate.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 300,
        min_lr: float = 1e-6,
        steps_per_epoch: Optional[int] = None,
        last_epoch: int = -1
    ) -> None:
        """
        Initialize cosine annealing scheduler with warmup.
        
        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs.
            max_epochs: Total number of training epochs.
            min_lr: Minimum learning rate.
            steps_per_epoch: Steps per epoch for step-wise scheduling.
            last_epoch: Last epoch index.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.steps_per_epoch = steps_per_epoch or 1
        
        self.warmup_steps = warmup_epochs * self.steps_per_epoch
        self.total_steps = max_epochs * self.steps_per_epoch
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            return [
                self.min_lr + alpha * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            import math
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupStepLR(_LRScheduler):
    """Step learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        step_size: int = 30,
        gamma: float = 0.1,
        steps_per_epoch: Optional[int] = None,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch or 1
        self.warmup_steps = warmup_epochs * self.steps_per_epoch
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return [alpha * base_lr for base_lr in self.base_lrs]
        else:
            epoch = (step - self.warmup_steps) // self.steps_per_epoch
            return [
                base_lr * (self.gamma ** (epoch // self.step_size))
                for base_lr in self.base_lrs
            ]


class WarmupMultiStepLR(_LRScheduler):
    """Multi-step learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        milestones: List[int] = [100, 200],
        gamma: float = 0.1,
        steps_per_epoch: Optional[int] = None,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch or 1
        self.warmup_steps = warmup_epochs * self.steps_per_epoch
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return [alpha * base_lr for base_lr in self.base_lrs]
        else:
            epoch = (step - self.warmup_steps) // self.steps_per_epoch
            factor = 1.0
            for milestone in self.milestones:
                if epoch >= milestone:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmupExponentialLR(_LRScheduler):
    """Exponential learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        gamma: float = 0.95,
        steps_per_epoch: Optional[int] = None,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch or 1
        self.warmup_steps = warmup_epochs * self.steps_per_epoch
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return [alpha * base_lr for base_lr in self.base_lrs]
        else:
            epoch = (step - self.warmup_steps) // self.steps_per_epoch
            return [base_lr * (self.gamma ** epoch) for base_lr in self.base_lrs]
