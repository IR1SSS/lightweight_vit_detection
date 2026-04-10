"""
Checkpoint management for model saving and loading.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup of old checkpoints.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = "mAP",
        metric_mode: str = "max",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save the best checkpoint separately
            metric_name: Name of the metric to compare
            metric_mode: 'max' or 'min' for metric comparison
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.best_metric = float("-inf") if metric_mode == "max" else float("inf")
        self.checkpoint_history: List[Path] = []
    
    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
            optimizer: Optimizer state
            scheduler: Scheduler state
            extra_state: Additional state to save
            filename: Custom filename (if None, auto-generates)
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint state
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
        }
        
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        
        if extra_state is not None:
            state["extra_state"] = extra_state
        
        # Generate filename
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        
        save_path = self.save_dir / filename
        torch.save(state, save_path)
        logger.info(f"Saved checkpoint: {save_path}")
        
        # Update checkpoint history
        self.checkpoint_history.append(save_path)
        
        # Check if this is the best model
        current_metric = metrics.get(self.metric_name, 0.0)
        is_best = self._is_better(current_metric)
        
        if is_best:
            self.best_metric = current_metric
            if self.save_best:
                best_path = self.save_dir / "best_model.pth"
                torch.save(state, best_path)
                logger.info(f"New best model! Saved to: {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return save_path
    
    def load(
        self,
        model: nn.Module,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        load_optimizer: bool = True,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint (if None, loads best or latest)
            load_best: Whether to load the best checkpoint
            load_optimizer: Whether to load optimizer state
            map_location: Device to map tensors to
            
        Returns:
            Loaded checkpoint state
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.save_dir / "best_model.pth"
            else:
                checkpoint_path = self._get_latest_checkpoint()
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            logger.warning(f"No checkpoint found at: {checkpoint_path}")
            return {}
        
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore best metric
        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
    
    def load_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Load optimizer state from checkpoint.
        
        Args:
            optimizer: Optimizer to load state into
            checkpoint: Checkpoint dictionary
        """
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state")
    
    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.metric_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint file."""
        checkpoints = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        
        # Sort by epoch number
        def get_epoch(path: Path) -> int:
            match = re.search(r"epoch_(\d+)", path.name)
            return int(match.group(1)) if match else 0
        
        return max(checkpoints, key=get_epoch)
    
    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        # Don't remove best_model.pth
        checkpoints = [
            p for p in self.checkpoint_history
            if "best_model" not in p.name
        ]
        
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug(f"Removed old checkpoint: {oldest}")


def save_checkpoint(
    save_path: Union[str, Path],
    model: nn.Module,
    epoch: int,
    metrics: Dict[str, float],
    optimizer: Optional[Optimizer] = None,
    **kwargs,
) -> None:
    """
    Simple checkpoint save function.
    
    Args:
        save_path: Path to save checkpoint
        model: Model to save
        epoch: Current epoch
        metrics: Current metrics
        optimizer: Optional optimizer
        **kwargs: Additional state to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        **kwargs,
    }
    
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(state, save_path)
    logger.info(f"Saved checkpoint: {save_path}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Simple checkpoint load function.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        map_location: Device to map tensors to
        strict: Whether to strictly enforce state_dict keys match
        
    Returns:
        Loaded checkpoint state
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    return checkpoint
