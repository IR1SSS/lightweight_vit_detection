"""
Base Model Abstract Class for Lightweight ViT Detection System.

This module defines the abstract base class that all detection models should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for detection models.
    
    All detection models should inherit from this class and implement
    the abstract methods for building backbone, neck, and head components.
    
    Attributes:
        config (Dict): Model configuration dictionary.
        backbone (nn.Module): Feature extraction backbone network.
        neck (nn.Module): Feature pyramid network for multi-scale features.
        head (nn.Module): Detection head for classification and regression.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary containing backbone,
                   neck, and head configurations.
        """
        super().__init__()
        self.config = config
        
        # Build model components
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = self._build_head()
        
    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """
        Build the backbone network.
        
        Returns:
            nn.Module: Backbone network for feature extraction.
        """
        raise NotImplementedError("Subclasses must implement _build_backbone()")
    
    @abstractmethod
    def _build_neck(self) -> Optional[nn.Module]:
        """
        Build the neck network (e.g., FPN).
        
        Returns:
            nn.Module or None: Neck network for feature fusion.
        """
        raise NotImplementedError("Subclasses must implement _build_neck()")
    
    @abstractmethod
    def _build_head(self) -> nn.Module:
        """
        Build the detection head.
        
        Returns:
            nn.Module: Detection head for prediction.
        """
        raise NotImplementedError("Subclasses must implement _build_head()")
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            targets: Optional list of target dictionaries for training.
                    Each dict contains 'boxes' and 'labels'.
                    
        Returns:
            Dict containing:
                - In training mode: loss dictionary
                - In inference mode: detection results
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    @abstractmethod
    def get_complexity(self) -> Dict[str, float]:
        """
        Calculate model complexity metrics.
        
        Returns:
            Dict containing:
                - 'params': Number of parameters (in millions)
                - 'flops': FLOPs for a single forward pass (in billions)
                - 'model_size': Model size in MB
        """
        raise NotImplementedError("Subclasses must implement get_complexity()")
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters.
            
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True) -> None:
        """
        Load pretrained weights.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            strict: Whether to strictly enforce that the keys in state_dict
                   match the keys returned by this module's state_dict().
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        self.load_state_dict(state_dict, strict=strict)
        
    def save_checkpoint(
        self, 
        save_path: str, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        **kwargs
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save the checkpoint.
            optimizer: Optional optimizer to save state.
            epoch: Current epoch number.
            **kwargs: Additional items to save.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        checkpoint.update(kwargs)
        torch.save(checkpoint, save_path)
