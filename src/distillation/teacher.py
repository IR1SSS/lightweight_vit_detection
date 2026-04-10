"""
Teacher model wrapper for knowledge distillation.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class TeacherModel(nn.Module):
    """
    Teacher model wrapper for knowledge distillation.
    
    Wraps a pre-trained model and provides hooks for extracting
    intermediate features for distillation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
        output_features: bool = True,
    ):
        """
        Initialize teacher model wrapper.
        
        Args:
            model: The pre-trained teacher model
            freeze: Whether to freeze teacher weights
            output_features: Whether to output intermediate features
        """
        super().__init__()
        self.model = model
        self.freeze = freeze
        self.output_features = output_features
        
        # Freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Feature extraction hooks
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        
        if output_features:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        # Hook function to capture features
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        # Register hooks based on model type
        if hasattr(self.model, "backbone"):
            # Hook backbone features
            for name, module in self.model.backbone.named_modules():
                if isinstance(module, nn.Conv2d) and "blocks" in name:
                    hook = module.register_forward_hook(hook_fn(f"backbone.{name}"))
                    self._hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (cls_pred, reg_pred, obj_pred, features)
        """
        # Clear previous feature maps
        self.feature_maps.clear()
        
        # Forward pass through model
        outputs = self.model(x)
        
        # Handle different model types
        if isinstance(outputs, (list, tuple)) and len(outputs) != 3:
            # Feature extractor: save and return features directly
            self._last_feature_outputs = outputs
            return outputs
        
        # Detection model: unpack predictions
        if isinstance(outputs, (list, tuple)):
            cls_pred, reg_pred, obj_pred = outputs
        else:
            # Single tensor output
            cls_pred, reg_pred, obj_pred = outputs, None, None
            
        # Return predictions and features
        if self.output_features:
            return cls_pred, reg_pred, obj_pred, self.feature_maps
        
        return cls_pred, reg_pred, obj_pred
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """
        Get captured feature maps.
        
        Returns:
            Dictionary of feature maps
        """
        # For feature extractors, return the last outputs
        if hasattr(self, '_last_feature_outputs') and self._last_feature_outputs:
            return self._last_feature_outputs
        return self.feature_maps
    
    def train(self, mode: bool = True):
        """Set training mode (teacher always in eval if frozen)."""
        if self.freeze:
            return self
        return super().train(mode)


def wrap_teacher(
    model: nn.Module,
    freeze: bool = True,
    output_features: bool = True,
) -> TeacherModel:
    """
    Wrap a model as a teacher model.
    
    Args:
        model: Model to wrap
        freeze: Whether to freeze weights
        output_features: Whether to output features
        
    Returns:
        TeacherModel wrapper
    """
    return TeacherModel(model, freeze=freeze, output_features=output_features)
