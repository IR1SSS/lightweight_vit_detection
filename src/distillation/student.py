"""
Student model wrapper for knowledge distillation.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class StudentModel(nn.Module):
    """
    Student model wrapper for knowledge distillation.
    
    Wraps a lightweight model and provides hooks for extracting
    intermediate features for distillation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        output_features: bool = True,
    ):
        """
        Initialize student model wrapper.
        
        Args:
            model: The lightweight student model
            output_features: Whether to output intermediate features
        """
        super().__init__()
        self.model = model
        self.output_features = output_features
        
        # Feature storage - stored as list for distillation compatibility
        self._backbone_features: List[torch.Tensor] = []
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        
        if output_features:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        def backbone_hook(module, input, output):
            """Hook to capture backbone output as list."""
            if isinstance(output, (list, tuple)):
                self._backbone_features = list(output)
            else:
                self._backbone_features = [output]
        
        # Register hook on backbone module to capture its output
        if hasattr(self.model, "backbone"):
            hook = self.model.backbone.register_forward_hook(backbone_hook)
            self._hooks.append(hook)
            
            # Also register hooks on individual conv layers for detailed feature maps
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
            Tuple of predictions and features
        """
        # Clear previous feature maps
        self.feature_maps.clear()
        self._backbone_features = []
        
        # Forward pass through model (hooks will capture backbone features)
        cls_pred, reg_pred, obj_pred = self.model(x)
        
        # Return predictions and features
        if self.output_features:
            return cls_pred, reg_pred, obj_pred, self.feature_maps
        
        return cls_pred, reg_pred, obj_pred
    
    def get_features(self) -> List[torch.Tensor]:
        """
        Get backbone feature maps as list for distillation.
        
        Returns:
            List of backbone feature tensors (compatible with teacher format)
        """
        return self._backbone_features
    
    def get_parameters(self):
        """Get trainable parameters."""
        return self.model.parameters()


def wrap_student(
    model: nn.Module,
    output_features: bool = True,
) -> StudentModel:
    """
    Wrap a model as a student model.
    
    Args:
        model: Model to wrap
        output_features: Whether to output features
        
    Returns:
        StudentModel wrapper
    """
    return StudentModel(model, output_features=output_features)
