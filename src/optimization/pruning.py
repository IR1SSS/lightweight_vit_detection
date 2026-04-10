"""
Model pruning utilities for model optimization.
Implements various pruning strategies for neural networks.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class ModelPruner:
    """
    Base class for model pruning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        prune_layers: Optional[List[str]] = None,
    ):
        """
        Initialize pruner.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of weights to prune
            prune_layers: List of layer names to prune (None for all)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.prune_layers = prune_layers
        self.pruned_layers = []
    
    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get layers that can be pruned.
        
        Returns:
            List of (name, module) tuples
        """
        prunable = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if self.prune_layers is None or name in self.prune_layers:
                    prunable.append((name, module))
        
        return prunable
    
    def apply_masks(self):
        """Apply pruning masks to model."""
        pass
    
    def remove_masks(self):
        """Remove pruning masks and make pruning permanent."""
        for name, module in self.get_prunable_layers():
            prune.remove(module, "weight")
    
    def get_sparsity(self) -> float:
        """
        Calculate model sparsity.
        
        Returns:
            Sparsity ratio
        """
        total_params = 0
        zero_params = 0
        
        for name, module in self.get_prunable_layers():
            if hasattr(module, "weight_mask"):
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
            else:
                weight = module.weight
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0


class L1UnstructuredPruner(ModelPruner):
    """
    L1 unstructured pruning.
    
    Prunes weights with smallest L1 magnitude regardless of position.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        prune_layers: Optional[List[str]] = None,
    ):
        """
        Initialize L1 unstructured pruner.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of weights to prune
            prune_layers: List of layer names to prune
        """
        super().__init__(model, pruning_ratio, prune_layers)
    
    def prune(self) -> nn.Module:
        """
        Apply L1 unstructured pruning.
        
        Returns:
            Pruned model
        """
        for name, module in self.get_prunable_layers():
            prune.l1_unstructured(module, name="weight", amount=self.pruning_ratio)
            self.pruned_layers.append(name)
        
        return self.model


class L1StructuredPruner(ModelPruner):
    """
    L1 structured pruning.
    
    Prunes entire channels/filters based on L1 magnitude.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        prune_layers: Optional[List[str]] = None,
        dim: int = 0,  # 0 for output channels, 1 for input channels
    ):
        """
        Initialize L1 structured pruner.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of channels to prune
            prune_layers: List of layer names to prune
            dim: Dimension to prune (0=output, 1=input)
        """
        super().__init__(model, pruning_ratio, prune_layers)
        self.dim = dim
    
    def prune(self) -> nn.Module:
        """
        Apply L1 structured pruning.
        
        Returns:
            Pruned model
        """
        for name, module in self.get_prunable_layers():
            # Calculate L1 norm for each channel
            if isinstance(module, nn.Conv2d):
                if self.dim == 0:
                    # Prune output channels
                    importance = module.weight.abs().sum(dim=(1, 2, 3))
                else:
                    # Prune input channels
                    importance = module.weight.abs().sum(dim=(0, 2, 3))
            elif isinstance(module, nn.Linear):
                if self.dim == 0:
                    importance = module.weight.abs().sum(dim=1)
                else:
                    importance = module.weight.abs().sum(dim=0)
            
            # Get threshold
            num_prune = int(len(importance) * self.pruning_ratio)
            threshold = torch.kthvalue(importance, num_prune).values
            
            # Create mask
            mask = importance > threshold
            
            # Apply structured pruning
            if self.dim == 0:
                prune.custom_from_mask(module, "weight", mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(module.weight))
            
            self.pruned_layers.append(name)
        
        return self.model


class ChannelPruner(ModelPruner):
    """
    Channel pruning with dependency-aware pruning.
    
    Handles the dependency between layers when pruning channels.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        input_shape: Tuple[int, int, int] = (3, 320, 320),
    ):
        """
        Initialize channel pruner.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of channels to prune
            input_shape: Input shape for tracing
        """
        super().__init__(model, pruning_ratio)
        self.input_shape = input_shape
    
    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """
        Analyze channel dependencies between layers.
        
        Returns:
            Dictionary mapping layers to their dependent layers
        """
        dependencies = {}
        
        # Trace model to find connections
        # This is a simplified version - actual implementation would need
        # proper graph analysis
        
        prunable = self.get_prunable_layers()
        for i, (name, module) in enumerate(prunable):
            if i < len(prunable) - 1:
                next_name = prunable[i + 1][0]
                dependencies[name] = [next_name]
            else:
                dependencies[name] = []
        
        return dependencies
    
    def compute_channel_importance(
        self,
        module: nn.Module,
    ) -> torch.Tensor:
        """
        Compute importance score for each channel.
        
        Args:
            module: Module to compute importance for
            
        Returns:
            Importance scores
        """
        if isinstance(module, nn.Conv2d):
            # L1 norm based importance
            importance = module.weight.abs().sum(dim=(2, 3))
            if module.bias is not None:
                importance += module.bias.abs().unsqueeze(1)
        elif isinstance(module, nn.Linear):
            importance = module.weight.abs().sum(dim=1)
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")
        
        return importance
    
    def prune(self) -> nn.Module:
        """
        Apply channel pruning.
        
        Returns:
            Pruned model
        """
        dependencies = self.analyze_dependencies()
        
        for name, module in self.get_prunable_layers():
            importance = self.compute_channel_importance(module)
            
            # Determine number of channels to keep
            num_channels = importance.shape[0]
            num_keep = max(1, int(num_channels * (1 - self.pruning_ratio)))
            
            # Get indices to keep
            _, indices = torch.topk(importance.sum(dim=1), num_keep)
            indices = indices.sort()[0]
            
            # Create channel mask
            mask = torch.zeros(num_channels, dtype=torch.bool)
            mask[indices] = True
            
            # Apply pruning
            prune.custom_from_mask(
                module, "weight",
                mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(module.weight)
            )
            
            self.pruned_layers.append(name)
        
        return self.model


def iterative_pruning(
    model: nn.Module,
    target_sparsity: float,
    num_iterations: int,
    train_fn: Callable,
    pruner_class: type = L1UnstructuredPruner,
) -> nn.Module:
    """
    Iterative pruning with fine-tuning.
    
    Args:
        model: Model to prune
        target_sparsity: Target sparsity ratio
        num_iterations: Number of pruning iterations
        train_fn: Fine-tuning function
        pruner_class: Pruner class to use
        
    Returns:
        Pruned and fine-tuned model
    """
    # Calculate per-iteration sparsity
    sparsity_per_iter = 1 - (1 - target_sparsity) ** (1 / num_iterations)
    
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        
        # Prune
        pruner = pruner_class(model, pruning_ratio=sparsity_per_iter)
        model = pruner.prune()
        
        # Fine-tune
        model = train_fn(model)
        
        # Report sparsity
        sparsity = pruner.get_sparsity()
        print(f"Current sparsity: {sparsity:.2%}")
    
    return model


def get_pruning_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Get pruning statistics for a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary of statistics
    """
    total_params = 0
    pruned_params = 0
    layer_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight
            total = weight.numel()
            zeros = (weight == 0).sum().item() if hasattr(weight, "numel") else 0
            
            total_params += total
            pruned_params += zeros
            
            layer_stats[name] = {
                "total": total,
                "zeros": zeros,
                "sparsity": zeros / total if total > 0 else 0.0,
            }
    
    return {
        "total_params": total_params,
        "pruned_params": pruned_params,
        "global_sparsity": pruned_params / total_params if total_params > 0 else 0.0,
        "layer_stats": layer_stats,
    }
