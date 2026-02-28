"""
Base Distiller Class for Knowledge Distillation.

This module implements the base class for knowledge distillation,
providing framework for teacher-student model training.
"""

from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn

from ..models.base_model import BaseModel


class BaseDistiller(nn.Module):
    """
    Base class for knowledge distillation.
    
    Manages teacher-student model pairs and provides
    framework for distillation loss computation.
    
    Attributes:
        teacher (nn.Module): Frozen teacher model.
        student (nn.Module): Student model to be trained.
        distillation_config (Dict): Configuration for distillation.
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize distiller.
        
        Args:
            teacher: Pre-trained teacher model.
            student: Student model to be trained.
            config: Distillation configuration dictionary.
        """
        super().__init__()
        
        self.teacher = teacher
        self.student = student
        self.config = config
        
        # Freeze teacher model
        self._freeze_teacher()
        
        # Store intermediate features
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}
        
        # Register hooks for feature extraction
        self._register_hooks()
        
    def _freeze_teacher(self) -> None:
        """Freeze all parameters of the teacher model."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def _register_hooks(self) -> None:
        """Register forward hooks for feature extraction."""
        feature_config = self.config.get('strategy', {}).get('feature_distillation', {})
        
        if not feature_config.get('enabled', False):
            return
            
        layers = feature_config.get('layers', [])
        
        # Register hooks for teacher
        for layer_name in layers:
            self._register_hook(self.teacher, layer_name, 'teacher')
            
        # Register hooks for student  
        for layer_name in layers:
            self._register_hook(self.student, layer_name, 'student')
            
    def _register_hook(
        self, 
        model: nn.Module, 
        layer_name: str, 
        model_type: str
    ) -> None:
        """
        Register forward hook on a specific layer.
        
        Args:
            model: Model to register hook on.
            layer_name: Name of the layer (e.g., 'backbone.layer2').
            model_type: 'teacher' or 'student'.
        """
        try:
            layer = self._get_layer(model, layer_name)
            
            def hook(module, input, output):
                if model_type == 'teacher':
                    self.teacher_features[layer_name] = output
                else:
                    self.student_features[layer_name] = output
                    
            layer.register_forward_hook(hook)
        except AttributeError:
            print(f"Warning: Layer {layer_name} not found in {model_type} model")
            
    def _get_layer(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Get a layer from model by name."""
        parts = layer_name.split('.')
        layer = model
        for part in parts:
            layer = getattr(layer, part)
        return layer
        
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with distillation.
        
        Args:
            x: Input tensor.
            targets: Optional ground truth targets.
            
        Returns:
            Dictionary containing student outputs and distillation losses.
        """
        # Clear stored features
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_output = self.teacher(x)
            
        # Student forward
        student_output = self.student(x, targets)
        
        # Compute distillation losses
        distill_losses = self.compute_distillation_loss(
            teacher_output, student_output
        )
        
        # Combine with task loss
        if 'loss_total' in student_output:
            total_loss = student_output['loss_total']
            
            distill_weight = self.config.get('training', {}).get(
                'loss_weights', {}
            ).get('distillation', 0.5)
            
            for key, loss in distill_losses.items():
                total_loss = total_loss + distill_weight * loss
                
            student_output['loss_total'] = total_loss
            
        student_output.update(distill_losses)
        
        return student_output
        
    def compute_distillation_loss(
        self,
        teacher_output: Dict[str, torch.Tensor],
        student_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation losses.
        
        Args:
            teacher_output: Teacher model outputs.
            student_output: Student model outputs.
            
        Returns:
            Dictionary of distillation losses.
        """
        losses = {}
        strategy = self.config.get('strategy', {})
        
        # Response distillation
        if strategy.get('response_distillation', {}).get('enabled', False):
            from .losses import ResponseDistillationLoss
            
            resp_config = strategy['response_distillation']
            resp_loss_fn = ResponseDistillationLoss(
                temperature=resp_config.get('temperature', 4.0)
            )
            
            losses['loss_response'] = resp_loss_fn(
                student_output.get('cls_scores', []),
                teacher_output.get('cls_scores', [])
            ) * resp_config.get('weight', 1.0)
            
        # Feature distillation
        if strategy.get('feature_distillation', {}).get('enabled', False):
            from .losses import FeatureDistillationLoss
            
            feat_config = strategy['feature_distillation']
            feat_loss_fn = FeatureDistillationLoss()
            
            feature_loss = torch.tensor(0.0, device=next(self.student.parameters()).device)
            
            for layer_name in self.teacher_features.keys():
                if layer_name in self.student_features:
                    feature_loss = feature_loss + feat_loss_fn(
                        self.student_features[layer_name],
                        self.teacher_features[layer_name]
                    )
                    
            losses['loss_feature'] = feature_loss * feat_config.get('weight', 0.5)
            
        # Relation distillation
        if strategy.get('relation_distillation', {}).get('enabled', False):
            from .losses import RelationDistillationLoss
            
            rel_config = strategy['relation_distillation']
            rel_loss_fn = RelationDistillationLoss()
            
            # Use the last feature layer for relation distillation
            if self.teacher_features and self.student_features:
                last_teacher = list(self.teacher_features.values())[-1]
                last_student = list(self.student_features.values())[-1]
                
                losses['loss_relation'] = rel_loss_fn(
                    last_student, last_teacher
                ) * rel_config.get('weight', 0.1)
                
        return losses


class DistillationTrainer:
    """
    Trainer class for knowledge distillation.
    
    Wraps the distiller and handles training loop,
    optimization, and logging.
    """
    
    def __init__(
        self,
        distiller: BaseDistiller,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> None:
        """
        Initialize distillation trainer.
        
        Args:
            distiller: BaseDistiller instance.
            optimizer: Optimizer for student model.
            scheduler: Optional learning rate scheduler.
            device: Device to train on.
        """
        self.distiller = distiller.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            batch: Tuple of (images, targets).
            
        Returns:
            Dictionary of loss values.
        """
        images, targets = batch
        images = images.to(self.device)
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
        
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.distiller(images, targets)
        
        # Backward pass
        loss = outputs.get('loss_total', torch.tensor(0.0))
        loss.backward()
        
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in outputs.items() if 'loss' in k}
        
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            max_batches: Maximum batches per epoch (0 = no limit).
            
        Returns:
            Dictionary of average loss values.
        """
        self.distiller.train()
        self.distiller.teacher.eval()  # Keep teacher in eval mode
        
        total_losses: Dict[str, float] = {}
        num_batches = 0
        
        for batch in dataloader:
            losses = self.train_step(batch)
            
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
            
            if max_batches > 0 and num_batches >= max_batches:
                break
            
        return {k: v / num_batches for k, v in total_losses.items()}


def build_distiller(
    teacher_model: nn.Module,
    student_model: nn.Module,
    config: Dict[str, Any]
) -> BaseDistiller:
    """
    Build distiller from configuration.
    
    Args:
        teacher_model: Pre-trained teacher model.
        student_model: Student model to train.
        config: Distillation configuration.
        
    Returns:
        BaseDistiller instance.
    """
    return BaseDistiller(teacher_model, student_model, config)
