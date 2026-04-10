"""
Distillation trainer for knowledge distillation training.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logger import get_logger, MetricLogger
from ..utils.checkpoint import CheckpointManager
from ..utils.metrics import DetectionMetrics
from .teacher import TeacherModel, wrap_teacher
from .student import StudentModel, wrap_student
from .losses import DistillationLoss
from .feature_adapter import MultiLevelFeatureAdapter, FeatureDistillationWithAdapter

logger = get_logger(__name__)


class DistillationTrainer:
    """
    Trainer for knowledge distillation.
    
    Handles the training loop for distilling knowledge from a teacher model
    to a student model.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        distill_criterion: Optional[DistillationLoss] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
        distill_config: Optional[Dict] = None,
        teacher_channels: Optional[List[int]] = None,
        student_channels: Optional[List[int]] = None,
        gradient_clip: float = 1.0,
        enable_qat: bool = False,
        qat_start_epoch: int = 0,
        qat_backend: str = "fbgemm",
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Lightweight student model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for student model
            scheduler: Learning rate scheduler
            criterion: Detection loss function
            distill_criterion: Distillation loss function
            device: Device to use
            output_dir: Output directory for checkpoints
            distill_config: Distillation configuration
            teacher_channels: Teacher feature channel dimensions
            student_channels: Student feature channel dimensions
            gradient_clip: Maximum gradient norm for clipping
            enable_qat: Enable quantization-aware training
            qat_start_epoch: Epoch to start QAT (default: 0, start immediately)
            qat_backend: Quantization backend ("fbgemm" or "qnnpack")
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip = gradient_clip
        
        # QAT settings
        self.enable_qat = enable_qat
        self.qat_start_epoch = qat_start_epoch
        self.qat_backend = qat_backend
        self.qat_prepared = False
        
        # Wrap models
        self.teacher = wrap_teacher(teacher_model, freeze=True, output_features=True)
        self.student = wrap_student(student_model, output_features=True)
        
        self.teacher.to(device)
        self.student.to(device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Distillation config (must be set before criterion)
        self.distill_config = distill_config or {
            "response": {"enabled": True, "weight": 1.0},
            "feature": {"enabled": True, "weight": 0.5},
            "relation": {"enabled": True, "weight": 0.3},
            "loss_weights": {
                "detection_loss": 1.0,
                "distill_loss": 2.0,
            },
        }
        
        # Loss functions
        self.criterion = criterion or self._build_detection_loss()
        self.distill_criterion = distill_criterion or DistillationLoss()
        
        # Feature adapter for cross-architecture distillation
        self.feature_distill = None
        
        if teacher_channels and student_channels:
            logger.info("Creating feature adapter for cross-architecture distillation")
            logger.info(f"  Teacher channels: {teacher_channels}")
            logger.info(f"  Student channels: {student_channels}")
            
            # Use the minimum number of levels
            num_levels = min(len(teacher_channels), len(student_channels))
            
            # Store num_levels for consistent feature slicing during training
            self.num_distill_levels = num_levels
            
            # Create feature distillation module with adapter
            # Use last N levels for both teacher and student (deeper features for detection)
            self.feature_distill = FeatureDistillationWithAdapter(
                student_channels=student_channels[-num_levels:],
                teacher_channels=teacher_channels[-num_levels:],
                projection_dim=256,
                loss_type="mse",
                use_projection=True,
            )
            self.feature_distill.to(device)
            
            # Add adapter parameters to optimizer
            self.optimizer.add_param_group({
                "params": self.feature_distill.parameters(),
                "lr": optimizer.param_groups[0]["lr"],
            })
            
            logger.info("Feature adapter created and added to optimizer")
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=output_dir,
            max_checkpoints=5,
            save_best=True,
            metric_name="mAP",
            metric_mode="max",
        )
        
        # Metrics
        self.metrics = DetectionMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_mAP = 0.0
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')
        self.use_amp = False  # Disabled for numerical stability in detection training
    
    def _build_detection_loss(self) -> nn.Module:
        """Build default detection loss."""
        # Get image size from config or use default
        image_size = 320
        if self.distill_config and "image_size" in self.distill_config:
            image_size = self.distill_config["image_size"]
        
        return DetectionLoss(
            num_classes=80,
            num_anchors=3,
            strides=[8, 16, 32],
            cls_weight=1.0,
            reg_weight=5.0,
            obj_weight=1.0,
            image_size=image_size,
        )
    
    def train(
        self,
        num_epochs: int,
        eval_interval: int = 10,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Run knowledge distillation training.
        
        Args:
            num_epochs: Number of training epochs
            eval_interval: Evaluation interval
            log_interval: Logging interval
            
        Returns:
            Dictionary of final metrics
        """
        logger.info(f"Starting distillation training for {num_epochs} epochs")
        logger.info(f"Teacher model: {self.teacher.model.__class__.__name__}")
        logger.info(f"Student model: {self.student.model.__class__.__name__}")
        if self.enable_qat:
            logger.info(f"QAT enabled, starting at epoch {self.qat_start_epoch}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
                    
            # Enable QAT at specified epoch
            if self.enable_qat and epoch >= self.qat_start_epoch and not self.qat_prepared:
                self._prepare_qat()
                    
            # Train one epoch
            train_metrics = self._train_epoch(epoch, log_interval)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Evaluate
            if (epoch + 1) % eval_interval == 0:
                val_metrics = self.evaluate()
                
                # Save checkpoint
                self.checkpoint_manager.save(
                    model=self.student.model,
                    epoch=epoch,
                    metrics=val_metrics,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                
                # Update best mAP (use mAP@0.5 as primary metric)
                current_map = val_metrics.get("mAP@0.5", 0.0)
                if current_map > self.best_mAP:
                    self.best_mAP = current_map
                    logger.info(f"New best mAP@0.5: {self.best_mAP:.4f}")
        
        logger.info(f"Training completed. Best mAP: {self.best_mAP:.4f}")
        
        return {"best_mAP": self.best_mAP}
    
    def _train_epoch(
        self,
        epoch: int,
        log_interval: int,
    ) -> Dict[str, float]:
        """
        Train one epoch.
        
        Args:
            epoch: Current epoch
            log_interval: Logging interval
            
        Returns:
            Dictionary of training metrics
        """
        self.student.train()
        self.teacher.eval()
        
        metric_logger = MetricLogger(logger, delimiter="  ")
        
        start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["images"].to(self.device)
            targets = self._move_targets(batch["targets"], self.device)
            
            # Forward pass - Teacher (no gradient needed)
            with torch.no_grad():
                teacher_outputs = self.teacher(images)
                teacher_features = self.teacher.get_features()
            
            # Forward pass - Student with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                student_outputs = self.student(images)
                student_features = self.student.get_features()
                
                # Compute detection loss
                det_loss = self.criterion(student_outputs[:3], targets)
                
                # Compute distillation loss
                if self.feature_distill is not None:
                    # Use feature adapter for cross-architecture distillation
                    # Use the last N levels that match adapter
                    num_levels = min(len(teacher_features), len(student_features))
                    t_feats = teacher_features[-num_levels:]
                    s_feats = student_features[-num_levels:]
                    
                    feature_loss = self.feature_distill(s_feats, t_feats)
                    distill_loss = feature_loss
                else:
                    # Standard distillation loss - handle both list and dict formats
                    if isinstance(student_features, list) and isinstance(teacher_features, list):
                        # Both are lists - compute simple feature matching loss
                        distill_loss = self._compute_list_distill_loss(
                            student_features, teacher_features
                        )
                    else:
                        # Dict format
                        distill_outputs = self.distill_criterion(
                            {"cls_pred": student_outputs[0]},
                            {"cls_pred": teacher_outputs[0]},
                            student_features,
                            teacher_features,
                        )
                        distill_loss = distill_outputs["total_distill_loss"]
                
                # Combined loss
                # Get loss weights from config (support both old and new format)
                loss_weights = self.distill_config.get("loss_weights", {})
                detection_weight = loss_weights.get("detection_loss", 1.0)
                distill_weight = loss_weights.get("distill_loss", 2.0)
                
                total_loss = (
                    detection_weight * det_loss +
                    distill_weight * distill_loss
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            
            # Check for NaN in loss - skip batch if detected
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN/Inf detected in loss at batch {batch_idx}, skipping batch")
                continue
            
            # Check for zero loss (potential model death)
            if total_loss.item() < 1e-10:
                logger.warning(f"Near-zero loss detected at batch {batch_idx}: {total_loss.item()}")
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    max_norm=self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    max_norm=self.gradient_clip
                )
                self.optimizer.step()
            
            # Health check: Monitor for model death (every 100 batches)
            if batch_idx % 100 == 0 and batch_idx > 0:
                with torch.no_grad():
                    # Check if model outputs are all the same (dead model)
                    out_std = student_outputs[0].std().item() if isinstance(student_outputs[0], torch.Tensor) else 0
                    if out_std < 1e-6:
                        logger.error(f"MODEL DEATH DETECTED at batch {batch_idx}! Output std = {out_std}")
                        logger.error("Training will continue but model may need to be restarted with lower learning rate")
            
            # Update metrics
            metric_logger.update(
                loss=total_loss.item(),
                det_loss=det_loss.item(),
                distill_loss=distill_loss.item(),
                lr=self.optimizer.param_groups[0]["lr"],
            )
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "det": f"{det_loss.item():.4f}",
                "distill": f"{distill_loss.item():.4f}",
            })
            
            # Log
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"[{batch_idx}/{len(self.train_loader)}] {metric_logger}"
                )
                # Clear CUDA cache periodically to prevent memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        return {
            "loss": metric_logger.metrics["loss"].avg,
            "det_loss": metric_logger.metrics["det_loss"].avg,
            "distill_loss": metric_logger.metrics["distill_loss"].avg,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate student model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.student.eval()
        self.metrics.reset()
        
        logger.info("Evaluating...")
        
        for batch in tqdm(self.val_loader, desc="Evaluation"):
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            
            # Forward pass
            cls_pred, reg_pred, obj_pred = self.student.model(images)
            
            # Get predictions
            predictions = self._decode_predictions(cls_pred, reg_pred, obj_pred)
            
            # Update metrics
            for pred, target in zip(predictions, targets):
                self.metrics.update(
                    {
                        "boxes": pred["boxes"].cpu().numpy(),
                        "scores": pred["scores"].cpu().numpy(),
                        "labels": pred["labels"].cpu().numpy(),
                    },
                    {
                        "boxes": target["boxes"].numpy(),
                        "labels": target["labels"].numpy(),
                    }
                )
        
        # Compute metrics
        metrics = self.metrics.compute()
        
        logger.info(f"Validation mAP@0.5: {metrics['mAP@0.5']:.4f}")
        logger.info(f"Validation mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        
        return metrics
    
    def _move_targets(
        self,
        targets: List[Dict],
        device: str,
    ) -> List[Dict]:
        """Move targets to device."""
        moved_targets = []
        for target in targets:
            moved_target = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in target.items()
            }
            moved_targets.append(moved_target)
        return moved_targets
    
    def _decode_predictions(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        obj_pred: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
    ) -> List[Dict]:
        """Decode model predictions to detections."""
        predictions = []
        
        B = cls_pred.shape[0]
        
        for b in range(B):
            # Get predictions for this image
            scores = cls_pred[b].sigmoid() * obj_pred[b].sigmoid()
            boxes = reg_pred[b]
            
            # Get top predictions
            max_scores, labels = scores.max(dim=-1)
            
            # Filter by confidence
            mask = max_scores > conf_threshold
            
            if mask.sum() == 0:
                predictions.append({
                    "boxes": torch.zeros(0, 4, device=self.device),
                    "scores": torch.zeros(0, device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                })
                continue
            
            filtered_boxes = boxes[mask]
            filtered_scores = max_scores[mask]
            filtered_labels = labels[mask]
            
            # Apply NMS
            keep = self._nms(filtered_boxes, filtered_scores, nms_threshold)
            
            predictions.append({
                "boxes": filtered_boxes[keep],
                "scores": filtered_scores[keep],
                "labels": filtered_labels[keep],
            })
        
        return predictions
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        """Non-maximum suppression."""
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    
    def save_checkpoint(self, path: str):
        """Save training state."""
        torch.save({
            "epoch": self.current_epoch,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mAP": self.best_mAP,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_epoch = checkpoint["epoch"]
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_mAP = checkpoint.get("best_mAP", 0.0)
    
    def _prepare_qat(self):
        """
        Prepare model for quantization-aware training.
        
        This method is called when QAT is enabled and the specified
        start epoch is reached. It wraps the student model with
        fake quantization modules.
        """
        if self.qat_prepared:
            return
        
        logger.info("Preparing model for quantization-aware training...")
        logger.info(f"Backend: {self.qat_backend}")
        
        try:
            from ..optimization.quantization import (
                QuantizationAwareTraining,
                get_vit_qconfig_mapping,
                prepare_qat_fx,
                FX_QUANT_AVAILABLE,
            )
            from torch.ao.quantization import get_default_qat_qconfig_mapping
            
            # Move student to CPU for QAT preparation
            self.student.cpu()
            
            if FX_QUANT_AVAILABLE:
                # Use FX mode for better ViT compatibility
                logger.info("Using FX graph mode QAT")
                
                qconfig_mapping = get_default_qat_qconfig_mapping(self.qat_backend)
                example_input = torch.randn(1, 3, 320, 320)
                
                # Prepare with FX
                prepared_model = prepare_qat_fx(
                    self.student.model,
                    qconfig_mapping,
                    example_input,
                )
                
                # Update student model
                self.student.model = prepared_model
            else:
                # Use eager mode QAT
                logger.info("Using eager mode QAT")
                
                qat = QuantizationAwareTraining(
                    self.student.model,
                    backend=self.qat_backend,
                )
                self.student.model = qat.prepare()
            
            # Move back to device
            self.student.to(self.device)
            self.qat_prepared = True
            
            logger.info("QAT preparation completed")
            
        except ImportError as e:
            logger.warning(f"QAT not available: {e}")
            self.enable_qat = False
    
    def convert_to_quantized(self) -> nn.Module:
        """
        Convert QAT-prepared model to quantized model.
        
        Should be called after training is complete.
        
        Returns:
            Quantized model
        """
        if not self.qat_prepared:
            logger.warning("Model was not prepared for QAT")
            return self.student.model
        
        logger.info("Converting to quantized model...")
        
        try:
            from ..optimization.quantization import convert_fx, FX_QUANT_AVAILABLE
            
            self.student.eval()
            self.student.cpu()
            
            if FX_QUANT_AVAILABLE:
                quantized_model = convert_fx(self.student.model)
            else:
                quantized_model = torch.quantization.convert(self.student.model)
            
            logger.info("Model converted to quantized version")
            return quantized_model
            
        except ImportError as e:
            logger.error(f"Failed to convert model: {e}")
            return self.student.model


class DetectionLoss(nn.Module):
    """
    Detection loss for object detection.
    Implements YOLO-style loss with anchor assignment.
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        num_anchors: int = 3,
        strides: List[int] = None,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
        obj_weight: float = 1.0,
        iou_threshold: float = 0.5,
        image_size: int = 320,
    ):
        """
        Initialize detection loss.
        
        Args:
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
            strides: Feature map strides
            cls_weight: Classification loss weight
            reg_weight: Regression loss weight
            obj_weight: Objectness loss weight
            iou_threshold: IoU threshold for positive assignment
            image_size: Input image size
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.strides = strides or [8, 16, 32]
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        
        # Anchor ratios (width/height)
        self.anchor_ratios = [
            (1.0, 1.0),      # Square
            (1.0, 2.0),      # Tall
            (2.0, 1.0),      # Wide
        ]
        
        # BCE loss for classification and objectness
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        Compute detection loss.
        
        Args:
            predictions: Model predictions (cls_pred, reg_pred, obj_pred)
                cls_pred: (B, A, N, C) classification logits
                reg_pred: (B, A, N, 4) box regression (cx, cy, w, h) normalized
                obj_pred: (B, A, N, 1) objectness logits
            targets: List of ground truth dicts with 'boxes' and 'labels'
            
        Returns:
            Total detection loss
        """
        cls_pred, reg_pred, obj_pred = predictions
        batch_size = cls_pred.shape[0]
        device = cls_pred.device
        
        # Generate anchor points
        anchors, anchor_indices = self._generate_anchors(cls_pred.shape, device)
        num_anchors = anchors.shape[0]
        
        # Flatten predictions: (B, A, N, C) -> (B, A*N, C)
        # The model outputs (B, num_anchors_per_position, total_positions, channels)
        cls_pred = cls_pred.reshape(batch_size, -1, self.num_classes)
        reg_pred = reg_pred.reshape(batch_size, -1, 4)
        obj_pred = obj_pred.reshape(batch_size, -1, 1)
        
        # Initialize target tensors
        cls_target = torch.zeros_like(cls_pred)
        obj_target = torch.zeros(batch_size, num_anchors, 1, device=device)
        reg_target = torch.zeros(batch_size, num_anchors, 4, device=device)
        pos_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)
        
        # Assign targets for each image
        for b in range(batch_size):
            if targets[b] is None or len(targets[b].get("boxes", [])) == 0:
                continue
            
            boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]
            
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes).to(device)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).to(device)
            
            if len(boxes) == 0:
                continue
            
            # Normalize boxes to [0, 1]
            boxes_norm = boxes.float() / self.image_size
            
            # Assign targets to anchors
            assigned = self._assign_targets(
                anchors, boxes_norm, labels, device
            )
            
            if assigned is not None:
                pos_idx, gt_idx = assigned
                pos_mask[b, pos_idx] = True
                
                # Set classification targets
                cls_target[b, pos_idx, labels[gt_idx].long()] = 1.0
                
                # Set objectness targets
                obj_target[b, pos_idx, 0] = 1.0
                
                # Set regression targets (normalized center format)
                gt_boxes = boxes_norm[gt_idx]
                anchor_centers = anchors[pos_idx, :2]
                anchor_sizes = anchors[pos_idx, 2:]
                
                # Target: offset from anchor center and size ratio
                reg_target[b, pos_idx, 0] = gt_boxes[:, 0] - anchor_centers[:, 0]  # cx offset
                reg_target[b, pos_idx, 1] = gt_boxes[:, 1] - anchor_centers[:, 1]  # cy offset
                reg_target[b, pos_idx, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]        # w
                reg_target[b, pos_idx, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]        # h
        
        # Compute losses
        num_pos = pos_mask.sum().clamp(min=1)
        
        # Classification loss (only for positive samples)
        # Clamp predictions for numerical stability
        cls_pred_clamped = cls_pred.clamp(min=-10.0, max=10.0)
        cls_loss = self.bce_loss(cls_pred_clamped, cls_target)
        cls_loss = cls_loss[pos_mask].sum() / num_pos
        
        # Objectness loss
        obj_pred_clamped = obj_pred.clamp(min=-10.0, max=10.0)
        obj_loss = self.bce_loss(obj_pred_clamped, obj_target)
        obj_loss = obj_loss.mean()
        
        # Regression loss (only for positive samples)
        if pos_mask.any():
            reg_loss = self._ciou_loss(
                reg_pred[pos_mask], 
                reg_target[pos_mask]
            )
            reg_loss = reg_loss.mean()
            # Check for NaN in regression loss
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                reg_loss = reg_pred.new_tensor(0.0)
        else:
            reg_loss = reg_pred.new_tensor(0.0)
        
        # Total loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.reg_weight * reg_loss +
            self.obj_weight * obj_loss
        )
        
        return total_loss
    
    def _generate_anchors(
        self, 
        pred_shape: Tuple[int, ...],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate anchor points for all feature map locations.
            
        Args:
            pred_shape: Shape of predictions (B, A, N, C) where:
                B = batch size
                A = num_anchors_per_position (typically 3)
                N = total positions across all feature levels
                C = num_classes for cls_pred, 4 for reg_pred, 1 for obj_pred
    
        Returns:
            anchors: (A*N, 4) anchor boxes in [cx, cy, w, h] normalized format
            indices: (A*N,) anchor indices for each anchor
        """
        B, A, N, C = pred_shape
            
        # Total anchors = A * N (anchors per position * positions)
        total_anchors = A * N
            
        # Estimate feature map sizes from strides
        # For strides [8, 16, 32] and image_size 320:
        # Level 0: 40x40 = 1600 positions
        # Level 1: 20x20 = 400 positions  
        # Level 2: 10x10 = 100 positions
        # Total: 2100 positions (but we only have N=150, so scale down)
            
        sizes = []
        for stride in self.strides:
            size = int(self.image_size / stride)
            sizes.append((size, size))
            
        all_anchors = []
            
        for level, (stride, (h, w)) in enumerate(zip(self.strides, sizes)):
            # Generate grid
            y_coords = torch.arange(h, device=device, dtype=torch.float32) + 0.5
            x_coords = torch.arange(w, device=device, dtype=torch.float32) + 0.5
                
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
            # Normalize to [0, 1]
            y_grid = y_grid.flatten() * stride / self.image_size
            x_grid = x_grid.flatten() * stride / self.image_size
                
            # Create anchors for each anchor ratio at this level
            for ratio_idx, ratio in enumerate(self.anchor_ratios):
                anchors = torch.stack([
                    x_grid, y_grid,
                    torch.full_like(x_grid, stride * 8 / self.image_size * ratio[0]),
                    torch.full_like(y_grid, stride * 8 / self.image_size * ratio[1]),
                ], dim=1)
                all_anchors.append(anchors)
            
        anchors = torch.cat(all_anchors, dim=0)
            
        # Ensure correct number of anchors (truncate or pad)
        if anchors.shape[0] < total_anchors:
            pad = total_anchors - anchors.shape[0]
            anchors = torch.cat([anchors, anchors[:pad]], dim=0)
        elif anchors.shape[0] > total_anchors:
            anchors = anchors[:total_anchors]
            
        return anchors, torch.arange(total_anchors, device=device)
    
    def _assign_targets(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Assign ground truth boxes to anchors using IoU matching.
        
        Args:
            anchors: (N, 4) anchor boxes [cx, cy, w, h]
            gt_boxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
            gt_labels: (M,) ground truth labels
            device: torch device
            
        Returns:
            Tuple of (positive anchor indices, matched gt indices)
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        
        if num_gt == 0:
            return None
        
        # Convert anchors to [x1, y1, x2, y2] format
        anchor_x1 = anchors[:, 0] - anchors[:, 2] / 2
        anchor_y1 = anchors[:, 1] - anchors[:, 3] / 2
        anchor_x2 = anchors[:, 0] + anchors[:, 2] / 2
        anchor_y2 = anchors[:, 1] + anchors[:, 3] / 2
        anchors_xyxy = torch.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], dim=1)
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(anchors_xyxy, gt_boxes)
        
        # SimOTA-style assignment
        # For each GT, find top-k anchors with highest IoU
        top_k = min(10, num_anchors // num_gt)
        
        # Get best matching anchor for each GT
        gt_matched_anchors = []
        gt_matched_gt_idx = []
        
        for gt_idx in range(num_gt):
            ious = iou_matrix[:, gt_idx]
            # Get top-k anchors for this GT
            topk_vals, topk_idx = ious.topk(min(top_k, len(ious)))
            
            # Filter by IoU threshold
            mask = topk_vals > 0.3  # Lower threshold to get more positives
            if mask.any():
                gt_matched_anchors.append(topk_idx[mask])
                gt_matched_gt_idx.append(torch.full((mask.sum().item(),), gt_idx, 
                                                    dtype=torch.long, device=device))
        
        if len(gt_matched_anchors) == 0:
            # Fallback: use highest IoU match for each GT
            max_ious, max_idx = iou_matrix.max(dim=0)
            mask = max_ious > 0.1
            if not mask.any():
                return None
            return max_idx[mask], torch.arange(num_gt, device=device)[mask]
        
        # Concatenate all assignments
        pos_anchor_idx = torch.cat(gt_matched_anchors)
        matched_gt_idx = torch.cat(gt_matched_gt_idx)
        
        return pos_anchor_idx, matched_gt_idx
    
    def _compute_iou_matrix(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU matrix between two sets of boxes.
        
        Args:
            boxes1: (N, 4) boxes in [x1, y1, x2, y2] format
            boxes2: (M, 4) boxes in [x1, y1, x2, y2] format
            
        Returns:
            (N, M) IoU matrix
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
        boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)
        
        # Intersection
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Union
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        return iou
    
    def _compute_list_distill_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute distillation loss for list-format features.
        
        Args:
            student_features: List of student feature tensors
            teacher_features: List of teacher feature tensors
            
        Returns:
            Distillation loss
        """
        total_loss = 0.0
        num_levels = min(len(student_features), len(teacher_features))
        
        if num_levels == 0:
            return student_features[0].new_tensor(0.0) if len(student_features) > 0 else torch.tensor(0.0)
        
        for i in range(num_levels):
            s_feat = student_features[i]
            t_feat = teacher_features[i]
            
            # Handle different shapes
            if s_feat.shape != t_feat.shape:
                # Match spatial dimensions
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.interpolate(
                        s_feat,
                        size=t_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Match channel dimensions via padding or truncation
                s_ch, t_ch = s_feat.shape[1], t_feat.shape[1]
                if s_ch < t_ch:
                    pad = t_ch - s_ch
                    s_feat = F.pad(s_feat, (0, 0, 0, 0, 0, pad))
                elif s_ch > t_ch:
                    s_feat = s_feat[:, :t_ch]
            
            # MSE loss on normalized features
            s_norm = F.normalize(s_feat, dim=1)
            t_norm = F.normalize(t_feat, dim=1)
            total_loss += F.mse_loss(s_norm, t_norm)
        
        return total_loss / num_levels
    
    def _ciou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CIoU loss for regression.
        
        Args:
            pred_boxes: (N, 4) predicted boxes [cx_off, cy_off, w, h]
            target_boxes: (N, 4) target boxes [cx_off, cy_off, w, h]
            
        Returns:
            CIoU loss
        """
        # Clamp boxes for numerical stability
        pred_boxes = pred_boxes.clamp(min=-10.0, max=10.0)
        target_boxes = target_boxes.clamp(min=-10.0, max=10.0)
        
        # Ensure positive width and height
        w_pred = pred_boxes[:, 2].clamp(min=1e-4)
        h_pred = pred_boxes[:, 3].clamp(min=1e-4)
        w_target = target_boxes[:, 2].clamp(min=1e-4)
        h_target = target_boxes[:, 3].clamp(min=1e-4)
        
        # Convert to x1y1x2y2 format
        pred_x1 = -w_pred / 2
        pred_y1 = -h_pred / 2
        pred_x2 = w_pred / 2
        pred_y2 = h_pred / 2
        
        target_x1 = -w_target / 2
        target_y1 = -h_target / 2
        target_x2 = w_target / 2
        target_y2 = h_target / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        pred_area = w_pred * h_pred
        target_area = w_target * h_target
        
        # Union
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_w = (enclose_x2 - enclose_x1).clamp(min=1e-7)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=1e-7)
        
        # Diagonal squared
        c2 = enclose_w ** 2 + enclose_h ** 2 + 1e-7
        
        # Center distance squared
        cx_pred = pred_boxes[:, 0]
        cy_pred = pred_boxes[:, 1]
        cx_target = target_boxes[:, 0]
        cy_target = target_boxes[:, 1]
        rho2 = (cx_pred - cx_target) ** 2 + (cy_pred - cy_target) ** 2
        
        # Aspect ratio consistency
        v = (4 / (3.14159 ** 2)) * torch.pow(
            torch.atan(w_target / h_target) - torch.atan(w_pred / h_pred), 2
        )
        alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU
        ciou = iou - (rho2 / c2) - alpha * v
        
        return 1 - ciou
