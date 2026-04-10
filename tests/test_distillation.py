"""
Unit tests for distillation modules.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.fc = nn.Linear(64, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


class SimpleDetector(nn.Module):
    """Simple detector for testing."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.head = nn.Conv2d(128, num_classes + 5, 1)
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.head(x)
        
        # Fake outputs
        B, _, H, W = x.shape
        cls_pred = x[:, :self.num_classes].flatten(2).transpose(1, 2)
        reg_pred = x[:, self.num_classes:self.num_classes+4].flatten(2).transpose(1, 2)
        obj_pred = x[:, -1:].flatten(2).transpose(1, 2)
        
        return cls_pred, reg_pred, obj_pred


class TestTeacherModel:
    """Test teacher model wrapper."""
    
    def test_teacher_wrapper(self):
        """Test teacher model wrapper."""
        from src.distillation.teacher import TeacherModel, wrap_teacher
        
        model = SimpleDetector()
        teacher = wrap_teacher(model, freeze=True)
        
        # Check frozen
        for param in teacher.parameters():
            assert not param.requires_grad
        
        # Test forward
        x = torch.randn(2, 3, 64, 64)
        outputs = teacher(x)
        
        assert len(outputs) >= 3  # cls, reg, obj, features
    
    def test_teacher_eval_mode(self):
        """Test teacher always in eval mode."""
        from src.distillation.teacher import TeacherModel
        
        model = SimpleDetector()
        teacher = TeacherModel(model, freeze=True)
        
        teacher.train()  # Try to set to train mode
        
        # Should still be in eval mode
        assert not teacher.training


class TestStudentModel:
    """Test student model wrapper."""
    
    def test_student_wrapper(self):
        """Test student model wrapper."""
        from src.distillation.student import StudentModel, wrap_student
        
        model = SimpleDetector()
        student = wrap_student(model)
        
        # Check trainable
        for param in student.parameters():
            assert param.requires_grad
        
        # Test forward
        x = torch.randn(2, 3, 64, 64)
        outputs = student(x)
        
        assert len(outputs) >= 3


class TestDistillationLosses:
    """Test distillation loss functions."""
    
    def test_response_distillation_loss(self):
        """Test response distillation loss."""
        from src.distillation.losses import ResponseDistillationLoss
        
        loss_fn = ResponseDistillationLoss(temperature=4.0)
        
        student_logits = torch.randn(2, 100, 80)
        teacher_logits = torch.randn(2, 100, 80)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
    
    def test_feature_distillation_loss(self):
        """Test feature distillation loss."""
        from src.distillation.losses import FeatureDistillationLoss
        
        loss_fn = FeatureDistillationLoss(loss_type="mse")
        
        student_features = torch.randn(2, 128, 8, 8)
        teacher_features = torch.randn(2, 128, 8, 8)
        
        loss = loss_fn(student_features, teacher_features)
        
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_relation_distillation_loss(self):
        """Test relation distillation loss."""
        from src.distillation.losses import RelationDistillationLoss
        
        loss_fn = RelationDistillationLoss(loss_type="spatial_attention")
        
        student_features = torch.randn(2, 128, 8, 8)
        teacher_features = torch.randn(2, 128, 8, 8)
        
        loss = loss_fn(student_features, teacher_features)
        
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_combined_distillation_loss(self):
        """Test combined distillation loss."""
        from src.distillation.losses import DistillationLoss
        
        loss_fn = DistillationLoss(
            response_weight=1.0,
            feature_weight=0.5,
            relation_weight=0.3,
        )
        
        student_outputs = {"cls_pred": torch.randn(2, 100, 80)}
        teacher_outputs = {"cls_pred": torch.randn(2, 100, 80)}
        student_features = {"layer1": torch.randn(2, 64, 8, 8)}
        teacher_features = {"layer1": torch.randn(2, 64, 8, 8)}
        
        losses = loss_fn(
            student_outputs,
            teacher_outputs,
            student_features,
            teacher_features,
        )
        
        assert "total_distill_loss" in losses
        assert losses["total_distill_loss"].item() >= 0


class TestDistillationTrainer:
    """Test distillation trainer."""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        from src.distillation.trainer import DistillationTrainer, DetectionLoss
        
        teacher = SimpleDetector()
        student = SimpleDetector()
        
        # Create fake dataloaders
        class FakeDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return {
                    "images": torch.randn(3, 64, 64),
                    "targets": [{
                        "boxes": torch.randn(2, 4),
                        "labels": torch.randint(0, 80, (2,)),
                    }]
                }
        
        train_loader = torch.utils.data.DataLoader(
            FakeDataset(), batch_size=2, collate_fn=lambda x: x[0]
        )
        val_loader = train_loader
        
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device="cpu",
        )
        
        assert trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
