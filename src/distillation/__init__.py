"""
Knowledge Distillation modules for the Lightweight ViT Detection System.
"""

from .teacher import TeacherModel, wrap_teacher
from .student import StudentModel, wrap_student
from .losses import (
    DistillationLoss,
    ResponseDistillationLoss,
    FeatureDistillationLoss,
    RelationDistillationLoss,
)
from .trainer import DistillationTrainer
from .efficientformer_teacher import EfficientFormerV2Teacher, create_efficientformer_teacher
from .feature_adapter import (
    FeatureAdapter,
    MultiLevelFeatureAdapter,
    FeatureDistillationWithAdapter,
)

__all__ = [
    "TeacherModel",
    "wrap_teacher",
    "StudentModel",
    "wrap_student",
    "DistillationLoss",
    "ResponseDistillationLoss",
    "FeatureDistillationLoss",
    "RelationDistillationLoss",
    "DistillationTrainer",
    "EfficientFormerV2Teacher",
    "create_efficientformer_teacher",
    "FeatureAdapter",
    "MultiLevelFeatureAdapter",
    "FeatureDistillationWithAdapter",
]
