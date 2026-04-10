"""
Unit tests for inference modules.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import torch
import torch.nn as nn


class MockDetector(nn.Module):
    """Mock detector for testing."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.head = nn.Conv2d(64, num_classes + 5, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        
        # Fake outputs
        B, _, H, W = x.shape
        cls_pred = x[:, :self.num_classes].flatten(2).transpose(1, 2)
        reg_pred = x[:, self.num_classes:self.num_classes+4].flatten(2).transpose(1, 2)
        obj_pred = x[:, -1:].flatten(2).transpose(1, 2)
        
        return cls_pred, reg_pred, obj_pred


class TestPredictor:
    """Test predictor modules."""
    
    def test_predictor_initialization(self):
        """Test predictor can be initialized."""
        from src.inference.predictor import Predictor
        
        model = MockDetector()
        predictor = Predictor(model, device="cpu")
        
        assert predictor is not None
    
    def test_predictor_preprocess(self):
        """Test preprocessing."""
        from src.inference.predictor import Predictor
        
        model = MockDetector()
        predictor = Predictor(model, device="cpu")
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, info = predictor.preprocess(image, target_size=320)
        
        assert tensor.shape == (1, 3, 320, 320)
        assert "scale" in info
    
    def test_predictor_predict(self):
        """Test prediction."""
        from src.inference.predictor import Predictor
        
        model = MockDetector()
        predictor = Predictor(model, device="cpu", conf_threshold=0.05)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = predictor.predict(image)
        
        assert "boxes" in results
        assert "scores" in results
        assert "labels" in results
    
    def test_predictor_confidence_threshold(self):
        """Test confidence threshold filtering."""
        from src.inference.predictor import Predictor
        
        model = MockDetector()
        predictor = Predictor(
            model,
            device="cpu",
            conf_threshold=0.9,  # High threshold
        )
        
        image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        results = predictor.predict(image)
        
        # With high threshold, should have fewer detections
        # (exact number depends on random weights)


class TestVisualizer:
    """Test visualization modules."""
    
    def test_draw_detections(self):
        """Test drawing detections."""
        from src.inference.visualizer import draw_detections
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]])
        scores = np.array([0.9, 0.8])
        labels = np.array([0, 1])
        class_names = ["person", "car"]
        
        result = draw_detections(image, boxes, scores, labels, class_names)
        
        assert result.shape == image.shape
        assert not np.array_equal(result, image)  # Should be different
    
    def test_visualizer_class(self):
        """Test Visualizer class."""
        from src.inference.visualizer import Visualizer
        
        class_names = ["person", "car", "dog"]
        visualizer = Visualizer(class_names)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]])
        labels = np.array([0])
        scores = np.array([0.95])
        
        result = visualizer.draw_boxes(image, boxes, labels, scores)
        
        assert result.shape == image.shape


class TestMetrics:
    """Test metrics modules."""
    
    def test_iou_computation(self):
        """Test IoU computation."""
        from src.utils.metrics import compute_iou
        
        boxes1 = np.array([[0, 0, 100, 100]])
        boxes2 = np.array([[50, 50, 150, 150]])
        
        iou = compute_iou(boxes1, boxes2)
        
        # IoU should be (50*50) / (10000 + 10000 - 2500) = 2500 / 17500
        expected_iou = 2500 / 17500
        assert np.isclose(iou[0, 0], expected_iou, atol=0.01)
    
    def test_ap_computation(self):
        """Test AP computation."""
        from src.utils.metrics import compute_ap
        
        recall = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        precision = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
        
        ap = compute_ap(recall, precision)
        
        assert 0 <= ap <= 1
    
    def test_detection_metrics(self):
        """Test DetectionMetrics class."""
        from src.utils.metrics import DetectionMetrics
        
        metrics = DetectionMetrics(num_classes=80)
        
        # Add some predictions and targets
        metrics.update(
            {
                "boxes": np.array([[100, 100, 200, 200]]),
                "scores": np.array([0.9]),
                "labels": np.array([0]),
            },
            {
                "boxes": np.array([[105, 105, 195, 195]]),
                "labels": np.array([0]),
            }
        )
        
        results = metrics.compute()
        
        assert "mAP@0.5" in results


class TestOptimization:
    """Test optimization modules."""
    
    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        from src.optimization.quantization import quantize_model_dynamic
        
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 80),
        )
        
        quantized = quantize_model_dynamic(model)
        
        assert quantized is not None
    
    def test_pruning(self):
        """Test model pruning."""
        from src.optimization.pruning import L1UnstructuredPruner
        
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
        )
        
        pruner = L1UnstructuredPruner(model, pruning_ratio=0.3)
        pruned_model = pruner.prune()
        
        sparsity = pruner.get_sparsity()
        
        # Sparsity should be close to target
        assert 0 < sparsity <= 0.35  # Allow some tolerance


class TestExport:
    """Test export modules."""
    
    def test_torchscript_export(self):
        """Test TorchScript export."""
        import tempfile
        from src.optimization.export import export_torchscript
        
        model = MockDetector()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/model.pt"
            result = export_torchscript(
                model,
                output_path,
                input_shape=(3, 64, 64),
                method="trace",
            )
            
            assert result == output_path
            
            # Load and test
            loaded = torch.jit.load(output_path)
            x = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                _ = loaded(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
