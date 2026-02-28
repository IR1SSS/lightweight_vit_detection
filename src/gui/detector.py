"""
Detector Thread for GUI Application.

This module provides a background thread for running object detection
without blocking the main UI thread.
"""

import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

import cv2
import numpy as np
import torch

# PyQt6 is optional - only needed for Qt-based GUI
try:
    from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # Dummy classes when PyQt6 is not available
    class QThread:
        pass
    def pyqtSignal(*args):
        return None
    class QMutex:
        pass
    class QWaitCondition:
        pass


@dataclass
class DetectionResult:
    """Detection result data class."""
    timestamp: datetime
    frame_id: int
    boxes: np.ndarray  # Shape: (N, 4) in xyxy format
    scores: np.ndarray  # Shape: (N,)
    labels: np.ndarray  # Shape: (N,)
    class_names: List[str]
    inference_time: float  # milliseconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'frame_id': self.frame_id,
            'num_detections': len(self.boxes),
            'inference_time_ms': self.inference_time,
            'detections': [
                {
                    'class': self.class_names[i] if i < len(self.class_names) else f'class_{self.labels[i]}',
                    'confidence': float(self.scores[i]),
                    'bbox': self.boxes[i].tolist()
                }
                for i in range(len(self.boxes))
            ]
        }


class DetectorThread(QThread):
    """
    Background thread for object detection.
    
    Signals:
        frame_ready: Emitted when a processed frame is ready (frame, result)
        detection_logged: Emitted when a detection should be logged (result_dict)
        error_occurred: Emitted when an error occurs (error_message)
        fps_updated: Emitted with current FPS value
    """
    
    frame_ready = pyqtSignal(np.ndarray, object)  # frame, DetectionResult
    detection_logged = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.model = None
        self.device = None
        self.running = False
        self.paused = False
        
        # Source settings
        self.source_type = None  # 'camera', 'image', 'video'
        self.source = None  # camera index, file path, or numpy array
        self.cap = None  # VideoCapture object
        
        # Detection settings
        self.conf_threshold = 0.5
        self.nms_threshold = 0.45
        self.target_size = (640, 640)
        
        # Thread control
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # FPS calculation
        self.frame_times = []
        self.frame_id = 0
        
    def load_model(self, model_path: Optional[str] = None, config_path: Optional[str] = None) -> bool:
        """
        Load detection model.
        
        Args:
            model_path: Path to model checkpoint (optional)
            config_path: Path to model config (optional)
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Import model builder
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from src.models.mobilevit import build_mobilevit
            from src.utils.config_loader import load_config
            
            # Load config
            if config_path and os.path.exists(config_path):
                config = load_config(config_path)
            else:
                # Use default config
                config = {
                    'model': {
                        'backbone': {'name': 'mobilevit_s'},
                        'neck': {'type': 'fpn', 'out_channels': 256},
                        'head': {'type': 'retina', 'num_classes': 80}
                    }
                }
            
            # Build model
            self.model = build_mobilevit(config)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load weights if provided
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                    
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model: {str(e)}")
            return False
            
    def set_source(self, source_type: str, source: Any):
        """
        Set detection source.
        
        Args:
            source_type: 'camera', 'image', or 'video'
            source: Camera index, file path, or numpy array
        """
        self.mutex.lock()
        self.source_type = source_type
        self.source = source
        
        # Release previous capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.mutex.unlock()
        
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold."""
        self.conf_threshold = max(0.0, min(1.0, threshold))
        
    def set_nms_threshold(self, threshold: float):
        """Set NMS threshold."""
        self.nms_threshold = max(0.0, min(1.0, threshold))
        
    def pause(self):
        """Pause detection."""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
    def resume(self):
        """Resume detection."""
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
    def stop(self):
        """Stop detection thread."""
        self.mutex.lock()
        self.running = False
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
    def run(self):
        """Main detection loop."""
        self.running = True
        self.frame_id = 0
        self.frame_times = []
        
        while self.running:
            # Check pause state
            self.mutex.lock()
            while self.paused and self.running:
                self.condition.wait(self.mutex)
            self.mutex.unlock()
            
            if not self.running:
                break
                
            try:
                frame = self._get_frame()
                if frame is None:
                    if self.source_type == 'video':
                        # Video ended
                        self.running = False
                    continue
                    
                # Run detection
                start_time = time.time()
                result = self._detect(frame)
                inference_time = (time.time() - start_time) * 1000
                
                # Update FPS
                self.frame_times.append(time.time())
                self.frame_times = [t for t in self.frame_times if time.time() - t < 1.0]
                fps = len(self.frame_times)
                self.fps_updated.emit(fps)
                
                # Create result
                if result is not None:
                    detection_result = DetectionResult(
                        timestamp=datetime.now(),
                        frame_id=self.frame_id,
                        boxes=result['boxes'],
                        scores=result['scores'],
                        labels=result['labels'],
                        class_names=self.COCO_CLASSES,
                        inference_time=inference_time
                    )
                    
                    # Draw results on frame
                    annotated_frame = self._draw_results(frame, detection_result)
                    
                    # Emit signals
                    self.frame_ready.emit(annotated_frame, detection_result)
                    
                    # Log if detections found
                    if len(result['boxes']) > 0:
                        self.detection_logged.emit(detection_result.to_dict())
                else:
                    self.frame_ready.emit(frame, None)
                    
                self.frame_id += 1
                
                # For image source, stop after one detection
                if self.source_type == 'image':
                    self.running = False
                    
            except Exception as e:
                self.error_occurred.emit(f"Detection error: {str(e)}")
                
        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get frame from source."""
        if self.source_type == 'camera':
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    self.error_occurred.emit(f"Failed to open camera {self.source}")
                    return None
            ret, frame = self.cap.read()
            return frame if ret else None
            
        elif self.source_type == 'video':
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    self.error_occurred.emit(f"Failed to open video: {self.source}")
                    return None
            ret, frame = self.cap.read()
            return frame if ret else None
            
        elif self.source_type == 'image':
            if isinstance(self.source, np.ndarray):
                return self.source.copy()
            elif isinstance(self.source, str):
                frame = cv2.imread(self.source)
                if frame is None:
                    self.error_occurred.emit(f"Failed to load image: {self.source}")
                return frame
                
        return None
        
    def _detect(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Run detection on frame."""
        if self.model is None:
            return None
            
        try:
            # Preprocess
            h, w = frame.shape[:2]
            input_tensor = self._preprocess(frame)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
            # Postprocess
            result = self._postprocess(outputs, (h, w))
            return result
            
        except Exception as e:
            self.error_occurred.emit(f"Inference error: {str(e)}")
            return None
            
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Resize
        resized = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # To tensor: HWC -> CHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
        
    def _postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Postprocess model outputs."""
        import torch.nn.functional as F
        
        # Get predictions - handle both list and tensor formats
        if 'cls_scores' in outputs and 'bbox_preds' in outputs:
            cls_scores = outputs['cls_scores']
            bbox_preds = outputs['bbox_preds']
        else:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'scores': np.array([]),
                'labels': np.array([], dtype=np.int64)
            }
        
        # Handle list format (per-level outputs)
        if isinstance(cls_scores, list):
            all_cls = []
            all_bbox = []
            strides = [8, 16, 32, 64, 128]
            
            for level_idx, (cls_level, bbox_level) in enumerate(zip(cls_scores, bbox_preds)):
                # cls_level: (B, num_anchors*num_classes, H, W)
                # bbox_level: (B, num_anchors*4, H, W)
                B, C, H, W = cls_level.shape
                num_classes = 80
                num_anchors = C // num_classes
                
                # Reshape: (B, num_anchors*num_classes, H, W) -> (B, H*W*num_anchors, num_classes)
                cls_reshaped = cls_level.permute(0, 2, 3, 1).reshape(B, H * W * num_anchors, num_classes)
                bbox_reshaped = bbox_level.permute(0, 2, 3, 1).reshape(B, H * W * num_anchors, 4)
                
                all_cls.append(cls_reshaped[0])  # (H*W*num_anchors, num_classes)
                all_bbox.append(bbox_reshaped[0])  # (H*W*num_anchors, 4)
            
            cls_scores = torch.cat(all_cls, dim=0)
            bbox_preds = torch.cat(all_bbox, dim=0)
        else:
            # Handle tensor format
            cls_scores = cls_scores[0]
            bbox_preds = bbox_preds[0]
        
        # Apply sigmoid to get scores
        scores = torch.sigmoid(cls_scores)
        
        # Get max score and label for each anchor
        max_scores, labels = scores.max(dim=1)
        
        # Filter by threshold
        score_threshold = 0.3
        keep = max_scores > score_threshold
        if keep.sum() == 0:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'scores': np.array([]),
                'labels': np.array([], dtype=np.int64)
            }
        
        max_scores = max_scores[keep]
        labels = labels[keep]
        bbox_preds = bbox_preds[keep]
        
        # Generate anchors and decode boxes
        h, w = original_size
        
        # Build anchors matching the model output
        anchors = []
        strides = [8, 16, 32, 64, 128]
        anchor_sizes = [(20, 20), (40, 40), (80, 80)]
        
        for level_idx, stride in enumerate(strides):
            feat_h = h // stride
            feat_w = w // stride
            
            for y in range(feat_h):
                for x in range(feat_w):
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    
                    for w_s, h_s in anchor_sizes:
                        x1 = cx - w_s / 2
                        y1 = cy - h_s / 2
                        x2 = cx + w_s / 2
                        y2 = cy + h_s / 2
                        anchors.append([x1, y1, x2, y2])
        
        anchors = torch.tensor(anchors, device=bbox_preds.device)
        
        # Ensure same length
        min_len = min(len(anchors), len(bbox_preds))
        anchors = anchors[:min_len]
        bbox_preds = bbox_preds[:min_len]
        max_scores = max_scores[:min_len]
        labels = labels[:min_len]
        
        # Decode boxes
        dx = bbox_preds[:, 0]
        dy = bbox_preds[:, 1]
        dw = bbox_preds[:, 2]
        dh = bbox_preds[:, 3]
        
        anc_x = (anchors[:, 0] + anchors[:, 2]) / 2
        anc_y = (anchors[:, 1] + anchors[:, 3]) / 2
        anc_w = anchors[:, 2] - anchors[:, 0]
        anc_h = anchors[:, 3] - anchors[:, 1]
        
        pred_cx = anc_x + dx * anc_w
        pred_cy = anc_y + dy * anc_h
        pred_w = anc_w * torch.exp(dw)
        pred_h = anc_h * torch.exp(dh)
        
        boxes = torch.stack([
            pred_cx - pred_w / 2,
            pred_cy - pred_h / 2,
            pred_cx + pred_w / 2,
            pred_cy + pred_h / 2
        ], dim=1)
        
        boxes[:, 0] = boxes[:, 0].clamp(0, w)
        boxes[:, 1] = boxes[:, 1].clamp(0, h)
        boxes[:, 2] = boxes[:, 2].clamp(0, w)
        boxes[:, 3] = boxes[:, 3].clamp(0, h)
        
        # Apply NMS per class
        nms_threshold = 0.5
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        
        for class_id in range(int(labels.max().item()) + 1):
            class_mask = labels == class_id
            if class_mask.sum() == 0:
                continue
            
            class_boxes = boxes[class_mask]
            class_scores = max_scores[class_mask]
            
            x1 = class_boxes[:, 0]
            y1 = class_boxes[:, 1]
            x2 = class_boxes[:, 2]
            y2 = class_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            
            _, order = class_scores.sort(descending=True)
            
            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                
                i = order[0].item()
                keep.append(i)
                
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                
                w_i = (xx2 - xx1).clamp(min=0)
                h_i = (yy2 - yy1).clamp(min=0)
                inter = w_i * h_i
                
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                
                ids = (ovr <= nms_threshold).nonzero(as_tuple=False).squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids + 1]
            
            for idx in keep:
                mask_idx = (class_mask.nonzero(as_tuple=True)[0])
                global_idx = mask_idx[idx].item()
                keep_boxes.append(boxes[global_idx].cpu().numpy())
                keep_scores.append(max_scores[global_idx].item())
                keep_labels.append(class_id)
        
        if len(keep_boxes) == 0:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'scores': np.array([]),
                'labels': np.array([], dtype=np.int64)
            }
        
        return {
            'boxes': np.array(keep_boxes),
            'scores': np.array(keep_scores),
            'labels': np.array(keep_labels, dtype=np.int64)
        }
        
    def _draw_results(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection results on frame."""
        annotated = frame.copy()
        
        for i in range(len(result.boxes)):
            box = result.boxes[i].astype(int)
            score = result.scores[i]
            label_idx = result.labels[i]
            
            # Get class name
            if label_idx < len(result.class_names):
                class_name = result.class_names[label_idx]
            else:
                class_name = f'class_{label_idx}'
                
            # Draw box
            color = self._get_color(label_idx)
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label
            label_text = f'{class_name}: {score:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (box[0], box[1] - text_height - 5),
                (box[0] + text_width, box[1]),
                color, -1
            )
            cv2.putText(
                annotated, label_text,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
        # Draw info
        info_text = f'FPS: {len(self.frame_times):.1f} | Detections: {len(result.boxes)}'
        cv2.putText(
            annotated, info_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        return annotated
        
    def _get_color(self, label_idx: int) -> Tuple[int, int, int]:
        """Get color for class label."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[label_idx % len(colors)]
