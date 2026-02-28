"""
Tkinter-based Main Window for Lightweight ViT Detection System.

This module provides a cross-platform GUI application using tkinter.
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch


@dataclass
class DetectionResult:
    """Detection result data class."""
    timestamp: datetime
    frame_id: int
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: List[str]
    inference_time: float
    
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


class DetectorThread(threading.Thread):
    """Background thread for object detection."""
    
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
    
    def __init__(self, callback=None, log_callback=None, fps_callback=None, error_callback=None):
        super().__init__(daemon=True)
        self.callback = callback
        self.log_callback = log_callback
        self.fps_callback = fps_callback
        self.error_callback = error_callback
        
        self.model = None
        self.device = None
        self.running = False
        self.paused = False
        
        self.source_type = None
        self.source = None
        self.cap = None
        
        self.conf_threshold = 0.5
        self.target_size = (640, 640)
        
        self.frame_times = []
        self.frame_id = 0
        
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()
        
    def load_model(self, model_path: Optional[str] = None, config_path: Optional[str] = None) -> bool:
        """
        Load pre-trained detection model for inference.
        
        Note: This method loads an already trained/distilled model.
        Training should be done separately before running the GUI.
        
        Args:
            model_path: Path to model weights (.pth file)
            config_path: Path to model config (.yaml file)
            
        Returns:
            True if model loaded successfully
        """
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from src.models.mobilevit import build_mobilevit
            from src.utils.config_loader import load_config
            
            # Default model paths (pre-trained/distilled models)
            default_model_paths = [
                os.path.join(project_root, 'models', 'mobilevit_distilled.pth'),
                os.path.join(project_root, 'models', 'mobilevit_best.pth'),
                os.path.join(project_root, 'models', 'checkpoint_best.pth'),
                os.path.join(project_root, 'outputs', 'mobilevit_distilled.pth'),
                os.path.join(project_root, 'outputs', 'best_model.pth'),
            ]
            
            # Default config path
            default_config = os.path.join(project_root, 'configs', 'model', 'mobilevit.yaml')
            
            # Find model file
            if model_path is None:
                for path in default_model_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
            
            # Load config
            if config_path is None and os.path.exists(default_config):
                config_path = default_config
                
            if config_path and os.path.exists(config_path):
                config = load_config(config_path)
            else:
                config = {
                    'model': {
                        'backbone': {'name': 'mobilevit_s'},
                        'neck': {'type': 'fpn', 'out_channels': 256},
                        'head': {'type': 'retina', 'num_classes': 80}
                    }
                }
            
            # Build model architecture
            self.model = build_mobilevit(config)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load pre-trained weights
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                if self.error_callback:
                    self.error_callback(f"INFO: Loaded model from {os.path.basename(model_path)}")
                return True
            else:
                # No pre-trained weights found - model will use random weights
                if self.error_callback:
                    self.error_callback(
                        "WARNING: No pre-trained model found. Please train a model first or load weights manually.\n"
                        "Expected locations: models/mobilevit_distilled.pth or outputs/best_model.pth"
                    )
                return True  # Model architecture loaded, but no weights
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Failed to load model: {str(e)}")
            return False
            
    def set_source(self, source_type: str, source: Any):
        """Set detection source."""
        with self._lock:
            self.source_type = source_type
            self.source = source
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold."""
        self.conf_threshold = max(0.0, min(1.0, threshold))
        
    def pause(self):
        """Pause detection."""
        self._pause_event.clear()
        self.paused = True
        
    def resume(self):
        """Resume detection."""
        self._pause_event.set()
        self.paused = False
        
    def stop(self):
        """Stop detection."""
        self.running = False
        self._pause_event.set()
        
    def run(self):
        """Main detection loop."""
        self.running = True
        self.frame_id = 0
        self.frame_times = []
        
        while self.running:
            self._pause_event.wait()
            if not self.running:
                break
                
            try:
                frame = self._get_frame()
                if frame is None:
                    if self.source_type == 'video':
                        self.running = False
                    time.sleep(0.01)
                    continue
                    
                start_time = time.time()
                result = self._detect(frame)
                inference_time = (time.time() - start_time) * 1000
                
                self.frame_times.append(time.time())
                self.frame_times = [t for t in self.frame_times if time.time() - t < 1.0]
                fps = len(self.frame_times)
                
                if self.fps_callback:
                    self.fps_callback(fps)
                
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
                    
                    annotated_frame = self._draw_results(frame, detection_result)
                    
                    if self.callback:
                        self.callback(annotated_frame, detection_result)
                    
                    if len(result['boxes']) > 0 and self.log_callback:
                        self.log_callback(detection_result.to_dict())
                else:
                    if self.callback:
                        self.callback(frame, None)
                        
                self.frame_id += 1
                
                if self.source_type == 'image':
                    self.running = False
                    
            except Exception as e:
                if self.error_callback:
                    self.error_callback(f"Detection error: {str(e)}")
                    
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get frame from source."""
        if self.source_type == 'camera':
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    if self.error_callback:
                        self.error_callback(f"Failed to open camera {self.source}")
                    return None
            ret, frame = self.cap.read()
            return frame if ret else None
            
        elif self.source_type == 'video':
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    if self.error_callback:
                        self.error_callback(f"Failed to open video: {self.source}")
                    return None
            ret, frame = self.cap.read()
            return frame if ret else None
            
        elif self.source_type == 'image':
            if isinstance(self.source, np.ndarray):
                return self.source.copy()
            elif isinstance(self.source, str):
                frame = cv2.imread(self.source)
                if frame is None and self.error_callback:
                    self.error_callback(f"Failed to load image: {self.source}")
                return frame
                
        return None
        
    def _detect(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Run detection on frame."""
        if self.model is None:
            return {'boxes': np.array([]).reshape(0, 4), 'scores': np.array([]), 'labels': np.array([], dtype=np.int64)}
            
        try:
            input_tensor = self._preprocess(frame)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Postprocess outputs
            return self._postprocess(outputs, frame.shape[:2])
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Inference error: {str(e)}")
            return None
    
    def _postprocess(self, outputs: Dict[str, torch.Tensor], original_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Postprocess model outputs."""
        # Get predictions - handle both list and tensor formats
        if 'cls_scores' in outputs and 'bbox_preds' in outputs:
            cls_scores = outputs['cls_scores']
            bbox_preds = outputs['bbox_preds']
        else:
            return {'boxes': np.array([]).reshape(0, 4), 'scores': np.array([]), 'labels': np.array([], dtype=np.int64)}
        
        # Handle list format (per-level outputs)
        if isinstance(cls_scores, list):
            all_cls = []
            all_bbox = []
            strides = [8, 16, 32, 64, 128]
            
            for level_idx, (cls_level, bbox_level) in enumerate(zip(cls_scores, bbox_preds)):
                B, C, H, W = cls_level.shape
                num_classes = 80
                num_anchors = C // num_classes
                
                cls_reshaped = cls_level.permute(0, 2, 3, 1).reshape(B, H * W * num_anchors, num_classes)
                bbox_reshaped = bbox_level.permute(0, 2, 3, 1).reshape(B, H * W * num_anchors, 4)
                
                all_cls.append(cls_reshaped[0])
                all_bbox.append(bbox_reshaped[0])
            
            cls_scores = torch.cat(all_cls, dim=0)
            bbox_preds = torch.cat(all_bbox, dim=0)
        else:
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
            return {'boxes': np.array([]).reshape(0, 4), 'scores': np.array([]), 'labels': np.array([], dtype=np.int64)}
        
        max_scores = max_scores[keep]
        labels = labels[keep]
        bbox_preds = bbox_preds[keep]
        
        # Generate anchors and decode boxes
        h, w = original_size
        anchors = []
        strides = [8, 16, 32, 64, 128]
        anchor_sizes = [(20, 20), (40, 40), (80, 80)]
        
        for stride in strides:
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
            return {'boxes': np.array([]).reshape(0, 4), 'scores': np.array([]), 'labels': np.array([], dtype=np.int64)}
        
        return {
            'boxes': np.array(keep_boxes),
            'scores': np.array(keep_scores),
            'labels': np.array(keep_labels, dtype=np.int64)
        }
            
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        resized = cv2.resize(frame, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor
        
    def _draw_results(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection results on frame."""
        annotated = frame.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
        
        for i in range(len(result.boxes)):
            box = result.boxes[i].astype(int)
            score = result.scores[i]
            label_idx = int(result.labels[i])
            
            class_name = result.class_names[label_idx] if label_idx < len(result.class_names) else f'class_{label_idx}'
            color = colors[label_idx % len(colors)]
            
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            label_text = f'{class_name}: {score:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (box[0], box[1] - text_h - 5), (box[0] + text_w, box[1]), color, -1)
            cv2.putText(annotated, label_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        info_text = f'FPS: {len(self.frame_times):.1f} | Detections: {len(result.boxes)}'
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated


class MainWindow:
    """Main application window using tkinter."""
    
    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root or tk.Tk()
        self.root.title("Lightweight ViT Detection System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        self.detector_thread: Optional[DetectorThread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        
        self.total_detections = 0
        self.frame_count = 0
        self.current_fps = 0.0
        
        self._setup_style()
        self._setup_ui()
        self._setup_detector()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_style(self):
        """Setup ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#2d2d2d')
        style.configure('TLabel', background='#2d2d2d', foreground='#ffffff')
        style.configure('TButton', padding=6)
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video display
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video canvas
        video_frame = ttk.LabelFrame(left_frame, text="Video Display", padding="5")
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(video_frame, bg='#1e1e1e', width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Log panel
        log_frame = ttk.LabelFrame(left_frame, text="Detection Log", padding="5")
        log_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, bg='#1e1e1e', fg='#d4d4d4',
                                                   font=('Consolas', 9))
        self.log_text.pack(fill=tk.X)
        
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(log_btn_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_btn_frame, text="Save Log", command=self._save_log).pack(side=tk.LEFT)
        
        # Right panel - Controls
        right_frame = ttk.Frame(main_frame, width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Input source
        source_frame = ttk.LabelFrame(right_frame, text="Input Source", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        cam_frame = ttk.Frame(source_frame)
        cam_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(cam_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(cam_frame, textvariable=self.camera_var, values=["0", "1", "2"], width=5)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(cam_frame, text="Start Camera", command=self._start_camera).pack(side=tk.LEFT)
        
        file_frame = ttk.Frame(source_frame)
        file_frame.pack(fill=tk.X)
        ttk.Button(file_frame, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Open Video", command=self._open_video).pack(side=tk.LEFT)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(right_frame, text="Detection Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = ttk.Scale(conf_frame, from_=0, to=1, variable=self.conf_var, orient=tk.HORIZONTAL)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.50", width=4)
        self.conf_label.pack(side=tk.LEFT)
        self.conf_scale.config(command=self._on_conf_change)
        
        # Controls
        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        self.pause_btn = ttk.Button(btn_frame, text="Pause", command=self._toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Model
        model_frame = ttk.LabelFrame(right_frame, text="Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(model_frame, text="Load Model Weights", command=self._load_model).pack(fill=tk.X)
        self.model_label = ttk.Label(model_frame, text="Model: Default", wraplength=250)
        self.model_label.pack(fill=tk.X, pady=(5, 0))
        
        # Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: --", style='Header.TLabel')
        self.fps_label.pack(anchor=tk.W)
        self.detection_label = ttk.Label(stats_frame, text="Total Detections: 0")
        self.detection_label.pack(anchor=tk.W)
        self.frame_label = ttk.Label(stats_frame, text="Frames Processed: 0")
        self.frame_label.pack(anchor=tk.W)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load a model or select input source")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _setup_detector(self):
        """Initialize detector for inference (not training)."""
        self._add_log("Initializing inference engine...", "INFO")
        self._add_log("Looking for pre-trained model weights...", "INFO")
        
        self.detector_thread = DetectorThread(
            callback=self._on_frame_ready,
            log_callback=self._on_detection_logged,
            fps_callback=self._on_fps_update,
            error_callback=lambda msg: self._add_log(msg, "WARNING" if "WARNING" in msg else "INFO")
        )
        
        if self.detector_thread.load_model():
            self._add_log("Model architecture loaded successfully", "INFO")
            self._add_log("Ready for inference. Select input source to start detection.", "INFO")
            self.model_label.config(text="Model: MobileViT (inference mode)")
        else:
            self._add_log("Failed to initialize model", "ERROR")
            self.model_label.config(text="Model: Not loaded")
            
    def _add_log(self, message: str, level: str = "INFO"):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] [{level}] {message}\n")
        self.log_text.see(tk.END)
        
    def _clear_log(self):
        """Clear log."""
        self.log_text.delete(1.0, tk.END)
        
    def _save_log(self):
        """Save log to file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self._add_log(f"Log saved to: {file_path}", "INFO")
            
    def _on_conf_change(self, value):
        """Handle confidence slider change."""
        conf = float(value)
        self.conf_label.config(text=f"{conf:.2f}")
        if self.detector_thread:
            self.detector_thread.set_confidence_threshold(conf)
            
    def _start_camera(self):
        """Start camera detection."""
        if self.detector_thread and self.detector_thread.is_alive():
            self._stop_detection()
            
        camera_idx = int(self.camera_var.get())
        self._add_log(f"Starting camera {camera_idx}...", "INFO")
        
        self.detector_thread = DetectorThread(
            callback=self._on_frame_ready,
            log_callback=self._on_detection_logged,
            fps_callback=self._on_fps_update,
            error_callback=self._on_error
        )
        self.detector_thread.load_model()
        self.detector_thread.set_source('camera', camera_idx)
        self.detector_thread.start()
        
        self._set_running_state(True)
        self.status_var.set(f"Camera {camera_idx} - Running")
        
    def _open_image(self):
        """Open image file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            if self.detector_thread and self.detector_thread.is_alive():
                self._stop_detection()
                
            self._add_log(f"Processing image: {os.path.basename(file_path)}", "INFO")
            
            self.detector_thread = DetectorThread(
                callback=self._on_frame_ready,
                log_callback=self._on_detection_logged,
                fps_callback=self._on_fps_update,
                error_callback=self._on_error
            )
            self.detector_thread.load_model()
            self.detector_thread.set_source('image', file_path)
            self.detector_thread.start()
            
            self._set_running_state(True)
            self.status_var.set(f"Image: {os.path.basename(file_path)}")
            
    def _open_video(self):
        """Open video file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        if file_path:
            if self.detector_thread and self.detector_thread.is_alive():
                self._stop_detection()
                
            self._add_log(f"Processing video: {os.path.basename(file_path)}", "INFO")
            
            self.detector_thread = DetectorThread(
                callback=self._on_frame_ready,
                log_callback=self._on_detection_logged,
                fps_callback=self._on_fps_update,
                error_callback=self._on_error
            )
            self.detector_thread.load_model()
            self.detector_thread.set_source('video', file_path)
            self.detector_thread.start()
            
            self._set_running_state(True)
            self.status_var.set(f"Video: {os.path.basename(file_path)}")
            
    def _toggle_pause(self):
        """Toggle pause/resume."""
        if self.detector_thread:
            if self.detector_thread.paused:
                self.detector_thread.resume()
                self.pause_btn.config(text="Pause")
                self._add_log("Detection resumed", "INFO")
            else:
                self.detector_thread.pause()
                self.pause_btn.config(text="Resume")
                self._add_log("Detection paused", "INFO")
                
    def _stop_detection(self):
        """Stop detection."""
        if self.detector_thread:
            self.detector_thread.stop()
            self.detector_thread.join(timeout=3.0)
            
        self._set_running_state(False)
        self._add_log("Detection stopped", "INFO")
        self.status_var.set("Stopped")
        
    def _load_model(self):
        """Load model weights."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Model files", "*.pth *.pt *.ckpt")]
        )
        if file_path:
            self._add_log(f"Loading model: {os.path.basename(file_path)}", "INFO")
            if self.detector_thread and self.detector_thread.load_model(model_path=file_path):
                self.model_label.config(text=f"Model: {os.path.basename(file_path)}")
                self._add_log("Model loaded successfully", "INFO")
            else:
                self._add_log("Failed to load model", "ERROR")
                
    def _set_running_state(self, running: bool):
        """Set UI state."""
        state = tk.NORMAL if running else tk.DISABLED
        self.pause_btn.config(state=state)
        self.stop_btn.config(state=state)
        self.pause_btn.config(text="Pause")
        
    def _on_frame_ready(self, frame: np.ndarray, result: Optional[DetectionResult]):
        """Handle processed frame."""
        self.current_frame = frame
        self.frame_count += 1
        
        # Schedule UI update in main thread
        self.root.after(0, self._update_video_display)
        self.root.after(0, lambda: self.frame_label.config(text=f"Frames Processed: {self.frame_count}"))
        
    def _update_video_display(self):
        """Update video display."""
        if self.current_frame is None:
            return
            
        frame = self.current_frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            
        image = Image.fromarray(frame_rgb)
        self.photo_image = ImageTk.PhotoImage(image)
        
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            self.video_canvas.winfo_width() // 2,
            self.video_canvas.winfo_height() // 2,
            image=self.photo_image, anchor=tk.CENTER
        )
        
    def _on_detection_logged(self, result_dict: dict):
        """Handle detection logging."""
        num_det = result_dict.get('num_detections', 0)
        self.total_detections += num_det
        
        if num_det > 0:
            detections = result_dict.get('detections', [])
            classes = [d['class'] for d in detections[:3]]
            classes_str = ", ".join(classes)
            if len(detections) > 3:
                classes_str += f" +{len(detections)-3} more"
                
            def update():
                self._add_log(f"Detected {num_det} objects: {classes_str}", "DETECTION")
                self.detection_label.config(text=f"Total Detections: {self.total_detections}")
                
            self.root.after(0, update)
            
    def _on_fps_update(self, fps: float):
        """Handle FPS update."""
        self.current_fps = fps
        self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f}"))
        
    def _on_error(self, error_msg: str):
        """Handle error."""
        self.root.after(0, lambda: self._add_log(error_msg, "ERROR"))
        
    def _on_closing(self):
        """Handle window close."""
        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.stop()
            self.detector_thread.join(timeout=3.0)
        self.root.destroy()
        
    def run(self):
        """Run the application."""
        self.root.mainloop()
        
    def show(self):
        """Show the window (compatibility method)."""
        pass  # tkinter shows immediately


def main():
    """Main entry point."""
    root = tk.Tk()
    app = MainWindow(root)
    app.run()


if __name__ == "__main__":
    main()
