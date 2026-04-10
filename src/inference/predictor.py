"""
Image predictor for object detection inference.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Predictor:
    """
    Base predictor for object detection models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Detection model
            device: Device for inference
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            max_detections: Maximum detections per image
        """
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Warm up
        self._warmup()
    
    def _warmup(self, input_size: int = 320):
        """Warm up model for inference."""
        dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, str, Path],
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on an image.
        
        Args:
            image: Input image (array or path)
            
        Returns:
            Dictionary with boxes, scores, labels
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor, preproc_info = self.preprocess(image)
        
        # Inference
        outputs = self.model(input_tensor)
        
        # Postprocess
        results = self.postprocess(outputs, preproc_info)
        
        return results
    
    def preprocess(
        self,
        image: np.ndarray,
        target_size: int = 320,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, C)
            target_size: Target size for resizing
            
        Returns:
            Tuple of (input tensor, preprocessing info)
        """
        h, w = image.shape[:2]
        
        # Resize with letterbox
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize and convert to tensor
        input_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Preprocessing info for postprocessing
        preproc_info = {
            "original_shape": (h, w),
            "scale": scale,
            "pad_top": pad_top,
            "pad_left": pad_left,
        }
        
        return input_tensor, preproc_info
    
    def postprocess(
        self,
        outputs: Tuple,
        preproc_info: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            preproc_info: Preprocessing information
            
        Returns:
            Dictionary with boxes, scores, labels
        """
        cls_pred, reg_pred, obj_pred = outputs[:3]
        
        # Convert to numpy
        cls_pred = cls_pred.cpu().numpy()
        reg_pred = reg_pred.cpu().numpy()
        obj_pred = obj_pred.cpu().numpy()
        
        # Get predictions
        boxes = reg_pred[0].reshape(-1, 4)
        scores = cls_pred[0].sigmoid().reshape(-1, self.model.num_classes)
        obj_scores = obj_pred[0].sigmoid().reshape(-1, 1)
        
        # Combined scores
        scores = scores * obj_scores
        max_scores = scores.max(axis=1)
        labels = scores.argmax(axis=1)
        
        # Filter by confidence
        mask = max_scores > self.conf_threshold
        boxes = boxes[mask]
        scores = max_scores[mask]
        labels = labels[mask]
        
        # Scale boxes back
        scale = preproc_info["scale"]
        pad_top = preproc_info["pad_top"]
        pad_left = preproc_info["pad_left"]
        
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        
        # Clip to original image bounds
        h, w = preproc_info["original_shape"]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        
        # NMS
        if len(boxes) > 0:
            keep = self._nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Limit detections
            if len(boxes) > self.max_detections:
                indices = np.argsort(scores)[::-1][:self.max_detections]
                boxes = boxes[indices]
                scores = scores[indices]
                labels = labels[indices]
        
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Non-maximum suppression."""
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Compute IoU
            ious = self._compute_iou(boxes[i:i+1], boxes[indices[1:]])
            
            # Keep boxes with IoU below threshold
            mask = ious.squeeze(0) < iou_threshold
            indices = indices[1:][mask]
        
        return np.array(keep)
    
    def _compute_iou(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU between box sets."""
        x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, np.newaxis] + area2 - intersection
        
        return intersection / (union + 1e-6)


class ImagePredictor(Predictor):
    """
    Image predictor with batch processing support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize image predictor.
        
        Args:
            model: Detection model
            class_names: List of class names
            **kwargs: Additional arguments for Predictor
        """
        super().__init__(model, **kwargs)
        self.class_names = class_names or [f"class_{i}" for i in range(80)]
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Predict on a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Dict[str, np.ndarray]]:
        """Process a single batch."""
        # Preprocess all images
        input_tensors = []
        preproc_infos = []
        
        for image in images:
            tensor, info = self.preprocess(image)
            input_tensors.append(tensor)
            preproc_infos.append(info)
        
        # Stack inputs
        input_batch = torch.cat(input_tensors, dim=0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_batch)
        
        # Postprocess each image
        results = []
        for i in range(len(images)):
            batch_outputs = [out[i:i+1] for out in outputs]
            result = self.postprocess(batch_outputs, preproc_infos[i])
            results.append(result)
        
        return results
    
    def predict_directory(
        self,
        directory: str,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Predict on all images in a directory.
        
        Args:
            directory: Directory path
            extensions: Image file extensions
            output_dir: Output directory for visualizations
            
        Returns:
            Dictionary mapping filenames to predictions
        """
        directory = Path(directory)
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
        
        results = {}
        for image_path in image_files:
            predictions = self.predict(str(image_path))
            results[image_path.name] = predictions
            
            if output_dir is not None:
                # Save visualization
                from .visualizer import draw_detections
                output_path = Path(output_dir) / image_path.name
                vis_image = draw_detections(
                    cv2.imread(str(image_path)),
                    predictions["boxes"],
                    predictions["scores"],
                    predictions["labels"],
                    self.class_names,
                )
                cv2.imwrite(str(output_path), vis_image)
        
        return results
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label."""
        return self.class_names[label] if label < len(self.class_names) else f"class_{label}"
