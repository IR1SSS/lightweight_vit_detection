"""
Detection metrics for object detection evaluation.
Implements COCO-style mAP computation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def compute_iou(
    box1: np.ndarray,
    box2: np.ndarray,
) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: First set of boxes (N, 4) in [x1, y1, x2, y2] format
        box2: Second set of boxes (M, 4) in [x1, y1, x2, y2] format
        
    Returns:
        IoU matrix (N, M)
    """
    # Ensure 2D arrays
    box1 = np.atleast_2d(box1)
    box2 = np.atleast_2d(box2)
    
    # Compute intersection
    x1 = np.maximum(box1[:, 0:1], box2[:, 0].T)
    y1 = np.maximum(box1[:, 1:2], box2[:, 1].T)
    x2 = np.minimum(box1[:, 2:3], box2[:, 2].T)
    y2 = np.minimum(box1[:, 3:4], box2[:, 3].T)
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, np.newaxis] + area2 - intersection
    
    return intersection / (union + 1e-7)


def compute_ap(
    recall: np.ndarray,
    precision: np.ndarray,
    use_07_metric: bool = False,
) -> float:
    """
    Compute Average Precision.
    
    Args:
        recall: Recall values
        precision: Precision values
        use_07_metric: Use VOC2007 11-point metric
        
    Returns:
        Average Precision value
    """
    if use_07_metric:
        # VOC2007 11-point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
    else:
        # COCO-style AP (all-point interpolation)
        # Prepend (0, 1) and append (1, 0)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        
        # Ensure precision is monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        
        # Find points where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Sum areas under the curve
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return float(ap)


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 80,
) -> Dict[str, float]:
    """
    Compute mean Average Precision.
    
    Args:
        predictions: List of predictions, each with 'boxes', 'scores', 'labels'
        targets: List of targets, each with 'boxes', 'labels'
        iou_threshold: IoU threshold for TP/FP determination
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    # Initialize per-class AP storage
    aps = {}
    
    for class_id in range(num_classes):
        # Collect predictions and targets for this class
        class_predictions = []
        class_targets = []
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Get predictions for this class
            mask = pred["labels"] == class_id
            boxes = pred["boxes"][mask] if len(mask) > 0 else np.array([])
            scores = pred["scores"][mask] if len(mask) > 0 else np.array([])
            
            for box, score in zip(boxes, scores):
                class_predictions.append({
                    "img_idx": img_idx,
                    "box": box,
                    "score": score,
                })
            
            # Get targets for this class
            t_mask = target["labels"] == class_id
            t_boxes = target["boxes"][t_mask] if len(t_mask) > 0 else np.array([])
            
            for box in t_boxes:
                class_targets.append({
                    "img_idx": img_idx,
                    "box": box,
                    "matched": False,
                })
        
        # Skip if no targets or predictions
        if len(class_targets) == 0:
            continue
        
        if len(class_predictions) == 0:
            aps[class_id] = 0.0
            continue
        
        # Sort predictions by score
        class_predictions.sort(key=lambda x: x["score"], reverse=True)
        
        # Compute TP/FP
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        
        for pred_idx, pred in enumerate(class_predictions):
            img_idx = pred["img_idx"]
            pred_box = pred["box"]
            
            # Find unmatched targets in the same image
            img_targets = [
                (t_idx, t) for t_idx, t in enumerate(class_targets)
                if t["img_idx"] == img_idx and not t["matched"]
            ]
            
            if len(img_targets) == 0:
                fp[pred_idx] = 1
                continue
            
            # Compute IoU with all unmatched targets
            target_boxes = np.array([t["box"] for _, t in img_targets])
            ious = compute_iou(np.array([pred_box]), target_boxes)[0]
            
            # Find best match
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]
            
            if best_iou >= iou_threshold:
                tp[pred_idx] = 1
                class_targets[img_targets[best_idx][0]]["matched"] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        recall = tp_cumsum / len(class_targets)
        
        # Compute AP
        aps[class_id] = compute_ap(recall, precision)
    
    # Compute mAP
    if len(aps) > 0:
        mAP = np.mean(list(aps.values()))
    else:
        mAP = 0.0
    
    return {
        "mAP": mAP,
        "APs": aps,
    }


class DetectionMetrics:
    """
    Comprehensive detection metrics tracker.
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        iou_thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes: Number of detection classes
            iou_thresholds: IoU thresholds for mAP computation
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ):
        """
        Add batch predictions and targets.
        
        Args:
            predictions: Dictionary with 'boxes', 'scores', 'labels'
            targets: Dictionary with 'boxes', 'labels'
        """
        self.predictions.append(predictions)
        self.targets.append(targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with computed metrics
        """
        metrics = {}
        
        # mAP@0.5
        map_50 = compute_map(self.predictions, self.targets, iou_threshold=0.5, num_classes=self.num_classes)
        metrics["mAP@0.5"] = map_50["mAP"]
        
        # mAP@0.5:0.95 (COCO-style)
        aps = []
        for iou_thresh in self.iou_thresholds:
            result = compute_map(
                self.predictions, self.targets,
                iou_threshold=iou_thresh,
                num_classes=self.num_classes
            )
            aps.append(result["mAP"])
        metrics["mAP@0.5:0.95"] = np.mean(aps)
        
        # Per-class AP at IoU=0.5
        for class_id, ap in map_50["APs"].items():
            metrics[f"AP_class_{class_id}"] = ap
        
        return metrics
    
    def reset(self):
        """Reset collected predictions and targets."""
        self.predictions = []
        self.targets = []


def non_max_suppression(
    predictions: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 100,
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression.
    
    Args:
        predictions: Predicted boxes (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score threshold
        max_detections: Maximum number of detections
        
    Returns:
        Indices of kept predictions
    """
    # Filter by score threshold
    mask = scores >= score_threshold
    predictions = predictions[mask]
    scores = scores[mask]
    
    if len(scores) == 0:
        return np.array([], dtype=np.int64)
    
    # Sort by score
    order = np.argsort(scores)[::-1]
    
    keep = []
    while len(order) > 0 and len(keep) < max_detections:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = compute_iou(predictions[i:i+1], predictions[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        order = order[1:][mask]
    
    return np.array(keep)
