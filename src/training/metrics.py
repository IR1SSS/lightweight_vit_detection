"""
Metrics Computation for Lightweight ViT Detection System.

This module provides evaluation metrics for object detection,
including COCO-style mAP computation.
"""

import json
import tempfile
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch


class COCOEvaluator:
    """
    COCO-style evaluator for object detection.
    
    Computes mAP at various IoU thresholds following
    COCO evaluation protocol.
    """
    
    def __init__(
        self,
        ann_file: Optional[str] = None,
        iou_thresholds: Optional[List[float]] = None
    ) -> None:
        """
        Initialize COCO evaluator.
        
        Args:
            ann_file: Path to COCO annotation file for evaluation.
            iou_thresholds: IoU thresholds for mAP computation.
                           Defaults to COCO thresholds [0.5:0.05:0.95].
        """
        self.ann_file = ann_file
        self.iou_thresholds = iou_thresholds or [
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
        ]
        
        # Try to load pycocotools
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            self.COCO = COCO
            self.COCOeval = COCOeval
            self.use_pycocotools = True
        except ImportError:
            self.use_pycocotools = False
            
    def __call__(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: List of prediction dictionaries.
            targets: List of target dictionaries.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.use_pycocotools and self.ann_file:
            return self._evaluate_coco(predictions, targets)
        else:
            return self._evaluate_simple(predictions, targets)
            
    def _evaluate_coco(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Evaluate using pycocotools."""
        # Load COCO ground truth
        coco_gt = self.COCO(self.ann_file)
        
        # Convert predictions to COCO format
        coco_results = []
        
        for pred, target in zip(predictions, targets):
            image_id = target['image_id'].item()
            
            if 'boxes' in pred and len(pred['boxes']) > 0:
                boxes = pred['boxes']
                scores = pred.get('scores', torch.ones(len(boxes)))
                labels = pred.get('labels', torch.zeros(len(boxes), dtype=torch.int64))
                
                # Convert to COCO format [x, y, width, height]
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': label.item() + 1,  # COCO categories are 1-indexed
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': score.item()
                    })
                    
        if not coco_results:
            return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}
            
        # Save results to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_results, f)
            results_file = f.name
            
        # Run COCO evaluation
        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = self.COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            'mAP': coco_eval.stats[0],      # mAP@[0.5:0.95]
            'mAP50': coco_eval.stats[1],    # mAP@0.5
            'mAP75': coco_eval.stats[2],    # mAP@0.75
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5]
        }
        
    def _evaluate_simple(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Simple mAP evaluation without pycocotools."""
        all_ap = []
        
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap(predictions, targets, iou_thresh)
            all_ap.append(ap)
            
        return {
            'mAP': np.mean(all_ap),
            'mAP50': self._compute_ap(predictions, targets, 0.5),
            'mAP75': self._compute_ap(predictions, targets, 0.75)
        }
        
    def _compute_ap(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        iou_threshold: float
    ) -> float:
        """
        Compute Average Precision at a single IoU threshold.
        
        Args:
            predictions: List of prediction dictionaries.
            targets: List of target dictionaries.
            iou_threshold: IoU threshold for matching.
            
        Returns:
            Average precision value.
        """
        # Collect all detections and ground truths
        all_scores = []
        all_matches = []
        total_gt = 0
        
        for pred, target in zip(predictions, targets):
            gt_boxes = target.get('boxes', torch.zeros(0, 4))
            gt_labels = target.get('labels', torch.zeros(0, dtype=torch.int64))
            
            pred_boxes = pred.get('boxes', torch.zeros(0, 4))
            pred_scores = pred.get('scores', torch.zeros(0))
            pred_labels = pred.get('labels', torch.zeros(0, dtype=torch.int64))
            
            total_gt += len(gt_boxes)
            
            if len(pred_boxes) == 0:
                continue
                
            # Match predictions to ground truth
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            
            # Sort by score
            sort_idx = torch.argsort(pred_scores, descending=True)
            
            for idx in sort_idx:
                if len(gt_boxes) == 0:
                    all_scores.append(pred_scores[idx].item())
                    all_matches.append(0)
                    continue
                    
                pred_box = pred_boxes[idx]
                pred_label = pred_labels[idx]
                
                # Compute IoU with all ground truth boxes
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
                
                # Find best match with same label
                best_iou = 0.0
                best_idx = -1
                
                for gt_idx, (iou, gt_label) in enumerate(zip(ious, gt_labels)):
                    if not matched[gt_idx] and gt_label == pred_label and iou > best_iou:
                        best_iou = iou.item()
                        best_idx = gt_idx
                        
                all_scores.append(pred_scores[idx].item())
                
                if best_iou >= iou_threshold:
                    matched[best_idx] = True
                    all_matches.append(1)
                else:
                    all_matches.append(0)
                    
        if total_gt == 0:
            return 0.0
            
        # Sort by score and compute precision-recall
        sort_idx = np.argsort(all_scores)[::-1]
        all_matches = np.array(all_matches)[sort_idx]
        
        tp = np.cumsum(all_matches)
        fp = np.cumsum(1 - all_matches)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / total_gt
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            prec_at_recall = precision[recall >= r]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall)
        ap /= 11
        
        return ap


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Boxes of shape [N, 4] in (x1, y1, x2, y2) format.
        boxes2: Boxes of shape [M, 4] in (x1, y1, x2, y2) format.
        
    Returns:
        IoU matrix of shape [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Compute IoU
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-8)
    
    return iou


def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mean Average Precision.
    
    Args:
        predictions: List of prediction dictionaries.
        targets: List of target dictionaries.
        iou_threshold: IoU threshold for matching.
        
    Returns:
        mAP value.
    """
    evaluator = COCOEvaluator()
    metrics = evaluator._compute_ap(predictions, targets, iou_threshold)
    return metrics


class DetectionMetrics:
    """
    Collection of detection evaluation metrics.
    """
    
    @staticmethod
    def precision_recall(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[float, float]:
        """
        Compute precision and recall.
        
        Args:
            predictions: Predicted scores.
            targets: Ground truth binary labels.
            threshold: Score threshold for positive prediction.
            
        Returns:
            Tuple of (precision, recall).
        """
        pred_binary = predictions >= threshold
        
        tp = (pred_binary & (targets == 1)).sum().item()
        fp = (pred_binary & (targets == 0)).sum().item()
        fn = (~pred_binary & (targets == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return precision, recall
        
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall."""
        return 2 * precision * recall / (precision + recall + 1e-8)
