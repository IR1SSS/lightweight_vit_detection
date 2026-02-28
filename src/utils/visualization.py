"""
Visualization Utilities for Lightweight ViT Detection System.

This module provides visualization functions for detection results,
including bounding box drawing and image display.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# Try to import visualization libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Default color palette for detection visualization
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Light Blue
    (255, 0, 128),    # Pink
]

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


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR format for OpenCV).
        boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format.
        labels: Optional class labels [N].
        scores: Optional confidence scores [N].
        class_names: Optional list of class names.
        score_threshold: Minimum score to display.
        line_thickness: Box line thickness.
        font_scale: Font scale for labels.
        
    Returns:
        Image with drawn boxes.
    """
    if not HAS_CV2:
        print("Warning: OpenCV not available. Cannot draw boxes.")
        return image
        
    image = image.copy()
    
    if class_names is None:
        class_names = COCO_CLASSES
        
    for i, box in enumerate(boxes):
        # Filter by score
        if scores is not None and scores[i] < score_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        
        # Get color
        label_idx = labels[i] if labels is not None else 0
        color = COLORS[int(label_idx) % len(COLORS)]
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if labels is not None:
            label = class_names[int(label_idx)] if int(label_idx) < len(class_names) else str(int(label_idx))
            
            if scores is not None:
                label = f'{label}: {scores[i]:.2f}'
                
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color, -1
            )
            
            # Draw text
            cv2.putText(
                image, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1
            )
            
    return image


def visualize_detections(
    image: Union[np.ndarray, str],
    detections: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    output_path: Optional[str] = None,
    show: bool = False
) -> np.ndarray:
    """
    Visualize detection results on image.
    
    Args:
        image: Input image array or path to image file.
        detections: Detection results dictionary with 'boxes', 'labels', 'scores'.
        class_names: Optional list of class names.
        score_threshold: Minimum score to display.
        output_path: Optional path to save visualization.
        show: Whether to display image.
        
    Returns:
        Visualization image.
    """
    # Load image if path
    if isinstance(image, str):
        if HAS_CV2:
            image = cv2.imread(image)
        elif HAS_PIL:
            image = np.array(Image.open(image))
        else:
            raise ImportError("Neither OpenCV nor PIL available")
            
    # Extract detection components
    boxes = detections.get('boxes', np.array([]))
    labels = detections.get('labels', None)
    scores = detections.get('scores', None)
    
    # Convert tensors to numpy
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    if labels is not None and hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    if scores is not None and hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
        
    # Draw boxes
    vis_image = draw_boxes(
        image, boxes, labels, scores,
        class_names, score_threshold
    )
    
    # Save if requested
    if output_path is not None:
        if HAS_CV2:
            cv2.imwrite(output_path, vis_image)
        elif HAS_PIL:
            Image.fromarray(vis_image).save(output_path)
            
    # Display if requested
    if show and HAS_CV2:
        cv2.imshow('Detection Results', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return vis_image


def visualize_batch(
    images: np.ndarray,
    detections_list: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    output_dir: Optional[str] = None,
    prefix: str = 'detection'
) -> List[np.ndarray]:
    """
    Visualize batch of detection results.
    
    Args:
        images: Batch of images [B, H, W, C].
        detections_list: List of detection dictionaries.
        class_names: Optional list of class names.
        score_threshold: Minimum score to display.
        output_dir: Optional directory to save visualizations.
        prefix: Filename prefix for saved images.
        
    Returns:
        List of visualization images.
    """
    import os
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
    vis_images = []
    
    for i, (image, detections) in enumerate(zip(images, detections_list)):
        output_path = None
        if output_dir is not None:
            output_path = os.path.join(output_dir, f'{prefix}_{i:04d}.jpg')
            
        vis_image = visualize_detections(
            image, detections, class_names, score_threshold, output_path
        )
        vis_images.append(vis_image)
        
    return vis_images


def create_comparison_image(
    image: np.ndarray,
    ground_truth: Dict[str, Any],
    predictions: Dict[str, Any],
    class_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of ground truth and predictions.
    
    Args:
        image: Input image.
        ground_truth: Ground truth detection dictionary.
        predictions: Predicted detection dictionary.
        class_names: Optional list of class names.
        
    Returns:
        Comparison image.
    """
    # Draw ground truth
    gt_image = draw_boxes(
        image.copy(),
        ground_truth.get('boxes', np.array([])),
        ground_truth.get('labels', None),
        class_names=class_names
    )
    
    # Draw predictions
    pred_image = draw_boxes(
        image.copy(),
        predictions.get('boxes', np.array([])),
        predictions.get('labels', None),
        predictions.get('scores', None),
        class_names=class_names
    )
    
    # Add labels
    if HAS_CV2:
        cv2.putText(gt_image, 'Ground Truth', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred_image, 'Predictions', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                   
    # Concatenate horizontally
    comparison = np.concatenate([gt_image, pred_image], axis=1)
    
    return comparison


def plot_training_curves(
    metrics_history: List[Dict[str, float]],
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training curves.
    
    Args:
        metrics_history: List of metric dictionaries per epoch.
        output_path: Optional path to save plot.
        show: Whether to display plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Cannot plot curves.")
        return
        
    if not metrics_history:
        return
        
    # Extract metrics
    epochs = range(1, len(metrics_history) + 1)
    metrics_keys = list(metrics_history[0].keys())
    
    # Create subplots
    n_metrics = len(metrics_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
        
    for ax, key in zip(axes, metrics_keys):
        values = [m[key] for m in metrics_history]
        ax.plot(epochs, values, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(f'{key} over Training')
        ax.grid(True)
        
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()
