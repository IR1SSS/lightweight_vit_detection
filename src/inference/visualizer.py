"""
Visualization utilities for detection results.
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# COCO class colors
COCO_COLORS = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
]


class Visualizer:
    """
    Visualizer for detection results.
    """
    
    def __init__(
        self,
        class_names: List[str],
        colors: Optional[List[Tuple[int, int, int]]] = None,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        """
        Initialize visualizer.
        
        Args:
            class_names: List of class names
            colors: List of BGR colors for each class
            line_thickness: Box line thickness
            font_scale: Font scale for labels
        """
        self.class_names = class_names
        self.colors = colors or self._generate_colors(len(class_names))
        self.line_thickness = line_thickness
        self.font_scale = font_scale
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate random colors for classes."""
        random.seed(42)  # For reproducibility
        colors = []
        for _ in range(num_classes):
            colors.append((
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ))
        return colors
    
    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            boxes: Bounding boxes (N, 4) in xyxy format
            labels: Class labels
            scores: Confidence scores
            
        Returns:
            Image with drawn boxes
        """
        image = image.copy()
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box)
            
            # Get color
            color = self.colors[int(label) % len(self.colors)]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Draw label
            class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f"class_{label}"
            
            if scores is not None:
                label_text = f"{class_name}: {scores[i]:.2f}"
            else:
                label_text = class_name
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1,
            )
            
            # Draw text
            cv2.putText(
                image,
                label_text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        
        return image
    
    def draw_masks(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Draw segmentation masks.
        
        Args:
            image: Input image
            boxes: Bounding boxes
            masks: Masks (N, H, W)
            alpha: Transparency
            
        Returns:
            Image with masks
        """
        image = image.copy()
        
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            color = self.colors[i % len(self.colors)]
            
            # Create colored mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            
            # Blend with image
            mask_area = mask > 0.5
            if mask_area.any():
                image[mask_area] = (
                    image[mask_area] * (1 - alpha) +
                    colored_mask[mask_area] * alpha
                ).astype(np.uint8)
        
        return image
    
    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Draw keypoints for pose estimation.
        
        Args:
            image: Input image
            keypoints: Keypoints (N, 2) or (N, 3)
            scores: Keypoint scores
            threshold: Score threshold
            
        Returns:
            Image with keypoints
        """
        image = image.copy()
        
        for i, kp in enumerate(keypoints):
            if scores is not None and scores[i] < threshold:
                continue
            
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        return image


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw detection results on image.
    
    Args:
        image: Input image (BGR)
        boxes: Bounding boxes (N, 4) in xyxy format
        scores: Confidence scores
        labels: Class labels
        class_names: List of class names
        colors: List of BGR colors
        line_thickness: Box line thickness
        font_scale: Font scale
        show_confidence: Show confidence scores
        
    Returns:
        Image with detections drawn
    """
    image = image.copy()
    
    # Default class names
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]
    
    # Default colors
    if colors is None:
        colors = COCO_COLORS
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        
        # Get color
        color = colors[int(label) % len(colors)]
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Prepare label
        class_name = class_names[int(label)] if int(label) < len(class_names) else f"class_{label}"
        
        if show_confidence:
            label_text = f"{class_name}: {score:.2f}"
        else:
            label_text = class_name
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Draw label background
        y1_text = max(y1, text_height + baseline + 5)
        cv2.rectangle(
            image,
            (x1, y1_text - text_height - baseline - 5),
            (x1 + text_width + 5, y1_text),
            color,
            -1,
        )
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (x1 + 2, y1_text - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    return image


def draw_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Before",
    title2: str = "After",
) -> np.ndarray:
    """
    Draw side-by-side comparison.
    
    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
        
    Returns:
        Comparison image
    """
    # Resize to same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    target_height = max(h1, h2)
    
    if h1 != target_height:
        scale = target_height / h1
        image1 = cv2.resize(image1, (int(w1 * scale), target_height))
    
    if h2 != target_height:
        scale = target_height / h2
        image2 = cv2.resize(image2, (int(w2 * scale), target_height))
    
    # Concatenate
    comparison = np.hstack([image1, image2])
    
    # Add titles
    cv2.putText(comparison, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, title2, (image1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return comparison


def save_detection_results(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None,
):
    """
    Save detection results as image.
    
    Args:
        image: Input image
        boxes: Bounding boxes
        scores: Confidence scores
        labels: Class labels
        output_path: Output file path
        class_names: List of class names
    """
    vis_image = draw_detections(image, boxes, scores, labels, class_names)
    cv2.imwrite(output_path, vis_image)
