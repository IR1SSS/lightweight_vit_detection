"""
Data augmentation transforms for object detection.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Any]):
        """
        Initialize Compose.
        
        Args:
            transforms: List of transforms to compose
        """
        self.transforms = transforms
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply all transforms.
        
        Args:
            image: Input image
            target: Target dictionary
            
        Returns:
            Transformed image and target
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """Resize image and adjust bounding boxes."""
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """
        Initialize Resize.
        
        Args:
            size: Target size (int for square, tuple for (h, w))
            interpolation: Interpolation method
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Resize image and boxes."""
        h, w = image.shape[:2]
        new_h, new_w = self.size
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)
        
        # Scale boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].copy()
            scale_x = new_w / w
            scale_y = new_h / h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes
        
        # Update image info
        if "image_info" in target:
            target["image_info"]["height"] = new_h
            target["image_info"]["width"] = new_w
        
        return image, target


class Letterbox:
    """Letterbox resize with padding."""
    
    def __init__(
        self,
        target_size: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114),
    ):
        """
        Initialize Letterbox.
        
        Args:
            target_size: Target size
            color: Padding color
        """
        self.target_size = target_size
        self.color = color
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply letterbox resize."""
        h, w = image.shape[:2]
        
        # Calculate scale
        scale = min(self.target_size / h, self.target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=self.color
        )
        
        # Adjust boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top
            target["boxes"] = boxes
        
        # Store padding info
        target["letterbox_info"] = {
            "scale": scale,
            "pad_top": pad_top,
            "pad_left": pad_left,
        }
        
        return image, target


class RandomFlip:
    """Random horizontal or vertical flip."""
    
    def __init__(
        self,
        prob: float = 0.5,
        direction: str = "horizontal",
    ):
        """
        Initialize RandomFlip.
        
        Args:
            prob: Flip probability
            direction: "horizontal" or "vertical"
        """
        self.prob = prob
        self.direction = direction
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply random flip."""
        if random.random() < self.prob:
            if self.direction == "horizontal":
                image = cv2.flip(image, 1)
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"].copy()
                    w = image.shape[1]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes
            else:
                image = cv2.flip(image, 0)
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"].copy()
                    h = image.shape[0]
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                    target["boxes"] = boxes
        
        return image, target


class RandomHSV:
    """Random HSV augmentation."""
    
    def __init__(
        self,
        hgain: float = 0.015,
        sgain: float = 0.7,
        vgain: float = 0.4,
    ):
        """
        Initialize RandomHSV.
        
        Args:
            hgain: Hue gain
            sgain: Saturation gain
            vgain: Value gain
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply random HSV augmentation."""
        if self.hgain or self.sgain or self.vgain:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Apply random gains
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            h += random.uniform(-self.hgain, self.hgain) * 179
            s += random.uniform(-self.sgain, self.sgain) * 255
            v += random.uniform(-self.vgain, self.vgain) * 255
            
            hsv[:, :, 0] = np.clip(h, 0, 179)
            hsv[:, :, 1] = np.clip(s, 0, 255)
            hsv[:, :, 2] = np.clip(v, 0, 255)
            
            # Convert back to BGR
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image, target


class Normalize:
    """Normalize image with mean and std."""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize Normalize.
        
        Args:
            mean: Mean values for each channel
            std: Standard deviation values for each channel
        """
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Normalize image."""
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, target


class Mosaic:
    """Mosaic data augmentation (combines 4 images)."""
    
    def __init__(
        self,
        prob: float = 0.5,
        target_size: int = 640,
    ):
        """
        Initialize Mosaic.
        
        Args:
            prob: Probability of applying mosaic
            target_size: Output image size
        """
        self.prob = prob
        self.target_size = target_size
        self._dataset = None
    
    def set_dataset(self, dataset: Any):
        """Set reference to dataset for loading additional images."""
        self._dataset = dataset
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply mosaic augmentation."""
        if random.random() > self.prob or self._dataset is None:
            return image, target
        
        # Load 3 additional images
        indices = [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
        
        images = [image]
        targets = [target]
        
        for idx in indices:
            img, tgt = self._dataset._load_image(self._dataset.ids[idx])
            ann = self._dataset._load_annotations(self._dataset.ids[idx])
            tgt = {
                "boxes": ann["boxes"],
                "labels": ann["labels"],
            }
            images.append(img)
            targets.append(tgt)
        
        # Create mosaic
        mosaic_size = self.target_size * 2
        mosaic_image = np.full(
            (mosaic_size, mosaic_size, 3),
            114, dtype=np.uint8
        )
        
        # Place images in 4 corners
        boxes_list = []
        labels_list = []
        
        for i, (img, tgt) in enumerate(zip(images, targets)):
            h, w = img.shape[:2]
            
            # Resize to target size
            scale = self.target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            
            # Place in mosaic
            if i == 0:  # Top left
                x1a, y1a = 0, 0
                x2a, y2a = new_w, new_h
                x1b, y1b = 0, 0
                x2b, y2b = new_w, new_h
            elif i == 1:  # Top right
                x1a, y1a = self.target_size, 0
                x2a, y2a = self.target_size + new_w, new_h
                x1b, y1b = 0, 0
                x2b, y2b = new_w, new_h
            elif i == 2:  # Bottom left
                x1a, y1a = 0, self.target_size
                x2a, y2a = new_w, self.target_size + new_h
                x1b, y1b = 0, 0
                x2b, y2b = new_w, new_h
            else:  # Bottom right
                x1a, y1a = self.target_size, self.target_size
                x2a, y2a = self.target_size + new_w, self.target_size + new_h
                x1b, y1b = 0, 0
                x2b, y2b = new_w, new_h
            
            mosaic_image[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust boxes
            if len(tgt["boxes"]) > 0:
                boxes = tgt["boxes"].copy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + x1a - x1b
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + y1a - y1b
                
                # Clip to mosaic bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, mosaic_size)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, mosaic_size)
                
                boxes_list.append(boxes)
                labels_list.append(tgt["labels"])
        
        # Combine all annotations
        if boxes_list:
            target = {
                "boxes": np.concatenate(boxes_list, axis=0),
                "labels": np.concatenate(labels_list, axis=0),
            }
        else:
            target = {"boxes": np.array([]), "labels": np.array([])}
        
        return mosaic_image, target


class MixUp:
    """MixUp data augmentation."""
    
    def __init__(
        self,
        prob: float = 0.1,
        alpha: float = 32.0,
        beta: float = 32.0,
    ):
        """
        Initialize MixUp.
        
        Args:
            prob: Probability of applying mixup
            alpha: Alpha parameter for beta distribution
            beta: Beta parameter for beta distribution
        """
        self.prob = prob
        self.alpha = alpha
        self.beta = beta
        self._dataset = None
    
    def set_dataset(self, dataset: Any):
        """Set reference to dataset for loading additional images."""
        self._dataset = dataset
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply mixup augmentation."""
        if random.random() > self.prob or self._dataset is None:
            return image, target
        
        # Load another random image
        idx = random.randint(0, len(self._dataset) - 1)
        img2, tgt2 = self._dataset._load_image(self._dataset.ids[idx])
        ann2 = self._dataset._load_annotations(self._dataset.ids[idx])
        tgt2 = {
            "boxes": ann2["boxes"],
            "labels": ann2["labels"],
        }
        
        # Resize to same size
        h, w = image.shape[:2]
        img2 = cv2.resize(img2, (w, h))
        
        # Mix images
        lam = np.random.beta(self.alpha, self.beta)
        image = (image * lam + img2 * (1 - lam)).astype(np.uint8)
        
        # Combine annotations
        if len(target["boxes"]) > 0 and len(tgt2["boxes"]) > 0:
            boxes = np.concatenate([target["boxes"], tgt2["boxes"]], axis=0)
            labels = np.concatenate([target["labels"], tgt2["labels"]], axis=0)
        elif len(target["boxes"]) > 0:
            boxes = target["boxes"]
            labels = target["labels"]
        elif len(tgt2["boxes"]) > 0:
            boxes = tgt2["boxes"]
            labels = tgt2["labels"]
        else:
            boxes = np.array([])
            labels = np.array([])
        
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        
        return image, target


class RandomAffine:
    """Random affine transformation."""
    
    def __init__(
        self,
        prob: float = 0.5,
        degrees: float = 10.0,
        translate: float = 0.1,
        scale: Tuple[float, float] = (0.5, 1.5),
        shear: float = 0.0,
    ):
        """
        Initialize RandomAffine.
        
        Args:
            prob: Probability of applying transform
            degrees: Rotation degrees
            translate: Translation ratio
            scale: Scale range
            shear: Shear degrees
        """
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
    
    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply random affine transform."""
        if random.random() > self.prob:
            return image, target
        
        h, w = image.shape[:2]
        
        # Generate random parameters
        angle = random.uniform(-self.degrees, self.degrees)
        translate_x = random.uniform(-self.translate, self.translate) * w
        translate_y = random.uniform(-self.translate, self.translate) * h
        scale = random.uniform(*self.scale)
        shear = random.uniform(-self.shear, self.shear)
        
        # Compute transformation matrix
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += translate_x
        M[1, 2] += translate_y
        
        # Apply transformation to image
        image = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        
        # Transform boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].copy()
            
            # Convert boxes to corners
            corners = np.zeros((len(boxes), 4, 2))
            corners[:, 0, 0] = boxes[:, 0]  # x1
            corners[:, 0, 1] = boxes[:, 1]  # y1
            corners[:, 1, 0] = boxes[:, 2]  # x2
            corners[:, 1, 1] = boxes[:, 1]  # y1
            corners[:, 2, 0] = boxes[:, 2]  # x2
            corners[:, 2, 1] = boxes[:, 3]  # y2
            corners[:, 3, 0] = boxes[:, 0]  # x1
            corners[:, 3, 1] = boxes[:, 3]  # y2
            
            # Apply transformation
            corners = corners.reshape(-1, 2)
            corners = np.dot(corners, M[:, :2].T) + M[:, 2]
            corners = corners.reshape(-1, 4, 2)
            
            # Get new boxes
            new_boxes = np.zeros_like(boxes)
            new_boxes[:, 0] = corners[:, :, 0].min(axis=1)
            new_boxes[:, 1] = corners[:, :, 1].min(axis=1)
            new_boxes[:, 2] = corners[:, :, 0].max(axis=1)
            new_boxes[:, 3] = corners[:, :, 1].max(axis=1)
            
            # Clip to image bounds
            new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, w)
            new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, h)
            
            # Filter invalid boxes
            valid = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
            target["boxes"] = new_boxes[valid]
            target["labels"] = target["labels"][valid]
        
        return image, target
