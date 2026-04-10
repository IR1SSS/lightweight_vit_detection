"""
COCO Dataset implementation for object detection.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO_API = True
except ImportError:
    HAS_COCO_API = False


class COCODataset:
    """
    COCO Dataset for object detection.
    
    Supports loading images and annotations from COCO format datasets.
    """
    
    def __init__(
        self,
        root: str,
        annotation_file: str,
        transforms: Optional[Any] = None,
        class_names: Optional[List[str]] = None,
        remove_empty: bool = False,
    ):
        """
        Initialize COCO dataset.
        
        Args:
            root: Root directory of images
            annotation_file: Path to annotation JSON file
            transforms: Data transformations
            class_names: List of class names (optional)
            remove_empty: Remove images without annotations
        """
        self.root = Path(root).resolve()  # Always use absolute path
        self.annotation_file = str(Path(annotation_file).resolve())  # Absolute path
        self.transforms = transforms
        self.remove_empty = remove_empty
        
        # Load annotations
        if HAS_COCO_API:
            self.coco = COCO(annotation_file)
            self.ids = list(self.coco.imgs.keys())
        else:
            self._load_annotations_manually()
        
        # Filter empty images
        if remove_empty:
            self._filter_empty_images()
        
        # Class info
        self._load_class_info(class_names)
        
    def _load_annotations_manually(self):
        """Load annotations without pycocotools."""
        with open(self.annotation_file, "r") as f:
            data = json.load(f)
        
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = {}
        
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.ids = list(self.images.keys())
        
        # Create category mapping
        self.categories = {cat["id"]: cat for cat in data["categories"]}
    
    def _filter_empty_images(self):
        """Remove images without annotations."""
        if HAS_COCO_API:
            valid_ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if len(ann_ids) > 0:
                    valid_ids.append(img_id)
            self.ids = valid_ids
        else:
            self.ids = [img_id for img_id in self.ids if img_id in self.annotations]
    
    def _load_class_info(self, class_names: Optional[List[str]] = None):
        """Load class information."""
        if HAS_COCO_API:
            self.cat_ids = sorted(self.coco.getCatIds())
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            if class_names is None:
                cats = self.coco.loadCats(self.cat_ids)
                self.class_names = [cat["name"] for cat in cats]
            else:
                self.class_names = class_names
        else:
            self.cat_ids = sorted(self.categories.keys())
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            if class_names is None:
                self.class_names = [self.categories[cat_id]["name"] for cat_id in self.cat_ids]
            else:
                self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict]:
        """
        Get an item.
        
        Args:
            index: Index of the item
            
        Returns:
            Tuple of (image, target)
        """
        img_id = self.ids[index]
        
        # Load image
        image, image_info = self._load_image(img_id)
        
        # Load annotations
        annotations = self._load_annotations(img_id)
        
        # Prepare target
        target = {
            "image_id": img_id,
            "boxes": annotations["boxes"],
            "labels": annotations["labels"],
            "area": annotations["area"],
            "iscrowd": annotations["iscrowd"],
            "image_info": image_info,
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def _load_image(self, img_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Load image by ID.
        
        Args:
            img_id: Image ID
            
        Returns:
            Tuple of (image array, image info dict)
        """
        if HAS_COCO_API:
            img_info = self.coco.loadImgs(img_id)[0]
        else:
            img_info = self.images[img_id]
        
        # Load image
        img_path = self.root / img_info["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Image info
        info = {
            "height": img_info["height"],
            "width": img_info["width"],
            "file_name": img_info["file_name"],
        }
        
        return image, info
    
    def _load_annotations(self, img_id: int) -> Dict:
        """
        Load annotations for an image.
        
        Args:
            img_id: Image ID
            
        Returns:
            Dictionary with boxes, labels, etc.
        """
        if HAS_COCO_API:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
        else:
            anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                continue
            
            # Get bounding box (COCO format: [x, y, w, h])
            bbox = ann["bbox"]
            
            # Convert to [x1, y1, x2, y2] format
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            # Filter invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))
        
        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "area": np.array(areas, dtype=np.float32),
            "iscrowd": np.array(iscrowd, dtype=np.int64),
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names
    
    def evaluate(
        self,
        predictions: List[Dict],
        iou_type: str = "bbox",
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predictions
            iou_type: IoU type for evaluation
            
        Returns:
            Dictionary of metrics
        """
        if not HAS_COCO_API:
            raise RuntimeError("pycocotools required for evaluation")
        
        # Convert predictions to COCO format
        coco_predictions = []
        for pred in predictions:
            image_id = pred["image_id"]
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                w = x2 - x1
                h = y2 - y1
                
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": self.cat_ids[labels[i]],
                    "bbox": [x1, y1, w, h],
                    "score": float(scores[i]),
                })
        
        # Create COCO predictions object
        coco_pred = self.coco.loadRes(coco_predictions)
        
        # Evaluate
        coco_eval = COCOeval(self.coco, coco_pred, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
        }
        
        return metrics


class COCOCategories:
    """COCO 80 class names."""
    
    NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]
    
    COLORS = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 184), (255, 255, 0),
        (2, 33, 105), (95, 54, 80), (252, 27, 6), (192, 101, 246), (124, 49, 76),
        (157, 75, 129), (160, 194, 56), (230, 69, 0), (194, 150, 102), (83, 123, 101),
        (255, 110, 71), (114, 56, 113), (198, 255, 0), (0, 167, 104), (98, 149, 54),
        (163, 75, 44), (93, 102, 18), (255, 163, 183), (148, 78, 189), (0, 167, 143),
        (235, 67, 52), (212, 103, 22), (242, 154, 0), (255, 180, 172), (193, 255, 0),
        (255, 225, 210), (122, 150, 198), (138, 87, 126), (222, 188, 255), (137, 151, 105),
        (255, 106, 141), (138, 255, 245), (0, 56, 113), (255, 105, 238), (186, 0, 247),
        (255, 225, 77), (75, 180, 170), (150, 255, 130), (234, 234, 0), (0, 118, 122),
        (128, 222, 164), (255, 165, 150), (240, 240, 240), (150, 150, 255),
    ]
