#!/usr/bin/env python
"""
Demo script for the Lightweight ViT Detection System.

Usage:
    python tools/demo.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --source image.jpg
    python tools/demo.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --source video.mp4
    python tools/demo.py --config configs/model/mobilevit_small.yaml --weights outputs/best_model.pth --source 0  # webcam
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.models.detector import build_detector
from src.inference.predictor import ImagePredictor
from src.inference.video_detector import VideoDetector
from src.inference.visualizer import draw_detections
from src.data.coco_dataset import COCOCategories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run detection demo")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source (image path, video path, or camera ID)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Input image size",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results in window",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output",
    )
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name="demo")
    logger.info("Starting detection demo...")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    config = load_config(args.config)
    model = build_detector(config)
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded weights from: {args.weights}")
    
    # Class names
    class_names = COCOCategories.NAMES
    
    # Determine source type
    source = args.source
    
    if source.isdigit():
        # Camera
        source = int(source)
        mode = "camera"
    elif Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        # Image
        mode = "image"
    elif Path(source).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        # Video
        mode = "video"
    else:
        logger.error(f"Unknown source type: {source}")
        return
    
    logger.info(f"Mode: {mode}")
    
    # Process based on mode
    if mode == "image":
        process_image(args, model, class_names, device, logger)
    elif mode == "video":
        process_video(args, model, class_names, device, logger)
    elif mode == "camera":
        process_camera(args, model, class_names, device, logger)


def process_image(args, model, class_names, device, logger):
    """Process single image."""
    # Load image
    image = cv2.imread(args.source)
    if image is None:
        logger.error(f"Could not load image: {args.source}")
        return
    
    logger.info(f"Processing image: {args.source}")
    
    # Create predictor
    predictor = ImagePredictor(
        model=model,
        class_names=class_names,
        device=str(device),
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
    )
    
    # Predict
    results = predictor.predict(image)
    
    # Draw results
    vis_image = draw_detections(
        image,
        results["boxes"],
        results["scores"],
        results["labels"],
        class_names,
    )
    
    # Print detections
    logger.info(f"Detected {len(results['boxes'])} objects:")
    for i, (box, score, label) in enumerate(zip(
        results["boxes"], results["scores"], results["labels"]
    )):
        class_name = class_names[label]
        logger.info(f"  {i+1}. {class_name}: {score:.2f} at {box}")
    
    # Save or show
    if args.output:
        cv2.imwrite(args.output, vis_image)
        logger.info(f"Result saved to: {args.output}")
    elif not args.no_save:
        output_path = Path(args.source).stem + "_detected.jpg"
        cv2.imwrite(output_path, vis_image)
        logger.info(f"Result saved to: {output_path}")
    
    if args.show:
        cv2.imshow("Detection Result", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(args, model, class_names, device, logger):
    """Process video file."""
    logger.info(f"Processing video: {args.source}")
    
    # Create video detector
    detector = VideoDetector(
        model=model,
        device=str(device),
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
    )
    
    # Determine output path
    output_path = args.output
    if output_path is None and not args.no_save:
        output_path = Path(args.source).stem + "_detected.mp4"
    
    # Process video
    stats = detector.process_video(
        args.source,
        output_path=output_path,
        show=args.show,
    )
    
    logger.info(f"Processed {stats['total_frames']} frames")
    logger.info(f"Average FPS: {stats['fps']:.1f}")


def process_camera(args, model, class_names, device, logger):
    """Process camera stream."""
    logger.info(f"Processing camera: {args.source}")
    
    # Create video detector
    detector = VideoDetector(
        model=model,
        device=str(device),
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
    )
    
    # Determine output path
    output_path = args.output
    if output_path is None and not args.no_save:
        output_path = "camera_detected.mp4"
    
    # Process camera
    stats = detector.process_webcam(
        camera_id=args.source,
        output_path=output_path if not args.no_save else None,
        show=True,  # Always show for camera
    )
    
    logger.info(f"Processed {stats['total_frames']} frames")
    logger.info(f"Average FPS: {stats['fps']:.1f}")


if __name__ == "__main__":
    main()
