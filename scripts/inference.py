#!/usr/bin/env python
"""
Inference Script for Lightweight ViT Detection System.

This script runs object detection inference on images or videos.

Usage:
    python inference.py --model outputs/best_model.pth --config configs/model/mobilevit.yaml --input image.jpg
    python inference.py --model outputs/best_model.pth --config configs/model/mobilevit.yaml --input video.mp4
    python inference.py --model outputs/best_model.pth --config configs/model/mobilevit.yaml --input webcam
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.visualization import visualize_detections, COCO_CLASSES
from src.models.mobilevit import build_mobilevit
from src.deployment.inference_engine import PyTorchEngine, ONNXEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Object Detection Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model checkpoint (.pth) or ONNX model (.onnx)'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input source: image path, video path, directory, or "webcam"'
    )
    parser.add_argument(
        '--output', type=str, default='outputs/inference',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.3,
        help='Confidence score threshold'
    )
    parser.add_argument(
        '--backend', type=str, default='pytorch',
        choices=['pytorch', 'onnx'],
        help='Inference backend'
    )
    parser.add_argument(
        '--save-txt', action='store_true',
        help='Save detection results as text files'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Disable result display'
    )
    
    return parser.parse_args()


def load_model(args, config):
    """Load detection model."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.backend == 'pytorch' or args.model.endswith('.pth'):
        # Build PyTorch model
        model = build_mobilevit(config)
        
        # Load checkpoint
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        model.eval()
        
        engine = PyTorchEngine(model, device=str(device))
        
    elif args.backend == 'onnx' or args.model.endswith('.onnx'):
        # Load ONNX model
        engine = ONNXEngine(args.model)
        
    else:
        raise ValueError(f"Unsupported model format: {args.model}")
        
    return engine, device


def preprocess_image(image, config):
    """Preprocess image for inference."""
    input_cfg = config.get('model', {}).get('input', {})
    target_size = input_cfg.get('image_size', [640, 640])
    
    # Resize
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    # BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    mean = input_cfg.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    std = input_cfg.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
    
    image_norm = image_rgb.astype(np.float32) / 255.0
    image_norm = (image_norm - mean) / std
    
    # To tensor format [1, C, H, W]
    image_tensor = np.transpose(image_norm, (2, 0, 1))[np.newaxis, ...]
    
    return image_tensor.astype(np.float32), original_size


def postprocess_detections(outputs, original_size, config, threshold):
    """Postprocess detection outputs."""
    # Extract outputs (format depends on model)
    boxes = outputs.get('boxes', np.array([]))
    scores = outputs.get('scores', np.array([]))
    labels = outputs.get('labels', np.array([]))
    
    if len(boxes) == 0:
        return {'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])}
        
    # Filter by threshold
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Scale boxes to original size
    input_size = config.get('model', {}).get('input', {}).get('image_size', [640, 640])
    scale_y = original_size[0] / input_size[0]
    scale_x = original_size[1] / input_size[1]
    
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
    return {'boxes': boxes, 'scores': scores, 'labels': labels}


def save_results(detections, output_path, image_name):
    """Save detection results to text file."""
    txt_path = os.path.join(output_path, f'{image_name}.txt')
    
    with open(txt_path, 'w') as f:
        for box, score, label in zip(
            detections['boxes'], detections['scores'], detections['labels']
        ):
            f.write(f'{int(label)} {score:.4f} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n')


def run_image_inference(engine, image_path, config, args, output_dir):
    """Run inference on single image."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image: {image_path}")
        return
        
    # Preprocess
    image_tensor, original_size = preprocess_image(image, config)
    
    # Inference
    start_time = time.time()
    outputs = engine.predict(image_tensor)
    inference_time = time.time() - start_time
    
    # Postprocess
    detections = postprocess_detections(outputs, original_size, config, args.threshold)
    
    print(f"Image: {image_path}")
    print(f"  Detected: {len(detections['boxes'])} objects")
    print(f"  Inference time: {inference_time*1000:.1f}ms")
    
    # Visualize
    vis_image = visualize_detections(
        image, detections, COCO_CLASSES, args.threshold
    )
    
    # Save results
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f'{image_name}_result.jpg')
    cv2.imwrite(output_path, vis_image)
    print(f"  Saved: {output_path}")
    
    if args.save_txt:
        save_results(detections, output_dir, image_name)
        
    if not args.no_display and HAS_CV2:
        cv2.imshow('Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_inference(engine, video_path, config, args, output_dir):
    """Run inference on video."""
    if not HAS_CV2:
        print("Error: OpenCV required for video inference")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f'{video_name}_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    frame_idx = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        image_tensor, original_size = preprocess_image(frame, config)
        
        # Inference
        start_time = time.time()
        outputs = engine.predict(image_tensor)
        total_time += time.time() - start_time
        
        # Postprocess
        detections = postprocess_detections(outputs, original_size, config, args.threshold)
        
        # Visualize
        vis_frame = visualize_detections(
            frame, detections, COCO_CLASSES, args.threshold
        )
        
        # Add FPS display
        avg_fps = (frame_idx + 1) / total_time if total_time > 0 else 0
        cv2.putText(vis_frame, f'FPS: {avg_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        writer.write(vis_frame)
        
        if not args.no_display:
            cv2.imshow('Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")
            
    cap.release()
    writer.release()
    
    if not args.no_display:
        cv2.destroyAllWindows()
        
    print(f"  Output saved: {output_path}")
    print(f"  Average FPS: {frame_idx / total_time:.1f}")


def run_webcam_inference(engine, config, args):
    """Run inference on webcam stream."""
    if not HAS_CV2:
        print("Error: OpenCV required for webcam inference")
        return
        
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
        
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        image_tensor, original_size = preprocess_image(frame, config)
        
        # Inference
        start_time = time.time()
        outputs = engine.predict(image_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = postprocess_detections(outputs, original_size, config, args.threshold)
        
        # Visualize
        vis_frame = visualize_detections(
            frame, detections, COCO_CLASSES, args.threshold
        )
        
        # Add FPS display
        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Detection', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model}")
    engine, device = load_model(args, config)
    print(f"Using device: {device}")
    
    # Run inference
    if args.input.lower() == 'webcam':
        run_webcam_inference(engine, config, args)
        
    elif os.path.isdir(args.input):
        # Process directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in sorted(os.listdir(args.input)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(args.input, filename)
                run_image_inference(engine, image_path, config, args, args.output)
                
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_video_inference(engine, args.input, config, args, args.output)
        
    else:
        run_image_inference(engine, args.input, config, args, args.output)


if __name__ == '__main__':
    main()
