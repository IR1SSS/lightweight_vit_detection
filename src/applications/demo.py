"""
Demo Application for Lightweight ViT Detection System.

This module provides a simple demonstration of the detection system,
including image and video inference capabilities.
"""

import os
import time
import argparse
from typing import Optional, Union, List, Dict, Any

import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..utils.config_loader import load_config
from ..utils.visualization import visualize_detections, COCO_CLASSES
from ..deployment.inference_engine import PyTorchEngine


class DetectionDemo:
    """
    Demo class for running object detection.
    
    Supports image and video inference with visualization.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        score_threshold: float = 0.3
    ) -> None:
        """
        Initialize detection demo.
        
        Args:
            model_path: Path to model checkpoint.
            config_path: Path to model configuration.
            device: Device to run inference on.
            score_threshold: Minimum confidence score.
        """
        self.device = device
        self.score_threshold = score_threshold
        
        # Load configuration
        self.config = {}
        if config_path is not None:
            self.config = load_config(config_path)
            
        # Load model
        self.model = None
        self.engine = None
        
        if model_path is not None:
            self.load_model(model_path)
            
    def load_model(self, model_path: str) -> None:
        """
        Load detection model.
        
        Args:
            model_path: Path to model checkpoint.
        """
        from ..models.mobilevit import build_mobilevit
        
        # Build model from config
        self.model = build_mobilevit(self.config)
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create inference engine
        self.engine = PyTorchEngine(self.model, device=self.device)
        
        print(f"Model loaded from: {model_path}")
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR format).
            
        Returns:
            Preprocessed tensor.
        """
        # Get target size from config
        input_cfg = self.config.get('model', {}).get('input', {})
        target_size = input_cfg.get('image_size', [640, 640])
        
        # Resize
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        mean = input_cfg.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
        std = input_cfg.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # To tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.float()
        
    def postprocess(
        self,
        outputs: Dict[str, Any],
        orig_shape: tuple
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs.
            orig_shape: Original image shape (H, W).
            
        Returns:
            Processed detection results.
        """
        # This is a simplified postprocess
        # Actual implementation depends on model output format
        
        boxes = outputs.get('boxes', np.array([]))
        scores = outputs.get('scores', np.array([]))
        labels = outputs.get('labels', np.array([]))
        
        # Filter by score
        if len(scores) > 0:
            keep = scores > self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
        # Scale boxes to original image size
        input_size = self.config.get('model', {}).get('input', {}).get('image_size', [640, 640])
        
        if len(boxes) > 0:
            scale_x = orig_shape[1] / input_size[1]
            scale_y = orig_shape[0] / input_size[0]
            
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
        
    @torch.no_grad()
    def detect_image(
        self,
        image: Union[str, np.ndarray],
        visualize: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run detection on single image.
        
        Args:
            image: Image path or numpy array.
            visualize: Whether to visualize results.
            output_path: Optional path to save visualization.
            
        Returns:
            Detection results dictionary.
        """
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)
            
        orig_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        outputs = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        results = self.postprocess(outputs, orig_shape)
        results['inference_time'] = inference_time
        
        # Visualize
        if visualize:
            vis_image = visualize_detections(
                image, results, COCO_CLASSES, self.score_threshold, output_path
            )
            results['visualization'] = vis_image
            
        return results
        
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True
    ) -> None:
        """
        Run detection on video.
        
        Args:
            video_path: Path to input video.
            output_path: Optional path to save output video.
            display: Whether to display video during processing.
        """
        if not HAS_CV2:
            print("Error: OpenCV required for video processing")
            return
            
        # Open video
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
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_idx = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Detect
            results = self.detect_image(frame, visualize=True)
            vis_frame = results.get('visualization', frame)
            total_time += results.get('inference_time', 0)
            
            # Display FPS
            avg_fps = (frame_idx + 1) / total_time if total_time > 0 else 0
            cv2.putText(
                vis_frame, f'FPS: {avg_fps:.1f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Write frame
            if writer is not None:
                writer.write(vis_frame)
                
            # Display
            if display:
                cv2.imshow('Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
                
        cap.release()
        
        if writer is not None:
            writer.release()
            
        if display:
            cv2.destroyAllWindows()
            
        print(f"Processed {frame_idx} frames")
        print(f"Average FPS: {frame_idx / total_time:.1f}")
        
    def detect_webcam(self, camera_id: int = 0) -> None:
        """
        Run detection on webcam stream.
        
        Args:
            camera_id: Camera device ID.
        """
        if not HAS_CV2:
            print("Error: OpenCV required for webcam")
            return
            
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
            
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Detect
            results = self.detect_image(frame, visualize=True)
            vis_frame = results.get('visualization', frame)
            
            # Display FPS
            fps = 1.0 / results.get('inference_time', 1)
            cv2.putText(
                vis_frame, f'FPS: {fps:.1f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            cv2.imshow('Detection', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()


def run_demo(
    model_path: str,
    config_path: str,
    input_source: str,
    output_path: Optional[str] = None,
    device: str = 'cuda',
    score_threshold: float = 0.3
) -> None:
    """
    Run detection demo.
    
    Args:
        model_path: Path to model checkpoint.
        config_path: Path to model configuration.
        input_source: Image path, video path, or 'webcam'.
        output_path: Optional path for output.
        device: Device to run on.
        score_threshold: Minimum confidence score.
    """
    demo = DetectionDemo(
        model_path=model_path,
        config_path=config_path,
        device=device,
        score_threshold=score_threshold
    )
    
    if input_source == 'webcam':
        demo.detect_webcam()
    elif input_source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        demo.detect_video(input_source, output_path)
    else:
        results = demo.detect_image(input_source, output_path=output_path)
        print(f"Detected {len(results['boxes'])} objects")
        print(f"Inference time: {results['inference_time']*1000:.1f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Demo')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--input', type=str, required=True, help='Input source')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--threshold', type=float, default=0.3, help='Score threshold')
    
    args = parser.parse_args()
    
    run_demo(
        model_path=args.model,
        config_path=args.config,
        input_source=args.input,
        output_path=args.output,
        device=args.device,
        score_threshold=args.threshold
    )
