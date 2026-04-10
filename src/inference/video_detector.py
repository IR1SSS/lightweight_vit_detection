"""
Video detection module for real-time inference.
"""

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from .predictor import Predictor


class VideoDetector(Predictor):
    """
    Video detector for processing video files and streams.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialize video detector.
        
        Args:
            model: Detection model
            device: Device for inference
            **kwargs: Additional arguments
        """
        super().__init__(model, device, **kwargs)
        
        # Performance tracking
        self.frame_times = []
        self.fps = 0.0
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video
            show: Whether to display video
            callback: Optional callback for each frame
            
        Returns:
            Dictionary with processing statistics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        self.frame_times = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Time the inference
                start_time = time.time()
                
                # Predict
                results = self.predict(frame)
                
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                
                # Draw detections
                from .visualizer import draw_detections
                vis_frame = draw_detections(frame, results["boxes"], results["scores"], results["labels"])
                
                # Add FPS display
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                cv2.putText(vis_frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write to output
                if writer:
                    writer.write(vis_frame)
                
                # Display
                if show:
                    cv2.imshow("Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                # Callback
                if callback:
                    callback(frame_count, frame, results)
                
                frame_count += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        # Calculate statistics
        if self.frame_times:
            self.fps = len(self.frame_times) / sum(self.frame_times)
        
        return {
            "total_frames": frame_count,
            "fps": self.fps,
            "avg_frame_time": sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
        }
    
    def process_webcam(
        self,
        camera_id: int = 0,
        output_path: Optional[str] = None,
        show: bool = True,
        max_frames: int = -1,
    ) -> Dict[str, Any]:
        """
        Process webcam stream.
        
        Args:
            camera_id: Camera device ID
            output_path: Path to save video
            show: Whether to display video
            max_frames: Maximum frames to process (-1 for unlimited)
            
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Default for webcam
        
        # Setup output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        self.frame_times = []
        frame_count = 0
        
        try:
            while True:
                if max_frames > 0 and frame_count >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                results = self.predict(frame)
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                
                # Draw
                from .visualizer import draw_detections
                vis_frame = draw_detections(frame, results["boxes"], results["scores"], results["labels"])
                
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                cv2.putText(vis_frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if writer:
                    writer.write(vis_frame)
                
                if show:
                    cv2.imshow("Webcam Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                frame_count += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        if self.frame_times:
            self.fps = len(self.frame_times) / sum(self.frame_times)
        
        return {
            "total_frames": frame_count,
            "fps": self.fps,
        }


class RealTimeDetector:
    """
    Real-time detector with asynchronous processing.
    
    Uses separate threads for capture and inference to maximize throughput.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        queue_size: int = 10,
        **kwargs,
    ):
        """
        Initialize real-time detector.
        
        Args:
            model: Detection model
            device: Device for inference
            queue_size: Frame queue size
            **kwargs: Additional arguments
        """
        self.predictor = Predictor(model, device, **kwargs)
        self.queue_size = queue_size
        
        # Queues for async processing
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        
        # Control flags
        self.running = False
        self.capture_thread = None
        self.inference_thread = None
    
    def start(self, source: Union[int, str]):
        """
        Start real-time detection.
        
        Args:
            source: Video source (camera ID or video path)
        """
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_worker,
            args=(source,)
        )
        self.capture_thread.start()
        
        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_worker
        )
        self.inference_thread.start()
    
    def stop(self):
        """Stop real-time detection."""
        self.running = False
        
        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
    
    def _capture_worker(self, source: Union[int, str]):
        """Capture thread worker."""
        cap = cv2.VideoCapture(source)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Drop frame if queue is full
                    pass
                
        finally:
            cap.release()
    
    def _inference_worker(self):
        """Inference thread worker."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                results = self.predictor.predict(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put((frame, results))
                    
            except queue.Empty:
                continue
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """
        Get the latest detection result.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame, results) or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def process_with_display(self, source: Union[int, str], show_fps: bool = True):
        """
        Process and display in real-time.
        
        Args:
            source: Video source
            show_fps: Show FPS counter
        """
        self.start(source)
        
        frame_times = []
        
        try:
            while True:
                result = self.get_result()
                
                if result is None:
                    if not self.running:
                        break
                    continue
                
                frame, detections = result
                
                # Draw
                from .visualizer import draw_detections
                vis_frame = draw_detections(
                    frame,
                    detections["boxes"],
                    detections["scores"],
                    detections["labels"],
                )
                
                # Calculate FPS
                if show_fps:
                    fps = len(frame_times) / sum(frame_times) if frame_times else 0
                    cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Real-time Detection", vis_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        finally:
            self.stop()
            cv2.destroyAllWindows()
