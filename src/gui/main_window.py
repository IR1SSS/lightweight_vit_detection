"""
Main Window for Lightweight ViT Detection System GUI.

This module provides the main application window with:
- Camera/Video/Image detection
- Real-time display
- Detection logging
- Settings panel
"""

import os
import sys
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QFileDialog, QTextEdit, QGroupBox, QStatusBar,
    QSplitter, QMessageBox, QProgressBar, QTabWidget,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction, QIcon

from .detector import DetectorThread, DetectionResult


class VideoWidget(QLabel):
    """Widget for displaying video frames."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
            }
        """)
        self.setText("No Video Source")
        self.setFont(QFont("Arial", 14))
        
    def update_frame(self, frame: np.ndarray):
        """Update displayed frame."""
        if frame is None:
            return
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Create QImage
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)


class LogPanel(QTextEdit):
    """Panel for displaying detection logs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        self.max_lines = 1000
        
    def add_log(self, message: str, level: str = "INFO"):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        colors = {
            "INFO": "#4ec9b0",
            "WARNING": "#dcdcaa",
            "ERROR": "#f14c4c",
            "DETECTION": "#569cd6"
        }
        color = colors.get(level, "#d4d4d4")
        
        html = f'<span style="color: #808080">[{timestamp}]</span> '
        html += f'<span style="color: {color}">[{level}]</span> '
        html += f'<span style="color: #d4d4d4">{message}</span><br>'
        
        self.insertHtml(html)
        self.ensureCursorVisible()
        
        # Limit lines
        if self.document().blockCount() > self.max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 100)
            cursor.removeSelectedText()
            
    def add_detection_log(self, result_dict: dict):
        """Add detection result log."""
        num_det = result_dict.get('num_detections', 0)
        time_ms = result_dict.get('inference_time_ms', 0)
        
        if num_det > 0:
            detections = result_dict.get('detections', [])
            classes = [d['class'] for d in detections[:5]]
            classes_str = ", ".join(classes)
            if len(detections) > 5:
                classes_str += f" +{len(detections)-5} more"
            self.add_log(
                f"Detected {num_det} objects ({time_ms:.1f}ms): {classes_str}",
                "DETECTION"
            )


class ControlPanel(QWidget):
    """Control panel for detection settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Source selection group
        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout(source_group)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        camera_layout.addWidget(self.camera_combo)
        self.btn_camera = QPushButton("Start Camera")
        self.btn_camera.setStyleSheet(self._get_button_style("#4CAF50"))
        camera_layout.addWidget(self.btn_camera)
        source_layout.addLayout(camera_layout)
        
        # File selection
        file_layout = QHBoxLayout()
        self.btn_open_image = QPushButton("Open Image")
        self.btn_open_image.setStyleSheet(self._get_button_style("#2196F3"))
        self.btn_open_video = QPushButton("Open Video")
        self.btn_open_video.setStyleSheet(self._get_button_style("#9C27B0"))
        file_layout.addWidget(self.btn_open_image)
        file_layout.addWidget(self.btn_open_video)
        source_layout.addLayout(file_layout)
        
        layout.addWidget(source_group)
        
        # Detection settings group
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        settings_layout.addLayout(conf_layout)
        
        # NMS threshold
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS:"))
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(0, 100)
        self.nms_slider.setValue(45)
        self.nms_label = QLabel("0.45")
        nms_layout.addWidget(self.nms_slider)
        nms_layout.addWidget(self.nms_label)
        settings_layout.addLayout(nms_layout)
        
        layout.addWidget(settings_group)
        
        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet(self._get_button_style("#FF9800"))
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._get_button_style("#f44336"))
        
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_stop)
        
        layout.addWidget(control_group)
        
        # Model settings
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        
        self.btn_load_model = QPushButton("Load Model Weights")
        self.btn_load_model.setStyleSheet(self._get_button_style("#607D8B"))
        model_layout.addWidget(self.btn_load_model)
        
        self.model_status = QLabel("Model: Default (not loaded)")
        self.model_status.setStyleSheet("color: #888;")
        model_layout.addWidget(self.model_status)
        
        layout.addWidget(model_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout.addWidget(self.fps_label)
        
        self.detection_count_label = QLabel("Total Detections: 0")
        stats_layout.addWidget(self.detection_count_label)
        
        self.frame_count_label = QLabel("Frames Processed: 0")
        stats_layout.addWidget(self.frame_count_label)
        
        layout.addWidget(stats_group)
        
        # Spacer
        layout.addStretch()
        
        # Connect slider signals
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.nms_slider.valueChanged.connect(self._on_nms_changed)
        
    def _get_button_style(self, color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}aa;
            }}
            QPushButton:disabled {{
                background-color: #555;
                color: #888;
            }}
        """
        
    def _on_conf_changed(self, value: int):
        self.conf_label.setText(f"{value/100:.2f}")
        
    def _on_nms_changed(self, value: int):
        self.nms_label.setText(f"{value/100:.2f}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.detector_thread = None
        self.total_detections = 0
        self.frame_count = 0
        
        self.setup_ui()
        self.setup_connections()
        self.setup_detector()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Lightweight ViT Detection System")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #252526;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #d4d4d4;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: #d4d4d4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3d3d3d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 3px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side - Video display and logs
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # Video display
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_widget, stretch=3)
        
        # Log panel
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout(log_group)
        self.log_panel = LogPanel()
        self.log_panel.setMaximumHeight(200)
        log_layout.addWidget(self.log_panel)
        
        # Log controls
        log_controls = QHBoxLayout()
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self.log_panel.clear)
        self.btn_save_log = QPushButton("Save Log")
        self.btn_save_log.clicked.connect(self.save_log)
        log_controls.addWidget(self.btn_clear_log)
        log_controls.addWidget(self.btn_save_log)
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        
        left_layout.addWidget(log_group, stretch=1)
        
        main_layout.addWidget(left_widget, stretch=3)
        
        # Right side - Control panel
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(300)
        main_layout.addWidget(self.control_panel)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a model or select input source")
        
        # Menu bar
        self.setup_menu()
        
    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #333;
                color: #d4d4d4;
            }
            QMenuBar::item:selected {
                background-color: #0078d4;
            }
            QMenu {
                background-color: #333;
                color: #d4d4d4;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_image_action = QAction("Open Image", self)
        open_image_action.setShortcut("Ctrl+I")
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)
        
        open_video_action = QAction("Open Video", self)
        open_video_action.setShortcut("Ctrl+V")
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)
        
        file_menu.addSeparator()
        
        save_log_action = QAction("Save Log", self)
        save_log_action.setShortcut("Ctrl+S")
        save_log_action.triggered.connect(self.save_log)
        file_menu.addAction(save_log_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Model menu
        model_menu = menubar.addMenu("Model")
        
        load_model_action = QAction("Load Model Weights", self)
        load_model_action.triggered.connect(self.load_model_weights)
        model_menu.addAction(load_model_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Setup signal-slot connections."""
        cp = self.control_panel
        
        cp.btn_camera.clicked.connect(self.start_camera)
        cp.btn_open_image.clicked.connect(self.open_image)
        cp.btn_open_video.clicked.connect(self.open_video)
        cp.btn_pause.clicked.connect(self.toggle_pause)
        cp.btn_stop.clicked.connect(self.stop_detection)
        cp.btn_load_model.clicked.connect(self.load_model_weights)
        
        cp.conf_slider.valueChanged.connect(self.update_confidence)
        cp.nms_slider.valueChanged.connect(self.update_nms)
        
    def setup_detector(self):
        """Initialize detector thread."""
        self.detector_thread = DetectorThread(self)
        self.detector_thread.frame_ready.connect(self.on_frame_ready)
        self.detector_thread.detection_logged.connect(self.on_detection_logged)
        self.detector_thread.error_occurred.connect(self.on_error)
        self.detector_thread.fps_updated.connect(self.on_fps_updated)
        self.detector_thread.finished.connect(self.on_detection_finished)
        
        # Load default model
        self.log_panel.add_log("Initializing model...", "INFO")
        if self.detector_thread.load_model():
            self.log_panel.add_log("Model loaded successfully", "INFO")
            self.control_panel.model_status.setText("Model: MobileViT (default)")
        else:
            self.log_panel.add_log("Failed to load model", "ERROR")
            
    @pyqtSlot()
    def start_camera(self):
        """Start camera detection."""
        if self.detector_thread.isRunning():
            self.stop_detection()
            
        camera_idx = self.control_panel.camera_combo.currentIndex()
        self.detector_thread.set_source('camera', camera_idx)
        
        self.log_panel.add_log(f"Starting camera {camera_idx}...", "INFO")
        self.detector_thread.start()
        
        self._set_running_state(True)
        self.status_bar.showMessage(f"Camera {camera_idx} - Running")
        
    @pyqtSlot()
    def open_image(self):
        """Open image file for detection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        
        if file_path:
            if self.detector_thread.isRunning():
                self.stop_detection()
                
            self.detector_thread.set_source('image', file_path)
            self.log_panel.add_log(f"Processing image: {os.path.basename(file_path)}", "INFO")
            self.detector_thread.start()
            
            self._set_running_state(True)
            self.status_bar.showMessage(f"Image: {os.path.basename(file_path)}")
            
    @pyqtSlot()
    def open_video(self):
        """Open video file for detection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            "", "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        
        if file_path:
            if self.detector_thread.isRunning():
                self.stop_detection()
                
            self.detector_thread.set_source('video', file_path)
            self.log_panel.add_log(f"Processing video: {os.path.basename(file_path)}", "INFO")
            self.detector_thread.start()
            
            self._set_running_state(True)
            self.status_bar.showMessage(f"Video: {os.path.basename(file_path)}")
            
    @pyqtSlot()
    def toggle_pause(self):
        """Toggle pause/resume."""
        if self.detector_thread.paused:
            self.detector_thread.resume()
            self.control_panel.btn_pause.setText("Pause")
            self.log_panel.add_log("Detection resumed", "INFO")
        else:
            self.detector_thread.pause()
            self.control_panel.btn_pause.setText("Resume")
            self.log_panel.add_log("Detection paused", "INFO")
            
    @pyqtSlot()
    def stop_detection(self):
        """Stop detection."""
        if self.detector_thread.isRunning():
            self.detector_thread.stop()
            self.detector_thread.wait(3000)
            
        self._set_running_state(False)
        self.log_panel.add_log("Detection stopped", "INFO")
        self.status_bar.showMessage("Stopped")
        
    @pyqtSlot()
    def load_model_weights(self):
        """Load model weights from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model Weights",
            "", "Model Files (*.pth *.pt *.ckpt)"
        )
        
        if file_path:
            self.log_panel.add_log(f"Loading model: {os.path.basename(file_path)}", "INFO")
            
            if self.detector_thread.load_model(model_path=file_path):
                self.control_panel.model_status.setText(
                    f"Model: {os.path.basename(file_path)}"
                )
                self.log_panel.add_log("Model loaded successfully", "INFO")
            else:
                self.log_panel.add_log("Failed to load model", "ERROR")
                
    @pyqtSlot(int)
    def update_confidence(self, value: int):
        """Update confidence threshold."""
        self.detector_thread.set_confidence_threshold(value / 100)
        
    @pyqtSlot(int)
    def update_nms(self, value: int):
        """Update NMS threshold."""
        self.detector_thread.set_nms_threshold(value / 100)
        
    @pyqtSlot(np.ndarray, object)
    def on_frame_ready(self, frame: np.ndarray, result: DetectionResult):
        """Handle processed frame."""
        self.video_widget.update_frame(frame)
        self.frame_count += 1
        self.control_panel.frame_count_label.setText(
            f"Frames Processed: {self.frame_count}"
        )
        
    @pyqtSlot(dict)
    def on_detection_logged(self, result_dict: dict):
        """Handle detection result logging."""
        self.log_panel.add_detection_log(result_dict)
        self.total_detections += result_dict.get('num_detections', 0)
        self.control_panel.detection_count_label.setText(
            f"Total Detections: {self.total_detections}"
        )
        
    @pyqtSlot(str)
    def on_error(self, error_msg: str):
        """Handle error from detector."""
        self.log_panel.add_log(error_msg, "ERROR")
        
    @pyqtSlot(float)
    def on_fps_updated(self, fps: float):
        """Update FPS display."""
        self.control_panel.fps_label.setText(f"FPS: {fps:.1f}")
        
    @pyqtSlot()
    def on_detection_finished(self):
        """Handle detection thread finished."""
        self._set_running_state(False)
        self.log_panel.add_log("Detection finished", "INFO")
        
    def _set_running_state(self, running: bool):
        """Set UI state based on running status."""
        self.control_panel.btn_pause.setEnabled(running)
        self.control_panel.btn_stop.setEnabled(running)
        self.control_panel.btn_pause.setText("Pause")
        
    @pyqtSlot()
    def save_log(self):
        """Save log to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log",
            f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_panel.toPlainText())
            self.log_panel.add_log(f"Log saved to: {file_path}", "INFO")
            
    @pyqtSlot()
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About",
            "Lightweight ViT Detection System\n\n"
            "A real-time object detection system based on\n"
            "lightweight Vision Transformer architecture.\n\n"
            "Features:\n"
            "- Real-time camera detection\n"
            "- Image and video file detection\n"
            "- Detection logging\n\n"
            "Version: 1.0.0"
        )
        
    def closeEvent(self, event):
        """Handle window close event."""
        if self.detector_thread.isRunning():
            self.detector_thread.stop()
            self.detector_thread.wait(3000)
        event.accept()
