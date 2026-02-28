"""
GUI module for Lightweight ViT Detection System.

This module provides a desktop application with:
- Real-time camera detection
- Image/video file detection
- Detection logging

Using tkinter for maximum compatibility.
"""

# Conditional imports based on available libraries
_gui_backend = None

try:
    import tkinter
    _gui_backend = 'tkinter'
except ImportError:
    pass

if _gui_backend == 'tkinter':
    from .main_window_tk import MainWindow, DetectorThread
else:
    # Fallback - no GUI available
    MainWindow = None
    DetectorThread = None

__all__ = [
    'MainWindow',
    'DetectorThread',
]
