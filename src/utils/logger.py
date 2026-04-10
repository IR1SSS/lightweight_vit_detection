"""
Logging utilities for the Lightweight ViT Detection System.
Provides colored logging with file and console handlers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import colorlog


def setup_logger(
    name: str = "vit_detection",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup logger with colored console output and optional file logging.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (if None, uses timestamp)
        level: Root logger level
        console_level: Console handler level
        file_level: File handler level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_format = (
        "%(log_color)s%(asctime)s | %(levelname)-8s | "
        "%(name)s:%(lineno)d | %(message)s%(reset)s"
    )
    console_formatter = colorlog.ColoredFormatter(
        console_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (no colors)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"train_{timestamp}.log"
        
        file_path = log_dir / log_file
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(
            file_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "vit_detection") -> logging.Logger:
    """
    Get existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Setup if not already configured
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log messages.
    """
    
    def __init__(self, logger: logging.Logger, prefix: str = ""):
        """
        Initialize adapter.
        
        Args:
            logger: Base logger
            prefix: Prefix to add to all messages
        """
        super().__init__(logger, {"prefix": prefix})
    
    def process(self, msg, kwargs):
        """Process log message with prefix."""
        prefix = self.extra.get("prefix", "")
        if prefix:
            return f"[{prefix}] {msg}", kwargs
        return msg, kwargs


class MetricLogger:
    """
    Logger for tracking and displaying training/evaluation metrics.
    """
    
    def __init__(self, logger: logging.Logger, delimiter: str = "\t"):
        """
        Initialize metric logger.
        
        Args:
            logger: Base logger
            delimiter: Delimiter between metrics
        """
        self.logger = logger
        self.delimiter = delimiter
        self.metrics = {}
    
    def update(self, **kwargs):
        """
        Update metric values.
        
        Args:
            **kwargs: Metric name-value pairs
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter(key)
            self.metrics[key].update(value)
    
    def __str__(self) -> str:
        """Return formatted metrics string."""
        return self.delimiter.join(
            str(meter) for meter in self.metrics.values()
        )
    
    def log(self, level: int = logging.INFO):
        """Log current metrics."""
        self.logger.log(level, str(self))
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str = "", fmt: str = ":f"):
        """
        Initialize average meter.
        
        Args:
            name: Metric name
            fmt: Format string for display
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update with new value.
        
        Args:
            val: New value
            n: Number of items (for batch averaging)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        """Return formatted string."""
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)


class ProgressTracker:
    """
    Tracks progress during training with time estimation.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize progress tracker.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.epoch_meters = {}
        self.batch_meters = {}
    
    def add_meter(self, name: str, is_epoch: bool = True):
        """
        Add a metric meter.
        
        Args:
            name: Meter name
            is_epoch: True for epoch-level metrics, False for batch-level
        """
        meter = AverageMeter(name)
        if is_epoch:
            self.epoch_meters[name] = meter
        else:
            self.batch_meters[name] = meter
    
    def update(self, name: str, value: float, n: int = 1):
        """
        Update a meter.
        
        Args:
            name: Meter name
            value: New value
            n: Batch size
        """
        if name in self.epoch_meters:
            self.epoch_meters[name].update(value, n)
        if name in self.batch_meters:
            self.batch_meters[name].update(value, n)
    
    def log_batch(self, batch_idx: int, num_batches: int):
        """
        Log batch progress.
        
        Args:
            batch_idx: Current batch index
            num_batches: Total number of batches
        """
        progress = f"[{batch_idx}/{num_batches}]"
        metrics = " | ".join(str(m) for m in self.batch_meters.values())
        self.logger.info(f"{progress} {metrics}")
    
    def log_epoch(self, epoch: int, num_epochs: int):
        """
        Log epoch summary.
        
        Args:
            epoch: Current epoch
            num_epochs: Total number of epochs
        """
        progress = f"Epoch [{epoch}/{num_epochs}]"
        metrics = " | ".join(str(m) for m in self.epoch_meters.values())
        self.logger.info(f"{progress} {metrics}")
        
        # Reset batch meters
        for meter in self.batch_meters.values():
            meter.reset()
