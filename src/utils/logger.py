"""
Logging Utilities for Lightweight ViT Detection System.

This module provides logging configuration and utilities for
consistent logging across the project.
"""

import os
import sys
import logging
from typing import Optional
from datetime import datetime


# Default format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logger(
    name: str = 'lightweight_vit',
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = 'outputs/logs',
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name.
        log_level: Logging level (e.g., logging.INFO).
        log_file: Optional specific log file name.
        log_dir: Directory for log files.
        console: Whether to output to console.
        format_string: Optional custom format string.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        format_string or DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # File handler
    if log_file is not None or log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{name}_{timestamp}.log'
            
        log_path = os.path.join(log_dir, log_file)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def get_logger(
    name: str = 'lightweight_vit',
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Get or create a logger by name.
    
    If logger doesn't exist, creates one with default settings.
    
    Args:
        name: Logger name.
        log_level: Logging level.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Set up if no handlers exist
    if not logger.handlers:
        logger = setup_logger(name, log_level)
        
    return logger


class LoggerManager:
    """
    Manager for multiple loggers.
    
    Provides centralized logging configuration for different modules.
    """
    
    _instance = None
    _loggers: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_level: int = logging.INFO,
        **kwargs
    ) -> logging.Logger:
        """
        Get or create logger by name.
        
        Args:
            name: Logger name.
            log_level: Logging level.
            **kwargs: Additional setup_logger arguments.
            
        Returns:
            Logger instance.
        """
        if name not in cls._loggers:
            cls._loggers[name] = setup_logger(name, log_level, **kwargs)
        return cls._loggers[name]
        
    @classmethod
    def set_level(cls, level: int) -> None:
        """Set log level for all loggers."""
        for logger in cls._loggers.values():
            logger.setLevel(level)


class ProgressLogger:
    """
    Logger for training progress with metrics tracking.
    
    Provides formatted output for training iterations and epochs.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        total_epochs: int = 100,
        log_interval: int = 50
    ) -> None:
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger instance.
            total_epochs: Total number of epochs.
            log_interval: Logging interval in batches.
        """
        self.logger = logger or get_logger('progress')
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        
        self.current_epoch = 0
        self.metrics_history = []
        
    def log_epoch_start(self, epoch: int) -> None:
        """Log epoch start."""
        self.current_epoch = epoch
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Epoch {epoch+1}/{self.total_epochs}")
        self.logger.info(f"{'='*60}")
        
    def log_batch(
        self,
        batch_idx: int,
        total_batches: int,
        metrics: dict
    ) -> None:
        """
        Log batch progress.
        
        Args:
            batch_idx: Current batch index.
            total_batches: Total number of batches.
            metrics: Dictionary of metric values.
        """
        if (batch_idx + 1) % self.log_interval == 0:
            metrics_str = ' | '.join([
                f'{k}: {v:.4f}' for k, v in metrics.items()
            ])
            self.logger.info(
                f"  Batch [{batch_idx+1}/{total_batches}] - {metrics_str}"
            )
            
    def log_epoch_end(self, metrics: dict) -> None:
        """
        Log epoch end with summary metrics.
        
        Args:
            metrics: Dictionary of epoch metrics.
        """
        self.metrics_history.append(metrics)
        
        metrics_str = ' | '.join([
            f'{k}: {v:.4f}' for k, v in metrics.items()
        ])
        self.logger.info(f"Epoch {self.current_epoch+1} Summary: {metrics_str}")
        
    def log_validation(self, metrics: dict) -> None:
        """
        Log validation results.
        
        Args:
            metrics: Dictionary of validation metrics.
        """
        metrics_str = ' | '.join([
            f'{k}: {v:.4f}' for k, v in metrics.items()
        ])
        self.logger.info(f"Validation: {metrics_str}")


# Convenience function for quick logging
def log_info(message: str, logger_name: str = 'lightweight_vit') -> None:
    """Log info message."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = 'lightweight_vit') -> None:
    """Log warning message."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = 'lightweight_vit') -> None:
    """Log error message."""
    get_logger(logger_name).error(message)
