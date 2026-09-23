"""
NeuralStream-Core Structured Logging System
Provides colored, formatted logging with file rotation support.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class LogFormatter(logging.Formatter):
    """Custom formatter with color coding for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "neuralstream",
    level: str = "INFO",
    log_dir: str = "logs",
    enable_file: bool = True,
    enable_console: bool = True
) -> logging.Logger:
    """
    Configure and return a structured logger instance.
    
    Args:
        name: Logger name identifier
        level: Logging level string
        log_dir: Directory for log file output
        enable_file: Whether to write logs to file
        enable_console: Whether to write logs to console
    
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handler attachment
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    
    # Console handler
    if enable_console:
        if HAS_RICH:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True
            )
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            console_formatter = LogFormatter(
                fmt='%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"neuralstream_{timestamp}.log"
        
        file_handler = logging.FileHandler(str(file_path), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "neuralstream") -> logging.Logger:
    """Retrieve an existing logger or create a default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger