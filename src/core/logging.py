"""
Logging configuration and setup for the RAG server.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from .config import LoggingConfig


def setup_logging(config: LoggingConfig):
    """Setup application logging with loguru"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        level=config.level,
        format=config.format,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler
    if config.file_path:
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            config.file_path,
            level=config.level,
            format=config.format,
            rotation=config.rotation,
            retention=config.retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Set up structured logging for different components
    logger.info("Logging initialized")


def get_logger(name: str):
    """Get a logger instance for a specific component"""
    return logger.bind(component=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        """Get logger instance for this class"""
        return get_logger(self.__class__.__name__)