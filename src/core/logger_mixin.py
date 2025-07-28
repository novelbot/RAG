"""
Logger mixin - circular import safe module.

This module provides logging capabilities without importing config.
"""

from loguru import logger


def get_logger(name: str):
    """Get a logger instance for a specific component"""
    return logger.bind(component=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        """Get logger instance for this class"""
        return get_logger(self.__class__.__name__)