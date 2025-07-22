"""
File System Data Ingestion Exceptions.

This module defines custom exceptions for file system data ingestion operations.
"""

from src.core.exceptions import BaseCustomException


class FileSystemError(BaseCustomException):
    """Base exception for file system operations."""
    pass


class ParsingError(FileSystemError):
    """Exception raised when file parsing fails."""
    pass


class UnsupportedFileTypeError(FileSystemError):
    """Exception raised when encountering unsupported file types."""
    pass


class FileChangeDetectionError(FileSystemError):
    """Exception raised when file change detection fails."""
    pass


class MetadataExtractionError(FileSystemError):
    """Exception raised when metadata extraction fails."""
    pass


class BatchProcessingError(FileSystemError):
    """Exception raised when batch processing fails."""
    pass


class FileCorruptionError(ParsingError):
    """Exception raised when a file is corrupted or cannot be read."""
    pass


class FilePermissionError(FileSystemError):
    """Exception raised when file permission issues occur."""
    pass


class DirectoryScanError(FileSystemError):
    """Exception raised when directory scanning fails."""
    pass