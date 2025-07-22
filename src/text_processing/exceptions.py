"""
Text Processing Exceptions.

This module defines custom exceptions for text processing operations
including cleaning, chunking, and metadata management.
"""


class TextProcessingError(Exception):
    """Base exception for text processing operations."""
    pass


class ChunkingError(TextProcessingError):
    """Exception raised during text chunking operations."""
    pass


class MetadataError(TextProcessingError):
    """Exception raised during metadata processing operations."""
    pass


class InvalidConfigurationError(TextProcessingError):
    """Exception raised when configuration is invalid."""
    pass


class CleaningError(TextProcessingError):
    """Exception raised during text cleaning operations."""
    pass


class UnsupportedEncodingError(TextProcessingError):
    """Exception raised when text encoding is not supported."""
    pass


class TextValidationError(TextProcessingError):
    """Exception raised when text validation fails."""
    pass