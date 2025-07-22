"""
Custom exceptions for data extraction operations.
"""

from src.core.exceptions import BaseCustomException


class ExtractionError(BaseCustomException):
    """Base exception for data extraction errors."""
    pass


class ExtractionTimeoutError(ExtractionError):
    """Exception raised when extraction operations timeout."""
    pass


class ExtractionValidationError(ExtractionError):
    """Exception raised when extracted data fails validation."""
    pass


class ExtractionConnectionError(ExtractionError):
    """Exception raised when database connection fails during extraction."""
    pass


class ExtractionQueryError(ExtractionError):
    """Exception raised when SQL query execution fails."""
    pass


class ExtractionConfigurationError(ExtractionError):
    """Exception raised when extraction configuration is invalid."""
    pass


class ExtractionPermissionError(ExtractionError):
    """Exception raised when extraction lacks required permissions."""
    pass