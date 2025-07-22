"""
Exception classes for Response Generation System.
"""

from typing import Optional, Dict, Any


class ResponseGenerationError(Exception):
    """Base exception for response generation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class PromptEngineeringError(ResponseGenerationError):
    """Exception raised during prompt engineering operations."""
    pass


class ResponseEvaluationError(ResponseGenerationError):
    """Exception raised during response quality evaluation."""
    pass


class ResponseProcessingError(ResponseGenerationError):
    """Exception raised during response post-processing."""
    pass


class EnsembleError(ResponseGenerationError):
    """Exception raised during ensemble response generation."""
    pass


class TimeoutError(ResponseGenerationError):
    """Exception raised when response generation times out."""
    
    def __init__(self, message: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, details)


class ProviderUnavailableError(ResponseGenerationError):
    """Exception raised when no LLM providers are available."""
    pass


class ContextTooLongError(ResponseGenerationError):
    """Exception raised when context exceeds token limits."""
    
    def __init__(self, message: str, token_count: int, max_tokens: int, details: Optional[Dict[str, Any]] = None):
        self.token_count = token_count
        self.max_tokens = max_tokens
        super().__init__(message, details)


class ResponseQualityError(ResponseGenerationError):
    """Exception raised when response quality is below threshold."""
    
    def __init__(self, message: str, quality_score: float, threshold: float, details: Optional[Dict[str, Any]] = None):
        self.quality_score = quality_score
        self.threshold = threshold
        super().__init__(message, details)