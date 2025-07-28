"""
Custom exceptions for the RAG server application.
"""

from typing import Optional, Any


class BaseCustomException(Exception):
    """Base custom exception class"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class RAGException(Exception):
    """Base exception for RAG server errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class ConfigurationError(RAGException):
    """Configuration related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class DatabaseError(RAGException):
    """Database related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class MilvusError(RAGException):
    """Milvus vector database related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class LLMError(RAGException):
    """LLM provider related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class EmbeddingError(RAGException):
    """Embedding model related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class AuthenticationError(RAGException):
    """Authentication related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 401, details)


class AuthorizationError(RAGException):
    """Authorization related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 403, details)


class ValidationError(RAGException):
    """Input validation errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 400, details)


class NotFoundError(RAGException):
    """Resource not found errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 404, details)


class RateLimitError(RAGException):
    """Rate limiting errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 429, details)


class ProcessingError(RAGException):
    """Data processing errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class PipelineError(RAGException):
    """Pipeline processing errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class ConnectionError(RAGException):
    """Connection related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class IndexError(RAGException):
    """Index related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class SchemaError(RAGException):
    """Schema related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class SearchError(RAGException):
    """Search related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class RBACError(RAGException):
    """RBAC related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class PermissionError(RAGException):
    """Permission related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 403, details)


class PerformanceError(RAGException):
    """Performance related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class HealthCheckError(RAGException):
    """Health check related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class CollectionError(RAGException):
    """Collection related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class VectorError(RAGException):
    """Vector related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)


class RetryError(RAGException):
    """Retry operation related errors"""
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, details: Optional[Any] = None):
        self.original_exception = original_exception
        super().__init__(message, 500, details)


class TokenLimitError(RAGException):
    """Token limit related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 400, details)


class CircuitBreakerError(RAGException):
    """Circuit breaker related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 503, details)


class StorageError(RAGException):
    """Storage related errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, 500, details)