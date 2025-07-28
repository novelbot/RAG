"""
Base classes and interfaces for embedding model integration.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from src.core.logger_mixin import LoggerMixin
from src.core.exceptions import EmbeddingError, ConfigurationError
from .types import EmbeddingProvider, EmbeddingConfig


@dataclass
class EmbeddingDimension:
    """Embedding dimension information."""
    dimensions: int
    model_name: str
    max_dimensions: Optional[int] = None
    supported_dimensions: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.supported_dimensions and self.dimensions not in self.supported_dimensions:
            raise ConfigurationError(
                f"Dimension {self.dimensions} not supported. "
                f"Supported dimensions: {self.supported_dimensions}"
            )


@dataclass
class EmbeddingUsage:
    """Embedding usage statistics."""
    prompt_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: 'EmbeddingUsage') -> 'EmbeddingUsage':
        return EmbeddingUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    input: Union[str, List[str]]
    model: Optional[str] = None
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    user: Optional[str] = None
    
    # Processing options
    truncate: bool = True
    batch_size: Optional[int] = None
    normalize: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.input, str):
            self.input = [self.input]
        
        if not self.input:
            raise ValueError("Input cannot be empty")
        
        # Validate input lengths
        for text in self.input:
            if not isinstance(text, str):
                raise ValueError("All inputs must be strings")
            if len(text.strip()) == 0:
                raise ValueError("Input text cannot be empty or whitespace only")


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""
    embeddings: List[List[float]]
    model: str
    usage: EmbeddingUsage
    dimensions: int
    
    # Performance metrics
    response_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.embeddings:
            raise ValueError("Embeddings cannot be empty")
        
        # Validate dimensions consistency
        if self.embeddings:
            actual_dim = len(self.embeddings[0])
            if actual_dim != self.dimensions:
                self.dimensions = actual_dim
            
            # Check all embeddings have same dimension
            for i, embedding in enumerate(self.embeddings):
                if len(embedding) != self.dimensions:
                    raise ValueError(f"Embedding {i} has dimension {len(embedding)}, expected {self.dimensions}")
    
    def to_numpy(self) -> np.ndarray:
        """Convert embeddings to numpy array."""
        return np.array(self.embeddings)
    
    def normalize(self) -> 'EmbeddingResponse':
        """Normalize embeddings to unit vectors."""
        embeddings_array = self.to_numpy()
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / norms
        
        return EmbeddingResponse(
            embeddings=normalized_embeddings.tolist(),
            model=self.model,
            usage=self.usage,
            dimensions=self.dimensions,
            response_time=self.response_time,
            metadata=self.metadata
        )
    
    def similarity(self, other: 'EmbeddingResponse', method: str = "cosine") -> np.ndarray:
        """Calculate similarity between embeddings."""
        if method == "cosine":
            # Cosine similarity
            a = self.to_numpy()
            b = other.to_numpy()
            return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))
        elif method == "euclidean":
            # Euclidean distance (smaller is more similar)
            a = self.to_numpy()
            b = other.to_numpy()
            return np.sqrt(np.sum((a[:, np.newaxis] - b[np.newaxis, :]) ** 2, axis=2))
        else:
            raise ValueError(f"Unsupported similarity method: {method}")


class BaseEmbeddingProvider(ABC, LoggerMixin):
    """
    Abstract base class for embedding providers.
    
    This class defines the interface that all embedding providers must implement.
    It provides common functionality and enforces the contract for embedding operations.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.provider_name = config.provider.value
        self._client = None
        self._async_client = None
        self._last_health_check = None
        self._health_status = True
        
        # Initialize provider-specific client
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    async def generate_embeddings_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings asynchronously.
        
        Args:
            request: Embedding request
            
        Returns:
            Embedding response
        """
        pass
    
    @abstractmethod
    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings synchronously.
        
        Args:
            request: Embedding request
            
        Returns:
            Embedding response
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self, model: str) -> EmbeddingDimension:
        """
        Get embedding dimension information for a model.
        
        Args:
            model: Model name
            
        Returns:
            Embedding dimension information
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate provider configuration.
        
        Returns:
            True if configuration is valid
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, num_tokens: int, model: str) -> float:
        """
        Estimate cost for embedding generation.
        
        Args:
            num_tokens: Number of tokens to estimate cost for
            model: Model name
            
        Returns:
            Estimated cost in dollars
        """
        pass
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform health check asynchronously.
        
        Returns:
            Health check result
        """
        try:
            start_time = time.time()
            
            # Simple health check with a minimal embedding request
            test_request = EmbeddingRequest(
                input=["health check"],
                model=self.config.model
            )
            
            await self.generate_embeddings_async(test_request)
            
            response_time = time.time() - start_time
            self._health_status = True
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.config.model,
                "response_time": response_time,
                "timestamp": self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._health_status = False
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.config.model,
                "error": str(e),
                "timestamp": self._last_health_check.isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check synchronously.
        
        Returns:
            Health check result
        """
        try:
            start_time = time.time()
            
            # Simple health check with a minimal embedding request
            test_request = EmbeddingRequest(
                input=["health check"],
                model=self.config.model
            )
            
            self.generate_embeddings(test_request)
            
            response_time = time.time() - start_time
            self._health_status = True
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.config.model,
                "response_time": response_time,
                "timestamp": self._last_health_check.isoformat()
            }
            
        except Exception as e:
            self._health_status = False
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.config.model,
                "error": str(e),
                "timestamp": self._last_health_check.isoformat()
            }
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self._health_status
    
    def get_last_health_check(self) -> Optional[datetime]:
        """Get timestamp of last health check."""
        return self._last_health_check
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit vectors."""
        if not self.config.normalize_embeddings:
            return embeddings
        
        normalized = []
        for embedding in embeddings:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                normalized.append((embedding_array / norm).tolist())
            else:
                normalized.append(embedding)
        
        return normalized
    
    def _batch_texts(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Split texts into batches."""
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i:i + batch_size])
        return batches
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass