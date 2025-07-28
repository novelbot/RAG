"""
Embedding Model Integration Layer - Unified interface for multiple embedding providers.
"""

from .types import EmbeddingProvider, EmbeddingConfig
from .base import (
    EmbeddingRequest, EmbeddingResponse, 
    EmbeddingUsage, EmbeddingDimension, BaseEmbeddingProvider
)
from .manager import EmbeddingManager, EmbeddingLoadBalancingStrategy, EmbeddingProviderConfig
from .providers import OpenAIEmbeddingProvider, GoogleEmbeddingProvider, OllamaEmbeddingProvider

__all__ = [
    # Core types
    "EmbeddingProvider",
    "EmbeddingConfig",
    "EmbeddingRequest", 
    "EmbeddingResponse",
    "EmbeddingUsage",
    "EmbeddingDimension",
    "BaseEmbeddingProvider",
    
    # Manager
    "EmbeddingManager",
    "EmbeddingLoadBalancingStrategy",
    "EmbeddingProviderConfig",
    
    # Providers
    "OpenAIEmbeddingProvider",
    "GoogleEmbeddingProvider",
    "OllamaEmbeddingProvider",
]