"""
Embedding Model Integration Layer - Unified interface for multiple embedding providers.
"""

from .base import (
    EmbeddingProvider, EmbeddingConfig, EmbeddingRequest, EmbeddingResponse, 
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