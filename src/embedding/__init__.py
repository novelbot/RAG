"""
Embedding Model Integration Layer - Unified interface for multiple embedding providers.
"""

from .types import EmbeddingProvider, EmbeddingConfig
from .base import (
    EmbeddingRequest, EmbeddingResponse, 
    EmbeddingUsage, EmbeddingDimension, BaseEmbeddingProvider
)
from .manager import EmbeddingManager, EmbeddingLoadBalancingStrategy, EmbeddingProviderConfig
# Import from LangChain-based factory
from .factory_langchain import get_embedding_client, get_embedding_manager
# Only import Ollama provider (others are handled by LangChain)
from .providers.ollama import OllamaEmbeddingProvider

# Create placeholder classes for backward compatibility
class OpenAIEmbeddingProvider:
    """Deprecated - use LangChain integration instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use get_embedding_client() with LangChain integration instead")

class GoogleEmbeddingProvider:
    """Deprecated - use LangChain integration instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use get_embedding_client() with LangChain integration instead")

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
    
    # Factory functions (now using LangChain)
    "get_embedding_client",
    "get_embedding_manager",
    
    # Providers (deprecated except Ollama)
    "OpenAIEmbeddingProvider",  # Deprecated
    "GoogleEmbeddingProvider",  # Deprecated
    "OllamaEmbeddingProvider",  # Still available for special features
]