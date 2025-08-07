"""
Multi-LLM Integration Layer - Unified interface for multiple LLM providers.
"""

from .base import (
    LLMProvider, LLMRequest, LLMResponse, LLMConfig, LLMMessage, 
    LLMRole, LLMUsage, LLMStreamChunk, BaseLLMProvider
)
# Import from manager (now using LangChain)
from .manager import (
    LangChainLLMManager as LLMManager,
    LoadBalancingStrategy, 
    ProviderConfig, 
    ProviderStats,
    create_llm_manager
)
# Only import Ollama provider (others are handled by LangChain)
from .providers.ollama import OllamaProvider

# Create placeholder classes for backward compatibility
class OpenAIProvider:
    """Deprecated - use LangChain integration instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use LangChain integration via LLMManager instead")

class GeminiProvider:
    """Deprecated - use LangChain integration instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use LangChain integration via LLMManager instead")

class ClaudeProvider:
    """Deprecated - use LangChain integration instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use LangChain integration via LLMManager instead")

__all__ = [
    # Core types
    "LLMProvider",
    "LLMRequest", 
    "LLMResponse",
    "LLMConfig",
    "LLMMessage",
    "LLMRole",
    "LLMUsage",
    "LLMStreamChunk",
    "BaseLLMProvider",
    
    # Manager
    "LLMManager",
    "LoadBalancingStrategy",
    "ProviderConfig",
    "ProviderStats",
    
    # Manager (now using LangChain)
    "create_llm_manager",
    
    # Providers (deprecated except Ollama)
    "OpenAIProvider",  # Deprecated
    "GeminiProvider",  # Deprecated
    "ClaudeProvider",  # Deprecated
    "OllamaProvider",  # Still available for special features
]