"""
Factory functions for creating embedding clients and managers.
"""

from typing import Union, List
from src.embedding.types import EmbeddingConfig, EmbeddingProvider
from src.embedding.manager import EmbeddingManager, EmbeddingProviderConfig
from src.embedding.providers import OpenAIEmbeddingProvider, GoogleEmbeddingProvider, OllamaEmbeddingProvider
from src.embedding.base import BaseEmbeddingProvider


def get_embedding_client(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """
    Factory function to create embedding client based on provider.
    
    Args:
        config: Embedding configuration
        
    Returns:
        BaseEmbeddingProvider: Configured embedding provider
    """
    if config.provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingProvider(config)
    elif config.provider == EmbeddingProvider.GOOGLE:
        return GoogleEmbeddingProvider(config)
    elif config.provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbeddingProvider(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")


def get_embedding_manager(configs: Union[List[EmbeddingConfig], List[EmbeddingProviderConfig]], 
                         enable_cache: bool = True) -> EmbeddingManager:
    """
    Factory function to create embedding manager with multiple providers.
    
    Args:
        configs: List of embedding configurations or provider configurations
        enable_cache: Whether to enable caching
        
    Returns:
        EmbeddingManager: Configured embedding manager
    """
    # Convert EmbeddingConfig to EmbeddingProviderConfig if needed
    provider_configs = []
    
    for config in configs:
        if isinstance(config, EmbeddingConfig):
            provider_configs.append(EmbeddingProviderConfig(
                provider=config.provider,
                config=config,
                enabled=True
            ))
        else:
            provider_configs.append(config)
    
    return EmbeddingManager(provider_configs, enable_cache=enable_cache)