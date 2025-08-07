"""
Factory functions for creating LangChain-based embedding clients and managers.
"""

from typing import Union, List
import os

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

# Local imports
from src.embedding.types import EmbeddingConfig, EmbeddingProvider
from src.embedding.base import BaseEmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage
from src.embedding.providers.ollama import OllamaEmbeddingProvider  # Keep for special features
from src.core.exceptions import EmbeddingError
import time
import asyncio


class LangChainEmbeddingAdapter(BaseEmbeddingProvider):
    """Adapter to make LangChain embeddings compatible with our BaseEmbeddingProvider interface."""
    
    def __init__(self, langchain_embeddings: Embeddings, config: EmbeddingConfig):
        """Initialize adapter with LangChain embeddings."""
        super().__init__(config)
        self.langchain_embeddings = langchain_embeddings
    
    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings synchronously."""
        start_time = time.time()
        
        try:
            # Ensure input is a list
            texts = request.input if isinstance(request.input, list) else [request.input]
            
            # Generate embeddings
            if len(texts) == 1:
                embeddings = [self.langchain_embeddings.embed_query(texts[0])]
            else:
                embeddings = self.langchain_embeddings.embed_documents(texts)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Estimate token usage
            total_tokens = sum(len(text.split()) for text in texts) * 1.3
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=request.model or self.config.model,
                usage=EmbeddingUsage(
                    prompt_tokens=int(total_tokens),
                    total_tokens=int(total_tokens)
                ),
                dimensions=len(embeddings[0]) if embeddings else 0,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"LangChain embedding generation failed: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    async def generate_embeddings_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings asynchronously."""
        start_time = time.time()
        
        try:
            texts = request.input if isinstance(request.input, list) else [request.input]
            
            # Check if async methods are available
            if hasattr(self.langchain_embeddings, 'aembed_documents'):
                if len(texts) == 1:
                    embeddings = [await self.langchain_embeddings.aembed_query(texts[0])]
                else:
                    embeddings = await self.langchain_embeddings.aembed_documents(texts)
            else:
                # Fallback to sync in thread pool
                loop = asyncio.get_event_loop()
                if len(texts) == 1:
                    embeddings = [await loop.run_in_executor(
                        None, self.langchain_embeddings.embed_query, texts[0]
                    )]
                else:
                    embeddings = await loop.run_in_executor(
                        None, self.langchain_embeddings.embed_documents, texts
                    )
            
            response_time = time.time() - start_time
            total_tokens = sum(len(text.split()) for text in texts) * 1.3
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=request.model or self.config.model,
                usage=EmbeddingUsage(
                    prompt_tokens=int(total_tokens),
                    total_tokens=int(total_tokens)
                ),
                dimensions=len(embeddings[0]) if embeddings else 0,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Async LangChain embedding generation failed: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def get_embedding_dimension(self, model: str) -> int:
        """Get embedding dimension for a model."""
        # Known dimensions for common models
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "models/embedding-001": 768,
            "models/text-embedding-004": 768,
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
        }
        
        return dimensions_map.get(model, 768)  # Default to 768


def get_langchain_embedding_client(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """
    Factory function to create LangChain-based embedding client.
    
    Args:
        config: Embedding configuration
        
    Returns:
        BaseEmbeddingProvider: Configured embedding provider
    """
    provider = config.provider
    if hasattr(provider, 'value'):
        provider = provider.value
    provider = provider.lower() if isinstance(provider, str) else str(provider).lower()
    
    if provider == "openai" or provider == "embeddingprovider.openai":
        langchain_embeddings = OpenAIEmbeddings(
            model=config.model or "text-embedding-3-small",
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        return LangChainEmbeddingAdapter(langchain_embeddings, config)
        
    elif provider == "google" or provider == "embeddingprovider.google":
        langchain_embeddings = GoogleGenerativeAIEmbeddings(
            model=config.model or "models/embedding-001",
            google_api_key=config.api_key or os.getenv("GOOGLE_API_KEY"),
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        return LangChainEmbeddingAdapter(langchain_embeddings, config)
        
    elif provider == "ollama" or provider == "embeddingprovider.ollama":
        # Use original Ollama provider for special features
        # (model pulling, health checks, instruction formatting)
        return OllamaEmbeddingProvider(config)
        
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_embedding_client(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """
    Backward compatibility wrapper - redirects to LangChain implementation.
    
    Args:
        config: Embedding configuration
        
    Returns:
        BaseEmbeddingProvider: Configured embedding provider
    """
    return get_langchain_embedding_client(config)


def get_embedding_manager(
    configs: Union[List[EmbeddingConfig], List], 
    enable_cache: bool = True
):
    """
    Factory function to create embedding manager with LangChain providers.
    
    Args:
        configs: List of embedding configurations
        enable_cache: Whether to enable caching
        
    Returns:
        EmbeddingManager: Configured embedding manager
    """
    # Import here to avoid circular dependency
    from src.embedding.manager import EmbeddingManager, EmbeddingProviderConfig
    
    # Convert EmbeddingConfig to EmbeddingProviderConfig if needed
    provider_configs = []
    
    for config in configs:
        if isinstance(config, EmbeddingConfig):
            # Create LangChain-based provider
            provider = get_langchain_embedding_client(config)
            
            provider_configs.append(EmbeddingProviderConfig(
                provider=config.provider,
                config=config,
                enabled=True
            ))
        else:
            provider_configs.append(config)
    
    return EmbeddingManager(provider_configs, enable_cache=enable_cache)