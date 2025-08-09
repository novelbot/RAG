"""
LangChain-based Embedding Provider Implementation.

This module provides unified embedding providers using LangChain's embedding interfaces,
supporting multiple providers (OpenAI, Google, Ollama, etc.) through a single interface.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os

# LangChain embedding imports
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.embedding.base import (
    BaseEmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage, EmbeddingDimension
)
from src.embedding.types import EmbeddingConfig, EmbeddingProvider
from src.core.exceptions import EmbeddingError, ConfigurationError
from src.core.logging import LoggerMixin


@dataclass
class LangChainEmbeddingConfig:
    """Configuration for LangChain-based embedding providers."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: Optional[int] = None
    batch_size: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    
    # Provider-specific configurations
    google_project_id: Optional[str] = None
    task_type: Optional[str] = None  # For Google embeddings
    encoding_format: Optional[str] = None  # For OpenAI


class LangChainEmbeddingProvider(BaseEmbeddingProvider, LoggerMixin):
    """
    Unified embedding provider using LangChain integrations.
    
    Features:
    - Support for multiple embedding providers (OpenAI, Google, Ollama, HuggingFace)
    - Async/batch processing capabilities
    - Automatic dimension detection
    - Consistent interface across providers
    - Built-in retry and error handling
    """
    
    # Model dimension mappings
    MODEL_DIMENSIONS = {
        # OpenAI models
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        
        # Google models
        "models/embedding-001": 768,
        "models/text-embedding-004": 768,
        
        # Ollama models
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "bge-m3": 768,
        
        # HuggingFace models
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    
    # Model max token limits
    MODEL_MAX_TOKENS = {
        # OpenAI models - all support 8191 tokens
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
        
        # Google models
        "models/embedding-001": 2048,
        "models/text-embedding-004": 2048,
        
        # Ollama models
        "nomic-embed-text": 8192,
        "mxbai-embed-large": 512,
        "all-minilm": 256,
        "bge-m3": 8192,
        
        # HuggingFace models
        "sentence-transformers/all-MiniLM-L6-v2": 512,
        "sentence-transformers/all-mpnet-base-v2": 512,
        "BAAI/bge-large-en-v1.5": 512,
    }
    
    def __init__(self, config: Union[EmbeddingConfig, LangChainEmbeddingConfig]):
        """Initialize LangChain embedding provider."""
        if isinstance(config, EmbeddingConfig):
            # Convert old config to new format
            self.config = self._convert_config(config)
        else:
            self.config = config
            
        self.embeddings_client: Optional[Embeddings] = None
        self._initialize_provider()
        
        # Set dimensions based on model
        if not self.config.dimensions:
            self.config.dimensions = self.MODEL_DIMENSIONS.get(
                self.config.model, 
                768  # Default dimension
            )
    
    def _convert_config(self, old_config: EmbeddingConfig) -> LangChainEmbeddingConfig:
        """Convert old EmbeddingConfig to LangChainEmbeddingConfig."""
        provider_value = old_config.provider
        if hasattr(provider_value, 'value'):
            provider_value = provider_value.value
            
        return LangChainEmbeddingConfig(
            provider=provider_value,
            model=old_config.model,
            api_key=old_config.api_key,
            base_url=old_config.base_url,
            dimensions=old_config.dimensions,
            batch_size=old_config.batch_size,
            timeout=old_config.timeout,
            retry_attempts=old_config.retry_attempts
        )
    
    def _initialize_provider(self) -> None:
        """Initialize the appropriate LangChain embedding provider."""
        provider = self.config.provider.lower()
        
        try:
            if provider == "openai":
                self._initialize_openai()
            elif provider in ["google", "gemini"]:
                self._initialize_google()
            elif provider == "ollama":
                self._initialize_ollama()
            elif provider == "huggingface":
                self._initialize_huggingface()
            else:
                raise ConfigurationError(f"Unsupported provider: {provider}")
                
            self.logger.info(f"Initialized LangChain embedding provider: {provider} with model: {self.config.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize provider {provider}: {e}")
            raise EmbeddingError(f"Provider initialization failed: {e}")
    
    def _initialize_openai(self) -> None:
        """Initialize OpenAI embeddings."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OpenAI API key not provided")
        
        kwargs = {
            "model": self.config.model or "text-embedding-3-small",
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.retry_attempts,
        }
        
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
            
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions
            
        self.embeddings_client = OpenAIEmbeddings(**kwargs)
    
    def _initialize_google(self) -> None:
        """Initialize Google Generative AI embeddings."""
        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError("Google API key not provided")
        
        self.embeddings_client = GoogleGenerativeAIEmbeddings(
            model=self.config.model or "models/embedding-001",
            google_api_key=api_key,
            task_type=self.config.task_type or "retrieval_document",
            timeout=self.config.timeout,
            max_retries=self.config.retry_attempts
        )
    
    def _initialize_ollama(self) -> None:
        """Initialize Ollama embeddings."""
        base_url = self.config.base_url or "http://localhost:11434"
        
        self.embeddings_client = OllamaEmbeddings(
            model=self.config.model or "nomic-embed-text",
            base_url=base_url,
            timeout=self.config.timeout
        )
    
    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace embeddings."""
        model_name = self.config.model or "sentence-transformers/all-MiniLM-L6-v2"
        
        model_kwargs = {}
        if self.config.api_key:
            model_kwargs["use_auth_token"] = self.config.api_key
            
        encode_kwargs = {
            "normalize_embeddings": True,
            "batch_size": self.config.batch_size
        }
        
        self.embeddings_client = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings synchronously using LangChain.
        
        Args:
            request: Embedding request with texts
            
        Returns:
            Embedding response with vectors
        """
        start_time = time.time()
        
        try:
            # Process in batches if needed
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(request.texts), self.config.batch_size):
                batch = request.texts[i:i + self.config.batch_size]
                
                # Generate embeddings using LangChain
                if len(batch) == 1:
                    # Single text
                    embedding = self.embeddings_client.embed_query(batch[0])
                    batch_embeddings = [embedding]
                else:
                    # Multiple texts
                    batch_embeddings = self.embeddings_client.embed_documents(batch)
                
                all_embeddings.extend(batch_embeddings)
                
                # Estimate token usage (rough approximation)
                for text in batch:
                    total_tokens += len(text.split()) * 1.3
            
            response_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=all_embeddings,
                model=self.config.model,
                dimensions=len(all_embeddings[0]) if all_embeddings else 0,
                usage=EmbeddingUsage(
                    prompt_tokens=int(total_tokens),
                    total_tokens=int(total_tokens)
                ),
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    async def embed_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings asynchronously using LangChain.
        
        Args:
            request: Embedding request with texts
            
        Returns:
            Embedding response with vectors
        """
        start_time = time.time()
        
        try:
            # Process in batches
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(request.texts), self.config.batch_size):
                batch = request.texts[i:i + self.config.batch_size]
                
                # Generate embeddings using async methods if available
                if hasattr(self.embeddings_client, 'aembed_documents'):
                    if len(batch) == 1:
                        embedding = await self.embeddings_client.aembed_query(batch[0])
                        batch_embeddings = [embedding]
                    else:
                        batch_embeddings = await self.embeddings_client.aembed_documents(batch)
                else:
                    # Fallback to sync in thread pool
                    loop = asyncio.get_event_loop()
                    if len(batch) == 1:
                        embedding = await loop.run_in_executor(
                            None, self.embeddings_client.embed_query, batch[0]
                        )
                        batch_embeddings = [embedding]
                    else:
                        batch_embeddings = await loop.run_in_executor(
                            None, self.embeddings_client.embed_documents, batch
                        )
                
                all_embeddings.extend(batch_embeddings)
                
                # Estimate token usage
                for text in batch:
                    total_tokens += len(text.split()) * 1.3
            
            response_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=all_embeddings,
                model=self.config.model,
                dimensions=len(all_embeddings[0]) if all_embeddings else 0,
                usage=EmbeddingUsage(
                    prompt_tokens=int(total_tokens),
                    total_tokens=int(total_tokens)
                ),
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Async embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "dimensions": self.config.dimensions,
            "max_tokens": self.MODEL_MAX_TOKENS.get(self.config.model, 2048),
            "max_batch_size": self.config.batch_size,
            "supports_async": hasattr(self.embeddings_client, 'aembed_documents'),
            "langchain_integration": True,
            "features": [
                "batch_processing",
                "async_generation",
                "multi_provider_support",
                "automatic_retry",
                "dimension_detection"
            ]
        }
    
    def validate_dimensions(self, embeddings: List[List[float]]) -> bool:
        """Validate embedding dimensions."""
        if not embeddings:
            return True
            
        expected_dim = self.config.dimensions
        actual_dim = len(embeddings[0])
        
        if expected_dim and actual_dim != expected_dim:
            self.logger.warning(
                f"Dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
            return False
            
        # Check consistency across all embeddings
        return all(len(emb) == actual_dim for emb in embeddings)


def create_langchain_embeddings(config: LangChainEmbeddingConfig) -> LangChainEmbeddingProvider:
    """Factory function to create LangChain embedding provider."""
    return LangChainEmbeddingProvider(config)