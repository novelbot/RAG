"""
Embedding types and enums - circular import safe module.

This module contains basic types and enums that can be imported 
without creating circular dependencies.
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: EmbeddingProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    normalize_embeddings: bool = True
    
    # Provider-specific configurations
    extra_headers: Optional[Dict[str, str]] = None
    extra_query: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Ensure provider is an enum instance
        if isinstance(self.provider, str):
            try:
                self.provider = EmbeddingProvider(self.provider.lower())
            except ValueError:
                raise ValueError(f"Unknown embedding provider: {self.provider}")
        
        # Provider-specific validation and defaults
        if self.provider == EmbeddingProvider.OPENAI and not self.api_key:
            raise ValueError("OpenAI API key is required")
        if self.provider == EmbeddingProvider.GOOGLE and not self.api_key:
            raise ValueError("Google API key is required")
        
        # Set provider-specific base URLs
        if self.provider == EmbeddingProvider.OLLAMA and not self.base_url:
            self.base_url = "http://localhost:11434"
        elif self.provider == EmbeddingProvider.GOOGLE:
            # Google API uses default endpoint, don't set base_url
            self.base_url = None
        elif self.provider == EmbeddingProvider.OPENAI:
            # OpenAI API uses default endpoint, don't set base_url
            self.base_url = None