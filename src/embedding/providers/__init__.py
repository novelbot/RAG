"""
Embedding Providers - Now using LangChain integrations.

Most providers are now handled through LangChain.
Only Ollama provider is kept for its special features.
"""

from .ollama import OllamaEmbeddingProvider

# Note: OpenAI and Google providers have been removed
# Use LangChain integration through get_embedding_client() instead

__all__ = [
    "OllamaEmbeddingProvider",
]