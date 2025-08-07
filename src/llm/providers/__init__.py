"""
LLM Providers - Now using LangChain integrations.

Most providers are now handled through LangChain.
Only Ollama provider is kept for its special features.
"""

from .ollama import OllamaProvider

# Note: OpenAI, Gemini, and Claude providers have been removed
# Use LangChain integration through LLMManager instead

__all__ = [
    "OllamaProvider",
]