"""
Factory functions for creating embedding clients and managers.

This module now redirects to LangChain-based implementations.
"""

# Import from LangChain-based factory
from .factory_langchain import (
    get_langchain_embedding_client as get_embedding_client,
    get_embedding_manager
)

# Re-export for backward compatibility
__all__ = [
    "get_embedding_client",
    "get_embedding_manager"
]