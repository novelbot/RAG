"""
Episode-based RAG processing module.

This module provides functionality for processing episode data from RDB,
generating embeddings, and managing episode-based vector search capabilities.
"""

from .processor import EpisodeEmbeddingProcessor
from .vector_store import EpisodeVectorStore
from .search_engine import EpisodeSearchEngine
from .manager import EpisodeRAGManager, EpisodeRAGConfig, create_episode_rag_manager
from .models import (
    EpisodeData, EpisodeSearchRequest, EpisodeSearchResult,
    EpisodeSortOrder, EpisodeProcessingStats
)

__all__ = [
    'EpisodeEmbeddingProcessor',
    'EpisodeVectorStore', 
    'EpisodeSearchEngine',
    'EpisodeRAGManager',
    'EpisodeRAGConfig',
    'create_episode_rag_manager',
    'EpisodeData',
    'EpisodeSearchRequest',
    'EpisodeSearchResult',
    'EpisodeSortOrder',
    'EpisodeProcessingStats'
]