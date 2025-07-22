"""
Text Processing and Chunking Module.

This module provides comprehensive text preprocessing and intelligent chunking
capabilities optimized for RAG applications, including text cleaning,
normalization, and multiple chunking strategies.
"""

from .text_cleaner import (
    TextCleaner,
    CleaningConfig,
    CleaningRule,
    CleaningResult
)
from .text_splitter import (
    TextSplitter,
    ChunkingStrategy,
    ChunkingConfig,
    ChunkResult,
    FixedSizeChunker,
    SemanticChunker,
    SentenceBasedChunker
)
from .metadata_manager import (
    MetadataManager,
    ChunkMetadata,
    MetadataConfig
)
from .exceptions import (
    TextProcessingError,
    ChunkingError,
    MetadataError,
    InvalidConfigurationError
)

__all__ = [
    # Text cleaning
    "TextCleaner",
    "CleaningConfig", 
    "CleaningRule",
    "CleaningResult",
    
    # Text splitting
    "TextSplitter",
    "ChunkingStrategy",
    "ChunkingConfig",
    "ChunkResult",
    "FixedSizeChunker",
    "SemanticChunker", 
    "SentenceBasedChunker",
    
    # Metadata management
    "MetadataManager",
    "ChunkMetadata",
    "MetadataConfig",
    
    # Exceptions
    "TextProcessingError",
    "ChunkingError",
    "MetadataError",
    "InvalidConfigurationError"
]