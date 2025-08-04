"""
Episode data models for RAG processing.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum


class EpisodeSortOrder(Enum):
    """Episode sorting order options."""
    SIMILARITY = "similarity"  # Sort by similarity score (default)
    EPISODE_NUMBER = "episode_number"  # Sort by episode number
    PUBLICATION_DATE = "publication_date"  # Sort by publication date
    

@dataclass
class EpisodeChunk:
    """
    Episode chunk data for long episodes that need to be split.
    """
    episode_id: int
    chunk_index: int  # 0-based chunk index
    content: str
    episode_number: int
    episode_title: str
    publication_date: Optional[date]
    novel_id: int
    
    # Original episode data
    total_chunks: int = 1
    
    # Computed fields
    embedding: Optional[List[float]] = None
    content_length: int = field(init=False)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.content_length = len(self.content) if self.content else 0
    
    @property
    def chunk_id(self) -> str:
        """Generate unique chunk ID."""
        return f"{self.episode_id}_{self.chunk_index}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "episode_id": self.episode_id,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "episode_number": self.episode_number,
            "episode_title": self.episode_title,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "novel_id": self.novel_id,
            "total_chunks": self.total_chunks,
            "content_length": self.content_length,
            "embedding": self.embedding
        }


@dataclass
class EpisodeData:
    """
    Episode data container for processing.
    
    Represents a single episode from the RDB with all necessary fields
    for embedding generation and metadata storage.
    """
    episode_id: int
    content: str
    episode_number: int
    episode_title: str
    publication_date: Optional[date]
    novel_id: int
    
    # Computed fields
    embedding: Optional[List[float]] = None
    content_length: int = field(init=False)
    chunks: Optional[List[EpisodeChunk]] = None  # For chunked episodes
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.content_length = len(self.content) if self.content else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "episode_id": self.episode_id,
            "content": self.content,
            "episode_number": self.episode_number,
            "episode_title": self.episode_title,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "novel_id": self.novel_id,
            "content_length": self.content_length,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'EpisodeData':
        """Create EpisodeData from database row."""
        return cls(
            episode_id=row['episode_id'],
            content=row['content'],
            episode_number=row['episode_number'],
            episode_title=row['episode_title'],
            publication_date=row.get('publication_date'),
            novel_id=row['novel_id']
        )


@dataclass
class EpisodeSearchRequest:
    """
    Episode-based search request specification.
    """
    query: str
    episode_ids: Optional[List[int]] = None  # Filter by specific episode IDs
    novel_ids: Optional[List[int]] = None   # Filter by specific novel IDs
    limit: int = 10
    similarity_threshold: Optional[float] = None
    sort_order: EpisodeSortOrder = EpisodeSortOrder.SIMILARITY
    include_content: bool = True
    include_metadata: bool = True
    
    # Date range filtering
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    
    # Episode number range filtering
    episode_num_from: Optional[int] = None
    episode_num_to: Optional[int] = None
    
    def build_filter_expression(self) -> Optional[str]:
        """Build Milvus filter expression from request parameters."""
        filters = []
        
        # Episode ID filtering
        if self.episode_ids:
            episode_filter = f"episode_id in [{','.join(map(str, self.episode_ids))}]"
            filters.append(episode_filter)
        
        # Novel ID filtering
        if self.novel_ids:
            novel_filter = f"novel_id in [{','.join(map(str, self.novel_ids))}]"
            filters.append(novel_filter)
        
        # Episode number range filtering
        if self.episode_num_from is not None:
            filters.append(f"episode_number >= {self.episode_num_from}")
        if self.episode_num_to is not None:
            filters.append(f"episode_number <= {self.episode_num_to}")
        
        # Date range filtering (convert to timestamp for Milvus)
        if self.date_from:
            timestamp_from = int(datetime.combine(self.date_from, datetime.min.time()).timestamp())
            filters.append(f"publication_timestamp >= {timestamp_from}")
        if self.date_to:
            timestamp_to = int(datetime.combine(self.date_to, datetime.max.time()).timestamp())
            filters.append(f"publication_timestamp <= {timestamp_to}")
        
        return " and ".join(filters) if filters else None
    
    def get_output_fields(self) -> List[str]:
        """Get list of fields to return in search results."""
        fields = ["episode_id", "episode_number", "episode_title", "novel_id"]
        
        if self.include_content:
            fields.append("content")
        
        if self.include_metadata:
            fields.extend(["publication_date", "content_length", "publication_timestamp"])
        
        return fields


@dataclass
class EpisodeSearchHit:
    """Individual episode search result hit."""
    episode_id: int
    episode_number: int
    episode_title: str
    novel_id: int
    similarity_score: float
    distance: float
    content: Optional[str] = None
    publication_date: Optional[date] = None
    content_length: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hit to dictionary."""
        return {
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "episode_title": self.episode_title,
            "novel_id": self.novel_id,
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "content": self.content,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "content_length": self.content_length,
            "metadata": self.metadata
        }


@dataclass
class EpisodeSearchResult:
    """Episode search result container."""
    hits: List[EpisodeSearchHit]
    total_count: int
    search_time: float
    query: str
    sort_order: EpisodeSortOrder
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_sorted_hits(self) -> List[EpisodeSearchHit]:
        """Get hits sorted according to the specified sort order."""
        if self.sort_order == EpisodeSortOrder.SIMILARITY:
            return sorted(self.hits, key=lambda h: h.similarity_score, reverse=True)
        elif self.sort_order == EpisodeSortOrder.EPISODE_NUMBER:
            return sorted(self.hits, key=lambda h: h.episode_number)
        elif self.sort_order == EpisodeSortOrder.PUBLICATION_DATE:
            return sorted(self.hits, key=lambda h: h.publication_date or date.min)
        else:
            return self.hits
    
    def get_context_text(self, separator: str = "\n\n---\n\n") -> str:
        """Get concatenated content from all hits as context."""
        sorted_hits = self.get_sorted_hits()
        contexts = []
        
        for hit in sorted_hits:
            if hit.content:
                context = f"Episode {hit.episode_number}: {hit.episode_title}\n{hit.content}"
                contexts.append(context)
        
        return separator.join(contexts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "total_count": self.total_count,
            "search_time": self.search_time,
            "query": self.query,
            "sort_order": self.sort_order.value,
            "metadata": self.metadata
        }


@dataclass
class EpisodeProcessingStats:
    """Statistics for episode processing operations."""
    total_episodes: int = 0
    processed_episodes: int = 0
    failed_episodes: int = 0
    total_processing_time: float = 0.0
    average_content_length: float = 0.0
    embedding_generation_time: float = 0.0
    storage_time: float = 0.0
    
    def get_success_rate(self) -> float:
        """Get processing success rate as percentage."""
        if self.total_episodes == 0:
            return 100.0
        return (self.processed_episodes / self.total_episodes) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "processed_episodes": self.processed_episodes,
            "failed_episodes": self.failed_episodes,
            "success_rate": self.get_success_rate(),
            "total_processing_time": self.total_processing_time,
            "average_content_length": self.average_content_length,
            "embedding_generation_time": self.embedding_generation_time,
            "storage_time": self.storage_time
        }