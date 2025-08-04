"""
Episode RAG Manager - Orchestrates all episode-based RAG components.

This module provides a high-level interface for managing episode data processing,
vector storage, and search operations.
"""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.core.logging import LoggerMixin
from src.core.exceptions import ProcessingError, SearchError, StorageError
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient

from .processor import EpisodeEmbeddingProcessor, EpisodeProcessingConfig
from .vector_store import EpisodeVectorStore, EpisodeVectorStoreConfig
from .search_engine import EpisodeSearchEngine
from .models import (
    EpisodeData, EpisodeSearchRequest, EpisodeSearchResult,
    EpisodeSortOrder, EpisodeProcessingStats
)


@dataclass
class EpisodeRAGConfig:
    """Configuration for Episode RAG Manager."""
    # Processing config
    processing_batch_size: int = 100
    embedding_model: Optional[str] = None  # Use embedding manager's default model
    enable_content_cleaning: bool = True
    
    # Vector store config
    collection_name: str = "episode_embeddings"
    vector_dimension: int = 1536
    index_params: Optional[Dict[str, Any]] = None
    
    # Search config
    default_search_limit: int = 10
    max_search_limit: int = 100
    
    def __post_init__(self):
        if self.index_params is None:
            self.index_params = {"nlist": 1024}


class EpisodeRAGManager(LoggerMixin):
    """
    High-level manager for episode-based RAG operations.
    
    Features:
    - End-to-end episode processing pipeline
    - Unified search interface
    - Component lifecycle management
    - Performance monitoring and statistics
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        embedding_manager: EmbeddingManager,
        milvus_client: MilvusClient,
        config: Optional[EpisodeRAGConfig] = None
    ):
        """
        Initialize Episode RAG Manager.
        
        Args:
            database_manager: Database connection manager
            embedding_manager: Embedding generation manager
            milvus_client: Milvus client instance
            config: RAG configuration
        """
        self.db_manager = database_manager
        self.embedding_manager = embedding_manager
        self.milvus_client = milvus_client
        self.config = config or EpisodeRAGConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Manager statistics
        self.stats = {
            "processed_novels": 0,
            "processed_episodes": 0,
            "total_searches": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        self.logger.info("EpisodeRAGManager initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all RAG components."""
        # Episode processor
        processor_config = EpisodeProcessingConfig(
            batch_size=self.config.processing_batch_size,
            embedding_model=self.config.embedding_model,
            enable_content_cleaning=self.config.enable_content_cleaning
        )
        self.processor = EpisodeEmbeddingProcessor(
            self.db_manager,
            self.embedding_manager,
            processor_config
        )
        
        # Vector store
        vector_store_config = EpisodeVectorStoreConfig(
            collection_name=self.config.collection_name,
            vector_dimension=self.config.vector_dimension,
            index_params=self.config.index_params
        )
        self.vector_store = EpisodeVectorStore(self.milvus_client, vector_store_config)
        
        # Search engine
        self.search_engine = EpisodeSearchEngine(
            self.milvus_client,
            self.embedding_manager,
            self.vector_store
        )
        
        self.logger.info("All RAG components initialized")
    
    async def setup_collection(self, drop_existing: bool = False) -> Dict[str, Any]:
        """
        Set up the episode collection and indexes.
        
        Args:
            drop_existing: Whether to drop existing collection
            
        Returns:
            Setup result with status
        """
        try:
            # Create collection
            self.vector_store.create_collection(drop_existing=drop_existing)
            
            # Create indexes
            self.vector_store.create_indexes()
            
            # Load collection into memory
            self.vector_store.load_collection()
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info()
            
            self.logger.info(f"Episode collection setup completed: {self.config.collection_name}")
            
            return {
                "status": "success",
                "collection_name": self.config.collection_name,
                "collection_info": collection_info,
                "message": "Collection setup completed successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Collection setup failed: {e}")
            self.stats["failed_operations"] += 1
            raise StorageError(f"Collection setup failed: {e}")
    
    async def process_novel(
        self,
        novel_id: int,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process all episodes for a novel.
        
        Args:
            novel_id: Novel ID to process
            force_reprocess: Whether to reprocess existing episodes
            
        Returns:
            Processing result with statistics
        """
        try:
            self.logger.info(f"Starting processing for novel {novel_id}")
            
            # Ensure collection is initialized
            if not hasattr(self.vector_store, 'collection') or self.vector_store.collection is None:
                self.logger.info("Collection not initialized, setting up collection...")
                await self.setup_collection(drop_existing=False)
            
            # Extract episodes from database
            episodes = self.processor.extract_episodes(novel_ids=[novel_id])
            
            if not episodes:
                return {
                    "status": "warning",
                    "novel_id": novel_id,
                    "message": "No episodes found for novel",
                    "episodes_processed": 0
                }
            
            # Process episodes (generate embeddings)
            processed_episodes = await self.processor.process_episodes_async(episodes)
            
            # Store in vector database
            if processed_episodes:
                if force_reprocess:
                    # Update existing episodes
                    storage_result = self.vector_store.update_episodes(processed_episodes)
                else:
                    # Insert new episodes
                    storage_result = self.vector_store.insert_episodes(processed_episodes)
            else:
                storage_result = {"inserted_count": 0}
            
            # Update statistics
            self.stats["processed_novels"] += 1
            self.stats["processed_episodes"] += len(processed_episodes)
            self.stats["successful_operations"] += 1
            
            processing_stats = self.processor.get_processing_stats()
            
            self.logger.info(f"Novel {novel_id} processing completed: {len(processed_episodes)} episodes")
            
            return {
                "status": "success",
                "novel_id": novel_id,
                "episodes_processed": len(processed_episodes),
                "episodes_stored": storage_result.get("inserted_count", 0),
                "processing_stats": processing_stats,
                "storage_result": storage_result
            }
            
        except Exception as e:
            self.logger.error(f"Novel processing failed for {novel_id}: {e}")
            self.stats["failed_operations"] += 1
            raise ProcessingError(f"Novel processing failed: {e}")
    
    async def search_episodes(
        self,
        query: str,
        episode_ids: Optional[List[int]] = None,
        novel_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
        sort_by_episode_number: bool = True,
        **kwargs
    ) -> EpisodeSearchResult:
        """
        Search episodes with simplified interface.
        
        Args:
            query: Search query
            episode_ids: Filter by specific episode IDs
            novel_ids: Filter by specific novel IDs
            limit: Maximum number of results
            sort_by_episode_number: Whether to sort by episode number
            **kwargs: Additional search parameters
            
        Returns:
            Episode search results
        """
        try:
            # Build search request
            request = EpisodeSearchRequest(
                query=query,
                episode_ids=episode_ids,
                novel_ids=novel_ids,
                limit=limit or self.config.default_search_limit,
                sort_order=EpisodeSortOrder.EPISODE_NUMBER if sort_by_episode_number else EpisodeSortOrder.SIMILARITY,
                **kwargs
            )
            
            # Perform search
            result = await self.search_engine.search_async(request)
            
            # Update statistics
            self.stats["total_searches"] += 1
            self.stats["successful_operations"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Episode search failed: {e}")
            self.stats["failed_operations"] += 1
            raise SearchError(f"Episode search failed: {e}")
    
    def get_episode_context(
        self,
        episode_ids: List[int],
        query: Optional[str] = None,
        max_context_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Get episode content as context for LLM.
        
        Args:
            episode_ids: List of episode IDs
            query: Optional query for relevance scoring
            max_context_length: Maximum total character length
            
        Returns:
            Context dictionary with ordered episode content
        """
        try:
            return self.search_engine.get_episode_context(
                episode_ids=episode_ids,
                query=query,
                max_context_length=max_context_length
            )
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            self.stats["failed_operations"] += 1
            raise SearchError(f"Context retrieval failed: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        return {
            "manager_stats": self.stats,
            "processor_stats": self.processor.get_processing_stats(),
            "vector_store_stats": self.vector_store.get_storage_stats(),
            "search_engine_stats": self.search_engine.get_search_stats(),
            "collection_info": self.vector_store.get_collection_info()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Check all components
            vector_store_health = self.vector_store.health_check()
            search_engine_health = self.search_engine.health_check()
            
            # Database connectivity check
            db_healthy = self.db_manager.test_connection()
            
            # Overall status
            all_healthy = (
                vector_store_health.get("status") == "healthy" and
                search_engine_health.get("status") == "healthy" and
                db_healthy
            )
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "components": {
                    "database": "healthy" if db_healthy else "unhealthy",
                    "vector_store": vector_store_health,
                    "search_engine": search_engine_health,
                    "embedding_manager": "healthy"  # Basic check
                },
                "statistics": self.get_manager_stats(),
                "configuration": {
                    "collection_name": self.config.collection_name,
                    "embedding_model": self.config.embedding_model,
                    "vector_dimension": self.config.vector_dimension
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {},
                "statistics": {}
            }
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.vector_store.release_collection()
            self.db_manager.close()
            self.logger.info("EpisodeRAGManager resources cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Factory function for easy initialization
async def create_episode_rag_manager(
    database_manager: DatabaseManager,
    embedding_manager: EmbeddingManager,
    milvus_client: MilvusClient,
    config: Optional[EpisodeRAGConfig] = None,
    setup_collection: bool = True
) -> EpisodeRAGManager:
    """
    Create and initialize an Episode RAG Manager.
    
    Args:
        database_manager: Database connection manager
        embedding_manager: Embedding generation manager
        milvus_client: Milvus client instance
        config: RAG configuration
        setup_collection: Whether to set up the collection
        
    Returns:
        Initialized EpisodeRAGManager
    """
    manager = EpisodeRAGManager(
        database_manager,
        embedding_manager,
        milvus_client,
        config
    )
    
    if setup_collection:
        await manager.setup_collection(drop_existing=False)
    
    return manager