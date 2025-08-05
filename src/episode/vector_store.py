"""
Episode Vector Store for Milvus Integration.

This module handles the storage and management of episode embeddings in Milvus,
including schema definition, indexing, and data operations.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date, timezone
from dataclasses import dataclass

from src.core.logging import LoggerMixin
from src.core.exceptions import StorageError, ConfigurationError
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection
from src.milvus.schema import create_field, DataType, CollectionSchema
from src.milvus.index import IndexType, MetricType
from .models import EpisodeData, EpisodeProcessingStats


@dataclass
class EpisodeVectorStoreConfig:
    """Configuration for episode vector store."""
    collection_name: str = "episode_embeddings"
    vector_dimension: int = 1536  # OpenAI ada-002 dimension
    index_type: IndexType = IndexType.IVF_FLAT
    metric_type: MetricType = MetricType.L2
    index_params: Optional[Dict[str, Any]] = None
    enable_dynamic_schema: bool = False
    shard_num: int = 2
    replica_num: int = 1
    
    def __post_init__(self):
        """Initialize default index parameters."""
        if self.index_params is None:
            self.index_params = {"nlist": 1024}


class EpisodeVectorStore(LoggerMixin):
    """
    Vector store for episode embeddings using Milvus.
    
    Features:
    - Optimized schema for episode metadata
    - Efficient indexing for episode-based searches
    - Batch insertion and updates
    - Metadata filtering support
    - Collection management
    """
    
    def __init__(
        self,
        milvus_client: MilvusClient,
        config: Optional[EpisodeVectorStoreConfig] = None
    ):
        """
        Initialize Episode Vector Store.
        
        Args:
            milvus_client: Milvus client instance
            config: Vector store configuration
        """
        self.milvus_client = milvus_client
        self.config = config or EpisodeVectorStoreConfig()
        from pymilvus import Collection
        self.collection: Optional[Collection] = None
        
        # Storage statistics
        self.stats = EpisodeProcessingStats()
        
        self.logger.info(f"EpisodeVectorStore initialized for collection '{self.config.collection_name}'")
    
    def create_collection(self, drop_existing: bool = False) -> None:
        """
        Create Milvus collection with episode schema.
        
        Args:
            drop_existing: Whether to drop existing collection
        """
        try:
            # Check if collection exists
            if self.milvus_client.has_collection(self.config.collection_name):
                if drop_existing:
                    self.logger.warning(f"Dropping existing collection '{self.config.collection_name}'")
                    self.milvus_client.drop_collection(self.config.collection_name)
                else:
                    self.logger.info(f"Collection '{self.config.collection_name}' already exists")
                    self.collection = self.milvus_client.get_collection(self.config.collection_name)
                    return
            
            # Define schema fields
            fields = [
                # Primary key - now uses a unique ID for each entry (episode or chunk)
                create_field(
                    name="entry_id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    description="Unique entry ID (auto-generated)"
                ),
                
                # Original RDB episode ID (for traceability)
                create_field(
                    name="episode_id",
                    dtype=DataType.INT64,
                    description="Original Episode ID from RDB"
                ),
                
                # Vector field
                create_field(
                    name="content_embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.config.vector_dimension,
                    description="Episode content embedding"
                ),
                
                # Metadata fields
                create_field(
                    name="novel_id",
                    dtype=DataType.INT64,
                    description="Novel ID from RDB"
                ),
                
                create_field(
                    name="episode_number",
                    dtype=DataType.INT32,
                    description="Episode number within novel"
                ),
                
                create_field(
                    name="episode_title",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Episode title"
                ),
                
                create_field(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=20000,  # Milvus VARCHAR max length
                    description="Episode content"
                ),
                
                create_field(
                    name="content_length",
                    dtype=DataType.INT32,
                    description="Content character count"
                ),
                
                # Chunk tracking fields
                create_field(
                    name="is_chunk",
                    dtype=DataType.BOOL,
                    description="Whether this entry is a chunk (True) or full episode (False)"
                ),
                
                create_field(
                    name="chunk_index",
                    dtype=DataType.INT32,
                    description="Chunk index (0-based, -1 for non-chunks)"
                ),
                
                create_field(
                    name="total_chunks",
                    dtype=DataType.INT32,
                    description="Total number of chunks for this episode (1 for non-chunks)"
                ),
                
                # Date as timestamp for filtering
                create_field(
                    name="publication_timestamp",
                    dtype=DataType.INT64,
                    description="Publication date as Unix timestamp"
                ),
                
                create_field(
                    name="publication_date",
                    dtype=DataType.VARCHAR,
                    max_length=20,
                    description="Publication date as ISO string"
                ),
                
                # Processing metadata
                create_field(
                    name="created_at",
                    dtype=DataType.INT64,
                    description="Creation timestamp"
                ),
                
                create_field(
                    name="updated_at",
                    dtype=DataType.INT64,
                    description="Last update timestamp"
                )
            ]
            
            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description=f"Episode embeddings for RAG search",
                enable_dynamic_field=self.config.enable_dynamic_schema
            )
            
            # Create collection
            self.collection = self.milvus_client.create_collection(
                name=self.config.collection_name,
                schema=schema
            )
            
            self.logger.info(f"Created collection '{self.config.collection_name}' with {len(fields)} fields")
            
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise StorageError(f"Collection creation failed: {e}")
    
    def create_indexes(self) -> None:
        """Create indexes on the collection for optimal search performance."""
        if not self.collection:
            raise StorageError("Collection not initialized")
        
        try:
            # Vector index
            vector_index_params = {
                "index_type": self.config.index_type.value,
                "metric_type": self.config.metric_type.value,
                "params": self.config.index_params
            }
            
            self.collection.create_index(
                field_name="content_embedding",
                index_params=vector_index_params
            )
            
            # Note: Milvus does not support scalar field indexes like BTREE
            # Scalar fields can be used for filtering via expr parameter in search/query
            # but cannot have dedicated indexes for performance optimization
            self.logger.info("Vector index created. Scalar fields available for filtering without indexes")
            
            self.logger.info("Created indexes on collection")
            
        except Exception as e:
            self.logger.error(f"Index creation failed: {e}")
            raise StorageError(f"Index creation failed: {e}")
    
    def insert_episodes(self, episodes: List[EpisodeData]) -> Dict[str, Any]:
        """
        Insert episode embeddings into the collection.
        Handles both regular episodes and chunked episodes.
        
        Args:
            episodes: List of episodes with embeddings
            
        Returns:
            Insertion result with statistics
        """
        if not self.collection:
            raise StorageError("Collection not initialized")
        
        if not episodes:
            return {"inserted_count": 0, "message": "No episodes to insert"}
        
        start_time = time.time()
        total_inserted = 0
        
        try:
            # Separate episodes into regular and chunked
            regular_episodes = []
            all_chunks = []
            
            for episode in episodes:
                if episode.chunks:
                    # Episode is chunked - insert chunks individually
                    for chunk in episode.chunks:
                        if chunk.embedding:  # Only insert chunks with embeddings
                            all_chunks.append(chunk)
                else:
                    # Regular episode - insert as is
                    if episode.embedding:  # Only insert episodes with embeddings
                        regular_episodes.append(episode)
            
            # Insert regular episodes
            if regular_episodes:
                data = self._prepare_insertion_data(regular_episodes)
                data_list = [data[field] for field in [
                    "entry_id", "episode_id", "content_embedding", "novel_id", "episode_number", 
                    "episode_title", "content", "content_length", "is_chunk", "chunk_index", 
                    "total_chunks", "publication_timestamp", "publication_date", "created_at", "updated_at"
                ]]
                insert_result = self.collection.insert(data_list)
                total_inserted += len(regular_episodes)
                self.logger.info(f"Inserted {len(regular_episodes)} regular episodes")
            
            # Insert chunks
            if all_chunks:
                chunk_data = self._prepare_chunk_insertion_data(all_chunks)
                chunk_data_list = [chunk_data[field] for field in [
                    "entry_id", "episode_id", "content_embedding", "novel_id", "episode_number", 
                    "episode_title", "content", "content_length", "is_chunk", "chunk_index", 
                    "total_chunks", "publication_timestamp", "publication_date", "created_at", "updated_at"
                ]]
                chunk_insert_result = self.collection.insert(chunk_data_list)
                total_inserted += len(all_chunks)
                self.logger.info(f"Inserted {len(all_chunks)} episode chunks")
            
            insert_result = {"inserted_count": total_inserted}
            
            # Update statistics
            inserted_count = len(episodes)
            self.stats.processed_episodes += inserted_count
            self.stats.storage_time += time.time() - start_time
            
            self.logger.info(f"Inserted {inserted_count} episodes into collection")
            
            return {
                "inserted_count": inserted_count,
                "insert_ids": insert_result.primary_keys if hasattr(insert_result, 'primary_keys') else [],
                "storage_time": time.time() - start_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Episode insertion failed: {e}")
            self.stats.failed_episodes += len(episodes)
            raise StorageError(f"Episode insertion failed: {e}")
    
    def update_episodes(self, episodes: List[EpisodeData]) -> Dict[str, Any]:
        """
        Update existing episode embeddings.
        
        Args:
            episodes: List of episodes to update
            
        Returns:
            Update result with statistics
        """
        if not self.collection:
            raise StorageError("Collection not initialized")
        
        # For now, implement as delete + insert
        # Milvus upsert might be available in newer versions
        start_time = time.time()
        
        try:
            # Delete existing episodes
            episode_ids = [ep.episode_id for ep in episodes]
            delete_expr = f"episode_id in [{','.join(map(str, episode_ids))}]"
            self.collection.delete(delete_expr)
            
            # Insert updated episodes
            result = self.insert_episodes(episodes)
            result["operation"] = "update"
            result["updated_count"] = result["inserted_count"]
            
            self.logger.info(f"Updated {len(episodes)} episodes")
            return result
            
        except Exception as e:
            self.logger.error(f"Episode update failed: {e}")
            raise StorageError(f"Episode update failed: {e}")
    
    def delete_episodes(self, episode_ids: List[int]) -> Dict[str, Any]:
        """
        Delete episodes by IDs.
        
        Args:
            episode_ids: List of episode IDs to delete
            
        Returns:
            Deletion result
        """
        if not self.collection:
            raise StorageError("Collection not initialized")
        
        if not episode_ids:
            return {"deleted_count": 0, "message": "No episodes to delete"}
        
        try:
            # Build delete expression
            delete_expr = f"episode_id in [{','.join(map(str, episode_ids))}]"
            
            # Execute deletion
            delete_result = self.collection.delete(delete_expr)
            
            self.logger.info(f"Deleted {len(episode_ids)} episodes")
            
            return {
                "deleted_count": len(episode_ids),
                "delete_expr": delete_expr,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Episode deletion failed: {e}")
            raise StorageError(f"Episode deletion failed: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            # Get basic info
            info = {
                "name": self.config.collection_name,
                "schema": self.collection.schema.to_dict() if hasattr(self.collection.schema, 'to_dict') else str(self.collection.schema),
                "num_entities": self.collection.num_entities,
                "indexes": []
            }
            
            # Get index information
            try:
                for index in self.collection.indexes:
                    index_info = {
                        "field_name": index.field_name,
                        "index_type": getattr(index, 'index_type', 'unknown'),
                        "params": getattr(index, 'params', {})
                    }
                    info["indexes"].append(index_info)
            except Exception as e:
                self.logger.warning(f"Failed to get index info: {e}")
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def load_collection(self) -> None:
        """Load collection into memory for searching."""
        if not self.collection:
            raise StorageError("Collection not initialized")
        
        try:
            self.collection.load()
            self.logger.info(f"Loaded collection '{self.config.collection_name}' into memory")
        except Exception as e:
            self.logger.error(f"Failed to load collection: {e}")
            raise StorageError(f"Collection loading failed: {e}")
    
    def release_collection(self) -> None:
        """Release collection from memory."""
        if not self.collection:
            return
        
        try:
            self.collection.release()
            self.logger.info(f"Released collection '{self.config.collection_name}' from memory")
        except Exception as e:
            self.logger.warning(f"Failed to release collection: {e}")
    
    def _prepare_insertion_data(self, episodes: List[EpisodeData]) -> Dict[str, List[Any]]:
        """Prepare episode data for Milvus insertion."""
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        
        data = {
            "entry_id": [],      # New primary key (auto-generated)
            "episode_id": [],    # Original RDB episode ID for traceability
            "content_embedding": [],
            "novel_id": [],
            "episode_number": [],
            "episode_title": [],
            "content": [],
            "content_length": [],
            "is_chunk": [],      # False for regular episodes
            "chunk_index": [],   # -1 for regular episodes
            "total_chunks": [],  # 1 for regular episodes
            "publication_timestamp": [],
            "publication_date": [],
            "created_at": [],
            "updated_at": []
        }
        
        for i, episode in enumerate(episodes):
            if not episode.embedding:
                raise StorageError(f"Episode {episode.episode_id} has no embedding")
            
            # Generate unique entry_id (primary key)
            # Use negative values for regular episodes to avoid conflicts with chunks
            entry_id = -(episode.episode_id * 1000 + 999)  # Ensures uniqueness and negative
            
            data["entry_id"].append(entry_id)
            data["episode_id"].append(episode.episode_id)  # Preserve original RDB ID
            data["content_embedding"].append(episode.embedding)
            data["novel_id"].append(episode.novel_id)
            data["episode_number"].append(episode.episode_number)
            data["episode_title"].append(episode.episode_title[:500])  # Truncate if too long
            data["content"].append(episode.content[:20000])  # Truncate if too long
            data["content_length"].append(episode.content_length)
            
            # Chunk tracking fields for regular episodes
            data["is_chunk"].append(False)  # Not a chunk
            data["chunk_index"].append(-1)   # No chunk index
            data["total_chunks"].append(1)   # Single piece
            
            # Handle publication date
            if episode.publication_date:
                pub_timestamp = int(datetime.combine(episode.publication_date, datetime.min.time()).timestamp())
                pub_date_str = episode.publication_date.isoformat()
            else:
                pub_timestamp = 0
                pub_date_str = ""
            
            data["publication_timestamp"].append(pub_timestamp)
            data["publication_date"].append(pub_date_str)
            data["created_at"].append(current_timestamp)
            data["updated_at"].append(current_timestamp)
        
        return data
    
    def _prepare_chunk_insertion_data(self, chunks) -> Dict[str, List[Any]]:
        """Prepare chunk data for Milvus insertion."""
        from .models import EpisodeChunk
        
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        
        data = {
            "entry_id": [],      # New primary key (auto-generated)
            "episode_id": [],    # Original RDB episode ID for traceability
            "content_embedding": [],
            "novel_id": [],
            "episode_number": [],
            "episode_title": [],
            "content": [],
            "content_length": [],
            "is_chunk": [],      # True for chunks
            "chunk_index": [],   # Chunk index (0-based)
            "total_chunks": [],  # Total chunks for this episode
            "publication_timestamp": [],
            "publication_date": [],
            "created_at": [],
            "updated_at": []
        }
        
        for chunk in chunks:
            if not chunk.embedding:
                raise StorageError(f"Chunk {chunk.chunk_id} has no embedding")
            
            # Generate unique entry_id for this chunk
            # Use positive values: episode_id * 1000 + chunk_index
            entry_id = chunk.episode_id * 1000 + chunk.chunk_index
            
            data["entry_id"].append(entry_id)
            data["episode_id"].append(chunk.episode_id)  # Preserve original RDB episode ID
            data["content_embedding"].append(chunk.embedding)
            data["novel_id"].append(chunk.novel_id)
            data["episode_number"].append(chunk.episode_number)
            
            # Add chunk info to title
            chunk_title = f"{chunk.episode_title} [Chunk {chunk.chunk_index+1}/{chunk.total_chunks}]"
            data["episode_title"].append(chunk_title[:500])  # Truncate if too long
            data["content"].append(chunk.content[:20000])  # Truncate if too long
            data["content_length"].append(chunk.content_length)
            
            # Chunk tracking fields
            data["is_chunk"].append(True)  # This is a chunk
            data["chunk_index"].append(chunk.chunk_index)  # 0-based chunk index
            data["total_chunks"].append(chunk.total_chunks)  # Total chunks for episode
            
            # Handle publication date
            if chunk.publication_date:
                pub_timestamp = int(chunk.publication_date.strftime('%s'))
                pub_date_str = chunk.publication_date.isoformat()
            else:
                pub_timestamp = current_timestamp
                pub_date_str = datetime.now(timezone.utc).date().isoformat()
            
            data["publication_timestamp"].append(pub_timestamp)
            data["publication_date"].append(pub_date_str)
            data["created_at"].append(current_timestamp)
            data["updated_at"].append(current_timestamp)
        
        return data
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.stats.to_dict()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            if not self.collection:
                return {
                    "status": "unhealthy",
                    "error": "Collection not initialized"
                }
            
            # Test basic operations
            num_entities = self.collection.num_entities
            
            return {
                "status": "healthy",
                "collection_name": self.config.collection_name,
                "num_entities": num_entities,
                "is_loaded": True,  # Assume loaded if collection exists
                "config": {
                    "vector_dimension": self.config.vector_dimension,
                    "index_type": self.config.index_type.value,
                    "metric_type": self.config.metric_type.value
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release_collection()