"""
Milvus collection management and vector operations.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pymilvus import SearchResult as PyMilvusSearchResult
from dataclasses import dataclass, field
from datetime import datetime
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import Collection, utility, MilvusException
from pymilvus.client.types import LoadState
from loguru import logger

from src.core.exceptions import MilvusError, CollectionError, VectorError
from src.core.logging import LoggerMixin
from src.milvus.client import MilvusClient
from src.milvus.schema import RAGCollectionSchema


@dataclass
class SearchResult:
    """Search result container."""
    hits: List[Dict[str, Any]]
    total_count: int
    query_time: float
    search_params: Dict[str, Any]
    
    def __post_init__(self):
        self.hit_count = len(self.hits)


@dataclass
class InsertResult:
    """Insert result container."""
    insert_count: int
    primary_keys: List[str]
    timestamp: int
    insert_time: float
    
    def __post_init__(self):
        self.success = self.insert_count > 0


@dataclass
class DeleteResult:
    """Delete result container."""
    delete_count: int
    timestamp: int
    delete_time: float
    
    def __post_init__(self):
        self.success = self.delete_count > 0


@dataclass
class BatchOperationResult:
    """Batch operation result container."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    results: List[Any]
    total_time: float
    
    def __post_init__(self):
        self.success_rate = self.successful_operations / max(1, self.total_operations)


class MilvusCollection(LoggerMixin):
    """
    Milvus collection wrapper with vector operations and batch processing.
    
    Based on Context7 documentation for PyMilvus collection operations:
    - Uses Collection class for data manipulation
    - Implements insert, search, delete, and query operations
    - Supports batch processing for performance
    - Provides comprehensive error handling
    """
    
    def __init__(self, 
                 client: MilvusClient, 
                 schema: RAGCollectionSchema):
        """
        Initialize Milvus collection.
        
        Args:
            client: Milvus client instance
            schema: Collection schema
        """
        self.client = client
        self.schema = schema
        self.collection_name = schema.collection_name
        self._collection: Optional[Collection] = None
        self._lock = threading.Lock()
        self._is_loaded = False
        self._batch_size = 1000
        self._max_workers = 4
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self) -> None:
        """Initialize or get existing collection."""
        try:
            if not self.client.is_connected():
                self.client.connect()
            
            # Check if collection exists
            if self.client.has_collection(self.collection_name):
                self._collection = Collection(
                    name=self.collection_name,
                    using=self.client.alias
                )
                self.logger.info(f"Connected to existing collection: {self.collection_name}")
            else:
                # Create new collection
                collection_schema = self.schema.create_collection_schema()
                self._collection = Collection(
                    name=self.collection_name,
                    schema=collection_schema,
                    using=self.client.alias,
                    consistency_level="Bounded"
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise CollectionError(f"Collection initialization failed: {e}")
    
    def load(self, timeout: Optional[float] = None) -> None:
        """
        Load collection into memory.
        
        Args:
            timeout: Load timeout in seconds
        """
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            # Check if already loaded
            if self.is_loaded():
                self.logger.info(f"Collection {self.collection_name} already loaded")
                return
            
            # Load collection
            self._collection.load()
            
            # Wait for loading to complete
            if timeout:
                utility.wait_for_loading_complete(
                    self.collection_name, 
                    timeout=timeout,
                    using=self.client.alias
                )
            
            self._is_loaded = True
            self.logger.info(f"Collection {self.collection_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load collection: {e}")
            raise CollectionError(f"Load failed: {e}")
    
    def release(self) -> None:
        """Release collection from memory."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            self._collection.release()
            self._is_loaded = False
            self.logger.info(f"Collection {self.collection_name} released")
            
        except Exception as e:
            self.logger.error(f"Failed to release collection: {e}")
            raise CollectionError(f"Release failed: {e}")
    
    def is_loaded(self) -> bool:
        """Check if collection is loaded."""
        try:
            if not self._collection:
                return False
            
            # Check loading state
            load_state = utility.loading_progress(
                self.collection_name,
                using=self.client.alias
            )
            
            return load_state is not None and load_state.get('loading_progress', 0) == 100
            
        except Exception as e:
            self.logger.error(f"Failed to check load status: {e}")
            return False
    
    def insert(self, 
               data: List[Dict[str, Any]], 
               partition_name: Optional[str] = None) -> InsertResult:
        """
        Insert data into collection.
        
        Args:
            data: List of entity dictionaries
            partition_name: Target partition name
            
        Returns:
            InsertResult: Insert operation result
        """
        start_time = time.time()
        
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            if not data:
                raise VectorError("No data provided for insertion")
            
            # Validate data structure
            self._validate_insert_data(data)
            
            # Convert data to column format
            column_data = self._convert_to_column_format(data)
            
            # Perform insert
            if partition_name:
                partition = self._collection.partition(partition_name)
                result = partition.insert(column_data)
            else:
                result = self._collection.insert(column_data)
            
            # Create result
            insert_time = time.time() - start_time
            insert_result = InsertResult(
                insert_count=result.insert_count,
                primary_keys=result.primary_keys,
                timestamp=result.timestamp,
                insert_time=insert_time
            )
            
            self.logger.info(f"Inserted {result.insert_count} entities in {insert_time:.3f}s")
            return insert_result
            
        except Exception as e:
            self.logger.error(f"Insert failed: {e}")
            raise VectorError(f"Insert operation failed: {e}")
    
    def batch_insert(self, 
                    data: List[Dict[str, Any]], 
                    batch_size: Optional[int] = None,
                    partition_name: Optional[str] = None) -> BatchOperationResult:
        """
        Insert data in batches for better performance.
        
        Args:
            data: List of entity dictionaries
            batch_size: Size of each batch
            partition_name: Target partition name
            
        Returns:
            BatchOperationResult: Batch operation result
        """
        start_time = time.time()
        batch_size = batch_size or self._batch_size
        
        try:
            if not data:
                raise VectorError("No data provided for batch insertion")
            
            # Split data into batches
            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            
            results = []
            successful_operations = 0
            failed_operations = 0
            
            # Process batches
            for batch in batches:
                try:
                    result = self.insert(batch, partition_name)
                    results.append(result)
                    successful_operations += 1
                except Exception as e:
                    self.logger.error(f"Batch insert failed: {e}")
                    results.append(None)
                    failed_operations += 1
            
            total_time = time.time() - start_time
            batch_result = BatchOperationResult(
                total_operations=len(batches),
                successful_operations=successful_operations,
                failed_operations=failed_operations,
                results=results,
                total_time=total_time
            )
            
            self.logger.info(f"Batch insert completed: {successful_operations}/{len(batches)} batches successful in {total_time:.3f}s")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            raise VectorError(f"Batch insert operation failed: {e}")
    
    def search(self, 
               query_vectors: List[List[float]], 
               limit: int = 10,
               search_params: Optional[Dict[str, Any]] = None,
               expr: Optional[str] = None,
               output_fields: Optional[List[str]] = None,
               partition_names: Optional[List[str]] = None) -> SearchResult:
        """
        Perform vector similarity search.
        
        Args:
            query_vectors: Query vectors
            limit: Maximum number of results
            search_params: Search parameters
            expr: Boolean expression for filtering
            output_fields: Fields to return
            partition_names: Partitions to search
            
        Returns:
            SearchResult: Search operation result
        """
        start_time = time.time()
        
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            if not self.is_loaded():
                self.load()
            
            if not query_vectors:
                raise VectorError("No query vectors provided")
            
            # Default search parameters
            if search_params is None:
                search_params = {
                    "metric_type": "L2",
                    "params": {"nprobe": 10}
                }
            
            # Default output fields
            if output_fields is None:
                output_fields = ["id", "content", "metadata", "created_at"]
            
            # Perform search
            search_response = self._collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names
            )
            
            # Handle both SearchFuture and direct result cases
            pymilvus_results: Any
            try:
                # Try to call result() method (for SearchFuture)
                pymilvus_results = cast(Any, search_response).result()
            except AttributeError:
                # No result() method, so it's already the result object
                pymilvus_results = search_response
            
            # Convert results to standard format
            hits = []
            # Ensure we can iterate over the results
            if hasattr(pymilvus_results, '__iter__'):
                for result in pymilvus_results:
                    for hit in result:
                        hit_data = {
                            "id": hit.id,
                            "distance": hit.distance,
                            "score": 1.0 / (1.0 + hit.distance),  # Convert distance to similarity score
                            "entity": {}
                        }
                        
                        # Add entity fields
                        if hasattr(hit, 'entity'):
                            for field in output_fields:
                                if hasattr(hit.entity, field):
                                    hit_data["entity"][field] = getattr(hit.entity, field)
                        
                        hits.append(hit_data)
            else:
                self.logger.warning("Search results are not iterable, returning empty results")
            
            query_time = time.time() - start_time
            search_result = SearchResult(
                hits=hits,
                total_count=len(hits),
                query_time=query_time,
                search_params=search_params
            )
            
            self.logger.info(f"Search completed: {len(hits)} hits in {query_time:.3f}s")
            return search_result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise VectorError(f"Search operation failed: {e}")
    
    def delete(self, 
               expr: str, 
               partition_name: Optional[str] = None) -> DeleteResult:
        """
        Delete entities by expression.
        
        Args:
            expr: Boolean expression for deletion
            partition_name: Target partition name
            
        Returns:
            DeleteResult: Delete operation result
        """
        start_time = time.time()
        
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            if not expr:
                raise VectorError("No expression provided for deletion")
            
            # Perform deletion
            if partition_name:
                partition = self._collection.partition(partition_name)
                delete_response = partition.delete(expr)
            else:
                delete_response = self._collection.delete(expr)
            
            # Handle both MutationFuture and direct result cases
            result: Any
            try:
                # Try to call result() method (for MutationFuture)
                result = cast(Any, delete_response).result()
            except AttributeError:
                # No result() method, so it's already the result object
                result = delete_response
            
            delete_time = time.time() - start_time
            delete_result = DeleteResult(
                delete_count=getattr(result, 'delete_count', 0),
                timestamp=getattr(result, 'timestamp', int(time.time() * 1000)),
                delete_time=delete_time
            )
            
            delete_count = getattr(result, 'delete_count', 0)
            self.logger.info(f"Deleted {delete_count} entities in {delete_time:.3f}s")
            return delete_result
            
        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            raise VectorError(f"Delete operation failed: {e}")
    
    def query(self, 
              expr: str, 
              output_fields: Optional[List[str]] = None,
              partition_names: Optional[List[str]] = None,
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query entities by expression.
        
        Args:
            expr: Boolean expression for querying
            output_fields: Fields to return
            partition_names: Partitions to query
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            if not self.is_loaded():
                self.load()
            
            if not expr:
                raise VectorError("No expression provided for query")
            
            # Default output fields
            if output_fields is None:
                output_fields = ["id", "content", "metadata", "created_at"]
            
            # Perform query
            results = self._collection.query(
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names,
                limit=limit
            )
            
            self.logger.info(f"Query completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise VectorError(f"Query operation failed: {e}")
    
    def upsert(self, 
               data: List[Dict[str, Any]], 
               partition_name: Optional[str] = None) -> InsertResult:
        """
        Upsert data into collection.
        
        Args:
            data: List of entity dictionaries
            partition_name: Target partition name
            
        Returns:
            InsertResult: Upsert operation result
        """
        start_time = time.time()
        
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            if not data:
                raise VectorError("No data provided for upsert")
            
            # Validate data structure
            self._validate_insert_data(data)
            
            # Convert data to column format
            column_data = self._convert_to_column_format(data)
            
            # Perform upsert
            if partition_name:
                partition = self._collection.partition(partition_name)
                result = partition.upsert(column_data)
            else:
                result = self._collection.upsert(column_data)
            
            # Create result
            upsert_time = time.time() - start_time
            upsert_result = InsertResult(
                insert_count=result.upsert_count,
                primary_keys=result.primary_keys,
                timestamp=result.timestamp,
                insert_time=upsert_time
            )
            
            self.logger.info(f"Upserted {result.upsert_count} entities in {upsert_time:.3f}s")
            return upsert_result
            
        except Exception as e:
            self.logger.error(f"Upsert failed: {e}")
            raise VectorError(f"Upsert operation failed: {e}")
    
    def get_entity_count(self) -> int:
        """Get number of entities in collection."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            return self._collection.num_entities
            
        except Exception as e:
            self.logger.error(f"Failed to get entity count: {e}")
            raise CollectionError(f"Entity count failed: {e}")
    
    def flush(self) -> None:
        """Flush collection data to disk."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            self._collection.flush()
            self.logger.info(f"Collection {self.collection_name} flushed")
            
        except Exception as e:
            self.logger.error(f"Failed to flush collection: {e}")
            raise CollectionError(f"Flush failed: {e}")
    
    def compact(self) -> None:
        """Compact collection data."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            self._collection.compact()
            self.logger.info(f"Collection {self.collection_name} compacted")
            
        except Exception as e:
            self.logger.error(f"Failed to compact collection: {e}")
            raise CollectionError(f"Compact failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            stats = {
                "name": self.collection_name,
                "num_entities": self.get_entity_count(),
                "is_loaded": self.is_loaded(),
                "schema": self.schema.get_schema_info(),
                "partitions": []
            }
            
            # Get partition information
            for partition in self._collection.partitions:
                partition_stats = {
                    "name": partition.name,
                    "num_entities": partition.num_entities,
                    "is_empty": partition.is_empty
                }
                stats["partitions"].append(partition_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            raise CollectionError(f"Stats collection failed: {e}")
    
    def drop(self) -> None:
        """Drop the collection."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            self._collection.drop()
            self._collection = None
            self.logger.info(f"Collection {self.collection_name} dropped")
            
        except Exception as e:
            self.logger.error(f"Failed to drop collection: {e}")
            raise CollectionError(f"Drop failed: {e}")
    
    def _validate_insert_data(self, data: List[Dict[str, Any]]) -> None:
        """Validate insert data structure."""
        if not data:
            raise VectorError("Empty data provided")
        
        # Check if all required fields are present
        required_fields = {"id", "vector", "content"}
        sample_entity = data[0]
        
        for field in required_fields:
            if field not in sample_entity:
                raise VectorError(f"Missing required field: {field}")
        
        # Validate vector dimensions
        vector_dim = len(sample_entity["vector"])
        if vector_dim != self.schema.vector_dim:
            raise VectorError(f"Vector dimension mismatch: expected {self.schema.vector_dim}, got {vector_dim}")
        
        # Validate all entities have consistent structure
        for i, entity in enumerate(data):
            if "id" not in entity:
                raise VectorError(f"Missing ID in entity {i}")
            if "vector" not in entity:
                raise VectorError(f"Missing vector in entity {i}")
            if len(entity["vector"]) != vector_dim:
                raise VectorError(f"Inconsistent vector dimension in entity {i}")
    
    def _convert_to_column_format(self, data: List[Dict[str, Any]]) -> List[List[Any]]:
        """Convert row format to column format for insertion."""
        if not data:
            return []
        
        # Get field names from schema
        field_names = [field.name for field in self.schema.config.fields]
        
        # Initialize column data
        column_data = []
        
        for field_name in field_names:
            column = []
            for entity in data:
                if field_name in entity:
                    column.append(entity[field_name])
                else:
                    # Provide default values for missing fields
                    if field_name == "created_at":
                        column.append(int(time.time() * 1000))
                    elif field_name == "updated_at":
                        column.append(int(time.time() * 1000))
                    elif field_name == "metadata":
                        column.append({})
                    elif field_name == "permissions":
                        column.append({})
                    elif field_name == "group_ids":
                        column.append([])
                    else:
                        column.append(None)
            
            column_data.append(column)
        
        return column_data
    
    def create_partition(self, partition_name: str, description: str = "") -> None:
        """Create a new partition."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            self._collection.create_partition(partition_name, description)
            self.logger.info(f"Created partition: {partition_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create partition: {e}")
            raise CollectionError(f"Partition creation failed: {e}")
    
    def drop_partition(self, partition_name: str) -> None:
        """Drop a partition."""
        try:
            if not self._collection:
                raise CollectionError("Collection not initialized")
            
            self._collection.drop_partition(partition_name)
            self.logger.info(f"Dropped partition: {partition_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to drop partition: {e}")
            raise CollectionError(f"Partition drop failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        del exc_type, exc_val, exc_tb  # Unused parameters
        if self._is_loaded:
            self.release()


class CollectionManager(LoggerMixin):
    """Manager for multiple Milvus collections."""
    
    def __init__(self, client: MilvusClient):
        """
        Initialize collection manager.
        
        Args:
            client: Milvus client instance
        """
        self.client = client
        self._collections: Dict[str, MilvusCollection] = {}
        self._lock = threading.Lock()
    
    def create_collection(self, schema: RAGCollectionSchema) -> MilvusCollection:
        """
        Create a new collection.
        
        Args:
            schema: Collection schema
            
        Returns:
            MilvusCollection: Created collection
        """
        with self._lock:
            if schema.collection_name in self._collections:
                raise CollectionError(f"Collection already exists: {schema.collection_name}")
            
            collection = MilvusCollection(self.client, schema)
            self._collections[schema.collection_name] = collection
            
            self.logger.info(f"Created collection: {schema.collection_name}")
            return collection
    
    def get_collection(self, collection_name: str) -> Optional[MilvusCollection]:
        """Get collection by name."""
        with self._lock:
            return self._collections.get(collection_name)
    
    def remove_collection(self, collection_name: str) -> None:
        """Remove collection from manager."""
        with self._lock:
            if collection_name in self._collections:
                del self._collections[collection_name]
                self.logger.info(f"Removed collection: {collection_name}")
    
    def list_collections(self) -> List[str]:
        """List all managed collections."""
        with self._lock:
            return list(self._collections.keys())
    
    def get_all_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        with self._lock:
            stats = {}
            for name, collection in self._collections.items():
                try:
                    stats[name] = collection.get_collection_stats()
                except Exception as e:
                    stats[name] = {"error": str(e)}
            return stats
    
    def health_check_all(self) -> Dict[str, bool]:
        """Health check for all collections."""
        with self._lock:
            results = {}
            for name, collection in self._collections.items():
                try:
                    # Simple health check by getting entity count
                    collection.get_entity_count()
                    results[name] = True
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    results[name] = False
            return results
    
    def close_all(self) -> None:
        """Close all collections."""
        with self._lock:
            for collection in self._collections.values():
                try:
                    if collection.is_loaded():
                        collection.release()
                except Exception as e:
                    self.logger.error(f"Error closing collection: {e}")
            
            self._collections.clear()
            self.logger.info("All collections closed")