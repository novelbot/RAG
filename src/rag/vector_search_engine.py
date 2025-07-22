"""
Vector Similarity Search Engine for RAG System.

This module provides comprehensive vector search capabilities with Milvus integration,
including configurable search parameters, distance metrics, and optimized search algorithms.
"""

import time
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

import numpy as np

from src.core.logging import LoggerMixin
from src.core.exceptions import SearchError, ConfigurationError
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection
from src.milvus.search import SearchManager, SearchQuery, SearchStrategy
from src.milvus.index import MetricType


class SearchMode(Enum):
    """Search modes for different use cases."""
    PRECISION = "precision"      # High precision, slower
    BALANCED = "balanced"        # Balanced precision/speed
    SPEED = "speed"             # High speed, lower precision
    ADAPTIVE = "adaptive"       # Adaptive based on query


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    L2 = "L2"                   # Euclidean distance
    IP = "IP"                   # Inner product (cosine for normalized vectors)
    COSINE = "COSINE"           # Cosine similarity
    HAMMING = "HAMMING"         # Hamming distance
    JACCARD = "JACCARD"         # Jaccard distance


@dataclass
class SearchConfig:
    """Configuration for vector search operations."""
    
    # Basic search parameters
    default_limit: int = 10
    max_limit: int = 1000
    default_metric: DistanceMetric = DistanceMetric.L2
    search_mode: SearchMode = SearchMode.BALANCED
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000
    batch_size: int = 100
    
    # Search parameter ranges
    min_similarity_threshold: float = 0.0
    max_similarity_threshold: float = 1.0
    search_radius: Optional[float] = None
    
    # Advanced options
    enable_search_optimization: bool = True
    enable_result_reranking: bool = True
    enable_diversity_filtering: bool = False
    diversity_threshold: float = 0.8
    
    # Timeout settings
    search_timeout: float = 30.0
    batch_timeout: float = 60.0


@dataclass
class VectorSearchRequest:
    """Vector search request specification."""
    query_vectors: List[List[float]]
    limit: int = 10
    metric: DistanceMetric = DistanceMetric.L2
    similarity_threshold: Optional[float] = None
    search_radius: Optional[float] = None
    filter_expression: Optional[str] = None
    output_fields: Optional[List[str]] = None
    search_mode: SearchMode = SearchMode.BALANCED
    custom_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.query_vectors:
            raise ValueError("Query vectors cannot be empty")
        
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        
        # Validate vector dimensions consistency
        if len(set(len(vec) for vec in self.query_vectors)) > 1:
            raise ValueError("All query vectors must have same dimension")


@dataclass 
class SearchHit:
    """Individual search result hit."""
    id: Union[int, str]
    distance: float
    score: float
    entity: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hit to dictionary."""
        return {
            "id": self.id,
            "distance": self.distance,
            "score": self.score,
            "entity": self.entity,
            "metadata": self.metadata
        }


@dataclass
class VectorSearchResult:
    """Vector search result container."""
    hits: List[List[SearchHit]]  # List of hit lists (one per query vector)
    total_count: int
    search_time: float
    search_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_flat_hits(self) -> List[SearchHit]:
        """Get all hits as a flat list."""
        flat_hits = []
        for hit_list in self.hits:
            flat_hits.extend(hit_list)
        return flat_hits
    
    def get_top_hits(self, k: int = None) -> List[SearchHit]:
        """Get top k hits across all queries."""
        all_hits = self.get_flat_hits()
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:k] if k else all_hits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "hits": [[hit.to_dict() for hit in hit_list] for hit_list in self.hits],
            "total_count": self.total_count,
            "search_time": self.search_time,
            "search_params": self.search_params,
            "metadata": self.metadata
        }


class VectorSearchEngine(LoggerMixin):
    """
    High-performance vector similarity search engine for RAG system.
    
    Features:
    - Milvus integration with optimized search parameters
    - Multiple distance metrics (L2, cosine, inner product)
    - Configurable search modes (precision, balanced, speed, adaptive)
    - Result caching and search optimization
    - Batch and asynchronous search support
    - Search result filtering and ranking
    - Comprehensive search metrics and monitoring
    """
    
    def __init__(
        self,
        milvus_client: MilvusClient,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize Vector Search Engine.
        
        Args:
            milvus_client: Milvus client instance
            config: Search configuration
        """
        self.milvus_client = milvus_client
        self.config = config or SearchConfig()
        
        # Initialize search manager
        self.search_manager = SearchManager(
            client=milvus_client,
            enable_cache=self.config.enable_caching,
            cache_size=self.config.max_cache_size,
            default_ttl=self.config.cache_ttl
        )
        
        # Search parameter configurations
        self.search_mode_configs = self._initialize_search_modes()
        
        # Performance metrics
        self.search_stats = {
            "total_searches": 0,
            "total_search_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "error_count": 0
        }
        
        self.logger.info("VectorSearchEngine initialized successfully")
    
    def search(
        self,
        collection_name: str,
        request: VectorSearchRequest
    ) -> VectorSearchResult:
        """
        Perform vector similarity search.
        
        Args:
            collection_name: Name of the collection to search
            request: Search request specification
            
        Returns:
            Vector search results
        """
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_search_request(request)
            
            # Get collection
            collection = self._get_collection(collection_name)
            
            # Convert to Milvus search query
            milvus_query = self._create_milvus_query(request)
            
            # Perform search
            milvus_result = self.search_manager.search(
                collection=collection,
                query=milvus_query,
                collection_name=collection_name
            )
            
            # Convert result format
            result = self._convert_milvus_result(milvus_result, request)
            
            # Apply post-processing
            if self.config.enable_result_reranking:
                result = self._rerank_results(result, request)
            
            if self.config.enable_diversity_filtering:
                result = self._filter_diverse_results(result, request)
            
            # Update metrics
            search_time = time.time() - start_time
            result.search_time = search_time
            self._update_search_stats(search_time, cached=False)
            
            self.logger.info(
                f"Vector search completed: {result.total_count} results "
                f"in {search_time:.3f}s for collection '{collection_name}'"
            )
            
            return result
            
        except Exception as e:
            search_time = time.time() - start_time
            self._update_search_stats(search_time, error=True)
            self.logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Vector search failed: {e}")
    
    async def search_async(
        self,
        collection_name: str,
        request: VectorSearchRequest
    ) -> VectorSearchResult:
        """
        Perform asynchronous vector similarity search.
        
        Args:
            collection_name: Name of the collection to search
            request: Search request specification
            
        Returns:
            Vector search results
        """
        # Run search in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.search, 
            collection_name, 
            request
        )
    
    def batch_search(
        self,
        collection_name: str,
        requests: List[VectorSearchRequest]
    ) -> List[VectorSearchResult]:
        """
        Perform batch vector searches.
        
        Args:
            collection_name: Name of the collection to search
            requests: List of search requests
            
        Returns:
            List of search results
        """
        if not requests:
            return []
        
        results = []
        
        # Process in batches for memory efficiency
        batch_size = self.config.batch_size
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Process batch
            for request in batch:
                try:
                    result = self.search(collection_name, request)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch search item failed: {e}")
                    # Create error result
                    error_result = VectorSearchResult(
                        hits=[],
                        total_count=0,
                        search_time=0.0,
                        search_params={},
                        metadata={"error": str(e)}
                    )
                    results.append(error_result)
        
        self.logger.info(f"Batch search completed: {len(results)} results")
        return results
    
    async def batch_search_async(
        self,
        collection_name: str,
        requests: List[VectorSearchRequest]
    ) -> List[VectorSearchResult]:
        """
        Perform asynchronous batch vector searches.
        
        Args:
            collection_name: Name of the collection to search
            requests: List of search requests
            
        Returns:
            List of search results
        """
        if not requests:
            return []
        
        # Create async tasks
        tasks = [
            self.search_async(collection_name, request)
            for request in requests
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Async batch search item {i} failed: {result}")
                error_result = VectorSearchResult(
                    hits=[],
                    total_count=0,
                    search_time=0.0,
                    search_params={},
                    metadata={"error": str(result)}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        self.logger.info(f"Async batch search completed: {len(processed_results)} results")
        return processed_results
    
    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = None,
        threshold: float = None,
        filter_expr: str = None,
        **kwargs
    ) -> List[SearchHit]:
        """
        Simplified similarity search interface.
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Single query vector
            limit: Maximum number of results
            threshold: Similarity threshold
            filter_expr: Filter expression
            **kwargs: Additional search parameters
            
        Returns:
            List of search hits
        """
        # Create search request
        request = VectorSearchRequest(
            query_vectors=[query_vector],
            limit=limit or self.config.default_limit,
            similarity_threshold=threshold,
            filter_expression=filter_expr,
            **kwargs
        )
        
        # Perform search
        result = self.search(collection_name, request)
        
        # Return flattened hits
        return result.get_flat_hits()
    
    def _validate_search_request(self, request: VectorSearchRequest) -> None:
        """Validate search request parameters."""
        if request.limit > self.config.max_limit:
            raise ValueError(f"Limit exceeds maximum: {self.config.max_limit}")
        
        if request.similarity_threshold is not None:
            if not (self.config.min_similarity_threshold <= 
                   request.similarity_threshold <= 
                   self.config.max_similarity_threshold):
                raise ValueError(
                    f"Similarity threshold must be between "
                    f"{self.config.min_similarity_threshold} and "
                    f"{self.config.max_similarity_threshold}"
                )
    
    def _get_collection(self, collection_name: str) -> MilvusCollection:
        """Get Milvus collection instance."""
        try:
            return self.milvus_client.get_collection(collection_name)
        except Exception as e:
            raise SearchError(f"Failed to get collection '{collection_name}': {e}")
    
    def _create_milvus_query(self, request: VectorSearchRequest) -> SearchQuery:
        """Convert VectorSearchRequest to Milvus SearchQuery."""
        # Map distance metrics
        metric_mapping = {
            DistanceMetric.L2: MetricType.L2,
            DistanceMetric.IP: MetricType.IP,
            DistanceMetric.COSINE: MetricType.COSINE,
        }
        
        milvus_metric = metric_mapping.get(request.metric, MetricType.L2)
        
        # Map search modes to strategies
        strategy_mapping = {
            SearchMode.PRECISION: SearchStrategy.EXACT,
            SearchMode.BALANCED: SearchStrategy.BALANCED,
            SearchMode.SPEED: SearchStrategy.FAST,
            SearchMode.ADAPTIVE: SearchStrategy.ADAPTIVE
        }
        
        strategy = strategy_mapping.get(request.search_mode, SearchStrategy.BALANCED)
        
        # Build search parameters
        search_params = self._get_search_params(request.search_mode, request.custom_params)
        
        return SearchQuery(
            vectors=request.query_vectors,
            limit=request.limit,
            metric_type=milvus_metric,
            search_params=search_params,
            filter_expr=request.filter_expression,
            output_fields=request.output_fields,
            strategy=strategy
        )
    
    def _get_search_params(
        self, 
        search_mode: SearchMode, 
        custom_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get optimized search parameters for the given mode."""
        # Base parameters from mode configuration
        base_params = self.search_mode_configs.get(search_mode, {}).copy()
        
        # Override with custom parameters
        if custom_params:
            base_params.update(custom_params)
        
        return base_params
    
    def _convert_milvus_result(
        self, 
        milvus_result, 
        request: VectorSearchRequest
    ) -> VectorSearchResult:
        """Convert Milvus search result to VectorSearchResult."""
        hits_per_query = []
        total_count = 0
        
        # Process hits from Milvus result
        if hasattr(milvus_result, 'hits'):
            # Handle different result formats
            milvus_hits = milvus_result.hits
            if not isinstance(milvus_hits[0], list):
                # Single query result
                milvus_hits = [milvus_hits]
            
            for query_hits in milvus_hits:
                converted_hits = []
                for hit in query_hits:
                    # Convert to SearchHit
                    search_hit = SearchHit(
                        id=hit.get("id", hit.get("pk")),
                        distance=hit.get("distance", 0.0),
                        score=self._distance_to_score(hit.get("distance", 0.0), request.metric),
                        entity=hit.get("entity", {}),
                        metadata=hit.get("metadata", {})
                    )
                    converted_hits.append(search_hit)
                
                hits_per_query.append(converted_hits)
                total_count += len(converted_hits)
        
        return VectorSearchResult(
            hits=hits_per_query,
            total_count=total_count,
            search_time=getattr(milvus_result, 'query_time', 0.0),
            search_params=getattr(milvus_result, 'search_params', {}),
            metadata={
                "search_mode": request.search_mode.value,
                "metric": request.metric.value,
                "query_count": len(request.query_vectors)
            }
        )
    
    def _distance_to_score(self, distance: float, metric: DistanceMetric) -> float:
        """Convert distance to similarity score (0-1, higher is better)."""
        if metric == DistanceMetric.IP or metric == DistanceMetric.COSINE:
            # For IP and COSINE, higher distance means higher similarity
            return distance
        elif metric == DistanceMetric.L2:
            # For L2, convert distance to similarity score
            return 1.0 / (1.0 + distance)
        else:
            # Default: assume lower distance means higher similarity
            return 1.0 / (1.0 + distance)
    
    def _rerank_results(
        self, 
        result: VectorSearchResult, 
        request: VectorSearchRequest
    ) -> VectorSearchResult:
        """Apply result reranking for improved relevance."""
        # Simple reranking by score (could be enhanced with ML models)
        for hit_list in result.hits:
            hit_list.sort(key=lambda h: h.score, reverse=True)
        
        result.metadata["reranked"] = True
        return result
    
    def _filter_diverse_results(
        self, 
        result: VectorSearchResult, 
        request: VectorSearchRequest
    ) -> VectorSearchResult:
        """Filter results for diversity."""
        if not self.config.enable_diversity_filtering:
            return result
        
        # Simple diversity filtering (could be enhanced)
        for i, hit_list in enumerate(result.hits):
            if len(hit_list) <= 1:
                continue
            
            diverse_hits = [hit_list[0]]  # Always keep top result
            
            for hit in hit_list[1:]:
                is_diverse = True
                for selected_hit in diverse_hits:
                    # Check if too similar to already selected hits
                    similarity = self._calculate_hit_similarity(hit, selected_hit)
                    if similarity > self.config.diversity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_hits.append(hit)
            
            result.hits[i] = diverse_hits
        
        # Update total count
        result.total_count = sum(len(hit_list) for hit_list in result.hits)
        result.metadata["diversity_filtered"] = True
        
        return result
    
    def _calculate_hit_similarity(self, hit1: SearchHit, hit2: SearchHit) -> float:
        """Calculate similarity between two hits (simplified)."""
        # This is a placeholder - in practice, you might compare
        # the actual vectors or use other similarity measures
        return abs(hit1.score - hit2.score)
    
    def _initialize_search_modes(self) -> Dict[SearchMode, Dict[str, Any]]:
        """Initialize search mode configurations."""
        return {
            SearchMode.PRECISION: {
                "nprobe": 256,
                "ef": 512,
                "search_k": 10000,
                "description": "High precision search"
            },
            SearchMode.BALANCED: {
                "nprobe": 64,
                "ef": 128,
                "search_k": 1000,
                "description": "Balanced precision and speed"
            },
            SearchMode.SPEED: {
                "nprobe": 16,
                "ef": 32,
                "search_k": 100,
                "description": "Fast search with lower precision"
            },
            SearchMode.ADAPTIVE: {
                "description": "Adaptive parameters based on context"
            }
        }
    
    def _update_search_stats(
        self, 
        search_time: float, 
        cached: bool = False, 
        error: bool = False
    ) -> None:
        """Update search statistics."""
        if error:
            self.search_stats["error_count"] += 1
        else:
            self.search_stats["total_searches"] += 1
            self.search_stats["total_search_time"] += search_time
            
            if cached:
                self.search_stats["cache_hits"] += 1
            else:
                self.search_stats["cache_misses"] += 1
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        total_searches = self.search_stats["total_searches"]
        avg_time = (
            self.search_stats["total_search_time"] / total_searches
            if total_searches > 0 else 0.0
        )
        
        cache_requests = self.search_stats["cache_hits"] + self.search_stats["cache_misses"]
        cache_rate = (
            self.search_stats["cache_hits"] / cache_requests
            if cache_requests > 0 else 0.0
        )
        
        return {
            "total_searches": total_searches,
            "average_search_time": avg_time,
            "cache_hit_rate": cache_rate,
            "cache_hits": self.search_stats["cache_hits"],
            "cache_misses": self.search_stats["cache_misses"],
            "error_count": self.search_stats["error_count"],
            "cache_stats": self.search_manager.get_cache_stats()
        }
    
    def clear_cache(self) -> None:
        """Clear search cache."""
        self.search_manager.clear_cache()
        self.logger.info("Vector search cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the search engine."""
        try:
            # Test basic functionality
            test_successful = True
            error_message = None
            
            # Check Milvus client connectivity
            if not self.milvus_client.is_connected():
                test_successful = False
                error_message = "Milvus client not connected"
            
            return {
                "status": "healthy" if test_successful else "unhealthy",
                "error": error_message,
                "config": {
                    "caching_enabled": self.config.enable_caching,
                    "max_cache_size": self.config.max_cache_size,
                    "default_limit": self.config.default_limit,
                    "search_timeout": self.config.search_timeout
                },
                "stats": self.get_search_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "config": {},
                "stats": {}
            }


def create_vector_search_engine(
    milvus_client: MilvusClient,
    config: Optional[SearchConfig] = None
) -> VectorSearchEngine:
    """
    Create and configure a VectorSearchEngine instance.
    
    Args:
        milvus_client: Milvus client instance
        config: Search configuration
        
    Returns:
        Configured VectorSearchEngine
    """
    return VectorSearchEngine(
        milvus_client=milvus_client,
        config=config or SearchConfig()
    )


def create_search_request(
    query_vectors: List[List[float]],
    limit: int = 10,
    metric: DistanceMetric = DistanceMetric.L2,
    mode: SearchMode = SearchMode.BALANCED,
    **kwargs
) -> VectorSearchRequest:
    """
    Create a vector search request with simplified interface.
    
    Args:
        query_vectors: List of query vectors
        limit: Maximum number of results
        metric: Distance metric to use
        mode: Search mode
        **kwargs: Additional request parameters
        
    Returns:
        Vector search request
    """
    return VectorSearchRequest(
        query_vectors=query_vectors,
        limit=limit,
        metric=metric,
        search_mode=mode,
        **kwargs
    )