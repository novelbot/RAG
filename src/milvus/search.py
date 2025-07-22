"""
Advanced search management for Milvus vector database with optimization and caching.
"""

import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np

from pymilvus import MilvusException
from loguru import logger

from src.core.exceptions import MilvusError, SearchError
from src.core.logging import LoggerMixin
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection, SearchResult
from src.milvus.index import IndexType, MetricType


class SearchStrategy(Enum):
    """Search strategy types."""
    EXACT = "exact"             # Exact search with highest accuracy
    BALANCED = "balanced"       # Balanced accuracy and speed
    FAST = "fast"               # Fast search with lower accuracy
    ADAPTIVE = "adaptive"       # Adaptive strategy based on query


@dataclass
class SearchQuery:
    """Search query specification."""
    vectors: List[List[float]]
    limit: int = 10
    metric_type: MetricType = MetricType.L2
    search_params: Optional[Dict[str, Any]] = None
    filter_expr: Optional[str] = None
    output_fields: Optional[List[str]] = None
    strategy: SearchStrategy = SearchStrategy.BALANCED
    
    def get_cache_key(self) -> str:
        """Generate cache key for query."""
        # Create deterministic hash of query parameters
        query_str = json.dumps({
            "vectors_hash": hashlib.md5(str(self.vectors).encode()).hexdigest(),
            "limit": self.limit,
            "metric_type": self.metric_type.value,
            "search_params": self.search_params,
            "filter_expr": self.filter_expr,
            "output_fields": self.output_fields
        }, sort_keys=True)
        
        return hashlib.sha256(query_str.encode()).hexdigest()


@dataclass
class CachedResult:
    """Cached search result."""
    result: SearchResult
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    def is_expired(self) -> bool:
        """Check if cached result is expired."""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl_seconds)


@dataclass
class SearchMetrics:
    """Search performance metrics."""
    query_count: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    last_query_time: Optional[datetime] = None
    
    @property
    def average_latency(self) -> float:
        """Calculate average query latency."""
        return self.total_latency / max(1, self.query_count)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total_requests)


class SearchManager(LoggerMixin):
    """
    Advanced search manager for Milvus with optimization and caching.
    
    Features:
    - Multiple search strategies with automatic optimization
    - Result caching with TTL
    - Query performance monitoring
    - Adaptive parameter tuning
    - Batch and hybrid search support
    """
    
    def __init__(self, 
                 client: MilvusClient,
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 default_ttl: int = 300):
        """
        Initialize Search Manager.
        
        Args:
            client: Milvus client instance
            enable_cache: Enable result caching
            cache_size: Maximum cache entries
            default_ttl: Default cache TTL in seconds
        """
        self.client = client
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.default_ttl = default_ttl
        
        # Result cache
        self._cache: Dict[str, CachedResult] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        
        # Performance metrics
        self._metrics: Dict[str, SearchMetrics] = {}
        
        # Strategy configurations
        self._strategy_configs = self._initialize_strategies()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _initialize_strategies(self) -> Dict[SearchStrategy, Dict[str, Any]]:
        """Initialize search strategy configurations."""
        return {
            SearchStrategy.EXACT: {
                "description": "Highest accuracy search",
                "params": {
                    "nprobe": 256,
                    "ef": 512,
                    "search_k": 10000
                },
                "timeout": 10.0
            },
            SearchStrategy.BALANCED: {
                "description": "Balanced accuracy and speed",
                "params": {
                    "nprobe": 64,
                    "ef": 128,
                    "search_k": 1000
                },
                "timeout": 5.0
            },
            SearchStrategy.FAST: {
                "description": "Fast search with lower accuracy",
                "params": {
                    "nprobe": 16,
                    "ef": 32,
                    "search_k": 100
                },
                "timeout": 2.0
            },
            SearchStrategy.ADAPTIVE: {
                "description": "Adaptive strategy based on query context",
                "params": {},  # Determined dynamically
                "timeout": 5.0
            }
        }
    
    def search(self,
               collection: MilvusCollection,
               query: SearchQuery,
               collection_name: Optional[str] = None) -> SearchResult:
        """
        Perform vector similarity search.
        
        Args:
            collection: Milvus collection
            query: Search query specification
            collection_name: Collection name for metrics (optional)
            
        Returns:
            Search results
        """
        collection_key = collection_name or collection.collection_name
        start_time = time.time()
        
        try:
            # Check cache first
            if self.enable_cache:
                cache_key = query.get_cache_key()
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    self._update_metrics(collection_key, 0, cache_hit=True)
                    self.logger.debug(f"Cache hit for search query")
                    return cached_result.result
            
            # Ensure collection is loaded
            if not collection.is_loaded():
                collection.load()
            
            # Get optimized search parameters
            search_params = self._get_optimized_params(
                collection=collection,
                query=query
            )
            
            # Perform search
            result = collection.search(
                query_vectors=query.vectors,
                limit=query.limit,
                search_params=search_params,
                expr=query.filter_expr,
                output_fields=query.output_fields
            )
            
            query_time = time.time() - start_time
            
            # Cache result if enabled
            if self.enable_cache:
                self._cache_result(cache_key, result, query_time)
            
            # Update metrics
            self._update_metrics(collection_key, query_time, cache_hit=False)
            
            self.logger.info(f"Search completed: {len(result.hits)} results in {query_time:.3f}s")
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            self._update_metrics(collection_key, query_time, error=True)
            self.logger.error(f"Search failed: {e}")
            raise SearchError(f"Search operation failed: {e}")
    
    def _get_optimized_params(self,
                            collection: MilvusCollection,
                            query: SearchQuery) -> Dict[str, Any]:
        """Get optimized search parameters based on strategy and index type."""
        # Base parameters
        search_params = {
            "metric_type": query.metric_type.value,
            "params": {}
        }
        
        # Get index information
        try:
            indexes = collection._collection.indexes
            if indexes:
                index_type = IndexType(indexes[0].params.get("index_type", "FLAT"))
            else:
                index_type = IndexType.FLAT
        except:
            index_type = IndexType.FLAT
        
        # Apply strategy-specific parameters
        if query.strategy == SearchStrategy.ADAPTIVE:
            # Adaptive strategy based on collection size and query characteristics
            collection_size = collection.get_entity_count()
            params = self._get_adaptive_params(index_type, collection_size, len(query.vectors))
        else:
            # Use predefined strategy parameters
            strategy_config = self._strategy_configs[query.strategy]
            params = self._filter_params_by_index(strategy_config["params"], index_type)
        
        # Override with user-provided parameters
        if query.search_params:
            params.update(query.search_params.get("params", {}))
            if "metric_type" in query.search_params:
                search_params["metric_type"] = query.search_params["metric_type"]
        
        search_params["params"] = params
        return search_params
    
    def _get_adaptive_params(self,
                           index_type: IndexType,
                           collection_size: int,
                           query_count: int) -> Dict[str, Any]:
        """Get adaptive parameters based on context."""
        params = {}
        
        if index_type in [IndexType.IVF_FLAT, IndexType.IVF_SQ8, IndexType.IVF_PQ]:
            # Adaptive nprobe based on collection size and query count
            if collection_size < 100000:
                base_nprobe = 32
            elif collection_size < 1000000:
                base_nprobe = 64
            else:
                base_nprobe = 128
            
            # Adjust based on query batch size
            if query_count > 10:
                params["nprobe"] = min(base_nprobe * 2, 512)
            else:
                params["nprobe"] = base_nprobe
                
        elif index_type == IndexType.HNSW:
            # Adaptive ef based on collection size
            if collection_size < 100000:
                params["ef"] = 64
            elif collection_size < 1000000:
                params["ef"] = 128
            else:
                params["ef"] = 256
                
        elif index_type == IndexType.ANNOY:
            # Adaptive search_k
            params["search_k"] = min(collection_size // 100, 5000)
        
        return params
    
    def _filter_params_by_index(self,
                              params: Dict[str, Any],
                              index_type: IndexType) -> Dict[str, Any]:
        """Filter parameters based on index type."""
        filtered = {}
        
        if index_type in [IndexType.IVF_FLAT, IndexType.IVF_SQ8, IndexType.IVF_PQ]:
            if "nprobe" in params:
                filtered["nprobe"] = params["nprobe"]
        elif index_type == IndexType.HNSW:
            if "ef" in params:
                filtered["ef"] = params["ef"]
        elif index_type == IndexType.ANNOY:
            if "search_k" in params:
                filtered["search_k"] = params["search_k"]
        
        return filtered
    
    def batch_search(self,
                    collection: MilvusCollection,
                    queries: List[SearchQuery],
                    parallel: bool = True) -> List[SearchResult]:
        """
        Perform batch search operations.
        
        Args:
            collection: Milvus collection
            queries: List of search queries
            parallel: Whether to process queries in parallel
            
        Returns:
            List of search results
        """
        results = []
        
        if parallel and len(queries) > 1:
            # Combine vectors for batch processing
            all_vectors = []
            query_ranges = []
            current_idx = 0
            
            for query in queries:
                start_idx = current_idx
                end_idx = current_idx + len(query.vectors)
                query_ranges.append((start_idx, end_idx, query))
                all_vectors.extend(query.vectors)
                current_idx = end_idx
            
            # Perform single batch search
            if all_vectors:
                # Use parameters from first query (assuming similar requirements)
                batch_query = SearchQuery(
                    vectors=all_vectors,
                    limit=max(q.limit for q in queries),
                    metric_type=queries[0].metric_type,
                    search_params=queries[0].search_params,
                    strategy=queries[0].strategy
                )
                
                batch_result = self.search(collection, batch_query)
                
                # Split results back to individual queries
                for start_idx, end_idx, original_query in query_ranges:
                    query_hits = batch_result.hits[start_idx * original_query.limit:(end_idx * original_query.limit)]
                    
                    query_result = SearchResult(
                        hits=query_hits[:original_query.limit],
                        total_count=len(query_hits),
                        query_time=batch_result.query_time / len(queries),
                        search_params=batch_result.search_params
                    )
                    results.append(query_result)
        else:
            # Sequential processing
            for query in queries:
                result = self.search(collection, query)
                results.append(result)
        
        return results
    
    def hybrid_search(self,
                     collection: MilvusCollection,
                     vector_query: SearchQuery,
                     scalar_filters: List[str],
                     fusion_method: str = "rrf") -> SearchResult:
        """
        Perform hybrid search combining vector similarity and scalar filtering.
        
        Args:
            collection: Milvus collection
            vector_query: Vector similarity query
            scalar_filters: List of scalar filter expressions
            fusion_method: Result fusion method ("rrf", "weighted")
            
        Returns:
            Fused search results
        """
        try:
            results = []
            
            # Perform vector search
            vector_result = self.search(collection, vector_query)
            results.append(("vector", vector_result))
            
            # Perform scalar-filtered searches
            for i, filter_expr in enumerate(scalar_filters):
                filtered_query = SearchQuery(
                    vectors=vector_query.vectors,
                    limit=vector_query.limit,
                    metric_type=vector_query.metric_type,
                    search_params=vector_query.search_params,
                    filter_expr=filter_expr,
                    output_fields=vector_query.output_fields,
                    strategy=vector_query.strategy
                )
                
                filtered_result = self.search(collection, filtered_query)
                results.append((f"filter_{i}", filtered_result))
            
            # Fuse results
            if fusion_method == "rrf":
                fused_result = self._reciprocal_rank_fusion(results)
            else:
                fused_result = self._weighted_fusion(results)
            
            self.logger.info(f"Hybrid search completed: {len(results)} sources fused")
            return fused_result
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}")
    
    def _reciprocal_rank_fusion(self,
                              results: List[Tuple[str, SearchResult]],
                              k: int = 60) -> SearchResult:
        """Fuse results using Reciprocal Rank Fusion."""
        entity_scores = {}
        
        for source, result in results:
            for rank, hit in enumerate(result.hits):
                entity_id = hit.get("id")
                if entity_id not in entity_scores:
                    entity_scores[entity_id] = {"score": 0, "hit": hit}
                
                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank + 1)
                entity_scores[entity_id]["score"] += rrf_score
        
        # Sort by fused score
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Create fused result
        fused_hits = []
        for entity_id, data in sorted_entities:
            hit = data["hit"].copy()
            hit["fused_score"] = data["score"]
            fused_hits.append(hit)
        
        return SearchResult(
            hits=fused_hits,
            total_count=len(fused_hits),
            query_time=sum(r.query_time for _, r in results),
            search_params={"fusion": "rrf", "k": k}
        )
    
    def _weighted_fusion(self,
                        results: List[Tuple[str, SearchResult]],
                        weights: Optional[List[float]] = None) -> SearchResult:
        """Fuse results using weighted combination."""
        if weights is None:
            weights = [1.0] * len(results)
        
        entity_scores = {}
        
        for (source, result), weight in zip(results, weights):
            for rank, hit in enumerate(result.hits):
                entity_id = hit.get("id")
                if entity_id not in entity_scores:
                    entity_scores[entity_id] = {"score": 0, "hit": hit}
                
                # Weighted score based on rank and distance
                distance = hit.get("distance", 1.0)
                normalized_score = 1.0 / (1.0 + distance)
                weighted_score = normalized_score * weight
                
                entity_scores[entity_id]["score"] += weighted_score
        
        # Sort by fused score
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Create fused result
        fused_hits = []
        for entity_id, data in sorted_entities:
            hit = data["hit"].copy()
            hit["fused_score"] = data["score"]
            fused_hits.append(hit)
        
        return SearchResult(
            hits=fused_hits,
            total_count=len(fused_hits),
            query_time=sum(r.query_time for _, r in results),
            search_params={"fusion": "weighted", "weights": weights}
        )
    
    def _get_from_cache(self, cache_key: str) -> Optional[CachedResult]:
        """Get result from cache if not expired."""
        with self._lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if not cached.is_expired():
                    self._cache_access_times[cache_key] = datetime.utcnow()
                    return cached
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
                    self._cache_access_times.pop(cache_key, None)
        
        return None
    
    def _cache_result(self,
                     cache_key: str,
                     result: SearchResult,
                     query_time: float) -> None:
        """Cache search result."""
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self._cache) >= self.cache_size:
                self._evict_oldest_entry()
            
            cached_result = CachedResult(
                result=result,
                timestamp=datetime.utcnow(),
                ttl_seconds=self.default_ttl
            )
            
            self._cache[cache_key] = cached_result
            self._cache_access_times[cache_key] = datetime.utcnow()
    
    def _evict_oldest_entry(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache_access_times:
            return
        
        oldest_key = min(
            self._cache_access_times.keys(),
            key=lambda k: self._cache_access_times[k]
        )
        
        self._cache.pop(oldest_key, None)
        self._cache_access_times.pop(oldest_key, None)
    
    def _update_metrics(self,
                       collection_name: str,
                       query_time: float,
                       cache_hit: bool = False,
                       error: bool = False) -> None:
        """Update search metrics."""
        with self._lock:
            if collection_name not in self._metrics:
                self._metrics[collection_name] = SearchMetrics()
            
            metrics = self._metrics[collection_name]
            
            if not error:
                metrics.query_count += 1
                metrics.total_latency += query_time
                
                if cache_hit:
                    metrics.cache_hits += 1
                else:
                    metrics.cache_misses += 1
            else:
                metrics.error_count += 1
            
            metrics.last_query_time = datetime.utcnow()
    
    def get_search_metrics(self, collection_name: str) -> Dict[str, Any]:
        """Get search metrics for collection."""
        with self._lock:
            if collection_name not in self._metrics:
                return {"error": "No metrics available"}
            
            metrics = self._metrics[collection_name]
            
            return {
                "collection_name": collection_name,
                "query_count": metrics.query_count,
                "average_latency": metrics.average_latency,
                "cache_hit_rate": metrics.cache_hit_rate,
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "error_count": metrics.error_count,
                "last_query_time": metrics.last_query_time.isoformat() if metrics.last_query_time else None,
                "cache_size": len(self._cache)
            }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._cache_access_times.clear()
            self.logger.info("Search cache cleared")
    
    def clear_metrics(self, collection_name: Optional[str] = None) -> None:
        """Clear search metrics."""
        with self._lock:
            if collection_name:
                self._metrics.pop(collection_name, None)
                self.logger.info(f"Cleared metrics for collection: {collection_name}")
            else:
                self._metrics.clear()
                self.logger.info("Cleared all search metrics")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            # Calculate cache age distribution
            now = datetime.utcnow()
            age_distribution = {}
            
            for key, cached in self._cache.items():
                age_seconds = (now - cached.timestamp).total_seconds()
                age_bucket = f"{int(age_seconds // 60)}min"
                age_distribution[age_bucket] = age_distribution.get(age_bucket, 0) + 1
            
            return {
                "cache_enabled": self.enable_cache,
                "cache_size": len(self._cache),
                "max_cache_size": self.cache_size,
                "default_ttl": self.default_ttl,
                "age_distribution": age_distribution,
                "total_collections": len(self._metrics)
            }


def create_search_manager(client: MilvusClient,
                        enable_cache: bool = True,
                        cache_size: int = 1000) -> SearchManager:
    """
    Create Search Manager instance.
    
    Args:
        client: Milvus client instance
        enable_cache: Enable result caching
        cache_size: Maximum cache entries
        
    Returns:
        Configured Search Manager
    """
    return SearchManager(
        client=client,
        enable_cache=enable_cache,
        cache_size=cache_size
    )


def create_search_query(vectors: List[List[float]],
                       limit: int = 10,
                       strategy: SearchStrategy = SearchStrategy.BALANCED,
                       filter_expr: Optional[str] = None,
                       **kwargs) -> SearchQuery:
    """
    Create search query with simplified interface.
    
    Args:
        vectors: Query vectors
        limit: Maximum results
        strategy: Search strategy
        filter_expr: Filter expression
        **kwargs: Additional query parameters
        
    Returns:
        Search query object
    """
    return SearchQuery(
        vectors=vectors,
        limit=limit,
        strategy=strategy,
        filter_expr=filter_expr,
        **kwargs
    )