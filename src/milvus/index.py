"""
Index management for Milvus vector database with performance optimization.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from pymilvus import utility, MilvusException, Index
from loguru import logger

from src.core.exceptions import MilvusError, IndexError, PerformanceError
from src.core.logging import LoggerMixin
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection


class IndexType(Enum):
    """Milvus index types with their characteristics."""
    FLAT = "FLAT"           # Default, 100% recall, no parameters needed
    IVF_FLAT = "IVF_FLAT"   # Basic IVF index with original data
    IVF_SQ8 = "IVF_SQ8"     # IVF with scalar quantization (8-bit)
    IVF_PQ = "IVF_PQ"       # IVF with product quantization
    HNSW = "HNSW"           # Hierarchical Navigable Small World
    ANNOY = "ANNOY"         # Approximate Nearest Neighbors Oh Yeah


class MetricType(Enum):
    """Distance metric types for vector similarity."""
    L2 = "L2"               # Euclidean distance
    IP = "IP"               # Inner product
    COSINE = "COSINE"       # Cosine similarity
    HAMMING = "HAMMING"     # Hamming distance (for binary vectors)
    JACCARD = "JACCARD"     # Jaccard distance (for binary vectors)


@dataclass
class IndexConfig:
    """Configuration for Milvus index creation."""
    index_type: IndexType
    metric_type: MetricType
    field_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for Milvus."""
        return {
            "index_type": self.index_type.value,
            "metric_type": self.metric_type.value,
            "params": self.params
        }


@dataclass
class IndexPerformance:
    """Index performance metrics."""
    index_type: IndexType
    field_name: str
    build_time: float
    index_size: Optional[int] = None
    search_latency: Optional[float] = None
    search_throughput: Optional[float] = None
    recall_rate: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchPerformance:
    """Search performance metrics."""
    query_time: float
    result_count: int
    search_params: Dict[str, Any]
    index_type: IndexType
    timestamp: datetime = field(default_factory=datetime.utcnow)


class IndexManager(LoggerMixin):
    """
    Index manager for Milvus vector database performance optimization.
    
    Based on Context7 documentation for PyMilvus index management:
    - Supports multiple index types: FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, ANNOY
    - Provides automatic parameter optimization based on data characteristics
    - Monitors index building progress and performance
    """
    
    def __init__(self, client: MilvusClient):
        """
        Initialize Index Manager.
        
        Args:
            client: Milvus client instance
        """
        self.client = client
        self._index_configs: Dict[str, IndexConfig] = {}
        self._performance_metrics: Dict[str, List[IndexPerformance]] = {}
        self._search_metrics: Dict[str, List[SearchPerformance]] = {}
        self._lock = threading.Lock()
        
        # Default index configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self) -> None:
        """Initialize default index configurations."""
        # FLAT index - default, 100% recall
        flat_config = IndexConfig(
            index_type=IndexType.FLAT,
            metric_type=MetricType.L2,
            field_name="vector",
            description="Default FLAT index with 100% recall"
        )
        
        # IVF_FLAT index - balanced performance
        ivf_flat_config = IndexConfig(
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            field_name="vector",
            params={"nlist": 1024},
            description="IVF_FLAT index for balanced performance"
        )
        
        # HNSW index - high performance
        hnsw_config = IndexConfig(
            index_type=IndexType.HNSW,
            metric_type=MetricType.L2,
            field_name="vector",
            params={"M": 16, "efConstruction": 200},
            description="HNSW index for high-performance search"
        )
        
        self._index_configs["default_flat"] = flat_config
        self._index_configs["default_ivf_flat"] = ivf_flat_config
        self._index_configs["default_hnsw"] = hnsw_config
    
    def create_index_config(self,
                          config_name: str,
                          index_type: IndexType,
                          metric_type: MetricType,
                          field_name: str = "vector",
                          **params) -> IndexConfig:
        """
        Create custom index configuration.
        
        Args:
            config_name: Name for the configuration
            index_type: Type of index to create
            metric_type: Distance metric to use
            field_name: Vector field name
            **params: Index-specific parameters
            
        Returns:
            Created index configuration
        """
        # Validate and optimize parameters
        optimized_params = self._optimize_index_params(index_type, **params)
        
        config = IndexConfig(
            index_type=index_type,
            metric_type=metric_type,
            field_name=field_name,
            params=optimized_params,
            description=f"Custom {index_type.value} index configuration"
        )
        
        with self._lock:
            self._index_configs[config_name] = config
        
        self.logger.info(f"Created index configuration: {config_name}")
        return config
    
    def _optimize_index_params(self, index_type: IndexType, **params) -> Dict[str, Any]:
        """
        Optimize index parameters based on type and best practices.
        
        Based on Context7 PyMilvus documentation for index parameters.
        """
        optimized = {}
        
        if index_type == IndexType.FLAT:
            # FLAT index has no parameters
            pass
            
        elif index_type == IndexType.IVF_FLAT:
            # IVF_FLAT parameters: nlist (1~65536)
            nlist = params.get("nlist", 1024)
            optimized["nlist"] = max(1, min(65536, nlist))
            
        elif index_type == IndexType.IVF_SQ8:
            # IVF_SQ8 parameters: nlist (1~65536)
            nlist = params.get("nlist", 1024)
            optimized["nlist"] = max(1, min(65536, nlist))
            
        elif index_type == IndexType.IVF_PQ:
            # IVF_PQ parameters: nlist, m, nbits
            nlist = params.get("nlist", 1024)
            m = params.get("m", 8)
            nbits = params.get("nbits", 8)
            
            optimized["nlist"] = max(1, min(65536, nlist))
            optimized["m"] = max(1, min(64, m))
            optimized["nbits"] = max(1, min(16, nbits))
            
        elif index_type == IndexType.HNSW:
            # HNSW parameters: M (4~64), efConstruction (8~512)
            M = params.get("M", 16)
            efConstruction = params.get("efConstruction", 200)
            
            optimized["M"] = max(4, min(64, M))
            optimized["efConstruction"] = max(8, min(512, efConstruction))
            
        elif index_type == IndexType.ANNOY:
            # ANNOY parameters: n_trees (1~1024)
            n_trees = params.get("n_trees", 8)
            optimized["n_trees"] = max(1, min(1024, n_trees))
        
        return optimized
    
    def create_index(self,
                    collection: MilvusCollection,
                    config_name: Optional[str] = None,
                    index_config: Optional[IndexConfig] = None,
                    wait_for_completion: bool = True,
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Create index on collection.
        
        Args:
            collection: Milvus collection
            config_name: Name of predefined configuration
            index_config: Custom index configuration
            wait_for_completion: Wait for index building to complete
            timeout: Build timeout in seconds
            
        Returns:
            Index creation result with performance metrics
        """
        try:
            # Get index configuration
            if index_config:
                config = index_config
            elif config_name and config_name in self._index_configs:
                config = self._index_configs[config_name]
            else:
                config = self._index_configs["default_ivf_flat"]
            
            self.logger.info(f"Creating {config.index_type.value} index on field '{config.field_name}'")
            
            # Check if collection is loaded and release if necessary
            if collection.is_loaded():
                collection.release()
                self.logger.info("Released collection for index creation")
            
            # Create index with performance monitoring
            start_time = time.time()
            
            collection._collection.create_index(
                field_name=config.field_name,
                index_params=config.to_dict()
            )
            
            # Wait for index building completion if requested
            if wait_for_completion:
                self._wait_for_index_completion(
                    collection.collection_name,
                    timeout=timeout
                )
            
            build_time = time.time() - start_time
            
            # Record performance metrics
            performance = IndexPerformance(
                index_type=config.index_type,
                field_name=config.field_name,
                build_time=build_time
            )
            
            with self._lock:
                if collection.collection_name not in self._performance_metrics:
                    self._performance_metrics[collection.collection_name] = []
                self._performance_metrics[collection.collection_name].append(performance)
            
            result = {
                "status": "success",
                "index_type": config.index_type.value,
                "field_name": config.field_name,
                "build_time": build_time,
                "config": config.to_dict()
            }
            
            self.logger.info(f"Index created successfully in {build_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            raise IndexError(f"Index creation failed: {e}")
    
    def _wait_for_index_completion(self,
                                 collection_name: str,
                                 timeout: Optional[float] = None) -> None:
        """Wait for index building to complete."""
        try:
            if timeout:
                utility.wait_for_index_building_complete(
                    collection_name,
                    timeout=timeout,
                    using=self.client.alias
                )
            else:
                utility.wait_for_index_building_complete(
                    collection_name,
                    using=self.client.alias
                )
            
            self.logger.info(f"Index building completed for: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Index building wait failed: {e}")
            raise IndexError(f"Index building timeout: {e}")
    
    def get_index_building_progress(self, collection_name: str) -> Dict[str, Any]:
        """
        Get index building progress.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Index building progress information
        """
        try:
            progress = utility.index_building_progress(
                collection_name,
                using=self.client.alias
            )
            
            return {
                "collection_name": collection_name,
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index progress: {e}")
            raise IndexError(f"Index progress check failed: {e}")
    
    def drop_index(self,
                  collection: MilvusCollection,
                  field_name: str = "vector") -> Dict[str, Any]:
        """
        Drop index from collection field.
        
        Args:
            collection: Milvus collection
            field_name: Field to drop index from
            
        Returns:
            Drop operation result
        """
        try:
            start_time = time.time()
            
            # Check if collection is loaded and release if necessary
            if collection.is_loaded():
                collection.release()
                self.logger.info("Released collection for index drop")
            
            collection._collection.drop_index(field_name)
            
            drop_time = time.time() - start_time
            
            result = {
                "status": "success",
                "field_name": field_name,
                "drop_time": drop_time
            }
            
            self.logger.info(f"Index dropped from field '{field_name}' in {drop_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to drop index: {e}")
            raise IndexError(f"Index drop failed: {e}")
    
    def list_indexes(self, collection: MilvusCollection) -> List[Dict[str, Any]]:
        """
        List all indexes on collection.
        
        Args:
            collection: Milvus collection
            
        Returns:
            List of index information
        """
        try:
            indexes = []
            
            for index in collection._collection.indexes:
                index_info = {
                    "field_name": index.field_name,
                    "index_type": index.params.get("index_type", "UNKNOWN"),
                    "metric_type": index.params.get("metric_type", "UNKNOWN"),
                    "params": index.params.get("params", {}),
                    "collection_name": index.collection_name
                }
                indexes.append(index_info)
            
            self.logger.info(f"Found {len(indexes)} indexes on collection")
            return indexes
            
        except Exception as e:
            self.logger.error(f"Failed to list indexes: {e}")
            raise IndexError(f"Index listing failed: {e}")
    
    def has_index(self,
                 collection: MilvusCollection,
                 field_name: str = "vector") -> bool:
        """
        Check if field has index.
        
        Args:
            collection: Milvus collection
            field_name: Field to check
            
        Returns:
            True if field has index
        """
        try:
            return collection._collection.has_index(field_name)
            
        except Exception as e:
            self.logger.error(f"Failed to check index existence: {e}")
            return False
    
    def get_optimal_search_params(self,
                                index_type: IndexType,
                                collection_size: Optional[int] = None,
                                target_recall: float = 0.9) -> Dict[str, Any]:
        """
        Get optimal search parameters for index type.
        
        Based on Context7 PyMilvus search parameter guidelines.
        
        Args:
            index_type: Type of index
            collection_size: Number of entities in collection
            target_recall: Target recall rate (0.0-1.0)
            
        Returns:
            Optimal search parameters
        """
        params = {}
        
        if index_type == IndexType.FLAT:
            # FLAT has no search parameters
            pass
            
        elif index_type in [IndexType.IVF_FLAT, IndexType.IVF_SQ8, IndexType.IVF_PQ]:
            # nprobe parameter for IVF-based indexes
            if collection_size:
                # Adaptive nprobe based on collection size
                if collection_size < 100000:
                    nprobe = 32
                elif collection_size < 1000000:
                    nprobe = 64
                else:
                    nprobe = 128
            else:
                nprobe = 64
            
            # Adjust for target recall
            if target_recall > 0.95:
                nprobe = min(nprobe * 2, 2048)
            elif target_recall < 0.8:
                nprobe = max(nprobe // 2, 1)
            
            params["nprobe"] = nprobe
            
        elif index_type == IndexType.HNSW:
            # ef parameter for HNSW
            if target_recall > 0.95:
                ef = 256
            elif target_recall > 0.9:
                ef = 128
            elif target_recall > 0.8:
                ef = 64
            else:
                ef = 32
            
            params["ef"] = ef
            
        elif index_type == IndexType.ANNOY:
            # search_k parameter for ANNOY
            if collection_size:
                search_k = min(collection_size // 100, 10000)
            else:
                search_k = 1000
            
            # Adjust for target recall
            if target_recall > 0.9:
                search_k = max(search_k * 2, 1000)
            
            params["search_k"] = search_k
        
        self.logger.debug(f"Optimal search params for {index_type.value}: {params}")
        return params
    
    def benchmark_search_performance(self,
                                   collection: MilvusCollection,
                                   query_vectors: List[List[float]],
                                   search_params: Optional[Dict[str, Any]] = None,
                                   limit: int = 10,
                                   iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark search performance with current index.
        
        Args:
            collection: Milvus collection
            query_vectors: Test query vectors
            search_params: Search parameters to test
            limit: Number of results per query
            iterations: Number of test iterations
            
        Returns:
            Performance benchmark results
        """
        try:
            if not collection.is_loaded():
                collection.load()
            
            # Get current index type
            indexes = self.list_indexes(collection)
            if not indexes:
                index_type = IndexType.FLAT
            else:
                index_type = IndexType(indexes[0]["index_type"])
            
            # Use default search params if not provided
            if search_params is None:
                search_params = self.get_optimal_search_params(index_type)
            
            latencies = []
            throughputs = []
            
            for i in range(iterations):
                start_time = time.time()
                
                result = collection.search(
                    query_vectors=query_vectors,
                    limit=limit,
                    search_params={"metric_type": "L2", "params": search_params}
                )
                
                end_time = time.time()
                query_time = end_time - start_time
                latencies.append(query_time)
                
                # Calculate throughput (queries per second)
                throughput = len(query_vectors) / query_time
                throughputs.append(throughput)
                
                # Record individual search performance
                search_perf = SearchPerformance(
                    query_time=query_time,
                    result_count=len(result.hits),
                    search_params=search_params,
                    index_type=index_type
                )
                
                with self._lock:
                    if collection.collection_name not in self._search_metrics:
                        self._search_metrics[collection.collection_name] = []
                    self._search_metrics[collection.collection_name].append(search_perf)
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            avg_throughput = sum(throughputs) / len(throughputs)
            
            benchmark_result = {
                "index_type": index_type.value,
                "search_params": search_params,
                "iterations": iterations,
                "query_count": len(query_vectors),
                "limit": limit,
                "average_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "average_throughput": avg_throughput,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Search benchmark completed: {avg_latency:.3f}s avg latency, "
                          f"{avg_throughput:.1f} QPS")
            
            return benchmark_result
            
        except Exception as e:
            self.logger.error(f"Search benchmark failed: {e}")
            raise PerformanceError(f"Benchmark failed: {e}")
    
    def get_performance_metrics(self, collection_name: str) -> Dict[str, Any]:
        """Get performance metrics for collection."""
        with self._lock:
            index_metrics = self._performance_metrics.get(collection_name, [])
            search_metrics = self._search_metrics.get(collection_name, [])
            
            return {
                "collection_name": collection_name,
                "index_performance": [
                    {
                        "index_type": metric.index_type.value,
                        "field_name": metric.field_name,
                        "build_time": metric.build_time,
                        "created_at": metric.created_at.isoformat()
                    }
                    for metric in index_metrics
                ],
                "search_performance": [
                    {
                        "query_time": metric.query_time,
                        "result_count": metric.result_count,
                        "search_params": metric.search_params,
                        "index_type": metric.index_type.value,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for metric in search_metrics[-10:]  # Last 10 searches
                ]
            }
    
    def recommend_index_type(self,
                           collection_size: int,
                           vector_dim: int,
                           query_frequency: str = "medium",
                           accuracy_requirement: str = "high") -> Tuple[IndexType, Dict[str, Any]]:
        """
        Recommend optimal index type based on data characteristics.
        
        Args:
            collection_size: Number of vectors in collection
            vector_dim: Vector dimension
            query_frequency: "low", "medium", "high"
            accuracy_requirement: "low", "medium", "high"
            
        Returns:
            Recommended index type and parameters
        """
        # Decision matrix based on collection characteristics
        if collection_size < 10000:
            if accuracy_requirement == "high":
                return IndexType.FLAT, {}
            else:
                return IndexType.IVF_FLAT, {"nlist": 128}
        
        elif collection_size < 1000000:
            if accuracy_requirement == "high" and query_frequency in ["low", "medium"]:
                return IndexType.IVF_FLAT, {"nlist": 1024}
            elif query_frequency == "high":
                return IndexType.HNSW, {"M": 16, "efConstruction": 200}
            else:
                return IndexType.IVF_SQ8, {"nlist": 1024}
        
        else:  # Large collections
            if query_frequency == "high":
                return IndexType.HNSW, {"M": 32, "efConstruction": 400}
            elif accuracy_requirement == "low":
                return IndexType.IVF_PQ, {"nlist": 2048, "m": 8}
            else:
                return IndexType.IVF_FLAT, {"nlist": 2048}
    
    def auto_optimize_index(self,
                          collection: MilvusCollection,
                          target_performance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Automatically optimize index for collection based on performance targets.
        
        Args:
            collection: Milvus collection
            target_performance: Performance targets (latency, throughput, recall)
            
        Returns:
            Optimization result
        """
        try:
            # Get collection statistics
            stats = collection.get_collection_stats()
            collection_size = stats["num_entities"]
            vector_dim = collection.schema.vector_dim
            
            # Default performance targets
            targets = target_performance or {
                "max_latency": 0.1,     # 100ms
                "min_throughput": 100,  # 100 QPS
                "min_recall": 0.9       # 90% recall
            }
            
            # Get recommendation
            recommended_type, recommended_params = self.recommend_index_type(
                collection_size=collection_size,
                vector_dim=vector_dim,
                query_frequency="medium",
                accuracy_requirement="high"
            )
            
            # Create optimized configuration
            config = IndexConfig(
                index_type=recommended_type,
                metric_type=MetricType.L2,
                field_name="vector",
                params=recommended_params,
                description=f"Auto-optimized {recommended_type.value} index"
            )
            
            # Apply the optimized index
            result = self.create_index(collection, index_config=config)
            
            # Reload collection for search
            collection.load()
            
            optimization_result = {
                "status": "success",
                "collection_size": collection_size,
                "vector_dim": vector_dim,
                "recommended_index": recommended_type.value,
                "recommended_params": recommended_params,
                "build_time": result["build_time"],
                "targets": targets
            }
            
            self.logger.info(f"Auto-optimization completed: {recommended_type.value} index")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Auto-optimization failed: {e}")
            raise PerformanceError(f"Index optimization failed: {e}")
    
    def get_index_recommendations(self) -> Dict[str, Any]:
        """Get index type recommendations and guidelines."""
        return {
            "index_types": {
                "FLAT": {
                    "description": "Default index with 100% recall",
                    "use_cases": ["Small datasets (<10K vectors)", "Highest accuracy required"],
                    "pros": ["Perfect recall", "No parameters needed"],
                    "cons": ["Slow for large datasets"]
                },
                "IVF_FLAT": {
                    "description": "Basic IVF index with original data",
                    "use_cases": ["Medium datasets (10K-1M vectors)", "Balanced performance"],
                    "pros": ["Good balance of speed and accuracy", "Configurable"],
                    "cons": ["Requires parameter tuning"]
                },
                "IVF_SQ8": {
                    "description": "IVF with 8-bit scalar quantization",
                    "use_cases": ["Large datasets with storage constraints"],
                    "pros": ["Smaller index size", "Good performance"],
                    "cons": ["Some accuracy loss"]
                },
                "IVF_PQ": {
                    "description": "IVF with product quantization",
                    "use_cases": ["Very large datasets (>1M vectors)"],
                    "pros": ["Smallest index size", "Scalable"],
                    "cons": ["Lower accuracy", "Complex parameters"]
                },
                "HNSW": {
                    "description": "Hierarchical navigable small world",
                    "use_cases": ["High-frequency queries", "Real-time applications"],
                    "pros": ["Fastest search", "Good recall"],
                    "cons": ["Larger memory usage", "Longer build time"]
                },
                "ANNOY": {
                    "description": "Approximate nearest neighbors",
                    "use_cases": ["Read-heavy workloads", "Static datasets"],
                    "pros": ["Memory efficient", "Fast queries"],
                    "cons": ["Read-only after build", "Parameter sensitive"]
                }
            },
            "selection_guidelines": {
                "small_dataset": "Use FLAT for <10K vectors",
                "medium_dataset": "Use IVF_FLAT for 10K-1M vectors",
                "large_dataset": "Use HNSW for >1M vectors with high QPS",
                "storage_constrained": "Use IVF_SQ8 or IVF_PQ",
                "high_accuracy": "Use FLAT or IVF_FLAT",
                "high_performance": "Use HNSW with optimized parameters"
            }
        }
    
    def clear_performance_metrics(self, collection_name: Optional[str] = None) -> None:
        """Clear performance metrics."""
        with self._lock:
            if collection_name:
                self._performance_metrics.pop(collection_name, None)
                self._search_metrics.pop(collection_name, None)
                self.logger.info(f"Cleared metrics for collection: {collection_name}")
            else:
                self._performance_metrics.clear()
                self._search_metrics.clear()
                self.logger.info("Cleared all performance metrics")


def create_index_manager(client: MilvusClient) -> IndexManager:
    """
    Create Index Manager instance.
    
    Args:
        client: Milvus client instance
        
    Returns:
        Configured Index Manager
    """
    return IndexManager(client=client)


def get_recommended_config(collection_size: int,
                         vector_dim: int,
                         performance_priority: str = "balanced") -> IndexConfig:
    """
    Get recommended index configuration.
    
    Args:
        collection_size: Number of vectors
        vector_dim: Vector dimension
        performance_priority: "speed", "accuracy", "balanced", "storage"
        
    Returns:
        Recommended index configuration
    """
    if performance_priority == "accuracy":
        return IndexConfig(
            index_type=IndexType.FLAT,
            metric_type=MetricType.L2,
            field_name="vector"
        )
    elif performance_priority == "speed":
        return IndexConfig(
            index_type=IndexType.HNSW,
            metric_type=MetricType.L2,
            field_name="vector",
            params={"M": 32, "efConstruction": 400}
        )
    elif performance_priority == "storage":
        return IndexConfig(
            index_type=IndexType.IVF_PQ,
            metric_type=MetricType.L2,
            field_name="vector",
            params={"nlist": 1024, "m": 8}
        )
    else:  # balanced
        nlist = min(max(collection_size // 39, 1), 65536)
        return IndexConfig(
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            field_name="vector",
            params={"nlist": nlist}
        )