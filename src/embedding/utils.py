"""
Embedding utilities for dimension management and optimization.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from datetime import datetime, timezone

from src.embedding.base import EmbeddingResponse, EmbeddingDimension
from src.core.logging import LoggerMixin


class DimensionReductionMethod(Enum):
    """Methods for dimension reduction."""
    PCA = "pca"
    TRUNCATE = "truncate"
    AVERAGE_POOLING = "average_pooling"
    MAX_POOLING = "max_pooling"


class SimilarityMetric(Enum):
    """Similarity metrics for embeddings."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class EmbeddingOptimizationConfig:
    """Configuration for embedding optimization."""
    target_dimensions: Optional[int] = None
    reduction_method: DimensionReductionMethod = DimensionReductionMethod.PCA
    normalize_before_reduction: bool = True
    normalize_after_reduction: bool = True
    preserve_variance_ratio: float = 0.95
    batch_size: int = 1000


class EmbeddingOptimizer(LoggerMixin):
    """
    Embedding optimizer for dimension management and performance optimization.
    
    Features:
    - Dimension reduction (PCA, truncation, pooling)
    - Similarity computation optimization
    - Batch processing optimization
    - Memory usage optimization
    - Performance profiling
    """
    
    def __init__(self, config: EmbeddingOptimizationConfig):
        """Initialize embedding optimizer."""
        self.config = config
        self._pca_model = None
        self._fit_data = None
        
    def reduce_dimensions(self, embeddings: Union[EmbeddingResponse, List[List[float]]], 
                         target_dimensions: Optional[int] = None) -> EmbeddingResponse:
        """
        Reduce embedding dimensions.
        
        Args:
            embeddings: Embedding response or list of embeddings
            target_dimensions: Target dimensions (overrides config)
            
        Returns:
            Optimized embedding response
        """
        if isinstance(embeddings, EmbeddingResponse):
            embedding_vectors = embeddings.embeddings
            original_response = embeddings
        else:
            embedding_vectors = embeddings
            original_response = None
        
        if not embedding_vectors:
            raise ValueError("No embeddings to optimize")
        
        target_dims = target_dimensions or self.config.target_dimensions
        current_dims = len(embedding_vectors[0])
        
        if not target_dims or target_dims >= current_dims:
            return original_response or EmbeddingResponse(
                embeddings=embedding_vectors,
                model="optimized",
                usage=None,
                dimensions=current_dims
            )
        
        # Convert to numpy array
        embeddings_array = np.array(embedding_vectors)
        
        # Normalize before reduction if configured
        if self.config.normalize_before_reduction:
            embeddings_array = self._normalize_embeddings(embeddings_array)
        
        # Apply reduction method
        if self.config.reduction_method == DimensionReductionMethod.PCA:
            reduced_embeddings = self._apply_pca(embeddings_array, target_dims)
        elif self.config.reduction_method == DimensionReductionMethod.TRUNCATE:
            reduced_embeddings = self._apply_truncation(embeddings_array, target_dims)
        elif self.config.reduction_method == DimensionReductionMethod.AVERAGE_POOLING:
            reduced_embeddings = self._apply_average_pooling(embeddings_array, target_dims)
        elif self.config.reduction_method == DimensionReductionMethod.MAX_POOLING:
            reduced_embeddings = self._apply_max_pooling(embeddings_array, target_dims)
        else:
            raise ValueError(f"Unsupported reduction method: {self.config.reduction_method}")
        
        # Normalize after reduction if configured
        if self.config.normalize_after_reduction:
            reduced_embeddings = self._normalize_embeddings(reduced_embeddings)
        
        # Create optimized response
        if original_response:
            optimized_response = EmbeddingResponse(
                embeddings=reduced_embeddings.tolist(),
                model=f"{original_response.model}-optimized",
                usage=original_response.usage,
                dimensions=target_dims,
                response_time=original_response.response_time,
                metadata={
                    **original_response.metadata,
                    "optimized": True,
                    "original_dimensions": current_dims,
                    "target_dimensions": target_dims,
                    "reduction_method": self.config.reduction_method.value,
                    "compression_ratio": target_dims / current_dims
                }
            )
        else:
            optimized_response = EmbeddingResponse(
                embeddings=reduced_embeddings.tolist(),
                model="optimized",
                usage=None,
                dimensions=target_dims,
                metadata={
                    "optimized": True,
                    "original_dimensions": current_dims,
                    "target_dimensions": target_dims,
                    "reduction_method": self.config.reduction_method.value,
                    "compression_ratio": target_dims / current_dims
                }
            )
        
        return optimized_response
    
    def _apply_pca(self, embeddings: np.ndarray, target_dims: int) -> np.ndarray:
        """Apply PCA dimension reduction."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("sklearn is required for PCA dimension reduction")
        
        if self._pca_model is None or self._pca_model.n_components != target_dims:
            self._pca_model = PCA(n_components=target_dims)
            self._pca_model.fit(embeddings)
            
            # Log variance explained
            variance_ratio = np.sum(self._pca_model.explained_variance_ratio_)
            self.logger.info(f"PCA preserves {variance_ratio:.3f} of original variance")
        
        return self._pca_model.transform(embeddings)
    
    def _apply_truncation(self, embeddings: np.ndarray, target_dims: int) -> np.ndarray:
        """Apply simple truncation."""
        return embeddings[:, :target_dims]
    
    def _apply_average_pooling(self, embeddings: np.ndarray, target_dims: int) -> np.ndarray:
        """Apply average pooling reduction."""
        original_dims = embeddings.shape[1]
        pool_size = original_dims // target_dims
        
        # Reshape and pool
        pooled_embeddings = []
        for i in range(target_dims):
            start_idx = i * pool_size
            end_idx = min((i + 1) * pool_size, original_dims)
            pooled_value = np.mean(embeddings[:, start_idx:end_idx], axis=1)
            pooled_embeddings.append(pooled_value)
        
        return np.array(pooled_embeddings).T
    
    def _apply_max_pooling(self, embeddings: np.ndarray, target_dims: int) -> np.ndarray:
        """Apply max pooling reduction."""
        original_dims = embeddings.shape[1]
        pool_size = original_dims // target_dims
        
        # Reshape and pool
        pooled_embeddings = []
        for i in range(target_dims):
            start_idx = i * pool_size
            end_idx = min((i + 1) * pool_size, original_dims)
            pooled_value = np.max(embeddings[:, start_idx:end_idx], axis=1)
            pooled_embeddings.append(pooled_value)
        
        return np.array(pooled_embeddings).T
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def compute_similarity(self, embeddings1: Union[EmbeddingResponse, np.ndarray], 
                          embeddings2: Union[EmbeddingResponse, np.ndarray],
                          metric: SimilarityMetric = SimilarityMetric.COSINE) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric to use
            
        Returns:
            Similarity matrix
        """
        # Convert to numpy arrays
        if isinstance(embeddings1, EmbeddingResponse):
            arr1 = np.array(embeddings1.embeddings)
        else:
            arr1 = np.array(embeddings1)
        
        if isinstance(embeddings2, EmbeddingResponse):
            arr2 = np.array(embeddings2.embeddings)
        else:
            arr2 = np.array(embeddings2)
        
        # Compute similarity based on metric
        if metric == SimilarityMetric.COSINE:
            # Normalize vectors
            arr1_normalized = arr1 / np.linalg.norm(arr1, axis=1, keepdims=True)
            arr2_normalized = arr2 / np.linalg.norm(arr2, axis=1, keepdims=True)
            return np.dot(arr1_normalized, arr2_normalized.T)
        
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (smaller is more similar)
            return np.sqrt(np.sum((arr1[:, np.newaxis] - arr2[np.newaxis, :]) ** 2, axis=2))
        
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return np.dot(arr1, arr2.T)
        
        elif metric == SimilarityMetric.MANHATTAN:
            # Manhattan distance (smaller is more similar)
            return np.sum(np.abs(arr1[:, np.newaxis] - arr2[np.newaxis, :]), axis=2)
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def optimize_batch_processing(self, embeddings: List[List[float]], 
                                 batch_size: Optional[int] = None) -> List[List[List[float]]]:
        """
        Optimize embeddings for batch processing.
        
        Args:
            embeddings: List of embeddings
            batch_size: Batch size (overrides config)
            
        Returns:
            List of batches
        """
        batch_size = batch_size or self.config.batch_size
        
        batches = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def estimate_memory_usage(self, num_embeddings: int, dimensions: int, 
                            dtype: str = "float32") -> Dict[str, Any]:
        """
        Estimate memory usage for embeddings.
        
        Args:
            num_embeddings: Number of embeddings
            dimensions: Embedding dimensions
            dtype: Data type
            
        Returns:
            Memory usage information
        """
        # Calculate bytes per value
        bytes_per_value = {
            "float32": 4,
            "float64": 8,
            "int32": 4,
            "int64": 8
        }.get(dtype, 4)
        
        # Calculate total memory
        total_bytes = num_embeddings * dimensions * bytes_per_value
        
        # Convert to human-readable format
        memory_mb = total_bytes / (1024 * 1024)
        memory_gb = memory_mb / 1024
        
        return {
            "num_embeddings": num_embeddings,
            "dimensions": dimensions,
            "dtype": dtype,
            "bytes_per_value": bytes_per_value,
            "total_bytes": total_bytes,
            "memory_mb": memory_mb,
            "memory_gb": memory_gb,
            "recommended_batch_size": max(1, int(100 * 1024 * 1024 / (dimensions * bytes_per_value)))  # 100MB batches
        }
    
    def profile_performance(self, embeddings: List[List[float]], 
                          operations: List[str] = None) -> Dict[str, Any]:
        """
        Profile embedding operations performance.
        
        Args:
            embeddings: Test embeddings
            operations: Operations to profile
            
        Returns:
            Performance metrics
        """
        import time
        
        if operations is None:
            operations = ["normalize", "similarity", "reduction"]
        
        embeddings_array = np.array(embeddings)
        results = {}
        
        # Profile normalization
        if "normalize" in operations:
            start_time = time.time()
            normalized = self._normalize_embeddings(embeddings_array)
            results["normalize"] = {
                "time_seconds": time.time() - start_time,
                "input_shape": embeddings_array.shape,
                "output_shape": normalized.shape
            }
        
        # Profile similarity computation
        if "similarity" in operations:
            start_time = time.time()
            similarity_matrix = self.compute_similarity(embeddings_array, embeddings_array)
            results["similarity"] = {
                "time_seconds": time.time() - start_time,
                "input_shape": embeddings_array.shape,
                "output_shape": similarity_matrix.shape,
                "matrix_size": similarity_matrix.size
            }
        
        # Profile dimension reduction
        if "reduction" in operations and self.config.target_dimensions:
            start_time = time.time()
            reduced = self.reduce_dimensions(embeddings)
            results["reduction"] = {
                "time_seconds": time.time() - start_time,
                "original_dimensions": embeddings_array.shape[1],
                "target_dimensions": self.config.target_dimensions,
                "compression_ratio": self.config.target_dimensions / embeddings_array.shape[1]
            }
        
        return results
    
    def get_dimension_analysis(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze embedding dimensions.
        
        Args:
            embeddings: List of embeddings
            
        Returns:
            Dimension analysis
        """
        embeddings_array = np.array(embeddings)
        
        # Basic statistics
        mean_values = np.mean(embeddings_array, axis=0)
        std_values = np.std(embeddings_array, axis=0)
        min_values = np.min(embeddings_array, axis=0)
        max_values = np.max(embeddings_array, axis=0)
        
        # Variance analysis
        variance_per_dimension = np.var(embeddings_array, axis=0)
        total_variance = np.sum(variance_per_dimension)
        variance_ratio = variance_per_dimension / total_variance
        
        # Dimension importance (based on variance)
        dimension_importance = np.argsort(variance_ratio)[::-1]
        
        return {
            "num_embeddings": len(embeddings),
            "dimensions": len(embeddings[0]),
            "statistics": {
                "mean": {
                    "overall": float(np.mean(mean_values)),
                    "std": float(np.std(mean_values)),
                    "min": float(np.min(mean_values)),
                    "max": float(np.max(mean_values))
                },
                "std": {
                    "overall": float(np.mean(std_values)),
                    "std": float(np.std(std_values)),
                    "min": float(np.min(std_values)),
                    "max": float(np.max(std_values))
                },
                "range": {
                    "overall": float(np.mean(max_values - min_values)),
                    "std": float(np.std(max_values - min_values)),
                    "min": float(np.min(max_values - min_values)),
                    "max": float(np.max(max_values - min_values))
                }
            },
            "variance": {
                "total": float(total_variance),
                "per_dimension": variance_per_dimension.tolist(),
                "normalized": variance_ratio.tolist(),
                "top_10_dimensions": dimension_importance[:10].tolist()
            },
            "recommendations": {
                "preserve_variance_dims": int(np.sum(np.cumsum(np.sort(variance_ratio)[::-1]) <= self.config.preserve_variance_ratio)),
                "low_variance_dims": int(np.sum(variance_ratio < 0.001)),
                "high_variance_dims": int(np.sum(variance_ratio > 0.01))
            }
        }


class EmbeddingCache:
    """
    Advanced caching system for embeddings with LRU eviction and persistence.
    """
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600, persist_file: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time to live in seconds
            persist_file: File path for cache persistence
        """
        self.max_size = max_size
        self.ttl = ttl
        self.persist_file = persist_file
        self.cache = {}
        self.access_times = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        
        # Load from file if specified
        if self.persist_file:
            self.load_from_file()
    
    def get_cache_key(self, text: str, model: str, dimensions: Optional[int] = None) -> str:
        """Generate cache key for text embedding."""
        key_data = {
            "text": text,
            "model": model,
            "dimensions": dimensions
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now(timezone.utc).timestamp() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            self.stats["evictions"] += 1
            self.stats["misses"] += 1
            return None
        
        # Update access time
        self.access_times[key] = datetime.now(timezone.utc).timestamp()
        self.stats["hits"] += 1
        
        return entry["embedding"]
    
    def put(self, key: str, embedding: List[float]) -> None:
        """Put embedding in cache."""
        # Check if we need to evict
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Store entry
        self.cache[key] = {
            "embedding": embedding,
            "timestamp": datetime.now(timezone.utc).timestamp()
        }
        self.access_times[key] = datetime.now(timezone.utc).timestamp()
        self.stats["size"] = len(self.cache)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove from cache
        del self.cache[lru_key]
        del self.access_times[lru_key]
        
        self.stats["evictions"] += 1
        self.stats["size"] = len(self.cache)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hit_rate": hit_rate,
            "memory_usage": sum(len(str(entry["embedding"])) for entry in self.cache.values())
        }
    
    def save_to_file(self) -> None:
        """Save cache to file."""
        if not self.persist_file:
            return
        
        try:
            with open(self.persist_file, 'w') as f:
                json.dump({
                    "cache": self.cache,
                    "access_times": self.access_times,
                    "stats": self.stats
                }, f)
        except Exception as e:
            print(f"Failed to save cache to file: {e}")
    
    def load_from_file(self) -> None:
        """Load cache from file."""
        if not self.persist_file:
            return
        
        try:
            with open(self.persist_file, 'r') as f:
                data = json.load(f)
                self.cache = data.get("cache", {})
                self.access_times = data.get("access_times", {})
                self.stats = data.get("stats", {
                    "hits": 0,
                    "misses": 0,
                    "evictions": 0,
                    "size": len(self.cache)
                })
        except Exception as e:
            print(f"Failed to load cache from file: {e}")