"""
Advanced Milvus Indexing Strategies for Vector Pipeline.
"""

import asyncio
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.core.logging import LoggerMixin
from src.core.exceptions import PipelineError, IndexError
from src.milvus.index import IndexManager, IndexType, MetricType, IndexConfig, IndexPerformance
from src.milvus.collection import MilvusCollection
from src.milvus.client import MilvusClient
from .monitoring import PipelineMetrics


class IndexStrategy(Enum):
    """Advanced indexing strategies."""
    ADAPTIVE = "adaptive"  # Automatically adapt based on data characteristics
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Optimize for query performance
    STORAGE_OPTIMIZED = "storage_optimized"  # Optimize for storage efficiency
    BALANCED = "balanced"  # Balance between performance and storage
    MULTI_TIER = "multi_tier"  # Use different indexes for different data tiers
    HYBRID = "hybrid"  # Combine multiple index types


class DataTier(Enum):
    """Data tiers for multi-tier indexing."""
    HOT = "hot"  # Frequently accessed data
    WARM = "warm"  # Occasionally accessed data
    COLD = "cold"  # Rarely accessed data
    ARCHIVE = "archive"  # Historical data


@dataclass
class IndexingProfile:
    """Indexing profile for different workload patterns."""
    name: str
    description: str
    primary_index: IndexType
    primary_params: Dict[str, Any]
    secondary_index: Optional[IndexType] = None
    secondary_params: Optional[Dict[str, Any]] = None
    metric_type: MetricType = MetricType.L2
    query_optimization: Dict[str, Any] = field(default_factory=dict)
    memory_optimization: bool = False
    storage_optimization: bool = False


@dataclass
class IndexStrategyConfig:
    """Configuration for index strategy."""
    strategy: IndexStrategy = IndexStrategy.ADAPTIVE
    collection_size_threshold: int = 100000
    query_frequency_threshold: float = 100.0  # queries per minute
    memory_limit_gb: float = 8.0
    storage_limit_gb: float = 100.0
    target_recall: float = 0.9
    target_latency_ms: float = 100.0
    target_throughput_qps: float = 100.0
    enable_auto_rebalancing: bool = True
    enable_performance_monitoring: bool = True
    rebalancing_interval_hours: int = 24


@dataclass
class IndexRecommendation:
    """Index recommendation with rationale."""
    index_type: IndexType
    params: Dict[str, Any]
    metric_type: MetricType
    estimated_performance: Dict[str, float]
    rationale: str
    confidence_score: float
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedIndexStrategy(LoggerMixin):
    """
    Advanced indexing strategy manager for vector pipeline.
    
    Provides intelligent index selection and optimization based on:
    - Data characteristics (size, dimension, distribution)
    - Query patterns (frequency, latency requirements)
    - Resource constraints (memory, storage)
    - Performance targets (recall, throughput)
    """
    
    def __init__(
        self,
        index_manager: IndexManager,
        config: IndexStrategyConfig,
        metrics: Optional[PipelineMetrics] = None
    ):
        """
        Initialize advanced index strategy.
        
        Args:
            index_manager: Base index manager
            config: Strategy configuration
            metrics: Pipeline metrics for monitoring
        """
        self.index_manager = index_manager
        self.config = config
        self.metrics = metrics
        
        # Predefined indexing profiles
        self.profiles = self._create_indexing_profiles()
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recommendations_cache: Dict[str, IndexRecommendation] = {}
        
        # Strategy state
        self.current_strategies: Dict[str, IndexStrategy] = {}
        self.last_optimization: Dict[str, datetime] = {}
    
    def _create_indexing_profiles(self) -> Dict[str, IndexingProfile]:
        """Create predefined indexing profiles for different scenarios."""
        profiles = {}
        
        # High-performance profile for real-time applications
        profiles["high_performance"] = IndexingProfile(
            name="high_performance",
            description="Optimized for lowest latency and highest throughput",
            primary_index=IndexType.HNSW,
            primary_params={"M": 32, "efConstruction": 400},
            query_optimization={
                "ef": 128,
                "enable_prefetch": True,
                "cache_size": "large"
            }
        )
        
        # Memory-efficient profile for resource-constrained environments
        profiles["memory_efficient"] = IndexingProfile(
            name="memory_efficient",
            description="Optimized for minimal memory usage",
            primary_index=IndexType.IVF_SQ8,
            primary_params={"nlist": 1024},
            memory_optimization=True,
            query_optimization={
                "nprobe": 32,
                "enable_compression": True
            }
        )
        
        # Storage-efficient profile for large datasets
        profiles["storage_efficient"] = IndexingProfile(
            name="storage_efficient",
            description="Optimized for minimal storage footprint",
            primary_index=IndexType.IVF_PQ,
            primary_params={"nlist": 2048, "m": 8, "nbits": 8},
            storage_optimization=True,
            query_optimization={
                "nprobe": 64
            }
        )
        
        # Balanced profile for general use
        profiles["balanced"] = IndexingProfile(
            name="balanced",
            description="Balanced performance, memory, and storage",
            primary_index=IndexType.IVF_FLAT,
            primary_params={"nlist": 1024},
            query_optimization={
                "nprobe": 64
            }
        )
        
        # Multi-tier profile for heterogeneous data
        profiles["multi_tier"] = IndexingProfile(
            name="multi_tier",
            description="Different indexes for different data tiers",
            primary_index=IndexType.HNSW,  # For hot data
            primary_params={"M": 16, "efConstruction": 200},
            secondary_index=IndexType.IVF_SQ8,  # For warm/cold data
            secondary_params={"nlist": 2048},
            query_optimization={
                "adaptive_routing": True
            }
        )
        
        # High-recall profile for critical applications
        profiles["high_recall"] = IndexingProfile(
            name="high_recall",
            description="Optimized for maximum recall accuracy",
            primary_index=IndexType.FLAT,
            primary_params={},
            secondary_index=IndexType.IVF_FLAT,
            secondary_params={"nlist": 512},
            query_optimization={
                "exhaustive_search": True
            }
        )
        
        return profiles
    
    async def analyze_collection_characteristics(self, collection: MilvusCollection) -> Dict[str, Any]:
        """
        Analyze collection characteristics for index strategy selection.
        
        Args:
            collection: Milvus collection to analyze
            
        Returns:
            Collection characteristics analysis
        """
        try:
            # Get basic collection statistics
            stats = collection.get_collection_stats()
            collection_size = stats.get("num_entities", 0)
            
            # Get schema information
            schema = collection.schema
            vector_dim = getattr(schema, 'vector_dim', 0)
            
            # Analyze data distribution (placeholder - would need actual vector sampling)
            data_analysis = await self._analyze_data_distribution(collection)
            
            # Analyze query patterns (if metrics available)
            query_patterns = await self._analyze_query_patterns(collection.collection_name)
            
            characteristics = {
                "collection_name": collection.collection_name,
                "size": collection_size,
                "vector_dimension": vector_dim,
                "data_distribution": data_analysis,
                "query_patterns": query_patterns,
                "resource_requirements": self._estimate_resource_requirements(collection_size, vector_dim),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Analyzed collection characteristics: {collection.collection_name}")
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze collection characteristics: {e}")
            raise PipelineError(f"Collection analysis failed: {e}")
    
    async def _analyze_data_distribution(self, collection: MilvusCollection) -> Dict[str, Any]:
        """Analyze vector data distribution patterns."""
        # This is a placeholder implementation
        # In practice, you would sample vectors and analyze their distribution
        return {
            "sparsity": "unknown",
            "clustering": "unknown",
            "dimensionality_distribution": "uniform",
            "outlier_percentage": 0.05
        }
    
    async def _analyze_query_patterns(self, collection_name: str) -> Dict[str, Any]:
        """Analyze historical query patterns."""
        if not self.metrics:
            return {"frequency": "unknown", "latency_sensitivity": "medium"}
        
        # Get query metrics from pipeline metrics
        try:
            performance_summary = self.metrics.get_performance_summary()
            
            return {
                "frequency": "medium",  # Would be calculated from actual metrics
                "latency_sensitivity": "medium",
                "typical_topk": 10,
                "query_complexity": "simple"
            }
        except Exception:
            return {"frequency": "unknown", "latency_sensitivity": "medium"}
    
    def _estimate_resource_requirements(self, collection_size: int, vector_dim: int) -> Dict[str, Any]:
        """Estimate resource requirements for different index types."""
        # Rough estimates based on common patterns
        vector_size_mb = collection_size * vector_dim * 4 / (1024 * 1024)  # float32
        
        requirements = {
            "FLAT": {
                "memory_mb": vector_size_mb,
                "storage_mb": vector_size_mb,
                "build_time_estimate": collection_size * 0.001  # seconds
            },
            "IVF_FLAT": {
                "memory_mb": vector_size_mb * 1.2,
                "storage_mb": vector_size_mb * 1.1,
                "build_time_estimate": collection_size * 0.01
            },
            "IVF_SQ8": {
                "memory_mb": vector_size_mb * 0.8,
                "storage_mb": vector_size_mb * 0.25,
                "build_time_estimate": collection_size * 0.015
            },
            "IVF_PQ": {
                "memory_mb": vector_size_mb * 0.6,
                "storage_mb": vector_size_mb * 0.1,
                "build_time_estimate": collection_size * 0.02
            },
            "HNSW": {
                "memory_mb": vector_size_mb * 2.0,
                "storage_mb": vector_size_mb * 1.5,
                "build_time_estimate": collection_size * 0.05
            }
        }
        
        return requirements
    
    async def recommend_index_strategy(
        self,
        collection: MilvusCollection,
        workload_profile: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> IndexRecommendation:
        """
        Recommend optimal index strategy for collection.
        
        Args:
            collection: Milvus collection
            workload_profile: Predefined workload profile name
            constraints: Resource and performance constraints
            
        Returns:
            Index recommendation with rationale
        """
        try:
            # Analyze collection characteristics
            characteristics = await self.analyze_collection_characteristics(collection)
            
            # Apply constraints
            constraints = constraints or {}
            memory_limit = constraints.get("memory_limit_gb", self.config.memory_limit_gb)
            storage_limit = constraints.get("storage_limit_gb", self.config.storage_limit_gb)
            target_recall = constraints.get("target_recall", self.config.target_recall)
            target_latency = constraints.get("target_latency_ms", self.config.target_latency_ms)
            
            # Get resource requirements
            resource_reqs = characteristics["resource_requirements"]
            collection_size = characteristics["size"]
            vector_dim = characteristics["vector_dimension"]
            
            # Decision logic based on multiple factors
            recommendation = self._select_optimal_index(
                collection_size=collection_size,
                vector_dim=vector_dim,
                resource_requirements=resource_reqs,
                memory_limit_gb=memory_limit,
                storage_limit_gb=storage_limit,
                target_recall=target_recall,
                target_latency_ms=target_latency,
                workload_profile=workload_profile
            )
            
            # Cache recommendation
            cache_key = f"{collection.collection_name}_{hash(str(constraints))}"
            self.recommendations_cache[cache_key] = recommendation
            
            self.logger.info(f"Generated index recommendation for {collection.collection_name}: {recommendation.index_type.value}")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to recommend index strategy: {e}")
            raise PipelineError(f"Index recommendation failed: {e}")
    
    def _select_optimal_index(
        self,
        collection_size: int,
        vector_dim: int,
        resource_requirements: Dict[str, Any],
        memory_limit_gb: float,
        storage_limit_gb: float,
        target_recall: float,
        target_latency_ms: float,
        workload_profile: Optional[str] = None
    ) -> IndexRecommendation:
        """Select optimal index based on comprehensive analysis."""
        
        # Convert limits to MB
        memory_limit_mb = memory_limit_gb * 1024
        storage_limit_mb = storage_limit_gb * 1024
        
        candidates = []
        
        # Evaluate each index type
        for index_type in IndexType:
            if index_type.value not in resource_requirements:
                continue
            
            reqs = resource_requirements[index_type.value]
            
            # Check resource constraints
            if reqs["memory_mb"] > memory_limit_mb or reqs["storage_mb"] > storage_limit_mb:
                continue
            
            # Estimate performance characteristics
            estimated_perf = self._estimate_performance(index_type, collection_size, vector_dim)
            
            # Calculate fitness score
            fitness_score = self._calculate_fitness_score(
                index_type=index_type,
                estimated_performance=estimated_perf,
                resource_usage=reqs,
                target_recall=target_recall,
                target_latency_ms=target_latency_ms,
                memory_limit_mb=memory_limit_mb,
                storage_limit_mb=storage_limit_mb
            )
            
            candidates.append({
                "index_type": index_type,
                "estimated_performance": estimated_perf,
                "resource_usage": reqs,
                "fitness_score": fitness_score
            })
        
        # Sort by fitness score
        candidates.sort(key=lambda x: x["fitness_score"], reverse=True)
        
        if not candidates:
            # Fallback to FLAT if no candidates meet constraints
            return IndexRecommendation(
                index_type=IndexType.FLAT,
                params={},
                metric_type=MetricType.L2,
                estimated_performance={"recall": 1.0, "latency_ms": 50.0, "throughput_qps": 10.0},
                rationale="Fallback to FLAT index due to resource constraints",
                confidence_score=0.5
            )
        
        # Select best candidate
        best = candidates[0]
        
        # Generate optimized parameters
        params = self._generate_optimized_params(
            best["index_type"],
            collection_size,
            vector_dim,
            target_recall
        )
        
        # Generate rationale
        rationale = self._generate_rationale(best, collection_size, memory_limit_gb, storage_limit_gb)
        
        # Prepare alternative options
        alternatives = [
            {
                "index_type": candidate["index_type"].value,
                "fitness_score": candidate["fitness_score"],
                "estimated_performance": candidate["estimated_performance"]
            }
            for candidate in candidates[1:3]  # Top 2 alternatives
        ]
        
        return IndexRecommendation(
            index_type=best["index_type"],
            params=params,
            metric_type=MetricType.L2,
            estimated_performance=best["estimated_performance"],
            rationale=rationale,
            confidence_score=min(best["fitness_score"] / 100.0, 1.0),
            alternative_options=alternatives
        )
    
    def _estimate_performance(self, index_type: IndexType, collection_size: int, vector_dim: int) -> Dict[str, float]:
        """Estimate performance characteristics for index type."""
        # These are rough estimates based on empirical data
        # In practice, you'd have more sophisticated models
        
        if index_type == IndexType.FLAT:
            return {
                "recall": 1.0,
                "latency_ms": max(1.0, collection_size / 10000),
                "throughput_qps": max(1.0, 10000 / collection_size),
                "memory_efficiency": 1.0,
                "storage_efficiency": 1.0
            }
        elif index_type == IndexType.IVF_FLAT:
            return {
                "recall": 0.95,
                "latency_ms": max(1.0, collection_size / 50000),
                "throughput_qps": max(10.0, 50000 / collection_size),
                "memory_efficiency": 0.9,
                "storage_efficiency": 0.9
            }
        elif index_type == IndexType.IVF_SQ8:
            return {
                "recall": 0.90,
                "latency_ms": max(1.0, collection_size / 60000),
                "throughput_qps": max(20.0, 60000 / collection_size),
                "memory_efficiency": 1.2,
                "storage_efficiency": 4.0
            }
        elif index_type == IndexType.IVF_PQ:
            return {
                "recall": 0.85,
                "latency_ms": max(2.0, collection_size / 70000),
                "throughput_qps": max(30.0, 70000 / collection_size),
                "memory_efficiency": 1.5,
                "storage_efficiency": 10.0
            }
        elif index_type == IndexType.HNSW:
            return {
                "recall": 0.95,
                "latency_ms": max(0.5, math.log10(collection_size)),
                "throughput_qps": max(100.0, 100000 / math.log10(max(collection_size, 10))),
                "memory_efficiency": 0.5,
                "storage_efficiency": 0.7
            }
        else:
            return {
                "recall": 0.8,
                "latency_ms": 10.0,
                "throughput_qps": 50.0,
                "memory_efficiency": 1.0,
                "storage_efficiency": 1.0
            }
    
    def _calculate_fitness_score(
        self,
        index_type: IndexType,
        estimated_performance: Dict[str, float],
        resource_usage: Dict[str, Any],
        target_recall: float,
        target_latency_ms: float,
        memory_limit_mb: float,
        storage_limit_mb: float
    ) -> float:
        """Calculate fitness score for index candidate."""
        score = 0.0
        
        # Recall score (40% weight)
        recall_score = min(estimated_performance["recall"] / target_recall, 1.0) * 40
        score += recall_score
        
        # Latency score (30% weight)
        if estimated_performance["latency_ms"] <= target_latency_ms:
            latency_score = 30
        else:
            latency_score = max(0, 30 * (target_latency_ms / estimated_performance["latency_ms"]))
        score += latency_score
        
        # Resource efficiency score (20% weight)
        memory_efficiency = 1.0 - (resource_usage["memory_mb"] / memory_limit_mb)
        storage_efficiency = 1.0 - (resource_usage["storage_mb"] / storage_limit_mb)
        resource_score = (memory_efficiency + storage_efficiency) / 2 * 20
        score += max(0, resource_score)
        
        # Throughput score (10% weight)
        throughput_score = min(estimated_performance["throughput_qps"] / 100.0, 1.0) * 10
        score += throughput_score
        
        return score
    
    def _generate_optimized_params(self, index_type: IndexType, collection_size: int, vector_dim: int, target_recall: float) -> Dict[str, Any]:
        """Generate optimized parameters for index type."""
        params = {}
        
        if index_type == IndexType.IVF_FLAT:
            # Optimize nlist based on collection size
            nlist = min(max(int(math.sqrt(collection_size)), 1), 65536)
            params["nlist"] = nlist
            
        elif index_type == IndexType.IVF_SQ8:
            nlist = min(max(int(math.sqrt(collection_size)), 1), 65536)
            params["nlist"] = nlist
            
        elif index_type == IndexType.IVF_PQ:
            nlist = min(max(int(math.sqrt(collection_size)), 1), 65536)
            # Optimize m based on vector dimension
            m = min(max(vector_dim // 64, 1), 64)
            params.update({"nlist": nlist, "m": m, "nbits": 8})
            
        elif index_type == IndexType.HNSW:
            # Optimize M and efConstruction based on recall target
            if target_recall > 0.95:
                M = 32
                efConstruction = 400
            elif target_recall > 0.9:
                M = 16
                efConstruction = 200
            else:
                M = 8
                efConstruction = 100
            params.update({"M": M, "efConstruction": efConstruction})
            
        elif index_type == IndexType.ANNOY:
            # Optimize n_trees based on collection size
            n_trees = min(max(int(math.log10(collection_size)), 1), 1024)
            params["n_trees"] = n_trees
        
        return params
    
    def _generate_rationale(self, candidate: Dict[str, Any], collection_size: int, memory_limit_gb: float, storage_limit_gb: float) -> str:
        """Generate human-readable rationale for index selection."""
        index_type = candidate["index_type"]
        perf = candidate["estimated_performance"]
        
        rationale_parts = []
        
        # Primary reason
        if index_type == IndexType.FLAT:
            rationale_parts.append("FLAT index selected for guaranteed 100% recall")
        elif index_type == IndexType.HNSW:
            rationale_parts.append("HNSW index selected for optimal query performance")
        elif index_type == IndexType.IVF_FLAT:
            rationale_parts.append("IVF_FLAT index selected for balanced performance and accuracy")
        elif index_type in [IndexType.IVF_SQ8, IndexType.IVF_PQ]:
            rationale_parts.append(f"{index_type.value} index selected for storage efficiency")
        
        # Performance characteristics
        rationale_parts.append(f"Expected recall: {perf['recall']:.1%}")
        rationale_parts.append(f"Estimated latency: {perf['latency_ms']:.1f}ms")
        rationale_parts.append(f"Expected throughput: {perf['throughput_qps']:.0f} QPS")
        
        # Resource considerations
        if collection_size > 1000000:
            rationale_parts.append("Large dataset size considered")
        if memory_limit_gb < 16:
            rationale_parts.append("Memory constraints considered")
        if storage_limit_gb < 100:
            rationale_parts.append("Storage constraints considered")
        
        return ". ".join(rationale_parts) + "."
    
    async def apply_index_strategy(
        self,
        collection: MilvusCollection,
        strategy: Optional[IndexStrategy] = None,
        recommendation: Optional[IndexRecommendation] = None
    ) -> Dict[str, Any]:
        """
        Apply index strategy to collection.
        
        Args:
            collection: Milvus collection
            strategy: Index strategy to apply
            recommendation: Specific index recommendation
            
        Returns:
            Application result
        """
        try:
            strategy = strategy or self.config.strategy
            
            if recommendation is None:
                recommendation = await self.recommend_index_strategy(collection)
            
            # Create index configuration
            index_config = IndexConfig(
                index_type=recommendation.index_type,
                metric_type=recommendation.metric_type,
                field_name="vector",  # Assuming vector field name
                params=recommendation.params
            )
            
            # Apply the index
            result = self.index_manager.create_index(
                collection=collection,
                index_config=index_config,
                wait_for_completion=True
            )
            
            # Track strategy application
            self.current_strategies[collection.collection_name] = strategy
            self.last_optimization[collection.collection_name] = datetime.utcnow()
            
            # Record performance baseline
            if collection.collection_name not in self.performance_history:
                self.performance_history[collection.collection_name] = []
            
            self.performance_history[collection.collection_name].append({
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": strategy.value,
                "index_type": recommendation.index_type.value,
                "build_time": result["build_time"],
                "confidence_score": recommendation.confidence_score
            })
            
            application_result = {
                "status": "success",
                "collection_name": collection.collection_name,
                "strategy": strategy.value,
                "index_type": recommendation.index_type.value,
                "params": recommendation.params,
                "build_time": result["build_time"],
                "rationale": recommendation.rationale,
                "confidence_score": recommendation.confidence_score,
                "estimated_performance": recommendation.estimated_performance
            }
            
            self.logger.info(f"Applied index strategy to {collection.collection_name}: {recommendation.index_type.value}")
            return application_result
            
        except Exception as e:
            self.logger.error(f"Failed to apply index strategy: {e}")
            raise PipelineError(f"Index strategy application failed: {e}")
    
    async def monitor_and_optimize(self, collection: MilvusCollection) -> Dict[str, Any]:
        """
        Monitor index performance and apply optimizations if needed.
        
        Args:
            collection: Milvus collection to monitor
            
        Returns:
            Monitoring and optimization results
        """
        try:
            collection_name = collection.collection_name
            
            # Check if optimization is needed
            if not self._should_optimize(collection_name):
                return {"status": "no_optimization_needed", "collection_name": collection_name}
            
            # Get current performance metrics
            current_metrics = self.index_manager.get_performance_metrics(collection_name)
            
            # Analyze performance trends
            performance_analysis = self._analyze_performance_trends(collection_name, current_metrics)
            
            # Determine if reindexing is beneficial
            if performance_analysis["recommend_reindex"]:
                # Generate new recommendation
                new_recommendation = await self.recommend_index_strategy(collection)
                
                # Apply new strategy if significantly better
                if self._is_significantly_better(performance_analysis, new_recommendation):
                    optimization_result = await self.apply_index_strategy(collection, recommendation=new_recommendation)
                    optimization_result["reason"] = "Performance degradation detected"
                    return optimization_result
            
            return {
                "status": "monitored",
                "collection_name": collection_name,
                "performance_analysis": performance_analysis,
                "optimization_applied": False
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor and optimize: {e}")
            return {"status": "error", "error": str(e)}
    
    def _should_optimize(self, collection_name: str) -> bool:
        """Check if collection should be optimized."""
        if not self.config.enable_auto_rebalancing:
            return False
        
        last_opt = self.last_optimization.get(collection_name)
        if not last_opt:
            return True
        
        # Check if enough time has passed
        hours_since_last = (datetime.utcnow() - last_opt).total_seconds() / 3600
        return hours_since_last >= self.config.rebalancing_interval_hours
    
    def _analyze_performance_trends(self, collection_name: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends for collection."""
        history = self.performance_history.get(collection_name, [])
        
        if len(history) < 2:
            return {"recommend_reindex": False, "reason": "Insufficient history"}
        
        # Simple trend analysis (placeholder)
        # In practice, you'd implement sophisticated trend analysis
        latest_search_metrics = current_metrics.get("search_performance", [])
        
        if not latest_search_metrics:
            return {"recommend_reindex": False, "reason": "No search metrics"}
        
        avg_latency = sum(m["query_time"] for m in latest_search_metrics) / len(latest_search_metrics)
        
        return {
            "recommend_reindex": avg_latency > self.config.target_latency_ms / 1000,
            "current_avg_latency": avg_latency,
            "target_latency": self.config.target_latency_ms / 1000,
            "trend": "degrading" if avg_latency > self.config.target_latency_ms / 1000 else "stable"
        }
    
    def _is_significantly_better(self, performance_analysis: Dict[str, Any], recommendation: IndexRecommendation) -> bool:
        """Check if new recommendation is significantly better than current setup."""
        if recommendation.confidence_score < 0.8:
            return False
        
        # Check if estimated performance meets targets
        estimated_latency = recommendation.estimated_performance.get("latency_ms", float('inf'))
        return estimated_latency < self.config.target_latency_ms
    
    def get_strategy_status(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of index strategies."""
        if collection_name:
            return {
                "collection_name": collection_name,
                "current_strategy": self.current_strategies.get(collection_name, "none"),
                "last_optimization": self.last_optimization.get(collection_name, "never"),
                "performance_history": self.performance_history.get(collection_name, [])
            }
        else:
            return {
                "total_collections": len(self.current_strategies),
                "strategies": dict(self.current_strategies),
                "last_optimizations": {
                    name: opt.isoformat() if isinstance(opt, datetime) else opt
                    for name, opt in self.last_optimization.items()
                },
                "cache_size": len(self.recommendations_cache)
            }
    
    def get_available_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get available indexing profiles."""
        return {
            name: {
                "name": profile.name,
                "description": profile.description,
                "primary_index": profile.primary_index.value,
                "primary_params": profile.primary_params,
                "secondary_index": profile.secondary_index.value if profile.secondary_index else None,
                "memory_optimization": profile.memory_optimization,
                "storage_optimization": profile.storage_optimization
            }
            for name, profile in self.profiles.items()
        }