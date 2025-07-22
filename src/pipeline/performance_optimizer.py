"""
Performance Optimization System for Large Dataset Processing.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from src.core.logging import LoggerMixin
from src.core.exceptions import PipelineError, PerformanceError
from .monitoring import PipelineMetrics, ResourceMetrics
from .batch_processor import BatchProcessor, BatchConfig


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    CONSERVATIVE = "conservative"  # Safe optimizations only
    MODERATE = "moderate"  # Balanced optimizations
    AGGRESSIVE = "aggressive"  # Maximum performance optimizations
    CUSTOM = "custom"  # Custom optimization rules


class OptimizationTarget(Enum):
    """Primary optimization targets."""
    THROUGHPUT = "throughput"  # Maximize items processed per second
    LATENCY = "latency"  # Minimize processing time per item
    MEMORY = "memory"  # Minimize memory usage
    CPU = "cpu"  # Optimize CPU utilization
    BALANCED = "balanced"  # Balance all metrics


@dataclass
class PerformanceTarget:
    """Performance optimization targets."""
    target_throughput_per_second: Optional[float] = None
    max_latency_seconds: Optional[float] = None
    max_memory_usage_mb: Optional[float] = None
    max_cpu_usage_percent: Optional[float] = None
    min_batch_efficiency: Optional[float] = None  # Items processed / time spent


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    level: OptimizationLevel = OptimizationLevel.MODERATE
    target: OptimizationTarget = OptimizationTarget.BALANCED
    performance_targets: PerformanceTarget = field(default_factory=PerformanceTarget)
    
    # Resource management
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_io_optimization: bool = True
    enable_garbage_collection_tuning: bool = True
    
    # Concurrency settings
    enable_adaptive_concurrency: bool = True
    max_worker_threads: Optional[int] = None
    max_worker_processes: Optional[int] = None
    
    # Caching and prefetching
    enable_smart_caching: bool = True
    enable_prefetching: bool = True
    cache_size_mb: float = 512.0
    
    # Monitoring and tuning
    monitoring_interval_seconds: float = 10.0
    tuning_interval_seconds: float = 60.0
    performance_history_size: int = 100
    
    # Safety limits
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 90.0
    emergency_throttling_threshold: float = 95.0


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics."""
    timestamp: datetime
    throughput_per_second: float
    average_latency: float
    memory_usage_mb: float
    cpu_usage_percent: float
    batch_efficiency: float
    queue_size: int
    active_workers: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "throughput_per_second": self.throughput_per_second,
            "average_latency": self.average_latency,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "batch_efficiency": self.batch_efficiency,
            "queue_size": self.queue_size,
            "active_workers": self.active_workers
        }


@dataclass
class OptimizationAction:
    """Optimization action taken."""
    timestamp: datetime
    action_type: str
    description: str
    parameters: Dict[str, Any]
    expected_impact: str
    success: bool = False
    actual_impact: Optional[Dict[str, float]] = None


class PerformanceOptimizer(LoggerMixin):
    """
    Performance optimization system for large dataset processing.
    
    Features:
    - Adaptive resource management
    - Dynamic concurrency tuning
    - Memory optimization and garbage collection
    - Intelligent caching and prefetching
    - Real-time performance monitoring
    - Automatic bottleneck detection and resolution
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        batch_processor: Optional[BatchProcessor] = None,
        metrics: Optional[PipelineMetrics] = None
    ):
        """
        Initialize performance optimizer.
        
        Args:
            config: Optimization configuration
            batch_processor: Batch processor to optimize
            metrics: Pipeline metrics for monitoring
        """
        self.config = config
        self.batch_processor = batch_processor
        self.metrics = metrics
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=config.performance_history_size)
        self.optimization_actions: List[OptimizationAction] = []
        
        # Resource management
        self.system_resources = self._get_system_resources()
        self.current_workers = self._get_optimal_worker_count()
        self.current_batch_size = 100  # Default
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = int(config.cache_size_mb * 1024 * 1024)  # Convert to bytes
        
        # Threading
        self._optimization_lock = threading.Lock()
        self._is_optimizing = False
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Performance state
        self.last_optimization_time = datetime.utcnow()
        self.performance_baseline: Optional[PerformanceSnapshot] = None
        
        self.logger.info(f"Performance optimizer initialized with {config.level.value} level")
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_io_counters": psutil.disk_io_counters(),
            "network_io_counters": psutil.net_io_counters()
        }
    
    def _get_optimal_worker_count(self) -> int:
        """Calculate optimal worker count based on system resources."""
        cpu_count = self.system_resources["cpu_count"]
        memory_gb = self.system_resources["memory_total_gb"]
        
        if self.config.max_worker_threads:
            return min(self.config.max_worker_threads, cpu_count * 2)
        
        # Heuristic based on system resources and optimization target
        if self.config.target == OptimizationTarget.CPU:
            return cpu_count
        elif self.config.target == OptimizationTarget.MEMORY:
            return max(1, min(cpu_count, int(memory_gb / 2)))
        elif self.config.target == OptimizationTarget.THROUGHPUT:
            return min(cpu_count * 2, int(memory_gb))
        else:  # BALANCED or LATENCY
            return max(1, min(cpu_count, int(memory_gb / 1.5)))
    
    async def start_optimization(self) -> None:
        """Start the performance optimization system."""
        if self._is_optimizing:
            return
        
        self._is_optimizing = True
        self.logger.info("Starting performance optimization system")
        
        # Take baseline measurement
        self.performance_baseline = await self._take_performance_snapshot()
        
        # Start optimization loop
        self._optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop_optimization(self) -> None:
        """Stop the performance optimization system."""
        if not self._is_optimizing:
            return
        
        self._is_optimizing = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped performance optimization system")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._is_optimizing:
            try:
                # Take performance snapshot
                snapshot = await self._take_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Check if optimization is needed
                if await self._should_optimize(snapshot):
                    await self._perform_optimization(snapshot)
                
                # Wait for next optimization cycle
                await asyncio.sleep(self.config.tuning_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(self.config.tuning_interval_seconds)
    
    async def _take_performance_snapshot(self) -> PerformanceSnapshot:
        """Take a snapshot of current performance metrics."""
        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get batch processor metrics if available
        batch_metrics = {}
        if self.batch_processor:
            batch_metrics = self.batch_processor.get_metrics()
        
        # Calculate derived metrics
        throughput = batch_metrics.get("throughput_items_per_second", 0.0)
        latency = batch_metrics.get("average_processing_time", 0.0)
        queue_size = batch_metrics.get("queue_size", 0)
        active_jobs = batch_metrics.get("active_jobs", 0)
        
        # Calculate batch efficiency
        total_time = batch_metrics.get("total_processing_time", 1.0)
        total_items = batch_metrics.get("total_items_processed", 0)
        batch_efficiency = total_items / total_time if total_time > 0 else 0.0
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            throughput_per_second=throughput,
            average_latency=latency,
            memory_usage_mb=memory_info.used / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            batch_efficiency=batch_efficiency,
            queue_size=queue_size,
            active_workers=active_jobs
        )
    
    async def _should_optimize(self, snapshot: PerformanceSnapshot) -> bool:
        """Determine if optimization should be performed."""
        # Check emergency conditions
        if (snapshot.memory_usage_mb / 1024 > self.system_resources["memory_total_gb"] * 0.01 * self.config.emergency_throttling_threshold or
            snapshot.cpu_usage_percent > self.config.emergency_throttling_threshold):
            return True
        
        # Check if enough time has passed since last optimization
        time_since_last = (datetime.utcnow() - self.last_optimization_time).total_seconds()
        if time_since_last < self.config.tuning_interval_seconds:
            return False
        
        # Check performance targets
        targets = self.config.performance_targets
        
        if targets.target_throughput_per_second and snapshot.throughput_per_second < targets.target_throughput_per_second * 0.8:
            return True
        
        if targets.max_latency_seconds and snapshot.average_latency > targets.max_latency_seconds:
            return True
        
        if targets.max_memory_usage_mb and snapshot.memory_usage_mb > targets.max_memory_usage_mb:
            return True
        
        if targets.max_cpu_usage_percent and snapshot.cpu_usage_percent > targets.max_cpu_usage_percent:
            return True
        
        # Check for performance degradation
        if len(self.performance_history) >= 5:
            recent_avg = sum(s.throughput_per_second for s in list(self.performance_history)[-5:]) / 5
            baseline_throughput = self.performance_baseline.throughput_per_second if self.performance_baseline else recent_avg
            
            if recent_avg < baseline_throughput * 0.8:  # 20% degradation
                return True
        
        return False
    
    async def _perform_optimization(self, snapshot: PerformanceSnapshot) -> None:
        """Perform optimization based on current performance snapshot."""
        with self._optimization_lock:
            self.logger.info("Performing performance optimization")
            
            optimizations_applied = []
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                memory_optimizations = await self._optimize_memory(snapshot)
                optimizations_applied.extend(memory_optimizations)
            
            # CPU optimization
            if self.config.enable_cpu_optimization:
                cpu_optimizations = await self._optimize_cpu(snapshot)
                optimizations_applied.extend(cpu_optimizations)
            
            # Batch processing optimization
            if self.batch_processor:
                batch_optimizations = await self._optimize_batch_processing(snapshot)
                optimizations_applied.extend(batch_optimizations)
            
            # Caching optimization
            if self.config.enable_smart_caching:
                cache_optimizations = await self._optimize_caching(snapshot)
                optimizations_applied.extend(cache_optimizations)
            
            # Record optimizations
            for opt in optimizations_applied:
                self.optimization_actions.append(opt)
            
            self.last_optimization_time = datetime.utcnow()
            
            if optimizations_applied:
                self.logger.info(f"Applied {len(optimizations_applied)} optimizations")
            else:
                self.logger.debug("No optimizations needed")
    
    async def _optimize_memory(self, snapshot: PerformanceSnapshot) -> List[OptimizationAction]:
        """Optimize memory usage."""
        optimizations = []
        
        memory_usage_percent = snapshot.memory_usage_mb / 1024 / self.system_resources["memory_total_gb"] * 100
        
        # Trigger garbage collection if memory usage is high
        if memory_usage_percent > self.config.max_memory_usage_percent:
            gc.collect()
            
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type="memory_gc",
                description="Triggered garbage collection due to high memory usage",
                parameters={"memory_usage_percent": memory_usage_percent},
                expected_impact="Reduce memory usage by 5-15%",
                success=True
            )
            optimizations.append(action)
        
        # Clear cache if memory is constrained
        if memory_usage_percent > 85 and len(self.cache) > 0:
            cache_size_before = len(self.cache)
            self.cache.clear()
            
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type="memory_cache_clear",
                description="Cleared cache to free memory",
                parameters={"cache_entries_cleared": cache_size_before},
                expected_impact="Free cache memory",
                success=True
            )
            optimizations.append(action)
        
        # Reduce batch size if memory usage is very high
        if (memory_usage_percent > 90 and 
            self.batch_processor and 
            hasattr(self.batch_processor, 'current_batch_size') and
            self.batch_processor.current_batch_size > 10):
            
            old_size = self.batch_processor.current_batch_size
            new_size = max(10, old_size // 2)
            self.batch_processor.current_batch_size = new_size
            
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type="memory_batch_reduction",
                description="Reduced batch size to conserve memory",
                parameters={"old_batch_size": old_size, "new_batch_size": new_size},
                expected_impact="Reduce memory usage per batch",
                success=True
            )
            optimizations.append(action)
        
        return optimizations
    
    async def _optimize_cpu(self, snapshot: PerformanceSnapshot) -> List[OptimizationAction]:
        """Optimize CPU utilization."""
        optimizations = []
        
        # Adjust worker count based on CPU usage
        if self.config.enable_adaptive_concurrency and self.batch_processor:
            target_cpu = min(self.config.max_cpu_usage_percent, 80.0)
            
            if snapshot.cpu_usage_percent < target_cpu * 0.6 and self.current_workers < self.system_resources["cpu_count"] * 2:
                # CPU is underutilized, increase workers
                new_workers = min(self.current_workers + 1, self.system_resources["cpu_count"] * 2)
                
                action = OptimizationAction(
                    timestamp=datetime.utcnow(),
                    action_type="cpu_increase_workers",
                    description="Increased worker count to utilize available CPU",
                    parameters={"old_workers": self.current_workers, "new_workers": new_workers},
                    expected_impact="Increase throughput",
                    success=True
                )
                optimizations.append(action)
                self.current_workers = new_workers
                
            elif snapshot.cpu_usage_percent > target_cpu and self.current_workers > 1:
                # CPU is overutilized, reduce workers
                new_workers = max(1, self.current_workers - 1)
                
                action = OptimizationAction(
                    timestamp=datetime.utcnow(),
                    action_type="cpu_decrease_workers",
                    description="Decreased worker count to reduce CPU pressure",
                    parameters={"old_workers": self.current_workers, "new_workers": new_workers},
                    expected_impact="Reduce CPU usage",
                    success=True
                )
                optimizations.append(action)
                self.current_workers = new_workers
        
        return optimizations
    
    async def _optimize_batch_processing(self, snapshot: PerformanceSnapshot) -> List[OptimizationAction]:
        """Optimize batch processing parameters."""
        optimizations = []
        
        if not self.batch_processor:
            return optimizations
        
        batch_metrics = self.batch_processor.get_metrics()
        
        # Optimize batch size based on efficiency
        if snapshot.batch_efficiency < 0.5 and self.current_batch_size > 10:
            # Low efficiency, reduce batch size
            old_size = self.current_batch_size
            new_size = max(10, old_size - 10)
            self.current_batch_size = new_size
            
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type="batch_reduce_size",
                description="Reduced batch size to improve efficiency",
                parameters={"old_size": old_size, "new_size": new_size},
                expected_impact="Improve batch processing efficiency",
                success=True
            )
            optimizations.append(action)
            
        elif (snapshot.batch_efficiency > 0.8 and 
              snapshot.queue_size > self.current_batch_size * 2 and
              self.current_batch_size < 500):
            # High efficiency and queue backlog, increase batch size
            old_size = self.current_batch_size
            new_size = min(500, old_size + 20)
            self.current_batch_size = new_size
            
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type="batch_increase_size",
                description="Increased batch size to process queue faster",
                parameters={"old_size": old_size, "new_size": new_size},
                expected_impact="Increase throughput for queued items",
                success=True
            )
            optimizations.append(action)
        
        return optimizations
    
    async def _optimize_caching(self, snapshot: PerformanceSnapshot) -> List[OptimizationAction]:
        """Optimize caching behavior."""
        optimizations = []
        
        if not self.config.enable_smart_caching:
            return optimizations
        
        # Calculate cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        # If hit rate is low and memory allows, don't clear cache aggressively
        memory_usage_percent = snapshot.memory_usage_mb / 1024 / self.system_resources["memory_total_gb"] * 100
        
        if hit_rate < 0.3 and memory_usage_percent < 70 and len(self.cache) > 100:
            # Low hit rate might indicate cache thrashing, reduce cache size
            items_to_remove = len(self.cache) // 4
            cache_items = list(self.cache.keys())
            for key in cache_items[:items_to_remove]:
                del self.cache[key]
            
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type="cache_optimize_size",
                description="Reduced cache size due to low hit rate",
                parameters={"items_removed": items_to_remove, "hit_rate": hit_rate},
                expected_impact="Improve cache efficiency",
                success=True
            )
            optimizations.append(action)
        
        return optimizations
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def add_to_cache(self, key: str, value: Any) -> None:
        """Add item to cache with size management."""
        if not self.config.enable_smart_caching:
            return
        
        # Simple size management (could be more sophisticated)
        if len(self.cache) >= self.max_cache_size // 1000:  # Rough estimate
            # Remove oldest items (simple FIFO)
            items_to_remove = len(self.cache) // 10
            cache_keys = list(self.cache.keys())
            for key_to_remove in cache_keys[:items_to_remove]:
                del self.cache[key_to_remove]
        
        self.cache[key] = value
    
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if not self.config.enable_smart_caching:
            self.cache_misses += 1
            return None
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_snapshots = list(self.performance_history)[-10:]  # Last 10 snapshots
        
        # Calculate averages
        avg_throughput = sum(s.throughput_per_second for s in recent_snapshots) / len(recent_snapshots)
        avg_latency = sum(s.average_latency for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_usage_mb for s in recent_snapshots) / len(recent_snapshots)
        avg_cpu = sum(s.cpu_usage_percent for s in recent_snapshots) / len(recent_snapshots)
        
        # Calculate trends
        if len(recent_snapshots) >= 2:
            throughput_trend = recent_snapshots[-1].throughput_per_second - recent_snapshots[0].throughput_per_second
            latency_trend = recent_snapshots[-1].average_latency - recent_snapshots[0].average_latency
        else:
            throughput_trend = latency_trend = 0.0
        
        return {
            "optimization_status": "active" if self._is_optimizing else "inactive",
            "current_performance": recent_snapshots[-1].to_dict() if recent_snapshots else {},
            "averages": {
                "throughput_per_second": avg_throughput,
                "average_latency": avg_latency,
                "memory_usage_mb": avg_memory,
                "cpu_usage_percent": avg_cpu
            },
            "trends": {
                "throughput_trend": throughput_trend,
                "latency_trend": latency_trend
            },
            "optimization_actions": [
                {
                    "timestamp": action.timestamp.isoformat(),
                    "type": action.action_type,
                    "description": action.description,
                    "success": action.success
                }
                for action in self.optimization_actions[-10:]  # Last 10 actions
            ],
            "cache_stats": self.get_cache_stats(),
            "system_resources": {
                "cpu_count": self.system_resources["cpu_count"],
                "memory_total_gb": self.system_resources["memory_total_gb"],
                "current_workers": self.current_workers
            },
            "configuration": {
                "level": self.config.level.value,
                "target": self.config.target.value,
                "max_memory_percent": self.config.max_memory_usage_percent,
                "max_cpu_percent": self.config.max_cpu_usage_percent
            }
        }
    
    async def manual_optimization(self, optimization_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Manually trigger specific optimization."""
        parameters = parameters or {}
        
        try:
            if optimization_type == "gc":
                gc.collect()
                result = {"status": "success", "message": "Garbage collection triggered"}
                
            elif optimization_type == "clear_cache":
                cache_size = len(self.cache)
                self.cache.clear()
                result = {"status": "success", "message": f"Cleared {cache_size} cache entries"}
                
            elif optimization_type == "adjust_workers":
                new_count = parameters.get("worker_count", self.current_workers)
                self.current_workers = max(1, min(new_count, self.system_resources["cpu_count"] * 2))
                result = {"status": "success", "message": f"Adjusted worker count to {self.current_workers}"}
                
            elif optimization_type == "adjust_batch_size":
                if self.batch_processor:
                    new_size = parameters.get("batch_size", self.current_batch_size)
                    self.current_batch_size = max(1, min(new_size, 1000))
                    result = {"status": "success", "message": f"Adjusted batch size to {self.current_batch_size}"}
                else:
                    result = {"status": "error", "message": "No batch processor available"}
                    
            else:
                result = {"status": "error", "message": f"Unknown optimization type: {optimization_type}"}
            
            # Record manual optimization
            action = OptimizationAction(
                timestamp=datetime.utcnow(),
                action_type=f"manual_{optimization_type}",
                description=f"Manual optimization: {optimization_type}",
                parameters=parameters,
                expected_impact="User-triggered optimization",
                success=result["status"] == "success"
            )
            self.optimization_actions.append(action)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Manual optimization failed: {e}")
            return {"status": "error", "message": str(e)}