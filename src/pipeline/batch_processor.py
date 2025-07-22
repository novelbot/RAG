"""
Batch processing system for efficient data handling.
"""

import asyncio
import time
import psutil
from typing import Any, Dict, List, Optional, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from collections import deque

from src.core.logging import LoggerMixin
from src.core.exceptions import PipelineError
from .stages import ProcessingData, ProcessingContext


class BatchProcessingStrategy(Enum):
    """Strategies for batch processing."""
    FIXED_SIZE = "fixed_size"  # Fixed batch size
    MEMORY_BASED = "memory_based"  # Based on memory usage
    TIME_BASED = "time_based"  # Based on time windows
    ADAPTIVE = "adaptive"  # Adaptive based on performance
    TOKEN_BASED = "token_based"  # Based on token count for embeddings


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    strategy: BatchProcessingStrategy = BatchProcessingStrategy.ADAPTIVE
    base_batch_size: int = 100
    min_batch_size: int = 10
    max_batch_size: int = 1000
    memory_limit_mb: float = 1024.0  # 1GB default
    time_window_seconds: float = 30.0
    max_tokens_per_batch: int = 8192  # For embedding APIs
    parallel_batches: int = 3
    queue_size_limit: int = 10000
    auto_scaling: bool = True
    performance_window: int = 10  # Number of batches to consider for adaptive sizing


@dataclass 
class BatchMetrics:
    """Metrics for batch processing."""
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_items_processed: int = 0
    total_processing_time: float = 0.0
    average_batch_size: float = 0.0
    average_processing_time: float = 0.0
    throughput_items_per_second: float = 0.0
    memory_usage_peak_mb: float = 0.0
    queue_wait_time: float = 0.0
    last_batch_time: Optional[datetime] = None
    
    def update_batch_completed(self, batch_size: int, processing_time: float, memory_usage: float):
        """Update metrics when a batch completes successfully."""
        self.total_batches += 1
        self.successful_batches += 1
        self.total_items_processed += batch_size
        self.total_processing_time += processing_time
        self.memory_usage_peak_mb = max(self.memory_usage_peak_mb, memory_usage)
        self.last_batch_time = datetime.utcnow()
        
        # Calculate averages
        self.average_batch_size = self.total_items_processed / max(self.total_batches, 1)
        self.average_processing_time = self.total_processing_time / max(self.successful_batches, 1)
        
        # Calculate throughput
        if self.total_processing_time > 0:
            self.throughput_items_per_second = self.total_items_processed / self.total_processing_time
    
    def update_batch_failed(self):
        """Update metrics when a batch fails."""
        self.total_batches += 1
        self.failed_batches += 1
        self.last_batch_time = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Get batch success rate as percentage."""
        if self.total_batches == 0:
            return 100.0
        return (self.successful_batches / self.total_batches) * 100.0


@dataclass
class BatchJob:
    """A batch processing job."""
    id: str
    items: List[ProcessingData]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    retry_count: int = 0
    priority: int = 0  # Higher priority jobs are processed first
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.items)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class BatchProcessor(LoggerMixin):
    """
    Batch processor for efficient handling of large datasets.
    
    Features:
    - Multiple batching strategies
    - Memory-aware processing
    - Adaptive batch sizing
    - Parallel batch processing
    - Progress tracking and monitoring
    - Queue management with backpressure
    """
    
    def __init__(self, config: BatchConfig):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.metrics = BatchMetrics()
        self.current_batch_size = config.base_batch_size
        self.processing_queue: deque = deque()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.performance_history: deque = deque(maxlen=config.performance_window)
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        # Semaphores for controlling parallelism
        self.batch_semaphore = asyncio.Semaphore(config.parallel_batches)
        self.queue_lock = asyncio.Lock()
        
    async def start(self) -> None:
        """Start the batch processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        self.logger.info("Batch processor started")
        
        # Start background tasks
        asyncio.create_task(self._batch_creator_task())
        asyncio.create_task(self._performance_monitor_task())
    
    async def stop(self) -> None:
        """Stop the batch processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for all active jobs to complete
        while self.active_jobs:
            await asyncio.sleep(0.1)
        
        self.logger.info("Batch processor stopped")
    
    async def submit_item(self, item: ProcessingData, priority: int = 0) -> str:
        """
        Submit an item for batch processing.
        
        Args:
            item: Data item to process
            priority: Priority level (higher = more urgent)
            
        Returns:
            str: Batch job ID when item is batched
            
        Raises:
            PipelineError: If queue is full
        """
        async with self.queue_lock:
            if len(self.processing_queue) >= self.config.queue_size_limit:
                raise PipelineError("Processing queue is full")
            
            self.processing_queue.append((item, priority, datetime.utcnow()))
            self.logger.debug(f"Added item to processing queue, queue size: {len(self.processing_queue)}")
    
    async def submit_batch(self, items: List[ProcessingData], priority: int = 0) -> str:
        """
        Submit a pre-formed batch for processing.
        
        Args:
            items: List of data items
            priority: Priority level
            
        Returns:
            str: Batch job ID
        """
        job_id = str(uuid.uuid4())
        job = BatchJob(
            id=job_id,
            items=items,
            created_at=datetime.utcnow(),
            priority=priority
        )
        
        self.active_jobs[job_id] = job
        asyncio.create_task(self._process_batch_job(job))
        
        return job_id
    
    async def process_batch(self, batch: List[ProcessingData], processor_func: Callable) -> List[Any]:
        """
        Process a batch of items using the provided processor function.
        
        Args:
            batch: List of items to process
            processor_func: Function to process the batch
            
        Returns:
            List of processed results
        """
        async with self.batch_semaphore:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                self.logger.debug(f"Processing batch of {len(batch)} items")
                
                # Process the batch
                results = await processor_func(batch)
                
                # Update metrics
                processing_time = time.time() - start_time
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = current_memory - start_memory
                
                self.metrics.update_batch_completed(len(batch), processing_time, current_memory)
                
                # Update performance history for adaptive sizing
                if self.config.auto_scaling:
                    self._update_performance_history(len(batch), processing_time, memory_used)
                
                self.logger.debug(f"Batch processed successfully in {processing_time:.2f}s")
                return results
                
            except Exception as e:
                self.metrics.update_batch_failed()
                self.logger.error(f"Batch processing failed: {e}")
                raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
        else:
            return None
        
        return {
            "id": job.id,
            "status": job.status,
            "size": job.size,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time": job.processing_time,
            "error": job.error,
            "retry_count": job.retry_count
        }
    
    def get_batch_size(self, items: List[ProcessingData]) -> int:
        """
        Calculate optimal batch size based on strategy.
        
        Args:
            items: Items to be batched
            
        Returns:
            int: Optimal batch size
        """
        if self.config.strategy == BatchProcessingStrategy.FIXED_SIZE:
            return min(self.config.base_batch_size, len(items))
        
        elif self.config.strategy == BatchProcessingStrategy.MEMORY_BASED:
            return self._calculate_memory_based_batch_size(items)
        
        elif self.config.strategy == BatchProcessingStrategy.TOKEN_BASED:
            return self._calculate_token_based_batch_size(items)
        
        elif self.config.strategy == BatchProcessingStrategy.ADAPTIVE:
            return self._calculate_adaptive_batch_size(items)
        
        else:  # TIME_BASED or fallback
            return min(self.current_batch_size, len(items))
    
    def _calculate_memory_based_batch_size(self, items: List[ProcessingData]) -> int:
        """Calculate batch size based on memory constraints."""
        if not items:
            return 0
        
        # Estimate memory per item (rough calculation)
        sample_item = items[0]
        estimated_size_mb = len(str(sample_item.content)) / 1024 / 1024  # Very rough estimate
        
        max_items = int(self.config.memory_limit_mb / max(estimated_size_mb, 0.1))
        return max(self.config.min_batch_size, 
                  min(max_items, self.config.max_batch_size, len(items)))
    
    def _calculate_token_based_batch_size(self, items: List[ProcessingData]) -> int:
        """Calculate batch size based on token limits."""
        if not items:
            return 0
        
        batch_size = 0
        token_count = 0
        
        for item in items:
            # Rough token estimation (1 token ≈ 4 characters)
            item_tokens = len(str(item.content)) // 4
            
            if token_count + item_tokens > self.config.max_tokens_per_batch and batch_size > 0:
                break
            
            token_count += item_tokens
            batch_size += 1
            
            if batch_size >= self.config.max_batch_size:
                break
        
        return max(batch_size, self.config.min_batch_size) if batch_size > 0 else self.config.min_batch_size
    
    def _calculate_adaptive_batch_size(self, items: List[ProcessingData]) -> int:
        """Calculate adaptive batch size based on performance history."""
        if not self.performance_history:
            return min(self.config.base_batch_size, len(items))
        
        # Find optimal batch size based on throughput
        best_throughput = 0
        best_size = self.config.base_batch_size
        
        for batch_size, processing_time, memory_used in self.performance_history:
            if memory_used <= self.config.memory_limit_mb:
                throughput = batch_size / processing_time if processing_time > 0 else 0
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_size = batch_size
        
        # Adjust size slightly for exploration
        if self.config.auto_scaling and len(self.performance_history) >= 5:
            # Small random adjustment for exploration
            import random
            adjustment = random.randint(-10, 10)
            best_size = max(self.config.min_batch_size, 
                           min(self.config.max_batch_size, best_size + adjustment))
        
        self.current_batch_size = best_size
        return min(best_size, len(items))
    
    def _update_performance_history(self, batch_size: int, processing_time: float, memory_used: float):
        """Update performance history for adaptive sizing."""
        self.performance_history.append((batch_size, processing_time, memory_used))
    
    async def _batch_creator_task(self):
        """Background task to create batches from the queue."""
        while self.is_running:
            try:
                await self._create_batches_from_queue()
                await asyncio.sleep(1.0)  # Check queue every second
            except Exception as e:
                self.logger.error(f"Error in batch creator task: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _create_batches_from_queue(self):
        """Create batches from queued items."""
        async with self.queue_lock:
            if not self.processing_queue:
                return
            
            # Sort by priority and age
            queue_items = list(self.processing_queue)
            queue_items.sort(key=lambda x: (-x[1], x[2]))  # Sort by priority desc, then by time
            
            # Create batches
            while queue_items:
                batch_items = []
                items_for_sizing = [item[0] for item in queue_items[:1000]]  # Sample for sizing
                optimal_size = self.get_batch_size(items_for_sizing)
                
                # Take items for batch
                for _ in range(min(optimal_size, len(queue_items))):
                    item_data, priority, timestamp = queue_items.pop(0)
                    batch_items.append(item_data)
                
                if batch_items:
                    # Create and submit batch job
                    job_id = str(uuid.uuid4())
                    job = BatchJob(
                        id=job_id,
                        items=batch_items,
                        created_at=datetime.utcnow(),
                        priority=priority
                    )
                    
                    self.active_jobs[job_id] = job
                    asyncio.create_task(self._process_batch_job(job))
                    
                    # Update queue
                    for item_tuple in queue_items:
                        self.processing_queue.append(item_tuple)
                    self.processing_queue.clear()
                    for item_tuple in queue_items:
                        self.processing_queue.append(item_tuple)
                    break
    
    async def _process_batch_job(self, job: BatchJob):
        """Process a batch job."""
        job.status = "processing"
        job.started_at = datetime.utcnow()
        
        try:
            async with self.batch_semaphore:
                self.logger.info(f"Processing batch job {job.id} with {job.size} items")
                
                # This is where the actual processing would happen
                # The processor function should be provided by the caller
                # For now, we just simulate processing
                await asyncio.sleep(0.1 * job.size)  # Simulate processing time
                
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                
                # Move to completed jobs
                self.completed_jobs[job.id] = job
                del self.active_jobs[job.id]
                
                self.logger.info(f"Batch job {job.id} completed successfully")
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            
            # Move to completed jobs (even if failed)
            self.completed_jobs[job.id] = job
            del self.active_jobs[job.id]
            
            self.logger.error(f"Batch job {job.id} failed: {e}")
    
    async def _performance_monitor_task(self):
        """Background task to monitor performance and adjust settings."""
        while self.is_running:
            try:
                await self._monitor_performance()
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in performance monitor task: {e}")
                await asyncio.sleep(60.0)  # Wait longer on error
    
    async def _monitor_performance(self):
        """Monitor performance and adjust batch sizing if needed."""
        if not self.config.auto_scaling:
            return
        
        # Check memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if current_memory > self.config.memory_limit_mb * 0.9:  # 90% of limit
            # Reduce batch size if memory is high
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            self.logger.warning(f"High memory usage, reducing batch size to {self.current_batch_size}")
        
        # Log performance metrics
        self.logger.debug(f"Performance metrics: {self.get_metrics()}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch processor metrics."""
        return {
            "total_batches": self.metrics.total_batches,
            "successful_batches": self.metrics.successful_batches,
            "failed_batches": self.metrics.failed_batches,
            "success_rate": self.metrics.get_success_rate(),
            "total_items_processed": self.metrics.total_items_processed,
            "average_batch_size": self.metrics.average_batch_size,
            "average_processing_time": self.metrics.average_processing_time,
            "throughput_items_per_second": self.metrics.throughput_items_per_second,
            "memory_usage_peak_mb": self.metrics.memory_usage_peak_mb,
            "current_batch_size": self.current_batch_size,
            "queue_size": len(self.processing_queue),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "is_running": self.is_running,
            "configuration": {
                "strategy": self.config.strategy.value,
                "base_batch_size": self.config.base_batch_size,
                "min_batch_size": self.config.min_batch_size,
                "max_batch_size": self.config.max_batch_size,
                "memory_limit_mb": self.config.memory_limit_mb,
                "parallel_batches": self.config.parallel_batches,
                "auto_scaling": self.config.auto_scaling
            }
        }