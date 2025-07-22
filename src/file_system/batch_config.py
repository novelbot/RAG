"""
Batch Processing Configuration.

This module defines configuration classes for the batch processing system,
including worker settings, memory limits, and retry policies.
"""

import os
import psutil
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ProcessingStrategy(Enum):
    """Strategy for processing files in batches."""
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    SIZE_ASC = "size_asc"  # Smallest files first
    SIZE_DESC = "size_desc"  # Largest files first
    PRIORITY = "priority"  # Priority-based


class RetryStrategy(Enum):
    """Strategy for retrying failed operations."""
    NONE = "none"  # No retries
    FIXED = "fixed"  # Fixed interval
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    
    # Worker Configuration
    max_workers: int = 4
    batch_size: int = 10
    worker_timeout: float = 300.0  # 5 minutes per worker
    
    # Memory Management
    memory_limit_mb: Optional[int] = None  # Max memory usage in MB
    memory_threshold: float = 0.8  # Memory usage threshold (80%)
    enable_gc: bool = True  # Enable garbage collection
    gc_interval: int = 100  # GC every N processed items
    
    # Processing Strategy
    processing_strategy: ProcessingStrategy = ProcessingStrategy.SIZE_ASC
    enable_priority_queue: bool = False
    
    # Retry Configuration
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_retry_delay: float = 1.0  # Base delay in seconds
    max_retry_delay: float = 60.0  # Maximum delay in seconds
    
    # Progress Tracking
    enable_progress_tracking: bool = True
    progress_update_interval: int = 1  # Update every N processed items
    enable_eta_calculation: bool = True
    
    # Queue Management
    max_queue_size: Optional[int] = None  # Maximum items in queue
    queue_timeout: float = 30.0  # Queue operation timeout
    
    # Checkpoint and Recovery
    enable_checkpoints: bool = True
    checkpoint_interval: int = 100  # Save checkpoint every N items
    checkpoint_file: Optional[str] = None
    
    # Performance Tuning
    enable_adaptive_batching: bool = True  # Adjust batch size based on performance
    min_batch_size: int = 1
    max_batch_size: int = 100
    performance_window: int = 50  # Window for performance calculations
    
    def __post_init__(self):
        """Validate and normalize configuration values."""
        # Auto-detect optimal worker count if not specified
        if self.max_workers <= 0:
            self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        # Auto-detect memory limit if not specified
        if self.memory_limit_mb is None:
            available_memory = psutil.virtual_memory().available
            # Use 50% of available memory as default limit
            self.memory_limit_mb = int(available_memory * 0.5 / (1024 * 1024))
        
        # Validate batch size constraints
        if self.batch_size < self.min_batch_size:
            self.batch_size = self.min_batch_size
        if self.batch_size > self.max_batch_size:
            self.batch_size = self.max_batch_size
        
        # Validate memory threshold
        if not 0.1 <= self.memory_threshold <= 1.0:
            self.memory_threshold = 0.8
        
        # Validate retry configuration
        if self.max_retries < 0:
            self.max_retries = 0
        if self.base_retry_delay <= 0:
            self.base_retry_delay = 1.0
        if self.max_retry_delay < self.base_retry_delay:
            self.max_retry_delay = self.base_retry_delay * 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "worker_timeout": self.worker_timeout,
            "memory_limit_mb": self.memory_limit_mb,
            "memory_threshold": self.memory_threshold,
            "enable_gc": self.enable_gc,
            "gc_interval": self.gc_interval,
            "processing_strategy": self.processing_strategy.value,
            "enable_priority_queue": self.enable_priority_queue,
            "retry_strategy": self.retry_strategy.value,
            "max_retries": self.max_retries,
            "base_retry_delay": self.base_retry_delay,
            "max_retry_delay": self.max_retry_delay,
            "enable_progress_tracking": self.enable_progress_tracking,
            "progress_update_interval": self.progress_update_interval,
            "enable_eta_calculation": self.enable_eta_calculation,
            "max_queue_size": self.max_queue_size,
            "queue_timeout": self.queue_timeout,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_file": self.checkpoint_file,
            "enable_adaptive_batching": self.enable_adaptive_batching,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "performance_window": self.performance_window
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchProcessingConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "processing_strategy" in data:
            data["processing_strategy"] = ProcessingStrategy(data["processing_strategy"])
        if "retry_strategy" in data:
            data["retry_strategy"] = RetryStrategy(data["retry_strategy"])
        
        return cls(**data)
    
    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes."""
        return self.memory_limit_mb * 1024 * 1024
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy and attempt number."""
        if self.retry_strategy == RetryStrategy.NONE:
            return 0.0
        elif self.retry_strategy == RetryStrategy.FIXED:
            return min(self.base_retry_delay, self.max_retry_delay)
        elif self.retry_strategy == RetryStrategy.LINEAR:
            delay = self.base_retry_delay * attempt
            return min(delay, self.max_retry_delay)
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_retry_delay * (2 ** (attempt - 1))
            return min(delay, self.max_retry_delay)
        else:
            return self.base_retry_delay
    
    def should_retry(self, attempt: int) -> bool:
        """Check if operation should be retried."""
        return (self.retry_strategy != RetryStrategy.NONE and 
                attempt <= self.max_retries)
    
    def get_optimal_batch_size(self, current_performance: float) -> int:
        """Calculate optimal batch size based on current performance."""
        if not self.enable_adaptive_batching:
            return self.batch_size
        
        # Simple adaptive batching: increase batch size if performance is good
        if current_performance > 0.8:  # High performance
            new_size = min(self.batch_size + 1, self.max_batch_size)
        elif current_performance < 0.4:  # Low performance
            new_size = max(self.batch_size - 1, self.min_batch_size)
        else:
            new_size = self.batch_size
        
        return new_size