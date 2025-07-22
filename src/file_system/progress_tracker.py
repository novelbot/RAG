"""
Progress Tracking and Statistics for Batch Processing.

This module provides functionality to track progress, calculate ETA,
and monitor performance during batch processing operations.
"""

import gc
import time
import psutil
import threading
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Deque

from src.core.logging import LoggerMixin


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    retried_items: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    
    total_bytes_processed: int = 0
    average_item_size: float = 0.0
    
    # Performance metrics
    items_per_second: float = 0.0
    bytes_per_second: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    
    # Error tracking
    error_count_by_type: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "retried_items": self.retried_items,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "total_bytes_processed": self.total_bytes_processed,
            "average_item_size": self.average_item_size,
            "items_per_second": self.items_per_second,
            "bytes_per_second": self.bytes_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "current_memory_mb": self.current_memory_mb,
            "error_count_by_type": self.error_count_by_type
        }


class ProgressTracker(LoggerMixin):
    """
    Progress tracking and performance monitoring for batch processing.
    
    Provides real-time progress updates, ETA calculations, performance metrics,
    and memory monitoring for long-running batch operations.
    """
    
    def __init__(self, total_items: int, enable_eta: bool = True, 
                 performance_window: int = 50):
        """
        Initialize the progress tracker.
        
        Args:
            total_items: Total number of items to process
            enable_eta: Whether to calculate ETA
            performance_window: Window size for performance calculations
        """
        self.stats = ProcessingStats(total_items=total_items)
        self.enable_eta = enable_eta
        self.performance_window = performance_window
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._performance_samples: Deque[float] = deque(maxlen=performance_window)
        self._throughput_samples: Deque[float] = deque(maxlen=performance_window)
        self._memory_samples: Deque[float] = deque(maxlen=performance_window)
        
        # ETA calculation
        self._eta_samples: Deque[float] = deque(maxlen=10)
        
        # Callbacks
        self._progress_callbacks: List[Callable[[ProcessingStats], None]] = []
        
        self.logger.info(f"Progress tracker initialized for {total_items} items")
    
    def start(self) -> None:
        """Start tracking progress."""
        with self._lock:
            self.stats.start_time = datetime.now()
            self.stats.last_update_time = self.stats.start_time
            self._update_memory_usage()
            self.logger.info("Progress tracking started")
    
    def finish(self) -> None:
        """Finish tracking progress."""
        with self._lock:
            self.stats.end_time = datetime.now()
            self._update_performance_metrics()
            self._update_memory_usage()
            
            duration = self.get_elapsed_time()
            self.logger.info(
                f"Progress tracking finished: {self.stats.processed_items}/{self.stats.total_items} "
                f"items processed in {duration:.2f}s "
                f"({self.stats.items_per_second:.2f} items/s)"
            )
    
    def update(self, processed_count: int = 1, bytes_processed: int = 0,
               error_type: Optional[str] = None, is_retry: bool = False) -> None:
        """
        Update progress with processed items.
        
        Args:
            processed_count: Number of items processed
            bytes_processed: Number of bytes processed
            error_type: Type of error if processing failed
            is_retry: Whether this is a retry attempt
        """
        with self._lock:
            self.stats.processed_items += processed_count
            self.stats.total_bytes_processed += bytes_processed
            
            if error_type:
                self.stats.failed_items += processed_count
                self.stats.error_count_by_type[error_type] = \
                    self.stats.error_count_by_type.get(error_type, 0) + processed_count
            else:
                self.stats.successful_items += processed_count
            
            if is_retry:
                self.stats.retried_items += processed_count
            
            self.stats.last_update_time = datetime.now()
            
            # Update performance metrics
            self._update_performance_metrics()
            self._update_memory_usage()
            
            # Update average item size
            if self.stats.processed_items > 0:
                self.stats.average_item_size = \
                    self.stats.total_bytes_processed / self.stats.processed_items
            
            # Trigger progress callbacks
            self._trigger_progress_callbacks()
    
    def skip(self, skip_count: int = 1) -> None:
        """Mark items as skipped."""
        with self._lock:
            self.stats.skipped_items += skip_count
            self.stats.processed_items += skip_count
            self.stats.last_update_time = datetime.now()
            
            self._update_performance_metrics()
            self._trigger_progress_callbacks()
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        with self._lock:
            if self.stats.total_items == 0:
                return 100.0
            return (self.stats.processed_items / self.stats.total_items) * 100.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        with self._lock:
            if not self.stats.start_time:
                return 0.0
            
            end_time = self.stats.end_time or datetime.now()
            return (end_time - self.stats.start_time).total_seconds()
    
    def get_eta_seconds(self) -> Optional[float]:
        """Get estimated time to completion in seconds."""
        if not self.enable_eta:
            return None
        
        with self._lock:
            if (self.stats.processed_items == 0 or 
                not self.stats.start_time or
                self.stats.items_per_second == 0):
                return None
            
            remaining_items = self.stats.total_items - self.stats.processed_items
            eta_seconds = remaining_items / self.stats.items_per_second
            
            # Smooth ETA with recent samples
            self._eta_samples.append(eta_seconds)
            if len(self._eta_samples) >= 3:
                # Use median of recent samples to reduce fluctuation
                sorted_samples = sorted(self._eta_samples)
                eta_seconds = sorted_samples[len(sorted_samples) // 2]
            
            return eta_seconds
    
    def get_eta_string(self) -> str:
        """Get ETA as formatted string."""
        eta_seconds = self.get_eta_seconds()
        if eta_seconds is None:
            return "Unknown"
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"
    
    def get_throughput(self) -> Dict[str, float]:
        """Get current throughput metrics."""
        with self._lock:
            return {
                "items_per_second": self.stats.items_per_second,
                "bytes_per_second": self.stats.bytes_per_second,
                "mb_per_second": self.stats.bytes_per_second / (1024 * 1024)
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage metrics."""
        with self._lock:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "current_mb": memory_info.rss / (1024 * 1024),
                "peak_mb": self.stats.peak_memory_mb,
                "virtual_mb": memory_info.vms / (1024 * 1024),
                "system_available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "system_used_percent": psutil.virtual_memory().percent
            }
    
    def get_performance_score(self) -> float:
        """Get overall performance score (0-1)."""
        with self._lock:
            if not self._performance_samples:
                return 0.0
            
            # Calculate average performance over recent samples
            avg_performance = sum(self._performance_samples) / len(self._performance_samples)
            return min(1.0, max(0.0, avg_performance))
    
    def add_progress_callback(self, callback: Callable[[ProcessingStats], None]) -> None:
        """Add a progress callback function."""
        with self._lock:
            self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProcessingStats], None]) -> None:
        """Remove a progress callback function."""
        with self._lock:
            if callback in self._progress_callbacks:
                self._progress_callbacks.remove(callback)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        with self._lock:
            progress_pct = self.get_progress_percentage()
            eta = self.get_eta_string()
            throughput = self.get_throughput()
            memory = self.get_memory_usage()
            
            return {
                "progress": {
                    "processed": self.stats.processed_items,
                    "total": self.stats.total_items,
                    "percentage": progress_pct,
                    "successful": self.stats.successful_items,
                    "failed": self.stats.failed_items,
                    "skipped": self.stats.skipped_items,
                    "retried": self.stats.retried_items
                },
                "timing": {
                    "elapsed_seconds": self.get_elapsed_time(),
                    "eta": eta,
                    "eta_seconds": self.get_eta_seconds()
                },
                "performance": throughput,
                "memory": memory,
                "errors": dict(self.stats.error_count_by_type)
            }
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on recent activity."""
        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            return
        
        # Calculate current throughput
        self.stats.items_per_second = self.stats.processed_items / elapsed
        self.stats.bytes_per_second = self.stats.total_bytes_processed / elapsed
        
        # Add to performance samples
        performance_score = min(1.0, self.stats.items_per_second / 10.0)  # Normalize to 0-1
        self._performance_samples.append(performance_score)
        self._throughput_samples.append(self.stats.items_per_second)
    
    def _update_memory_usage(self) -> None:
        """Update memory usage tracking."""
        try:
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            self.stats.current_memory_mb = current_memory_mb
            if current_memory_mb > self.stats.peak_memory_mb:
                self.stats.peak_memory_mb = current_memory_mb
            
            self._memory_samples.append(current_memory_mb)
            
        except Exception as e:
            self.logger.warning(f"Failed to update memory usage: {e}")
    
    def _trigger_progress_callbacks(self) -> None:
        """Trigger all registered progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self.stats)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
    
    def force_gc(self) -> None:
        """Force garbage collection and update memory metrics."""
        with self._lock:
            gc.collect()
            self._update_memory_usage()
            self.logger.debug("Forced garbage collection completed")
    
    def is_memory_threshold_exceeded(self, threshold_mb: float) -> bool:
        """Check if memory usage exceeds threshold."""
        with self._lock:
            return self.stats.current_memory_mb > threshold_mb
    
    def get_statistics(self) -> ProcessingStats:
        """Get a copy of current statistics."""
        with self._lock:
            # Create a deep copy of stats
            stats_copy = ProcessingStats()
            stats_copy.total_items = self.stats.total_items
            stats_copy.processed_items = self.stats.processed_items
            stats_copy.successful_items = self.stats.successful_items
            stats_copy.failed_items = self.stats.failed_items
            stats_copy.skipped_items = self.stats.skipped_items
            stats_copy.retried_items = self.stats.retried_items
            stats_copy.start_time = self.stats.start_time
            stats_copy.end_time = self.stats.end_time
            stats_copy.last_update_time = self.stats.last_update_time
            stats_copy.total_bytes_processed = self.stats.total_bytes_processed
            stats_copy.average_item_size = self.stats.average_item_size
            stats_copy.items_per_second = self.stats.items_per_second
            stats_copy.bytes_per_second = self.stats.bytes_per_second
            stats_copy.peak_memory_mb = self.stats.peak_memory_mb
            stats_copy.current_memory_mb = self.stats.current_memory_mb
            stats_copy.error_count_by_type = self.stats.error_count_by_type.copy()
            
            return stats_copy