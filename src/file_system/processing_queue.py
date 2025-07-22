"""
Processing Queue Management for Batch Operations.

This module provides queue management functionality for batch processing,
including priority queues, work distribution, and memory-aware batching.
"""

import heapq
import threading
from enum import Enum
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union, Callable

from src.core.logging import LoggerMixin
from .batch_config import ProcessingStrategy
from .scanner import ScanResult


@dataclass
class QueueItem:
    """Item in the processing queue."""
    path: Path
    priority: int = 0
    size: int = 0
    metadata: Dict[str, Any] = None
    retry_count: int = 0
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        
        # Convert string path to Path object
        if isinstance(self.path, str):
            self.path = Path(self.path)
        
        # Get file size if not provided
        if self.size == 0 and self.path.exists():
            try:
                self.size = self.path.stat().st_size
            except OSError:
                self.size = 0
    
    def __lt__(self, other):
        """Compare items for priority queue ordering."""
        # Higher priority value = higher priority (processed first)
        return self.priority > other.priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "priority": self.priority,
            "size": self.size,
            "metadata": self.metadata,
            "retry_count": self.retry_count
        }


class ProcessingQueue(LoggerMixin):
    """
    Queue management for batch processing operations.
    
    Provides efficient work distribution, priority handling, and memory-aware
    batching for file processing operations.
    """
    
    def __init__(self, strategy: ProcessingStrategy = ProcessingStrategy.FIFO,
                 max_size: Optional[int] = None, enable_priority: bool = False):
        """
        Initialize the processing queue.
        
        Args:
            strategy: Strategy for ordering items in the queue
            max_size: Maximum number of items in queue (None for unlimited)
            enable_priority: Whether to enable priority-based processing
        """
        self.strategy = strategy
        self.max_size = max_size
        self.enable_priority = enable_priority
        
        # Queue data structures
        if enable_priority:
            self._priority_queue: List[QueueItem] = []
        else:
            self._queue: deque = deque()
        
        # Thread safety
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        # Statistics
        self._total_added = 0
        self._total_removed = 0
        self._total_bytes = 0
        
        self.logger.info(f"Processing queue initialized with strategy: {strategy.value}")
    
    def add_item(self, item: QueueItem, timeout: Optional[float] = None) -> bool:
        """
        Add an item to the queue.
        
        Args:
            item: Item to add to the queue
            timeout: Timeout for adding item (None for no timeout)
            
        Returns:
            True if item was added, False if timeout occurred
        """
        with self._not_full:
            # Wait for space if queue is full
            if self.max_size and self.size() >= self.max_size:
                if timeout is None:
                    self._not_full.wait()
                elif timeout > 0:
                    if not self._not_full.wait(timeout):
                        return False
                else:
                    return False
            
            # Add item to appropriate queue
            if self.enable_priority:
                heapq.heappush(self._priority_queue, item)
            else:
                if self.strategy == ProcessingStrategy.LIFO:
                    self._queue.append(item)
                else:
                    self._queue.appendleft(item)
            
            self._total_added += 1
            self._total_bytes += item.size
            
            # Notify waiting consumers
            self._not_empty.notify()
            
            self.logger.debug(f"Added item to queue: {item.path} (size: {self.size()})")
            return True
    
    def add_items(self, items: List[QueueItem]) -> int:
        """
        Add multiple items to the queue.
        
        Args:
            items: List of items to add
            
        Returns:
            Number of items successfully added
        """
        # Sort items according to strategy before adding
        sorted_items = self._sort_items(items)
        
        added_count = 0
        for item in sorted_items:
            if self.add_item(item):
                added_count += 1
            else:
                break
        
        self.logger.info(f"Added {added_count}/{len(items)} items to queue")
        return added_count
    
    def get_item(self, timeout: Optional[float] = None) -> Optional[QueueItem]:
        """
        Get an item from the queue.
        
        Args:
            timeout: Timeout for getting item (None for no timeout)
            
        Returns:
            Queue item or None if timeout occurred
        """
        with self._not_empty:
            # Wait for items if queue is empty
            while self.is_empty():
                if timeout is None:
                    self._not_empty.wait()
                elif timeout > 0:
                    if not self._not_empty.wait(timeout):
                        return None
                else:
                    return None
            
            # Get item from appropriate queue
            if self.enable_priority:
                item = heapq.heappop(self._priority_queue)
            else:
                if self.strategy == ProcessingStrategy.LIFO:
                    item = self._queue.pop()
                else:
                    item = self._queue.popleft()
            
            self._total_removed += 1
            
            # Notify waiting producers
            self._not_full.notify()
            
            self.logger.debug(f"Retrieved item from queue: {item.path}")
            return item
    
    def get_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[QueueItem]:
        """
        Get a batch of items from the queue.
        
        Args:
            batch_size: Maximum number of items in batch
            timeout: Timeout for getting items
            
        Returns:
            List of queue items (may be smaller than batch_size)
        """
        batch = []
        remaining_timeout = timeout
        
        for _ in range(batch_size):
            start_time = None
            if remaining_timeout is not None:
                start_time = threading.get_ident()  # Use as timestamp
            
            item = self.get_item(remaining_timeout)
            if item is None:
                break
            
            batch.append(item)
            
            # Update remaining timeout
            if remaining_timeout is not None and start_time is not None:
                # Simplified timeout handling
                remaining_timeout = max(0, remaining_timeout - 0.1)
                if remaining_timeout <= 0:
                    break
        
        self.logger.debug(f"Retrieved batch of {len(batch)} items")
        return batch
    
    def peek(self) -> Optional[QueueItem]:
        """
        Peek at the next item without removing it.
        
        Returns:
            Next queue item or None if queue is empty
        """
        with self._lock:
            if self.is_empty():
                return None
            
            if self.enable_priority:
                return self._priority_queue[0]
            else:
                if self.strategy == ProcessingStrategy.LIFO:
                    return self._queue[-1]
                else:
                    return self._queue[0]
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            if self.enable_priority:
                return len(self._priority_queue)
            else:
                return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        if self.max_size is None:
            return False
        return self.size() >= self.max_size
    
    def clear(self) -> int:
        """
        Clear all items from the queue.
        
        Returns:
            Number of items that were cleared
        """
        with self._lock:
            count = self.size()
            
            if self.enable_priority:
                self._priority_queue.clear()
            else:
                self._queue.clear()
            
            # Notify all waiting threads
            self._not_full.notify_all()
            
            self.logger.info(f"Cleared {count} items from queue")
            return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "current_size": self.size(),
                "max_size": self.max_size,
                "total_added": self._total_added,
                "total_removed": self._total_removed,
                "total_bytes": self._total_bytes,
                "strategy": self.strategy.value,
                "enable_priority": self.enable_priority,
                "is_empty": self.is_empty(),
                "is_full": self.is_full()
            }
    
    def reorder_by_strategy(self, new_strategy: ProcessingStrategy) -> None:
        """
        Reorder queue items according to new strategy.
        
        Args:
            new_strategy: New processing strategy to apply
        """
        with self._lock:
            if new_strategy == self.strategy:
                return
            
            # Get all current items
            items = []
            while not self.is_empty():
                if self.enable_priority:
                    items.append(heapq.heappop(self._priority_queue))
                else:
                    items.append(self._queue.popleft())
            
            # Update strategy and re-add items
            self.strategy = new_strategy
            sorted_items = self._sort_items(items)
            
            for item in sorted_items:
                if self.enable_priority:
                    heapq.heappush(self._priority_queue, item)
                else:
                    self._queue.append(item)
            
            self.logger.info(f"Reordered queue with strategy: {new_strategy.value}")
    
    def update_priority(self, path: Path, new_priority: int) -> bool:
        """
        Update priority of an item in the queue.
        
        Args:
            path: Path of item to update
            new_priority: New priority value
            
        Returns:
            True if item was found and updated
        """
        if not self.enable_priority:
            return False
        
        with self._lock:
            # Find and update item
            for item in self._priority_queue:
                if item.path == path:
                    item.priority = new_priority
                    heapq.heapify(self._priority_queue)  # Re-heapify
                    self.logger.debug(f"Updated priority for {path}: {new_priority}")
                    return True
            
            return False
    
    def get_items_by_size_range(self, min_size: int, max_size: int) -> List[QueueItem]:
        """
        Get items within a specific size range.
        
        Args:
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            
        Returns:
            List of items within size range
        """
        with self._lock:
            items = []
            queue_items = self._priority_queue if self.enable_priority else list(self._queue)
            
            for item in queue_items:
                if min_size <= item.size <= max_size:
                    items.append(item)
            
            return items
    
    def get_memory_optimized_batch(self, max_batch_memory: int, max_items: int) -> List[QueueItem]:
        """
        Get a batch optimized for memory usage.
        
        Args:
            max_batch_memory: Maximum memory for batch in bytes
            max_items: Maximum number of items in batch
            
        Returns:
            Memory-optimized batch of items
        """
        batch = []
        current_memory = 0
        
        with self._lock:
            temp_items = []
            
            # Extract items while checking memory constraint
            for _ in range(max_items):
                if self.is_empty():
                    break
                
                item = self.get_item(timeout=0)
                if item is None:
                    break
                
                if current_memory + item.size <= max_batch_memory:
                    batch.append(item)
                    current_memory += item.size
                else:
                    # Put item back if it doesn't fit
                    temp_items.append(item)
                    break
            
            # Put back items that didn't fit
            for item in reversed(temp_items):
                if self.enable_priority:
                    heapq.heappush(self._priority_queue, item)
                else:
                    self._queue.appendleft(item)
        
        self.logger.debug(f"Created memory-optimized batch: {len(batch)} items, {current_memory} bytes")
        return batch
    
    def _sort_items(self, items: List[QueueItem]) -> List[QueueItem]:
        """Sort items according to current strategy."""
        if self.strategy == ProcessingStrategy.FIFO:
            return items  # Keep original order
        elif self.strategy == ProcessingStrategy.LIFO:
            return list(reversed(items))
        elif self.strategy == ProcessingStrategy.SIZE_ASC:
            return sorted(items, key=lambda x: x.size)
        elif self.strategy == ProcessingStrategy.SIZE_DESC:
            return sorted(items, key=lambda x: x.size, reverse=True)
        elif self.strategy == ProcessingStrategy.PRIORITY:
            return sorted(items, key=lambda x: x.priority, reverse=True)
        else:
            return items
    
    @classmethod
    def from_scan_result(cls, scan_result: ScanResult, 
                        strategy: ProcessingStrategy = ProcessingStrategy.SIZE_ASC,
                        enable_priority: bool = False) -> "ProcessingQueue":
        """
        Create a processing queue from a scan result.
        
        Args:
            scan_result: Directory scan result
            strategy: Processing strategy
            enable_priority: Whether to enable priority queue
            
        Returns:
            New processing queue with items from scan result
        """
        queue = cls(strategy=strategy, enable_priority=enable_priority)
        
        # Convert file paths to queue items
        items = []
        for file_path in scan_result.files:
            try:
                size = file_path.stat().st_size if file_path.exists() else 0
                item = QueueItem(
                    path=file_path,
                    size=size,
                    priority=0,
                    metadata={"scan_timestamp": scan_result.scan_timestamp}
                )
                items.append(item)
            except Exception as e:
                queue.logger.warning(f"Failed to create queue item for {file_path}: {e}")
        
        # Add all items to queue
        queue.add_items(items)
        
        queue.logger.info(f"Created queue from scan result: {len(items)} items")
        return queue