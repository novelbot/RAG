"""
Batch Processing System for File Operations.

This module provides a comprehensive batch processing system with parallel execution,
progress tracking, memory management, and error handling for file operations.
"""

import gc
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

from src.core.logging import LoggerMixin
from .exceptions import FileSystemError, BatchProcessingError
from .scanner import ScanResult
from .batch_config import BatchProcessingConfig, RetryStrategy
from .progress_tracker import ProgressTracker, ProcessingStats
from .processing_queue import ProcessingQueue, QueueItem
from .metadata_extractor import MetadataExtractor


class ProcessingState:
    """Enumeration of processing states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BatchProcessor(LoggerMixin):
    """
    Comprehensive batch processing system for file operations.
    
    Provides parallel processing, progress tracking, memory management,
    error handling, and pause/resume capabilities for large file sets.
    """
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchProcessingConfig()
        self.state = ProcessingState.IDLE
        
        # Core components
        self.queue: Optional[ProcessingQueue] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # State management
        self._state_lock = threading.RLock()
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        
        # Processing context
        self._active_futures: Dict[Future, QueueItem] = {}
        self._results: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        
        # Memory management
        self._last_gc_count = 0
        
        # Checkpointing
        self._checkpoint_data: Dict[str, Any] = {}
        
        self.logger.info("Batch processor initialized")
    
    def process_files(self, scan_result: ScanResult, 
                     processor_func: Callable[[Path], Any],
                     progress_callback: Optional[Callable[[ProcessingStats], None]] = None) -> Dict[str, Any]:
        """
        Process files from scan result using the provided processor function.
        
        Args:
            scan_result: Scan result containing files to process
            processor_func: Function to process each file
            progress_callback: Optional progress callback function
            
        Returns:
            Processing results and statistics
            
        Raises:
            BatchProcessingError: If processing fails
        """
        try:
            self._validate_processing_state()
            self._initialize_processing(scan_result, progress_callback)
            
            self.logger.info(f"Starting batch processing of {len(scan_result.files)} files")
            
            with self._state_lock:
                self.state = ProcessingState.RUNNING
            
            # Start progress tracking
            self.progress_tracker.start()
            
            # Create and start thread pool
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="batch_worker"
            )
            
            try:
                # Process files in batches
                self._process_batches(processor_func)
                
                # Wait for all remaining futures to complete
                self._wait_for_completion()
                
            finally:
                # Clean up executor
                if self.executor:
                    self.executor.shutdown(wait=True)
                    self.executor = None
            
            # Finish progress tracking
            self.progress_tracker.finish()
            
            # Generate final results
            results = self._generate_results()
            
            with self._state_lock:
                self.state = ProcessingState.STOPPED
            
            self.logger.info(f"Batch processing completed: {len(self._results)} successful, {len(self._errors)} failed")
            
            return results
            
        except Exception as e:
            with self._state_lock:
                self.state = ProcessingState.ERROR
            
            self.logger.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(f"Batch processing failed: {e}")
    
    def pause(self) -> None:
        """Pause processing."""
        with self._state_lock:
            if self.state == ProcessingState.RUNNING:
                self.state = ProcessingState.PAUSED
                self._pause_event.clear()
                self.logger.info("Batch processing paused")
    
    def resume(self) -> None:
        """Resume processing."""
        with self._state_lock:
            if self.state == ProcessingState.PAUSED:
                self.state = ProcessingState.RUNNING
                self._pause_event.set()
                self.logger.info("Batch processing resumed")
    
    def stop(self) -> None:
        """Stop processing."""
        with self._state_lock:
            if self.state in [ProcessingState.RUNNING, ProcessingState.PAUSED]:
                self.state = ProcessingState.STOPPING
                self._stop_event.set()
                self._pause_event.set()  # Unblock any paused threads
                self.logger.info("Batch processing stop requested")
    
    def get_state(self) -> str:
        """Get current processing state."""
        with self._state_lock:
            return self.state
    
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get current progress information."""
        if self.progress_tracker:
            return self.progress_tracker.get_status_summary()
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = {
            "state": self.get_state(),
            "config": self.config.to_dict(),
            "results_count": len(self._results),
            "errors_count": len(self._errors),
            "active_futures": len(self._active_futures)
        }
        
        if self.progress_tracker:
            stats["progress"] = self.progress_tracker.get_status_summary()
        
        if self.queue:
            stats["queue"] = self.queue.get_statistics()
        
        return stats
    
    def save_checkpoint(self, checkpoint_file: Optional[str] = None) -> None:
        """Save processing checkpoint."""
        if not self.config.enable_checkpoints:
            return
        
        checkpoint_file = checkpoint_file or self.config.checkpoint_file
        if not checkpoint_file:
            return
        
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "state": self.get_state(),
                "progress": self.get_progress(),
                "remaining_items": self.queue.size() if self.queue else 0,
                "results_count": len(self._results),
                "errors_count": len(self._errors)
            }
            
            # Save checkpoint (implementation would depend on desired format)
            self.logger.debug(f"Checkpoint saved: {checkpoint_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _validate_processing_state(self) -> None:
        """Validate that processing can be started."""
        with self._state_lock:
            if self.state not in [ProcessingState.IDLE, ProcessingState.STOPPED]:
                raise BatchProcessingError(f"Cannot start processing in state: {self.state}")
    
    def _initialize_processing(self, scan_result: ScanResult, 
                             progress_callback: Optional[Callable[[ProcessingStats], None]]) -> None:
        """Initialize processing components."""
        # Reset state
        self._active_futures.clear()
        self._results.clear()
        self._errors.clear()
        self._stop_event.clear()
        self._pause_event.set()
        
        # Create processing queue
        self.queue = ProcessingQueue.from_scan_result(
            scan_result, 
            strategy=self.config.processing_strategy,
            enable_priority=self.config.enable_priority_queue
        )
        
        # Create progress tracker
        self.progress_tracker = ProgressTracker(
            total_items=len(scan_result.files),
            enable_eta=self.config.enable_eta_calculation,
            performance_window=self.config.performance_window
        )
        
        # Add progress callback if provided
        if progress_callback:
            self.progress_tracker.add_progress_callback(progress_callback)
        
        # Add checkpoint callback if enabled
        if self.config.enable_checkpoints:
            self.progress_tracker.add_progress_callback(
                lambda stats: self._checkpoint_callback(stats)
            )
    
    def _process_batches(self, processor_func: Callable[[Path], Any]) -> None:
        """Process files in batches."""
        while not self._stop_event.is_set() and not self.queue.is_empty():
            # Wait if paused
            self._pause_event.wait()
            
            if self._stop_event.is_set():
                break
            
            # Check memory usage
            if self._should_trigger_gc():
                self._perform_gc()
            
            # Get next batch
            batch = self._get_next_batch()
            if not batch:
                break
            
            # Submit batch for processing
            self._submit_batch(batch, processor_func)
            
            # Process completed futures
            self._process_completed_futures()
    
    def _get_next_batch(self) -> List[QueueItem]:
        """Get the next batch of items to process."""
        if self.config.enable_adaptive_batching:
            # Adjust batch size based on performance
            performance = self.progress_tracker.get_performance_score()
            batch_size = self.config.get_optimal_batch_size(performance)
        else:
            batch_size = self.config.batch_size
        
        # Get memory-optimized batch if memory limit is set
        if self.config.memory_limit_mb:
            max_batch_memory = self.config.get_memory_limit_bytes() // 4  # Use 1/4 of limit per batch
            return self.queue.get_memory_optimized_batch(max_batch_memory, batch_size)
        else:
            return self.queue.get_batch(batch_size, timeout=1.0)
    
    def _submit_batch(self, batch: List[QueueItem], processor_func: Callable[[Path], Any]) -> None:
        """Submit a batch of items for processing."""
        for item in batch:
            if self._stop_event.is_set():
                break
            
            # Check if we have available workers
            if len(self._active_futures) >= self.config.max_workers:
                self._process_completed_futures()
            
            # Submit item for processing
            future = self.executor.submit(
                self._process_item_with_retry,
                item,
                processor_func
            )
            
            self._active_futures[future] = item
    
    def _process_item_with_retry(self, item: QueueItem, 
                               processor_func: Callable[[Path], Any]) -> Tuple[bool, Any, Optional[str]]:
        """Process a single item with retry logic."""
        last_error = None
        
        for attempt in range(1, self.config.max_retries + 2):  # +1 for initial attempt
            try:
                # Wait if paused
                self._pause_event.wait()
                
                if self._stop_event.is_set():
                    return False, None, "Processing stopped"
                
                # Process the item
                start_time = time.time()
                result = processor_func(item.path)
                end_time = time.time()
                
                # Calculate processing time
                processing_time = end_time - start_time
                
                return True, {
                    "result": result,
                    "processing_time": processing_time,
                    "attempt": attempt,
                    "file_size": item.size
                }, None
                
            except Exception as e:
                last_error = str(e)
                error_type = type(e).__name__
                
                self.logger.warning(f"Processing failed for {item.path} (attempt {attempt}): {e}")
                
                # Check if we should retry
                if not self.config.should_retry(attempt):
                    break
                
                # Calculate retry delay
                delay = self.config.calculate_retry_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
        
        return False, None, last_error
    
    def _process_completed_futures(self) -> None:
        """Process completed futures and update progress."""
        completed_futures = []
        
        for future in list(self._active_futures.keys()):
            if future.done():
                completed_futures.append(future)
        
        for future in completed_futures:
            item = self._active_futures.pop(future)
            
            try:
                success, result, error = future.result(timeout=0.1)
                
                if success:
                    self._results.append({
                        "path": str(item.path),
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update progress
                    file_size = result.get("file_size", 0) if isinstance(result, dict) else item.size
                    self.progress_tracker.update(
                        processed_count=1,
                        bytes_processed=file_size
                    )
                else:
                    self._errors.append({
                        "path": str(item.path),
                        "error": error,
                        "timestamp": datetime.now().isoformat(),
                        "retry_count": item.retry_count
                    })
                    
                    # Update progress with error
                    self.progress_tracker.update(
                        processed_count=1,
                        error_type=type(Exception(error)).__name__ if error else "Unknown"
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to process future result for {item.path}: {e}")
                
                self._errors.append({
                    "path": str(item.path),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": item.retry_count
                })
                
                self.progress_tracker.update(
                    processed_count=1,
                    error_type=type(e).__name__
                )
    
    def _wait_for_completion(self) -> None:
        """Wait for all active futures to complete."""
        while self._active_futures and not self._stop_event.is_set():
            self._process_completed_futures()
            time.sleep(0.1)
        
        # Cancel any remaining futures if stopping
        if self._stop_event.is_set():
            for future in self._active_futures.keys():
                future.cancel()
    
    def _should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        if not self.config.enable_gc:
            return False
        
        processed_since_gc = self.progress_tracker.stats.processed_items - self._last_gc_count
        return processed_since_gc >= self.config.gc_interval
    
    def _perform_gc(self) -> None:
        """Perform garbage collection and update memory metrics."""
        self.progress_tracker.force_gc()
        self._last_gc_count = self.progress_tracker.stats.processed_items
        
        # Check memory threshold
        if self.config.memory_limit_mb:
            memory_usage = self.progress_tracker.get_memory_usage()
            memory_threshold = self.config.memory_limit_mb * self.config.memory_threshold
            
            if memory_usage["current_mb"] > memory_threshold:
                self.logger.warning(
                    f"Memory usage ({memory_usage['current_mb']:.1f}MB) "
                    f"exceeds threshold ({memory_threshold:.1f}MB)"
                )
    
    def _checkpoint_callback(self, stats: ProcessingStats) -> None:
        """Callback for saving checkpoints during processing."""
        if stats.processed_items % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate final processing results."""
        return {
            "summary": {
                "total_files": self.progress_tracker.stats.total_items,
                "successful": len(self._results),
                "failed": len(self._errors),
                "success_rate": len(self._results) / max(1, self.progress_tracker.stats.total_items),
                "processing_time": self.progress_tracker.get_elapsed_time()
            },
            "performance": self.progress_tracker.get_throughput(),
            "memory": self.progress_tracker.get_memory_usage(),
            "results": self._results,
            "errors": self._errors,
            "stats": self.progress_tracker.get_statistics().to_dict()
        }