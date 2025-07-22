"""
Pipeline stages and processing components.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from src.core.logging import LoggerMixin
from src.core.exceptions import PipelineError


class ProcessingStageType(Enum):
    """Types of processing stages in the pipeline."""
    DOCUMENT_VALIDATION = "document_validation"
    TEXT_CLEANING = "text_cleaning"
    TEXT_SPLITTING = "text_splitting"
    EMBEDDING_GENERATION = "embedding_generation"
    METADATA_ENRICHMENT = "metadata_enrichment"
    ACCESS_CONTROL_TAGGING = "access_control_tagging"
    VECTOR_STORAGE = "vector_storage"
    INDEX_OPTIMIZATION = "index_optimization"


@dataclass
class StageMetrics:
    """Metrics for a processing stage."""
    stage_name: str
    processed_items: int = 0
    failed_items: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_per_second: float = 0.0
    last_processed: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_success(self, processing_time: float, memory_usage: float = 0.0):
        """Update metrics for successful processing."""
        self.processed_items += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / max(self.processed_items, 1)
        self.memory_usage_mb = max(self.memory_usage_mb, memory_usage)
        self.last_processed = datetime.utcnow()
        
        # Calculate throughput
        if self.total_processing_time > 0:
            self.throughput_per_second = self.processed_items / self.total_processing_time
    
    def update_failure(self, error: str):
        """Update metrics for failed processing."""
        self.failed_items += 1
        self.error_count += 1
        self.last_error = error
        self.last_processed = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        total = self.processed_items + self.failed_items
        if total == 0:
            return 100.0
        return (self.processed_items / total) * 100.0


@dataclass
class StageConfig:
    """Configuration for a processing stage."""
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    parallel_workers: int = 1
    batch_size: int = 10
    memory_limit_mb: Optional[float] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingContext:
    """Context passed through pipeline stages."""
    document_id: str
    batch_id: Optional[str] = None
    user_id: Optional[str] = None
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_control_tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProcessingData:
    """Data container for pipeline processing."""
    content: Any
    context: ProcessingContext
    stage_outputs: Dict[ProcessingStageType, Any] = field(default_factory=dict)
    
    def update_context(self, **kwargs):
        """Update context metadata."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value
        self.context.updated_at = datetime.utcnow()


class PipelineStage(ABC, LoggerMixin):
    """
    Abstract base class for pipeline stages.
    
    Each stage processes data and passes it to the next stage.
    Stages are designed to be modular, reusable, and configurable.
    """
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig):
        """
        Initialize pipeline stage.
        
        Args:
            stage_type: Type of the processing stage
            config: Configuration for the stage
        """
        self.stage_type = stage_type
        self.config = config
        self.metrics = StageMetrics(stage_name=stage_type.value)
        self.is_initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the stage. Called once before processing."""
        if not self.is_initialized:
            await self._initialize_stage()
            self.is_initialized = True
            self.logger.info(f"Stage {self.stage_type.value} initialized")
    
    @abstractmethod
    async def _initialize_stage(self) -> None:
        """Subclasses implement their initialization logic here."""
        pass
    
    async def process(self, data: ProcessingData) -> ProcessingData:
        """
        Process data through this stage.
        
        Args:
            data: Input data to process
            
        Returns:
            ProcessingData: Processed data
            
        Raises:
            PipelineError: If processing fails after retries
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.config.enabled:
            self.logger.info(f"Stage {self.stage_type.value} is disabled, skipping")
            return data
        
        async with self._lock:
            start_time = time.time()
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    # Apply timeout if configured
                    if self.config.timeout:
                        processed_data = await asyncio.wait_for(
                            self._process_data(data),
                            timeout=self.config.timeout
                        )
                    else:
                        processed_data = await self._process_data(data)
                    
                    # Update metrics
                    processing_time = time.time() - start_time
                    self.metrics.update_success(processing_time)
                    
                    # Store stage output
                    processed_data.stage_outputs[self.stage_type] = processed_data.content
                    
                    self.logger.debug(
                        f"Stage {self.stage_type.value} completed in {processing_time:.3f}s"
                    )
                    
                    return processed_data
                    
                except Exception as e:
                    error_msg = f"Stage {self.stage_type.value} failed (attempt {attempt + 1}): {e}"
                    self.logger.warning(error_msg)
                    
                    if attempt == self.config.max_retries:
                        self.metrics.update_failure(str(e))
                        raise PipelineError(f"Stage {self.stage_type.value} failed after {self.config.max_retries + 1} attempts: {e}")
                    
                    # Wait before retrying
                    if self.config.retry_delay > 0:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    @abstractmethod
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """
        Subclasses implement their processing logic here.
        
        Args:
            data: Input data to process
            
        Returns:
            ProcessingData: Processed data
        """
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources. Called when pipeline is shut down."""
        await self._cleanup_stage()
        self.logger.info(f"Stage {self.stage_type.value} cleaned up")
    
    async def _cleanup_stage(self) -> None:
        """Subclasses implement their cleanup logic here."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stage metrics."""
        return {
            "stage_name": self.metrics.stage_name,
            "stage_type": self.stage_type.value,
            "enabled": self.config.enabled,
            "processed_items": self.metrics.processed_items,
            "failed_items": self.metrics.failed_items,
            "success_rate": self.metrics.get_success_rate(),
            "average_processing_time": self.metrics.average_processing_time,
            "total_processing_time": self.metrics.total_processing_time,
            "throughput_per_second": self.metrics.throughput_per_second,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "error_count": self.metrics.error_count,
            "last_error": self.metrics.last_error,
            "last_processed": self.metrics.last_processed.isoformat() if self.metrics.last_processed else None,
            "configuration": {
                "max_retries": self.config.max_retries,
                "retry_delay": self.config.retry_delay,
                "timeout": self.config.timeout,
                "parallel_workers": self.config.parallel_workers,
                "batch_size": self.config.batch_size,
                "memory_limit_mb": self.config.memory_limit_mb
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the stage."""
        is_healthy = (
            self.is_initialized and
            self.config.enabled and
            self.metrics.get_success_rate() > 80.0 and
            self.metrics.error_count < 10
        )
        
        return {
            "stage_type": self.stage_type.value,
            "healthy": is_healthy,
            "initialized": self.is_initialized,
            "enabled": self.config.enabled,
            "success_rate": self.metrics.get_success_rate(),
            "error_count": self.metrics.error_count,
            "last_error": self.metrics.last_error
        }


class ParallelPipelineStage(PipelineStage):
    """
    Pipeline stage that supports parallel processing of batch data.
    """
    
    async def process_batch(self, data_batch: List[ProcessingData]) -> List[ProcessingData]:
        """
        Process a batch of data in parallel.
        
        Args:
            data_batch: List of data items to process
            
        Returns:
            List[ProcessingData]: Processed data items
        """
        if not self.config.enabled:
            self.logger.info(f"Stage {self.stage_type.value} is disabled, skipping batch")
            return data_batch
        
        # Determine number of workers
        num_workers = min(self.config.parallel_workers, len(data_batch))
        batch_size = max(1, len(data_batch) // num_workers)
        
        # Split data into chunks for parallel processing
        chunks = [
            data_batch[i:i + batch_size] 
            for i in range(0, len(data_batch), batch_size)
        ]
        
        # Process chunks in parallel
        tasks = [
            self._process_chunk(chunk)
            for chunk in chunks
        ]
        
        processed_chunks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        results = []
        for chunk_result in processed_chunks:
            if isinstance(chunk_result, Exception):
                self.logger.error(f"Chunk processing failed: {chunk_result}")
                raise PipelineError(f"Batch processing failed: {chunk_result}")
            results.extend(chunk_result)
        
        return results
    
    async def _process_chunk(self, chunk: List[ProcessingData]) -> List[ProcessingData]:
        """Process a chunk of data items."""
        processed_items = []
        
        for data_item in chunk:
            try:
                processed_item = await self.process(data_item)
                processed_items.append(processed_item)
            except Exception as e:
                self.logger.error(f"Failed to process item {data_item.context.document_id}: {e}")
                raise
        
        return processed_items


# Utility function for creating stage configurations
def create_stage_config(
    enabled: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    parallel_workers: int = 1,
    batch_size: int = 10,
    memory_limit_mb: Optional[float] = None,
    **custom_params
) -> StageConfig:
    """Create a stage configuration with common parameters."""
    return StageConfig(
        enabled=enabled,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
        parallel_workers=parallel_workers,
        batch_size=batch_size,
        memory_limit_mb=memory_limit_mb,
        custom_params=custom_params
    )