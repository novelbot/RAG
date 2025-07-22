"""
Vector Processing Pipeline Package.

This package provides a comprehensive framework for processing documents through
a vector pipeline that includes text processing, embedding generation, and vector storage.
"""

from .pipeline import VectorPipeline, PipelineConfig, ProcessingResult
from .stages import PipelineStage, ProcessingStageType
from .batch_processor import BatchProcessor, BatchConfig
from .monitoring import PipelineMetrics, PipelineStatus

__all__ = [
    "VectorPipeline",
    "PipelineConfig", 
    "ProcessingResult",
    "PipelineStage",
    "ProcessingStageType",
    "BatchProcessor",
    "BatchConfig",
    "PipelineMetrics",
    "PipelineStatus"
]