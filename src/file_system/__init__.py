"""
File System Data Ingestion Module.

This module provides comprehensive file system data ingestion capabilities
including file parsing, change detection, metadata extraction, batch processing,
and error handling with corruption detection.
"""

from .extractor import FileSystemExtractor
from .parsers import (
    BaseParser,
    TextParser,
    PDFParser,
    WordParser,
    ExcelParser,
    MarkdownParser,
    ParserFactory
)
from .scanner import DirectoryScanner, ScanResult
from .change_detector import FileChangeDetector, ChangeRecord, ChangeType, FileState
from .metadata_extractor import MetadataExtractor, FileMetadata
from .metadata_types import (
    BasicFileMetadata,
    FormatSpecificMetadata,
    PDFMetadata,
    WordMetadata,
    ExcelMetadata,
    TextMetadata,
    ImageMetadata,
    MetadataStats
)
from .batch_processor import BatchProcessor, ProcessingState
from .batch_config import BatchProcessingConfig, ProcessingStrategy, RetryStrategy
from .progress_tracker import ProgressTracker, ProcessingStats
from .processing_queue import ProcessingQueue, QueueItem
from .error_handler import (
    FileErrorHandler,
    FileValidator,
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy
)
from .file_processor import FileProcessor
from .exceptions import (
    FileSystemError,
    ParsingError,
    UnsupportedFileTypeError,
    FileChangeDetectionError,
    MetadataExtractionError,
    BatchProcessingError,
    FileCorruptionError,
    FilePermissionError,
    DirectoryScanError
)

__all__ = [
    # Main components
    "FileSystemExtractor",
    "FileProcessor",
    
    # Parsers
    "BaseParser",
    "TextParser",
    "PDFParser",
    "WordParser",
    "ExcelParser",
    "MarkdownParser",
    "ParserFactory",
    
    # Directory scanning
    "DirectoryScanner",
    "ScanResult",
    
    # Change detection
    "FileChangeDetector",
    "ChangeRecord",
    "ChangeType",
    "FileState",
    
    # Metadata extraction
    "MetadataExtractor",
    "FileMetadata",
    "BasicFileMetadata",
    "FormatSpecificMetadata",
    "PDFMetadata",
    "WordMetadata",
    "ExcelMetadata",
    "TextMetadata",
    "ImageMetadata",
    "MetadataStats",
    
    # Batch processing
    "BatchProcessor",
    "ProcessingState",
    "BatchProcessingConfig",
    "ProcessingStrategy",
    "RetryStrategy",
    "ProgressTracker",
    "ProcessingStats",
    "ProcessingQueue",
    "QueueItem",
    
    # Error handling
    "FileErrorHandler",
    "FileValidator",
    "ErrorRecord",
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    
    # Exceptions
    "FileSystemError",
    "ParsingError",
    "UnsupportedFileTypeError",
    "FileChangeDetectionError",
    "MetadataExtractionError",
    "BatchProcessingError",
    "FileCorruptionError",
    "FilePermissionError",
    "DirectoryScanError"
]