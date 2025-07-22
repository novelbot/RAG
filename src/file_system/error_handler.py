"""
Comprehensive Error Handling System for File Processing.

This module provides robust error handling, classification, recovery mechanisms,
and corruption detection for file processing operations.
"""

import os
import hashlib
import traceback
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Type

from src.core.logging import LoggerMixin
from .exceptions import (
    FileSystemError, ParsingError, UnsupportedFileTypeError,
    FileCorruptionError, FilePermissionError, MetadataExtractionError
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of file processing errors."""
    CORRUPTION = "corruption"
    PERMISSION = "permission"
    UNSUPPORTED_FORMAT = "unsupported_format"
    PARSING_ERROR = "parsing_error"
    METADATA_ERROR = "metadata_error"
    IO_ERROR = "io_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    QUARANTINE = "quarantine"
    REPAIR = "repair"
    ABORT = "abort"


@dataclass
class ErrorRecord:
    """Record of a file processing error."""
    file_path: Path
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: Optional[str] = None
    recovery_strategy: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary."""
        return {
            "file_path": str(self.file_path),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }


class FileValidator(LoggerMixin):
    """File validation and corruption detection utilities."""
    
    def __init__(self):
        """Initialize the file validator."""
        # File signatures for common formats
        self.file_signatures = {
            '.pdf': [b'%PDF'],
            '.docx': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
            '.xlsx': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
            '.zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
            '.jpg': [b'\xff\xd8\xff'],
            '.jpeg': [b'\xff\xd8\xff'],
            '.png': [b'\x89PNG\r\n\x1a\n'],
            '.gif': [b'GIF87a', b'GIF89a'],
            '.bmp': [b'BM'],
            '.tiff': [b'II*\x00', b'MM\x00*'],
            '.mp3': [b'ID3', b'\xff\xfb'],
            '.mp4': [b'ftyp'],
            '.avi': [b'RIFF'],
        }
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a file for corruption and accessibility.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Validation result with details
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {},
            "validation_checks": {}
        }
        
        try:
            # Check if file exists
            if not file_path.exists():
                result["is_valid"] = False
                result["errors"].append("File does not exist")
                return result
            
            # Check if it's actually a file
            if not file_path.is_file():
                result["is_valid"] = False
                result["errors"].append("Path is not a file")
                return result
            
            # Get basic file info
            stat_info = file_path.stat()
            result["file_info"] = {
                "size": stat_info.st_size,
                "modified_time": stat_info.st_mtime,
                "permissions": oct(stat_info.st_mode)[-3:],
                "extension": file_path.suffix.lower()
            }
            
            # Check file size
            if stat_info.st_size == 0:
                result["warnings"].append("File is empty")
            
            # Check file permissions
            result["validation_checks"]["readable"] = os.access(file_path, os.R_OK)
            if not result["validation_checks"]["readable"]:
                result["is_valid"] = False
                result["errors"].append("File is not readable")
            
            # Validate file signature
            signature_result = self._validate_file_signature(file_path)
            result["validation_checks"]["signature"] = signature_result
            if not signature_result["valid"]:
                result["warnings"].extend(signature_result["warnings"])
            
            # Check for basic corruption indicators
            corruption_result = self._check_basic_corruption(file_path)
            result["validation_checks"]["corruption"] = corruption_result
            if corruption_result["likely_corrupted"]:
                result["is_valid"] = False
                result["errors"].extend(corruption_result["errors"])
            
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(f"Validation failed: {e}")
            self.logger.error(f"File validation failed for {file_path}: {e}")
        
        return result
    
    def _validate_file_signature(self, file_path: Path) -> Dict[str, Any]:
        """Validate file signature matches extension."""
        result = {
            "valid": True,
            "warnings": [],
            "detected_type": None
        }
        
        try:
            extension = file_path.suffix.lower()
            if extension not in self.file_signatures:
                result["warnings"].append(f"Unknown file type: {extension}")
                return result
            
            # Read first few bytes
            with open(file_path, 'rb') as f:
                header = f.read(32)
            
            if not header:
                result["valid"] = False
                result["warnings"].append("File appears to be empty")
                return result
            
            # Check against known signatures
            expected_signatures = self.file_signatures[extension]
            signature_matches = any(header.startswith(sig) for sig in expected_signatures)
            
            if not signature_matches:
                result["valid"] = False
                result["warnings"].append(f"File signature does not match extension {extension}")
                
                # Try to detect actual type
                for ext, signatures in self.file_signatures.items():
                    if any(header.startswith(sig) for sig in signatures):
                        result["detected_type"] = ext
                        result["warnings"].append(f"File appears to be {ext} format")
                        break
            
        except Exception as e:
            result["warnings"].append(f"Signature validation failed: {e}")
        
        return result
    
    def _check_basic_corruption(self, file_path: Path) -> Dict[str, Any]:
        """Check for basic corruption indicators."""
        result = {
            "likely_corrupted": False,
            "errors": [],
            "checks_performed": []
        }
        
        try:
            # Check if file can be opened and read
            result["checks_performed"].append("read_test")
            try:
                with open(file_path, 'rb') as f:
                    # Try to read the entire file in chunks
                    chunk_size = 8192
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
            except Exception as e:
                result["likely_corrupted"] = True
                result["errors"].append(f"File read error: {e}")
            
            # Check for null bytes in text files
            if file_path.suffix.lower() in ['.txt', '.md', '.py', '.json', '.xml', '.html', '.css']:
                result["checks_performed"].append("null_byte_test")
                try:
                    with open(file_path, 'rb') as f:
                        sample = f.read(1024)  # Read first 1KB
                        if b'\x00' in sample:
                            result["errors"].append("Null bytes found in text file")
                except Exception:
                    pass  # Already handled in read test
            
            # Check for extremely small files that should be larger
            stat_info = file_path.stat()
            extension = file_path.suffix.lower()
            min_sizes = {
                '.pdf': 100,
                '.docx': 1000,
                '.xlsx': 1000,
                '.jpg': 50,
                '.png': 50,
            }
            
            if extension in min_sizes and stat_info.st_size < min_sizes[extension]:
                result["errors"].append(f"File too small for {extension} format")
            
        except Exception as e:
            result["errors"].append(f"Corruption check failed: {e}")
        
        return result
    
    def calculate_file_checksum(self, file_path: Path, algorithm: str = "md5") -> Optional[str]:
        """Calculate file checksum for integrity verification."""
        try:
            if algorithm == "md5":
                hasher = hashlib.md5()
            elif algorithm == "sha256":
                hasher = hashlib.sha256()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Checksum calculation failed for {file_path}: {e}")
            return None


class FileErrorHandler(LoggerMixin):
    """
    Comprehensive error handling system for file processing.
    
    Provides error classification, recovery strategies, logging,
    and corruption detection for robust file processing.
    """
    
    def __init__(self, 
                 max_retry_attempts: int = 3,
                 enable_quarantine: bool = True,
                 quarantine_directory: Optional[Path] = None):
        """
        Initialize the error handler.
        
        Args:
            max_retry_attempts: Maximum number of retry attempts
            enable_quarantine: Whether to quarantine corrupted files
            quarantine_directory: Directory for quarantined files
        """
        self.max_retry_attempts = max_retry_attempts
        self.enable_quarantine = enable_quarantine
        self.quarantine_directory = quarantine_directory
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.error_statistics: Dict[str, int] = {}
        
        # File validator
        self.validator = FileValidator()
        
        # Error classification rules
        self.error_classification_rules = self._initialize_classification_rules()
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        self.logger.info("File error handler initialized")
    
    def handle_error(self, 
                    file_path: Path, 
                    error: Exception, 
                    operation: str = "processing",
                    retry_count: int = 0) -> Dict[str, Any]:
        """
        Handle a file processing error.
        
        Args:
            file_path: Path to the file that caused the error
            error: The exception that occurred
            operation: Description of the operation that failed
            retry_count: Number of previous retry attempts
            
        Returns:
            Error handling result with recovery strategy
        """
        try:
            # Classify the error
            category = self._classify_error(error, file_path)
            severity = self._determine_severity(error, category)
            
            # Create error record
            error_record = ErrorRecord(
                file_path=file_path,
                error_type=type(error).__name__,
                error_message=str(error),
                category=category,
                severity=severity,
                timestamp=datetime.now(),
                stack_trace=traceback.format_exc(),
                retry_count=retry_count,
                metadata={"operation": operation}
            )
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(error_record)
            error_record.recovery_strategy = recovery_strategy
            
            # Log the error
            self._log_error(error_record)
            
            # Update statistics
            self._update_error_statistics(error_record)
            
            # Store error record
            self.error_records.append(error_record)
            
            # Execute recovery strategy
            recovery_result = self._execute_recovery_strategy(error_record)
            
            return {
                "error_record": error_record.to_dict(),
                "recovery_strategy": recovery_strategy.value,
                "recovery_result": recovery_result,
                "should_retry": recovery_result.get("should_retry", False),
                "should_skip": recovery_result.get("should_skip", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error handling failed for {file_path}: {e}")
            return {
                "error_record": None,
                "recovery_strategy": RecoveryStrategy.SKIP.value,
                "recovery_result": {"should_skip": True},
                "should_retry": False,
                "should_skip": True
            }
    
    def validate_file_before_processing(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a file before processing to catch issues early.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Validation result
        """
        return self.validator.validate_file(file_path)
    
    def quarantine_file(self, file_path: Path, reason: str) -> bool:
        """
        Move a corrupted or problematic file to quarantine.
        
        Args:
            file_path: Path to the file to quarantine
            reason: Reason for quarantine
            
        Returns:
            True if file was successfully quarantined
        """
        if not self.enable_quarantine:
            return False
        
        try:
            if not self.quarantine_directory:
                self.quarantine_directory = file_path.parent / ".quarantine"
            
            # Create quarantine directory if it doesn't exist
            self.quarantine_directory.mkdir(parents=True, exist_ok=True)
            
            # Generate unique quarantine filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            quarantine_path = self.quarantine_directory / quarantine_name
            
            # Move file to quarantine
            file_path.rename(quarantine_path)
            
            # Create reason file
            reason_file = quarantine_path.with_suffix(f"{quarantine_path.suffix}.reason")
            with open(reason_file, 'w') as f:
                f.write(f"Quarantined: {datetime.now().isoformat()}\n")
                f.write(f"Original path: {file_path}\n")
                f.write(f"Reason: {reason}\n")
            
            self.logger.info(f"File quarantined: {file_path} -> {quarantine_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to quarantine file {file_path}: {e}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "total_errors": len(self.error_records),
            "error_by_category": self._get_errors_by_category(),
            "error_by_severity": self._get_errors_by_severity(),
            "error_by_type": self._get_errors_by_type(),
            "recovery_strategies_used": self._get_recovery_strategies_used(),
            "most_common_errors": self._get_most_common_errors(),
            "files_with_multiple_errors": self._get_files_with_multiple_errors()
        }
    
    def get_problematic_files(self) -> List[Dict[str, Any]]:
        """Get list of files that had processing issues."""
        files = {}
        
        for record in self.error_records:
            path_str = str(record.file_path)
            if path_str not in files:
                files[path_str] = {
                    "file_path": path_str,
                    "error_count": 0,
                    "errors": [],
                    "highest_severity": ErrorSeverity.LOW,
                    "categories": set()
                }
            
            file_info = files[path_str]
            file_info["error_count"] += 1
            file_info["errors"].append(record.to_dict())
            file_info["categories"].add(record.category.value)
            
            # Update highest severity
            severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            if severities.index(record.severity) > severities.index(file_info["highest_severity"]):
                file_info["highest_severity"] = record.severity
        
        # Convert sets to lists for JSON serialization
        for file_info in files.values():
            file_info["categories"] = list(file_info["categories"])
            file_info["highest_severity"] = file_info["highest_severity"].value
        
        return list(files.values())
    
    def _classify_error(self, error: Exception, file_path: Path) -> ErrorCategory:
        """Classify an error into appropriate category."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check classification rules
        for rule in self.error_classification_rules:
            if rule["condition"](error, error_type, error_message, file_path):
                return rule["category"]
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category."""
        # Critical errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.CORRUPTION, ErrorCategory.SYSTEM_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.PERMISSION, ErrorCategory.IO_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, error_record: ErrorRecord) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for an error."""
        category = error_record.category
        severity = error_record.severity
        retry_count = error_record.retry_count
        
        # Check if max retries exceeded
        if retry_count >= self.max_retry_attempts:
            if category == ErrorCategory.CORRUPTION:
                return RecoveryStrategy.QUARANTINE
            else:
                return RecoveryStrategy.SKIP
        
        # Strategy based on category
        strategy_map = {
            ErrorCategory.CORRUPTION: RecoveryStrategy.QUARANTINE,
            ErrorCategory.PERMISSION: RecoveryStrategy.SKIP,
            ErrorCategory.UNSUPPORTED_FORMAT: RecoveryStrategy.SKIP,
            ErrorCategory.PARSING_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.METADATA_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.IO_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.MEMORY_ERROR: RecoveryStrategy.SKIP,
            ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.SYSTEM_ERROR: RecoveryStrategy.ABORT,
        }
        
        return strategy_map.get(category, RecoveryStrategy.SKIP)
    
    def _execute_recovery_strategy(self, error_record: ErrorRecord) -> Dict[str, Any]:
        """Execute the recovery strategy for an error."""
        strategy = error_record.recovery_strategy
        result = {"executed": True, "should_retry": False, "should_skip": False}
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                result["should_retry"] = True
                
            elif strategy == RecoveryStrategy.SKIP:
                result["should_skip"] = True
                
            elif strategy == RecoveryStrategy.QUARANTINE:
                success = self.quarantine_file(
                    error_record.file_path, 
                    f"{error_record.category.value}: {error_record.error_message}"
                )
                result["quarantined"] = success
                result["should_skip"] = True
                
            elif strategy == RecoveryStrategy.FALLBACK:
                # Implement fallback logic (e.g., basic text extraction)
                result["fallback_attempted"] = True
                result["should_retry"] = True
                
            elif strategy == RecoveryStrategy.ABORT:
                result["should_abort"] = True
                
        except Exception as e:
            self.logger.error(f"Recovery strategy execution failed: {e}")
            result["executed"] = False
            result["error"] = str(e)
        
        return result
    
    def _initialize_classification_rules(self) -> List[Dict[str, Any]]:
        """Initialize error classification rules."""
        return [
            {
                "condition": lambda e, t, m, p: isinstance(e, FileCorruptionError) or "corrupt" in m,
                "category": ErrorCategory.CORRUPTION
            },
            {
                "condition": lambda e, t, m, p: isinstance(e, FilePermissionError) or "permission" in m,
                "category": ErrorCategory.PERMISSION
            },
            {
                "condition": lambda e, t, m, p: isinstance(e, UnsupportedFileTypeError) or "unsupported" in m,
                "category": ErrorCategory.UNSUPPORTED_FORMAT
            },
            {
                "condition": lambda e, t, m, p: isinstance(e, ParsingError) or "parse" in m,
                "category": ErrorCategory.PARSING_ERROR
            },
            {
                "condition": lambda e, t, m, p: isinstance(e, MetadataExtractionError) or "metadata" in m,
                "category": ErrorCategory.METADATA_ERROR
            },
            {
                "condition": lambda e, t, m, p: "timeout" in m or t == "TimeoutError",
                "category": ErrorCategory.TIMEOUT_ERROR
            },
            {
                "condition": lambda e, t, m, p: isinstance(e, MemoryError) or "memory" in m,
                "category": ErrorCategory.MEMORY_ERROR
            },
            {
                "condition": lambda e, t, m, p: "network" in m or "connection" in m,
                "category": ErrorCategory.NETWORK_ERROR
            },
            {
                "condition": lambda e, t, m, p: t in ["OSError", "IOError"] or "i/o" in m,
                "category": ErrorCategory.IO_ERROR
            }
        ]
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
        """Initialize default recovery strategies."""
        return {
            ErrorCategory.CORRUPTION: RecoveryStrategy.QUARANTINE,
            ErrorCategory.PERMISSION: RecoveryStrategy.SKIP,
            ErrorCategory.UNSUPPORTED_FORMAT: RecoveryStrategy.SKIP,
            ErrorCategory.PARSING_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.METADATA_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.IO_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.MEMORY_ERROR: RecoveryStrategy.SKIP,
            ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.SYSTEM_ERROR: RecoveryStrategy.ABORT,
        }
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log an error record appropriately."""
        level_map = {
            ErrorSeverity.LOW: "debug",
            ErrorSeverity.MEDIUM: "warning",
            ErrorSeverity.HIGH: "error",
            ErrorSeverity.CRITICAL: "critical"
        }
        
        log_level = level_map.get(error_record.severity, "error")
        message = (f"{error_record.category.value.title()} error in {error_record.file_path}: "
                  f"{error_record.error_message}")
        
        getattr(self.logger, log_level)(message)
    
    def _update_error_statistics(self, error_record: ErrorRecord) -> None:
        """Update error statistics."""
        key = f"{error_record.category.value}_{error_record.error_type}"
        self.error_statistics[key] = self.error_statistics.get(key, 0) + 1
    
    def _get_errors_by_category(self) -> Dict[str, int]:
        """Get error count by category."""
        counts = {}
        for record in self.error_records:
            category = record.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _get_errors_by_severity(self) -> Dict[str, int]:
        """Get error count by severity."""
        counts = {}
        for record in self.error_records:
            severity = record.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _get_errors_by_type(self) -> Dict[str, int]:
        """Get error count by type."""
        counts = {}
        for record in self.error_records:
            error_type = record.error_type
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts
    
    def _get_recovery_strategies_used(self) -> Dict[str, int]:
        """Get count of recovery strategies used."""
        counts = {}
        for record in self.error_records:
            if record.recovery_strategy:
                strategy = record.recovery_strategy.value
                counts[strategy] = counts.get(strategy, 0) + 1
        return counts
    
    def _get_most_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common errors."""
        sorted_errors = sorted(
            self.error_statistics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"error": error, "count": count}
            for error, count in sorted_errors[:limit]
        ]
    
    def _get_files_with_multiple_errors(self) -> List[str]:
        """Get files that had multiple errors."""
        file_error_counts = {}
        
        for record in self.error_records:
            path_str = str(record.file_path)
            file_error_counts[path_str] = file_error_counts.get(path_str, 0) + 1
        
        return [
            path for path, count in file_error_counts.items()
            if count > 1
        ]