"""
File System Extractor with Comprehensive Error Handling.

This module provides a high-level file system data extraction interface that integrates
all file processing components with robust error handling, validation, and recovery.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

from src.core.logging import LoggerMixin
from .file_processor import FileProcessor
from .error_handler import FileErrorHandler, FileValidator, ErrorRecord, RecoveryStrategy
from .batch_config import BatchProcessingConfig
from .scanner import ScanResult
from .progress_tracker import ProcessingStats
from .exceptions import FileSystemError, FileCorruptionError


class FileSystemExtractor(LoggerMixin):
    """
    High-level file system data extractor with comprehensive error handling.
    
    Provides a robust interface for file system data extraction that integrates
    file validation, error handling, recovery strategies, and comprehensive
    reporting for production use.
    """
    
    def __init__(self, 
                 batch_config: Optional[BatchProcessingConfig] = None,
                 enable_change_detection: bool = True,
                 enable_metadata_extraction: bool = True,
                 enable_file_validation: bool = True,
                 enable_quarantine: bool = True,
                 quarantine_directory: Optional[Path] = None,
                 max_retry_attempts: int = 3,
                 change_detector_config: Optional[Dict[str, Any]] = None,
                 metadata_extractor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the file system extractor.
        
        Args:
            batch_config: Configuration for batch processing
            enable_change_detection: Whether to enable change detection
            enable_metadata_extraction: Whether to enable metadata extraction
            enable_file_validation: Whether to validate files before processing
            enable_quarantine: Whether to quarantine corrupted files
            quarantine_directory: Directory for quarantined files
            max_retry_attempts: Maximum retry attempts for failed operations
            change_detector_config: Configuration for change detector
            metadata_extractor_config: Configuration for metadata extractor
        """
        self.enable_file_validation = enable_file_validation
        self.enable_quarantine = enable_quarantine
        self.max_retry_attempts = max_retry_attempts
        
        # Initialize file processor
        self.file_processor = FileProcessor(
            batch_config=batch_config,
            enable_change_detection=enable_change_detection,
            enable_metadata_extraction=enable_metadata_extraction,
            change_detector_config=change_detector_config,
            metadata_extractor_config=metadata_extractor_config
        )
        
        # Initialize error handler
        self.error_handler = FileErrorHandler(
            max_retry_attempts=max_retry_attempts,
            enable_quarantine=enable_quarantine,
            quarantine_directory=quarantine_directory
        )
        
        # Initialize file validator
        self.file_validator = FileValidator()
        
        # Processing statistics
        self.processing_stats = {
            "total_files_processed": 0,
            "successful_files": 0,
            "failed_files": 0,
            "quarantined_files": 0,
            "validation_failures": 0,
            "retry_attempts": 0,
            "start_time": None,
            "end_time": None
        }
        
        self.logger.info("File system extractor initialized with error handling")
    
    def extract_from_directory(self, 
                              directory: Union[str, Path],
                              file_patterns: Optional[List[str]] = None,
                              exclude_patterns: Optional[List[str]] = None,
                              max_depth: Optional[int] = None,
                              follow_symlinks: bool = False,
                              only_changed_files: bool = True,
                              validate_files: Optional[bool] = None,
                              progress_callback: Optional[Callable[[ProcessingStats], None]] = None) -> Dict[str, Any]:
        """
        Extract data from a directory with comprehensive error handling.
        
        Args:
            directory: Directory to process
            file_patterns: Patterns for files to include
            exclude_patterns: Patterns for files to exclude
            max_depth: Maximum recursion depth
            follow_symlinks: Whether to follow symbolic links
            only_changed_files: Whether to only process changed files
            validate_files: Whether to validate files (overrides instance setting)
            progress_callback: Optional progress callback
            
        Returns:
            Comprehensive extraction results including errors and statistics
        """
        self._reset_processing_stats()
        self.processing_stats["start_time"] = datetime.now()
        
        try:
            self.logger.info(f"Starting directory extraction: {directory}")
            
            # Determine validation setting
            should_validate = validate_files if validate_files is not None else self.enable_file_validation
            
            # Wrap the file processor with error handling
            def enhanced_processor_func(file_path: Path) -> Dict[str, Any]:
                return self._process_file_with_error_handling(file_path, should_validate)
            
            # Process directory using file processor but with enhanced error handling
            results = self._process_directory_with_error_handling(
                directory=directory,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                max_depth=max_depth,
                follow_symlinks=follow_symlinks,
                only_changed_files=only_changed_files,
                processor_func=enhanced_processor_func,
                progress_callback=progress_callback
            )
            
            # Enhance results with error information
            results["error_handling"] = self._get_error_summary()
            results["processing_statistics"] = self._get_processing_statistics()
            
            self.processing_stats["end_time"] = datetime.now()
            self.logger.info(f"Directory extraction completed: {self.processing_stats['successful_files']} successful, {self.processing_stats['failed_files']} failed")
            
            return results
            
        except Exception as e:
            self.processing_stats["end_time"] = datetime.now()
            self.logger.error(f"Directory extraction failed: {e}")
            raise FileSystemError(f"Directory extraction failed: {e}")
    
    def extract_from_files(self, 
                          file_paths: List[Union[str, Path]],
                          validate_files: Optional[bool] = None,
                          progress_callback: Optional[Callable[[ProcessingStats], None]] = None) -> Dict[str, Any]:
        """
        Extract data from specific files with error handling.
        
        Args:
            file_paths: List of file paths to process
            validate_files: Whether to validate files (overrides instance setting)
            progress_callback: Optional progress callback
            
        Returns:
            Extraction results with error handling information
        """
        self._reset_processing_stats()
        self.processing_stats["start_time"] = datetime.now()
        
        try:
            self.logger.info(f"Starting file extraction: {len(file_paths)} files")
            
            # Determine validation setting
            should_validate = validate_files if validate_files is not None else self.enable_file_validation
            
            # Process each file with error handling
            results = []
            errors = []
            
            for file_path in file_paths:
                try:
                    result = self._process_file_with_error_handling(Path(file_path), should_validate)
                    results.append(result)
                    
                    if result.get("errors"):
                        self.processing_stats["failed_files"] += 1
                    else:
                        self.processing_stats["successful_files"] += 1
                        
                except Exception as e:
                    error_result = {
                        "path": str(file_path),
                        "errors": [f"Processing failed: {e}"],
                        "metadata": None
                    }
                    results.append(error_result)
                    errors.append(str(e))
                    self.processing_stats["failed_files"] += 1
                    self.logger.error(f"File processing failed for {file_path}: {e}")
                
                self.processing_stats["total_files_processed"] += 1
            
            # Create final results
            final_results = {
                "file_processing": {
                    "summary": {
                        "total_files": len(file_paths),
                        "successful": self.processing_stats["successful_files"],
                        "failed": self.processing_stats["failed_files"]
                    },
                    "results": results,
                    "errors": errors
                },
                "error_handling": self._get_error_summary(),
                "processing_statistics": self._get_processing_statistics(),
                "input_files": len(file_paths)
            }
            
            self.processing_stats["end_time"] = datetime.now()
            self.logger.info(f"File extraction completed: {self.processing_stats['successful_files']} successful, {self.processing_stats['failed_files']} failed")
            
            return final_results
            
        except Exception as e:
            self.processing_stats["end_time"] = datetime.now()
            self.logger.error(f"File extraction failed: {e}")
            raise FileSystemError(f"File extraction failed: {e}")
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a single file for corruption and accessibility.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Validation result with detailed information
        """
        try:
            return self.file_validator.validate_file(Path(file_path))
        except Exception as e:
            self.logger.error(f"File validation failed for {file_path}: {e}")
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
                "file_info": {},
                "validation_checks": {}
            }
    
    def get_error_report(self) -> Dict[str, Any]:
        """
        Get comprehensive error report.
        
        Returns:
            Detailed error report including statistics and problematic files
        """
        error_stats = self.error_handler.get_error_statistics()
        problematic_files = self.error_handler.get_problematic_files()
        
        return {
            "error_statistics": error_stats,
            "problematic_files": problematic_files,
            "processing_statistics": self._get_processing_statistics(),
            "quarantine_info": self._get_quarantine_info()
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        return self._get_processing_statistics()
    
    def clear_error_history(self) -> None:
        """Clear error history and reset statistics."""
        self.error_handler.error_records.clear()
        self.error_handler.error_statistics.clear()
        self._reset_processing_stats()
        self.logger.info("Error history and statistics cleared")
    
    def _process_directory_with_error_handling(self, 
                                              directory: Union[str, Path],
                                              file_patterns: Optional[List[str]],
                                              exclude_patterns: Optional[List[str]],
                                              max_depth: Optional[int],
                                              follow_symlinks: bool,
                                              only_changed_files: bool,
                                              processor_func: Callable[[Path], Dict[str, Any]],
                                              progress_callback: Optional[Callable[[ProcessingStats], None]]) -> Dict[str, Any]:
        """Process directory with enhanced error handling."""
        try:
            # Get the original file processor method but override the processor function
            original_method = self.file_processor._process_single_file
            self.file_processor._process_single_file = processor_func
            
            # Call the original directory processing
            results = self.file_processor.process_directory(
                directory=directory,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                max_depth=max_depth,
                follow_symlinks=follow_symlinks,
                only_changed_files=only_changed_files,
                progress_callback=progress_callback
            )
            
            # Restore original method
            self.file_processor._process_single_file = original_method
            
            return results
            
        except Exception as e:
            # Restore original method in case of error
            if hasattr(self.file_processor, '_process_single_file'):
                self.file_processor._process_single_file = self.file_processor._process_single_file
            raise e
    
    def _process_file_with_error_handling(self, file_path: Path, validate_file: bool = True) -> Dict[str, Any]:
        """
        Process a single file with comprehensive error handling.
        
        Args:
            file_path: Path to the file to process
            validate_file: Whether to validate the file before processing
            
        Returns:
            Processing result with error handling information
        """
        result = {
            "path": str(file_path),
            "metadata": None,
            "errors": [],
            "warnings": [],
            "validation_result": None,
            "error_handling": None
        }
        
        retry_count = 0
        max_retries = self.max_retry_attempts
        
        while retry_count <= max_retries:
            try:
                # Step 1: Validate file if enabled
                if validate_file:
                    validation_result = self.file_validator.validate_file(file_path)
                    result["validation_result"] = validation_result
                    
                    if not validation_result["is_valid"]:
                        self.processing_stats["validation_failures"] += 1
                        result["errors"].extend(validation_result["errors"])
                        result["warnings"].extend(validation_result["warnings"])
                        
                        # Handle validation failure as an error
                        validation_error = FileCorruptionError(f"File validation failed: {validation_result['errors']}")
                        error_handling_result = self.error_handler.handle_error(
                            file_path=file_path,
                            error=validation_error,
                            operation="validation",
                            retry_count=retry_count
                        )
                        
                        result["error_handling"] = error_handling_result
                        
                        # Check recovery strategy
                        if error_handling_result["should_skip"]:
                            break
                        elif error_handling_result["should_retry"] and retry_count < max_retries:
                            retry_count += 1
                            self.processing_stats["retry_attempts"] += 1
                            continue
                        else:
                            break
                
                # Step 2: Process file using original file processor
                processing_result = self.file_processor._process_single_file(file_path)
                
                # Merge results
                result["metadata"] = processing_result.get("metadata")
                result["basic_info"] = processing_result.get("basic_info")
                if processing_result.get("errors"):
                    result["errors"].extend(processing_result["errors"])
                
                # If we got here without exceptions, processing was successful
                break
                
            except Exception as e:
                # Handle processing error
                error_handling_result = self.error_handler.handle_error(
                    file_path=file_path,
                    error=e,
                    operation="processing",
                    retry_count=retry_count
                )
                
                result["error_handling"] = error_handling_result
                result["errors"].append(f"Processing error (attempt {retry_count + 1}): {e}")
                
                # Check recovery strategy
                if error_handling_result["should_skip"]:
                    break
                elif error_handling_result["should_retry"] and retry_count < max_retries:
                    retry_count += 1
                    self.processing_stats["retry_attempts"] += 1
                    continue
                else:
                    break
        
        # Update quarantine statistics
        if result.get("error_handling", {}).get("recovery_result", {}).get("quarantined"):
            self.processing_stats["quarantined_files"] += 1
        
        return result
    
    def _get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary."""
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "total_errors": error_stats["total_errors"],
            "error_categories": error_stats["error_by_category"],
            "error_severities": error_stats["error_by_severity"],
            "recovery_strategies": error_stats["recovery_strategies_used"],
            "most_common_errors": error_stats["most_common_errors"][:5]  # Top 5
        }
    
    def _get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        
        # Calculate processing time
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["processing_duration_seconds"] = duration.total_seconds()
        
        # Calculate success rate
        if stats["total_files_processed"] > 0:
            stats["success_rate"] = stats["successful_files"] / stats["total_files_processed"]
            stats["failure_rate"] = stats["failed_files"] / stats["total_files_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    def _get_quarantine_info(self) -> Dict[str, Any]:
        """Get quarantine information."""
        return {
            "enabled": self.enable_quarantine,
            "directory": str(self.error_handler.quarantine_directory) if self.error_handler.quarantine_directory else None,
            "quarantined_files": self.processing_stats["quarantined_files"]
        }
    
    def _reset_processing_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats.update({
            "total_files_processed": 0,
            "successful_files": 0,
            "failed_files": 0,
            "quarantined_files": 0,
            "validation_failures": 0,
            "retry_attempts": 0,
            "start_time": None,
            "end_time": None
        })
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        base_config = self.file_processor.get_configuration()
        
        return {
            **base_config,
            "error_handling": {
                "enable_file_validation": self.enable_file_validation,
                "enable_quarantine": self.enable_quarantine,
                "max_retry_attempts": self.max_retry_attempts,
                "quarantine_directory": str(self.error_handler.quarantine_directory) if self.error_handler.quarantine_directory else None
            }
        }