"""
Integrated File Processing System.

This module provides a high-level interface that integrates all file system components:
directory scanning, change detection, metadata extraction, and batch processing.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union

from src.core.logging import LoggerMixin
from .scanner import DirectoryScanner, ScanResult
from .change_detector import FileChangeDetector, ChangeRecord, ChangeType
from .metadata_extractor import MetadataExtractor, FileMetadata
from .batch_processor import BatchProcessor
from .batch_config import BatchProcessingConfig
from .progress_tracker import ProcessingStats
from .exceptions import FileSystemError, BatchProcessingError


class FileProcessor(LoggerMixin):
    """
    Integrated file processing system.
    
    Combines directory scanning, change detection, metadata extraction,
    and batch processing into a unified interface for file operations.
    """
    
    def __init__(self, 
                 batch_config: Optional[BatchProcessingConfig] = None,
                 enable_change_detection: bool = True,
                 enable_metadata_extraction: bool = True,
                 change_detector_config: Optional[Dict[str, Any]] = None,
                 metadata_extractor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the file processor.
        
        Args:
            batch_config: Configuration for batch processing
            enable_change_detection: Whether to enable change detection
            enable_metadata_extraction: Whether to enable metadata extraction
            change_detector_config: Configuration for change detector
            metadata_extractor_config: Configuration for metadata extractor
        """
        self.batch_config = batch_config or BatchProcessingConfig()
        self.enable_change_detection = enable_change_detection
        self.enable_metadata_extraction = enable_metadata_extraction
        
        # Initialize components
        self.scanner = DirectoryScanner()
        
        if enable_change_detection:
            detector_config = change_detector_config or {}
            self.change_detector = FileChangeDetector(**detector_config)
        else:
            self.change_detector = None
        
        if enable_metadata_extraction:
            extractor_config = metadata_extractor_config or {}
            self.metadata_extractor = MetadataExtractor(**extractor_config)
        else:
            self.metadata_extractor = None
        
        self.batch_processor = BatchProcessor(self.batch_config)
        
        self.logger.info("File processor initialized")
    
    def process_directory(self, 
                         directory: Union[str, Path],
                         file_patterns: Optional[List[str]] = None,
                         exclude_patterns: Optional[List[str]] = None,
                         max_depth: Optional[int] = None,
                         follow_symlinks: bool = False,
                         only_changed_files: bool = True,
                         progress_callback: Optional[Callable[[ProcessingStats], None]] = None) -> Dict[str, Any]:
        """
        Process all files in a directory with full pipeline.
        
        Args:
            directory: Directory to process
            file_patterns: Patterns for files to include
            exclude_patterns: Patterns for files to exclude
            max_depth: Maximum recursion depth
            follow_symlinks: Whether to follow symbolic links
            only_changed_files: Whether to only process changed files
            progress_callback: Optional progress callback
            
        Returns:
            Processing results including metadata and statistics
        """
        try:
            self.logger.info(f"Starting directory processing: {directory}")
            
            # Step 1: Scan directory
            scan_result = self.scanner.scan_directory(
                directory=directory,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                max_depth=max_depth,
                follow_symlinks=follow_symlinks
            )
            
            self.logger.info(f"Directory scan completed: {len(scan_result.files)} files found")
            
            # Step 2: Detect changes if enabled
            files_to_process = scan_result.files
            changes = []
            
            if self.enable_change_detection and self.change_detector and only_changed_files:
                changes = self.change_detector.detect_changes(scan_result)
                
                # Filter to only changed files
                changed_paths = {
                    change.path for change in changes 
                    if change.change_type in [ChangeType.NEW, ChangeType.MODIFIED]
                }
                
                files_to_process = [f for f in scan_result.files if f in changed_paths]
                
                self.logger.info(f"Change detection completed: {len(files_to_process)} files to process")
            
            # Step 3: Process files with batch processor
            if files_to_process:
                # Create a new scan result with only files to process
                filtered_scan_result = ScanResult(
                    directory=scan_result.directory,
                    files=files_to_process,
                    scan_timestamp=scan_result.scan_timestamp,
                    patterns=scan_result.patterns,
                    exclude_patterns=scan_result.exclude_patterns,
                    errors=scan_result.errors
                )
                
                # Define processing function
                def process_file(file_path: Path) -> Dict[str, Any]:
                    return self._process_single_file(file_path)
                
                # Run batch processing
                batch_results = self.batch_processor.process_files(
                    scan_result=filtered_scan_result,
                    processor_func=process_file,
                    progress_callback=progress_callback
                )
                
                # Combine results
                results = {
                    "scan_result": {
                        "total_files": len(scan_result.files),
                        "files_to_process": len(files_to_process),
                        "scan_time": scan_result.scan_timestamp,
                        "scan_errors": scan_result.errors
                    },
                    "change_detection": {
                        "enabled": self.enable_change_detection,
                        "changes": [change.to_dict() for change in changes] if changes else []
                    },
                    "batch_processing": batch_results
                }
                
            else:
                self.logger.info("No files to process")
                results = {
                    "scan_result": {
                        "total_files": len(scan_result.files),
                        "files_to_process": 0,
                        "scan_time": scan_result.scan_timestamp,
                        "scan_errors": scan_result.errors
                    },
                    "change_detection": {
                        "enabled": self.enable_change_detection,
                        "changes": [change.to_dict() for change in changes] if changes else []
                    },
                    "batch_processing": {
                        "summary": {"total_files": 0, "successful": 0, "failed": 0},
                        "results": [],
                        "errors": []
                    }
                }
            
            self.logger.info(f"Directory processing completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {e}")
            raise FileSystemError(f"Directory processing failed: {e}")
    
    def process_files(self, 
                     file_paths: List[Union[str, Path]],
                     progress_callback: Optional[Callable[[ProcessingStats], None]] = None) -> Dict[str, Any]:
        """
        Process a specific list of files.
        
        Args:
            file_paths: List of file paths to process
            progress_callback: Optional progress callback
            
        Returns:
            Processing results
        """
        try:
            self.logger.info(f"Starting file processing: {len(file_paths)} files")
            
            # Convert to Path objects and create scan result
            paths = [Path(p) for p in file_paths]
            scan_result = ScanResult(
                directory=Path.cwd(),  # Dummy directory
                files=paths,
                scan_timestamp=self.scanner._get_current_timestamp(),
                patterns=[],
                exclude_patterns=[],
                errors=[]
            )
            
            # Define processing function
            def process_file(file_path: Path) -> Dict[str, Any]:
                return self._process_single_file(file_path)
            
            # Run batch processing
            batch_results = self.batch_processor.process_files(
                scan_result=scan_result,
                processor_func=process_file,
                progress_callback=progress_callback
            )
            
            results = {
                "file_processing": batch_results,
                "input_files": len(file_paths)
            }
            
            self.logger.info(f"File processing completed")
            return results
            
        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            raise FileSystemError(f"File processing failed: {e}")
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """
        Extract metadata from a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata or None if extraction failed
        """
        if not self.enable_metadata_extraction or not self.metadata_extractor:
            return None
        
        try:
            return self.metadata_extractor.extract_metadata(file_path)
        except Exception as e:
            self.logger.error(f"Metadata extraction failed for {file_path}: {e}")
            return None
    
    def detect_file_changes(self, scan_result: ScanResult) -> List[ChangeRecord]:
        """
        Detect changes in files from a scan result.
        
        Args:
            scan_result: Directory scan result
            
        Returns:
            List of detected changes
        """
        if not self.enable_change_detection or not self.change_detector:
            return []
        
        try:
            return self.change_detector.detect_changes(scan_result)
        except Exception as e:
            self.logger.error(f"Change detection failed: {e}")
            return []
    
    def pause_processing(self) -> None:
        """Pause batch processing."""
        self.batch_processor.pause()
    
    def resume_processing(self) -> None:
        """Resume batch processing."""
        self.batch_processor.resume()
    
    def stop_processing(self) -> None:
        """Stop batch processing."""
        self.batch_processor.stop()
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "state": self.batch_processor.get_state(),
            "progress": self.batch_processor.get_progress(),
            "statistics": self.batch_processor.get_statistics()
        }
    
    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processing result
        """
        result = {
            "path": str(file_path),
            "metadata": None,
            "errors": []
        }
        
        try:
            # Extract metadata if enabled
            if self.enable_metadata_extraction and self.metadata_extractor:
                try:
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    result["metadata"] = metadata.to_dict() if metadata else None
                except Exception as e:
                    error_msg = f"Metadata extraction failed: {e}"
                    result["errors"].append(error_msg)
                    self.logger.warning(f"{error_msg} for {file_path}")
            
            # Add basic file information
            if file_path.exists():
                stat = file_path.stat()
                result["basic_info"] = {
                    "size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "is_file": file_path.is_file(),
                    "is_dir": file_path.is_dir(),
                    "exists": True
                }
            else:
                result["basic_info"] = {"exists": False}
                result["errors"].append("File does not exist")
            
        except Exception as e:
            error_msg = f"File processing failed: {e}"
            result["errors"].append(error_msg)
            self.logger.error(f"{error_msg} for {file_path}")
        
        return result
    
    def clear_change_detection_cache(self) -> None:
        """Clear the change detection cache."""
        if self.change_detector:
            self.change_detector.clear_cache()
    
    def validate_change_detection_cache(self) -> Dict[str, Any]:
        """Validate the change detection cache."""
        if self.change_detector:
            return self.change_detector.validate_cache()
        return {"valid": False, "errors": ["Change detection not enabled"]}
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "batch_config": self.batch_config.to_dict(),
            "enable_change_detection": self.enable_change_detection,
            "enable_metadata_extraction": self.enable_metadata_extraction,
            "components": {
                "scanner": True,
                "change_detector": self.change_detector is not None,
                "metadata_extractor": self.metadata_extractor is not None,
                "batch_processor": True
            }
        }