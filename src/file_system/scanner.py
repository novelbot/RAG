"""
Directory Scanner for Recursive File Discovery.

This module provides recursive directory scanning functionality that can identify
supported file types and apply filtering criteria.
"""

import os
import re
import fnmatch
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Callable, Iterator
from datetime import datetime
from dataclasses import dataclass, field

from src.core.logging import LoggerMixin
from .exceptions import DirectoryScanError, FileSystemError
from .parsers import ParserFactory


@dataclass
class ScanResult:
    """Result of a directory scan operation."""
    files: List[Path] = field(default_factory=list)
    directories: List[Path] = field(default_factory=list)
    skipped_files: List[Path] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    scan_time: float = 0.0
    total_size: int = 0
    file_count_by_type: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert scan result to dictionary."""
        return {
            "files": [str(f) for f in self.files],
            "directories": [str(d) for d in self.directories],
            "skipped_files": [str(f) for f in self.skipped_files],
            "errors": self.errors,
            "scan_time": self.scan_time,
            "total_size": self.total_size,
            "file_count_by_type": self.file_count_by_type,
            "total_files": len(self.files),
            "total_directories": len(self.directories),
            "total_skipped": len(self.skipped_files),
            "total_errors": len(self.errors)
        }


@dataclass
class ScanOptions:
    """Options for directory scanning."""
    max_depth: Optional[int] = None
    follow_symlinks: bool = False
    include_hidden: bool = False
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    supported_extensions_only: bool = True
    max_file_size: Optional[int] = None  # in bytes
    min_file_size: Optional[int] = None  # in bytes
    file_filter: Optional[Callable[[Path], bool]] = None
    directory_filter: Optional[Callable[[Path], bool]] = None
    
    def matches_include_pattern(self, path: Path) -> bool:
        """Check if path matches any include pattern."""
        if not self.include_patterns:
            return True
        
        path_str = str(path)
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False
    
    def matches_exclude_pattern(self, path: Path) -> bool:
        """Check if path matches any exclude pattern."""
        if not self.exclude_patterns:
            return False
        
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False


class DirectoryScanner(LoggerMixin):
    """
    Recursive Directory Scanner.
    
    Provides functionality to recursively scan directories and identify
    supported file types with flexible filtering options.
    """
    
    def __init__(self, parser_factory: Optional[ParserFactory] = None):
        """
        Initialize the directory scanner.
        
        Args:
            parser_factory: Factory for file parsers (optional)
        """
        self.parser_factory = parser_factory or ParserFactory()
        self.supported_extensions = set(self.parser_factory.get_supported_extensions())
        
        # Default exclude patterns
        self.default_exclude_patterns = [
            '*.tmp', '*.temp', '*.log', '*.cache',
            '*.pyc', '*.pyo', '*.pyd', '__pycache__',
            '.git', '.svn', '.hg', '.bzr',
            'node_modules', '.DS_Store', 'Thumbs.db',
            '*.lock', '*.pid', '*.sock'
        ]
        
        self.logger.info("Directory Scanner initialized successfully")
    
    def scan(self, 
             root_path: Union[str, Path], 
             options: Optional[ScanOptions] = None) -> ScanResult:
        """
        Scan a directory recursively.
        
        Args:
            root_path: Root directory to scan
            options: Scan options
            
        Returns:
            Scan result containing found files and metadata
            
        Raises:
            DirectoryScanError: If scanning fails
        """
        try:
            start_time = datetime.now()
            root_path = Path(root_path)
            options = options or ScanOptions()
            
            # Validate root path
            if not root_path.exists():
                raise DirectoryScanError(f"Root path does not exist: {root_path}")
            
            if not root_path.is_dir():
                raise DirectoryScanError(f"Root path is not a directory: {root_path}")
            
            # Initialize result
            result = ScanResult()
            
            # Start scanning
            self._scan_recursive(root_path, options, result, depth=0)
            
            # Calculate scan time
            end_time = datetime.now()
            result.scan_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Directory scan completed: {len(result.files)} files found in {result.scan_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Directory scan failed: {e}")
            raise DirectoryScanError(f"Failed to scan directory: {e}")
    
    def _scan_recursive(self, 
                       path: Path, 
                       options: ScanOptions, 
                       result: ScanResult, 
                       depth: int = 0) -> None:
        """
        Recursively scan a directory.
        
        Args:
            path: Current directory path
            options: Scan options
            result: Result object to populate
            depth: Current recursion depth
        """
        try:
            # Check depth limit
            if options.max_depth is not None and depth > options.max_depth:
                return
            
            # Apply directory filter
            if options.directory_filter and not options.directory_filter(path):
                return
            
            # Get directory contents
            try:
                entries = list(path.iterdir())
            except (PermissionError, OSError) as e:
                result.errors.append({
                    "path": str(path),
                    "error": f"Permission denied or OS error: {e}",
                    "type": "directory_access"
                })
                return
            
            # Process entries
            for entry in entries:
                try:
                    # Skip hidden files/directories if not included
                    if not options.include_hidden and entry.name.startswith('.'):
                        continue
                    
                    # Handle symlinks
                    if entry.is_symlink():
                        if not options.follow_symlinks:
                            continue
                        
                        # Check for circular references
                        try:
                            real_path = entry.resolve()
                            if real_path == path or path in real_path.parents:
                                self.logger.warning(f"Circular symlink detected: {entry}")
                                continue
                        except (OSError, RuntimeError):
                            result.errors.append({
                                "path": str(entry),
                                "error": "Failed to resolve symlink",
                                "type": "symlink_resolution"
                            })
                            continue
                    
                    if entry.is_dir():
                        # Process directory
                        result.directories.append(entry)
                        
                        # Apply exclude patterns
                        if options.matches_exclude_pattern(entry):
                            continue
                        
                        # Recurse into directory
                        self._scan_recursive(entry, options, result, depth + 1)
                    
                    elif entry.is_file():
                        # Process file
                        self._process_file(entry, options, result)
                
                except Exception as e:
                    result.errors.append({
                        "path": str(entry),
                        "error": str(e),
                        "type": "entry_processing"
                    })
                    continue
        
        except Exception as e:
            result.errors.append({
                "path": str(path),
                "error": str(e),
                "type": "directory_scan"
            })
    
    def _process_file(self, file_path: Path, options: ScanOptions, result: ScanResult) -> None:
        """
        Process a single file.
        
        Args:
            file_path: Path to the file
            options: Scan options
            result: Result object to populate
        """
        try:
            # Apply exclude patterns
            if options.matches_exclude_pattern(file_path):
                result.skipped_files.append(file_path)
                return
            
            # Apply include patterns
            if not options.matches_include_pattern(file_path):
                result.skipped_files.append(file_path)
                return
            
            # Check file extension if required
            if options.supported_extensions_only:
                if not self.parser_factory.is_supported(file_path):
                    result.skipped_files.append(file_path)
                    return
            
            # Get file stats
            try:
                stat = file_path.stat()
                file_size = stat.st_size
            except (OSError, PermissionError) as e:
                result.errors.append({
                    "path": str(file_path),
                    "error": f"Failed to get file stats: {e}",
                    "type": "file_stats"
                })
                return
            
            # Check file size limits
            if options.max_file_size is not None and file_size > options.max_file_size:
                result.skipped_files.append(file_path)
                return
            
            if options.min_file_size is not None and file_size < options.min_file_size:
                result.skipped_files.append(file_path)
                return
            
            # Apply custom file filter
            if options.file_filter and not options.file_filter(file_path):
                result.skipped_files.append(file_path)
                return
            
            # File passed all filters
            result.files.append(file_path)
            result.total_size += file_size
            
            # Track file type counts
            extension = file_path.suffix.lower()
            result.file_count_by_type[extension] = result.file_count_by_type.get(extension, 0) + 1
            
        except Exception as e:
            result.errors.append({
                "path": str(file_path),
                "error": str(e),
                "type": "file_processing"
            })
    
    def scan_with_filter(self, 
                        root_path: Union[str, Path],
                        extensions: Optional[List[str]] = None,
                        max_depth: Optional[int] = None,
                        exclude_patterns: Optional[List[str]] = None) -> ScanResult:
        """
        Convenience method for scanning with common filters.
        
        Args:
            root_path: Root directory to scan
            extensions: List of extensions to include (e.g., ['.txt', '.pdf'])
            max_depth: Maximum recursion depth
            exclude_patterns: Patterns to exclude
            
        Returns:
            Scan result
        """
        options = ScanOptions(
            max_depth=max_depth,
            exclude_patterns=exclude_patterns or self.default_exclude_patterns
        )
        
        # Set supported extensions if provided
        if extensions:
            options.supported_extensions_only = False
            options.include_patterns = [f"*{ext}" for ext in extensions]
        
        return self.scan(root_path, options)
    
    def find_files_by_pattern(self, 
                             root_path: Union[str, Path],
                             pattern: str,
                             max_depth: Optional[int] = None) -> List[Path]:
        """
        Find files matching a specific pattern.
        
        Args:
            root_path: Root directory to search
            pattern: File name pattern (supports wildcards)
            max_depth: Maximum recursion depth
            
        Returns:
            List of matching files
        """
        options = ScanOptions(
            max_depth=max_depth,
            include_patterns=[pattern],
            supported_extensions_only=False
        )
        
        result = self.scan(root_path, options)
        return result.files
    
    def get_directory_stats(self, root_path: Union[str, Path]) -> Dict[str, any]:
        """
        Get comprehensive statistics about a directory.
        
        Args:
            root_path: Root directory to analyze
            
        Returns:
            Dictionary containing directory statistics
        """
        options = ScanOptions(supported_extensions_only=False)
        result = self.scan(root_path, options)
        
        return {
            "total_files": len(result.files),
            "total_directories": len(result.directories),
            "total_size": result.total_size,
            "file_count_by_type": result.file_count_by_type,
            "supported_files": len([f for f in result.files if self.parser_factory.is_supported(f)]),
            "scan_time": result.scan_time,
            "errors": len(result.errors),
            "skipped_files": len(result.skipped_files)
        }
    
    def validate_directory(self, path: Union[str, Path]) -> Dict[str, any]:
        """
        Validate a directory for scanning.
        
        Args:
            path: Directory path to validate
            
        Returns:
            Validation result
        """
        path = Path(path)
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "accessible": False,
            "readable": False,
            "size_estimate": 0,
            "file_count_estimate": 0
        }
        
        try:
            # Check if path exists
            if not path.exists():
                result["valid"] = False
                result["errors"].append(f"Path does not exist: {path}")
                return result
            
            # Check if it's a directory
            if not path.is_dir():
                result["valid"] = False
                result["errors"].append(f"Path is not a directory: {path}")
                return result
            
            # Check accessibility
            try:
                list(path.iterdir())
                result["accessible"] = True
                result["readable"] = True
            except PermissionError:
                result["errors"].append(f"Permission denied: {path}")
                return result
            except OSError as e:
                result["errors"].append(f"OS error accessing directory: {e}")
                return result
            
            # Quick size and count estimate
            try:
                quick_scan = self.scan_with_filter(path, max_depth=2)
                result["size_estimate"] = quick_scan.total_size
                result["file_count_estimate"] = len(quick_scan.files)
                
                if quick_scan.errors:
                    result["warnings"].extend([error["error"] for error in quick_scan.errors])
                
            except Exception as e:
                result["warnings"].append(f"Failed to estimate directory size: {e}")
        
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation failed: {e}")
        
        return result
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(self.supported_extensions)
    
    def create_scan_options(self, **kwargs) -> ScanOptions:
        """
        Create scan options with defaults.
        
        Args:
            **kwargs: Option parameters
            
        Returns:
            ScanOptions instance
        """
        # Set default exclude patterns
        if 'exclude_patterns' not in kwargs:
            kwargs['exclude_patterns'] = self.default_exclude_patterns
        
        return ScanOptions(**kwargs)