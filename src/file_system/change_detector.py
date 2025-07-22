"""
File Change Detection System.

This module provides functionality to detect file system changes using modification
timestamps and optional file hashing for integrity verification.
"""

import json
import hashlib
import tempfile
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Union, Any

from src.core.logging import LoggerMixin
from .exceptions import FileChangeDetectionError, FileSystemError
from .scanner import ScanResult


class ChangeType(Enum):
    """Types of file changes that can be detected."""
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class FileState:
    """Represents the state of a file at a point in time."""
    path: str
    mtime: float
    size: int
    hash: Optional[str] = None
    last_scanned: str = ""
    
    def __post_init__(self):
        """Set last_scanned to current time if not provided."""
        if not self.last_scanned:
            self.last_scanned = datetime.now().isoformat()
    
    @classmethod
    def from_path(cls, file_path: Path, calculate_hash: bool = False, 
                  hash_algorithm: str = "md5") -> "FileState":
        """Create FileState from a file path."""
        try:
            stat = file_path.stat()
            hash_value = None
            
            if calculate_hash:
                hash_value = FileChangeDetector._calculate_file_hash(
                    file_path, hash_algorithm
                )
            
            return cls(
                path=str(file_path),
                mtime=stat.st_mtime,
                size=stat.st_size,
                hash=hash_value
            )
        except Exception as e:
            raise FileChangeDetectionError(f"Failed to get file state for {file_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        """Create from dictionary (for JSON deserialization)."""
        return cls(**data)


@dataclass
class ChangeRecord:
    """Represents a detected file change."""
    path: Path
    change_type: ChangeType
    previous_state: Optional[FileState] = None
    current_state: Optional[FileState] = None
    
    def __post_init__(self):
        """Convert string path to Path object."""
        if isinstance(self.path, str):
            self.path = Path(self.path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "change_type": self.change_type.value,
            "previous_state": self.previous_state.to_dict() if self.previous_state else None,
            "current_state": self.current_state.to_dict() if self.current_state else None
        }


class FileChangeDetector(LoggerMixin):
    """
    File Change Detection System.
    
    Detects file system changes using modification timestamps and optional
    file hashing. Maintains a cache of file states to enable incremental
    scanning and change detection.
    """
    
    def __init__(self, 
                 cache_file: Optional[Union[str, Path]] = None,
                 use_hashing: bool = False,
                 hash_algorithm: str = "md5",
                 mtime_threshold: float = 1.0):
        """
        Initialize the file change detector.
        
        Args:
            cache_file: Path to cache file for storing file states
            use_hashing: Whether to use file hashing for integrity checks
            hash_algorithm: Hash algorithm to use ('md5', 'sha256', 'sha1')
            mtime_threshold: Minimum time difference (seconds) to consider as modification
        """
        self.cache_file = Path(cache_file) if cache_file else Path.cwd() / ".file_states.json"
        self.use_hashing = use_hashing
        self.hash_algorithm = hash_algorithm.lower()
        self.mtime_threshold = mtime_threshold
        
        # Validate hash algorithm
        if self.hash_algorithm not in ['md5', 'sha1', 'sha256', 'sha512']:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
        
        self.logger.info(f"File Change Detector initialized with cache: {self.cache_file}")
        
        # Statistics
        self.stats = {
            "files_scanned": 0,
            "new_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "unchanged_files": 0,
            "hash_calculations": 0,
            "cache_hits": 0
        }
    
    def detect_changes(self, scan_result: ScanResult) -> List[ChangeRecord]:
        """
        Detect file changes from a scan result.
        
        Args:
            scan_result: Result from DirectoryScanner
            
        Returns:
            List of detected changes
            
        Raises:
            FileChangeDetectionError: If change detection fails
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting change detection for {len(scan_result.files)} files")
            
            # Reset statistics
            self.stats = {key: 0 for key in self.stats}
            
            # Load previous file states
            previous_states = self._load_cache()
            current_states = {}
            changes = []
            
            # Track current file paths for deletion detection
            current_paths = {str(file_path) for file_path in scan_result.files}
            
            # Process current files
            for file_path in scan_result.files:
                try:
                    current_state = self._get_file_state(file_path)
                    current_states[str(file_path)] = current_state
                    
                    path_str = str(file_path)
                    previous_state = previous_states.get(path_str)
                    
                    change_type = self._determine_change_type(previous_state, current_state)
                    
                    if change_type != ChangeType.UNCHANGED:
                        changes.append(ChangeRecord(
                            path=file_path,
                            change_type=change_type,
                            previous_state=previous_state,
                            current_state=current_state
                        ))
                    
                    # Update statistics
                    self.stats["files_scanned"] += 1
                    if change_type == ChangeType.NEW:
                        self.stats["new_files"] += 1
                    elif change_type == ChangeType.MODIFIED:
                        self.stats["modified_files"] += 1
                    elif change_type == ChangeType.UNCHANGED:
                        self.stats["unchanged_files"] += 1
                        self.stats["cache_hits"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process file {file_path}: {e}")
                    scan_result.errors.append({
                        "path": str(file_path),
                        "error": str(e),
                        "type": "change_detection"
                    })
            
            # Detect deleted files
            for path_str, previous_state in previous_states.items():
                if path_str not in current_paths:
                    changes.append(ChangeRecord(
                        path=Path(path_str),
                        change_type=ChangeType.DELETED,
                        previous_state=previous_state,
                        current_state=None
                    ))
                    self.stats["deleted_files"] += 1
            
            # Save updated cache
            self._save_cache(current_states)
            
            # Log results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(
                f"Change detection completed in {duration:.2f}s: "
                f"{self.stats['new_files']} new, "
                f"{self.stats['modified_files']} modified, "
                f"{self.stats['deleted_files']} deleted, "
                f"{self.stats['unchanged_files']} unchanged"
            )
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Change detection failed: {e}")
            raise FileChangeDetectionError(f"Failed to detect changes: {e}")
    
    def _get_file_state(self, file_path: Path) -> FileState:
        """Get current state of a file."""
        return FileState.from_path(
            file_path, 
            calculate_hash=self.use_hashing,
            hash_algorithm=self.hash_algorithm
        )
    
    def _determine_change_type(self, previous: Optional[FileState], 
                              current: FileState) -> ChangeType:
        """Determine the type of change between two file states."""
        if previous is None:
            return ChangeType.NEW
        
        # Check modification time (with threshold)
        if abs(current.mtime - previous.mtime) > self.mtime_threshold:
            return ChangeType.MODIFIED
        
        # Check file size
        if current.size != previous.size:
            return ChangeType.MODIFIED
        
        # Check hash if available
        if self.use_hashing and previous.hash and current.hash:
            if current.hash != previous.hash:
                return ChangeType.MODIFIED
        
        return ChangeType.UNCHANGED
    
    def _load_cache(self) -> Dict[str, FileState]:
        """Load file states from cache."""
        if not self.cache_file.exists():
            self.logger.debug(f"Cache file does not exist: {self.cache_file}")
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            states = {}
            for path_str, state_data in data.items():
                try:
                    states[path_str] = FileState.from_dict(state_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load state for {path_str}: {e}")
            
            self.logger.debug(f"Loaded {len(states)} file states from cache")
            return states
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Cache file corrupted: {e}")
            # Backup corrupted cache and start fresh
            backup_path = self.cache_file.with_suffix('.json.backup')
            self.cache_file.rename(backup_path)
            self.logger.info(f"Corrupted cache backed up to {backup_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self, file_states: Dict[str, FileState]) -> None:
        """Save file states to cache with atomic write."""
        try:
            # Prepare data for JSON serialization
            data = {path_str: state.to_dict() for path_str, state in file_states.items()}
            
            # Atomic write using temporary file
            cache_dir = self.cache_file.parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=cache_dir,
                delete=False,
                suffix='.tmp',
                encoding='utf-8'
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2, ensure_ascii=False)
                tmp_path = Path(tmp_file.name)
            
            # Atomic move
            tmp_path.replace(self.cache_file)
            
            self.logger.debug(f"Saved {len(file_states)} file states to cache")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            # Clean up temporary file if it exists
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise FileChangeDetectionError(f"Failed to save cache: {e}")
    
    @staticmethod
    def _calculate_file_hash(file_path: Path, algorithm: str = "md5") -> str:
        """Calculate hash of a file using specified algorithm."""
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "sha512":
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files efficiently
                chunk_size = 8192
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            raise FileChangeDetectionError(f"Failed to calculate hash for {file_path}: {e}")
    
    def get_changed_files(self, scan_result: ScanResult, 
                         change_types: Optional[Set[ChangeType]] = None) -> List[Path]:
        """
        Get list of files that have changed.
        
        Args:
            scan_result: Scan result to check
            change_types: Types of changes to include (default: NEW, MODIFIED)
            
        Returns:
            List of changed file paths
        """
        if change_types is None:
            change_types = {ChangeType.NEW, ChangeType.MODIFIED}
        
        changes = self.detect_changes(scan_result)
        return [
            change.path for change in changes 
            if change.change_type in change_types
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.stats.copy()
    
    def clear_cache(self) -> None:
        """Clear the file states cache."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                self.logger.info(f"Cache file cleared: {self.cache_file}")
            else:
                self.logger.info("Cache file does not exist")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise FileChangeDetectionError(f"Failed to clear cache: {e}")
    
    def validate_cache(self) -> Dict[str, Any]:
        """
        Validate the cache file and return validation results.
        
        Returns:
            Validation report
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "cache_exists": False,
            "cache_size": 0,
            "file_count": 0,
            "oldest_entry": None,
            "newest_entry": None
        }
        
        try:
            if not self.cache_file.exists():
                result["errors"].append("Cache file does not exist")
                result["valid"] = False
                return result
            
            result["cache_exists"] = True
            result["cache_size"] = self.cache_file.stat().st_size
            
            # Load and validate cache contents
            file_states = self._load_cache()
            result["file_count"] = len(file_states)
            
            if not file_states:
                result["warnings"].append("Cache is empty")
                return result
            
            # Find oldest and newest entries
            scan_times = []
            for state in file_states.values():
                try:
                    if state.last_scanned:
                        scan_time = datetime.fromisoformat(state.last_scanned)
                        scan_times.append(scan_time)
                except ValueError:
                    result["warnings"].append(f"Invalid timestamp for {state.path}")
            
            if scan_times:
                result["oldest_entry"] = min(scan_times).isoformat()
                result["newest_entry"] = max(scan_times).isoformat()
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))
        
        return result