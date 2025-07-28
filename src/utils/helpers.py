"""
Utility functions and helpers for the RAG server.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone, timedelta
import uuid
import re
from pathlib import Path


def generate_uuid() -> str:
    """Generate a unique UUID string"""
    return str(uuid.uuid4())


def generate_hash(data: Union[str, bytes]) -> str:
    """Generate MD5 hash of data"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()


def generate_sha256(data: Union[str, bytes]) -> str:
    """Generate SHA256 hash of data"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON data with fallback"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely dump data to JSON with fallback"""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default


def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters"""
    # Remove invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()


def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported"""
    supported_extensions = {'.txt', '.pdf', '.docx', '.xlsx', '.md'}
    return get_file_extension(filename) in supported_extensions


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten nested list"""
    return [item for sublist in nested_list for item in sublist]


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries with dict2 taking precedence"""
    result = dict1.copy()
    result.update(dict2)
    return result


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_text(text: str) -> str:
    """Sanitize text by removing control characters"""
    # Remove control characters except newlines and tabs
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = r'^https?://(?:[-\w.])+(?::\d+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return re.match(pattern, url) is not None


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)


def format_timestamp(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp to string"""
    return timestamp.strftime(format_str)


def parse_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse timestamp from string"""
    return datetime.strptime(timestamp_str, format_str)


def add_time_delta(timestamp: datetime, **kwargs) -> datetime:
    """Add time delta to timestamp"""
    return timestamp + timedelta(**kwargs)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    return Path(file_path).stat().st_size


def get_file_modified_time(file_path: Union[str, Path]) -> datetime:
    """Get file modification time"""
    return datetime.fromtimestamp(Path(file_path).stat().st_mtime)


def is_file_newer(file_path: Union[str, Path], reference_time: datetime) -> bool:
    """Check if file is newer than reference time"""
    return get_file_modified_time(file_path) > reference_time