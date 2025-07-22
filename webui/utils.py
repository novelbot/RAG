"""
Utility functions for RAG Server Web UI
Common helper functions and utilities
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import json
import base64
from pathlib import Path
import hashlib
import time

def format_bytes(bytes_value: Union[int, float]) -> str:
    """Format bytes to human readable string"""
    if bytes_value == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    size = abs(bytes_value)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"

def format_datetime(dt: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string"""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            return dt  # Return as-is if parsing fails
    
    if isinstance(dt, datetime):
        return dt.strftime(format_str)
    
    return str(dt)

def time_ago(dt: Union[str, datetime]) -> str:
    """Get human-readable time ago string"""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            return dt
    
    if not isinstance(dt, datetime):
        return str(dt)
    
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    diff = now - dt
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 2592000:  # 30 days
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return format_datetime(dt, "%Y-%m-%d")

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, indent: Optional[int] = None) -> str:
    """Safely serialize object to JSON string"""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except (TypeError, ValueError):
        return str(obj)

def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def get_file_type_icon(filename: str) -> str:
    """Get emoji icon for file type"""
    ext = Path(filename).suffix.lower()
    
    icons = {
        '.pdf': '📄',
        '.doc': '📝', '.docx': '📝',
        '.xls': '📊', '.xlsx': '📊',
        '.txt': '📄',
        '.md': '📝',
        '.py': '🐍',
        '.js': '📜',
        '.html': '🌐',
        '.css': '🎨',
        '.json': '📋',
        '.xml': '📋',
        '.zip': '📦',
        '.tar': '📦',
        '.gz': '📦',
        '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.gif': '🖼️',
        '.mp4': '🎥', '.avi': '🎥', '.mov': '🎥',
        '.mp3': '🎵', '.wav': '🎵',
    }
    
    return icons.get(ext, '📄')

def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate if file type is allowed"""
    ext = Path(filename).suffix.lower().lstrip('.')
    return ext in [t.lower().lstrip('.') for t in allowed_types]

def get_status_color(status: str) -> str:
    """Get color for status display"""
    colors = {
        'success': 'green',
        'completed': 'green',
        'active': 'green',
        'healthy': 'green',
        'processing': 'orange',
        'pending': 'orange',
        'warning': 'orange',
        'failed': 'red',
        'error': 'red',
        'inactive': 'red',
        'unhealthy': 'red',
        'cancelled': 'gray',
        'unknown': 'gray'
    }
    return colors.get(status.lower(), 'gray')

def get_status_emoji(status: str) -> str:
    """Get emoji for status display"""
    emojis = {
        'success': '✅',
        'completed': '✅',
        'active': '🟢',
        'healthy': '🟢',
        'processing': '🟡',
        'pending': '🟡',
        'warning': '🟠',
        'failed': '❌',
        'error': '🔴',
        'inactive': '⚪',
        'unhealthy': '🔴',
        'cancelled': '⚫',
        'unknown': '❓'
    }
    return emojis.get(status.lower(), '❓')

def show_loading_spinner(text: str = "Loading..."):
    """Show loading spinner with text"""
    return st.spinner(text)

def show_success_message(message: str, duration: int = 3):
    """Show temporary success message"""
    success_placeholder = st.empty()
    success_placeholder.success(message)
    time.sleep(duration)
    success_placeholder.empty()

def show_error_message(message: str, duration: Optional[int] = None):
    """Show error message"""
    if duration:
        error_placeholder = st.empty()
        error_placeholder.error(message)
        time.sleep(duration)
        error_placeholder.empty()
    else:
        st.error(message)

def create_download_link(data: Union[str, bytes], filename: str, mime_type: str = "text/plain") -> str:
    """Create download link for data"""
    if isinstance(data, str):
        data = data.encode()
    
    b64_data = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">Download {filename}</a>'

def paginate_data(data: List[Any], page: int, items_per_page: int = 10) -> tuple[List[Any], int, int]:
    """Paginate data and return current page items, total pages, and total items"""
    total_items = len(data)
    total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 0
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    current_page_data = data[start_idx:end_idx]
    
    return current_page_data, total_pages, total_items

def filter_data(data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter list of dictionaries based on filters"""
    filtered_data = []
    
    for item in data:
        include_item = True
        
        for key, value in filters.items():
            if key not in item:
                continue
            
            item_value = item[key]
            
            # Handle different filter types
            if isinstance(value, str) and value.lower() == "all":
                continue
            elif isinstance(value, str):
                if isinstance(item_value, str) and value.lower() not in item_value.lower():
                    include_item = False
                    break
            elif value != item_value:
                include_item = False
                break
        
        if include_item:
            filtered_data.append(item)
    
    return filtered_data

def sort_data(data: List[Dict[str, Any]], sort_key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """Sort list of dictionaries by key"""
    def sort_func(item):
        value = item.get(sort_key, '')
        # Handle different types for sorting
        if isinstance(value, str):
            return value.lower()
        return value
    
    return sorted(data, key=sort_func, reverse=reverse)

def search_data(data: List[Dict[str, Any]], search_term: str, search_fields: List[str]) -> List[Dict[str, Any]]:
    """Search through data based on search term and fields"""
    if not search_term:
        return data
    
    search_term = search_term.lower()
    filtered_data = []
    
    for item in data:
        found = False
        
        for field in search_fields:
            if field in item:
                field_value = str(item[field]).lower()
                if search_term in field_value:
                    found = True
                    break
        
        if found:
            filtered_data.append(item)
    
    return filtered_data

def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    import psutil
    
    try:
        return {
            "platform": platform.system(),
            "platform_version": platform.release(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:\\').percent
        }
    except ImportError:
        # psutil not available
        return {
            "platform": platform.system(),
            "platform_version": platform.release(),
            "python_version": platform.python_version(),
            "cpu_count": "N/A",
            "memory_total": "N/A",
            "memory_available": "N/A",
            "disk_usage": "N/A"
        }

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters"""
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename

def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength and return feedback"""
    import re
    
    checks = {
        "length": len(password) >= 8,
        "uppercase": bool(re.search(r'[A-Z]', password)),
        "lowercase": bool(re.search(r'[a-z]', password)),
        "digit": bool(re.search(r'\d', password)),
        "special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    
    score = sum(checks.values())
    
    if score <= 2:
        strength = "Weak"
    elif score <= 4:
        strength = "Medium" 
    else:
        strength = "Strong"
    
    return {
        "strength": strength,
        "score": score,
        "checks": checks,
        "is_valid": score >= 3  # At least 3 criteria must be met
    }

class SessionManager:
    """Utility class for managing session state"""
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state with default"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state"""
        st.session_state[key] = value
    
    @staticmethod
    def remove(key: str) -> None:
        """Remove key from session state"""
        if key in st.session_state:
            del st.session_state[key]
    
    @staticmethod
    def clear() -> None:
        """Clear all session state"""
        st.session_state.clear()
    
    @staticmethod
    def has(key: str) -> bool:
        """Check if key exists in session state"""
        return key in st.session_state

def create_progress_bar(progress: float, text: str = "") -> None:
    """Create progress bar with optional text"""
    if text:
        st.text(text)
    st.progress(progress)

def format_currency(amount: float, currency: str = "USD", symbol: str = "$") -> str:
    """Format amount as currency"""
    return f"{symbol}{amount:,.2f} {currency}"

def calculate_percentage(part: Union[int, float], total: Union[int, float]) -> float:
    """Calculate percentage safely"""
    if total == 0:
        return 0.0
    return (part / total) * 100

def create_metric_card(title: str, value: str, delta: Optional[str] = None) -> None:
    """Create a metric card display"""
    st.metric(label=title, value=value, delta=delta)