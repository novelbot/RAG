"""
Base models and schemas for the RAG server application.
"""

from typing import Optional, Any, Dict, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp fields"""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[Any] = None


class PaginationMixin(BaseModel):
    """Mixin for paginated responses"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    total: int = 0
    total_pages: int = 0


class PaginatedResponse(BaseResponse):
    """Paginated response model"""
    pagination: PaginationMixin


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    services: Dict[str, str] = Field(default_factory=dict)


class DatabaseStatus(str, Enum):
    """Database connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ServiceStatus(BaseModel):
    """Service status model"""
    name: str
    status: DatabaseStatus
    message: Optional[str] = None
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RAGMode(str, Enum):
    """RAG operation modes"""
    SINGLE = "single"
    MULTI = "multi"


class DataSourceType(str, Enum):
    """Data source types"""
    RDB = "rdb"
    FILE = "file"


class FileType(str, Enum):
    """Supported file types"""
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    MD = "md"


class LLMProvider(str, Enum):
    """LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    """Embedding providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class QueryRequest(BaseModel):
    """RAG query request model"""
    query: str = Field(..., min_length=1, max_length=1000)
    mode: RAGMode = RAGMode.SINGLE
    k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """RAG query response model"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float