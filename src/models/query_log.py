"""
Query logging models for analytics and auditing.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from src.core.user_database import UserBase as Base


class QueryStatus(Enum):
    """Query execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class QueryType(Enum):
    """Type of query performed"""
    SEARCH = "search"
    RAG = "rag"
    BATCH_SEARCH = "batch_search"
    SIMILARITY = "similarity"


class QueryLog(Base):
    """Query log model for storing query analytics and audit information"""
    
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True)
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_type = Column(SQLEnum(QueryType), nullable=False, default=QueryType.SEARCH)
    query_hash = Column(String(64), nullable=True, index=True)  # For detecting duplicate queries
    
    # User context
    user_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # Execution details
    status = Column(SQLEnum(QueryStatus), nullable=False, default=QueryStatus.SUCCESS)
    error_message = Column(Text, nullable=True)
    
    # Performance metrics
    response_time_ms = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    embedding_time_ms = Column(Float, nullable=True)
    search_time_ms = Column(Float, nullable=True)
    llm_time_ms = Column(Float, nullable=True)
    
    # Token usage
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    
    # Search parameters
    search_limit = Column(Integer, nullable=True)
    search_offset = Column(Integer, nullable=True)
    search_filter = Column(JSON, nullable=True)
    
    # Results metadata
    results_count = Column(Integer, nullable=True)
    max_similarity_score = Column(Float, nullable=True)
    min_similarity_score = Column(Float, nullable=True)
    avg_similarity_score = Column(Float, nullable=True)
    
    # LLM details
    model_used = Column(String(100), nullable=True)
    llm_provider = Column(String(50), nullable=True)
    finish_reason = Column(String(50), nullable=True)
    
    # Additional metadata
    request_metadata = Column(JSON, nullable=True)
    response_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"QueryLog(id={self.id}, user_id='{self.user_id}', query_type='{self.query_type.value}', status='{self.status.value}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query log to dictionary format"""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "query_hash": self.query_hash,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "error_message": self.error_message,
            "response_time_ms": self.response_time_ms,
            "processing_time_ms": self.processing_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "search_time_ms": self.search_time_ms,
            "llm_time_ms": self.llm_time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "search_limit": self.search_limit,
            "search_offset": self.search_offset,
            "search_filter": self.search_filter,
            "results_count": self.results_count,
            "max_similarity_score": self.max_similarity_score,
            "min_similarity_score": self.min_similarity_score,
            "avg_similarity_score": self.avg_similarity_score,
            "model_used": self.model_used,
            "llm_provider": self.llm_provider,
            "finish_reason": self.finish_reason,
            "request_metadata": self.request_metadata,
            "response_metadata": self.response_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def create_query_log(cls, query_text: str, query_type: QueryType, 
                        user_id: str, **kwargs) -> "QueryLog":
        """Create a new query log entry"""
        return cls(
            query_text=query_text,
            query_type=query_type,
            user_id=user_id,
            **kwargs
        )
    
    def mark_success(self, response_time_ms: float = None, 
                    results_count: int = None, **kwargs):
        """Mark query as successful"""
        self.status = QueryStatus.SUCCESS
        if response_time_ms is not None:
            self.response_time_ms = response_time_ms
        if results_count is not None:
            self.results_count = results_count
        
        # Update any additional metrics
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def mark_failed(self, error_message: str, response_time_ms: float = None):
        """Mark query as failed"""
        self.status = QueryStatus.FAILED
        self.error_message = error_message
        if response_time_ms is not None:
            self.response_time_ms = response_time_ms