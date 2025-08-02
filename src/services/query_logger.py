"""
Query logging service for analytics and audit trails.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.orm import Session
from fastapi import Request

from ..core.user_database import get_user_session
from ..models.query_log import QueryLog, QueryStatus, QueryType
from ..core.logging import LoggerMixin


@dataclass
class QueryMetrics:
    """Container for query performance metrics"""
    response_time_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    embedding_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None
    llm_time_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    results_count: Optional[int] = None
    max_similarity_score: Optional[float] = None
    min_similarity_score: Optional[float] = None
    avg_similarity_score: Optional[float] = None


@dataclass
class QueryContext:
    """Container for query context information"""
    user_id: str
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    request_metadata: Optional[Dict[str, Any]] = None


class QueryLogger(LoggerMixin):
    """
    Service for logging query activities with structured data and performance metrics.
    
    Provides comprehensive logging for RAG queries including:
    - Performance metrics (response time, token usage)
    - User context and audit information
    - Search parameters and results
    - Error tracking and debugging information
    """
    
    def __init__(self):
        """Initialize the query logger"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        
    def _generate_query_hash(self, query_text: str) -> str:
        """Generate a hash for the query to detect duplicates"""
        return hashlib.sha256(query_text.encode('utf-8')).hexdigest()[:16]
    
    def _extract_request_context(self, request: Optional[Request] = None) -> Dict[str, Any]:
        """Extract context information from the request"""
        if not request:
            return {}
            
        return {
            "user_agent": request.headers.get("user-agent"),
            "ip_address": request.client.host if request.client else None,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers) if hasattr(request, 'headers') else {}
        }
    
    async def create_query_log(self, 
                              query_text: str,
                              query_type: QueryType,
                              context: QueryContext,
                              search_params: Optional[Dict[str, Any]] = None,
                              request: Optional[Request] = None) -> QueryLog:
        """
        Create a new query log entry.
        
        Args:
            query_text: The search query text
            query_type: Type of query (search, rag, etc.)
            context: User context information
            search_params: Search parameters used
            request: FastAPI request object for additional context
            
        Returns:
            QueryLog: Created query log entry
        """
        db = get_user_session()
        try:
            # Generate query hash for duplicate detection
            query_hash = self._generate_query_hash(query_text)
            
            # Extract request context
            request_context = self._extract_request_context(request)
            
            # Merge request metadata
            request_metadata = {**(context.request_metadata or {}), **request_context}
            
            # Create query log entry
            query_log = QueryLog.create_query_log(
                query_text=query_text,
                query_type=query_type,
                user_id=context.user_id,
                query_hash=query_hash,
                session_id=context.session_id,
                user_agent=context.user_agent or request_context.get("user_agent"),
                ip_address=context.ip_address or request_context.get("ip_address"),
                search_limit=search_params.get("limit") if search_params else None,
                search_offset=search_params.get("offset") if search_params else None,
                search_filter=search_params.get("filter") if search_params else None,
                request_metadata=request_metadata
            )
            
            db.add(query_log)
            db.commit()
            db.refresh(query_log)
            
            self._logger.info(f"Created query log entry {query_log.id} for user {context.user_id}")
            return query_log
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to create query log: {e}")
            raise
        finally:
            db.close()
    
    async def update_query_metrics(self, 
                                  query_log_id: int,
                                  metrics: QueryMetrics,
                                  model_info: Optional[Dict[str, str]] = None,
                                  response_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update query log with performance metrics and results.
        
        Args:
            query_log_id: ID of the query log entry
            metrics: Performance metrics to record
            model_info: Information about the model used
            response_metadata: Additional response metadata
        """
        db = get_user_session()
        try:
            query_log = db.query(QueryLog).filter(QueryLog.id == query_log_id).first()
            if not query_log:
                self._logger.warning(f"Query log {query_log_id} not found for metrics update")
                return
                
            # Update performance metrics
            if metrics.response_time_ms is not None:
                query_log.response_time_ms = metrics.response_time_ms
            if metrics.processing_time_ms is not None:
                query_log.processing_time_ms = metrics.processing_time_ms
            if metrics.embedding_time_ms is not None:
                query_log.embedding_time_ms = metrics.embedding_time_ms
            if metrics.search_time_ms is not None:
                query_log.search_time_ms = metrics.search_time_ms
            if metrics.llm_time_ms is not None:
                query_log.llm_time_ms = metrics.llm_time_ms
                
            # Update token usage
            if metrics.prompt_tokens is not None:
                query_log.prompt_tokens = metrics.prompt_tokens
            if metrics.completion_tokens is not None:
                query_log.completion_tokens = metrics.completion_tokens
            if metrics.total_tokens is not None:
                query_log.total_tokens = metrics.total_tokens
                
            # Update result metrics
            if metrics.results_count is not None:
                query_log.results_count = metrics.results_count
            if metrics.max_similarity_score is not None:
                query_log.max_similarity_score = metrics.max_similarity_score
            if metrics.min_similarity_score is not None:
                query_log.min_similarity_score = metrics.min_similarity_score
            if metrics.avg_similarity_score is not None:
                query_log.avg_similarity_score = metrics.avg_similarity_score
                
            # Update model information
            if model_info:
                query_log.model_used = model_info.get("model_used")
                query_log.llm_provider = model_info.get("llm_provider")
                query_log.finish_reason = model_info.get("finish_reason")
                
            # Update response metadata
            if response_metadata:
                query_log.response_metadata = response_metadata
                
            db.commit()
            self._logger.debug(f"Updated metrics for query log {query_log_id}")
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to update query metrics: {e}")
            raise
        finally:
            db.close()
    
    async def mark_query_success(self, 
                                query_log_id: int,
                                metrics: Optional[QueryMetrics] = None) -> None:
        """
        Mark a query as successful and optionally update metrics.
        
        Args:
            query_log_id: ID of the query log entry
            metrics: Optional performance metrics
        """
        db = get_user_session()
        try:
            query_log = db.query(QueryLog).filter(QueryLog.id == query_log_id).first()
            if not query_log:
                self._logger.warning(f"Query log {query_log_id} not found for success marking")
                return
                
            query_log.mark_success(
                response_time_ms=metrics.response_time_ms if metrics else None,
                results_count=metrics.results_count if metrics else None
            )
            
            db.commit()
            self._logger.debug(f"Marked query log {query_log_id} as successful")
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to mark query success: {e}")
            raise
        finally:
            db.close()
    
    async def mark_query_failed(self, 
                               query_log_id: int,
                               error_message: str,
                               response_time_ms: Optional[float] = None) -> None:
        """
        Mark a query as failed with error information.
        
        Args:
            query_log_id: ID of the query log entry
            error_message: Error message describing the failure
            response_time_ms: Optional response time
        """
        db = get_user_session()
        try:
            query_log = db.query(QueryLog).filter(QueryLog.id == query_log_id).first()
            if not query_log:
                self._logger.warning(f"Query log {query_log_id} not found for failure marking")
                return
                
            query_log.mark_failed(error_message, response_time_ms)
            
            db.commit()
            self._logger.info(f"Marked query log {query_log_id} as failed: {error_message}")
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to mark query failure: {e}")
            raise
        finally:
            db.close()
    
    async def get_user_query_history(self, 
                                    user_id: str,
                                    limit: int = 100,
                                    offset: int = 0) -> List[QueryLog]:
        """
        Get query history for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of query log entries
        """
        db = get_user_session()
        try:
            query_logs = db.query(QueryLog)\
                          .filter(QueryLog.user_id == user_id)\
                          .order_by(QueryLog.created_at.desc())\
                          .offset(offset)\
                          .limit(limit)\
                          .all()
            
            return query_logs
            
        except Exception as e:
            self._logger.error(f"Failed to get user query history: {e}")
            raise
        finally:
            db.close()
    
    @asynccontextmanager
    async def log_query_context(self, 
                               query_text: str,
                               query_type: QueryType,
                               context: QueryContext,
                               search_params: Optional[Dict[str, Any]] = None,
                               request: Optional[Request] = None):
        """
        Context manager for query logging with automatic success/failure handling.
        
        Usage:
            async with query_logger.log_query_context(query, QueryType.RAG, context) as log_id:
                # Perform query operations
                result = await process_query()
                # Context manager automatically marks as success
        """
        start_time = time.time()
        query_log = await self.create_query_log(
            query_text, query_type, context, search_params, request
        )
        
        try:
            yield query_log.id
            
            # Automatically mark as success if no exception
            response_time_ms = (time.time() - start_time) * 1000
            await self.mark_query_success(
                query_log.id, 
                QueryMetrics(response_time_ms=response_time_ms)
            )
            
        except Exception as e:
            # Automatically mark as failed if exception occurs
            response_time_ms = (time.time() - start_time) * 1000
            await self.mark_query_failed(
                query_log.id, 
                str(e), 
                response_time_ms
            )
            raise


# Global query logger instance
query_logger = QueryLogger()