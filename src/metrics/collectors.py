"""
Data collection middleware and event handlers for metrics system.
"""

import time
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import logging
from contextlib import asynccontextmanager

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.security import HTTPBearer
from starlette.types import ASGIApp

from .database import get_metrics_db

logger = logging.getLogger(__name__)


class MetricsCollectionMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to collect request metrics and user activity."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request."""
        start_time = time.time()
        
        # Extract user information
        user_id = await self._extract_user_id(request)
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Update user activity if authenticated
        if user_id:
            try:
                metrics_db = await get_metrics_db()
                await metrics_db.update_user_activity(user_id)
            except Exception as e:
                logger.error(f"Failed to update user activity: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log request metrics for API endpoints
        if request.url.path.startswith("/api/"):
            await self._log_api_request(
                request, response, response_time_ms, user_id, ip_address, user_agent
            )
        
        # Add response time header
        response.headers["X-Response-Time"] = str(response_time_ms)
        
        return response
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from JWT token or session."""
        try:
            # Try to get from authorization header
            authorization = request.headers.get("authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split("Bearer ")[1]
                
                # Import auth manager to verify token
                from ..auth.sqlite_auth import auth_manager
                session_data = auth_manager.verify_token(token)
                
                if session_data:
                    return str(session_data.get('user_id'))
                
                # Fallback for mock tokens during development
                if token == "demo_access_token":
                    return "demo_user_id"
                elif token == "admin_token":
                    return "admin_user_id"
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract user ID: {e}")
            return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    async def _log_api_request(
        self,
        request: Request,
        response: Response,
        response_time_ms: int,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str
    ) -> None:
        """Log API request metrics."""
        try:
            metrics_db = await get_metrics_db()
            
            # Special handling for query endpoints
            if "/query" in request.url.path or "/chat" in request.url.path:
                # Try to extract query text from request body if available
                query_text = "API Query"  # Default placeholder
                success = 200 <= response.status_code < 400
                
                # For more detailed query logging, you could store request body in middleware
                # and extract actual query text here
                
                await metrics_db.log_query(
                    user_id=user_id or "anonymous",
                    query_text=query_text,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=None if success else f"HTTP {response.status_code}",
                    result_count=1 if success else 0,  # Assume 1 result for successful queries
                    tokens_used=0,   # Would calculate based on LLM usage
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                # Also log as system event for successful queries
                if success and user_id:
                    await metrics_db.log_system_event(
                        event_type="query_performed",
                        user_id=user_id,
                        description=f"Performed API query",
                        details={
                            "endpoint": str(request.url.path),
                            "response_time_ms": response_time_ms,
                            "ip_address": ip_address
                        }
                    )
                
        except Exception as e:
            logger.error(f"Failed to log API request metrics: {e}")


class DocumentEventCollector:
    """Collector for document upload/delete/update events."""
    
    def __init__(self):
        self.metrics_db = None
    
    async def _get_db(self):
        """Get metrics database instance."""
        if self.metrics_db is None:
            self.metrics_db = await get_metrics_db()
        return self.metrics_db
    
    async def log_document_upload(
        self,
        document_id: str,
        filename: str,
        user_id: str,
        file_size_bytes: int,
        processing_time_ms: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log document upload event."""
        try:
            db = await self._get_db()
            await db.log_document_event(
                document_id=document_id,
                filename=filename,
                event_type="upload",
                user_id=user_id,
                file_size_bytes=file_size_bytes,
                processing_time_ms=processing_time_ms,
                metadata=metadata
            )
            
            # Also log as system event for recent activity
            await db.log_system_event(
                event_type="document_uploaded",
                user_id=user_id,
                description=f"Uploaded document '{filename}'",
                details={
                    "document_id": document_id,
                    "file_size_bytes": file_size_bytes,
                    "processing_time_ms": processing_time_ms
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log document upload: {e}")
    
    async def log_document_delete(
        self,
        document_id: str,
        filename: str,
        user_id: str
    ) -> None:
        """Log document deletion event."""
        try:
            db = await self._get_db()
            await db.log_document_event(
                document_id=document_id,
                filename=filename,
                event_type="delete",
                user_id=user_id
            )
            
            # Also log as system event
            await db.log_system_event(
                event_type="document_deleted",
                user_id=user_id,
                description=f"Deleted document '{filename}'",
                details={"document_id": document_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to log document deletion: {e}")


class QueryEventCollector:
    """Collector for RAG query events."""
    
    def __init__(self):
        self.metrics_db = None
    
    async def _get_db(self):
        """Get metrics database instance."""
        if self.metrics_db is None:
            self.metrics_db = await get_metrics_db()
        return self.metrics_db
    
    async def log_query_event(
        self,
        user_id: str,
        query_text: str,
        response_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        result_count: int = 0,
        tokens_used: int = 0,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log a RAG query event with full details."""
        try:
            db = await self._get_db()
            await db.log_query(
                user_id=user_id,
                query_text=query_text,
                response_time_ms=response_time_ms,
                success=success,
                error_message=error_message,
                result_count=result_count,
                tokens_used=tokens_used,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Log as system event for recent activity (successful queries only)
            if success and result_count > 0:
                await db.log_system_event(
                    event_type="query_performed",
                    user_id=user_id,
                    description=f"Performed query with {result_count} results",
                    details={
                        "query_preview": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                        "response_time_ms": response_time_ms,
                        "result_count": result_count
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to log query event: {e}")


class UserSessionCollector:
    """Collector for user session events."""
    
    def __init__(self):
        self.metrics_db = None
        self.active_sessions: Dict[str, int] = {}
    
    async def _get_db(self):
        """Get metrics database instance."""
        if self.metrics_db is None:
            self.metrics_db = await get_metrics_db()
        return self.metrics_db
    
    async def start_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> int:
        """Start a new user session."""
        try:
            db = await self._get_db()
            session_id = await db.start_user_session(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Store active session
            self.active_sessions[user_id] = session_id
            
            # Log system event
            await db.log_system_event(
                event_type="user_login",
                user_id=user_id,
                description=f"User logged in",
                details={
                    "ip_address": ip_address,
                    "user_agent": user_agent
                }
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start user session: {e}")
            return 0
    
    async def end_session(self, user_id: str) -> None:
        """End a user session."""
        try:
            session_id = self.active_sessions.get(user_id)
            if session_id:
                db = await self._get_db()
                await db.end_user_session(session_id)
                del self.active_sessions[user_id]
                
                # Log system event
                await db.log_system_event(
                    event_type="user_logout",
                    user_id=user_id,
                    description=f"User logged out"
                )
                
        except Exception as e:
            logger.error(f"Failed to end user session: {e}")


class PerformanceMetricsCollector:
    """Collector for system performance metrics."""
    
    def __init__(self):
        self.metrics_db = None
        self._collection_task = None
        self._running = False
    
    async def _get_db(self):
        """Get metrics database instance."""
        if self.metrics_db is None:
            self.metrics_db = await get_metrics_db()
        return self.metrics_db
    
    async def start_collection(self, interval_seconds: int = 300) -> None:
        """Start background performance metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(
            self._collect_metrics_loop(interval_seconds)
        )
        logger.info(f"Started performance metrics collection (interval: {interval_seconds}s)")
    
    async def stop_collection(self) -> None:
        """Stop background performance metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped performance metrics collection")
    
    async def _collect_metrics_loop(self, interval_seconds: int) -> None:
        """Background loop to collect performance metrics."""
        while self._running:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _collect_current_metrics(self) -> None:
        """Collect current system performance metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            db = await self._get_db()
            await db.log_performance_metrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                storage_usage_percent=(disk.used / disk.total) * 100,
                active_connections=0,  # Would get from database connection pool
                cache_hit_rate=0.0,    # Would get from cache system
                error_rate=0.0         # Would calculate from recent error logs
            )
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")


class DailyAggregationTask:
    """Background task for daily metrics aggregation."""
    
    def __init__(self):
        self.metrics_db = None
        self._aggregation_task = None
        self._running = False
    
    async def _get_db(self):
        """Get metrics database instance."""
        if self.metrics_db is None:
            self.metrics_db = await get_metrics_db()
        return self.metrics_db
    
    async def start_aggregation(self) -> None:
        """Start daily aggregation task."""
        if self._running:
            return
        
        self._running = True
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        logger.info("Started daily metrics aggregation task")
    
    async def stop_aggregation(self) -> None:
        """Stop daily aggregation task."""
        self._running = False
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped daily metrics aggregation task")
    
    async def _aggregation_loop(self) -> None:
        """Background loop for daily aggregation."""
        while self._running:
            try:
                # Calculate time until next midnight
                now = datetime.now()
                tomorrow = now.replace(hour=0, minute=5, second=0, microsecond=0) + timedelta(days=1)
                sleep_seconds = (tomorrow - now).total_seconds()
                
                # Wait until 00:05 next day
                await asyncio.sleep(sleep_seconds)
                
                # Run aggregation for yesterday
                yesterday = (now - timedelta(days=1)).date()
                await self._run_daily_aggregation(yesterday)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in daily aggregation: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _run_daily_aggregation(self, target_date) -> None:
        """Run daily metrics aggregation for a specific date."""
        try:
            db = await self._get_db()
            await db.aggregate_daily_metrics(target_date)
            logger.info(f"Completed daily aggregation for {target_date}")
        except Exception as e:
            logger.error(f"Failed daily aggregation for {target_date}: {e}")


# Global collector instances
document_collector = DocumentEventCollector()
query_collector = QueryEventCollector()
session_collector = UserSessionCollector()
performance_collector = PerformanceMetricsCollector()
daily_aggregator = DailyAggregationTask()


async def start_background_tasks() -> None:
    """Start all background metric collection tasks."""
    await performance_collector.start_collection(interval_seconds=300)  # Every 5 minutes
    await daily_aggregator.start_aggregation()
    logger.info("Started all background metrics collection tasks")


async def stop_background_tasks() -> None:
    """Stop all background metric collection tasks."""
    await performance_collector.stop_collection()
    await daily_aggregator.stop_aggregation()
    logger.info("Stopped all background metrics collection tasks")


# Context manager for background tasks
@asynccontextmanager
async def metrics_context():
    """Context manager for metrics collection lifecycle."""
    try:
        await start_background_tasks()
        yield
    finally:
        await stop_background_tasks()