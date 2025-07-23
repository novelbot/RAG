"""
Integration module for adding metrics collection to FastAPI application.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .collectors import (
    MetricsCollectionMiddleware,
    start_background_tasks,
    stop_background_tasks,
    document_collector,
    query_collector,
    session_collector
)
from .database import init_metrics_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def metrics_lifespan(app: FastAPI):
    """Application lifespan context manager for metrics system."""
    logger.info("Starting metrics collection system...")
    
    try:
        # Initialize metrics database
        await init_metrics_db("metrics.db")
        logger.info("Metrics database initialized")
        
        # Start background tasks
        await start_background_tasks()
        logger.info("Background metrics collection tasks started")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start metrics system: {e}")
        yield
        
    finally:
        # Stop background tasks
        logger.info("Stopping metrics collection system...")
        try:
            await stop_background_tasks()
            logger.info("Background metrics collection tasks stopped")
        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")


def setup_metrics(app: FastAPI) -> None:
    """Setup metrics collection for a FastAPI application."""
    
    # Add metrics collection middleware
    app.add_middleware(MetricsCollectionMiddleware)
    logger.info("Added metrics collection middleware")
    
    # Note: The lifespan should be set when creating the FastAPI app:
    # app = FastAPI(lifespan=metrics_lifespan)


# Helper functions to be used throughout the application
async def log_document_upload(
    document_id: str,
    filename: str,
    user_id: str,
    file_size_bytes: int,
    processing_time_ms: int,
    metadata: dict = None
) -> None:
    """Helper to log document upload events."""
    await document_collector.log_document_upload(
        document_id=document_id,
        filename=filename,
        user_id=user_id,
        file_size_bytes=file_size_bytes,
        processing_time_ms=processing_time_ms,
        metadata=metadata
    )


async def log_document_delete(
    document_id: str,
    filename: str,
    user_id: str
) -> None:
    """Helper to log document deletion events."""
    await document_collector.log_document_delete(
        document_id=document_id,
        filename=filename,
        user_id=user_id
    )


async def log_query_event(
    user_id: str,
    query_text: str,
    response_time_ms: int,
    success: bool,
    error_message: str = None,
    result_count: int = 0,
    tokens_used: int = 0,
    ip_address: str = None,
    user_agent: str = None
) -> None:
    """Helper to log RAG query events."""
    await query_collector.log_query_event(
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


async def log_user_login(
    user_id: str,
    ip_address: str = None,
    user_agent: str = None
) -> int:
    """Helper to log user login events."""
    return await session_collector.start_session(
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent
    )


async def log_user_logout(user_id: str) -> None:
    """Helper to log user logout events."""
    await session_collector.end_session(user_id)


# Decorator for automatic query logging
def log_query_execution(func):
    """Decorator to automatically log query execution metrics."""
    import time
    import asyncio
    from functools import wraps
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        success = False
        error_message = None
        result_count = 0
        
        try:
            result = await func(*args, **kwargs)
            success = True
            
            # Try to extract result count from response
            if isinstance(result, dict):
                if "results" in result:
                    result_count = len(result["results"])
                elif "documents" in result:
                    result_count = len(result["documents"])
                elif "answer" in result:
                    result_count = 1
            
            return result
            
        except Exception as e:
            error_message = str(e)
            raise
            
        finally:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract query information from args/kwargs
            query_text = "Unknown query"
            user_id = "unknown"
            
            # Try to find query and user info in function arguments
            if args:
                for arg in args:
                    if isinstance(arg, str) and len(arg) > 10:
                        query_text = arg
                        break
                    elif isinstance(arg, dict) and "query" in arg:
                        query_text = arg["query"]
                        break
            
            if "query" in kwargs:
                query_text = kwargs["query"]
            if "user_id" in kwargs:
                user_id = kwargs["user_id"]
                
            # Log the query asynchronously
            try:
                await log_query_event(
                    user_id=user_id,
                    query_text=query_text,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=error_message,
                    result_count=result_count
                )
            except Exception as e:
                logger.error(f"Failed to log query metrics: {e}")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        success = False
        error_message = None
        result_count = 0
        
        try:
            result = func(*args, **kwargs)
            success = True
            
            # Try to extract result count from response
            if isinstance(result, dict):
                if "results" in result:
                    result_count = len(result["results"])
                elif "documents" in result:
                    result_count = len(result["documents"])
                elif "answer" in result:
                    result_count = 1
            
            return result
            
        except Exception as e:
            error_message = str(e)
            raise
            
        finally:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract query information from args/kwargs
            query_text = "Unknown query"
            user_id = "unknown"
            
            # Try to find query and user info in function arguments
            if args:
                for arg in args:
                    if isinstance(arg, str) and len(arg) > 10:
                        query_text = arg
                        break
                    elif isinstance(arg, dict) and "query" in arg:
                        query_text = arg["query"]
                        break
            
            if "query" in kwargs:
                query_text = kwargs["query"]
            if "user_id" in kwargs:
                user_id = kwargs["user_id"]
                
            # Log the query asynchronously (run in background)
            try:
                asyncio.create_task(log_query_event(
                    user_id=user_id,
                    query_text=query_text,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=error_message,
                    result_count=result_count
                ))
            except Exception as e:
                logger.error(f"Failed to log query metrics: {e}")
    
    # Return appropriate wrapper based on whether the function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper