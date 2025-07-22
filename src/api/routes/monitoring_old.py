"""
Monitoring and health check API routes for system status and metrics.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Dict, List, Any, Optional
import asyncio
import time
from datetime import datetime, timedelta

from ...auth.dependencies import get_current_user, MockUser

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
security = HTTPBearer()


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for all system components.
    
    Returns:
        Dict: System health status and component details
    """
    # Simulate async health checks for various components
    start_time = time.time()
    
    # Check database connectivity
    await asyncio.sleep(0.1)
    db_status = "healthy"
    
    # Check vector database connectivity
    await asyncio.sleep(0.1)
    vector_db_status = "healthy"
    
    # Check LLM services
    await asyncio.sleep(0.1)
    llm_status = "healthy"
    
    # Check embedding services
    await asyncio.sleep(0.1)
    embedding_status = "healthy"
    
    response_time = (time.time() - start_time) * 1000
    
    overall_status = "healthy"  # All components healthy
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "response_time_ms": round(response_time, 2),
        "components": {
            "database": {
                "status": db_status,
                "last_check": datetime.utcnow().isoformat()
            },
            "vector_database": {
                "status": vector_db_status,
                "last_check": datetime.utcnow().isoformat()
            },
            "llm_services": {
                "status": llm_status,
                "last_check": datetime.utcnow().isoformat()
            },
            "embedding_services": {
                "status": embedding_status,
                "last_check": datetime.utcnow().isoformat()
            }
        },
        "version": "0.1.0"
    }


@router.get("/health/simple", response_model=Dict[str, str])
async def simple_health_check() -> Dict[str, str]:
    """
    Simple health check for load balancers and basic monitoring.
    
    Returns:
        Dict: Basic health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed system performance metrics.
    
    Args:
        current_user: Authenticated user (admin role required)
        
    Returns:
        Dict: System performance metrics
        
    Raises:
        HTTPException: 403 if user doesn't have admin role
    """
    # Check admin permissions
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for metrics access"
        )
    
    # Simulate async metrics collection
    await asyncio.sleep(0.2)
    
    # TODO: Implement actual metrics collection
    return {
        "system": {
            "cpu_usage_percent": 45.2,
            "memory_usage_percent": 67.8,
            "disk_usage_percent": 23.1,
            "uptime_seconds": 86400
        },
        "application": {
            "total_requests": 15420,
            "requests_per_minute": 23.5,
            "average_response_time_ms": 156.7,
            "error_rate_percent": 0.12,
            "active_connections": 45
        },
        "database": {
            "connection_pool_usage": 8,
            "connection_pool_size": 20,
            "query_count": 8934,
            "average_query_time_ms": 12.3
        },
        "vector_database": {
            "collection_count": 5,
            "total_vectors": 125000,
            "index_status": "ready",
            "search_latency_ms": 23.1
        },
        "llm_services": {
            "total_requests": 342,
            "successful_requests": 338,
            "average_latency_ms": 1250.5,
            "token_usage": 45230
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/logs", response_model=Dict[str, Any])
async def get_logs(
    level: Optional[str] = "INFO",
    limit: int = 100,
    since: Optional[str] = None,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get application logs with filtering.
    
    Args:
        level: Log level filter (DEBUG, INFO, WARNING, ERROR)
        limit: Maximum number of log entries
        since: ISO timestamp to get logs since
        current_user: Authenticated user (admin role required)
        
    Returns:
        Dict: Filtered log entries
        
    Raises:
        HTTPException: 403 if user doesn't have admin role
    """
    # Check admin permissions
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for log access"
        )
    
    # Simulate async log retrieval
    await asyncio.sleep(0.3)
    
    # Parse since parameter
    since_datetime = None
    if since:
        try:
            since_datetime = datetime.fromisoformat(since.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 'since' timestamp format. Use ISO format."
            )
    
    # TODO: Implement actual log retrieval
    mock_logs = [
        {
            "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            "level": "INFO" if i % 3 != 0 else "WARNING",
            "module": f"module_{i % 5}",
            "message": f"Sample log message {i}",
            "request_id": f"req_{i}"
        }
        for i in range(1, 201)
    ]
    
    # Apply filters
    if level and level != "ALL":
        mock_logs = [log for log in mock_logs if log["level"] == level]
    
    if since_datetime:
        mock_logs = [
            log for log in mock_logs 
            if datetime.fromisoformat(log["timestamp"]) >= since_datetime
        ]
    
    return {
        "logs": mock_logs[:limit],
        "total": len(mock_logs),
        "filters": {
            "level": level,
            "since": since,
            "limit": limit
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/stats", response_model=Dict[str, Any])
async def get_usage_statistics(
    period: str = "24h",
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get usage statistics for the specified period.
    
    Args:
        period: Time period (1h, 24h, 7d, 30d)
        current_user: Authenticated user
        
    Returns:
        Dict: Usage statistics
        
    Raises:
        HTTPException: 400 if invalid period specified
    """
    valid_periods = {"1h", "24h", "7d", "30d"}
    if period not in valid_periods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period. Valid options: {valid_periods}"
        )
    
    # Simulate async statistics collection
    await asyncio.sleep(0.2)
    
    # TODO: Implement actual statistics collection
    period_multiplier = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}[period]
    base_requests = 100
    
    return {
        "period": period,
        "user_id": current_user.id,
        "statistics": {
            "total_queries": base_requests * period_multiplier,
            "search_queries": int(base_requests * period_multiplier * 0.6),
            "ask_queries": int(base_requests * period_multiplier * 0.4),
            "documents_uploaded": max(1, period_multiplier // 24),
            "average_query_time_ms": 245.3,
            "total_tokens_used": base_requests * period_multiplier * 50,
            "error_count": max(0, period_multiplier // 10)
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/alerts", response_model=Dict[str, str])
async def create_alert(
    alert_data: Dict[str, Any],
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Create a monitoring alert configuration.
    
    Args:
        alert_data: Alert configuration data
        current_user: Authenticated user (admin role required)
        
    Returns:
        Dict: Alert creation confirmation
        
    Raises:
        HTTPException: 403 if user doesn't have admin role
    """
    # Check admin permissions
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for alert management"
        )
    
    # Simulate async alert creation
    await asyncio.sleep(0.1)
    
    # TODO: Implement actual alert management
    alert_id = f"alert_{hash(str(alert_data))}"
    
    return {
        "message": "Alert created successfully",
        "alert_id": alert_id
    }


@router.get("/status", response_model=Dict[str, Any])
async def get_service_status() -> Dict[str, Any]:
    """
    Get current status of all services and dependencies.
    
    Returns:
        Dict: Service status overview
    """
    from ...core.config import get_config
    import psutil
    import pymysql
    from pymilvus import connections, utility
    import ollama
    
    config = get_config()
    start_time = time.time()
    
    # Calculate actual server uptime
    boot_time = psutil.boot_time()
    current_time = time.time()
    uptime_seconds = current_time - boot_time
    uptime_days = int(uptime_seconds // 86400)
    uptime_hours = int((uptime_seconds % 86400) // 3600)
    uptime_minutes = int((uptime_seconds % 3600) // 60)
    uptime_str = f"{uptime_days}d {uptime_hours}h {uptime_minutes}m"
    
    # Test Database Connection
    database_status = {"status": "disconnected", "response_time_ms": None, "connection_count": 0}
    try:
        db_start = time.time()
        connection = pymysql.connect(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.name,
            connect_timeout=5
        )
        db_time = (time.time() - db_start) * 1000
        
        with connection.cursor() as cursor:
            cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
            result = cursor.fetchone()
            connection_count = int(result[1]) if result else 0
        
        connection.close()
        database_status = {
            "status": "connected",
            "response_time_ms": round(db_time, 2),
            "connection_count": connection_count,
            "host": config.database.host,
            "database": config.database.name
        }
    except Exception as e:
        database_status = {
            "status": "disconnected",
            "error": str(e),
            "host": config.database.host,
            "database": config.database.name
        }
    
    # Test Milvus Vector Database Connection
    vector_db_status = {"status": "disconnected", "response_time_ms": None, "collection_count": 0}
    try:
        milvus_start = time.time()
        connections.connect(
            alias="status_check",
            host=config.milvus.host,
            port=config.milvus.port
        )
        milvus_time = (time.time() - milvus_start) * 1000
        
        collections = utility.list_collections()
        server_version = utility.get_server_version()
        
        connections.disconnect("status_check")
        vector_db_status = {
            "status": "connected",
            "response_time_ms": round(milvus_time, 2),
            "collection_count": len(collections),
            "collections": collections,
            "version": server_version,
            "host": config.milvus.host
        }
    except Exception as e:
        vector_db_status = {
            "status": "disconnected",
            "error": str(e),
            "host": config.milvus.host
        }
    
    # Test LLM Providers (Ollama)
    llm_providers = {}
    if config.llm.provider.lower() == "ollama":
        try:
            llm_start = time.time()
            # Test LLM model
            llm_response = ollama.chat(
                model=config.llm.model,
                messages=[{"role": "user", "content": "test"}],
                options={"num_predict": 1}  # Minimal response
            )
            llm_time = (time.time() - llm_start) * 1000
            
            llm_providers["ollama"] = {
                "status": "available",
                "model": config.llm.model,
                "latency_ms": round(llm_time, 2),
                "response": "success"
            }
        except Exception as e:
            llm_providers["ollama"] = {
                "status": "unavailable",
                "model": config.llm.model,
                "error": str(e)
            }
    
    # Test Embedding Providers (Ollama)
    embedding_providers = {}
    if config.embedding.provider.lower() == "ollama":
        try:
            embed_start = time.time()
            # Test embedding model
            embed_response = ollama.embeddings(
                model=config.embedding.model,
                prompt="test"
            )
            embed_time = (time.time() - embed_start) * 1000
            
            embedding_providers["ollama"] = {
                "status": "available",
                "model": config.embedding.model,
                "latency_ms": round(embed_time, 2),
                "dimension": len(embed_response["embedding"])
            }
        except Exception as e:
            embedding_providers["ollama"] = {
                "status": "unavailable",
                "model": config.embedding.model,
                "error": str(e)
            }
    
    # Determine overall status
    services_status = [
        database_status["status"] == "connected",
        vector_db_status["status"] == "connected",
        any(provider.get("status") == "available" for provider in llm_providers.values()),
        any(provider.get("status") == "available" for provider in embedding_providers.values())
    ]
    
    if all(services_status):
        overall_status = "operational"
    elif any(services_status):
        overall_status = "degraded"
    else:
        overall_status = "down"
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "services": {
            "api_server": {
                "status": "running",
                "uptime": uptime_str,
                "version": config.version,
                "environment": config.environment
            },
            "database": database_status,
            "vector_database": vector_db_status,
            "llm_providers": llm_providers,
            "embedding_providers": embedding_providers
        },
        "overall_status": overall_status,
        "response_time_ms": round(total_time, 2),
        "last_updated": datetime.utcnow().isoformat(),
        "configuration": {
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "embedding_provider": config.embedding.provider,
            "embedding_model": config.embedding.model
        }
    }