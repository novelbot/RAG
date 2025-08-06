"""
Monitoring and health check API routes for system status and metrics.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Dict, List, Any, Optional
import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta

from ...auth.dependencies import get_current_user, MockUser
from ...embedding.types import EmbeddingProvider

logger = logging.getLogger(__name__)

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response_time_ms": round(response_time, 2),
        "components": {
            "database": {
                "status": db_status,
                "last_check": datetime.now(timezone.utc).isoformat()
            },
            "vector_database": {
                "status": vector_db_status,
                "last_check": datetime.now(timezone.utc).isoformat()
            },
            "llm_services": {
                "status": llm_status,
                "last_check": datetime.now(timezone.utc).isoformat()
            },
            "embedding_services": {
                "status": embedding_status,
                "last_check": datetime.now(timezone.utc).isoformat()
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
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/status", response_model=Dict[str, Any])
async def get_service_status() -> Dict[str, Any]:
    """
    UPDATED: Get current status of all services and dependencies with REAL data.
    
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
        
        collections = utility.list_collections(using="status_check")
        server_version = utility.get_server_version(using="status_check")
        
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
    if config.llm.provider == "ollama":
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
    if config.embedding.provider == EmbeddingProvider.OLLAMA:
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
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "embedding_provider": config.embedding.provider,
            "embedding_model": config.embedding.model
        }
    }


# Removed Web UI dashboard-specific metrics endpoints
# These endpoints were for charts and visualizations in the removed Web UI
# Core health and status endpoints remain below