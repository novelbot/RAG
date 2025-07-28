"""
Main API router that includes all sub-routers.
"""

from fastapi import APIRouter

from .auth import router as auth_router
from .query import router as query_router
from .documents import router as documents_router
from .monitoring import router as monitoring_router
from .admin import router as admin_router
from .episode import router as episode_router

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include all sub-routers
api_router.include_router(auth_router)
api_router.include_router(query_router)
api_router.include_router(documents_router)
api_router.include_router(monitoring_router)
api_router.include_router(admin_router)
api_router.include_router(episode_router)

# Export for use in main application
__all__ = ["api_router"]