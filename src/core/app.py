"""
Main application factory and setup for the RAG server.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
import uvicorn
from typing import AsyncGenerator

from .config import get_config
from .exceptions import RAGException
from .logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG Server...")
    config = get_config()
    
    # Initialize components
    await initialize_components(config)
    logger.info("RAG Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Server...")
    await cleanup_components()
    logger.info("RAG Server shutdown complete")


async def initialize_components(config):
    """Initialize all application components"""
    # TODO: Initialize database connections
    # TODO: Initialize Milvus connection
    # TODO: Initialize LLM clients
    # TODO: Initialize embedding models
    pass


async def cleanup_components():
    """Clean up application components"""
    # TODO: Close database connections
    # TODO: Close Milvus connection
    # TODO: Cleanup LLM clients
    pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    config = get_config()
    
    # Setup logging
    setup_logging(config.logging)
    
    # Create FastAPI app
    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description="RAG Server with Vector Database - A comprehensive RAG system with Milvus vector database, multi-LLM support, and fine-grained access control",
        docs_url="/docs",  # Always enable docs for development
        redoc_url="/redoc",  # Always enable redoc for development
        lifespan=lifespan
    )
    
    # Add middleware
    add_middleware(app, config)
    
    # Add exception handlers
    add_exception_handlers(app)
    
    # Register routes
    register_routes(app)
    
    return app


def add_middleware(app: FastAPI, config):
    """Add middleware to the application"""
    from ..api.middleware import (
        AuthenticationMiddleware, 
        RequestLoggingMiddleware, 
        RateLimitMiddleware
    )
    
    # Request logging middleware (first to capture all requests)
    app.add_middleware(RequestLoggingMiddleware)
    
    # TEMPORARILY DISABLE OTHER MIDDLEWARE FOR DEBUGGING
    # # Rate limiting middleware
    # app.add_middleware(
    #     RateLimitMiddleware,
    #     max_requests=100,
    #     window_seconds=60
    # )
    # 
    # # Authentication middleware
    # app.add_middleware(AuthenticationMiddleware)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    if not config.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure as needed
        )


def add_exception_handlers(app: FastAPI):
    """Add exception handlers to the application"""
    
    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "details": exc.details}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )


def register_routes(app: FastAPI):
    """Register all API routes"""
    from ..api.routes import api_router
    
    # Register main API router
    app.include_router(api_router)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "message": "RAG Server is running"}
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {"message": "RAG Server API", "version": get_config().version}


def run_server():
    """Run the server with uvicorn"""
    config = get_config()
    
    # Create app instance directly for debugging
    app = create_app()
    
    uvicorn.run(
        app,  # Direct app instance instead of factory
        host=config.api.host,
        port=config.api.port,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    run_server()