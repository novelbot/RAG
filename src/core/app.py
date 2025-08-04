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
    # Initialize metrics database
    try:
        from ..metrics.database import init_metrics_db
        await init_metrics_db("metrics.db")
        logger.info("✅ Metrics database initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize metrics database: {e}")
    
    # Initialize database connections
    try:
        from ..database.base import DatabaseFactory
        from ..core.database import init_database
        
        # Initialize primary database connection
        init_database()
        logger.info("✅ Primary database connection initialized")
        
        # Initialize RDB connections if configured
        if hasattr(config, 'rdb_connections') and config.rdb_connections:
            for conn_name, db_config in config.rdb_connections.items():
                try:
                    manager = DatabaseFactory.create_manager(db_config)
                    # Test connection
                    if manager.test_connection():
                        logger.info(f"✅ RDB connection '{conn_name}' initialized and tested")
                    else:
                        logger.warning(f"⚠️ RDB connection '{conn_name}' initialized but test failed")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize RDB connection '{conn_name}': {e}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database connections: {e}")
    
    # Initialize Milvus connection
    try:
        from ..milvus.client import MilvusClient
        
        milvus_client = MilvusClient(config.milvus)
        if milvus_client.connect():
            logger.info("✅ Milvus connection initialized")
            
            # Store client globally for access by other components
            import sys
            sys.modules[__name__].milvus_client = milvus_client
        else:
            logger.error("❌ Failed to connect to Milvus")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Milvus connection: {e}")
    
    # Initialize LLM clients
    try:
        from ..llm.manager import LLMManager, ProviderConfig, LoadBalancingStrategy
        from ..llm.base import LLMProvider
        
        # Create provider configurations from config
        provider_configs = []
        
        # Add primary LLM provider
        if config.llm.provider and (config.llm.api_key or config.llm.provider.lower() == 'ollama'):
            try:
                provider_enum = LLMProvider(config.llm.provider.lower())
                provider_config = ProviderConfig(
                    provider=provider_enum,
                    config=config.llm,
                    priority=1,
                    enabled=True
                )
                provider_configs.append(provider_config)
            except ValueError:
                logger.warning(f"Unknown LLM provider: {config.llm.provider}")
        
        if provider_configs:
            llm_manager = LLMManager(provider_configs)
            llm_manager.set_load_balancing_strategy(LoadBalancingStrategy.HEALTH_BASED)
            
            # Store manager globally for access by other components
            import sys
            sys.modules[__name__].llm_manager = llm_manager
            logger.info("✅ LLM manager initialized")
        else:
            logger.warning("⚠️ No LLM providers configured")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM clients: {e}")
    
    # Initialize embedding models
    try:
        from ..embedding.manager import EmbeddingManager, EmbeddingProviderConfig
        from ..embedding.types import EmbeddingProvider
        
        # Create provider configurations from config
        provider_configs = []
        
        # Add primary embedding provider
        if hasattr(config, 'embedding') and config.embedding:
            provider_config = EmbeddingProviderConfig(
                provider=config.embedding.provider,
                config=config.embedding,
                priority=1,
                enabled=True
            )
            provider_configs.append(provider_config)
        
        # Add additional embedding providers if configured
        if hasattr(config, 'embedding_providers') and config.embedding_providers:
            for provider_name, embedding_config in config.embedding_providers.items():
                if provider_name != "default":  # Skip default as it's already added
                    provider_config = EmbeddingProviderConfig(
                        provider=embedding_config.provider,
                        config=embedding_config,
                        priority=2,
                        enabled=True
                    )
                    provider_configs.append(provider_config)
        
        if provider_configs:
            embedding_manager = EmbeddingManager(provider_configs, enable_cache=True)
            
            # Store manager globally for access by other components
            import sys
            current_module = sys.modules[__name__]
            current_module.embedding_manager = embedding_manager
            # Also store in the main app module for easier access
            sys.modules['src.core.app'].embedding_manager = embedding_manager
            logger.info("✅ Embedding manager initialized")
        else:
            logger.warning("⚠️ No embedding providers configured")
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedding models: {e}")


async def cleanup_components():
    """Clean up application components"""
    # Close database connections
    try:
        from ..core.database import engine
        if engine:
            engine.dispose()
            logger.info("✅ Primary database connections closed")
    except Exception as e:
        logger.error(f"❌ Failed to close database connections: {e}")
    
    # Close Milvus connection
    try:
        import sys
        milvus_client = getattr(sys.modules.get(__name__), 'milvus_client', None)
        if milvus_client:
            milvus_client.disconnect()
            logger.info("✅ Milvus connection closed")
    except Exception as e:
        logger.error(f"❌ Failed to close Milvus connection: {e}")
    
    # Cleanup LLM clients
    try:
        import sys
        llm_manager = getattr(sys.modules.get(__name__), 'llm_manager', None)
        if llm_manager:
            # LLM Manager doesn't need explicit cleanup
            logger.info("✅ LLM clients cleaned up")
    except Exception as e:
        logger.error(f"❌ Failed to cleanup LLM clients: {e}")
    
    # Cleanup embedding models
    try:
        import sys
        embedding_manager = getattr(sys.modules.get('src.core.app'), 'embedding_manager', None)
        if embedding_manager:
            # Clear embedding cache
            embedding_manager.clear_cache()
            logger.info("✅ Embedding models cleaned up")
    except Exception as e:
        logger.error(f"❌ Failed to cleanup embedding models: {e}")


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


def run_server(host=None, port=None, reload=False):
    """Run the server with uvicorn"""
    config = get_config()
    
    # Use provided values or fall back to config
    server_host = host if host is not None else config.api.host
    server_port = port if port is not None else config.api.port
    
    if reload:
        # For reload mode, pass module path instead of app instance
        uvicorn.run(
            "src.core.app:create_app",
            factory=True,
            host=server_host,
            port=server_port,
            log_level=config.logging.level.lower(),
            reload=True
        )
    else:
        # Create app instance directly for production
        app = create_app()
        uvicorn.run(
            app,
            host=server_host,
            port=server_port,
            log_level=config.logging.level.lower(),
        )


if __name__ == "__main__":
    run_server()