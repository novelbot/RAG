"""
Database base models and configuration.
"""

from typing import Generator, TYPE_CHECKING
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, func, create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone

from .logging import get_logger

if TYPE_CHECKING:
    from .config import AppConfig

logger = get_logger(__name__)

# Create the declarative base for all models
Base = declarative_base()

# Global session maker
SessionLocal = None
engine = None


def init_database():
    """Initialize database connection and session maker."""
    global SessionLocal, engine
    
    from .config import get_config
    config = get_config()
    
    # Build database URL
    if config.database.password:
        database_url = f"{config.database.driver}://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.name}"
    else:
        database_url = f"{config.database.driver}://{config.database.user}@{config.database.host}:{config.database.port}/{config.database.name}"
    
    # Create engine
    engine = create_engine(
        database_url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        pool_timeout=config.database.pool_timeout,
        pool_pre_ping=True
    )
    
    # Create session maker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    logger.info(f"Database initialized with {config.database.driver} driver")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Yields:
        Session: SQLAlchemy session
    """
    global SessionLocal
    
    if SessionLocal is None:
        init_database()
    
    if SessionLocal is None:
        raise RuntimeError("Failed to initialize database session")
    
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_session() -> Session:
    """
    Get a database session (for non-FastAPI usage).
    
    Returns:
        Session: SQLAlchemy session (caller must close)
    """
    global SessionLocal
    
    if SessionLocal is None:
        init_database()
    
    if SessionLocal is None:
        raise RuntimeError("Failed to initialize database session")
    
    return SessionLocal()


def get_db_session() -> Generator[Session, None, None]:
    """
    Alias for get_db() for backward compatibility.
    
    Yields:
        Session: SQLAlchemy session
    """
    yield from get_db()


def get_db(database_url: str = None):
    """
    Get database engine or session based on usage context.
    
    Args:
        database_url: Optional database URL for custom connection
        
    Returns:
        Engine or Session based on context
    """
    if database_url:
        # Return engine for custom database URL
        return create_engine(database_url, pool_pre_ping=True)
    else:
        # Return session generator for default database
        return get_db_session()


class TimestampMixin:
    """Mixin to add timestamp fields to models."""
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)