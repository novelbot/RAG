"""
Database base models and configuration.
"""

from typing import Generator
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, func, create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)

# Create the declarative base for all models
Base = declarative_base()

# Global session maker
SessionLocal = None
engine = None


def init_database():
    """Initialize database connection and session maker."""
    global SessionLocal, engine
    
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
    
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


class TimestampMixin:
    """Mixin to add timestamp fields to models."""
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)