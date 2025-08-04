"""
User database configuration using SQLite for user data and conversations.
Separate from the read-only RDB used for embeddings.
"""

import os
from typing import Generator
from sqlalchemy import create_engine, Column, Integer, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone

from .logging import get_logger

logger = get_logger(__name__)

# Create the declarative base for user-related models
UserBase = declarative_base()

# Global session maker for user database
UserSessionLocal = None
user_engine = None


def get_user_db_path() -> str:
    """Get the path for the SQLite user database."""
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    return os.path.join(data_dir, "user_data.db")


def init_user_database():
    """Initialize SQLite database connection for user data."""
    global UserSessionLocal, user_engine
    
    database_path = get_user_db_path()
    database_url = f"sqlite:///{database_path}"
    
    # Create engine with SQLite-specific settings
    user_engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False,  # Set to True for SQL debugging
        connect_args={"check_same_thread": False}  # Allow SQLite to be used in multiple threads
    )
    
    # Create session maker
    UserSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=user_engine)
    
    logger.info(f"User database initialized: {database_path}")
    
    # Create all tables
    create_user_tables()


def create_user_tables():
    """Create all user-related tables."""
    global user_engine
    
    if user_engine is None:
        init_user_database()
    
    # Import models to register them
    from ..models.conversation import ConversationSession, ConversationTurn
    from ..models.query_log import QueryLog
    
    # Update Base metadata to use UserBase
    for model_class in [ConversationSession, ConversationTurn, QueryLog]:
        if hasattr(model_class, '__table__'):
            table_name = model_class.__table__.name
            if table_name not in UserBase.metadata.tables:
                model_class.__table__.tometadata(UserBase.metadata)
    
    # Create tables
    UserBase.metadata.create_all(bind=user_engine)
    logger.info("User database tables created/verified")


def get_user_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for user database sessions.
    
    Yields:
        Session: SQLAlchemy session for user database
    """
    global UserSessionLocal
    
    if UserSessionLocal is None:
        init_user_database()
    
    if UserSessionLocal is None:
        raise RuntimeError("Failed to initialize user database session")
    
    db = UserSessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"User database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_user_session() -> Session:
    """
    Get a user database session (for non-FastAPI usage).
    
    Returns:
        Session: SQLAlchemy session for user database (caller must close)
    """
    global UserSessionLocal
    
    if UserSessionLocal is None:
        init_user_database()
    
    if UserSessionLocal is None:
        raise RuntimeError("Failed to initialize user database session")
    
    return UserSessionLocal()


class UserTimestampMixin:
    """Mixin to add timestamp fields to user models."""
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


# Initialize on import
init_user_database()