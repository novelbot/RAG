"""
Base database connection management using SQLAlchemy.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine, MetaData, Table, text
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.engine import Connection
from contextlib import contextmanager
from loguru import logger
import time
import threading
from datetime import datetime, timezone

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, ConfigurationError
from src.core.logging import LoggerMixin


class DatabaseManager(LoggerMixin):
    """
    Base SQLAlchemy database manager with connection pooling and health checks.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager with configuration.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._metadata: Optional[MetaData] = None
        self._lock = threading.Lock()
        self._is_connected = False
        self._last_health_check = None
        
        # Initialize engine
        self._create_engine()
        
    def _create_engine(self) -> None:
        """Create SQLAlchemy engine with connection pooling."""
        try:
            # Build connection URL
            connection_url = self._build_connection_url()
            
            # Engine configuration
            engine_config = {
                'echo': False,  # Set to True for SQL debugging
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.pool_timeout,
                'pool_recycle': 3600,  # Recycle connections after 1 hour
                'pool_pre_ping': True,  # Enable pessimistic disconnect detection
                'pool_use_lifo': True,  # Use LIFO for better connection reuse
            }
            
            # Create engine
            self._engine = create_engine(connection_url, **engine_config)
            
            # Set up event listeners
            self._setup_event_listeners()
            
            self.logger.info(f"Database engine created for {self.config.driver}")
            
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {e}")
            raise DatabaseError(f"Engine creation failed: {e}")
    
    def _build_connection_url(self) -> str:
        """Build database connection URL."""
        if self.config.password:
            return f"{self.config.driver}://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
        else:
            return f"{self.config.driver}://{self.config.user}@{self.config.host}:{self.config.port}/{self.config.database}"
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for connection management."""
        if not self._engine:
            return
            
        from sqlalchemy import event
            
        # Listen for connection events
        @event.listens_for(self._engine, "connect")
        def connect_handler(dbapi_connection, connection_record):
            """Handle new connections."""
            self.logger.debug("New database connection established")
            
        @event.listens_for(self._engine, "checkout")
        def checkout_handler(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            self.logger.debug("Connection checked out from pool")
            
        @event.listens_for(self._engine, "checkin")
        def checkin_handler(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            self.logger.debug("Connection returned to pool")
            
        @event.listens_for(self._engine, "invalidate")
        def invalidate_handler(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            self.logger.warning(f"Connection invalidated: {exception}")
    
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        if not self._engine:
            raise DatabaseError("Database engine not initialized")
        return self._engine
    
    @property
    def metadata(self) -> MetaData:
        """Get SQLAlchemy metadata."""
        if not self._metadata:
            self._metadata = MetaData()
        return self._metadata
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection as context manager.
        
        Yields:
            Connection: SQLAlchemy connection
        """
        if not self._engine:
            raise DatabaseError("Database engine not initialized")
            
        connection = None
        try:
            connection = self._engine.connect()
            self.logger.debug("Database connection acquired")
            yield connection
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            if connection:
                connection.invalidate()
            raise DatabaseError(f"Connection error: {e}")
        finally:
            if connection:
                connection.close()
                self.logger.debug("Database connection released")
    
    @contextmanager
    def get_transaction(self):
        """
        Get database transaction as context manager.
        
        Yields:
            Connection: SQLAlchemy connection within transaction
        """
        if not self._engine:
            raise DatabaseError("Database engine not initialized")
            
        connection = None
        transaction = None
        try:
            connection = self._engine.connect()
            transaction = connection.begin()
            self.logger.debug("Database transaction started")
            yield connection
            transaction.commit()
            self.logger.debug("Database transaction committed")
        except Exception as e:
            self.logger.error(f"Database transaction error: {e}")
            if transaction:
                transaction.rollback()
                self.logger.debug("Database transaction rolled back")
            if connection:
                connection.invalidate()
            raise DatabaseError(f"Transaction error: {e}")
        finally:
            if connection:
                connection.close()
                self.logger.debug("Database connection released")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query result
        """
        with self.get_connection() as conn:
            result = conn.execute(text(query), params or {})
            return result
    
    def execute_transaction(self, queries: List[str], params: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of SQL queries
            params: List of query parameters
        """
        with self.get_transaction() as conn:
            for i, query in enumerate(queries):
                query_params = params[i] if params and i < len(params) else {}
                conn.execute(text(query), query_params)
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            with self.get_connection() as conn:
                # Simple query to test connection
                conn.execute(text("SELECT 1"))
                self._is_connected = True
                self._last_health_check = datetime.now(timezone.utc)
                self.logger.debug("Database connection test successful")
                return True
        except Exception as e:
            self._is_connected = False
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status.
        
        Returns:
            Dict containing pool statistics
        """
        if not self._engine:
            return {}
            
        pool = self._engine.pool
        
        # Try to get pool status safely
        try:
            # For QueuePool and similar pool implementations
            status_info = {}
            
            # Pool size (configured size)
            if hasattr(pool, 'size'):
                status_info['size'] = getattr(pool, 'size', lambda: 0)() if callable(getattr(pool, 'size', None)) else getattr(pool, 'size', 0)
            
            # Checked in connections
            if hasattr(pool, 'checkedin'):
                status_info['checked_in'] = getattr(pool, 'checkedin', lambda: 0)() if callable(getattr(pool, 'checkedin', None)) else getattr(pool, 'checkedin', 0)
            
            # Checked out connections  
            if hasattr(pool, 'checkedout'):
                status_info['checked_out'] = getattr(pool, 'checkedout', lambda: 0)() if callable(getattr(pool, 'checkedout', None)) else getattr(pool, 'checkedout', 0)
            
            # Overflow connections
            if hasattr(pool, 'overflow'):
                status_info['overflow'] = getattr(pool, 'overflow', lambda: 0)() if callable(getattr(pool, 'overflow', None)) else getattr(pool, 'overflow', 0)
            
            # Use status() method if available
            if hasattr(pool, 'status') and callable(getattr(pool, 'status')):
                status_str = pool.status()
                status_info['status_string'] = status_str
                
            return status_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get pool status: {e}")
            return {'error': f'Unable to retrieve pool status: {e}'}
    
    def close(self) -> None:
        """Close database connections and dispose engine."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._is_connected = False
            self.logger.info("Database connections closed")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on database connection.
        
        Returns:
            Dict with health check results
        """
        try:
            with self.get_connection() as conn:
                # Simple query to test connection
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                return {
                    "status": "healthy",
                    "message": "Database connection is healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Database connection failed: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DatabaseFactory:
    """Factory class for creating database managers."""
    
    @staticmethod
    def create_manager(config: DatabaseConfig) -> DatabaseManager:
        """
        Create database manager based on configuration.
        
        Args:
            config: Database configuration
            
        Returns:
            DatabaseManager instance
        """
        # Validate configuration
        if not config.driver:
            raise ConfigurationError("Database driver not specified")
            
        if not config.host:
            raise ConfigurationError("Database host not specified")
            
        if not config.database:
            raise ConfigurationError("Database name not specified")
        
        # Create manager
        return DatabaseManager(config)
    
    @staticmethod
    def create_from_url(url: str, **kwargs) -> DatabaseManager:
        """
        Create database manager from connection URL.
        
        Args:
            url: Database connection URL
            **kwargs: Additional configuration options
            
        Returns:
            DatabaseManager instance
        """
        # Parse URL to extract configuration
        # This is a simplified implementation
        # In practice, you'd use urllib.parse or SQLAlchemy's URL parsing
        
        # For now, create a basic config
        # Extract known fields from kwargs
        known_fields = {'driver', 'host', 'port', 'name', 'database', 'user', 'password'}
        config_kwargs = {k: v for k, v in kwargs.items() if k not in known_fields}
        
        config = DatabaseConfig(
            driver=kwargs.get('driver', 'postgresql'),
            host=kwargs.get('host', 'localhost'),
            port=kwargs.get('port', 5432),
            database=kwargs.get('database', kwargs.get('name', 'test')),  # Use database instead of name
            user=kwargs.get('user', 'test'),
            password=kwargs.get('password', ''),
            **config_kwargs
        )
        
        return DatabaseManager(config)