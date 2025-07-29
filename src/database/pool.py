"""
Advanced connection pooling with configuration and monitoring.
"""

import time
import threading
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.pool import QueuePool, NullPool, StaticPool, Pool
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, TimeoutError
from loguru import logger

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, ConfigurationError
from src.core.logging import LoggerMixin
from src.database.drivers import DatabaseDriverFactory


@dataclass
class PoolMetrics:
    """Connection pool metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    checked_out: int = 0
    overflow: int = 0
    invalid_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    connection_errors: int = 0
    last_updated: Optional[datetime] = None


class PoolMonitor:
    """Monitor connection pool performance and health."""
    
    def __init__(self, pool_name: str):
        self.pool_name = pool_name
        self.metrics = PoolMetrics()
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)
        
    def update_metrics(self, pool: Pool) -> None:
        """Update pool metrics."""
        with self._lock:
            try:
                # Use direct pool attributes with safe getattr
                self.metrics.total_connections = getattr(pool, 'size', 0) or 0
                self.metrics.checked_out = getattr(pool, 'checkedout', 0) or 0
                self.metrics.overflow = getattr(pool, 'overflow', 0) or 0
                self.metrics.idle_connections = getattr(pool, 'checkedin', 0) or 0
                self.metrics.active_connections = self.metrics.checked_out
                self.metrics.invalid_connections = 0
            except Exception:
                # Fallback to zero values if pool metrics unavailable
                self.metrics.total_connections = 0
                self.metrics.checked_out = 0
                self.metrics.overflow = 0
                self.metrics.idle_connections = 0
                self.metrics.active_connections = 0
                self.metrics.invalid_connections = 0
            
            self.metrics.last_updated = datetime.now(timezone.utc)
    
    def increment_hits(self) -> None:
        """Increment pool hit counter."""
        with self._lock:
            self.metrics.pool_hits += 1
    
    def increment_misses(self) -> None:
        """Increment pool miss counter."""
        with self._lock:
            self.metrics.pool_misses += 1
    
    def increment_errors(self) -> None:
        """Increment connection error counter."""
        with self._lock:
            self.metrics.connection_errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics."""
        with self._lock:
            uptime = datetime.now(timezone.utc) - self._start_time
            return {
                'pool_name': self.pool_name,
                'total_connections': self.metrics.total_connections,
                'active_connections': self.metrics.active_connections,
                'idle_connections': self.metrics.idle_connections,
                'checked_out': self.metrics.checked_out,
                'overflow': self.metrics.overflow,
                'invalid_connections': self.metrics.invalid_connections,
                'pool_hits': self.metrics.pool_hits,
                'pool_misses': self.metrics.pool_misses,
                'connection_errors': self.metrics.connection_errors,
                'hit_ratio': self._calculate_hit_ratio(),
                'uptime_seconds': uptime.total_seconds(),
                'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
            }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate pool hit ratio."""
        total_requests = self.metrics.pool_hits + self.metrics.pool_misses
        if total_requests == 0:
            return 0.0
        return self.metrics.pool_hits / total_requests


class AdvancedConnectionPool(LoggerMixin):
    """Advanced connection pool with monitoring and configuration."""
    
    def __init__(self, config: DatabaseConfig, pool_name: str = "default"):
        self.config = config
        self.pool_name = pool_name
        self.monitor = PoolMonitor(pool_name)
        self._engine: Optional[Engine] = None
        self._driver = None
        self._setup_pool()
        
    def _setup_pool(self) -> None:
        """Setup connection pool with configuration."""
        try:
            # Create database driver
            self._driver = DatabaseDriverFactory.create_driver(self.config)
            
            # Build connection URL
            connection_url = self._driver.build_connection_url()
            
            # Get engine options
            engine_options = self._driver.get_engine_options()
            
            # Enhanced pool configuration
            pool_config = self._get_pool_configuration()
            engine_options.update(pool_config)
            
            # Create engine
            self._engine = create_engine(connection_url, **engine_options)
            
            # Setup event listeners
            self._setup_event_listeners()
            
            self.logger.info(f"Connection pool '{self.pool_name}' created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup connection pool: {e}")
            raise DatabaseError(f"Pool setup failed: {e}")
    
    def _get_pool_configuration(self) -> Dict[str, Any]:
        """Get connection pool configuration."""
        return {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': 3600,  # 1 hour
            'pool_pre_ping': True,
            'pool_use_lifo': True,  # Use LIFO for better connection reuse
            'pool_reset_on_return': 'commit',  # Reset behavior
            'echo_pool': False,  # Set to True for debugging
        }
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners."""
        if not self._engine:
            return
            
        @event.listens_for(self._engine, "connect")
        def connect_handler(_dbapi_connection, connection_record):
            """Handle new connection creation."""
            self.logger.debug(f"New connection created in pool '{self.pool_name}'")
            connection_record.info['created_at'] = datetime.now(timezone.utc)
            
        @event.listens_for(self._engine, "checkout")
        def checkout_handler(_dbapi_connection, connection_record, _connection_proxy):
            """Handle connection checkout."""
            self.logger.debug(f"Connection checked out from pool '{self.pool_name}'")
            connection_record.info['checked_out_at'] = datetime.now(timezone.utc)
            self.monitor.increment_hits()
            
        @event.listens_for(self._engine, "checkin")
        def checkin_handler(_dbapi_connection, connection_record):
            """Handle connection checkin."""
            self.logger.debug(f"Connection checked in to pool '{self.pool_name}'")
            connection_record.info['checked_in_at'] = datetime.now(timezone.utc)
            
        @event.listens_for(self._engine, "invalidate")
        def invalidate_handler(_dbapi_connection, _connection_record, exception):
            """Handle connection invalidation."""
            self.logger.warning(f"Connection invalidated in pool '{self.pool_name}': {exception}")
            self.monitor.increment_errors()
            
        @event.listens_for(self._engine, "soft_invalidate")
        def soft_invalidate_handler(_dbapi_connection, _connection_record, exception):
            """Handle soft connection invalidation."""
            self.logger.debug(f"Connection soft invalidated in pool '{self.pool_name}': {exception}")
            
        @event.listens_for(self._engine, "close")
        def close_handler(_dbapi_connection, _connection_record):
            """Handle connection close."""
            self.logger.debug(f"Connection closed in pool '{self.pool_name}'")
            
        @event.listens_for(self._engine, "detach")
        def detach_handler(_dbapi_connection, _connection_record):
            """Handle connection detach."""
            self.logger.debug(f"Connection detached from pool '{self.pool_name}'")
    
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        if not self._engine:
            raise DatabaseError("Connection pool not initialized")
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """
        Get connection from pool as context manager.
        
        Yields:
            Connection: SQLAlchemy connection
        """
        connection = None
        start_time = time.time()
        
        try:
            if not self._engine:
                raise DatabaseError("Connection pool not initialized")
            connection = self._engine.connect()
            self.logger.debug(f"Connection acquired from pool '{self.pool_name}'")
            
            # Update metrics
            self.monitor.update_metrics(self._engine.pool)
            
            yield connection
            
        except TimeoutError as e:
            self.logger.error(f"Connection timeout in pool '{self.pool_name}': {e}")
            self.monitor.increment_errors()
            raise DatabaseError(f"Connection timeout: {e}")
            
        except DisconnectionError as e:
            self.logger.error(f"Connection disconnected in pool '{self.pool_name}': {e}")
            self.monitor.increment_errors()
            if connection:
                connection.invalidate()
            raise DatabaseError(f"Connection disconnected: {e}")
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error in pool '{self.pool_name}': {e}")
            self.monitor.increment_errors()
            if connection:
                connection.invalidate()
            raise DatabaseError(f"Database error: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in pool '{self.pool_name}': {e}")
            self.monitor.increment_errors()
            if connection:
                connection.invalidate()
            raise DatabaseError(f"Unexpected error: {e}")
            
        finally:
            if connection:
                connection.close()
                execution_time = time.time() - start_time
                self.logger.debug(f"Connection released to pool '{self.pool_name}' (execution time: {execution_time:.3f}s)")
    
    @contextmanager
    def get_transaction(self):
        """
        Get transaction from pool as context manager.
        
        Yields:
            Connection: SQLAlchemy connection within transaction
        """
        connection = None
        transaction = None
        start_time = time.time()
        
        try:
            if not self._engine:
                raise DatabaseError("Connection pool not initialized")
            connection = self._engine.connect()
            transaction = connection.begin()
            self.logger.debug(f"Transaction started in pool '{self.pool_name}'")
            
            # Update metrics
            self.monitor.update_metrics(self._engine.pool)
            
            yield connection
            
            transaction.commit()
            self.logger.debug(f"Transaction committed in pool '{self.pool_name}'")
            
        except Exception as e:
            self.logger.error(f"Transaction error in pool '{self.pool_name}': {e}")
            self.monitor.increment_errors()
            
            if transaction:
                transaction.rollback()
                self.logger.debug(f"Transaction rolled back in pool '{self.pool_name}'")
            
            if connection:
                connection.invalidate()
                
            raise DatabaseError(f"Transaction error: {e}")
            
        finally:
            if connection:
                connection.close()
                execution_time = time.time() - start_time
                self.logger.debug(f"Transaction connection released to pool '{self.pool_name}' (execution time: {execution_time:.3f}s)")
    
    def test_connection(self) -> bool:
        """Test connection pool health."""
        try:
            with self.get_connection() as conn:
                if not self._driver:
                    return False
                health_query = self._driver.get_health_check_query()
                result = conn.execute(text(health_query))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Pool health check failed for '{self.pool_name}': {e}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get detailed pool status."""
        try:
            if self._engine:
                self.monitor.update_metrics(self._engine.pool)
            return self.monitor.get_metrics()
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {e}")
            return {'error': str(e)}
    
    def invalidate_pool(self) -> None:
        """Invalidate all connections in pool."""
        try:
            if self._engine:
                # Use engine.dispose() to invalidate and replace the pool
                self._engine.dispose()
                # Recreate the engine to get a fresh pool
                self._setup_pool()
                self.logger.info(f"Pool '{self.pool_name}' invalidated and recreated")
        except Exception as e:
            self.logger.error(f"Failed to invalidate pool '{self.pool_name}': {e}")
    
    def dispose(self) -> None:
        """Dispose of the connection pool."""
        try:
            if self._engine:
                self._engine.dispose()
                self.logger.info(f"Pool '{self.pool_name}' disposed")
        except Exception as e:
            self.logger.error(f"Failed to dispose pool '{self.pool_name}': {e}")
    
    def recreate_pool(self) -> None:
        """Recreate the connection pool."""
        try:
            self.dispose()
            self._setup_pool()
            self.logger.info(f"Pool '{self.pool_name}' recreated")
        except Exception as e:
            self.logger.error(f"Failed to recreate pool '{self.pool_name}': {e}")
            raise DatabaseError(f"Pool recreation failed: {e}")


class PoolManager(LoggerMixin):
    """Manager for multiple connection pools."""
    
    def __init__(self):
        self._pools: Dict[str, AdvancedConnectionPool] = {}
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._monitoring = False
        
    def create_pool(self, name: str, config: DatabaseConfig) -> AdvancedConnectionPool:
        """
        Create a new connection pool.
        
        Args:
            name: Pool name
            config: Database configuration
            
        Returns:
            AdvancedConnectionPool instance
        """
        with self._lock:
            if name in self._pools:
                raise DatabaseError(f"Pool '{name}' already exists")
            
            pool = AdvancedConnectionPool(config, name)
            self._pools[name] = pool
            
            self.logger.info(f"Created connection pool: {name}")
            return pool
    
    def get_pool(self, name: str) -> AdvancedConnectionPool:
        """
        Get connection pool by name.
        
        Args:
            name: Pool name
            
        Returns:
            AdvancedConnectionPool instance
        """
        with self._lock:
            if name not in self._pools:
                raise DatabaseError(f"Pool '{name}' not found")
            return self._pools[name]
    
    def remove_pool(self, name: str) -> None:
        """
        Remove connection pool.
        
        Args:
            name: Pool name
        """
        with self._lock:
            if name not in self._pools:
                raise DatabaseError(f"Pool '{name}' not found")
            
            pool = self._pools[name]
            pool.dispose()
            del self._pools[name]
            
            self.logger.info(f"Removed connection pool: {name}")
    
    def get_all_pools(self) -> Dict[str, AdvancedConnectionPool]:
        """Get all connection pools."""
        with self._lock:
            return self._pools.copy()
    
    def test_all_pools(self) -> Dict[str, bool]:
        """Test all connection pools."""
        results = {}
        with self._lock:
            for name, pool in self._pools.items():
                results[name] = pool.test_connection()
        return results
    
    def get_all_pool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pools."""
        status = {}
        with self._lock:
            for name, pool in self._pools.items():
                status[name] = pool.get_pool_status()
        return status
    
    def invalidate_all_pools(self) -> None:
        """Invalidate all connection pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.invalidate_pool()
    
    def dispose_all_pools(self) -> None:
        """Dispose all connection pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.dispose()
            self._pools.clear()
            self.logger.info("All connection pools disposed")
    
    def start_monitoring(self, interval: int = 60) -> None:
        """
        Start pool monitoring thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_pools,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Pool monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop pool monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Pool monitoring stopped")
    
    def _monitor_pools(self, interval: int) -> None:
        """Monitor pools periodically."""
        while self._monitoring:
            try:
                status = self.get_all_pool_status()
                
                # Log pool statistics
                for pool_name, pool_status in status.items():
                    if 'error' not in pool_status:
                        self.logger.debug(
                            f"Pool '{pool_name}': "
                            f"Active={pool_status.get('active_connections', 0)}, "
                            f"Idle={pool_status.get('idle_connections', 0)}, "
                            f"Hits={pool_status.get('pool_hits', 0)}, "
                            f"Errors={pool_status.get('connection_errors', 0)}"
                        )
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Pool monitoring error: {e}")
                time.sleep(interval)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        self.dispose_all_pools()