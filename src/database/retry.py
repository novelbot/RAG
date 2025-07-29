"""
Database error handling and retry mechanisms with comprehensive recovery strategies.
"""

import time
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import exc, text, select, event
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import Pool
from sqlalchemy.sql import literal_column
from loguru import logger

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, RetryError, CircuitBreakerError
from src.core.logging import LoggerMixin
from src.database.pool import AdvancedConnectionPool
from src.database.drivers import DatabaseDriver

T = TypeVar('T')


class ErrorType(Enum):
    """Types of database errors for categorization."""
    CONNECTION_LOST = "connection_lost"
    TIMEOUT = "timeout"
    DEADLOCK = "deadlock"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INVALID_TRANSACTION = "invalid_transaction"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategy types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.CONNECTION_LOST,
        ErrorType.TIMEOUT,
        ErrorType.DEADLOCK,
        ErrorType.RESOURCE_EXHAUSTED
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass
class ErrorMetrics:
    """Metrics for error tracking."""
    total_errors: int = 0
    errors_by_type: Dict[ErrorType, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    last_error_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    retry_attempts: int = 0
    retry_successes: int = 0


class DatabaseErrorClassifier(LoggerMixin):
    """Classifier for database error types and retry decisions."""
    
    def __init__(self):
        self._error_patterns = self._build_error_patterns()
        self._retryable_exceptions = self._build_retryable_exceptions()
    
    def _build_error_patterns(self) -> Dict[ErrorType, List[str]]:
        """Build error patterns for different database types."""
        return {
            ErrorType.CONNECTION_LOST: [
                "Lost connection to MySQL server",
                "MySQL server has gone away",
                "SSL connection has been closed unexpectedly",
                "Connection was killed",
                "Connection reset by peer",
                "Broken pipe",
                "Connection refused",
                "Connection timed out",
                "Connection aborted",
                "Connection closed",
                "Connection not available",
                "Connection pool exhausted",
                "server closed the connection unexpectedly",
                "SSL SYSCALL error: Success",
                "SSL SYSCALL error: EOF detected",
                "terminating connection due to administrator command",
                "could not receive data from server",
                "connection to server was lost",
                "connection not open",
                "connection is closed",
                "Connection is closed",
                "ORA-03113",  # Oracle: end-of-file on communication channel
                "ORA-03114",  # Oracle: not connected to ORACLE
                "ORA-01041",  # Oracle: internal error. hostdef extension doesn't exist
                "ORA-12541",  # Oracle: TNS:no listener
                "ORA-12154",  # Oracle: TNS:could not resolve the connect identifier
                "TNS-12535",  # Oracle: operation timed out
                "TNS-12537",  # Oracle: connection closed
                "DPY-1001",   # Oracle: connection was terminated
                "DPY-4011",   # Oracle: connection timed out
            ],
            ErrorType.TIMEOUT: [
                "timeout",
                "timed out",
                "Timeout",
                "Query was cancelled",
                "Query execution was interrupted",
                "Lock wait timeout exceeded",
                "Statement timeout",
                "Connection timeout",
                "Read timeout",
                "Write timeout",
                "ORA-01013",  # Oracle: user requested cancel of current operation
                "ORA-00936",  # Oracle: missing expression (can indicate timeout)
            ],
            ErrorType.DEADLOCK: [
                "deadlock",
                "Deadlock found",
                "Lock wait timeout exceeded",
                "Transaction deadlock",
                "ORA-00060",  # Oracle: deadlock detected while waiting for resource
                "40001",      # PostgreSQL: serialization failure
                "40P01",      # PostgreSQL: deadlock detected
            ],
            ErrorType.CONSTRAINT_VIOLATION: [
                "constraint",
                "foreign key constraint",
                "unique constraint",
                "check constraint",
                "duplicate key",
                "UNIQUE constraint failed",
                "IntegrityError",
                "CHECK constraint",
                "NOT NULL constraint",
                "FOREIGN KEY constraint",
                "ORA-00001",  # Oracle: unique constraint violated
                "ORA-02290",  # Oracle: check constraint violated
                "ORA-02291",  # Oracle: integrity constraint violated - parent key not found
                "ORA-02292",  # Oracle: integrity constraint violated - child record found
                "23000",      # Generic: integrity constraint violation
                "23001",      # Generic: restrict violation
                "23502",      # PostgreSQL: not null violation
                "23503",      # PostgreSQL: foreign key violation
                "23505",      # PostgreSQL: unique violation
                "23514",      # PostgreSQL: check violation
            ],
            ErrorType.INVALID_TRANSACTION: [
                "invalid transaction",
                "transaction is not active",
                "transaction already committed",
                "transaction already rolled back",
                "Can't reconnect until invalid transaction is rolled back",
                "This transaction is inactive",
                "This connection is in a failed state",
                "This session is in a failed state",
                "invalid transaction is rolled back",
                "25P02",      # PostgreSQL: in failed sql transaction
                "25001",      # PostgreSQL: active sql transaction
            ],
            ErrorType.PERMISSION_DENIED: [
                "permission denied",
                "access denied",
                "insufficient privilege",
                "not authorized",
                "authentication failed",
                "login failed",
                "ORA-00942",  # Oracle: table or view does not exist
                "ORA-00980",  # Oracle: synonym translation is no longer valid
                "ORA-01031",  # Oracle: insufficient privileges
                "42000",      # Generic: syntax error or access violation
                "42501",      # PostgreSQL: insufficient privilege
            ],
            ErrorType.RESOURCE_EXHAUSTED: [
                "out of memory",
                "disk full",
                "too many connections",
                "connection limit exceeded",
                "resource temporarily unavailable",
                "Cannot allocate memory",
                "Out of memory",
                "WSREP has not yet prepared node for application use",
                "Too many connections",
                "User limit of connections exceeded",
                "ORA-00020",  # Oracle: maximum number of processes exceeded
                "ORA-00018",  # Oracle: maximum number of sessions exceeded
                "ORA-04031",  # Oracle: unable to allocate memory
                "53000",      # Generic: insufficient resources
                "53100",      # PostgreSQL: disk full
                "53200",      # PostgreSQL: out of memory
                "53300",      # PostgreSQL: too many connections
            ]
        }
    
    def _build_retryable_exceptions(self) -> List[type]:
        """Build list of retryable exception types."""
        return [
            exc.DisconnectionError,
            exc.TimeoutError,
            exc.ResourceClosedError,
            exc.InvalidRequestError,
            exc.OperationalError,
            exc.DatabaseError,
            exc.InterfaceError,
            exc.InternalError,
            # Add DBAPI-specific exceptions that are typically retryable
            ConnectionError,
            OSError,
            BrokenPipeError,
            ConnectionResetError,
            ConnectionRefusedError,
            ConnectionAbortedError,
        ]
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify database error into error type.
        
        Args:
            error: Exception to classify
            
        Returns:
            ErrorType classification
        """
        error_message = str(error).lower()
        
        # Check error patterns
        for error_type, patterns in self._error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message:
                    return error_type
        
        # Check exception types
        if isinstance(error, (exc.DisconnectionError, ConnectionError, BrokenPipeError)):
            return ErrorType.CONNECTION_LOST
        elif isinstance(error, (exc.TimeoutError, TimeoutError)):
            return ErrorType.TIMEOUT
        elif isinstance(error, exc.IntegrityError):
            return ErrorType.CONSTRAINT_VIOLATION
        elif isinstance(error, exc.InvalidRequestError):
            return ErrorType.INVALID_TRANSACTION
        elif isinstance(error, (exc.OperationalError, exc.DatabaseError)):
            # These are generic, need to check message
            if any(pattern in error_message for pattern in self._error_patterns[ErrorType.CONNECTION_LOST]):
                return ErrorType.CONNECTION_LOST
            elif any(pattern in error_message for pattern in self._error_patterns[ErrorType.TIMEOUT]):
                return ErrorType.TIMEOUT
            elif any(pattern in error_message for pattern in self._error_patterns[ErrorType.DEADLOCK]):
                return ErrorType.DEADLOCK
            elif any(pattern in error_message for pattern in self._error_patterns[ErrorType.RESOURCE_EXHAUSTED]):
                return ErrorType.RESOURCE_EXHAUSTED
        
        return ErrorType.UNKNOWN
    
    def is_retryable(self, error: Exception, retryable_errors: List[ErrorType]) -> bool:
        """
        Determine if error is retryable.
        
        Args:
            error: Exception to check
            retryable_errors: List of retryable error types
            
        Returns:
            True if error is retryable
        """
        error_type = self.classify_error(error)
        return error_type in retryable_errors
    
    def should_invalidate_connection(self, error: Exception) -> bool:
        """
        Determine if connection should be invalidated for error.
        
        Args:
            error: Exception to check
            
        Returns:
            True if connection should be invalidated
        """
        error_type = self.classify_error(error)
        return error_type in [
            ErrorType.CONNECTION_LOST,
            ErrorType.INVALID_TRANSACTION,
            ErrorType.RESOURCE_EXHAUSTED
        ]


class RetryDelayCalculator:
    """Calculator for retry delays with different strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self._fibonacci_cache = [1, 1]
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._get_fibonacci(attempt)
        else:  # CUSTOM or fallback
            delay = self.config.base_delay * (2 ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number with caching."""
        while len(self._fibonacci_cache) <= n:
            self._fibonacci_cache.append(
                self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            )
        return self._fibonacci_cache[n]


class CircuitBreaker(LoggerMixin):
    """Circuit breaker for database operations."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.half_open_successes = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} HALF_OPEN call limit exceeded")
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now(timezone.utc) - self.last_failure_time
        return elapsed.total_seconds() >= self.config.recovery_timeout
    
    def _record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
            else:
                self.failure_count = 0
    
    def _record_failure(self) -> None:
        """Record failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} transitioning to OPEN (half-open failure)")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker {self.name} transitioning to OPEN (failure threshold reached)")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'half_open_calls': self.half_open_calls,
                'half_open_successes': self.half_open_successes,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
            }


class DatabaseRetryHandler(LoggerMixin):
    """Main retry handler for database operations."""
    
    def __init__(self, 
                 pool: AdvancedConnectionPool,
                 retry_config: Optional[RetryConfig] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None):
        self.pool = pool
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        
        self.classifier = DatabaseErrorClassifier()
        self.delay_calculator = RetryDelayCalculator(self.retry_config)
        self.circuit_breaker = CircuitBreaker(f"db_{pool.pool_name}", self.circuit_config)
        self.metrics = ErrorMetrics()
        self._lock = threading.Lock()
        
        # Setup event listeners for connection handling
        self._setup_event_listeners()
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for error handling."""
        engine = self.pool.engine
        
        @event.listens_for(engine, "handle_error")
        def handle_error(context):
            """Handle database errors with custom logic."""
            error = context.original_exception
            error_type = self.classifier.classify_error(error)
            
            self.logger.debug(f"Handling error: {error_type.value} - {error}")
            
            # Update error metrics
            with self._lock:
                self.metrics.total_errors += 1
                if error_type not in self.metrics.errors_by_type:
                    self.metrics.errors_by_type[error_type] = 0
                self.metrics.errors_by_type[error_type] += 1
                self.metrics.last_error_time = datetime.now(timezone.utc)
            
            # Determine if this is a disconnect error
            if self.classifier.should_invalidate_connection(error):
                context.is_disconnect = True
                self.logger.warning(f"Marking connection as disconnected due to: {error}")
        
        @event.listens_for(engine, "invalidate")
        def connection_invalidated(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            self.logger.warning(f"Connection invalidated: {exception}")
            
            with self._lock:
                self.metrics.consecutive_failures += 1
    
    def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryError: If all retries exhausted
        """
        def retry_wrapper():
            return self._execute_with_retry_internal(func, *args, **kwargs)
        
        return self.circuit_breaker.call(retry_wrapper)
    
    def _execute_with_retry_internal(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Internal retry execution logic."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                with self._lock:
                    self.metrics.retry_attempts += 1
                
                result = func(*args, **kwargs)
                
                # Success - reset consecutive failures
                with self._lock:
                    self.metrics.consecutive_failures = 0
                    self.metrics.last_success_time = datetime.now(timezone.utc)
                    if attempt > 0:
                        self.metrics.retry_successes += 1
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self.classifier.classify_error(e)
                
                self.logger.warning(f"Attempt {attempt + 1} failed: {error_type.value} - {e}")
                
                # Check if error is retryable
                if not self.classifier.is_retryable(e, self.retry_config.retryable_errors):
                    self.logger.info(f"Error not retryable: {error_type.value}")
                    break
                
                # If this is the last attempt, don't delay
                if attempt >= self.retry_config.max_retries:
                    break
                
                # Calculate and apply delay
                delay = self.delay_calculator.calculate_delay(attempt)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                
                # Handle connection invalidation
                if self.classifier.should_invalidate_connection(e):
                    self._handle_connection_invalidation()
        
        # All retries exhausted
        with self._lock:
            self.metrics.consecutive_failures += 1
        
        raise RetryError(
            f"Operation failed after {self.retry_config.max_retries + 1} attempts",
            last_exception
        )
    
    def _handle_connection_invalidation(self) -> None:
        """Handle connection invalidation and pool cleanup."""
        try:
            self.pool.invalidate_pool()
            self.logger.info("Connection pool invalidated due to connection error")
        except Exception as e:
            self.logger.error(f"Failed to invalidate connection pool: {e}")
    
    @contextmanager
    def resilient_connection(self):
        """
        Get a resilient database connection with retry logic.
        
        Yields:
            Database connection with retry capabilities
        """
        def get_connection():
            return self.pool.get_connection()
        
        connection_manager = self.execute_with_retry(get_connection)
        
        try:
            with connection_manager as conn:
                yield ResilientConnection(conn, self)
        except Exception as e:
            error_type = self.classifier.classify_error(e)
            if self.classifier.should_invalidate_connection(e):
                self._handle_connection_invalidation()
            raise
    
    @contextmanager
    def resilient_transaction(self):
        """
        Get a resilient database transaction with retry logic.
        
        Yields:
            Database connection within transaction with retry capabilities
        """
        def get_transaction():
            return self.pool.get_transaction()
        
        transaction_manager = self.execute_with_retry(get_transaction)
        
        try:
            with transaction_manager as conn:
                yield ResilientConnection(conn, self)
        except Exception as e:
            error_type = self.classifier.classify_error(e)
            if self.classifier.should_invalidate_connection(e):
                self._handle_connection_invalidation()
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current error and retry metrics."""
        with self._lock:
            return {
                'total_errors': self.metrics.total_errors,
                'errors_by_type': {k.value: v for k, v in self.metrics.errors_by_type.items()},
                'consecutive_failures': self.metrics.consecutive_failures,
                'last_error_time': self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
                'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                'retry_attempts': self.metrics.retry_attempts,
                'retry_successes': self.metrics.retry_successes,
                'retry_success_rate': self.metrics.retry_successes / max(1, self.metrics.retry_attempts),
                'circuit_breaker': self.circuit_breaker.get_state()
            }
    
    def reset_metrics(self) -> None:
        """Reset error and retry metrics."""
        with self._lock:
            self.metrics = ErrorMetrics()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check with retry logic."""
        def ping():
            with self.pool.get_connection() as conn:
                if not self.pool._driver:
                    raise DatabaseError("Database driver not initialized")
                result = conn.execute(text(self.pool._driver.get_health_check_query()))
                return result.scalar()
        
        try:
            start_time = time.time()
            result = self.execute_with_retry(ping)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'result': result,
                'metrics': self.get_metrics()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'error_type': self.classifier.classify_error(e).value,
                'metrics': self.get_metrics()
            }


class ResilientConnection:
    """Wrapper for database connection with retry capabilities."""
    
    def __init__(self, connection: Connection, retry_handler: DatabaseRetryHandler):
        self.connection = connection
        self.retry_handler = retry_handler
    
    def execute(self, statement, parameters=None):
        """Execute statement with retry logic."""
        def execute_stmt():
            return self.connection.execute(statement, parameters)
        
        return self.retry_handler.execute_with_retry(execute_stmt)
    
    def scalar(self, statement, parameters=None):
        """Execute scalar query with retry logic."""
        def execute_scalar():
            return self.connection.scalar(statement, parameters)
        
        return self.retry_handler.execute_with_retry(execute_scalar)
    
    def __getattr__(self, name):
        """Delegate other attributes to underlying connection."""
        return getattr(self.connection, name)


def retry_database_operation(retry_config: Optional[RetryConfig] = None,
                           circuit_config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for database operations with retry logic.
    
    Args:
        retry_config: Retry configuration
        circuit_config: Circuit breaker configuration
        
    Returns:
        Decorated function with retry capabilities
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract pool from first argument if it's a method
            pool = None
            if args and hasattr(args[0], 'pool'):
                pool = args[0].pool
            elif args and isinstance(args[0], AdvancedConnectionPool):
                pool = args[0]
            
            if pool is None:
                raise ValueError("Cannot determine database pool for retry operation")
            
            # Create retry handler
            retry_handler = DatabaseRetryHandler(
                pool=pool,
                retry_config=retry_config,
                circuit_config=circuit_config
            )
            
            # Execute with retry
            return retry_handler.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


class DatabaseRetryManager(LoggerMixin):
    """Manager for multiple database retry handlers."""
    
    def __init__(self):
        self._handlers: Dict[str, DatabaseRetryHandler] = {}
        self._lock = threading.Lock()
    
    def add_handler(self, name: str, handler: DatabaseRetryHandler) -> None:
        """Add retry handler."""
        with self._lock:
            self._handlers[name] = handler
            self.logger.info(f"Added retry handler: {name}")
    
    def get_handler(self, name: str) -> Optional[DatabaseRetryHandler]:
        """Get retry handler by name."""
        with self._lock:
            return self._handlers.get(name)
    
    def remove_handler(self, name: str) -> None:
        """Remove retry handler."""
        with self._lock:
            if name in self._handlers:
                del self._handlers[name]
                self.logger.info(f"Removed retry handler: {name}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all handlers."""
        with self._lock:
            return {
                name: handler.get_metrics()
                for name, handler in self._handlers.items()
            }
    
    def reset_all_metrics(self) -> None:
        """Reset metrics for all handlers."""
        with self._lock:
            for handler in self._handlers.values():
                handler.reset_metrics()
    
    def health_check_all(self) -> Dict[str, Any]:
        """Health check for all handlers."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self._handlers)) as executor:
            future_to_name = {
                executor.submit(handler.health_check): name
                for name, handler in self._handlers.items()
            }
            
            for future in future_to_name:
                name = future_to_name[future]
                try:
                    results[name] = future.result(timeout=30)
                except Exception as e:
                    results[name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return results