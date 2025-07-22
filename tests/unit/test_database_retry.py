"""
Unit tests for Database Retry and Error Handling functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
from sqlalchemy import exc

from src.database.retry import (
    DatabaseErrorClassifier,
    ErrorType,
    RetryConfig,
    RetryStrategy,
    RetryDelayCalculator,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    DatabaseRetryHandler,
    ResilientConnection,
    DatabaseRetryManager,
    ErrorMetrics,
    retry_database_operation
)
from src.core.exceptions import DatabaseError, RetryError, CircuitBreakerError


class TestErrorType:
    """Test ErrorType enum."""
    
    def test_error_type_values(self):
        """Test ErrorType enum values."""
        assert ErrorType.CONNECTION_LOST.value == "connection_lost"
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.DEADLOCK.value == "deadlock"
        assert ErrorType.CONSTRAINT_VIOLATION.value == "constraint_violation"
        assert ErrorType.INVALID_TRANSACTION.value == "invalid_transaction"
        assert ErrorType.PERMISSION_DENIED.value == "permission_denied"
        assert ErrorType.RESOURCE_EXHAUSTED.value == "resource_exhausted"
        assert ErrorType.UNKNOWN.value == "unknown"


class TestRetryConfig:
    """Test RetryConfig dataclass."""
    
    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert ErrorType.CONNECTION_LOST in config.retryable_errors
        assert ErrorType.TIMEOUT in config.retryable_errors
        assert ErrorType.DEADLOCK in config.retryable_errors
        assert ErrorType.RESOURCE_EXHAUSTED in config.retryable_errors

    def test_retry_config_custom(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5,
            strategy=RetryStrategy.LINEAR,
            base_delay=2.0,
            max_delay=60.0,
            backoff_multiplier=3.0,
            jitter=False,
            retryable_errors=[ErrorType.CONNECTION_LOST, ErrorType.TIMEOUT]
        )
        
        assert config.max_retries == 5
        assert config.strategy == RetryStrategy.LINEAR
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False
        assert len(config.retryable_errors) == 2
        assert ErrorType.CONNECTION_LOST in config.retryable_errors
        assert ErrorType.TIMEOUT in config.retryable_errors


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""
    
    def test_circuit_breaker_config_defaults(self):
        """Test CircuitBreakerConfig default values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2

    def test_circuit_breaker_config_custom(self):
        """Test CircuitBreakerConfig with custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            half_open_max_calls=5,
            success_threshold=3
        )
        
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120.0
        assert config.half_open_max_calls == 5
        assert config.success_threshold == 3


class TestErrorMetrics:
    """Test ErrorMetrics dataclass."""
    
    def test_error_metrics_defaults(self):
        """Test ErrorMetrics default values."""
        metrics = ErrorMetrics()
        
        assert metrics.total_errors == 0
        assert metrics.errors_by_type == {}
        assert metrics.consecutive_failures == 0
        assert metrics.last_error_time is None
        assert metrics.last_success_time is None
        assert metrics.retry_attempts == 0
        assert metrics.retry_successes == 0


class TestDatabaseErrorClassifier:
    """Test DatabaseErrorClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create DatabaseErrorClassifier."""
        return DatabaseErrorClassifier()

    def test_classifier_creation(self, classifier):
        """Test DatabaseErrorClassifier creation."""
        assert classifier._error_patterns is not None
        assert classifier._retryable_exceptions is not None
        assert len(classifier._error_patterns) == 7  # All ErrorType values except UNKNOWN

    def test_classify_connection_lost_by_message(self, classifier):
        """Test classification of connection lost errors by message."""
        errors = [
            Exception("Lost connection to MySQL server"),
            Exception("Connection refused"),
            Exception("Connection timed out"),
            Exception("server closed the connection unexpectedly"),
            Exception("ORA-03113: end-of-file on communication channel")
        ]
        
        for error in errors:
            error_type = classifier.classify_error(error)
            assert error_type == ErrorType.CONNECTION_LOST

    def test_classify_timeout_by_message(self, classifier):
        """Test classification of timeout errors by message."""
        errors = [
            Exception("timeout occurred"),
            Exception("Query was cancelled"),
            Exception("Lock wait timeout exceeded"),
            Exception("ORA-01013: user requested cancel of current operation")
        ]
        
        for error in errors:
            error_type = classifier.classify_error(error)
            assert error_type == ErrorType.TIMEOUT

    def test_classify_deadlock_by_message(self, classifier):
        """Test classification of deadlock errors by message."""
        errors = [
            Exception("deadlock found"),
            Exception("Transaction deadlock"),
            Exception("ORA-00060: deadlock detected while waiting for resource"),
            Exception("40001: serialization failure")
        ]
        
        for error in errors:
            error_type = classifier.classify_error(error)
            assert error_type == ErrorType.DEADLOCK

    def test_classify_constraint_violation_by_message(self, classifier):
        """Test classification of constraint violation errors by message."""
        errors = [
            Exception("UNIQUE constraint failed"),
            Exception("foreign key constraint"),
            Exception("IntegrityError occurred"),
            Exception("ORA-00001: unique constraint violated")
        ]
        
        for error in errors:
            error_type = classifier.classify_error(error)
            assert error_type == ErrorType.CONSTRAINT_VIOLATION

    def test_classify_by_exception_type(self, classifier):
        """Test classification by exception type."""
        test_cases = [
            (exc.DisconnectionError("test"), ErrorType.CONNECTION_LOST),
            (exc.TimeoutError("test"), ErrorType.TIMEOUT),
            (exc.IntegrityError("test", None, None), ErrorType.CONSTRAINT_VIOLATION),
            (exc.InvalidRequestError("test"), ErrorType.INVALID_TRANSACTION),
            (ConnectionError("test"), ErrorType.CONNECTION_LOST),
            (BrokenPipeError("test"), ErrorType.CONNECTION_LOST),
        ]
        
        for error, expected_type in test_cases:
            error_type = classifier.classify_error(error)
            assert error_type == expected_type

    def test_classify_unknown_error(self, classifier):
        """Test classification of unknown errors."""
        error = Exception("some unknown error message")
        error_type = classifier.classify_error(error)
        assert error_type == ErrorType.UNKNOWN

    def test_is_retryable_true(self, classifier):
        """Test retryable error detection."""
        retryable_errors = [ErrorType.CONNECTION_LOST, ErrorType.TIMEOUT]
        
        # Connection lost error should be retryable
        error = exc.DisconnectionError("connection lost")
        assert classifier.is_retryable(error, retryable_errors) is True
        
        # Timeout error should be retryable
        error = Exception("timeout occurred")
        assert classifier.is_retryable(error, retryable_errors) is True

    def test_is_retryable_false(self, classifier):
        """Test non-retryable error detection."""
        retryable_errors = [ErrorType.CONNECTION_LOST, ErrorType.TIMEOUT]
        
        # Constraint violation should not be retryable
        error = exc.IntegrityError("unique constraint", None, None)
        assert classifier.is_retryable(error, retryable_errors) is False
        
        # Unknown error should not be retryable
        error = Exception("unknown error")
        assert classifier.is_retryable(error, retryable_errors) is False

    def test_should_invalidate_connection_true(self, classifier):
        """Test connection invalidation decision - true cases."""
        invalidating_errors = [
            exc.DisconnectionError("connection lost"),
            exc.InvalidRequestError("invalid transaction"),
            Exception("out of memory")
        ]
        
        for error in invalidating_errors:
            assert classifier.should_invalidate_connection(error) is True

    def test_should_invalidate_connection_false(self, classifier):
        """Test connection invalidation decision - false cases."""
        non_invalidating_errors = [
            exc.IntegrityError("constraint violation", None, None),
            Exception("deadlock found"),
            Exception("permission denied")
        ]
        
        for error in non_invalidating_errors:
            assert classifier.should_invalidate_connection(error) is False


class TestRetryDelayCalculator:
    """Test RetryDelayCalculator class."""
    
    def test_linear_strategy(self):
        """Test linear retry delay calculation."""
        config = RetryConfig(strategy=RetryStrategy.LINEAR, base_delay=2.0, jitter=False)
        calculator = RetryDelayCalculator(config)
        
        assert calculator.calculate_delay(0) == 2.0  # base_delay * 1
        assert calculator.calculate_delay(1) == 4.0  # base_delay * 2
        assert calculator.calculate_delay(2) == 6.0  # base_delay * 3

    def test_exponential_strategy(self):
        """Test exponential retry delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        calculator = RetryDelayCalculator(config)
        
        assert calculator.calculate_delay(0) == 1.0  # base_delay * (2^0)
        assert calculator.calculate_delay(1) == 2.0  # base_delay * (2^1)
        assert calculator.calculate_delay(2) == 4.0  # base_delay * (2^2)
        assert calculator.calculate_delay(3) == 8.0  # base_delay * (2^3)

    def test_fibonacci_strategy(self):
        """Test Fibonacci retry delay calculation."""
        config = RetryConfig(strategy=RetryStrategy.FIBONACCI, base_delay=1.0, jitter=False)
        calculator = RetryDelayCalculator(config)
        
        assert calculator.calculate_delay(0) == 1.0  # base_delay * fib(0) = 1 * 1
        assert calculator.calculate_delay(1) == 1.0  # base_delay * fib(1) = 1 * 1
        assert calculator.calculate_delay(2) == 2.0  # base_delay * fib(2) = 1 * 2
        assert calculator.calculate_delay(3) == 3.0  # base_delay * fib(3) = 1 * 3
        assert calculator.calculate_delay(4) == 5.0  # base_delay * fib(4) = 1 * 5

    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=False
        )
        calculator = RetryDelayCalculator(config)
        
        # Large attempt should be capped at max_delay
        assert calculator.calculate_delay(10) == 10.0

    def test_jitter_application(self):
        """Test jitter application."""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            base_delay=10.0,
            jitter=True
        )
        calculator = RetryDelayCalculator(config)
        
        # With jitter, delays should vary
        delays = [calculator.calculate_delay(0) for _ in range(10)]
        
        # All delays should be around base_delay but not exactly the same
        assert all(9.0 <= delay <= 11.0 for delay in delays)
        assert len(set(delays)) > 1  # Should have some variation

    def test_fibonacci_caching(self):
        """Test Fibonacci number caching."""
        config = RetryConfig(strategy=RetryStrategy.FIBONACCI, base_delay=1.0, jitter=False)
        calculator = RetryDelayCalculator(config)
        
        # First call should build cache
        calculator.calculate_delay(5)
        cache_size_after_first = len(calculator._fibonacci_cache)
        
        # Second call should use cached values
        calculator.calculate_delay(3)
        cache_size_after_second = len(calculator._fibonacci_cache)
        
        # Cache should not grow for smaller indices
        assert cache_size_after_second == cache_size_after_first


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    @pytest.fixture
    def circuit_config(self):
        """Circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_max_calls=2,
            success_threshold=2
        )

    @pytest.fixture
    def circuit_breaker(self, circuit_config):
        """Create CircuitBreaker."""
        return CircuitBreaker("test_circuit", circuit_config)

    def test_circuit_breaker_creation(self, circuit_breaker, circuit_config):
        """Test CircuitBreaker creation."""
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.config == circuit_config
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_successful_operation_closed_state(self, circuit_breaker):
        """Test successful operation in closed state."""
        def success_func():
            return "success"
        
        result = circuit_breaker.call(success_func)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_failed_operation_closed_state(self, circuit_breaker):
        """Test failed operation in closed state."""
        def failing_func():
            raise Exception("test error")
        
        # First few failures should keep circuit closed
        for i in range(2):
            with pytest.raises(Exception, match="test error"):
                circuit_breaker.call(failing_func)
            assert circuit_breaker.state == CircuitState.CLOSED
            assert circuit_breaker.failure_count == i + 1

    def test_circuit_opens_on_threshold(self, circuit_breaker):
        """Test circuit opens when failure threshold is reached."""
        def failing_func():
            raise Exception("test error")
        
        # Reach failure threshold
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 3

    def test_circuit_blocks_calls_when_open(self, circuit_breaker):
        """Test circuit blocks calls when open."""
        def failing_func():
            raise Exception("test error")
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        # Now calls should be blocked
        def any_func():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerError, match="Circuit breaker test_circuit is OPEN"):
            circuit_breaker.call(any_func)

    def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after recovery timeout."""
        def failing_func():
            raise Exception("test error")
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        # Mock time passage
        with patch('src.database.retry.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(seconds=10)
            
            def success_func():
                return "success"
            
            result = circuit_breaker.call(success_func)
            
            assert result == "success"
            assert circuit_breaker.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self, circuit_breaker):
        """Test successful operations in half-open state close the circuit."""
        # Manually set to half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = 0
        circuit_breaker.half_open_successes = 0
        
        def success_func():
            return "success"
        
        # Make successful calls to meet success threshold
        for i in range(2):
            result = circuit_breaker.call(success_func)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_half_open_failure_opens_circuit(self, circuit_breaker):
        """Test failure in half-open state opens the circuit."""
        # Manually set to half-open state
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = 0
        circuit_breaker.half_open_successes = 0
        
        def failing_func():
            raise Exception("half-open failure")
        
        with pytest.raises(Exception, match="half-open failure"):
            circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == CircuitState.OPEN

    def test_half_open_call_limit(self, circuit_breaker):
        """Test half-open call limit enforcement."""
        # Manually set to half-open state with max calls reached
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = 2  # At limit
        circuit_breaker.half_open_successes = 0
        
        def any_func():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerError, match="HALF_OPEN call limit exceeded"):
            circuit_breaker.call(any_func)

    def test_get_state(self, circuit_breaker):
        """Test circuit breaker state retrieval."""
        state = circuit_breaker.get_state()
        
        assert state['name'] == "test_circuit"
        assert state['state'] == CircuitState.CLOSED.value
        assert state['failure_count'] == 0
        assert state['half_open_calls'] == 0
        assert state['half_open_successes'] == 0
        assert state['last_failure_time'] is None


class TestDatabaseRetryHandler:
    """Test DatabaseRetryHandler class."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool."""
        pool = Mock()
        pool.pool_name = "test_pool"
        pool.engine = Mock()
        pool._driver = Mock()
        pool._driver.get_health_check_query.return_value = "SELECT 1"
        return pool

    @pytest.fixture
    def retry_config(self):
        """Retry configuration."""
        return RetryConfig(max_retries=2, base_delay=0.1, jitter=False)

    @pytest.fixture
    def circuit_config(self):
        """Circuit breaker configuration."""
        return CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)

    @pytest.fixture
    def retry_handler(self, mock_pool, retry_config, circuit_config):
        """Create DatabaseRetryHandler."""
        with patch('src.database.retry.event'):
            return DatabaseRetryHandler(mock_pool, retry_config, circuit_config)

    def test_retry_handler_creation(self, retry_handler, mock_pool, retry_config, circuit_config):
        """Test DatabaseRetryHandler creation."""
        assert retry_handler.pool == mock_pool
        assert retry_handler.retry_config == retry_config
        assert retry_handler.circuit_config == circuit_config
        assert retry_handler.classifier is not None
        assert retry_handler.delay_calculator is not None
        assert retry_handler.circuit_breaker is not None

    def test_execute_with_retry_success_first_attempt(self, retry_handler):
        """Test successful execution on first attempt."""
        def success_func():
            return "success"
        
        result = retry_handler.execute_with_retry(success_func)
        
        assert result == "success"
        assert retry_handler.metrics.retry_attempts == 1
        assert retry_handler.metrics.consecutive_failures == 0

    def test_execute_with_retry_success_after_failures(self, retry_handler):
        """Test successful execution after some failures."""
        attempt_count = 0
        
        def sometimes_failing_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise exc.DisconnectionError("connection lost")
            return "success"
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = retry_handler.execute_with_retry(sometimes_failing_func)
        
        assert result == "success"
        assert retry_handler.metrics.retry_attempts == 2
        assert retry_handler.metrics.retry_successes == 1

    def test_execute_with_retry_all_attempts_fail(self, retry_handler):
        """Test retry exhaustion when all attempts fail."""
        def always_failing_func():
            raise exc.DisconnectionError("connection lost")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(RetryError, match="Operation failed after 3 attempts"):
                retry_handler.execute_with_retry(always_failing_func)
        
        assert retry_handler.metrics.consecutive_failures == 1

    def test_execute_with_retry_non_retryable_error(self, retry_handler):
        """Test immediate failure for non-retryable errors."""
        def non_retryable_func():
            raise exc.IntegrityError("constraint violation", None, None)
        
        # Should not retry constraint violations
        with pytest.raises(RetryError):
            retry_handler.execute_with_retry(non_retryable_func)
        
        # Should only have one attempt
        assert retry_handler.metrics.retry_attempts == 1

    @patch('time.sleep')
    def test_retry_delay_calculation(self, mock_sleep, retry_handler):
        """Test retry delay calculation and application."""
        attempt_count = 0
        
        def failing_func():
            nonlocal attempt_count
            attempt_count += 1
            raise exc.DisconnectionError("connection lost")
        
        with pytest.raises(RetryError):
            retry_handler.execute_with_retry(failing_func)
        
        # Should have called sleep for retry delays
        assert mock_sleep.call_count == 2  # max_retries

    def test_connection_invalidation_handling(self, retry_handler):
        """Test connection invalidation on specific errors."""
        def invalidating_func():
            raise exc.DisconnectionError("connection lost")
        
        retry_handler.pool.invalidate_pool = Mock()
        
        with patch('time.sleep'):
            with pytest.raises(RetryError):
                retry_handler.execute_with_retry(invalidating_func)
        
        # Should have attempted to invalidate pool
        assert retry_handler.pool.invalidate_pool.call_count >= 1

    def test_resilient_connection_context_manager(self, retry_handler):
        """Test resilient connection context manager."""
        mock_connection = Mock()
        mock_connection_manager = Mock()
        mock_connection_manager.__enter__ = Mock(return_value=mock_connection)
        mock_connection_manager.__exit__ = Mock(return_value=None)
        
        retry_handler.pool.get_connection.return_value = mock_connection_manager
        
        with retry_handler.resilient_connection() as conn:
            assert isinstance(conn, ResilientConnection)
            assert conn.connection == mock_connection
            assert conn.retry_handler == retry_handler

    def test_resilient_transaction_context_manager(self, retry_handler):
        """Test resilient transaction context manager."""
        mock_connection = Mock()
        mock_transaction_manager = Mock()
        mock_transaction_manager.__enter__ = Mock(return_value=mock_connection)
        mock_transaction_manager.__exit__ = Mock(return_value=None)
        
        retry_handler.pool.get_transaction.return_value = mock_transaction_manager
        
        with retry_handler.resilient_transaction() as conn:
            assert isinstance(conn, ResilientConnection)
            assert conn.connection == mock_connection
            assert conn.retry_handler == retry_handler

    def test_get_metrics(self, retry_handler):
        """Test metrics retrieval."""
        # Simulate some activity
        retry_handler.metrics.total_errors = 5
        retry_handler.metrics.retry_attempts = 10
        retry_handler.metrics.retry_successes = 7
        
        metrics = retry_handler.get_metrics()
        
        assert metrics['total_errors'] == 5
        assert metrics['retry_attempts'] == 10
        assert metrics['retry_successes'] == 7
        assert metrics['retry_success_rate'] == 0.7
        assert 'circuit_breaker' in metrics

    def test_reset_metrics(self, retry_handler):
        """Test metrics reset."""
        # Simulate some activity
        retry_handler.metrics.total_errors = 5
        retry_handler.metrics.retry_attempts = 10
        
        retry_handler.reset_metrics()
        
        assert retry_handler.metrics.total_errors == 0
        assert retry_handler.metrics.retry_attempts == 0

    def test_health_check_success(self, retry_handler):
        """Test successful health check."""
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=None)
        
        retry_handler.pool.get_connection.return_value = mock_connection
        
        result = retry_handler.health_check()
        
        assert result['status'] == 'healthy'
        assert 'response_time' in result
        assert result['result'] == 1
        assert 'metrics' in result

    def test_health_check_failure(self, retry_handler):
        """Test failed health check."""
        retry_handler.pool.get_connection.side_effect = Exception("connection failed")
        
        result = retry_handler.health_check()
        
        assert result['status'] == 'unhealthy'
        assert 'error' in result
        assert 'error_type' in result
        assert 'metrics' in result


class TestResilientConnection:
    """Test ResilientConnection class."""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock database connection."""
        return Mock()

    @pytest.fixture
    def mock_retry_handler(self):
        """Mock retry handler."""
        handler = Mock()
        handler.execute_with_retry.side_effect = lambda func: func()
        return handler

    @pytest.fixture
    def resilient_conn(self, mock_connection, mock_retry_handler):
        """Create ResilientConnection."""
        return ResilientConnection(mock_connection, mock_retry_handler)

    def test_resilient_connection_creation(self, resilient_conn, mock_connection, mock_retry_handler):
        """Test ResilientConnection creation."""
        assert resilient_conn.connection == mock_connection
        assert resilient_conn.retry_handler == mock_retry_handler

    def test_execute_with_retry(self, resilient_conn, mock_connection):
        """Test execute method with retry logic."""
        mock_result = Mock()
        mock_connection.execute.return_value = mock_result
        
        statement = "SELECT 1"
        result = resilient_conn.execute(statement)
        
        mock_connection.execute.assert_called_once_with(statement, None)
        assert result == mock_result

    def test_scalar_with_retry(self, resilient_conn, mock_connection):
        """Test scalar method with retry logic."""
        mock_connection.scalar.return_value = 42
        
        statement = "SELECT COUNT(*)"
        result = resilient_conn.scalar(statement)
        
        mock_connection.scalar.assert_called_once_with(statement, None)
        assert result == 42

    def test_attribute_delegation(self, resilient_conn, mock_connection):
        """Test attribute delegation to underlying connection."""
        mock_connection.some_attribute = "test_value"
        
        assert resilient_conn.some_attribute == "test_value"


class TestRetryDatabaseOperationDecorator:
    """Test retry_database_operation decorator."""
    
    def test_decorator_with_pool_method(self):
        """Test decorator on method with pool attribute."""
        class TestClass:
            def __init__(self):
                self.pool = Mock()
                self.pool.pool_name = "test_pool"
                self.pool.engine = Mock()
                self.pool._driver = Mock()
        
        obj = TestClass()
        
        @retry_database_operation()
        def test_method(self):
            return "success"
        
        # Bind method to object
        bound_method = test_method.__get__(obj, TestClass)
        
        with patch('src.database.retry.DatabaseRetryHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.execute_with_retry.return_value = "success"
            mock_handler_class.return_value = mock_handler
            
            result = bound_method()
            
            assert result == "success"
            mock_handler_class.assert_called_once()

    def test_decorator_with_pool_argument(self):
        """Test decorator with pool as first argument."""
        mock_pool = Mock()
        mock_pool.pool_name = "test_pool"
        mock_pool.engine = Mock()
        mock_pool._driver = Mock()
        
        @retry_database_operation()
        def test_function(pool):
            return "success"
        
        with patch('src.database.retry.DatabaseRetryHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.execute_with_retry.return_value = "success"
            mock_handler_class.return_value = mock_handler
            
            result = test_function(mock_pool)
            
            assert result == "success"
            mock_handler_class.assert_called_once()

    def test_decorator_no_pool_error(self):
        """Test decorator raises error when no pool found."""
        @retry_database_operation()
        def test_function():
            return "success"
        
        with pytest.raises(ValueError, match="Cannot determine database pool"):
            test_function()


class TestDatabaseRetryManager:
    """Test DatabaseRetryManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create DatabaseRetryManager."""
        return DatabaseRetryManager()

    @pytest.fixture
    def mock_handler(self):
        """Mock retry handler."""
        handler = Mock()
        handler.get_metrics.return_value = {'total_errors': 0}
        handler.health_check.return_value = {'status': 'healthy'}
        return handler

    def test_manager_creation(self, manager):
        """Test DatabaseRetryManager creation."""
        assert len(manager._handlers) == 0

    def test_add_handler(self, manager, mock_handler):
        """Test adding retry handler."""
        manager.add_handler("test_db", mock_handler)
        
        assert "test_db" in manager._handlers
        assert manager._handlers["test_db"] == mock_handler

    def test_get_handler(self, manager, mock_handler):
        """Test getting retry handler."""
        manager.add_handler("test_db", mock_handler)
        
        retrieved = manager.get_handler("test_db")
        assert retrieved == mock_handler
        
        # Test non-existent handler
        non_existent = manager.get_handler("non_existent")
        assert non_existent is None

    def test_remove_handler(self, manager, mock_handler):
        """Test removing retry handler."""
        manager.add_handler("test_db", mock_handler)
        manager.remove_handler("test_db")
        
        assert "test_db" not in manager._handlers

    def test_get_all_metrics(self, manager, mock_handler):
        """Test getting metrics for all handlers."""
        manager.add_handler("db1", mock_handler)
        manager.add_handler("db2", mock_handler)
        
        metrics = manager.get_all_metrics()
        
        assert "db1" in metrics
        assert "db2" in metrics
        assert metrics["db1"] == {'total_errors': 0}

    def test_reset_all_metrics(self, manager, mock_handler):
        """Test resetting metrics for all handlers."""
        manager.add_handler("db1", mock_handler)
        manager.add_handler("db2", mock_handler)
        
        manager.reset_all_metrics()
        
        # Should have called reset_metrics on all handlers
        assert mock_handler.reset_metrics.call_count == 2

    def test_health_check_all(self, manager, mock_handler):
        """Test health check for all handlers."""
        manager.add_handler("db1", mock_handler)
        manager.add_handler("db2", mock_handler)
        
        results = manager.health_check_all()
        
        assert "db1" in results
        assert "db2" in results
        assert results["db1"]["status"] == "healthy"
        assert results["db2"]["status"] == "healthy"