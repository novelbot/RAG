"""
Unit tests for Database Health Monitor module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from src.database.health import DatabaseHealthChecker, HealthCheckManager, HealthStatus, HealthCheckResult
from src.core.exceptions import DatabaseError, HealthCheckError


class TestHealthStatus:
    """Test HealthStatus enum."""
    
    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""
    
    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        check = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            response_time=0.001,
            timestamp=datetime.now(timezone.utc),
            message="Connection OK"
        )
        
        assert check.status == HealthStatus.HEALTHY
        assert check.response_time == 0.001
        assert check.message == "Connection OK"
        assert isinstance(check.timestamp, datetime)

    def test_health_check_result_with_error(self):
        """Test HealthCheckResult with error."""
        check = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            response_time=0.5,
            timestamp=datetime.now(timezone.utc),
            message="Connection failed",
            error="Timeout occurred"
        )
        
        assert check.status == HealthStatus.UNHEALTHY
        assert check.error == "Timeout occurred"

    def test_health_check_result_with_details(self):
        """Test HealthCheckResult with details."""
        timestamp = datetime.now(timezone.utc)
        check = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            response_time=0.001,
            timestamp=timestamp,
            message="OK",
            details={"pool_size": 10, "active_connections": 5}
        )
        
        assert check.details["pool_size"] == 10
        assert check.details["active_connections"] == 5


class TestDatabaseHealthChecker:
    """Test DatabaseHealthChecker class."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool."""
        pool = Mock()
        pool._driver = Mock()
        pool._driver.get_health_check_query.return_value = "SELECT 1"
        return pool
    
    @pytest.fixture
    def mock_config(self):
        """Mock database config."""
        from src.core.config import DatabaseConfig
        config = Mock(spec=DatabaseConfig)
        config.host = "localhost"
        config.port = 5432
        config.name = "testdb"
        return config
    
    def test_health_checker_creation(self, mock_pool, mock_config):
        """Test DatabaseHealthChecker creation."""
        checker = DatabaseHealthChecker(mock_pool, mock_config)
        
        assert checker.pool == mock_pool
        assert checker.config == mock_config
        assert checker.driver == mock_pool._driver
        assert len(checker._health_history) == 0

    @patch('time.time')
    def test_ping_connection_success(self, mock_time, mock_pool, mock_config):
        """Test successful ping connection."""
        mock_time.side_effect = [1.0, 1.001]  # 1ms response time
        
        # Setup mock connection
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        checker = DatabaseHealthChecker(mock_pool, mock_config)
        result = checker.ping_connection()
        
        assert result.status == HealthStatus.HEALTHY
        assert abs(result.response_time - 0.001) < 0.0001  # Allow for floating point precision
        assert "Ping successful" in result.message

    @patch('time.time')
    def test_ping_connection_slow(self, mock_time, mock_pool, mock_config):
        """Test slow ping connection."""
        mock_time.side_effect = [1.0, 7.0]  # 6s response time (over 5s threshold)
        
        # Setup mock connection
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        checker = DatabaseHealthChecker(mock_pool, mock_config)
        result = checker.ping_connection()
        
        assert result.status == HealthStatus.DEGRADED
        assert result.response_time == 6.0
        assert "Slow response" in result.message

    def test_ping_connection_failure(self, mock_pool, mock_config):
        """Test ping connection failure."""
        # Setup mock connection to raise exception
        mock_pool.get_connection.side_effect = OperationalError("Connection failed", None, None)
        
        checker = DatabaseHealthChecker(mock_pool, mock_config)
        result = checker.ping_connection()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.error

    def test_ping_connection_wrong_result(self, mock_pool, mock_config):
        """Test ping connection with wrong result."""
        # Setup mock connection with wrong result
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 2  # Wrong value, expected 1
        mock_connection.execute.return_value = mock_result
        mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)
        
        checker = DatabaseHealthChecker(mock_pool, mock_config)
        result = checker.ping_connection()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Unexpected ping result" in result.error


class TestHealthCheckManager:
    """Test HealthCheckManager class."""
    
    @pytest.fixture
    def mock_checker(self):
        """Mock health checker."""
        checker = Mock()
        return checker
    
    def test_health_check_manager_creation(self):
        """Test HealthCheckManager creation."""
        manager = HealthCheckManager()
        
        assert len(manager._checkers) == 0

    def test_health_check_manager_add_checker(self, mock_checker):
        """Test adding health checker."""
        manager = HealthCheckManager()
        manager.add_checker("test_db", mock_checker)
        
        assert len(manager._checkers) == 1
        assert manager._checkers["test_db"] == mock_checker

    def test_health_check_manager_remove_checker(self, mock_checker):
        """Test removing health checker."""
        manager = HealthCheckManager()
        manager.add_checker("test_db", mock_checker)
        
        assert len(manager._checkers) == 1
        
        manager.remove_checker("test_db")
        
        assert len(manager._checkers) == 0

    def test_health_check_manager_get_checker(self, mock_checker):
        """Test getting health checker."""
        manager = HealthCheckManager()
        manager.add_checker("test_db", mock_checker)
        
        retrieved_checker = manager.get_checker("test_db")
        assert retrieved_checker == mock_checker
        
        # Test non-existent checker
        non_existent = manager.get_checker("non_existent")
        assert non_existent is None

    def test_health_check_manager_multiple_checkers(self):
        """Test manager with multiple checkers."""
        checker1 = Mock()
        checker2 = Mock()
        
        manager = HealthCheckManager()
        manager.add_checker("db1", checker1)
        manager.add_checker("db2", checker2)
        
        assert len(manager._checkers) == 2
        assert manager._checkers["db1"] == checker1
        assert manager._checkers["db2"] == checker2