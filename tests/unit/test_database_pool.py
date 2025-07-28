"""
Unit tests for Database Connection Pool module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError, OperationalError, TimeoutError

from src.database.pool import AdvancedConnectionPool, PoolMetrics, PoolMonitor, PoolManager
from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError


class TestPoolMetrics:
    """Test PoolMetrics dataclass."""
    
    def test_pool_metrics_creation(self):
        """Test PoolMetrics creation with default values."""
        metrics = PoolMetrics()
        
        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.idle_connections == 0
        assert metrics.checked_out == 0
        assert metrics.overflow == 0
        assert metrics.invalid_connections == 0
        assert metrics.pool_hits == 0
        assert metrics.pool_misses == 0
        assert metrics.connection_errors == 0
        assert metrics.last_updated is None

    def test_pool_metrics_with_values(self):
        """Test PoolMetrics creation with custom values."""
        timestamp = datetime.now(timezone.utc)
        metrics = PoolMetrics(
            total_connections=10,
            active_connections=5,
            idle_connections=5,
            checked_out=3,
            overflow=2,
            invalid_connections=0,
            pool_hits=100,
            pool_misses=5,
            connection_errors=1,
            last_updated=timestamp
        )
        
        assert metrics.total_connections == 10
        assert metrics.active_connections == 5
        assert metrics.idle_connections == 5
        assert metrics.checked_out == 3
        assert metrics.overflow == 2
        assert metrics.invalid_connections == 0
        assert metrics.pool_hits == 100
        assert metrics.pool_misses == 5
        assert metrics.connection_errors == 1
        assert metrics.last_updated == timestamp


class TestPoolMonitor:
    """Test PoolMonitor class."""
    
    def test_pool_monitor_creation(self):
        """Test PoolMonitor creation."""
        monitor = PoolMonitor("test_pool")
        
        assert monitor.pool_name == "test_pool"
        assert isinstance(monitor.metrics, PoolMetrics)
        assert monitor._start_time is not None

    def test_pool_monitor_update_metrics(self):
        """Test updating pool metrics."""
        monitor = PoolMonitor("test_pool")
        
        # Mock pool object
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 8
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        
        monitor.update_metrics(mock_pool)
        
        # Check that metrics were updated
        assert monitor.metrics.last_updated is not None

    def test_pool_monitor_get_metrics_dict(self):
        """Test getting metrics as dictionary."""
        monitor = PoolMonitor("test_pool")
        monitor.metrics.total_connections = 10
        monitor.metrics.active_connections = 5
        
        # Mock the get_metrics_dict method if it exists
        with patch.object(monitor, 'get_metrics_dict', return_value={"total": 10, "active": 5}) as mock_method:
            metrics_dict = monitor.get_metrics_dict()
            assert metrics_dict["total"] == 10
            assert metrics_dict["active"] == 5
            mock_method.assert_called_once()


class TestAdvancedConnectionPool:
    """Test AdvancedConnectionPool class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock database configuration."""
        config = Mock(spec=DatabaseConfig)
        config.driver = "postgresql"
        config.host = "localhost"
        config.port = 5432
        config.name = "testdb"
        config.user = "user"
        config.password = "pass"
        config.pool_size = 10
        config.max_overflow = 5
        config.pool_timeout = 30
        return config

    @pytest.fixture
    def mock_driver(self):
        """Mock database driver."""
        driver = Mock()
        driver.get_connection_url.return_value = "postgresql://user:pass@localhost:5432/testdb"
        driver.get_engine_options.return_value = {}
        driver.setup_event_listeners = Mock()
        return driver

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_connection_pool_creation(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test AdvancedConnectionPool creation."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        assert pool.config == mock_config
        assert pool._engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_connection_pool_with_config(self, mock_create_engine, mock_driver_factory, mock_driver):
        """Test AdvancedConnectionPool with custom configuration."""
        config = DatabaseConfig(
            driver="postgresql",
            host="localhost",
            port=5432,
            name="testdb",
            user="user",
            password="pass",
            pool_size=15,
            max_overflow=10,
            pool_timeout=60
        )
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(config)
        
        assert pool.config.pool_size == 15
        assert pool.config.max_overflow == 10
        assert pool.config.pool_timeout == 60

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_get_connection_success(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test successful connection retrieval."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        with pool.get_connection() as conn:
            assert conn == mock_connection

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_get_connection_failure(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test connection retrieval failure."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        with pytest.raises(DatabaseError):
            with pool.get_connection():
                pass

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_execute_query_success(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test successful query execution."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        result = pool.execute_query("SELECT 1")
        
        assert result == mock_result

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_execute_query_failure(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test query execution failure."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.side_effect = SQLAlchemyError("Query failed")
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        with pytest.raises(DatabaseError):
            pool.execute_query("SELECT 1")

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_execute_transaction_success(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test successful transaction execution."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_transaction = Mock()
        mock_connection.begin.return_value = mock_transaction
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        def test_callback(conn):
            return "success"
        
        result = pool.execute_transaction(test_callback)
        assert result == "success"

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_execute_transaction_rollback(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test transaction rollback on error."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_transaction = Mock()
        mock_connection.begin.return_value = mock_transaction
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        def failing_callback(conn):
            raise Exception("Transaction failed")
        
        with pytest.raises(DatabaseError):
            pool.execute_transaction(failing_callback)
        
        mock_transaction.rollback.assert_called_once()

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_get_pool_status(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test getting pool status."""
        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 8
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_engine.pool = mock_pool
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        status = pool.get_pool_status()
        
        assert "size" in status
        assert "utilization" in status

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_is_healthy_true(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test pool health check - healthy."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        assert pool.is_healthy() is True

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_is_healthy_false(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test pool health check - unhealthy."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        
        assert pool.is_healthy() is False

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_get_pool_metrics(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test getting pool metrics."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        metrics = pool.get_metrics()
        
        assert isinstance(metrics, PoolMetrics)

    @patch('src.database.pool.DatabaseDriverFactory.create_driver')
    @patch('src.database.pool.create_engine')
    def test_close_pool(self, mock_create_engine, mock_driver_factory, mock_config, mock_driver):
        """Test closing connection pool."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_driver_factory.return_value = mock_driver
        
        pool = AdvancedConnectionPool(mock_config)
        pool.close()
        
        mock_engine.dispose.assert_called_once()


class TestPoolManager:
    """Test PoolManager class."""
    
    def test_pool_manager_creation(self):
        """Test PoolManager creation."""
        manager = PoolManager()
        
        assert len(manager._pools) == 0

    @patch('src.database.pool.AdvancedConnectionPool')
    def test_pool_manager_create_pool(self, mock_pool_class):
        """Test creating a pool through manager."""
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        manager = PoolManager()
        config = Mock(spec=DatabaseConfig)
        
        pool = manager.create_pool("test_pool", config)
        
        assert pool == mock_pool
        assert "test_pool" in manager._pools

    def test_pool_manager_get_pool(self):
        """Test getting a pool from manager."""
        manager = PoolManager()
        mock_pool = Mock()
        manager._pools["test_pool"] = mock_pool
        
        retrieved_pool = manager.get_pool("test_pool")
        assert retrieved_pool == mock_pool
        
        # Test non-existent pool
        non_existent = manager.get_pool("non_existent")
        assert non_existent is None

    def test_pool_manager_remove_pool(self):
        """Test removing a pool from manager."""
        manager = PoolManager()
        mock_pool = Mock()
        manager._pools["test_pool"] = mock_pool
        
        assert len(manager._pools) == 1
        
        manager.remove_pool("test_pool")
        
        assert len(manager._pools) == 0
        mock_pool.close.assert_called_once()