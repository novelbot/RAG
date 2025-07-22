"""
Unit tests for Database Base module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, ConfigurationError


class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_database_config_creation(self):
        """Test DatabaseConfig creation with required parameters."""
        config = DatabaseConfig(
            driver="postgresql",
            host="localhost",
            port=5432,
            name="testdb",
            user="user",
            password="pass"
        )
        
        assert config.driver == "postgresql"
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "testdb"
        assert config.user == "user"
        assert config.password == "pass"

    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig(
            driver="postgresql",
            host="localhost",
            name="testdb",
            user="user",
            password="pass"
        )
        
        # Should have default port
        assert config.port == 5432


@pytest.mark.parametrize("driver,expected_port", [
    ("postgresql", 5432),
    ("mysql", 3306),
    ("oracle", 1521),
])
def test_database_config_driver_defaults(driver, expected_port):
    """Test DatabaseConfig defaults for different drivers."""
    config = DatabaseConfig(
        driver=driver,
        host="localhost",
        name="testdb",
        user="user",
        password="pass"
    )
    
    assert config.driver == driver


class TestDatabaseManagerMocked:
    """Test DatabaseManager with mocked dependencies."""
    
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

    @patch('src.database.base.create_engine')
    def test_database_manager_creation(self, mock_create_engine, mock_config):
        """Test DatabaseManager creation."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        assert manager.config == mock_config
        assert manager._engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch('src.database.base.create_engine')
    def test_database_manager_connection_url_building(self, mock_create_engine, mock_config):
        """Test connection URL building."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        url = manager._build_connection_url()
        
        expected = "postgresql://user:pass@localhost:5432/testdb"
        assert url == expected

    @patch('src.database.base.create_engine')
    def test_database_manager_connection_url_no_password(self, mock_create_engine):
        """Test connection URL building without password."""
        from src.database.base import DatabaseManager
        
        config = Mock(spec=DatabaseConfig)
        config.driver = "postgresql"
        config.host = "localhost"
        config.port = 5432
        config.name = "testdb"
        config.user = "user"
        config.password = None
        config.pool_size = 10
        config.max_overflow = 5
        config.pool_timeout = 30
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(config)
        url = manager._build_connection_url()
        
        expected = "postgresql://user@localhost:5432/testdb"
        assert url == expected

    @patch('src.database.base.create_engine')
    def test_database_manager_test_connection_success(self, mock_create_engine, mock_config):
        """Test successful connection test."""
        from src.database.base import DatabaseManager
        from sqlalchemy import text
        
        mock_engine = Mock()
        mock_connection = Mock()
        # get_connection은 generator이므로 직접 connection을 반환하도록 설정
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        result = manager.test_connection()
        
        assert result is True
        # Verify execute was called with a text clause containing "SELECT 1"
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args[0]
        assert len(call_args) == 1
        assert str(call_args[0]) == "SELECT 1"

    @patch('src.database.base.create_engine')
    def test_database_manager_test_connection_failure(self, mock_create_engine, mock_config):
        """Test connection test failure."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        result = manager.test_connection()
        
        assert result is False

    @patch('src.database.base.create_engine')
    def test_database_manager_get_pool_status(self, mock_create_engine, mock_config):
        """Test pool status retrieval."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 10
        mock_pool.checkedin.return_value = 8
        mock_pool.checkedout.return_value = 2
        mock_pool.overflow.return_value = 0
        mock_pool.invalid.return_value = 0
        mock_engine.pool = mock_pool
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        status = manager.get_pool_status()
        
        assert status["size"] == 10
        assert status["checked_in"] == 8
        assert status["checked_out"] == 2
        assert status["overflow"] == 0
        assert status["invalid"] == 0

    @patch('src.database.base.create_engine')
    def test_database_manager_close(self, mock_create_engine, mock_config):
        """Test database manager closure."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        manager.close()
        
        mock_engine.dispose.assert_called_once()
        assert manager._engine is None

    @patch('src.database.base.create_engine')
    def test_database_manager_context_manager(self, mock_create_engine, mock_config):
        """Test DatabaseManager as context manager."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        with DatabaseManager(mock_config) as manager:
            assert manager._engine == mock_engine
        
        mock_engine.dispose.assert_called_once()


class TestDatabaseFactory:
    """Test DatabaseFactory class."""
    
    @patch('src.database.base.DatabaseManager')
    def test_factory_create_manager(self, mock_manager_class):
        """Test factory create manager."""
        from src.database.base import DatabaseFactory
        
        mock_config = Mock(spec=DatabaseConfig)
        mock_config.driver = "postgresql"
        mock_config.host = "localhost"
        mock_config.name = "testdb"
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = DatabaseFactory.create_manager(mock_config)
        
        assert result == mock_manager
        mock_manager_class.assert_called_once_with(mock_config)

    def test_factory_create_manager_missing_driver(self):
        """Test factory with missing driver."""
        from src.database.base import DatabaseFactory
        
        mock_config = Mock(spec=DatabaseConfig)
        mock_config.driver = None
        mock_config.host = "localhost"
        mock_config.name = "testdb"
        
        with pytest.raises(ConfigurationError, match="Database driver not specified"):
            DatabaseFactory.create_manager(mock_config)

    def test_factory_create_manager_missing_host(self):
        """Test factory with missing host."""
        from src.database.base import DatabaseFactory
        
        mock_config = Mock(spec=DatabaseConfig)
        mock_config.driver = "postgresql"
        mock_config.host = None
        mock_config.name = "testdb"
        
        with pytest.raises(ConfigurationError, match="Database host not specified"):
            DatabaseFactory.create_manager(mock_config)

    def test_factory_create_manager_missing_name(self):
        """Test factory with missing database name."""
        from src.database.base import DatabaseFactory
        
        mock_config = Mock(spec=DatabaseConfig)
        mock_config.driver = "postgresql"
        mock_config.host = "localhost"
        mock_config.name = None
        
        with pytest.raises(ConfigurationError, match="Database name not specified"):
            DatabaseFactory.create_manager(mock_config)

    @patch('src.database.base.DatabaseManager')
    def test_factory_create_from_url(self, mock_manager_class):
        """Test factory create from URL."""
        from src.database.base import DatabaseFactory
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = DatabaseFactory.create_from_url(
            "postgresql://user:pass@localhost:5432/testdb",
            driver="postgresql",
            host="localhost",
            port=5432,
            name="testdb",
            user="user",
            password="pass"
        )
        
        assert result == mock_manager
        mock_manager_class.assert_called_once()


class TestDatabaseExceptionHandling:
    """Test database exception handling."""
    
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
    
    @patch('src.database.base.create_engine')
    def test_engine_creation_failure(self, mock_create_engine, mock_config):
        """Test engine creation failure."""
        from src.database.base import DatabaseManager
        
        mock_create_engine.side_effect = Exception("Engine creation failed")
        
        with pytest.raises(DatabaseError, match="Engine creation failed"):
            DatabaseManager(mock_config)

    @patch('src.database.base.create_engine')
    def test_execute_query_failure(self, mock_create_engine, mock_config):
        """Test query execution failure."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.side_effect = SQLAlchemyError("Query failed")
        # get_connection is a generator, so mock the direct connection
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        with pytest.raises(DatabaseError, match="Connection error"):
            manager.execute_query("SELECT 1")

    @patch('src.database.base.create_engine')
    def test_transaction_rollback(self, mock_create_engine, mock_config):
        """Test transaction rollback on error."""
        from src.database.base import DatabaseManager
        
        mock_engine = Mock()
        mock_connection = Mock()
        mock_transaction = Mock()
        # connection.begin() directly returns the transaction object
        mock_connection.begin.return_value = mock_transaction
        mock_connection.execute.side_effect = SQLAlchemyError("Query failed")
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        with pytest.raises(DatabaseError, match="Transaction error"):
            manager.execute_transaction(["SELECT 1"])
        
        # Transaction should be rolled back
        mock_transaction.rollback.assert_called_once()