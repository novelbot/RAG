"""
Unit tests for Database Engine (DatabaseManager) module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from src.database.base import DatabaseManager, DatabaseFactory
from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, ConfigurationError


class TestDatabaseConfig:
    """Test DatabaseConfig class (used by DatabaseManager)."""
    
    def test_database_config_creation(self):
        """Test DatabaseConfig creation with valid parameters."""
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
        
        # Should have default port for PostgreSQL
        assert config.port == 5432

    def test_database_config_mysql_with_custom_port(self):
        """Test DatabaseConfig with MySQL driver and custom port."""
        config = DatabaseConfig(
            driver="mysql",
            host="localhost",
            port=3306,
            name="testdb",
            user="user",
            password="pass"
        )
        
        assert config.driver == "mysql"
        assert config.port == 3306

    def test_database_config_to_dict(self):
        """Test DatabaseConfig serialization."""
        config = DatabaseConfig(
            driver="postgresql",
            host="localhost",
            port=5432,
            name="testdb",
            user="user",
            password="pass",
            pool_size=10
        )
        
        config_dict = config.model_dump()
        
        assert config_dict["driver"] == "postgresql"
        assert config_dict["host"] == "localhost"
        assert config_dict["port"] == 5432
        assert config_dict["name"] == "testdb"
        assert config_dict["user"] == "user"
        assert config_dict["password"] == "pass"
        assert config_dict["pool_size"] == 10


class TestDatabaseManager:
    """Test DatabaseManager (Engine) class."""
    
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
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        assert manager.config == mock_config
        assert manager._engine == mock_engine
        mock_create_engine.assert_called_once()

    @patch('src.database.base.create_engine')
    def test_initialize_success(self, mock_create_engine, mock_config):
        """Test successful DatabaseManager initialization."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        # Manager should be initialized during construction
        assert manager._engine is not None
        assert manager.config == mock_config

    @patch('src.database.base.create_engine')
    def test_initialize_invalid_config(self, mock_create_engine):
        """Test DatabaseManager with invalid configuration."""
        mock_create_engine.side_effect = Exception("Invalid configuration")
        
        config = Mock(spec=DatabaseConfig)
        config.driver = "invalid_driver"
        config.host = "localhost"
        config.port = 5432
        config.name = "testdb"
        config.user = "user"
        config.password = "pass"
        config.pool_size = 10
        config.max_overflow = 5
        config.pool_timeout = 30
        
        with pytest.raises(DatabaseError, match="Engine creation failed"):
            DatabaseManager(config)

    @patch('src.database.base.create_engine')
    def test_get_connection_url(self, mock_create_engine, mock_config):
        """Test connection URL building."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        url = manager._build_connection_url()
        
        expected = "postgresql://user:pass@localhost:5432/testdb"
        assert url == expected

    @patch('src.database.base.create_engine')
    def test_get_connection_url_no_password(self, mock_create_engine):
        """Test connection URL building without password."""
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
    def test_test_connection_success(self, mock_create_engine, mock_config):
        """Test successful connection test."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        result = manager.test_connection()
        
        assert result is True

    @patch('src.database.base.create_engine')
    def test_test_connection_failure(self, mock_create_engine, mock_config):
        """Test connection test failure."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        result = manager.test_connection()
        
        assert result is False

    @patch('src.database.base.create_engine')
    def test_get_pool_status(self, mock_create_engine, mock_config):
        """Test getting pool status information."""
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
    def test_close(self, mock_create_engine, mock_config):
        """Test closing database manager."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        manager.close()
        
        mock_engine.dispose.assert_called_once()
        assert manager._engine is None

    @patch('src.database.base.create_engine')
    def test_engine_property_not_initialized(self, mock_create_engine):
        """Test engine property when not initialized."""
        # Create manager that fails initialization
        mock_create_engine.side_effect = Exception("Failed")
        
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
        
        with pytest.raises(DatabaseError):
            manager = DatabaseManager(config)

    @patch('src.database.base.create_engine')
    def test_engine_property_initialized(self, mock_create_engine, mock_config):
        """Test engine property when initialized."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        assert manager.engine == mock_engine

    @patch('src.database.base.create_engine')
    def test_metadata_property(self, mock_create_engine, mock_config):
        """Test metadata property."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        metadata = manager.metadata
        
        assert metadata is not None
        # Second call should return same metadata
        assert manager.metadata is metadata

    @patch('src.database.base.create_engine')
    def test_context_manager(self, mock_create_engine, mock_config):
        """Test DatabaseManager as context manager."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        with DatabaseManager(mock_config) as manager:
            assert manager._engine == mock_engine
        
        mock_engine.dispose.assert_called_once()

    def test_validate_config_all_defaults(self):
        """Test config validation with all default values."""
        # DatabaseConfig has default values for all fields
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "ragdb"
        assert config.user == "postgres"
        assert config.driver == "postgresql"

    def test_validate_config_invalid_driver(self):
        """Test config validation with invalid driver."""
        with pytest.raises(ValueError, match="Driver must be one of"):
            DatabaseConfig(
                driver="invalid_driver",
                host="localhost",
                name="testdb",
                user="user",
                password="pass"
            )

    @patch('src.database.base.create_engine')
    def test_create_engine_parameters(self, mock_create_engine, mock_config):
        """Test engine creation parameters."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager(mock_config)
        
        # Verify create_engine was called with correct parameters
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args
        
        # Check connection URL
        connection_url = call_args[0][0]
        assert "postgresql://" in connection_url
        assert "localhost:5432" in connection_url
        assert "testdb" in connection_url
        
        # Check engine parameters
        engine_params = call_args[1]
        assert engine_params["pool_size"] == 10
        assert engine_params["max_overflow"] == 5
        assert engine_params["pool_timeout"] == 30

    def test_driver_specific_url_generation(self):
        """Test URL generation for different drivers."""
        configs = [
            {
                "driver": "postgresql",
                "expected_prefix": "postgresql://"
            },
            {
                "driver": "mysql",
                "expected_prefix": "mysql://"
            },
            {
                "driver": "oracle",
                "expected_prefix": "oracle://"
            }
        ]
        
        for config_data in configs:
            config = DatabaseConfig(
                driver=config_data["driver"],
                host="localhost",
                name="testdb",
                user="user",
                password="pass"
            )
            
            with patch('src.database.base.create_engine') as mock_create_engine:
                mock_engine = Mock()
                mock_create_engine.return_value = mock_engine
                
                manager = DatabaseManager(config)
                url = manager._build_connection_url()
                
                assert url.startswith(config_data["expected_prefix"])