"""
Global test configuration and fixtures.
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock, patch

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import ConfigManager
from src.core.exceptions import ConfigurationError as ConfigError
from src.database.engine import DatabaseEngine, EngineConfig
from src.database.pool import ConnectionPool, PoolConfig
from src.database.health import DatabaseHealthMonitor
from src.milvus.client import MilvusClient, MilvusConnectionPool


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configurations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config_dict() -> Dict[str, Any]:
    """Test configuration dictionary."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_ragdb",
            "user": "test_user",
            "password": "test_password",
            "driver": "postgresql",
            "pool_size": 5,
            "max_overflow": 5,
            "pool_timeout": 10,
            "pool_recycle": 3600,
            "echo": False
        },
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "user": "test_user",
            "password": "test_password",
            "secure": False,
            "db_name": "test_db",
            "alias": "test_alias",
            "max_retries": 3,
            "retry_delay": 1.0,
            "collections": {
                "test_collection": {
                    "vector_dim": 768,
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "nlist": 1024
                }
            },
            "rbac": {
                "enable_rbac": True,
                "default_permissions": ["read"]
            }
        },
        "logging": {
            "level": "DEBUG",
            "format": "{time} | {level} | {name} | {message}",
            "file": "logs/test.log",
            "rotation": "1 MB",
            "retention": "1 day"
        }
    }


@pytest.fixture
def test_config_file(temp_config_dir, test_config_dict):
    """Create a test configuration file."""
    config_file = temp_config_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config_dict, f)
    return config_file


@pytest.fixture
def config_manager(test_config_file) -> ConfigManager:
    """ConfigManager instance with test configuration."""
    return ConfigManager(config_path=str(test_config_file))


@pytest.fixture
def mock_config_manager(test_config_dict) -> Mock:
    """Mock ConfigManager for unit tests."""
    mock = Mock(spec=ConfigManager)
    mock.get_config.return_value = test_config_dict
    mock.get_database_config.return_value = test_config_dict["database"]
    mock.get_milvus_config.return_value = test_config_dict["milvus"]
    mock.get_logging_config.return_value = test_config_dict["logging"]
    return mock


# Database Fixtures
@pytest.fixture
def engine_config() -> EngineConfig:
    """Test database engine configuration."""
    return EngineConfig(
        host="localhost",
        port=5432,
        database="test_ragdb",
        username="test_user",
        password="test_password",
        driver="postgresql",
        echo=False,
        pool_size=5,
        max_overflow=5,
        pool_timeout=10,
        pool_recycle=3600
    )


@pytest.fixture
def pool_config() -> PoolConfig:
    """Test connection pool configuration."""
    return PoolConfig(
        pool_size=5,
        max_overflow=5,
        pool_timeout=10,
        pool_recycle=3600,
        pool_pre_ping=True,
        pool_reset_on_return="commit"
    )


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine."""
    engine = Mock()
    engine.url = "postgresql://test_user:***@localhost:5432/test_ragdb"
    engine.dialect.name = "postgresql"
    engine.connect.return_value.__enter__ = Mock()
    engine.connect.return_value.__exit__ = Mock()
    return engine


@pytest.fixture
def mock_database_engine(mock_config_manager, mock_engine) -> Mock:
    """Mock DatabaseEngine."""
    mock = Mock(spec=DatabaseEngine)
    mock.config_manager = mock_config_manager
    mock.engine = mock_engine
    mock._config = EngineConfig(
        host="localhost",
        port=5432,
        database="test_ragdb",
        username="test_user", 
        password="test_password",
        driver="postgresql"
    )
    mock.get_connection_url.return_value = "postgresql://test_user:***@localhost:5432/test_ragdb"
    mock.test_connection.return_value = True
    mock.get_engine_info.return_value = {
        "driver": "postgresql",
        "server_version": "13.0",
        "pool_size": 5
    }
    return mock


@pytest.fixture
def mock_connection_pool(mock_database_engine) -> Mock:
    """Mock ConnectionPool."""
    mock = Mock(spec=ConnectionPool)
    mock.engine = mock_database_engine
    mock.is_healthy.return_value = True
    mock.get_pool_status.return_value = {
        "size": 5,
        "checked_in": 3,
        "checked_out": 2,
        "overflow": 0,
        "invalid": 0
    }
    return mock


@pytest.fixture
def mock_health_monitor() -> Mock:
    """Mock DatabaseHealthMonitor."""
    mock = Mock(spec=DatabaseHealthMonitor)
    mock.is_healthy.return_value = True
    mock.check_connection.return_value = {"status": "healthy", "latency": 0.001}
    mock.check_pool_health.return_value = {"status": "healthy", "pool_usage": 40}
    mock.get_health_report.return_value = {
        "overall_status": "healthy",
        "checks": [
            {"name": "connection", "status": "healthy"},
            {"name": "pool", "status": "healthy"}
        ]
    }
    return mock


# Milvus Fixtures
@pytest.fixture
def mock_milvus_client() -> Mock:
    """Mock MilvusClient."""
    mock = Mock(spec=MilvusClient)
    mock.alias = "test_alias"
    mock.config = {
        "host": "localhost",
        "port": 19530,
        "user": "test_user"
    }
    mock.is_connected.return_value = True
    mock.connect.return_value = True
    mock.disconnect.return_value = None
    mock.get_server_version.return_value = "2.3.0"
    mock.list_databases.return_value = ["default", "test_db"]
    return mock


@pytest.fixture
def mock_milvus_connection_pool() -> Mock:
    """Mock MilvusConnectionPool."""
    mock = Mock(spec=MilvusConnectionPool)
    mock.size = 5
    mock.get_connection.return_value = Mock()
    mock.return_connection.return_value = None
    mock.close_all.return_value = None
    mock.get_pool_stats.return_value = {
        "total_connections": 5,
        "active_connections": 2,
        "idle_connections": 3
    }
    return mock


@pytest.fixture
def mock_milvus_collection() -> Mock:
    """Mock MilvusCollection."""
    mock = Mock()
    mock.collection_name = "test_collection"
    mock.schema = Mock()
    mock.schema.vector_dim = 768
    mock.is_loaded.return_value = True
    mock.load.return_value = None
    mock.release.return_value = None
    mock.get_entity_count.return_value = 1000
    mock.insert.return_value = Mock(insert_count=10, primary_keys=[1, 2, 3])
    mock.search.return_value = Mock(hits=[{"id": 1, "distance": 0.1}])
    mock.query.return_value = [{"id": 1, "vector": [0.1] * 768}]
    mock.delete.return_value = Mock(delete_count=1)
    return mock


# Test Data Fixtures
@pytest.fixture
def sample_vectors():
    """Sample vector data for testing."""
    return [
        [0.1] * 768,
        [0.2] * 768,
        [0.3] * 768,
        [0.4] * 768,
        [0.5] * 768
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return [
        {
            "id": 1,
            "text": "This is a test document",
            "user_id": "user1",
            "group_ids": ["group1", "group2"],
            "permissions": ["read"],
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": 2,
            "text": "Another test document",
            "user_id": "user2", 
            "group_ids": ["group2", "group3"],
            "permissions": ["read", "write"],
            "created_at": "2024-01-02T00:00:00Z"
        }
    ]


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    test_env = {
        "APP_ENV": "test",
        "LOG_LEVEL": "DEBUG",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_ragdb",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530",
        "MILVUS_USER": "test_user",
        "MILVUS_PASSWORD": "test_password"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


# Mock external dependencies
@pytest.fixture
def mock_pymilvus():
    """Mock pymilvus package."""
    with patch.multiple(
        'src.milvus.client',
        connections=MagicMock(),
        Collection=MagicMock(),
        CollectionSchema=MagicMock(),
        FieldSchema=MagicMock(),
        DataType=MagicMock(),
        utility=MagicMock()
    ):
        yield


@pytest.fixture  
def mock_sqlalchemy():
    """Mock SQLAlchemy components."""
    with patch.multiple(
        'src.database.engine',
        create_engine=MagicMock(),
        sessionmaker=MagicMock()
    ):
        yield


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_logs():
    """Clean up test logs after each test."""
    yield
    # Clean up any test log files
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in log_dir.glob("test*.log"):
            try:
                log_file.unlink()
            except FileNotFoundError:
                pass


@pytest.fixture(scope="function")
def isolated_test():
    """Ensure test isolation."""
    # Reset any global state before test
    yield
    # Clean up after test


# Integration test fixtures
@pytest.fixture(scope="session")
def test_database_url():
    """Database URL for integration tests."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://test_user:test_password@localhost:5432/test_ragdb"
    )


@pytest.fixture(scope="session") 
def test_milvus_uri():
    """Milvus URI for integration tests."""
    return os.getenv(
        "TEST_MILVUS_URI",
        "http://localhost:19530"
    )


@pytest.fixture
def skip_if_no_database():
    """Skip test if database is not available."""
    def _skip_if_no_database():
        try:
            from sqlalchemy import create_engine
            engine = create_engine("postgresql://test_user:test_password@localhost:5432/test_ragdb")
            engine.connect().close()
        except Exception:
            pytest.skip("Database not available for integration tests")
    return _skip_if_no_database


@pytest.fixture
def skip_if_no_milvus():
    """Skip test if Milvus is not available."""
    def _skip_if_no_milvus():
        try:
            from pymilvus import connections
            connections.connect(alias="test", host="localhost", port="19530")
            connections.disconnect(alias="test")
        except Exception:
            pytest.skip("Milvus not available for integration tests")
    return _skip_if_no_milvus