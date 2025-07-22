"""
Simple test configuration and fixtures.
"""
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for tests."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "user": "test_user",
            "password": "test_password",
            "driver": "postgresql"
        },
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_collection"
        },
        "logging": {
            "level": "DEBUG"
        }
    }


@pytest.fixture
def mock_client():
    """Mock client for testing."""
    mock = Mock()
    mock.connect.return_value = True
    mock.disconnect.return_value = None
    mock.is_connected.return_value = True
    return mock


@pytest.fixture
def sample_vectors():
    """Sample vector data."""
    import random
    return [[random.random() for _ in range(128)] for _ in range(10)]


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Cleanup code would go here