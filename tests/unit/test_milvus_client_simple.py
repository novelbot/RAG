"""
Unit tests for Milvus Client module (simple version).
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.milvus.client import MilvusClient
from src.core.config import MilvusConfig
from src.core.exceptions import MilvusError, ConnectionError


class TestMilvusClientBasic:
    """Basic tests for MilvusClient."""
    
    @pytest.fixture
    def client_config(self):
        """Basic client configuration."""
        return MilvusConfig(
            host="localhost",
            port=19530,
            alias="test_alias"
        )

    def test_milvus_client_creation(self, client_config):
        """Test MilvusClient creation."""
        client = MilvusClient(client_config)
        
        assert client.config.host == "localhost"
        assert client.config.port == 19530
        assert client.alias == "test_alias"

    def test_milvus_client_defaults(self):
        """Test MilvusClient with minimal config."""
        minimal_config = MilvusConfig(host="localhost")
        client = MilvusClient(minimal_config)
        
        assert client.config.host == "localhost"
        assert client.config.port == 19530  # Default port
        assert client.alias == "default"  # Default alias

    @patch('src.milvus.client.connections')
    def test_connect_success(self, mock_connections, client_config):
        """Test successful connection."""
        mock_connections.connect.return_value = None
        mock_connections.list_connections.return_value = [("test_alias", None)]
        
        client = MilvusClient(client_config)
        result = client.connect()
        
        assert result is True
        mock_connections.connect.assert_called_once()

    @patch('src.milvus.client.connections')
    def test_connect_failure(self, mock_connections, client_config):
        """Test connection failure."""
        from pymilvus import MilvusException
        mock_connections.connect.side_effect = MilvusException("Connection failed")
        
        client = MilvusClient(client_config)
        result = client.connect()
        
        assert result is False

    @patch('src.milvus.client.connections')
    def test_disconnect(self, mock_connections, client_config):
        """Test disconnection."""
        mock_connections.disconnect.return_value = None
        
        client = MilvusClient(client_config)
        client.disconnect()
        
        mock_connections.disconnect.assert_called_once_with(alias="test_alias")

    @patch('src.milvus.client.connections')
    def test_is_connected_true(self, mock_connections, client_config):
        """Test is_connected when connected."""
        mock_connections.list_connections.return_value = [("test_alias", None)]
        
        client = MilvusClient(client_config)
        
        assert client.is_connected() is True

    @patch('src.milvus.client.connections')
    def test_is_connected_false(self, mock_connections, client_config):
        """Test is_connected when not connected."""
        mock_connections.list_connections.return_value = []
        
        client = MilvusClient(client_config)
        
        assert client.is_connected() is False

    @patch('src.milvus.client.utility')
    def test_get_server_version(self, mock_utility, client_config):
        """Test server version retrieval."""
        mock_utility.get_server_version.return_value = "2.3.0"
        
        client = MilvusClient(client_config)
        client._status = "connected"  # Mock connected status
        
        version = client.get_server_version()
        
        assert version == "2.3.0"
        mock_utility.get_server_version.assert_called_once()

    @patch('src.milvus.client.utility')
    def test_list_databases(self, mock_utility, client_config):
        """Test database listing."""
        mock_utility.list_database.return_value = ["default", "test_db"]
        
        client = MilvusClient(client_config)
        client._status = "connected"  # Mock connected status
        
        databases = client.list_databases()
        
        assert databases == ["default", "test_db"]
        mock_utility.list_database.assert_called_once()

    def test_get_connection_info(self, client_config):
        """Test connection info retrieval."""
        client = MilvusClient(client_config)
        
        info = client.get_connection_info()
        
        assert info["host"] == "localhost"
        assert info["port"] == 19530
        assert info["alias"] == "test_alias"

    @patch('src.milvus.client.utility')
    def test_ping_success(self, mock_utility, client_config):
        """Test successful ping."""
        mock_utility.get_server_version.return_value = "2.3.0"
        
        client = MilvusClient(client_config)
        client._status = "connected"
        
        result = client.ping()
        
        assert result is True

    @patch('src.milvus.client.utility')  
    def test_ping_failure(self, mock_utility, client_config):
        """Test ping failure."""
        from pymilvus import MilvusException
        mock_utility.get_server_version.side_effect = MilvusException("Server unavailable")
        
        client = MilvusClient(client_config)
        client._status = "connected"
        
        result = client.ping()
        
        assert result is False

    def test_ping_not_connected(self, client_config):
        """Test ping when not connected."""
        client = MilvusClient(client_config)
        client._status = "disconnected"
        
        result = client.ping()
        
        assert result is False

    @patch('src.milvus.client.connections')
    @patch('src.milvus.client.utility')
    def test_health_check_healthy(self, mock_utility, mock_connections, client_config):
        """Test health check - healthy."""
        mock_connections.list_connections.return_value = [("test_alias", None)]
        mock_utility.get_server_version.return_value = "2.3.0"
        
        client = MilvusClient(client_config)
        client._status = "connected"
        
        health = client.health_check()
        
        assert health["status"] == "healthy"
        assert health["connected"] is True

    @patch('src.milvus.client.connections')
    def test_health_check_unhealthy(self, mock_connections, client_config):
        """Test health check - unhealthy."""
        mock_connections.list_connections.return_value = []
        
        client = MilvusClient(client_config)
        
        health = client.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["connected"] is False


class TestMilvusClientAdvanced:
    """Advanced tests for MilvusClient."""
    
    @pytest.fixture
    def full_config(self):
        """Full client configuration."""
        return MilvusConfig(
            host="localhost",
            port=19530,
            user="test_user",
            password="test_password",
            secure=False,
            db_name="test_db",
            alias="test_alias",
            max_retries=3,
            retry_delay=1.0
        )

    def test_milvus_client_full_config(self, full_config):
        """Test MilvusClient with full configuration."""
        client = MilvusClient(full_config)
        
        assert client.config.host == "localhost"
        assert client.config.port == 19530
        assert client.config.user == "test_user"
        assert client.config.password == "test_password"
        assert client.config.secure is False
        assert client.config.db_name == "test_db"
        assert client.alias == "test_alias"
        assert client.config.max_retries == 3
        assert client.config.retry_delay == 1.0

    @patch('src.milvus.client.connections')
    @patch('time.sleep')
    def test_connect_with_retry(self, mock_sleep, mock_connections, full_config):
        """Test connection with retry mechanism."""
        from pymilvus import MilvusException
        
        # First attempt fails, second succeeds
        mock_connections.connect.side_effect = [
            MilvusException("Temporary failure"),
            None
        ]
        mock_connections.list_connections.return_value = [("test_alias", None)]
        
        client = MilvusClient(full_config)
        result = client.connect()
        
        assert result is True
        assert mock_connections.connect.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch('src.milvus.client.connections')
    @patch('time.sleep')
    def test_connect_retry_exhausted(self, mock_sleep, mock_connections, full_config):
        """Test connection retry exhaustion."""
        from pymilvus import MilvusException
        mock_connections.connect.side_effect = MilvusException("Persistent failure")
        
        client = MilvusClient(full_config)
        result = client.connect()
        
        assert result is False
        assert mock_connections.connect.call_count == 4  # initial + 3 retries
        assert mock_sleep.call_count == 3  # 3 retry delays

    @patch('src.milvus.client.connections')
    def test_context_manager_success(self, mock_connections, full_config):
        """Test context manager usage."""
        mock_connections.connect.return_value = None
        mock_connections.list_connections.return_value = [("test_alias", None)]
        mock_connections.disconnect.return_value = None
        
        with MilvusClient(full_config) as client:
            assert client.is_connected() is True
        
        mock_connections.disconnect.assert_called_once()

    @patch('src.milvus.client.connections')
    def test_context_manager_connection_failure(self, mock_connections, full_config):
        """Test context manager with connection failure."""
        from pymilvus import MilvusException
        mock_connections.connect.side_effect = MilvusException("Connection failed")
        
        with pytest.raises(MilvusError):
            with MilvusClient(full_config):
                pass


class TestMilvusClientErrorHandling:
    """Test error handling in MilvusClient."""
    
    @pytest.fixture
    def client_config(self):
        """Basic client configuration."""
        return MilvusConfig(host="localhost", alias="test_alias")

    def test_get_server_version_not_connected(self, client_config):
        """Test server version when not connected."""
        client = MilvusClient(client_config)
        client._status = "disconnected"
        
        with pytest.raises(MilvusError, match="Not connected"):
            client.get_server_version()

    def test_list_databases_not_connected(self, client_config):
        """Test database listing when not connected."""
        client = MilvusClient(client_config)
        client._status = "disconnected"
        
        with pytest.raises(MilvusError, match="Not connected"):
            client.list_databases()

    @patch('src.milvus.client.connections')
    def test_disconnect_failure_handling(self, mock_connections, client_config):
        """Test disconnect failure handling."""
        from pymilvus import MilvusException
        mock_connections.disconnect.side_effect = MilvusException("Disconnect failed")
        
        client = MilvusClient(client_config)
        
        # Should not raise exception, just log the error
        client.disconnect()

    @patch('src.milvus.client.connections')
    def test_is_connected_exception_handling(self, mock_connections, client_config):
        """Test is_connected exception handling."""
        from pymilvus import MilvusException
        mock_connections.list_connections.side_effect = MilvusException("Check failed")
        
        client = MilvusClient(client_config)
        
        # Should return False instead of raising exception
        assert client.is_connected() is False