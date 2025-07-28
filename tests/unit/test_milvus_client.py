"""
Unit tests for Milvus Client module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pymilvus import MilvusException

from src.milvus.client import MilvusClient, MilvusConnectionPool, ConnectionStatus
from src.core.config import MilvusConfig
from src.core.exceptions import MilvusError, ConnectionError


class TestConnectionStatus:
    """Test ConnectionStatus dataclass."""
    
    def test_connection_status_creation(self):
        """Test ConnectionStatus creation."""
        status = ConnectionStatus(connected=True)
        
        assert status.connected is True
        assert status.connection_time is None
        assert status.last_ping is None
        assert status.error_message is None
        assert status.response_time is None

    def test_connection_status_with_values(self):
        """Test ConnectionStatus with all values."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        status = ConnectionStatus(
            connected=True,
            connection_time=now,
            last_ping=now,
            error_message="test error",
            response_time=0.1
        )
        
        assert status.connected is True
        assert status.connection_time == now
        assert status.last_ping == now
        assert status.error_message == "test error"
        assert status.response_time == 0.1


class TestMilvusClient:
    """Test MilvusClient class."""
    
    @pytest.fixture
    def client_config(self):
        """Milvus client configuration."""
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

    @pytest.fixture
    def mock_connections(self):
        """Mock pymilvus connections."""
        with patch('src.milvus.client.connections') as mock:
            yield mock

    @pytest.fixture
    def mock_utility(self):
        """Mock pymilvus utility."""
        with patch('src.milvus.client.utility') as mock:
            yield mock

    def test_milvus_client_creation(self, client_config):
        """Test MilvusClient creation."""
        client = MilvusClient(client_config)
        
        assert client.config.host == "localhost"
        assert client.config.port == 19530
        assert client.alias == "test_alias"
        assert client._max_retries == 3
        assert client._retry_delay == 1.0
        assert client._connection_status.connected == False

    def test_milvus_client_defaults(self):
        """Test MilvusClient with default values."""
        minimal_config = MilvusConfig(
            host="localhost"
        )
        
        client = MilvusClient(minimal_config)
        
        assert client.config.port == 19530
        assert client.alias == "default"
        assert client._max_retries == 3
        assert client._retry_delay == 1.0

    def test_connect_success(self, client_config, mock_connections, mock_utility):
        """Test successful connection."""
        mock_connections.connect.return_value = None
        mock_utility.list_collections.return_value = ["test_collection"]
        
        client = MilvusClient(client_config)
        result = client.connect()
        
        assert result is True
        assert client._connection_status.connected == True
        mock_connections.connect.assert_called_once()

    def test_connect_failure(self, client_config, mock_connections):
        """Test connection failure."""
        mock_connections.connect.side_effect = MilvusException("Connection failed")
        
        client = MilvusClient(client_config)
        
        with pytest.raises(ConnectionError):
            client.connect()

    def test_connect_with_retry(self, client_config, mock_connections, mock_utility):
        """Test connection with retry mechanism."""
        # First attempt fails, second succeeds
        mock_connections.connect.side_effect = [
            MilvusException("Temporary failure"),
            None,
            None  # Extra call for subsequent attempts
        ]
        mock_utility.list_collections.side_effect = [
            MilvusException("Temporary failure"),
            ["test_collection"]
        ]
        
        client = MilvusClient(client_config)
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = client.connect()
        
        assert result is True
        assert mock_connections.connect.call_count == 3  # Setup call + 2 connection attempts

    def test_connect_retry_exhausted(self, client_config, mock_connections):
        """Test connection retry exhaustion."""
        mock_connections.connect.side_effect = MilvusException("Persistent failure")
        
        client = MilvusClient(client_config)
        
        with patch('time.sleep'):
            with pytest.raises(ConnectionError):
                client.connect()
        
        assert mock_connections.connect.call_count == 3  # max_retries

    def test_disconnect_success(self, client_config, mock_connections):
        """Test successful disconnection."""
        mock_connections.disconnect.return_value = None
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        client.disconnect()
        
        assert client._connection_status.connected == False
        mock_connections.disconnect.assert_called_once_with(alias="test_alias")

    def test_disconnect_failure(self, client_config, mock_connections):
        """Test disconnection failure."""
        mock_connections.disconnect.side_effect = MilvusException("Disconnect failed")
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        # Should raise MilvusError
        with pytest.raises(MilvusError):
            client.disconnect()

    def test_is_connected_true(self, client_config):
        """Test is_connected when connected."""
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        assert client.is_connected() is True

    def test_is_connected_false(self, client_config):
        """Test is_connected when not connected."""
        client = MilvusClient(client_config)
        client._connection_status.connected = False
        
        assert client.is_connected() is False

    def test_is_connected_exception(self, client_config):
        """Test is_connected without exception handling."""
        client = MilvusClient(client_config)
        client._connection_status.connected = False
        
        assert client.is_connected() is False

    def test_list_collections(self, client_config, mock_utility):
        """Test collection listing."""
        mock_utility.list_collections.return_value = ["collection1", "collection2"]
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        collections = client.list_collections()
        
        assert collections == ["collection1", "collection2"]
        mock_utility.list_collections.assert_called_once_with(using="test_alias")

    def test_list_collections_not_connected(self, client_config):
        """Test collection listing when not connected."""
        client = MilvusClient(client_config)
        client._connection_status.connected = False
        
        with pytest.raises(MilvusError, match="Not connected"):
            client.list_collections()

    def test_has_collection(self, client_config, mock_utility):
        """Test collection existence check."""
        mock_utility.has_collection.return_value = True
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        result = client.has_collection("test_collection")
        
        assert result is True
        mock_utility.has_collection.assert_called_once_with("test_collection", using="test_alias")

    def test_has_collection_not_connected(self, client_config):
        """Test collection existence check when not connected."""
        client = MilvusClient(client_config)
        client._connection_status.connected = False
        
        with pytest.raises(MilvusError, match="Not connected"):
            client.has_collection("test_collection")

    def test_get_connection_info(self, client_config):
        """Test connection info retrieval."""
        client = MilvusClient(client_config)
        
        info = client.get_connection_info()
        
        assert info["host"] == "localhost"
        assert info["port"] == 19530
        assert info["alias"] == "test_alias"
        assert info["connected"] == False

    def test_ping_success(self, client_config, mock_utility):
        """Test successful ping."""
        mock_utility.list_collections.return_value = ["collection1"]
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        result = client.ping()
        
        assert result["status"] == "healthy"
        assert "response_time" in result
        assert "collections_count" in result

    def test_ping_failure(self, client_config, mock_utility):
        """Test ping failure."""
        mock_utility.list_collections.side_effect = MilvusException("Server unavailable")
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        result = client.ping()
        
        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_ping_not_connected(self, client_config):
        """Test ping when not connected."""
        client = MilvusClient(client_config)
        client._connection_status.connected = False
        
        result = client.ping()
        
        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_get_connection_info_method(self, client_config):
        """Test get_connection_info method."""
        client = MilvusClient(client_config)
        
        info = client.get_connection_info()
        
        assert info["host"] == "localhost"
        assert info["port"] == 19530
        assert "user" in info

    def test_context_manager_success(self, client_config, mock_connections, mock_utility):
        """Test context manager usage - success."""
        mock_connections.connect.return_value = None
        mock_utility.list_collections.return_value = ["collection1"]
        
        client = MilvusClient(client_config)
        
        with client:
            # Context manager auto-connects if not connected
            pass

    def test_context_manager_connection_failure(self, client_config, mock_connections):
        """Test context manager with connection failure."""
        mock_connections.connect.side_effect = MilvusException("Connection failed")
        
        client = MilvusClient(client_config)
        
        with pytest.raises(ConnectionError):
            with client:
                pass

    def test_health_check_healthy(self, client_config, mock_utility):
        """Test health check - healthy."""
        mock_utility.list_collections.return_value = ["collection1"]
        
        client = MilvusClient(client_config)
        client._connection_status.connected = True
        
        with patch.object(client, 'ping', return_value={"status": "healthy"}):
            health = client.health_check()
        
        assert health["overall_status"] == "healthy"
        assert "connection" in health
        assert "ping" in health

    def test_health_check_unhealthy(self, client_config):
        """Test health check - unhealthy."""
        client = MilvusClient(client_config)
        client._connection_status.connected = False
        
        health = client.health_check()
        
        assert health["overall_status"] == "unhealthy"
        assert "error" in health


class TestMilvusConnectionPool:
    """Test MilvusConnectionPool class."""
    
    @pytest.fixture
    def pool_config(self):
        """Connection pool configuration."""
        return MilvusConfig(
            host="localhost",
            port=19530,
            user="test_user",
            password="test_password"
        )

    def test_connection_pool_creation(self, pool_config):
        """Test MilvusConnectionPool creation."""
        pool = MilvusConnectionPool(pool_config, pool_size=5)
        
        assert pool.pool_size == 5
        assert len(pool._pool) == 5

    def test_get_connection_round_robin(self, pool_config):
        """Test getting connection using round-robin."""
        with patch('src.milvus.client.MilvusClient') as mock_client_class:
            mock_clients = []
            for i in range(3):
                mock_client = Mock()
                mock_client.alias = f"default_{i}"
                mock_clients.append(mock_client)
            
            mock_client_class.side_effect = mock_clients
            
            pool = MilvusConnectionPool(pool_config, pool_size=3)
            
            # Get connections in round-robin fashion
            conn1 = pool.get_connection()
            conn2 = pool.get_connection()
            conn3 = pool.get_connection()
            conn4 = pool.get_connection()  # Should wrap around
            
            # Fourth connection should be same as first (round-robin)
            assert conn4.alias == conn1.alias

    def test_pool_health_check_all(self, pool_config):
        """Test health check for all connections in pool."""
        pool = MilvusConnectionPool(pool_config, pool_size=2)
        
        # Mock health checks for all clients
        for client in pool._pool:
            with patch.object(client, 'health_check', return_value={"overall_status": "healthy"}):
                pass
        
        health_results = pool.health_check_all()
        
        assert len(health_results) == 2
        assert "connection_0" in health_results
        assert "connection_1" in health_results

    def test_close_all_connections(self, pool_config):
        """Test closing all connections in pool."""
        pool = MilvusConnectionPool(pool_config, pool_size=3)
        
        # Mock disconnect for all clients
        for client in pool._pool:
            with patch.object(client, 'disconnect'):
                pass
        
        pool.close_all()
        
        # Pool should be empty after closing
        assert len(pool._pool) == 0