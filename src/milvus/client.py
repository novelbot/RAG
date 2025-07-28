"""
Milvus client implementation with connection management and health checks.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
from contextlib import contextmanager

from pymilvus import connections, utility, MilvusException, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus.exceptions import ConnectionNotExistException, MilvusException as PyMilvusException
from loguru import logger

from src.core.config import MilvusConfig
from src.core.exceptions import MilvusError, ConnectionError, HealthCheckError
from src.core.logging import LoggerMixin


@dataclass
class ConnectionStatus:
    """Status of a Milvus connection."""
    connected: bool
    connection_time: Optional[datetime] = None
    last_ping: Optional[datetime] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None


class MilvusClient(LoggerMixin):
    """
    Milvus client with connection management, health checks, and retry logic.
    
    Based on Context7 documentation for pymilvus connection management:
    - Uses connections.connect() for establishing connections
    - Implements health checks with utility.has_collection()
    - Provides connection pooling and retry mechanisms
    """
    
    def __init__(self, config: MilvusConfig):
        """
        Initialize Milvus client.
        
        Args:
            config: Milvus configuration settings
        """
        self.config = config
        self.alias = config.alias or "default"
        self._connection_status = ConnectionStatus(connected=False)
        self._lock = threading.Lock()
        self._retry_count = 0
        self._max_retries = config.max_retries or 3
        self._retry_delay = config.retry_delay or 1.0
        
        # Connection parameters
        self._connection_params = {
            "host": config.host,
            "port": config.port,
            "user": config.user,
            "password": config.password,
            "secure": config.secure,
            "db_name": config.db_name or ""
        }
        
        # Initialize connection
        self._setup_connection()
    
    def _setup_connection(self) -> None:
        """Setup Milvus connection configuration."""
        try:
            # Store connection parameters for later use
            self.logger.info(f"Milvus connection parameters configured: {self.alias}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Milvus connection: {e}")
            raise MilvusError(f"Connection setup failed: {e}")
    
    def connect(self) -> bool:
        """
        Connect to Milvus server.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            ConnectionError: If connection fails
        """
        with self._lock:
            if self._connection_status.connected:
                return True
            
            start_time = time.time()
            
            for attempt in range(self._max_retries):
                try:
                    # Establish connection using recommended approach from Context7
                    connections.connect(
                        alias=self.alias,
                        host=self._connection_params["host"],
                        port=str(self._connection_params["port"]),
                        user=self._connection_params["user"],
                        password=self._connection_params["password"],
                        secure=self._connection_params["secure"]
                    )
                    
                    # Verify connection with a simple utility call
                    utility.list_collections(using=self.alias)
                    
                    # Update connection status
                    self._connection_status = ConnectionStatus(
                        connected=True,
                        connection_time=datetime.utcnow(),
                        last_ping=datetime.utcnow(),
                        response_time=time.time() - start_time
                    )
                    
                    self.logger.info(f"Connected to Milvus server: {self.alias}")
                    return True
                    
                except Exception as e:
                    self._retry_count += 1
                    error_msg = f"Connection attempt {attempt + 1} failed: {e}"
                    self.logger.warning(error_msg)
                    
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        self._connection_status = ConnectionStatus(
                            connected=False,
                            error_message=str(e)
                        )
                        raise ConnectionError(f"Failed to connect after {self._max_retries} attempts: {e}")
            
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        with self._lock:
            try:
                if self._connection_status.connected:
                    connections.disconnect(alias=self.alias)
                    self.logger.info(f"Disconnected from Milvus server: {self.alias}")
                
                self._connection_status = ConnectionStatus(connected=False)
                
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
                raise MilvusError(f"Disconnect failed: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Milvus server."""
        with self._lock:
            return self._connection_status.connected
    
    def ping(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Ping Milvus server to check connectivity.
        
        Args:
            timeout: Ping timeout in seconds
            
        Returns:
            Dict with ping results
        """
        start_time = time.time()
        
        try:
            # Try to connect first if not connected
            if not self.is_connected():
                self.connect()
            
            # Perform ping by listing collections
            collections = utility.list_collections(using=self.alias)
            response_time = time.time() - start_time
            
            if response_time > timeout:
                status = "degraded"
                message = f"Slow response: {response_time:.3f}s"
            else:
                status = "healthy"
                message = f"Ping successful: {response_time:.3f}s"
            
            # Update last ping time
            with self._lock:
                self._connection_status.last_ping = datetime.utcnow()
                self._connection_status.response_time = response_time
            
            return {
                "status": status,
                "response_time": response_time,
                "message": message,
                "collections_count": len(collections),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Ping failed: {e}")
            
            return {
                "status": "unhealthy",
                "response_time": response_time,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        with self._lock:
            return {
                "alias": self.alias,
                "host": self._connection_params["host"],
                "port": self._connection_params["port"],
                "user": self._connection_params["user"],
                "db_name": self._connection_params["db_name"],
                "secure": self._connection_params["secure"],
                "connected": self._connection_status.connected,
                "connection_time": self._connection_status.connection_time.isoformat() if self._connection_status.connection_time else None,
                "last_ping": self._connection_status.last_ping.isoformat() if self._connection_status.last_ping else None,
                "response_time": self._connection_status.response_time,
                "error_message": self._connection_status.error_message,
                "retry_count": self._retry_count
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dict with health check results
        """
        results = {
            "connection": self.get_connection_info(),
            "ping": None,
            "collections": None,
            "overall_status": "unknown"
        }
        
        try:
            # Check connection status
            if not self.is_connected():
                results["overall_status"] = "unhealthy"
                results["error"] = "Not connected to Milvus server"
                return results
            
            # Perform ping
            ping_result = self.ping()
            results["ping"] = ping_result
            
            # List collections
            collections = utility.list_collections(using=self.alias)
            results["collections"] = {
                "count": len(collections),
                "names": collections
            }
            
            # Determine overall status
            if ping_result["status"] == "healthy":
                results["overall_status"] = "healthy"
            elif ping_result["status"] == "degraded":
                results["overall_status"] = "degraded"
            else:
                results["overall_status"] = "unhealthy"
                
        except Exception as e:
            results["overall_status"] = "unhealthy"
            results["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return results
    
    def list_collections(self) -> List[str]:
        """
        List all collections in Milvus.
        
        Returns:
            List of collection names
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            return utility.list_collections(using=self.alias)
            
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            raise MilvusError(f"List collections failed: {e}")
    
    def has_collection(self, collection_name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if collection exists
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            return utility.has_collection(collection_name, using=self.alias)
            
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            raise MilvusError(f"Collection existence check failed: {e}")
    
    def drop_collection(self, collection_name: str) -> None:
        """
        Drop a collection.
        
        Args:
            collection_name: Name of the collection to drop
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            utility.drop_collection(collection_name, using=self.alias)
            self.logger.info(f"Dropped collection: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to drop collection {collection_name}: {e}")
            raise MilvusError(f"Drop collection failed: {e}")

    def get_collection(self, collection_name: str) -> "Collection":
        """
        Get an existing collection by name.
        
        Args:
            collection_name: Name of the collection to get
            
        Returns:
            Collection: PyMilvus Collection object
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            if not utility.has_collection(collection_name, using=self.alias):
                raise MilvusError(f"Collection {collection_name} does not exist")
            
            from pymilvus import Collection
            collection = Collection(collection_name, using=self.alias)
            self.logger.info(f"Retrieved collection: {collection_name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to get collection {collection_name}: {e}")
            raise MilvusError(f"Get collection failed: {e}")

    def create_collection(self, name: str, schema: "CollectionSchema", **kwargs) -> "Collection":
        """
        Create a new collection with the given schema.
        
        Args:
            name: Name of the collection
            schema: Collection schema
            **kwargs: Additional arguments
            
        Returns:
            Collection: PyMilvus Collection object
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            from pymilvus import Collection
            collection = Collection(name, schema, using=self.alias, **kwargs)
            self.logger.info(f"Created collection: {name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {name}: {e}")
            raise MilvusError(f"Create collection failed: {e}")
    
    def create_collection_if_not_exists(
        self, 
        collection_name: str, 
        dim: int = 1024,
        description: str = "Auto-generated collection for RAG documents"
    ) -> Collection:
        """
        Create a collection if it doesn't exist, based on Context7 PyMilvus documentation.
        
        Args:
            collection_name: Name of the collection to create
            dim: Dimension of the vector field
            description: Description of the collection
            
        Returns:
            Collection: The created or existing collection
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            # Check if collection already exists
            if utility.has_collection(collection_name, using=self.alias):
                self.logger.info(f"Collection {collection_name} already exists")
                return Collection(collection_name, using=self.alias)
            
            # Define schema based on Context7 PyMilvus documentation
            fields = [
                FieldSchema(
                    name="id", 
                    dtype=DataType.VARCHAR, 
                    is_primary=True, 
                    auto_id=False, 
                    max_length=100
                ),
                FieldSchema(
                    name="content", 
                    dtype=DataType.VARCHAR, 
                    max_length=65535
                ),
                FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=dim
                ),
                FieldSchema(
                    name="metadata", 
                    dtype=DataType.JSON
                ),
                FieldSchema(
                    name="access_tags", 
                    dtype=DataType.VARCHAR,
                    max_length=1000
                )
            ]
            
            # Create schema
            schema = CollectionSchema(fields, description)
            
            # Create collection with Strong consistency (from Context7 docs)
            collection = Collection(
                collection_name, 
                schema, 
                consistency_level="Strong", 
                using=self.alias
            )
            
            # Create IVF_FLAT index on embedding field (from Context7 docs)
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2", 
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            
            # Load collection into memory (from Context7 docs)
            collection.load()
            
            self.logger.info(f"Created and loaded collection: {collection_name} with dimension {dim}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise MilvusError(f"Collection creation failed: {e}")
    
    def get_entity_count(self, collection_name: str) -> int:
        """
        Get the number of entities in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of entities in the collection
        """
        try:
            if not self.is_connected():
                raise MilvusError("Not connected to Milvus server")
            
            if not utility.has_collection(collection_name, using=self.alias):
                return 0
            
            collection = Collection(collection_name, using=self.alias)
            return collection.num_entities
            
        except Exception as e:
            self.logger.error(f"Failed to get entity count for collection {collection_name}: {e}")
            raise MilvusError(f"Get entity count failed: {e}")
    
    @contextmanager
    def connection_context(self):
        """
        Context manager for Milvus connection.
        
        Yields:
            MilvusClient: The connected client
        """
        if not self.is_connected():
            self.connect()
        
        try:
            yield self
        finally:
            # Connection remains open for reuse
            pass
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_connected():
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Keep connection open for reuse
        pass
    
    def __del__(self):
        """Cleanup on object deletion."""
        try:
            self.disconnect()
        except:
            pass


class MilvusConnectionPool(LoggerMixin):
    """Connection pool for managing multiple Milvus connections."""
    
    def __init__(self, config: MilvusConfig, pool_size: int = 5):
        """
        Initialize connection pool.
        
        Args:
            config: Milvus configuration
            pool_size: Number of connections in pool
        """
        self.config = config
        self.pool_size = pool_size
        self._pool: List[MilvusClient] = []
        self._lock = threading.Lock()
        self._connection_count = 0
        
        # Initialize connection pool
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        for i in range(self.pool_size):
            # Create unique alias for each connection
            config = self.config.copy()
            config.alias = f"{config.alias}_{i}"
            
            client = MilvusClient(config)
            self._pool.append(client)
    
    def get_connection(self) -> MilvusClient:
        """
        Get a connection from the pool.
        
        Returns:
            MilvusClient: Available connection
        """
        with self._lock:
            if not self._pool:
                raise MilvusError("No connections available in pool")
            
            # Get round-robin connection
            client = self._pool[self._connection_count % len(self._pool)]
            self._connection_count += 1
            
            # Ensure connection is active
            if not client.is_connected():
                client.connect()
            
            return client
    
    def health_check_all(self) -> Dict[str, Any]:
        """
        Health check for all connections in pool.
        
        Returns:
            Dict with health check results for all connections
        """
        results = {}
        
        with self._lock:
            for i, client in enumerate(self._pool):
                try:
                    results[f"connection_{i}"] = client.health_check()
                except Exception as e:
                    results[f"connection_{i}"] = {
                        "overall_status": "unhealthy",
                        "error": str(e)
                    }
        
        return results
    
    def close_all(self) -> None:
        """Close all connections in pool."""
        with self._lock:
            for client in self._pool:
                try:
                    client.disconnect()
                except Exception as e:
                    self.logger.error(f"Error closing connection: {e}")
            
            self._pool.clear()
            self.logger.info("All connections closed")