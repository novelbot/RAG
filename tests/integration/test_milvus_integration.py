"""
Integration tests for Milvus integration.
"""
import pytest
import os
import time
import random
from unittest.mock import patch
from typing import List

from src.milvus.client import MilvusClient, MilvusConnectionPool
from src.milvus.schema import RAGCollectionSchema, create_default_rag_schema
from src.milvus.collection import MilvusCollection, CollectionManager
from src.milvus.rbac import RBACManager, UserContext, Permission, AccessScope
from src.milvus.index import IndexManager, IndexType, MetricType
from src.milvus.search import SearchManager, SearchQuery, SearchStrategy


@pytest.mark.integration
@pytest.mark.milvus
class TestMilvusClientIntegration:
    """Integration tests for MilvusClient."""
    
    @pytest.fixture
    def milvus_config(self):
        """Milvus configuration for integration tests."""
        return {
            "host": os.getenv("TEST_MILVUS_HOST", "localhost"),
            "port": int(os.getenv("TEST_MILVUS_PORT", "19530")),
            "user": os.getenv("TEST_MILVUS_USER", ""),
            "password": os.getenv("TEST_MILVUS_PASSWORD", ""),
            "secure": False,
            "db_name": "default",
            "alias": f"test_alias_{int(time.time())}",
            "max_retries": 3,
            "retry_delay": 1.0
        }

    def test_client_connection_lifecycle(self, milvus_config, skip_if_no_milvus):
        """Test complete client connection lifecycle."""
        skip_if_no_milvus()
        
        client = MilvusClient(milvus_config)
        
        try:
            # Test connection
            assert client.connect() is True
            assert client.is_connected() is True
            
            # Test server information
            version = client.get_server_version()
            assert version is not None
            assert isinstance(version, str)
            
            # Test database listing
            databases = client.list_databases()
            assert isinstance(databases, list)
            assert "default" in databases
            
            # Test ping
            assert client.ping() is True
            
            # Test health check
            health = client.health_check()
            assert health["status"] == "healthy"
            assert health["connected"] is True
            
        finally:
            client.disconnect()
            assert client.is_connected() is False

    def test_client_context_manager(self, milvus_config, skip_if_no_milvus):
        """Test client as context manager."""
        skip_if_no_milvus()
        
        with MilvusClient(milvus_config) as client:
            assert client.is_connected() is True
            
            # Test operations within context
            version = client.get_server_version()
            assert version is not None
        
        # Client should be disconnected after context
        assert client.is_connected() is False

    def test_connection_pool_integration(self, milvus_config, skip_if_no_milvus):
        """Test Milvus connection pool."""
        skip_if_no_milvus()
        
        pool_config = milvus_config.copy()
        pool_config["pool_size"] = 3
        
        pool = MilvusConnectionPool(pool_config)
        
        try:
            pool.initialize()
            
            # Test pool statistics
            stats = pool.get_pool_stats()
            assert stats["total_connections"] == 3
            assert stats["idle_connections"] == 3
            assert stats["active_connections"] == 0
            
            # Test getting connections
            connections = []
            for _ in range(2):
                conn = pool.get_connection()
                assert conn.is_connected() is True
                connections.append(conn)
            
            # Check pool stats after getting connections
            stats = pool.get_pool_stats()
            assert stats["active_connections"] == 2
            assert stats["idle_connections"] == 1
            
            # Return connections
            for conn in connections:
                pool.return_connection(conn)
            
            # Pool should be back to initial state
            stats = pool.get_pool_stats()
            assert stats["active_connections"] == 0
            assert stats["idle_connections"] == 3
            
        finally:
            pool.close_all()


@pytest.mark.integration
@pytest.mark.milvus
class TestMilvusCollectionIntegration:
    """Integration tests for Milvus collections."""
    
    @pytest.fixture
    def milvus_client(self, milvus_config, skip_if_no_milvus):
        """Connected Milvus client."""
        skip_if_no_milvus()
        
        client = MilvusClient(milvus_config)
        client.connect()
        
        yield client
        
        client.disconnect()

    @pytest.fixture
    def test_schema(self):
        """Test collection schema."""
        return create_default_rag_schema(
            collection_name=f"test_collection_{int(time.time())}",
            vector_dim=128,
            enable_rbac=True
        )

    @pytest.fixture
    def test_collection(self, milvus_client, test_schema):
        """Test collection with cleanup."""
        from pymilvus import utility
        
        collection = MilvusCollection(milvus_client, test_schema)
        
        # Create collection if it doesn't exist
        if not collection.exists():
            collection.create()
        
        yield collection
        
        # Cleanup
        try:
            if collection.exists():
                collection.drop()
        except:
            pass  # Ignore cleanup errors

    def test_collection_lifecycle(self, milvus_client, test_schema):
        """Test complete collection lifecycle."""
        collection = MilvusCollection(milvus_client, test_schema)
        
        try:
            # Test creation
            assert not collection.exists()
            collection.create()
            assert collection.exists()
            
            # Test schema retrieval
            schema = collection.get_schema()
            assert schema is not None
            
            # Test loading
            collection.load()
            assert collection.is_loaded() is True
            
            # Test collection info
            info = collection.get_collection_info()
            assert info["name"] == test_schema.collection_name
            assert info["schema"] is not None
            
        finally:
            if collection.exists():
                collection.drop()

    def test_vector_operations(self, test_collection):
        """Test vector CRUD operations."""
        # Generate test data
        vectors = [[random.random() for _ in range(128)] for _ in range(10)]
        entities = []
        
        for i, vector in enumerate(vectors):
            entity = {
                "vector": vector,
                "text": f"Test document {i}",
                "user_id": f"user_{i % 3}",
                "group_ids": [f"group_{i % 2}"],
                "permissions": ["read", "write"]
            }
            entities.append(entity)
        
        # Test insertion
        result = test_collection.insert(entities)
        assert result.insert_count == 10
        assert len(result.primary_keys) == 10
        
        # Flush to ensure data is persisted
        test_collection.flush()
        
        # Load collection for search
        test_collection.load()
        
        # Test entity count
        count = test_collection.get_entity_count()
        assert count == 10
        
        # Test search
        query_vector = [random.random() for _ in range(128)]
        search_result = test_collection.search(
            query_vectors=[query_vector],
            limit=5,
            search_params={"metric_type": "L2", "params": {}}
        )
        
        assert len(search_result.hits) <= 5
        
        # Test query with filter
        query_result = test_collection.query(
            expr='user_id == "user_1"',
            output_fields=["text", "user_id"]
        )
        
        assert len(query_result) > 0
        for entity in query_result:
            assert entity["user_id"] == "user_1"

    def test_batch_operations(self, test_collection):
        """Test batch vector operations."""
        # Generate larger dataset
        batch_size = 50
        vectors = [[random.random() for _ in range(128)] for _ in range(batch_size)]
        entities = []
        
        for i, vector in enumerate(vectors):
            entity = {
                "vector": vector,
                "text": f"Batch document {i}",
                "user_id": f"batch_user_{i % 5}",
                "group_ids": [f"batch_group_{i % 3}"],
                "permissions": ["read"]
            }
            entities.append(entity)
        
        # Test batch insertion
        start_time = time.time()
        result = test_collection.insert_batch(entities, batch_size=20)
        end_time = time.time()
        
        assert result.insert_count == batch_size
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds
        
        test_collection.flush()
        test_collection.load()
        
        # Test batch search
        query_vectors = [[random.random() for _ in range(128)] for _ in range(5)]
        search_results = test_collection.search_batch(
            query_vectors=query_vectors,
            limit=3,
            search_params={"metric_type": "L2", "params": {}}
        )
        
        assert len(search_results) == 5
        for result in search_results:
            assert len(result.hits) <= 3


@pytest.mark.integration
@pytest.mark.milvus
class TestMilvusRBACIntegration:
    """Integration tests for Milvus RBAC."""
    
    @pytest.fixture
    def rbac_collection(self, milvus_client, skip_if_no_milvus):
        """Collection with RBAC enabled."""
        skip_if_no_milvus()
        
        schema = create_default_rag_schema(
            collection_name=f"rbac_test_{int(time.time())}",
            vector_dim=128,
            enable_rbac=True
        )
        
        collection = MilvusCollection(milvus_client, schema)
        
        if not collection.exists():
            collection.create()
        
        yield collection
        
        try:
            if collection.exists():
                collection.drop()
        except:
            pass

    @pytest.fixture
    def rbac_manager(self, milvus_client):
        """RBAC manager for testing."""
        return RBACManager(milvus_client)

    def test_rbac_user_isolation(self, rbac_collection, rbac_manager):
        """Test user data isolation with RBAC."""
        # Create test users
        user1 = UserContext(
            user_id="user1",
            group_ids=["group1"],
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        user2 = UserContext(
            user_id="user2",
            group_ids=["group2"],
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        rbac_manager.add_user_context(user1)
        rbac_manager.add_user_context(user2)
        
        # Insert data for different users
        user1_data = [
            {
                "vector": [random.random() for _ in range(128)],
                "text": "User 1 document 1",
                "user_id": "user1",
                "group_ids": ["group1"],
                "permissions": ["read", "write"]
            },
            {
                "vector": [random.random() for _ in range(128)],
                "text": "User 1 document 2",
                "user_id": "user1",
                "group_ids": ["group1"],
                "permissions": ["read", "write"]
            }
        ]
        
        user2_data = [
            {
                "vector": [random.random() for _ in range(128)],
                "text": "User 2 document 1",
                "user_id": "user2",
                "group_ids": ["group2"],
                "permissions": ["read", "write"]
            }
        ]
        
        # Insert data
        rbac_collection.insert(user1_data)
        rbac_collection.insert(user2_data)
        rbac_collection.flush()
        rbac_collection.load()
        
        # Test user1 can only see their data
        user1_filter = rbac_manager.get_access_filter("user1")
        user1_results = rbac_collection.query(
            expr=user1_filter,
            output_fields=["text", "user_id"]
        )
        
        assert len(user1_results) == 2
        for result in user1_results:
            assert result["user_id"] == "user1"
        
        # Test user2 can only see their data
        user2_filter = rbac_manager.get_access_filter("user2")
        user2_results = rbac_collection.query(
            expr=user2_filter,
            output_fields=["text", "user_id"]
        )
        
        assert len(user2_results) == 1
        for result in user2_results:
            assert result["user_id"] == "user2"

    def test_rbac_group_access(self, rbac_collection, rbac_manager):
        """Test group-based access control."""
        # Create user with multiple groups
        user = UserContext(
            user_id="multi_user",
            group_ids=["analysts", "researchers"],
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        rbac_manager.add_user_context(user)
        
        # Insert data for different groups
        group_data = [
            {
                "vector": [random.random() for _ in range(128)],
                "text": "Analysts document",
                "user_id": "other_user",
                "group_ids": ["analysts"],
                "permissions": ["read"]
            },
            {
                "vector": [random.random() for _ in range(128)],
                "text": "Researchers document",
                "user_id": "another_user",
                "group_ids": ["researchers"],
                "permissions": ["read"]
            },
            {
                "vector": [random.random() for _ in range(128)],
                "text": "Private document",
                "user_id": "private_user",
                "group_ids": ["private_group"],
                "permissions": ["read"]
            }
        ]
        
        rbac_collection.insert(group_data)
        rbac_collection.flush()
        rbac_collection.load()
        
        # Test user can access documents from their groups
        user_filter = rbac_manager.get_access_filter("multi_user")
        results = rbac_collection.query(
            expr=user_filter,
            output_fields=["text", "group_ids"]
        )
        
        # Should see documents from both analyst and researcher groups
        assert len(results) >= 2
        
        accessible_groups = set()
        for result in results:
            accessible_groups.update(result["group_ids"])
        
        assert "analysts" in accessible_groups
        assert "researchers" in accessible_groups
        assert "private_group" not in accessible_groups


@pytest.mark.integration
@pytest.mark.milvus
class TestMilvusIndexIntegration:
    """Integration tests for Milvus indexing."""
    
    @pytest.fixture
    def indexed_collection(self, milvus_client, skip_if_no_milvus):
        """Collection for index testing."""
        skip_if_no_milvus()
        
        schema = create_default_rag_schema(
            collection_name=f"index_test_{int(time.time())}",
            vector_dim=128
        )
        
        collection = MilvusCollection(milvus_client, schema)
        
        if not collection.exists():
            collection.create()
            
            # Insert some data for indexing
            test_data = []
            for i in range(1000):  # Larger dataset for meaningful indexing
                entity = {
                    "vector": [random.random() for _ in range(128)],
                    "text": f"Index test document {i}",
                    "user_id": f"user_{i % 10}",
                    "group_ids": [f"group_{i % 5}"],
                    "permissions": ["read"]
                }
                test_data.append(entity)
            
            collection.insert(test_data)
            collection.flush()
        
        yield collection
        
        try:
            if collection.exists():
                collection.drop()
        except:
            pass

    def test_index_creation_and_search_performance(self, indexed_collection, milvus_client):
        """Test index creation and search performance."""
        index_manager = IndexManager(milvus_client)
        
        # Test different index types
        index_types = [
            (IndexType.FLAT, {}),
            (IndexType.IVF_FLAT, {"nlist": 128}),
            (IndexType.HNSW, {"M": 8, "efConstruction": 128})
        ]
        
        for index_type, params in index_types:
            # Create index
            config = index_manager.create_index_config(
                config_name=f"test_{index_type.value}",
                index_type=index_type,
                metric_type=MetricType.L2,
                **params
            )
            
            result = index_manager.create_index(
                collection=indexed_collection,
                index_config=config,
                wait_for_completion=True
            )
            
            assert result["status"] == "success"
            assert result["index_type"] == index_type.value
            
            # Load collection for search
            indexed_collection.load()
            
            # Test search performance
            query_vectors = [[random.random() for _ in range(128)] for _ in range(10)]
            
            start_time = time.time()
            search_result = indexed_collection.search(
                query_vectors=query_vectors,
                limit=10,
                search_params={"metric_type": "L2", "params": {}}
            )
            end_time = time.time()
            
            search_time = end_time - start_time
            
            # Search should complete quickly with index
            assert search_time < 5.0  # 5 seconds for 10 queries
            assert len(search_result.hits) <= 10
            
            # Drop index for next test
            index_manager.drop_index(indexed_collection)

    def test_index_optimization(self, indexed_collection, milvus_client):
        """Test automatic index optimization."""
        index_manager = IndexManager(milvus_client)
        
        # Test auto-optimization
        result = index_manager.auto_optimize_index(indexed_collection)
        
        assert result["status"] == "success"
        assert "recommended_index" in result
        assert "build_time" in result
        
        # Verify index was created
        indexes = index_manager.list_indexes(indexed_collection)
        assert len(indexes) > 0
        
        # Test search performance after optimization
        indexed_collection.load()
        
        query_vector = [random.random() for _ in range(128)]
        start_time = time.time()
        search_result = indexed_collection.search(
            query_vectors=[query_vector],
            limit=5,
            search_params={"metric_type": "L2", "params": {}}
        )
        end_time = time.time()
        
        search_time = end_time - start_time
        assert search_time < 1.0  # Should be very fast with optimized index


@pytest.mark.integration
@pytest.mark.milvus
class TestMilvusSearchIntegration:
    """Integration tests for Milvus search functionality."""
    
    @pytest.fixture
    def search_collection(self, milvus_client, skip_if_no_milvus):
        """Collection for search testing."""
        skip_if_no_milvus()
        
        schema = create_default_rag_schema(
            collection_name=f"search_test_{int(time.time())}",
            vector_dim=128
        )
        
        collection = MilvusCollection(milvus_client, schema)
        
        if not collection.exists():
            collection.create()
            
            # Insert diverse test data
            test_data = []
            categories = ["technology", "science", "business", "health", "education"]
            
            for i in range(500):
                category = categories[i % len(categories)]
                entity = {
                    "vector": [random.random() for _ in range(128)],
                    "text": f"{category} document {i}",
                    "category": category,
                    "user_id": f"user_{i % 20}",
                    "group_ids": [f"group_{i % 10}"],
                    "permissions": ["read"]
                }
                test_data.append(entity)
            
            collection.insert(test_data)
            collection.flush()
            
            # Create index for better search performance
            from src.milvus.index import IndexManager, IndexType, MetricType
            index_manager = IndexManager(milvus_client)
            config = index_manager.create_index_config(
                config_name="search_test_index",
                index_type=IndexType.IVF_FLAT,
                metric_type=MetricType.L2,
                nlist=64
            )
            index_manager.create_index(collection, index_config=config)
            
            collection.load()
        
        yield collection
        
        try:
            if collection.exists():
                collection.drop()
        except:
            pass

    def test_search_strategies(self, search_collection, milvus_client):
        """Test different search strategies."""
        search_manager = SearchManager(milvus_client)
        
        query_vector = [random.random() for _ in range(128)]
        
        # Test different search strategies
        strategies = [
            SearchStrategy.FAST,
            SearchStrategy.BALANCED,
            SearchStrategy.EXACT,
            SearchStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            query = SearchQuery(
                vectors=[query_vector],
                limit=10,
                strategy=strategy
            )
            
            start_time = time.time()
            result = search_manager.search(search_collection, query)
            end_time = time.time()
            
            search_time = end_time - start_time
            
            assert len(result.hits) <= 10
            assert result.query_time > 0
            assert search_time < 5.0  # Should complete within 5 seconds

    def test_search_with_filters(self, search_collection, milvus_client):
        """Test search with metadata filters."""
        search_manager = SearchManager(milvus_client)
        
        query_vector = [random.random() for _ in range(128)]
        
        # Test category filter
        query = SearchQuery(
            vectors=[query_vector],
            limit=10,
            filter_expr='category == "technology"',
            output_fields=["text", "category"]
        )
        
        result = search_manager.search(search_collection, query)
        
        assert len(result.hits) <= 10
        for hit in result.hits:
            if "category" in hit:  # Field may not always be included
                assert hit["category"] == "technology"

    def test_batch_search(self, search_collection, milvus_client):
        """Test batch search operations."""
        search_manager = SearchManager(milvus_client)
        
        # Create multiple queries
        queries = []
        for i in range(5):
            query_vector = [random.random() for _ in range(128)]
            query = SearchQuery(
                vectors=[query_vector],
                limit=5,
                strategy=SearchStrategy.BALANCED
            )
            queries.append(query)
        
        # Test batch search
        start_time = time.time()
        results = search_manager.batch_search(search_collection, queries, parallel=True)
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        assert len(results) == 5
        for result in results:
            assert len(result.hits) <= 5
        
        # Batch search should be efficient
        assert batch_time < 10.0

    def test_search_caching(self, search_collection, milvus_client):
        """Test search result caching."""
        search_manager = SearchManager(milvus_client, enable_cache=True, cache_size=100)
        
        query_vector = [random.random() for _ in range(128)]
        query = SearchQuery(
            vectors=[query_vector],
            limit=5,
            strategy=SearchStrategy.BALANCED
        )
        
        # First search - cache miss
        start_time = time.time()
        result1 = search_manager.search(search_collection, query)
        first_time = time.time() - start_time
        
        # Second search - cache hit
        start_time = time.time()
        result2 = search_manager.search(search_collection, query)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert len(result1.hits) == len(result2.hits)
        
        # Second search should be faster (cache hit)
        assert second_time < first_time
        
        # Check cache statistics
        cache_stats = search_manager.get_cache_stats()
        assert cache_stats["cache_size"] > 0

    def test_hybrid_search(self, search_collection, milvus_client):
        """Test hybrid search combining vector and scalar search."""
        search_manager = SearchManager(milvus_client)
        
        query_vector = [random.random() for _ in range(128)]
        vector_query = SearchQuery(
            vectors=[query_vector],
            limit=20,
            strategy=SearchStrategy.BALANCED
        )
        
        # Define scalar filters
        scalar_filters = [
            'category == "technology"',
            'category == "science"'
        ]
        
        # Perform hybrid search
        result = search_manager.hybrid_search(
            collection=search_collection,
            vector_query=vector_query,
            scalar_filters=scalar_filters,
            fusion_method="rrf"
        )
        
        assert len(result.hits) > 0
        assert "fused_score" in result.hits[0]  # Should have fusion scores


@pytest.mark.integration
@pytest.mark.milvus
@pytest.mark.slow
class TestMilvusPerformanceIntegration:
    """Performance integration tests for Milvus."""
    
    def test_large_scale_operations(self, milvus_client, skip_if_no_milvus):
        """Test large-scale vector operations."""
        skip_if_no_milvus()
        
        # Create collection for large-scale testing
        schema = create_default_rag_schema(
            collection_name=f"perf_test_{int(time.time())}",
            vector_dim=256,
            enable_rbac=False
        )
        
        collection = MilvusCollection(milvus_client, schema)
        
        try:
            collection.create()
            
            # Test large batch insertion
            batch_size = 1000
            total_entities = 10000
            
            start_time = time.time()
            
            for batch_start in range(0, total_entities, batch_size):
                batch_data = []
                for i in range(batch_start, min(batch_start + batch_size, total_entities)):
                    entity = {
                        "vector": [random.random() for _ in range(256)],
                        "text": f"Performance test document {i}",
                        "user_id": f"user_{i % 100}",
                        "group_ids": [f"group_{i % 20}"],
                        "permissions": ["read"]
                    }
                    batch_data.append(entity)
                
                collection.insert(batch_data)
            
            collection.flush()
            insertion_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert insertion_time < 60.0  # 1 minute for 10k entities
            
            # Verify entity count
            assert collection.get_entity_count() == total_entities
            
            # Create index for search performance
            from src.milvus.index import IndexManager, IndexType, MetricType
            index_manager = IndexManager(milvus_client)
            
            config = index_manager.create_index_config(
                config_name="perf_index",
                index_type=IndexType.IVF_FLAT,
                metric_type=MetricType.L2,
                nlist=512
            )
            
            start_time = time.time()
            index_manager.create_index(collection, index_config=config)
            index_time = time.time() - start_time
            
            # Index creation should be reasonable
            assert index_time < 120.0  # 2 minutes for indexing
            
            collection.load()
            
            # Test search performance
            search_manager = SearchManager(milvus_client)
            query_vectors = [[random.random() for _ in range(256)] for _ in range(100)]
            
            start_time = time.time()
            
            for query_vector in query_vectors:
                query = SearchQuery(
                    vectors=[query_vector],
                    limit=10,
                    strategy=SearchStrategy.BALANCED
                )
                result = search_manager.search(collection, query)
                assert len(result.hits) <= 10
            
            search_time = time.time() - start_time
            
            # 100 searches should complete quickly
            assert search_time < 30.0  # 30 seconds for 100 searches
            
        finally:
            if collection.exists():
                collection.drop()

    def test_concurrent_operations(self, milvus_client, skip_if_no_milvus):
        """Test concurrent Milvus operations."""
        skip_if_no_milvus()
        
        import threading
        
        # Create collection for concurrency testing
        schema = create_default_rag_schema(
            collection_name=f"concurrent_test_{int(time.time())}",
            vector_dim=128
        )
        
        collection = MilvusCollection(milvus_client, schema)
        
        try:
            collection.create()
            
            # Prepare initial data
            initial_data = []
            for i in range(100):
                entity = {
                    "vector": [random.random() for _ in range(128)],
                    "text": f"Initial document {i}",
                    "user_id": f"user_{i}",
                    "group_ids": [f"group_{i % 5}"],
                    "permissions": ["read"]
                }
                initial_data.append(entity)
            
            collection.insert(initial_data)
            collection.flush()
            collection.load()
            
            # Test concurrent searches
            search_results = []
            search_errors = []
            
            def concurrent_search(thread_id):
                try:
                    search_manager = SearchManager(milvus_client)
                    
                    for i in range(10):
                        query_vector = [random.random() for _ in range(128)]
                        query = SearchQuery(
                            vectors=[query_vector],
                            limit=5,
                            strategy=SearchStrategy.FAST
                        )
                        
                        result = search_manager.search(collection, query)
                        search_results.append((thread_id, i, len(result.hits)))
                        
                        time.sleep(0.1)  # Small delay to simulate real usage
                        
                except Exception as e:
                    search_errors.append((thread_id, str(e)))
            
            # Start multiple search threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=concurrent_search, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check results
            assert len(search_errors) == 0  # No errors should occur
            assert len(search_results) == 50  # 5 threads * 10 searches each
            
            # All searches should return results
            for thread_id, search_id, hit_count in search_results:
                assert hit_count <= 5
                assert hit_count >= 0
            
        finally:
            if collection.exists():
                collection.drop()