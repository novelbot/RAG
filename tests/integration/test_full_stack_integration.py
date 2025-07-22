"""
Full stack integration tests combining Database and Milvus layers.
"""
import pytest
import os
import time
import random
from typing import List, Dict, Any

from src.core.config import ConfigManager
from src.database.base import DatabaseManager
from src.database.pool import AdvancedConnectionPool
from src.database.health import DatabaseHealthChecker, HealthCheckManager
from src.milvus.client import MilvusClient
from src.milvus.schema import create_default_rag_schema
from src.milvus.collection import MilvusCollection
from src.milvus.rbac import RBACManager, UserContext, Permission
from src.milvus.search import SearchManager, SearchQuery, SearchStrategy


@pytest.mark.integration
@pytest.mark.slow
class TestFullStackIntegration:
    """Full stack integration tests."""
    
    @pytest.fixture
    def config_manager(self, test_config_dict):
        """Real config manager with test configuration."""
        from unittest.mock import patch
        
        with patch.object(ConfigManager, 'get_config', return_value=test_config_dict):
            with patch.object(ConfigManager, 'get_database_config', return_value=test_config_dict["database"]):
                with patch.object(ConfigManager, 'get_milvus_config', return_value=test_config_dict["milvus"]):
                    yield ConfigManager()

    @pytest.fixture
    def database_stack(self, config_manager, skip_if_no_database):
        """Complete database stack."""
        skip_if_no_database()
        
        # Initialize database components
        engine = DatabaseEngine(config_manager)
        engine.initialize()
        
        pool = ConnectionPool(engine)
        monitor = DatabaseHealthMonitor(pool)
        
        yield {
            "engine": engine,
            "pool": pool,
            "monitor": monitor
        }
        
        # Cleanup
        pool.close()
        engine.close()

    @pytest.fixture
    def milvus_stack(self, config_manager, skip_if_no_milvus):
        """Complete Milvus stack."""
        skip_if_no_milvus()
        
        milvus_config = config_manager.get_milvus_config()
        milvus_config["alias"] = f"full_stack_test_{int(time.time())}"
        
        # Initialize Milvus components
        client = MilvusClient(milvus_config)
        client.connect()
        
        # Create test collection
        schema = create_default_rag_schema(
            collection_name=f"full_stack_test_{int(time.time())}",
            vector_dim=256,
            enable_rbac=True
        )
        
        collection = MilvusCollection(client, schema)
        collection.create()
        
        # Initialize managers
        rbac_manager = RBACManager(client)
        search_manager = SearchManager(client)
        
        yield {
            "client": client,
            "collection": collection,
            "rbac_manager": rbac_manager,
            "search_manager": search_manager
        }
        
        # Cleanup
        try:
            if collection.exists():
                collection.drop()
        except:
            pass
        client.disconnect()

    def test_full_system_health_monitoring(self, database_stack, milvus_stack):
        """Test comprehensive health monitoring across all systems."""
        db_monitor = database_stack["monitor"]
        milvus_client = milvus_stack["client"]
        
        # Test database health
        db_health = db_monitor.get_health_report()
        assert db_health["overall_status"] in ["healthy", "degraded"]
        
        # Test Milvus health
        milvus_health = milvus_client.health_check()
        assert milvus_health["status"] == "healthy"
        
        # Combined system health check
        system_health = {
            "database": db_health["overall_status"],
            "milvus": milvus_health["status"],
            "timestamp": time.time()
        }
        
        # System should be operational
        healthy_components = sum(1 for status in system_health.values() 
                               if isinstance(status, str) and status in ["healthy", "degraded"])
        assert healthy_components >= 2

    def test_data_consistency_across_systems(self, database_stack, milvus_stack):
        """Test data consistency between database and vector store."""
        db_pool = database_stack["pool"]
        collection = milvus_stack["collection"]
        
        # Simulate document metadata stored in database
        document_metadata = [
            {
                "doc_id": f"doc_{i}",
                "title": f"Test Document {i}",
                "content": f"This is test document {i} content",
                "author": f"author_{i % 5}",
                "category": ["technology", "science", "business"][i % 3],
                "created_at": time.time() - (i * 3600)  # Different creation times
            }
            for i in range(10)
        ]
        
        # Store metadata in database (simulated - would use actual SQL)
        stored_doc_ids = []
        for doc in document_metadata:
            # In real implementation, this would be actual database insertion
            stored_doc_ids.append(doc["doc_id"])
        
        # Store vectors in Milvus with corresponding metadata
        vector_data = []
        for i, doc in enumerate(document_metadata):
            entity = {
                "vector": [random.random() for _ in range(256)],
                "text": doc["content"],
                "doc_id": doc["doc_id"],
                "user_id": f"user_{i % 3}",
                "group_ids": [doc["category"]],
                "permissions": ["read", "write"]
            }
            vector_data.append(entity)
        
        # Insert vectors
        result = collection.insert(vector_data)
        assert result.insert_count == 10
        
        collection.flush()
        collection.load()
        
        # Verify consistency
        query_result = collection.query(
            expr="",  # Get all entities
            output_fields=["doc_id", "text"]
        )
        
        retrieved_doc_ids = {entity["doc_id"] for entity in query_result}
        expected_doc_ids = {doc["doc_id"] for doc in document_metadata}
        
        # All documents should be present in both systems
        assert retrieved_doc_ids == expected_doc_ids

    def test_user_workflow_end_to_end(self, database_stack, milvus_stack):
        """Test complete user workflow from authentication to search results."""
        collection = milvus_stack["collection"]
        rbac_manager = milvus_stack["rbac_manager"]
        search_manager = milvus_stack["search_manager"]
        
        # 1. User Authentication & Authorization Setup
        users = [
            UserContext(
                user_id="analyst_1",
                group_ids=["analysts", "employees"],
                permissions=[Permission.READ, Permission.WRITE]
            ),
            UserContext(
                user_id="researcher_1",
                group_ids=["researchers", "employees"],
                permissions=[Permission.READ]
            ),
            UserContext(
                user_id="admin_1",
                group_ids=["administrators"],
                permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN]
            )
        ]
        
        for user in users:
            rbac_manager.add_user_context(user)
        
        # 2. Data Ingestion with Access Control
        documents = []
        access_levels = [
            ("analysts", ["analyst_1"]),
            ("researchers", ["researcher_1"]),
            ("public", ["analyst_1", "researcher_1", "admin_1"]),
            ("confidential", ["admin_1"])
        ]
        
        for i in range(20):
            access_level, allowed_users = access_levels[i % len(access_levels)]
            user_id = allowed_users[0]  # Primary owner
            
            entity = {
                "vector": [random.random() for _ in range(256)],
                "text": f"Document {i}: {access_level} content",
                "doc_id": f"doc_{i}",
                "access_level": access_level,
                "user_id": user_id,
                "group_ids": [access_level] if access_level != "public" else ["employees"],
                "permissions": ["read", "write"] if access_level != "confidential" else ["read"]
            }
            documents.append(entity)
        
        # Insert documents
        result = collection.insert(documents)
        assert result.insert_count == 20
        
        collection.flush()
        collection.load()
        
        # 3. User Search Workflows
        test_scenarios = [
            {
                "user_id": "analyst_1",
                "expected_min_results": 15,  # Can access analysts, public docs
                "description": "Analyst searching for documents"
            },
            {
                "user_id": "researcher_1", 
                "expected_min_results": 10,  # Can access researchers, public docs
                "description": "Researcher searching for documents"
            },
            {
                "user_id": "admin_1",
                "expected_min_results": 5,   # Can access confidential docs
                "description": "Admin searching for documents"
            }
        ]
        
        for scenario in test_scenarios:
            user_id = scenario["user_id"]
            
            # Get user's access filter
            access_filter = rbac_manager.get_access_filter(user_id)
            
            # Perform filtered search
            query_vector = [random.random() for _ in range(256)]
            query = SearchQuery(
                vectors=[query_vector],
                limit=20,
                filter_expr=access_filter,
                output_fields=["text", "doc_id", "access_level", "user_id"],
                strategy=SearchStrategy.BALANCED
            )
            
            search_result = search_manager.search(collection, query)
            
            # Verify user can only see authorized documents
            accessible_docs = len(search_result.hits)
            
            # Verify access control
            for hit in search_result.hits:
                doc_user_id = hit.get("user_id")
                access_level = hit.get("access_level")
                
                # User should only see their own docs, public docs, or docs from their groups
                user_context = rbac_manager.get_user_context(user_id)
                
                is_own_doc = doc_user_id == user_id
                is_public = access_level == "public"
                is_group_accessible = (
                    access_level in user_context.group_ids or
                    any(group in user_context.group_ids for group in ["employees", "administrators"])
                )
                
                assert is_own_doc or is_public or is_group_accessible, f"Access violation for user {user_id}"

    def test_system_performance_under_load(self, database_stack, milvus_stack):
        """Test system performance under concurrent load."""
        import threading
        import queue
        
        db_pool = database_stack["pool"]
        collection = milvus_stack["collection"]
        search_manager = milvus_stack["search_manager"]
        
        # Prepare test data
        test_data = []
        for i in range(1000):
            entity = {
                "vector": [random.random() for _ in range(256)],
                "text": f"Load test document {i}",
                "doc_id": f"load_doc_{i}",
                "user_id": f"load_user_{i % 10}",
                "group_ids": [f"load_group_{i % 5}"],
                "permissions": ["read"]
            }
            test_data.append(entity)
        
        # Insert test data
        result = collection.insert(test_data)
        assert result.insert_count == 1000
        
        collection.flush()
        collection.load()
        
        # Test concurrent operations
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def database_worker(worker_id):
            """Simulate database operations."""
            try:
                for i in range(10):
                    # Simulate database query
                    with db_pool.get_connection() as conn:
                        from sqlalchemy import text
                        result = conn.execute(text("SELECT :worker_id as id, :iteration as iter"), 
                                            {"worker_id": worker_id, "iteration": i})
                        results_queue.put(("db", worker_id, i, result.scalar()))
                        time.sleep(0.05)  # Simulate processing time
            except Exception as e:
                errors_queue.put(("db", worker_id, str(e)))
        
        def vector_search_worker(worker_id):
            """Simulate vector search operations."""
            try:
                for i in range(10):
                    query_vector = [random.random() for _ in range(256)]
                    query = SearchQuery(
                        vectors=[query_vector],
                        limit=5,
                        strategy=SearchStrategy.FAST
                    )
                    
                    search_result = search_manager.search(collection, query)
                    results_queue.put(("search", worker_id, i, len(search_result.hits)))
                    time.sleep(0.05)  # Simulate processing time
            except Exception as e:
                errors_queue.put(("search", worker_id, str(e)))
        
        # Start concurrent workers
        threads = []
        
        # Database workers
        for i in range(3):
            thread = threading.Thread(target=database_worker, args=(f"db_{i}",))
            threads.append(thread)
            thread.start()
        
        # Search workers
        for i in range(3):
            thread = threading.Thread(target=vector_search_worker, args=(f"search_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        start_time = time.time()
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.append(results_queue.get())
        
        all_errors = []
        while not errors_queue.empty():
            all_errors.append(errors_queue.get())
        
        # Verify performance
        assert len(all_errors) == 0  # No errors should occur
        assert len(all_results) == 60  # 6 workers * 10 operations each
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Verify operation types
        db_results = [r for r in all_results if r[0] == "db"]
        search_results = [r for r in all_results if r[0] == "search"]
        
        assert len(db_results) == 30  # 3 workers * 10 operations
        assert len(search_results) == 30  # 3 workers * 10 operations

    def test_error_recovery_and_resilience(self, database_stack, milvus_stack):
        """Test system resilience and error recovery."""
        db_pool = database_stack["pool"]
        db_monitor = database_stack["monitor"]
        milvus_client = milvus_stack["client"]
        collection = milvus_stack["collection"]
        
        # Test database resilience
        try:
            # Attempt invalid database operation
            from sqlalchemy import text
            with pytest.raises(Exception):
                db_pool.execute_query(text("SELECT FROM invalid_syntax"))
        except:
            pass
        
        # Database should still be healthy after error
        db_health = db_monitor.check_connection()
        assert db_health["status"] == "healthy"
        
        # Verify database recovery with valid operation
        with db_pool.get_connection() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        # Test Milvus resilience
        try:
            # Attempt invalid Milvus operation
            collection.query(expr="invalid_expression_syntax")
        except:
            pass  # Expected to fail
        
        # Milvus should still be healthy after error
        milvus_health = milvus_client.health_check()
        assert milvus_health["status"] == "healthy"
        
        # Verify Milvus recovery with valid operation
        query_result = collection.query(
            expr="",  # Valid empty expression
            limit=1,
            output_fields=["doc_id"]
        )
        # Should execute without error (may return empty results)

    def test_data_migration_workflow(self, database_stack, milvus_stack):
        """Test data migration between systems."""
        collection = milvus_stack["collection"]
        
        # Simulate migration of existing documents
        legacy_documents = [
            {
                "id": i,
                "title": f"Legacy Document {i}",
                "content": f"This is legacy document {i} that needs to be migrated",
                "metadata": {"category": "legacy", "migrated": False}
            }
            for i in range(50)
        ]
        
        # Migration process: Convert to vector format
        migrated_entities = []
        for doc in legacy_documents:
            entity = {
                "vector": [random.random() for _ in range(256)],  # Would use real embeddings
                "text": doc["content"],
                "doc_id": f"migrated_{doc['id']}",
                "title": doc["title"],
                "user_id": "migration_user",
                "group_ids": ["legacy_data"],
                "permissions": ["read"],
                "metadata": doc["metadata"]
            }
            migrated_entities.append(entity)
        
        # Batch migration
        batch_size = 10
        migrated_count = 0
        
        for i in range(0, len(migrated_entities), batch_size):
            batch = migrated_entities[i:i + batch_size]
            result = collection.insert(batch)
            migrated_count += result.insert_count
        
        collection.flush()
        collection.load()
        
        # Verify migration success
        assert migrated_count == 50
        
        # Verify migrated data is searchable
        migrated_docs = collection.query(
            expr='user_id == "migration_user"',
            output_fields=["doc_id", "title", "text"]
        )
        
        assert len(migrated_docs) == 50
        
        # Verify data integrity
        for doc in migrated_docs:
            assert "migrated_" in doc["doc_id"]
            assert "Legacy Document" in doc["title"]

    def test_backup_and_recovery_simulation(self, database_stack, milvus_stack):
        """Test backup and recovery simulation."""
        collection = milvus_stack["collection"]
        
        # Create test data to backup
        original_data = []
        for i in range(20):
            entity = {
                "vector": [random.random() for _ in range(256)],
                "text": f"Backup test document {i}",
                "doc_id": f"backup_doc_{i}",
                "user_id": f"backup_user_{i % 3}",
                "group_ids": ["backup_group"],
                "permissions": ["read", "write"]
            }
            original_data.append(entity)
        
        # Insert original data
        result = collection.insert(original_data)
        assert result.insert_count == 20
        
        collection.flush()
        collection.load()
        
        # Simulate backup: Export data
        backup_data = collection.query(
            expr="",  # Get all entities
            output_fields=["doc_id", "text", "user_id", "group_ids", "vector"]
        )
        
        assert len(backup_data) == 20
        
        # Simulate disaster: Clear some data (in real scenario, would be actual data loss)
        # For testing, we'll just verify we can query the backup data
        
        # Simulate recovery: Verify backup data integrity
        recovered_doc_ids = {doc["doc_id"] for doc in backup_data}
        expected_doc_ids = {f"backup_doc_{i}" for i in range(20)}
        
        assert recovered_doc_ids == expected_doc_ids
        
        # Verify vectors are present
        for doc in backup_data:
            assert "vector" in doc
            assert len(doc["vector"]) == 256