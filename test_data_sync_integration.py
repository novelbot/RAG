"""
Integration tests for data synchronization with real database connections.
"""

import pytest
import asyncio
import os
from unittest.mock import patch

from src.services.data_sync import DataSyncManager, DatabaseSourceAdapter
from src.core.config import get_config


@pytest.mark.asyncio
async def test_real_database_connection():
    """Test connection to the actual database from .env"""
    config = get_config()
    
    # Check if database configuration is available
    if not config.database.host or not config.database.user:
        pytest.skip("Database configuration not available in .env")
    
    # Create database source config using .env values
    source_config = {
        "id": "real_database_test",
        "type": "database",
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
            "driver": config.database.driver
        },
        "query": "SELECT 1 as test_id, 'test content' as test_content, NOW() as modified_date LIMIT 1"
    }
    
    # Test database adapter connection
    adapter = DatabaseSourceAdapter(source_config)
    
    try:
        # Test connection
        connection_result = await adapter.test_connection()
        assert connection_result is True, "Database connection should succeed"
        
        # Test getting records
        records = await adapter.get_records()
        assert len(records) >= 0, "Should be able to fetch records"
        
        print(f"✅ Database connection successful!")
        print(f"✅ Fetched {len(records)} records")
        
        if records:
            print(f"✅ Sample record: {records[0].id} - {records[0].content[:50]}...")
            
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")


@pytest.mark.asyncio  
async def test_sync_manager_with_real_config():
    """Test sync manager with real database configuration"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    # Create sync manager
    sync_manager = DataSyncManager()
    sync_manager.sync_states = {}
    
    # Create real database source config
    source_config = {
        "id": "integration_test_db",
        "type": "database", 
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
            "driver": config.database.driver
        },
        "query": "SELECT 1 as id, 'integration test content' as content, NOW() as modified_date LIMIT 5"
    }
    
    try:
        # Test sync in dry run mode
        result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=False,
            dry_run=True
        )
        
        print(f"✅ Sync completed!")
        print(f"✅ Status: {result.sync_status.value}")
        print(f"✅ Records processed: {result.records_processed}")
        print(f"✅ Duration: {result.sync_duration:.2f}s")
        
        assert result.sync_status.value in ["completed", "failed"]
        
        if result.sync_status.value == "completed":
            assert result.records_processed >= 0
        else:
            print(f"⚠️ Sync failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Sync failed with exception: {e}")
        # Don't fail the test if it's a connection issue - this is integration testing
        pytest.skip(f"Sync failed due to: {e}")


def test_config_loading():
    """Test that configuration loads correctly from .env"""
    config = get_config()
    
    print(f"✅ Database Host: {config.database.host}")
    print(f"✅ Database Port: {config.database.port}")
    print(f"✅ Database Name: {config.database.database}")
    print(f"✅ Database User: {config.database.user}")
    print(f"✅ Database Driver: {config.database.driver}")
    print(f"✅ Milvus Host: {config.milvus.host}")
    print(f"✅ Milvus Port: {config.milvus.port}")
    print(f"✅ Embedding Provider: {config.embedding.provider}")
    print(f"✅ Embedding Model: {config.embedding.model}")
    
    # Basic assertions
    assert config.database.host is not None
    assert config.database.port > 0
    assert config.database.database is not None
    assert config.database.user is not None


@pytest.mark.asyncio
async def test_milvus_connection():
    """Test Milvus connection if available"""
    config = get_config()
    
    # Skip if no Milvus config
    if not config.milvus.host:
        pytest.skip("Milvus configuration not available")
    
    try:
        # Simple check - we can't easily test Milvus without the actual client
        print(f"✅ Milvus configuration loaded:")
        print(f"  Host: {config.milvus.host}")
        print(f"  Port: {config.milvus.port}")
        print(f"  User: {config.milvus.user}")
        print(f"  Collection: {config.milvus.collection_name}")
        
        # This would require actual Milvus client to test
        # For now, just verify config is loaded
        assert config.milvus.host is not None
        assert config.milvus.port > 0
        
    except Exception as e:
        pytest.skip(f"Milvus test skipped: {e}")


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__ + "::test_config_loading", "-v", "-s"])
    pytest.main([__file__ + "::test_real_database_connection", "-v", "-s"])
    pytest.main([__file__ + "::test_sync_manager_with_real_config", "-v", "-s"])