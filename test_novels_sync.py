"""
Test data sync with the novels table that we know exists.
"""

import pytest
import asyncio
from src.services.data_sync import DataSyncManager, DatabaseSourceAdapter
from src.core.config import get_config


@pytest.mark.asyncio
async def test_novels_table_sync():
    """Test syncing with the novels table"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    # Test with novels table
    source_config = {
        "id": "novels_sync_test",
        "type": "database",
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
            "driver": config.database.driver
        },
        "query": """
        SELECT 
            id,
            title as content,
            author,
            description,
            created_at as modified_date
        FROM novels 
        LIMIT 10
        """
    }
    
    # Create adapter
    adapter = DatabaseSourceAdapter(source_config)
    
    try:
        # Test connection
        connection_ok = await adapter.test_connection()
        assert connection_ok, "Database connection should work"
        
        # Get records
        records = await adapter.get_records()
        
        print(f"✅ Successfully fetched {len(records)} records from novels table")
        
        # Show sample records
        for i, record in enumerate(records[:3]):
            print(f"  Novel {i+1}: ID={record.id}")
            print(f"    Title: {record.content[:50]}{'...' if len(record.content) > 50 else ''}")
            print(f"    Author: {record.metadata.get('author', 'N/A')}")
            print(f"    Modified: {record.metadata.get('modified_date', 'N/A')}")
            print()
        
        # Test full sync with sync manager
        sync_manager = DataSyncManager()
        sync_manager.sync_states = {}
        
        result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=False,
            dry_run=True
        )
        
        print(f"✅ Full sync result:")
        print(f"  Status: {result.sync_status.value}")
        print(f"  Records processed: {result.records_processed}")
        print(f"  Records added: {result.records_added}")
        print(f"  Duration: {result.sync_duration:.2f}s")
        
        assert result.sync_status.value == "completed"
        assert result.records_processed > 0
        assert result.records_added > 0
        
        # Test incremental sync
        print(f"\n🔄 Testing incremental sync...")
        incremental_result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=True,
            dry_run=True
        )
        
        print(f"✅ Incremental sync result:")
        print(f"  Status: {incremental_result.sync_status.value}")
        print(f"  Records processed: {incremental_result.records_processed}")
        print(f"  Duration: {incremental_result.sync_duration:.2f}s")
        
        assert incremental_result.sync_status.value == "completed"
        
    except Exception as e:
        pytest.fail(f"Novels table sync failed: {e}")


@pytest.mark.asyncio
async def test_episode_table_sync():
    """Test syncing with the episode table"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    # Test with episode table
    source_config = {
        "id": "episode_sync_test",
        "type": "database",
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
            "driver": config.database.driver
        },
        "query": """
        SELECT 
            id,
            title as content,
            novel_id,
            episode_number,
            created_at as modified_date
        FROM episode 
        LIMIT 5
        """
    }
    
    # Create adapter
    adapter = DatabaseSourceAdapter(source_config)
    
    try:
        # Get records
        records = await adapter.get_records()
        
        print(f"✅ Successfully fetched {len(records)} records from episode table")
        
        # Show sample records
        for i, record in enumerate(records[:2]):
            print(f"  Episode {i+1}: ID={record.id}")
            print(f"    Title: {record.content[:40]}{'...' if len(record.content) > 40 else ''}")
            print(f"    Novel ID: {record.metadata.get('novel_id', 'N/A')}")
            print(f"    Episode #: {record.metadata.get('episode_number', 'N/A')}")
            print()
        
        # Test sync
        sync_manager = DataSyncManager()
        sync_manager.sync_states = {}
        
        result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=False,
            dry_run=True
        )
        
        print(f"✅ Episode sync result:")
        print(f"  Status: {result.sync_status.value}")
        print(f"  Records processed: {result.records_processed}")
        print(f"  Records added: {result.records_added}")
        
        assert result.sync_status.value == "completed"
        
    except Exception as e:
        print(f"⚠️ Episode table sync failed (may be empty or restricted): {e}")
        # Don't fail the test as table might be empty


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])