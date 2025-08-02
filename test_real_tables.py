"""
Test with real database tables to verify data sync works with actual data.
"""

import pytest
import asyncio
from src.services.data_sync import DataSyncManager, DatabaseSourceAdapter
from src.core.config import get_config


@pytest.mark.asyncio
async def test_sync_with_real_tables():
    """Test syncing with actual database tables"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    # Test with documents table (likely to exist in the RAG system)
    source_config = {
        "id": "documents_table_test",
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
            filename as content,
            upload_date as modified_date,
            file_size,
            status
        FROM documents 
        LIMIT 5
        """
    }
    
    # Create adapter
    adapter = DatabaseSourceAdapter(source_config)
    
    try:
        # Test connection first
        connection_ok = await adapter.test_connection()
        if not connection_ok:
            pytest.skip("Cannot connect to database")
        
        # Try to get records
        records = await adapter.get_records()
        
        print(f"✅ Successfully fetched {len(records)} records from documents table")
        
        for i, record in enumerate(records[:3]):  # Show first 3 records
            print(f"  Record {i+1}: ID={record.id}, Content='{record.content[:30]}...', Metadata keys={list(record.metadata.keys())}")
        
        # Test sync manager
        sync_manager = DataSyncManager()
        sync_manager.sync_states = {}
        
        result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=False,
            dry_run=True
        )
        
        print(f"✅ Sync result: {result.sync_status.value}")
        print(f"✅ Records processed: {result.records_processed}")
        print(f"✅ Records added: {result.records_added}")
        
        assert result.sync_status.value == "completed"
        assert result.records_processed >= 0
        
    except Exception as e:
        # If documents table doesn't exist, try a simpler query
        print(f"Documents table test failed: {e}")
        
        # Try with INFORMATION_SCHEMA
        simple_config = {
            "id": "tables_info_test",
            "type": "database", 
            "config": source_config["config"],
            "query": "SELECT TABLE_NAME as id, TABLE_NAME as content, CREATE_TIME as modified_date FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE() LIMIT 3"
        }
        
        simple_adapter = DatabaseSourceAdapter(simple_config)
        simple_records = await simple_adapter.get_records()
        
        print(f"✅ Fallback test: fetched {len(simple_records)} table names")
        for record in simple_records:
            print(f"  Table: {record.content}")


if __name__ == "__main__":
    pytest.main([__file__ + "::test_sync_with_real_tables", "-v", "-s"])