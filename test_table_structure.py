"""
Test to discover the actual structure of tables in the database.
"""

import pytest
import asyncio
from src.services.data_sync import DatabaseSourceAdapter
from src.core.config import get_config


@pytest.mark.asyncio
async def test_discover_table_structures():
    """Discover the actual structure of tables"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    tables_to_check = ["novels", "episode"]
    
    for table_name in tables_to_check:
        print(f"\n🔍 Checking structure of table: {table_name}")
        
        # Get column information
        source_config = {
            "id": f"{table_name}_structure_test",
            "type": "database",
            "config": {
                "host": config.database.host,
                "port": config.database.port,
                "database": config.database.database,
                "user": config.database.user,
                "password": config.database.password,
                "driver": config.database.driver
            },
            "query": f"""
            DESCRIBE {table_name}
            """
        }
        
        adapter = DatabaseSourceAdapter(source_config)
        
        try:
            records = await adapter.get_records()
            print(f"✅ Table '{table_name}' has {len(records)} columns:")
            
            for record in records:
                field_name = record.id
                field_type = record.metadata.get('Type', 'unknown')
                null_allowed = record.metadata.get('Null', 'unknown')
                key_info = record.metadata.get('Key', '')
                default_val = record.metadata.get('Default', '')
                
                print(f"  - {field_name}: {field_type} (Null: {null_allowed}, Key: {key_info}, Default: {default_val})")
            
            # Try to get a sample record
            sample_config = {
                "id": f"{table_name}_sample_test", 
                "type": "database",
                "config": source_config["config"],
                "query": f"SELECT * FROM {table_name} LIMIT 1"
            }
            
            sample_adapter = DatabaseSourceAdapter(sample_config)
            sample_records = await sample_adapter.get_records()
            
            if sample_records:
                print(f"  📄 Sample record from {table_name}:")
                sample = sample_records[0]
                print(f"    ID: {sample.id}")
                print(f"    Content: {sample.content[:100]}{'...' if len(sample.content) > 100 else ''}")
                print(f"    Metadata keys: {list(sample.metadata.keys())}")
            else:
                print(f"  📄 Table {table_name} appears to be empty")
                
        except Exception as e:
            print(f"❌ Failed to check table {table_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])