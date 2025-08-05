#!/usr/bin/env python3
"""
Detailed VectorDB status check with multiple verification methods.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient

def check_vectordb_detailed():
    """Check VectorDB status using multiple methods."""
    print("🔍 Detailed VectorDB Status Check")
    print("=" * 50)
    
    try:
        config = get_config()
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        # Method 1: Check if collection exists
        collection_exists = milvus_client.has_collection("episode_embeddings")
        print(f"\n📋 Collection Status:")
        print(f"   Collection exists: {collection_exists}")
        
        if not collection_exists:
            print("❌ Collection 'episode_embeddings' does not exist")
            return
        
        # Method 2: Get collection object and check entity count
        collection = milvus_client.get_collection("episode_embeddings")
        print(f"\n📊 Collection Object:")
        print(f"   Collection name: {collection.name}")
        
        # Method 3: Check num_entities (fastest way)
        entity_count = collection.num_entities
        print(f"   Entity count (num_entities): {entity_count}")
        
        # Method 4: Check collection status/loading state
        try:
            # Load collection to ensure it's accessible
            collection.load()
            print(f"   Collection load status: ✅ Loaded")
        except Exception as e:
            print(f"   Collection load status: ❌ Failed to load: {e}")
        
        # Method 5: Try to query sample data
        if entity_count > 0:
            try:
                # Try to get a small sample
                sample_data = collection.query(
                    expr="",  # Empty expression = get all
                    output_fields=["episode_id"],
                    limit=1
                )
                print(f"   Sample query result: ✅ Found {len(sample_data)} records")
                if sample_data:
                    print(f"   Sample episode_id: {sample_data[0].get('episode_id', 'N/A')}")
            except Exception as e:
                print(f"   Sample query result: ❌ Query failed: {e}")
        else:
            print(f"   Sample query result: ⚪ No data to query")
        
        # Method 6: Check schema fields (to verify new vs old schema)
        try:
            schema_fields = [field.name for field in collection.schema.fields]
            print(f"\n🏗️ Schema Information:")
            print(f"   Total fields: {len(schema_fields)}")
            print(f"   Fields: {schema_fields}")
            
            # Check for new schema fields
            new_schema_fields = ['entry_id', 'is_chunk', 'chunk_index', 'total_chunks']
            has_new_schema = all(field in schema_fields for field in new_schema_fields)
            print(f"   New schema detected: {'✅ YES' if has_new_schema else '❌ NO (old schema)'}")
            
        except Exception as e:
            print(f"   Schema check failed: {e}")
        
        # Method 7: Collection statistics
        try:
            # Get more detailed stats if available
            print(f"\n📈 Collection Statistics:")
            print(f"   Collection name: {collection.name}")
            print(f"   Total entities: {entity_count}")
            
            if entity_count == 0:
                print(f"   Status: 🔴 EMPTY - No data in collection")
            elif entity_count < 100:
                print(f"   Status: 🟡 MINIMAL - Very little data")
            elif entity_count < 1000:
                print(f"   Status: 🟠 PARTIAL - Some data present")
            else:
                print(f"   Status: 🟢 POPULATED - Significant data present")
                
        except Exception as e:
            print(f"   Statistics failed: {e}")
        
        # Summary
        print(f"\n🎯 Summary:")
        if entity_count == 0:
            print(f"   VectorDB is EMPTY (0 entities)")
            print(f"   Recommendation: Run data ingestion")
        else:
            print(f"   VectorDB has {entity_count} entities")
            print(f"   Status: Data is present")
            
        return entity_count > 0
        
    except Exception as e:
        print(f"❌ Error checking VectorDB: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
            print("🔌 Disconnected from Milvus")

if __name__ == "__main__":
    has_data = check_vectordb_detailed()
    print(f"\n{'='*50}")
    print(f"Result: VectorDB {'HAS DATA' if has_data else 'IS EMPTY'}")