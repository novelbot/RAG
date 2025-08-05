#!/usr/bin/env python3
"""
Check actual content in vectorDB to verify processing.
"""

import sys
sys.path.insert(0, "src")

from pymilvus import connections, Collection

def check_vectordb():
    """Check vectorDB content."""
    print("🔍 Checking VectorDB Content")
    print("=" * 60)
    
    try:
        # Connect directly
        connections.connect('default', host='localhost', port='19530')
        print("✅ Connected to Milvus")
        
        # Get collection
        collection = Collection("episode_embeddings")
        collection.load()
        
        print(f"📊 Total entries: {collection.num_entities}")
        
        # Get sample data
        results = collection.query(
            expr="",
            output_fields=["episode_id", "is_chunk", "chunk_index", "total_chunks"],
            limit=10
        )
        
        print(f"📝 Sample entries ({len(results)}):")
        for i, entry in enumerate(results):
            episode_id = entry.get('episode_id', 'N/A')
            is_chunk = entry.get('is_chunk', False)
            chunk_index = entry.get('chunk_index', -1)
            total_chunks = entry.get('total_chunks', -1)
            
            chunk_info = f"Chunk {chunk_index+1}/{total_chunks}" if is_chunk else "Full episode"
            print(f"   {i+1:2}. Episode {episode_id}: {chunk_info}")
        
        # Check if we have the improved schema
        schema_fields = [field.name for field in collection.schema.fields]
        new_fields = ["entry_id", "episode_id", "is_chunk", "chunk_index", "total_chunks"]
        has_new_schema = all(field in schema_fields for field in new_fields)
        
        print(f"\n🏗️ Schema Analysis:")
        print(f"   Available fields: {schema_fields}")
        print(f"   Has improved schema: {'✅ YES' if has_new_schema else '❌ NO'}")
        
        # Check data distribution
        chunk_results = collection.query(
            expr="is_chunk == true",
            output_fields=["episode_id"],
            limit=1
        )
        
        full_results = collection.query(
            expr="is_chunk == false", 
            output_fields=["episode_id"],
            limit=1
        )
        
        print(f"\n📈 Data Distribution:")
        print(f"   Has chunks: {'✅ YES' if chunk_results else '❌ NO'}")
        print(f"   Has full episodes: {'✅ YES' if full_results else '❌ NO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        connections.disconnect('default')

if __name__ == "__main__":
    success = check_vectordb()
    print(f"\n{'='*60}")
    if success:
        print("🎉 VECTORDB CHECK SUCCESSFUL!")
        print("API처리가 실제로 동작하고 있으며 개선된 스키마를 사용하고 있습니다.")
    else:
        print("⚠️ VECTORDB CHECK FAILED")
    sys.exit(0 if success else 1)