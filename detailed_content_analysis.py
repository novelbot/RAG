#!/usr/bin/env python3
"""
Detailed content analysis to understand the mismatch.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from sqlalchemy import text

def detailed_content_analysis():
    """Analyze content at character level."""
    print("🔬 Detailed Content Analysis")
    print("=" * 60)
    
    try:
        config = get_config()
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        db_manager = DatabaseManager(config.database)
        
        test_episode_id = 239
        
        # Get RDB content
        with db_manager.get_connection() as conn:
            rdb_result = conn.execute(text("""
                SELECT content FROM episode WHERE episode_id = :episode_id
            """), {'episode_id': test_episode_id})
            rdb_episode = rdb_result.fetchone()
        
        # Get VectorDB content
        collection = milvus_client.get_collection("episode_embeddings")
        collection.load()
        
        vector_results = collection.query(
            expr=f"episode_id == {test_episode_id}",
            output_fields=["content", "chunk_index"],
            limit=10
        )
        
        # Sort by chunk index
        vector_results.sort(key=lambda x: x.get('chunk_index', 0))
        
        print(f"📊 Analysis for Episode {test_episode_id}:")
        print(f"   RDB content length: {len(rdb_episode.content)} chars")
        print(f"   VectorDB chunks: {len(vector_results)}")
        
        # Reconstruct content from chunks
        reconstructed_content = ""
        for chunk in vector_results:
            reconstructed_content += chunk['content']
        
        print(f"   Reconstructed length: {len(reconstructed_content)} chars")
        
        # Character-by-character comparison
        rdb_content = rdb_episode.content
        
        print(f"\n🔍 Character-level comparison:")
        
        # Find first difference
        min_length = min(len(rdb_content), len(reconstructed_content))
        first_diff = None
        
        for i in range(min_length):
            if rdb_content[i] != reconstructed_content[i]:
                first_diff = i
                break
        
        if first_diff is not None:
            print(f"   First difference at position: {first_diff}")
            print(f"   RDB char: '{rdb_content[first_diff]}' (ord: {ord(rdb_content[first_diff])})")
            print(f"   Vector char: '{reconstructed_content[first_diff]}' (ord: {ord(reconstructed_content[first_diff])})")
            
            # Show context around difference
            start = max(0, first_diff - 50)
            end = min(len(rdb_content), first_diff + 50)
            
            print(f"\n   Context around difference:")
            print(f"   RDB:    ...{rdb_content[start:end]}...")
            print(f"   Vector: ...{reconstructed_content[start:end]}...")
        else:
            if len(rdb_content) == len(reconstructed_content):
                print(f"   ✅ Content matches perfectly!")
            else:
                print(f"   Length difference: RDB={len(rdb_content)}, Vector={len(reconstructed_content)}")
                longer = "RDB" if len(rdb_content) > len(reconstructed_content) else "Vector"
                print(f"   {longer} is longer by {abs(len(rdb_content) - len(reconstructed_content))} chars")
        
        # Check if chunks overlap properly
        print(f"\n🧩 Chunk Analysis:")
        rdb_pos = 0
        
        for i, chunk in enumerate(vector_results):
            chunk_content = chunk['content']
            chunk_index = chunk.get('chunk_index', i)
            
            # Try to find this chunk in RDB content
            found_pos = rdb_content.find(chunk_content)
            
            print(f"   Chunk {chunk_index} ({len(chunk_content)} chars):")
            
            if found_pos >= 0:
                print(f"     ✅ Found at position {found_pos}")
                if found_pos >= rdb_pos:
                    print(f"     ✅ Correct sequence (expected >= {rdb_pos})")
                    rdb_pos = found_pos + len(chunk_content)
                else:
                    print(f"     ⚠️ Out of sequence (expected >= {rdb_pos})")
            else:
                print(f"     ❌ Not found in RDB content")
                
                # Check for partial matches
                words_in_chunk = chunk_content.split()[:10]  # First 10 words
                partial_search = " ".join(words_in_chunk)
                partial_pos = rdb_content.find(partial_search)
                
                if partial_pos >= 0:
                    print(f"     🟡 Partial match found at position {partial_pos}")
                else:
                    print(f"     ❌ No partial match found")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    detailed_content_analysis()