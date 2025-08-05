#!/usr/bin/env python3
"""
Debug the content mismatch issue.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from sqlalchemy import text

def debug_content_mismatch():
    """Debug why content doesn't match between RDB and VectorDB."""
    print("🔍 Debugging Content Mismatch Issue")
    print("=" * 60)
    
    try:
        config = get_config()
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        db_manager = DatabaseManager(config.database)
        
        # Test one specific episode in detail
        test_episode_id = 239
        
        print(f"🎯 Testing Episode {test_episode_id}")
        print("-" * 40)
        
        # Get RDB content
        with db_manager.get_connection() as conn:
            rdb_result = conn.execute(text("""
                SELECT episode_id, novel_id, episode_number, episode_title, content
                FROM episode 
                WHERE episode_id = :episode_id
            """), {'episode_id': test_episode_id})
            rdb_episode = rdb_result.fetchone()
        
        if not rdb_episode:
            print(f"❌ Episode {test_episode_id} not found in RDB")
            return False
        
        print(f"📊 RDB Episode {test_episode_id}:")
        print(f"   Title: {rdb_episode.episode_title}")
        print(f"   Content length: {len(rdb_episode.content)} chars")
        print(f"   First 200 chars: {rdb_episode.content[:200]}...")
        print(f"   Last 200 chars: ...{rdb_episode.content[-200:]}")
        
        # Get VectorDB content
        collection = milvus_client.get_collection("episode_embeddings")
        collection.load()
        
        vector_results = collection.query(
            expr=f"episode_id == {test_episode_id}",
            output_fields=["episode_id", "content", "is_chunk", "chunk_index", "episode_title"],
            limit=10
        )
        
        print(f"\n📊 VectorDB Results ({len(vector_results)} entries):")
        
        for i, entry in enumerate(vector_results):
            print(f"\n   Entry {i+1}:")
            print(f"     Episode ID: {entry['episode_id']}")
            print(f"     Is Chunk: {entry.get('is_chunk', 'N/A')}")
            print(f"     Chunk Index: {entry.get('chunk_index', 'N/A')}")
            print(f"     Title: {entry['episode_title']}")
            print(f"     Content length: {len(entry['content'])} chars")
            print(f"     First 200 chars: {entry['content'][:200]}...")
            print(f"     Last 200 chars: ...{entry['content'][-200:]}")
            
            # Check if this chunk content exists in RDB content
            if entry['content'] in rdb_episode.content:
                print(f"     ✅ Content FOUND in RDB episode")
            else:
                print(f"     ❌ Content NOT FOUND in RDB episode")
                
                # Check if it's a substring issue
                rdb_words = set(rdb_episode.content.split())
                vector_words = set(entry['content'].split())
                overlap = len(rdb_words & vector_words)
                total_vector_words = len(vector_words)
                
                if total_vector_words > 0:
                    overlap_pct = (overlap / total_vector_words) * 100
                    print(f"     Word overlap: {overlap}/{total_vector_words} ({overlap_pct:.1f}%)")
                    
                    if overlap_pct < 50:
                        print(f"     🚨 LOW OVERLAP - This might be content from a different episode!")
        
        # Cross-check: Search for this content in other RDB episodes
        print(f"\n🔍 Cross-checking VectorDB content against other RDB episodes...")
        
        if vector_results:
            test_vector_content = vector_results[0]['content'][:100]  # First 100 chars
            
            with db_manager.get_connection() as conn:
                search_result = conn.execute(text("""
                    SELECT episode_id, episode_title, SUBSTRING(content, 1, 100) as content_preview
                    FROM episode 
                    WHERE content LIKE :search_pattern
                    LIMIT 5
                """), {'search_pattern': f'%{test_vector_content[:50]}%'})
                matches = search_result.fetchall()
            
            print(f"   Found {len(matches)} RDB episodes with similar content:")
            for match in matches:
                print(f"     Episode {match.episode_id}: {match.episode_title}")
                if match.episode_id != test_episode_id:
                    print(f"       🚨 WRONG EPISODE! VectorDB content belongs to Episode {match.episode_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    debug_content_mismatch()