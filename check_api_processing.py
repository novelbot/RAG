#!/usr/bin/env python3
"""
Check if API processing is actually working.
"""

import asyncio
import sys
import time
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from sqlalchemy import text

async def check_processing_status():
    """Check if API processing is actually working."""
    print("🔍 Checking API Processing Status")
    print("=" * 60)
    
    try:
        config = get_config()
        
        # Check Milvus connection and collection status
        print("1. 📡 Checking Milvus Connection...")
        try:
            milvus_client = MilvusClient(config.milvus)
            collections = milvus_client.list_collections()
            collection_names = [c.name for c in collections]
            
            print(f"   Available collections: {collection_names}")
            
            if "episode_embeddings" in collection_names:
                print("   ✅ episode_embeddings collection exists")
                
                # Get collection stats
                stats = milvus_client.get_collection_stats("episode_embeddings")
                entity_count = stats.get('row_count', 0)
                print(f"   📊 Current entity count: {entity_count}")
                
                if entity_count > 0:
                    print("   ✅ Collection has data")
                    
                    # Sample some data to verify schema
                    sample_results = milvus_client.query(
                        collection_name="episode_embeddings",
                        expr="",
                        output_fields=["episode_id", "is_chunk", "chunk_index"],
                        limit=5
                    )
                    
                    print(f"   📝 Sample entries: {len(sample_results)}")
                    for i, entry in enumerate(sample_results[:3]):
                        print(f"      {i+1}. Episode ID: {entry.get('episode_id')}, Is Chunk: {entry.get('is_chunk')}, Chunk Index: {entry.get('chunk_index')}")
                else:
                    print("   ⚠️  Collection is empty - processing may not be working")
            else:
                print("   ❌ episode_embeddings collection not found")
                
        except Exception as e:
            print(f"   ❌ Milvus connection failed: {e}")
            
        # Check database connection
        print("\n2. 🗄️ Checking Database Connection...")
        try:
            db_manager = DatabaseManager(config.database)
            
            with db_manager.get_connection() as conn:
                # Get total episode count
                result = conn.execute(text("SELECT COUNT(*) as total FROM episode"))
                total_episodes = result.scalar()
                print(f"   📊 Total episodes in RDB: {total_episodes}")
                
                # Get sample episode
                result = conn.execute(text("SELECT episode_id, episode_title, LENGTH(content) as content_length FROM episode LIMIT 3"))
                episodes = result.fetchall()
                
                print("   📝 Sample episodes:")
                for ep in episodes:
                    print(f"      Episode {ep.episode_id}: '{ep.episode_title}' ({ep.content_length} chars)")
                    
            print("   ✅ Database connection working")
                    
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            
        # Check background processing by monitoring for a bit
        print("\n3. ⏳ Monitoring Processing Activity...")
        
        if "episode_embeddings" in collection_names:
            initial_count = milvus_client.get_collection_stats("episode_embeddings").get('row_count', 0)
            print(f"   Initial count: {initial_count}")
            
            print("   Waiting 30 seconds to see if count increases...")
            await asyncio.sleep(30)
            
            final_count = milvus_client.get_collection_stats("episode_embeddings").get('row_count', 0)
            print(f"   Final count: {final_count}")
            
            if final_count > initial_count:
                print("   ✅ Processing is active - count increased!")
                increase = final_count - initial_count
                print(f"   📈 Added {increase} new entries in 30s")
                return True
            else:
                print("   ❌ No processing activity detected")
                return False
        else:
            print("   ❌ Cannot monitor - collection doesn't exist")
            return False
            
    except Exception as e:
        print(f"❌ Check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    result = asyncio.run(check_processing_status())
    print(f"\n{'='*60}")
    if result:
        print("🎉 API PROCESSING IS WORKING!")
    else:
        print("⚠️ API PROCESSING ISSUES DETECTED")
    sys.exit(0 if result else 1)