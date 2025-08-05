#!/usr/bin/env python3
"""
Check current status of VectorDB data.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from sqlalchemy import text

def check_current_status():
    """Check current VectorDB vs RDB status."""
    print("📊 Current RDB-VectorDB Status Check")
    print("=" * 50)
    
    try:
        config = get_config()
        
        # Check VectorDB
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        if milvus_client.has_collection("episode_embeddings"):
            collection = milvus_client.get_collection("episode_embeddings")
            
            # Ensure collection is loaded before checking entities
            try:
                collection.load()
                print("✅ Collection loaded successfully")
            except Exception as e:
                print(f"⚠️ Collection load warning: {e}")
            
            # Get total count
            total_entries = collection.num_entities
            print(f"📈 VectorDB: {total_entries} total entries")
            
            if total_entries > 0:
                # Sample data to check schema
                sample_data = collection.query(
                    expr="",
                    output_fields=["entry_id", "episode_id", "is_chunk"],
                    limit=10
                )
                
                if sample_data and 'is_chunk' in sample_data[0]:
                    # New schema - count chunks vs episodes
                    try:
                        all_data = collection.query(
                            expr="",
                            output_fields=["episode_id", "is_chunk"],
                            limit=5000  # Increased limit to get more complete data
                        )
                        
                        regular_episodes = len([e for e in all_data if not e.get('is_chunk', True)])
                        chunk_entries = len([e for e in all_data if e.get('is_chunk', True)])
                        unique_episodes = len(set(e['episode_id'] for e in all_data))
                        
                        print(f"   - Regular episodes: {regular_episodes}")
                        print(f"   - Chunk entries: {chunk_entries}")
                        print(f"   - Unique episodes: {unique_episodes}")
                        print(f"   - Schema: ✅ New (15 fields)")
                        
                        # More accurate coverage calculation
                        actual_entries_checked = len(all_data)
                        if actual_entries_checked < total_entries:
                            print(f"   - Note: Showing sample of {actual_entries_checked}/{total_entries} entries")
                        
                    except Exception as e:
                        print(f"   - Query error: {e}")
                        print(f"   - Schema: ✅ New (15 fields) - but query failed")
                        unique_episodes = 0  # Fallback
                else:
                    print(f"   - Schema: ❌ Old (11 fields)")
                    unique_episodes = total_entries  # Fallback for old schema
            else:
                print("   - No data found")
                unique_episodes = 0
        else:
            print("❌ VectorDB: Collection not found")
            return
        
        # Check RDB
        db_manager = DatabaseManager(config.database)
        with db_manager.get_connection() as conn:
            total_rdb_episodes = conn.execute(text("SELECT COUNT(*) FROM episode")).scalar()
            total_novels = conn.execute(text("SELECT COUNT(*) FROM novels")).scalar()
        
        print(f"📈 RDB: {total_novels} novels, {total_rdb_episodes} episodes")
        
        # Coverage calculation
        if total_entries > 0 and total_rdb_episodes > 0:
            coverage = (unique_episodes / total_rdb_episodes) * 100
            print(f"\n🎯 Coverage: {unique_episodes}/{total_rdb_episodes} ({coverage:.1f}%)")
            
            if coverage >= 95:
                status = "🌟 COMPLETE: Almost all data processed"
            elif coverage >= 50:
                status = "🔄 IN PROGRESS: Partial processing completed"
            elif coverage >= 10:
                status = "🚀 STARTED: Initial processing done"
            else:
                status = "⏳ MINIMAL: Very little data processed"
            
            print(f"Status: {status}")
            
            if coverage < 95:
                print(f"\n📋 Next Steps:")
                print(f"   Continue: uv run rag-cli data ingest --episode-mode --database --force")
                print(f"   Expected time: ~{max(1, (total_rdb_episodes - unique_episodes) * 2 / 60):.0f} minutes")
        elif total_entries == 0:
            print(f"\n⚠️ No data found in VectorDB")
            print(f"📋 Next Steps:")
            print(f"   Start: uv run rag-cli data ingest --episode-mode --database --force")
            print(f"   Expected time: ~{total_rdb_episodes * 2 / 60:.0f} minutes")
        else:
            print(f"\n❌ Cannot calculate coverage (missing RDB or VectorDB data)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    check_current_status()