#!/usr/bin/env python3
"""
Analyze why some episodes weren't embedded.
Compare RDB novels/episodes with Milvus data to identify gaps.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from src.milvus.client import MilvusClient
from sqlalchemy import text
from collections import defaultdict

def analyze_embedding_gaps():
    """Analyze gaps between RDB and Milvus data."""
    print("🔍 Analyzing Embedding Gaps")
    print("=" * 60)
    
    try:
        config = get_config()
        
        # Initialize connections
        db_manager = DatabaseManager(config.database)
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        print("✅ Connected to RDB and Milvus")
        
        # Get RDB data
        print("\n📊 Analyzing RDB Data...")
        with db_manager.get_connection() as conn:
            # Get all novels with episode counts from RDB
            result = conn.execute(text("""
                SELECT novel_id, COUNT(*) as episode_count
                FROM episode 
                GROUP BY novel_id
                ORDER BY novel_id
            """))
            rdb_novels = {row[0]: row[1] for row in result.fetchall()}
            
            # Get specific episode data
            result = conn.execute(text("""
                SELECT novel_id, episode_id, episode_number
                FROM episode
                ORDER BY novel_id, episode_number
            """))
            rdb_episodes = defaultdict(list)
            for novel_id, episode_id, episode_num in result.fetchall():
                rdb_episodes[novel_id].append({
                    'episode_id': episode_id,
                    'episode_number': episode_num
                })
        
        print(f"- RDB novels: {len(rdb_novels)}")
        print(f"- RDB episodes: {sum(rdb_novels.values())}")
        
        # Get Milvus data
        print("\n🧠 Analyzing Milvus Data...")
        if milvus_client.has_collection("episode_embeddings"):
            collection = milvus_client.get_collection("episode_embeddings")
            
            query_results = collection.query(
                expr="",
                output_fields=["novel_id", "episode_id", "episode_number"],
                limit=1000
            )
            
            milvus_novels = defaultdict(list)
            for ep in query_results:
                milvus_novels[ep['novel_id']].append({
                    'episode_id': ep['episode_id'],
                    'episode_number': ep['episode_number']
                })
            
            print(f"- Milvus novels: {len(milvus_novels)}")
            print(f"- Milvus episodes: {len(query_results)}")
        
        # Analyze gaps
        print("\n🔍 Gap Analysis:")
        print("-" * 60)
        
        missing_novels = []
        partial_novels = []
        complete_novels = []
        
        for novel_id, rdb_count in rdb_novels.items():
            if novel_id not in milvus_novels:
                missing_novels.append((novel_id, rdb_count))
            else:
                milvus_count = len(milvus_novels[novel_id])
                if milvus_count < rdb_count:
                    partial_novels.append((novel_id, rdb_count, milvus_count))
                else:
                    complete_novels.append((novel_id, rdb_count))
        
        print(f"📊 Processing Status:")
        print(f"- Complete novels: {len(complete_novels)}")
        print(f"- Partial novels: {len(partial_novels)}")
        print(f"- Missing novels: {len(missing_novels)}")
        
        # Show missing novels (especially large ones)
        if missing_novels:
            missing_novels.sort(key=lambda x: x[1], reverse=True)
            print(f"\n❌ Missing Novels (Top 10 by episode count):")
            print(f"{'Novel ID':<10} {'Episodes':<10} {'Reason Analysis':<30}")
            print("-" * 50)
            
            for novel_id, episode_count in missing_novels[:10]:
                # Analyze possible reasons
                reason = ""
                if episode_count >= 15:
                    reason = "Large novel - may timeout"
                elif novel_id > 70:
                    reason = "High novel_id - may be newer"
                elif episode_count == 1:
                    reason = "Single episode - may be test data"
                else:
                    reason = "Unknown processing error"
                
                print(f"{novel_id:<10} {episode_count:<10} {reason:<30}")
        
        # Show partial novels
        if partial_novels:
            partial_novels.sort(key=lambda x: x[1], reverse=True)
            print(f"\n⚠️ Partial Novels (Top 10):")
            print(f"{'Novel ID':<10} {'RDB':<6} {'Milvus':<8} {'Missing':<8} {'Issue':<25}")
            print("-" * 60)
            
            for novel_id, rdb_count, milvus_count in partial_novels[:10]:
                missing_count = rdb_count - milvus_count
                missing_pct = (missing_count / rdb_count) * 100
                
                # Check which episodes are missing
                rdb_episode_ids = {ep['episode_id'] for ep in rdb_episodes[novel_id]}
                milvus_episode_ids = {ep['episode_id'] for ep in milvus_novels[novel_id]}
                missing_episode_ids = rdb_episode_ids - milvus_episode_ids
                
                issue = ""
                if missing_pct > 50:
                    issue = "Major processing failure"
                elif len(missing_episode_ids) == missing_count:
                    issue = "Specific episodes failed"
                else:
                    issue = "Partial batch failure"
                
                print(f"{novel_id:<10} {rdb_count:<6} {milvus_count:<8} {missing_count:<8} {issue:<25}")
        
        # Check for processing errors in logs
        print(f"\n📋 Processing Statistics:")
        print(f"- Success rate: {len(milvus_novels)}/{len(rdb_novels)} novels ({len(milvus_novels)/len(rdb_novels)*100:.1f}%)")
        total_rdb_episodes = sum(rdb_novels.values())
        total_milvus_episodes = len(query_results)
        print(f"- Episode success rate: {total_milvus_episodes}/{total_rdb_episodes} episodes ({total_milvus_episodes/total_rdb_episodes*100:.1f}%)")
        
        # Identify potential causes
        print(f"\n🔧 Potential Causes Analysis:")
        
        # Check for large novels that might timeout
        large_missing = [nid for nid, count in missing_novels if count >= 15]
        if large_missing:
            print(f"- Large novels (15+ episodes) that failed: {len(large_missing)} novels")
            print(f"  Novel IDs: {large_missing[:5]}{'...' if len(large_missing) > 5 else ''}")
        
        # Check for novel ID patterns
        high_id_novels = [nid for nid, count in missing_novels if nid > 70]
        if high_id_novels:
            print(f"- High novel_id (>70) that failed: {len(high_id_novels)} novels")
            print(f"  Novel IDs: {high_id_novels[:5]}{'...' if len(high_id_novels) > 5 else ''}")
            print(f"  These may be newer novels added after processing")
        
        # Check embedding processing limits
        if len(missing_novels) > 0:
            print(f"- Processing may have stopped early due to:")
            print(f"  * Timeout or memory limits")
            print(f"  * API rate limiting")
            print(f"  * Database connection issues")
            print(f"  * Embedding model failures")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_manager' in locals():
            db_manager.close()
        if 'milvus_client' in locals():
            milvus_client.disconnect()

if __name__ == "__main__":
    analyze_embedding_gaps()