#!/usr/bin/env python3
"""
Final comprehensive verification of RDB-VectorDB correlation.
Now that we know the exact RDB schema.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from collections import defaultdict
from sqlalchemy import text

def final_rdb_vectordb_verification():
    """Final comprehensive verification using raw SQL queries."""
    print("🔗 Final RDB-VectorDB Correlation Verification")
    print("=" * 70)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize connections
        print("🔌 Initializing database connections...")
        
        # Milvus connection
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        # RDB connection
        db_manager = DatabaseManager(config.database)
        print("✅ Connected to RDB")
        
        # Get VectorDB data
        print("\n📊 Analyzing VectorDB data...")
        if not milvus_client.has_collection("episode_embeddings"):
            print("❌ Collection 'episode_embeddings' does not exist")
            return False
            
        collection = milvus_client.get_collection("episode_embeddings")
        
        # Get all episodes from VectorDB
        vector_episodes = collection.query(
            expr="",
            output_fields=[
                "episode_id", "novel_id", "episode_number", "episode_title", 
                "content_length", "publication_date"
            ],
            limit=2000
        )
        
        print(f"📈 VectorDB: {len(vector_episodes)} episodes found")
        
        # Group VectorDB data
        vector_by_novel = defaultdict(list)
        vector_episode_ids = set()
        
        for ep in vector_episodes:
            vector_by_novel[ep['novel_id']].append(ep)
            vector_episode_ids.add(ep['episode_id'])
        
        # Get RDB data using raw SQL
        print("\n📊 Analyzing RDB data...")
        
        with db_manager.get_connection() as conn:
            # Get novel counts
            novel_count_result = conn.execute(text("""
                SELECT novel_id, COUNT(*) as episode_count 
                FROM episode 
                GROUP BY novel_id 
                ORDER BY novel_id
            """))
            rdb_novel_counts = dict(novel_count_result.fetchall())
            
            # Get total episode count
            total_episodes_result = conn.execute(text("SELECT COUNT(*) FROM episode"))
            total_rdb_episodes = total_episodes_result.scalar()
            
            # Get novels info
            novels_result = conn.execute(text("SELECT novel_id, title, author FROM novels"))
            novels_info = {row.novel_id: {'title': row.title, 'author': row.author} 
                          for row in novels_result.fetchall()}
            
            # Get sample episode data for validation
            sample_episodes_result = conn.execute(text("""
                SELECT episode_id, novel_id, episode_number, episode_title, 
                       CHAR_LENGTH(content) as content_length, publication_date
                FROM episode 
                LIMIT 10
            """))
            rdb_sample_episodes = sample_episodes_result.fetchall()
        
        print(f"📈 RDB: {len(novels_info)} novels, {total_rdb_episodes} episodes")
        
        # Compare data
        print("\n🔍 Detailed Correlation Analysis:")
        print("=" * 70)
        
        vector_novels = set(vector_by_novel.keys())
        rdb_novels = set(rdb_novel_counts.keys())
        
        common_novels = vector_novels & rdb_novels
        vector_only = vector_novels - rdb_novels
        rdb_only = rdb_novels - vector_novels
        
        print(f"📊 Novel-level Analysis:")
        print(f"  - Common novels (in both): {len(common_novels)}")
        print(f"  - VectorDB only: {len(vector_only)}")
        print(f"  - RDB only: {len(rdb_only)}")
        
        if vector_only:
            print(f"  - VectorDB-only novel IDs: {sorted(list(vector_only))}")
        if rdb_only:
            print(f"  - RDB-only novel IDs: {sorted(list(rdb_only))}")
        
        # Episode count comparison
        print(f"\n📊 Episode Count Comparison (Common Novels):")
        print("-" * 80)
        print(f"{'Novel':<6} {'RDB':<6} {'Vector':<8} {'Match':<6} {'Title':<40}")
        print("-" * 80)
        
        perfect_matches = 0
        
        for novel_id in sorted(common_novels):
            rdb_count = rdb_novel_counts.get(novel_id, 0)
            vector_count = len(vector_by_novel.get(novel_id, []))
            
            if rdb_count == vector_count:
                match_status = "✅ FULL"
                perfect_matches += 1
            else:
                match_status = f"❌ {rdb_count-vector_count:+d}"  # Show difference
            
            novel_title = novels_info.get(novel_id, {}).get('title', 'N/A')[:35]
            
            print(f"{novel_id:<6} {rdb_count:<6} {vector_count:<8} {match_status:<6} {novel_title:<40}")
        
        # Metadata validation
        print(f"\n🔬 Metadata Validation (Sample Episodes):")
        print("-" * 70)
        
        metadata_matches = 0
        metadata_issues = []
        
        for rdb_ep in rdb_sample_episodes:
            # Find corresponding vector episode
            vector_ep = None
            for v_ep in vector_episodes:
                if v_ep['episode_id'] == rdb_ep.episode_id:
                    vector_ep = v_ep
                    break
            
            if vector_ep:
                # Check metadata matches
                title_match = rdb_ep.episode_title == vector_ep['episode_title']
                novel_match = rdb_ep.novel_id == vector_ep['novel_id']
                number_match = rdb_ep.episode_number == vector_ep['episode_number']
                length_match = rdb_ep.content_length == vector_ep['content_length']
                
                if all([title_match, novel_match, number_match, length_match]):
                    metadata_matches += 1
                    status = "✅"
                else:
                    status = "❌"
                    issues = []
                    if not title_match: issues.append("title")
                    if not novel_match: issues.append("novel_id")
                    if not number_match: issues.append("episode_number")
                    if not length_match: issues.append("content_length")
                    metadata_issues.append(f"Episode {rdb_ep.episode_id}: {', '.join(issues)}")
                
                print(f"    {status} Episode {rdb_ep.episode_id}: {rdb_ep.episode_title[:50]}...")
            else:
                print(f"    ❌ Episode {rdb_ep.episode_id}: NOT FOUND in VectorDB")
                metadata_issues.append(f"Episode {rdb_ep.episode_id}: missing from VectorDB")
        
        # Coverage analysis
        print(f"\n📈 Coverage Analysis:")
        print("-" * 50)
        
        rdb_episode_ids = set()
        with db_manager.get_connection() as conn:
            all_episode_ids = conn.execute(text("SELECT episode_id FROM episode"))
            rdb_episode_ids = {row.episode_id for row in all_episode_ids.fetchall()}
        
        vector_coverage = len(vector_episode_ids & rdb_episode_ids)
        total_coverage_pct = (vector_coverage / len(rdb_episode_ids)) * 100 if rdb_episode_ids else 0
        
        print(f"  - RDB episodes: {len(rdb_episode_ids)}")
        print(f"  - VectorDB episodes: {len(vector_episode_ids)}")
        print(f"  - Episodes in both: {vector_coverage}")
        print(f"  - Coverage percentage: {total_coverage_pct:.1f}%")
        
        # Final Assessment
        print(f"\n🎯 Final Assessment:")
        print("=" * 50)
        
        novel_match_pct = (perfect_matches / len(common_novels)) * 100 if common_novels else 0
        metadata_match_pct = (metadata_matches / len(rdb_sample_episodes)) * 100 if rdb_sample_episodes else 0
        
        print(f"Novel-level correlation: {perfect_matches}/{len(common_novels)} ({novel_match_pct:.1f}%)")
        print(f"Metadata accuracy: {metadata_matches}/{len(rdb_sample_episodes)} ({metadata_match_pct:.1f}%)")
        print(f"Episode coverage: {total_coverage_pct:.1f}%")
        
        if metadata_issues:
            print(f"\n⚠️ Metadata Issues Found:")
            for issue in metadata_issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(metadata_issues) > 5:
                print(f"  ... and {len(metadata_issues) - 5} more issues")
        
        # Overall Score
        overall_score = (
            (novel_match_pct * 0.3) + 
            (metadata_match_pct * 0.3) + 
            (total_coverage_pct * 0.4)
        )
        
        print(f"\n🏆 Overall Correlation Score: {overall_score:.1f}%")
        
        if overall_score >= 90:
            verdict = "🌟 EXCELLENT: Perfect correlation between RDB and VectorDB"
        elif overall_score >= 75:
            verdict = "✅ GOOD: Strong correlation with minor discrepancies"
        elif overall_score >= 50:
            verdict = "⚠️ FAIR: Moderate correlation with some issues"
        else:
            verdict = "❌ POOR: Significant correlation problems"
        
        print(f"Verdict: {verdict}")
        
        return overall_score >= 75
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    success = final_rdb_vectordb_verification()
    print(f"\n{'='*50}")
    if success:
        print("🎉 VERIFICATION PASSED: RDB and VectorDB are properly correlated!")
    else:
        print("⚠️ VERIFICATION ISSUES: Check the correlation problems above.")
    sys.exit(0 if success else 1)