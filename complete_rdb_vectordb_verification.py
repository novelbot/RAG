#!/usr/bin/env python3
"""
Complete verification of RDB-VectorDB correlation using proper session management.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from sqlalchemy.orm import sessionmaker
from collections import defaultdict

def complete_rdb_vectordb_verification():
    """Complete verification of RDB and VectorDB correlation."""
    print("🔗 Complete RDB-VectorDB Correlation Verification")
    print("=" * 70)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize database connections
        print("🔌 Initializing database connections...")
        
        # Milvus connection
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        # RDB connection with session
        db_manager = DatabaseManager(config.database)
        Session = sessionmaker(bind=db_manager.engine)
        print("✅ Connected to RDB")
        
        # Import models
        from src.database.models import Episode, Novel
        
        # Check VectorDB data
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
            limit=2000  # Get more data
        )
        
        print(f"📈 VectorDB: {len(vector_episodes)} episodes found")
        
        # Analyze VectorDB data structure
        vector_by_novel = defaultdict(list)
        vector_episode_ids = set()
        
        for ep in vector_episodes:
            vector_by_novel[ep['novel_id']].append(ep)
            vector_episode_ids.add(ep['episode_id'])
        
        print(f"📈 VectorDB: {len(vector_by_novel)} novels represented")
        
        # Check RDB data
        print("\n📊 Analyzing RDB data...")
        with Session() as session:
            # Get all novels with episode counts
            novels_with_episodes = session.query(
                Novel.id,
                Novel.title,
                Novel.author,
                Novel.status
            ).join(Episode, Novel.id == Episode.novel_id).distinct().all()
            
            print(f"📈 RDB: {len(novels_with_episodes)} novels with episodes")
            
            # Get episode counts by novel
            rdb_novel_counts = {}
            rdb_episodes_by_novel = defaultdict(list)
            
            for novel in novels_with_episodes:
                episodes = session.query(Episode).filter(
                    Episode.novel_id == novel.id
                ).all()
                
                rdb_novel_counts[novel.id] = len(episodes)
                rdb_episodes_by_novel[novel.id] = episodes
            
            total_rdb_episodes = sum(rdb_novel_counts.values())
            print(f"📈 RDB: {total_rdb_episodes} total episodes")
        
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
            print(f"  - VectorDB-only novel IDs: {sorted(list(vector_only))[:10]}")
        if rdb_only:
            print(f"  - RDB-only novel IDs: {sorted(list(rdb_only))[:10]}")
        
        # Episode count comparison for common novels
        print(f"\n📊 Episode Count Comparison:")
        print("-" * 70)
        print(f"{'Novel':<6} {'RDB':<6} {'Vector':<8} {'Match':<6} {'Title':<30}")
        print("-" * 70)
        
        perfect_matches = 0
        partial_matches = 0
        
        for novel_id in sorted(common_novels):
            rdb_count = rdb_novel_counts.get(novel_id, 0)
            vector_count = len(vector_by_novel.get(novel_id, []))
            
            if rdb_count == vector_count:
                match_status = "✅ FULL"
                perfect_matches += 1
            elif vector_count > 0:
                match_status = "⚠️ PART"
                partial_matches += 1
            else:
                match_status = "❌ NONE"
            
            # Get novel title from RDB
            with Session() as session:
                novel = session.query(Novel).filter(Novel.id == novel_id).first()
                novel_title = novel.title[:25] if novel and novel.title else "N/A"
            
            print(f"{novel_id:<6} {rdb_count:<6} {vector_count:<8} {match_status:<6} {novel_title:<30}")
        
        print(f"\n📈 Match Summary:")
        print(f"  - Perfect matches: {perfect_matches}")
        print(f"  - Partial matches: {partial_matches}")
        print(f"  - No matches: {len(common_novels) - perfect_matches - partial_matches}")
        
        # Episode-level verification for top novels
        print(f"\n🔬 Episode-level Verification (Top 5 Novels):")
        print("-" * 70)
        
        # Get top 5 novels by vector episode count
        top_novels = sorted(
            [(nid, len(eps)) for nid, eps in vector_by_novel.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        episode_level_verified = 0
        
        for novel_id, vector_count in top_novels:
            print(f"\n📚 Novel {novel_id} ({vector_count} vector episodes):")
            
            # Get RDB episodes for this novel
            with Session() as session:
                rdb_episodes = session.query(Episode).filter(
                    Episode.novel_id == novel_id
                ).order_by(Episode.episode_number).all()
                
                novel = session.query(Novel).filter(Novel.id == novel_id).first()
                print(f"    Title: {novel.title if novel else 'N/A'}")
                print(f"    RDB episodes: {len(rdb_episodes)}")
            
            # Check episode ID matches
            vector_eps = vector_by_novel[novel_id]
            vector_ep_ids = {ep['episode_id'] for ep in vector_eps}
            rdb_ep_ids = {ep.id for ep in rdb_episodes}
            
            common_episode_ids = vector_ep_ids & rdb_ep_ids
            vector_only_ids = vector_ep_ids - rdb_ep_ids
            rdb_only_ids = rdb_ep_ids - vector_ep_ids
            
            print(f"    Episode ID matches: {len(common_episode_ids)}")
            print(f"    Vector-only episode IDs: {len(vector_only_ids)}")
            print(f"    RDB-only episode IDs: {len(rdb_only_ids)}")
            
            if len(common_episode_ids) == len(vector_ep_ids) == len(rdb_ep_ids):
                print("    ✅ Perfect episode-level match")
                episode_level_verified += 1
            else:
                print("    ⚠️ Episode-level discrepancy")
        
        # Metadata quality check
        print(f"\n🔬 Metadata Quality Check:")
        print("-" * 50)
        
        # Check sample episodes for metadata accuracy
        sample_episodes = vector_episodes[:10]
        metadata_matches = 0
        
        with Session() as session:
            for vector_ep in sample_episodes:
                ep_id = vector_ep['episode_id']
                rdb_ep = session.query(Episode).filter(Episode.id == ep_id).first()
                
                if rdb_ep:
                    title_match = rdb_ep.title == vector_ep['episode_title']
                    novel_match = rdb_ep.novel_id == vector_ep['novel_id']
                    number_match = rdb_ep.episode_number == vector_ep['episode_number']
                    
                    content_len_match = True
                    if rdb_ep.content and vector_ep['content_length']:
                        content_len_match = len(rdb_ep.content) == vector_ep['content_length']
                    
                    if title_match and novel_match and number_match and content_len_match:
                        metadata_matches += 1
                        status = "✅"
                    else:
                        status = "❌"
                        print(f"      Episode {ep_id}: Title={title_match}, Novel={novel_match}, Number={number_match}, Length={content_len_match}")
                    
                    print(f"    {status} Episode {ep_id}: {vector_ep['episode_title'][:40]}...")
        
        print(f"\n📊 Final Assessment:")
        print("=" * 50)
        print(f"RDB Status:")
        print(f"  - Total novels: {len(rdb_novels)}")
        print(f"  - Total episodes: {total_rdb_episodes}")
        
        print(f"VectorDB Status:")
        print(f"  - Total novels: {len(vector_novels)}")
        print(f"  - Total episodes: {len(vector_episodes)}")
        
        print(f"Correlation Status:")
        print(f"  - Novel-level matches: {perfect_matches}/{len(common_novels)}")
        print(f"  - Episode-level verified: {episode_level_verified}/5")
        print(f"  - Metadata accuracy: {metadata_matches}/10")
        
        # Final verdict
        overall_score = (
            (perfect_matches / max(len(common_novels), 1)) * 0.4 +
            (episode_level_verified / 5) * 0.3 +
            (metadata_matches / 10) * 0.3
        ) * 100
        
        print(f"  - Overall correlation score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("\n🎯 ✅ EXCELLENT: RDB and VectorDB are well correlated")
        elif overall_score >= 60:
            print("\n🎯 ⚠️ GOOD: RDB and VectorDB have good correlation with minor issues")
        else:
            print("\n🎯 ❌ POOR: RDB and VectorDB have significant correlation issues")
        
        return overall_score >= 60
        
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
    success = complete_rdb_vectordb_verification()
    sys.exit(0 if success else 1)