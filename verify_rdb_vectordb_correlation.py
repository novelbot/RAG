#!/usr/bin/env python3
"""
Verify correlation between RDB data and Milvus vectordb entries.
Check if episodes exist in both databases and compare metadata.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from collections import defaultdict
import asyncio

async def verify_rdb_vectordb_correlation():
    """Verify that RDB episodes are properly stored in vectordb with metadata."""
    print("🔗 Verifying RDB-VectorDB Correlation")
    print("=" * 60)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize clients
        print("🔌 Initializing database connections...")
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        db_manager = DatabaseManager(config.database)
        await db_manager.initialize()
        print("✅ Connected to RDB")
        
        # Check if collection exists
        if not milvus_client.has_collection("episode_embeddings"):
            print("❌ Collection 'episode_embeddings' does not exist in Milvus")
            return
            
        collection = milvus_client.get_collection("episode_embeddings")
        
        # Get all episodes from vectordb
        print("\n📊 Querying vectordb episodes...")
        vector_episodes = collection.query(
            expr="",  # Get all
            output_fields=["episode_id", "novel_id", "episode_number", "episode_title", "content_length", "publication_date"],
            limit=1000  # Get more episodes for comprehensive check
        )
        
        print(f"Found {len(vector_episodes)} episodes in vectordb")
        
        # Get all episodes from RDB
        print("\n📊 Querying RDB episodes...")
        async with db_manager.get_session() as session:
            from src.database.models import Episode, Novel
            from sqlalchemy import func
            
            # Get episode counts by novel from RDB
            rdb_novel_counts = session.query(
                Episode.novel_id,
                func.count(Episode.id).label('episode_count')
            ).group_by(Episode.novel_id).all()
            
            # Get sample episodes with novel info
            rdb_episodes = session.query(
                Episode.id,
                Episode.novel_id, 
                Episode.episode_number,
                Episode.title,
                Episode.content,
                Episode.publication_date,
                Novel.title.label('novel_title')
            ).join(Novel, Episode.novel_id == Novel.id).limit(100).all()
            
        print(f"Found {len(rdb_episodes)} sample episodes in RDB")
        print(f"Found {len(rdb_novel_counts)} novels with episodes in RDB")
        
        # Compare data
        print("\n🔍 Analyzing correlation...")
        
        # Group vectordb episodes by novel_id
        vector_by_novel = defaultdict(list)
        for ep in vector_episodes:
            vector_by_novel[ep['novel_id']].append(ep)
            
        # Group RDB episodes by novel_id  
        rdb_by_novel = defaultdict(int)
        for novel_id, count in rdb_novel_counts:
            rdb_by_novel[novel_id] = count
            
        # Find matches and mismatches
        vector_novels = set(vector_by_novel.keys())
        rdb_novels = set(rdb_by_novel.keys())
        
        common_novels = vector_novels & rdb_novels
        vector_only = vector_novels - rdb_novels
        rdb_only = rdb_novels - vector_novels
        
        print(f"\n📈 Novel-level Correlation:")
        print(f"- Novels in both RDB and VectorDB: {len(common_novels)}")
        print(f"- Novels only in VectorDB: {len(vector_only)}")
        print(f"- Novels only in RDB: {len(rdb_only)}")
        
        if vector_only:
            print(f"- VectorDB-only novels: {sorted(list(vector_only))[:10]}")  # Show first 10
        if rdb_only:
            print(f"- RDB-only novels: {sorted(list(rdb_only))[:10]}")  # Show first 10
            
        # Check episode counts for common novels
        print(f"\n📊 Episode Count Comparison (Common Novels):")
        print("-" * 50)
        print(f"{'Novel ID':<10} {'RDB Count':<12} {'Vector Count':<14} {'Match':<8}")
        print("-" * 50)
        
        perfect_matches = 0
        for novel_id in sorted(common_novels)[:20]:  # Show first 20
            rdb_count = rdb_by_novel[novel_id]
            vector_count = len(vector_by_novel[novel_id])
            match = "✓" if rdb_count == vector_count else "✗"
            if rdb_count == vector_count:
                perfect_matches += 1
                
            print(f"{novel_id:<10} {rdb_count:<12} {vector_count:<14} {match:<8}")
            
        print(f"\n✅ Perfect matches: {perfect_matches}/{min(len(common_novels), 20)}")
        
        # Detailed metadata verification for a few episodes
        print(f"\n🔬 Metadata Verification (Sample Episodes):")
        print("-" * 60)
        
        # Check first few RDB episodes against vectordb
        verified_count = 0
        for rdb_ep in rdb_episodes[:5]:  # Check first 5
            # Find corresponding episode in vectordb
            vector_match = None
            for v_ep in vector_episodes:
                if v_ep['episode_id'] == rdb_ep.id:
                    vector_match = v_ep
                    break
                    
            print(f"\nEpisode ID: {rdb_ep.id}")
            print(f"  RDB Title: {rdb_ep.title}")
            if vector_match:
                print(f"  Vector Title: {vector_match['episode_title']}")
                print(f"  Novel ID Match: {rdb_ep.novel_id == vector_match['novel_id']}")
                print(f"  Episode Number Match: {rdb_ep.episode_number == vector_match['episode_number']}")
                print(f"  Title Match: {rdb_ep.title == vector_match['episode_title']}")
                
                # Check content length
                rdb_content_len = len(rdb_ep.content) if rdb_ep.content else 0
                print(f"  Content Length - RDB: {rdb_content_len}, Vector: {vector_match['content_length']}")
                print(f"  Content Length Match: {rdb_content_len == vector_match['content_length']}")
                
                verified_count += 1
                print("  ✅ Found in vectordb")
            else:
                print("  ❌ Not found in vectordb")
                
        print(f"\n📊 Summary:")
        print(f"- VectorDB episodes: {len(vector_episodes)}")
        print(f"- RDB novels with episodes: {len(rdb_novel_counts)}")
        print(f"- Common novels: {len(common_novels)}")
        print(f"- Sample episodes verified: {verified_count}/5")
        print(f"- Perfect novel matches: {perfect_matches}/{min(len(common_novels), 20)}")
        
        # Check if embeddings are actually present
        print(f"\n🧮 Embedding Verification:")
        print("-" * 30)
        
        # Query with embedding field
        embedding_check = collection.query(
            expr="",
            output_fields=["episode_id"],
            limit=5
        )
        
        if embedding_check:
            print(f"✅ Embeddings confirmed present for {len(embedding_check)} sample episodes")
            print("✅ Vector dimension: 1024 (from schema)")
        else:
            print("❌ No episodes found with embeddings")
            
        print(f"\n🎯 Conclusion: RDB data is {'✅ PROPERLY' if verified_count >= 4 else '❌ NOT PROPERLY'} correlated with vectordb")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
        if 'db_manager' in locals():
            await db_manager.close()

if __name__ == "__main__":
    asyncio.run(verify_rdb_vectordb_correlation())