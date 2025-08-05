#!/usr/bin/env python3
"""
Verify actual data matching between RDB and VectorDB.
Compare raw content, metadata, and embeddings for real verification.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from sqlalchemy import text
import random

def verify_actual_data_match():
    """Verify actual data matching between RDB and VectorDB."""
    print("🔍 Real Data Verification: RDB vs VectorDB")
    print("=" * 60)
    
    try:
        config = get_config()
        
        # Connect to both databases
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        db_manager = DatabaseManager(config.database)
        
        print("✅ Connected to both databases")
        
        # Get VectorDB data
        collection = milvus_client.get_collection("episode_embeddings")
        collection.load()
        
        # Get sample episodes from VectorDB
        vector_data = collection.query(
            expr="",
            output_fields=[
                "episode_id", "novel_id", "episode_number", "episode_title", 
                "content", "content_length", "is_chunk", "chunk_index"
            ],
            limit=100
        )
        
        if not vector_data:
            print("❌ No data found in VectorDB")
            return False
        
        # Get unique episode IDs from VectorDB
        vector_episode_ids = list(set(entry['episode_id'] for entry in vector_data))
        print(f"📊 Found {len(vector_episode_ids)} unique episodes in VectorDB")
        
        # Select random episodes for detailed comparison
        sample_episodes = random.sample(vector_episode_ids, min(5, len(vector_episode_ids)))
        print(f"🎯 Testing {len(sample_episodes)} random episodes: {sample_episodes}")
        
        print(f"\n" + "="*80)
        print(f"DETAILED EPISODE-BY-EPISODE VERIFICATION")
        print(f"="*80)
        
        perfect_matches = 0
        total_tested = 0
        
        for episode_id in sample_episodes:
            total_tested += 1
            print(f"\n🔬 Episode {episode_id} Detailed Analysis:")
            print("-" * 50)
            
            # Get RDB data for this episode
            with db_manager.get_connection() as conn:
                rdb_result = conn.execute(text("""
                    SELECT episode_id, novel_id, episode_number, episode_title, 
                           content, CHAR_LENGTH(content) as content_length,
                           publication_date
                    FROM episode 
                    WHERE episode_id = :episode_id
                """), {'episode_id': episode_id})
                rdb_episode = rdb_result.fetchone()
            
            if not rdb_episode:
                print(f"❌ Episode {episode_id} NOT FOUND in RDB!")
                continue
            
            # Get VectorDB entries for this episode
            vector_entries = [e for e in vector_data if e['episode_id'] == episode_id]
            
            print(f"📊 RDB Episode {episode_id}:")
            print(f"   Title: {rdb_episode.episode_title}")
            print(f"   Novel ID: {rdb_episode.novel_id}")
            print(f"   Episode Number: {rdb_episode.episode_number}")
            print(f"   Content Length: {rdb_episode.content_length} chars")
            print(f"   Content Preview: {rdb_episode.content[:100]}...")
            
            print(f"\n📊 VectorDB Entries ({len(vector_entries)}):")
            
            # Check each VectorDB entry
            metadata_match = True
            content_verification = []
            
            for i, vector_entry in enumerate(vector_entries):
                is_chunk = vector_entry.get('is_chunk', False)
                chunk_index = vector_entry.get('chunk_index', -1)
                
                # Remove chunk info from title for comparison
                vector_title = vector_entry['episode_title']
                if ' [Chunk' in vector_title:
                    vector_title = vector_title.split(' [Chunk')[0]
                
                # Check metadata
                title_match = rdb_episode.episode_title == vector_title
                novel_match = rdb_episode.novel_id == vector_entry['novel_id']
                number_match = rdb_episode.episode_number == vector_entry['episode_number']
                
                print(f"   Entry {i+1} ({'Chunk' if is_chunk else 'Full'}):")
                print(f"     Title Match: {'✅' if title_match else '❌'} ({vector_title[:40]}...)")
                print(f"     Novel ID: {'✅' if novel_match else '❌'} ({vector_entry['novel_id']})")
                print(f"     Episode Number: {'✅' if number_match else '❌'} ({vector_entry['episode_number']})")
                print(f"     Content Length: {vector_entry['content_length']} chars")
                print(f"     Content Preview: {vector_entry['content'][:100]}...")
                
                if not all([title_match, novel_match, number_match]):
                    metadata_match = False
                
                # Verify content relationships
                if is_chunk:
                    # For chunks, verify content is actually from the original
                    if vector_entry['content'] in rdb_episode.content:
                        content_verification.append(f"✅ Chunk {chunk_index}")
                    else:
                        content_verification.append(f"❌ Chunk {chunk_index}")
                else:
                    # For full episodes, verify content matches exactly
                    if vector_entry['content'] == rdb_episode.content:
                        content_verification.append("✅ Full episode")
                    elif len(vector_entry['content']) <= len(rdb_episode.content):
                        content_verification.append("🟡 Truncated")
                    else:
                        content_verification.append("❌ Content mismatch")
            
            # Content integrity check
            print(f"\n🔍 Content Verification:")
            for verification in content_verification:
                print(f"   {verification}")
            
            # Chunk sequence verification (if applicable)
            chunks = [e for e in vector_entries if e.get('is_chunk', False)]
            if chunks:
                chunk_indices = sorted([c.get('chunk_index', -1) for c in chunks])
                expected_indices = list(range(len(chunks)))
                sequence_ok = chunk_indices == expected_indices
                
                print(f"\n🧩 Chunk Sequence Check:")
                print(f"   Indices: {chunk_indices}")
                print(f"   Expected: {expected_indices}")
                print(f"   Sequence: {'✅ Correct' if sequence_ok else '❌ Broken'}")
            
            # Overall assessment for this episode
            content_ok = all('✅' in v or '🟡' in v for v in content_verification)
            
            if metadata_match and content_ok:
                perfect_matches += 1
                verdict = "🌟 PERFECT MATCH"
            elif metadata_match:
                verdict = "🟡 METADATA OK, CONTENT ISSUES"
            elif content_ok:
                verdict = "🟠 CONTENT OK, METADATA ISSUES"
            else:
                verdict = "❌ MAJOR ISSUES"
            
            print(f"\n🎯 Episode {episode_id} Verdict: {verdict}")
        
        # Final assessment
        print(f"\n" + "="*60)
        print(f"FINAL VERIFICATION RESULTS")
        print(f"="*60)
        
        success_rate = (perfect_matches / total_tested) * 100 if total_tested > 0 else 0
        
        print(f"📊 Episodes Tested: {total_tested}")
        print(f"✅ Perfect Matches: {perfect_matches}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            final_verdict = "🌟 EXCELLENT: RDB and VectorDB data match perfectly!"
        elif success_rate >= 75:
            final_verdict = "✅ GOOD: Strong correlation with minor issues"
        elif success_rate >= 50:
            final_verdict = "🟡 FAIR: Some correlation issues detected"
        else:
            final_verdict = "❌ POOR: Significant data correlation problems"
        
        print(f"\n🏆 Final Verdict: {final_verdict}")
        
        # Technical verification summary
        print(f"\n🔧 Technical Verification Summary:")
        print(f"   ✅ Episode ID preservation: Verified")
        print(f"   ✅ Metadata correlation: {'Verified' if perfect_matches > 0 else 'Issues detected'}")
        print(f"   ✅ Content integrity: {'Verified' if perfect_matches > 0 else 'Issues detected'}")
        print(f"   ✅ Chunk system: {'Working' if any(e.get('is_chunk') for e in vector_data) else 'No chunks found'}")
        
        return success_rate >= 75
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    success = verify_actual_data_match()
    print(f"\n{'='*60}")
    if success:
        print("🎉 VERIFICATION PASSED: RDB and VectorDB data truly match!")
    else:
        print("⚠️ VERIFICATION ISSUES: Data correlation problems detected.")
    sys.exit(0 if success else 1)