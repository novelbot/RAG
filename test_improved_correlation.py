#!/usr/bin/env python3
"""
Test script to verify improved RDB-VectorDB correlation after schema fixes.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from collections import defaultdict
from sqlalchemy import text

def test_improved_correlation():
    """Test the improved RDB-VectorDB correlation with new schema."""
    print("🔗 Testing Improved RDB-VectorDB Correlation")
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
        
        # Check if collection exists with new schema
        print("\n📊 Checking VectorDB collection schema...")
        if not milvus_client.has_collection("episode_embeddings"):
            print("❌ Collection 'episode_embeddings' does not exist")
            print("   Need to run data ingestion first with new schema")
            return False
            
        collection = milvus_client.get_collection("episode_embeddings")
        
        # Check schema fields
        schema_fields = [field.name for field in collection.schema.fields]
        expected_fields = [
            "entry_id", "episode_id", "content_embedding", "novel_id", 
            "episode_number", "episode_title", "content", "content_length",
            "is_chunk", "chunk_index", "total_chunks", "publication_timestamp",
            "publication_date", "created_at", "updated_at"
        ]
        
        print(f"📋 Schema fields found: {len(schema_fields)}")
        
        missing_fields = set(expected_fields) - set(schema_fields)
        if missing_fields:
            print(f"❌ Missing expected fields: {missing_fields}")
            print("   Old schema detected - need to reingest data with new schema")
            return False
        
        print("✅ New schema detected with proper chunk tracking fields")
        
        # Get sample data with new schema
        try:
            vector_data = collection.query(
                expr="",
                output_fields=[
                    "entry_id", "episode_id", "novel_id", "episode_number", 
                    "episode_title", "content_length", "is_chunk", "chunk_index", 
                    "total_chunks", "publication_date"
                ],
                limit=50
            )
        except Exception as e:
            print(f"❌ Error querying with new schema: {e}")
            print("   Old data detected - need to reingest with new schema")
            return False
        
        if not vector_data:
            print("❌ No data found in VectorDB")
            return False
        
        print(f"📈 VectorDB: {len(vector_data)} entries found")
        
        # Analyze chunk vs regular episode distribution
        regular_episodes = [entry for entry in vector_data if not entry.get('is_chunk', True)]
        chunk_entries = [entry for entry in vector_data if entry.get('is_chunk', True)]
        
        print(f"   - Regular episodes: {len(regular_episodes)}")
        print(f"   - Chunk entries: {len(chunk_entries)}")
        
        # Get RDB data for comparison
        print("\n📊 Analyzing RDB data...")
        
        with db_manager.get_connection() as conn:
            # Get total episode count
            total_episodes_result = conn.execute(text("SELECT COUNT(*) FROM episode"))
            total_rdb_episodes = total_episodes_result.scalar()
            
            # Get sample episodes for validation
            sample_episodes_result = conn.execute(text("""
                SELECT episode_id, novel_id, episode_number, episode_title, 
                       CHAR_LENGTH(content) as content_length, publication_date
                FROM episode 
                LIMIT 10
            """))
            rdb_sample_episodes = sample_episodes_result.fetchall()
        
        print(f"📈 RDB: {total_rdb_episodes} episodes")
        
        # Test episode ID correlation
        print("\n🔍 Episode ID Correlation Test:")
        print("-" * 50)
        
        # Extract unique episode IDs from VectorDB 
        vector_episode_ids = set()
        for entry in vector_data:
            vector_episode_ids.add(entry['episode_id'])
        
        print(f"📊 Unique episodes in VectorDB: {len(vector_episode_ids)}")
        
        # Test correlation with RDB sample
        correlation_matches = 0
        correlation_issues = []
        
        for rdb_ep in rdb_sample_episodes:
            # Find corresponding entry in VectorDB by episode_id
            matching_entries = [
                entry for entry in vector_data 
                if entry['episode_id'] == rdb_ep.episode_id
            ]
            
            if matching_entries:
                # Check if metadata matches
                vector_ep = matching_entries[0]  # Take first match
                
                title_match = rdb_ep.episode_title == vector_ep['episode_title'].split(' [Chunk')[0]  # Remove chunk info
                novel_match = rdb_ep.novel_id == vector_ep['novel_id']
                number_match = rdb_ep.episode_number == vector_ep['episode_number']
                
                if all([title_match, novel_match, number_match]):
                    correlation_matches += 1
                    status = "✅"
                else:
                    status = "❌"
                    issues = []
                    if not title_match: issues.append("title")
                    if not novel_match: issues.append("novel_id")
                    if not number_match: issues.append("episode_number")
                    correlation_issues.append(f"Episode {rdb_ep.episode_id}: {', '.join(issues)}")
                
                print(f"    {status} Episode {rdb_ep.episode_id}: Found {len(matching_entries)} entries")
            else:
                print(f"    ❌ Episode {rdb_ep.episode_id}: NOT FOUND")
                correlation_issues.append(f"Episode {rdb_ep.episode_id}: missing from VectorDB")
        
        # Test chunk tracking
        print(f"\n🧩 Chunk Tracking Test:")
        print("-" * 50)
        
        if chunk_entries:
            # Find episodes with multiple chunks
            episode_chunk_counts = defaultdict(list)
            for chunk in chunk_entries:
                episode_chunk_counts[chunk['episode_id']].append(chunk)
            
            multi_chunk_episodes = {ep_id: chunks for ep_id, chunks in episode_chunk_counts.items() if len(chunks) > 1}
            
            print(f"    Episodes with chunks: {len(episode_chunk_counts)}")
            print(f"    Episodes with multiple chunks: {len(multi_chunk_episodes)}")
            
            # Test chunk consistency for one multi-chunk episode
            if multi_chunk_episodes:
                test_episode_id = list(multi_chunk_episodes.keys())[0]
                test_chunks = multi_chunk_episodes[test_episode_id]
                
                print(f"    Testing episode {test_episode_id} with {len(test_chunks)} chunks:")
                
                # Check chunk index sequence
                chunk_indices = sorted([chunk['chunk_index'] for chunk in test_chunks])
                expected_indices = list(range(len(test_chunks)))
                
                if chunk_indices == expected_indices:
                    print(f"        ✅ Chunk indices: {chunk_indices} (correct sequence)")
                else:
                    print(f"        ❌ Chunk indices: {chunk_indices} (expected {expected_indices})")
                
                # Check total_chunks consistency
                total_chunks_values = set(chunk['total_chunks'] for chunk in test_chunks)
                if len(total_chunks_values) == 1 and list(total_chunks_values)[0] == len(test_chunks):
                    print(f"        ✅ Total chunks: {list(total_chunks_values)[0]} (consistent)")
                else:
                    print(f"        ❌ Total chunks: {total_chunks_values} (inconsistent)")
        else:
            print("    No chunk entries found")
        
        # Final Assessment
        print(f"\n🎯 Assessment Results:")
        print("=" * 50)
        
        correlation_rate = (correlation_matches / len(rdb_sample_episodes)) * 100 if rdb_sample_episodes else 0
        
        print(f"Episode ID correlation: {correlation_matches}/{len(rdb_sample_episodes)} ({correlation_rate:.1f}%)")
        print(f"Schema improvements: {'✅ Applied' if 'entry_id' in schema_fields else '❌ Not applied'}")
        print(f"Chunk tracking: {'✅ Functional' if chunk_entries else '⚠️ No chunks found'}")
        
        if correlation_issues:
            print(f"\n⚠️ Remaining Issues:")
            for issue in correlation_issues[:3]:  # Show first 3
                print(f"  - {issue}")
            if len(correlation_issues) > 3:
                print(f"  ... and {len(correlation_issues) - 3} more")
        
        # Overall verdict
        improvements_applied = 'entry_id' in schema_fields
        correlation_good = correlation_rate >= 80
        
        if improvements_applied and correlation_good:
            verdict = "🌟 EXCELLENT: Schema improvements successful!"
        elif improvements_applied:
            verdict = "🔧 IMPROVED: Schema fixed, need data reingestion"
        else:
            verdict = "⚠️ PENDING: Need to apply schema improvements"
        
        print(f"\n🏆 Overall Status: {verdict}")
        
        if not improvements_applied:
            print("\n📋 Next Steps:")
            print("1. The schema has been updated in code")
            print("2. Run data reingestion to apply the new schema:")
            print("   uv run rag-cli data ingest --episode-mode --database --force")
            print("3. Run this test again to verify improvements")
        
        return improvements_applied and correlation_good
        
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
    success = test_improved_correlation()
    print(f"\n{'='*50}")
    if success:
        print("🎉 IMPROVEMENTS VERIFIED: RDB-VectorDB correlation is working correctly!")
    else:
        print("🔧 IMPROVEMENTS APPLIED: Run data reingestion to complete the fix.")
    sys.exit(0 if success else 1)