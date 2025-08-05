#!/usr/bin/env python3
"""
Simple verification that vectordb contains properly structured episode data with metadata.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from collections import defaultdict

def verify_vectordb_data():
    """Verify vectordb contains proper episode data with metadata."""
    print("🔍 Verifying VectorDB Episode Data")
    print("=" * 60)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        if not milvus_client.has_collection("episode_embeddings"):
            print("❌ Collection 'episode_embeddings' does not exist")
            return False
            
        collection = milvus_client.get_collection("episode_embeddings")
        print("✅ Collection 'episode_embeddings' exists")
        
        # Check collection schema
        schema = collection.schema
        field_names = [field.name for field in schema.fields]
        expected_fields = [
            'episode_id', 'content_embedding', 'novel_id', 'episode_number',
            'episode_title', 'content', 'content_length', 'publication_timestamp',
            'publication_date', 'created_at', 'updated_at'
        ]
        
        print(f"\n📋 Schema Verification:")
        all_fields_present = True
        for field in expected_fields:
            present = field in field_names
            status = "✅" if present else "❌"
            print(f"  {status} {field}")
            if not present:
                all_fields_present = False
                
        if all_fields_present:
            print("✅ All expected metadata fields are present")
        else:
            print("❌ Some metadata fields are missing")
        
        # Check vector dimension
        embedding_field = next((f for f in schema.fields if f.name == 'content_embedding'), None)
        if embedding_field and hasattr(embedding_field, 'params'):
            vector_dim = embedding_field.params.get('dim', 'Unknown')
            print(f"✅ Vector dimension: {vector_dim}")
        
        # Get sample data to verify content
        print(f"\n📊 Data Content Verification:")
        sample_episodes = collection.query(
            expr="",
            output_fields=[
                "episode_id", "novel_id", "episode_number", "episode_title", 
                "content_length", "publication_date"
            ],
            limit=10
        )
        
        if not sample_episodes:
            print("❌ No episodes found in collection")
            return False
            
        print(f"✅ Found {len(sample_episodes)} sample episodes")
        
        # Verify data quality
        valid_episodes = 0
        for i, episode in enumerate(sample_episodes):
            valid = True
            issues = []
            
            # Check required fields
            if not episode.get('episode_id'):
                issues.append("Missing episode_id")
                valid = False
            if not episode.get('novel_id'):
                issues.append("Missing novel_id")
                valid = False
            if not episode.get('episode_title'):
                issues.append("Missing episode_title")
                valid = False
            if not episode.get('content_length', 0) > 0:
                issues.append("No content_length")
                valid = False
                
            if valid:
                valid_episodes += 1
                
            status = "✅" if valid else "❌"
            print(f"  {status} Episode {episode['episode_id']}: {episode['episode_title'][:30]}...")
            if issues:
                print(f"      Issues: {', '.join(issues)}")
        
        print(f"\n📈 Data Quality Summary:")
        print(f"  - Valid episodes: {valid_episodes}/{len(sample_episodes)}")
        quality_percentage = (valid_episodes / len(sample_episodes)) * 100
        print(f"  - Quality score: {quality_percentage:.1f}%")
        
        # Check novel distribution
        print(f"\n📚 Novel Distribution Analysis:")
        novel_episodes = defaultdict(list)
        
        # Get more episodes for distribution analysis
        all_episodes = collection.query(
            expr="",
            output_fields=["novel_id", "episode_id", "episode_number", "episode_title"],
            limit=1000
        )
        
        for episode in all_episodes:
            novel_episodes[episode['novel_id']].append(episode)
            
        total_novels = len(novel_episodes)
        total_episodes = len(all_episodes)
        
        print(f"  - Total novels: {total_novels}")
        print(f"  - Total episodes: {total_episodes}")
        print(f"  - Average episodes per novel: {total_episodes/total_novels:.1f}")
        
        # Show top novels by episode count
        novels_by_count = sorted(
            [(novel_id, len(episodes)) for novel_id, episodes in novel_episodes.items()],
            key=lambda x: x[1], reverse=True
        )
        
        print(f"\n🏆 Top 10 Novels by Episode Count:")
        for i, (novel_id, count) in enumerate(novels_by_count[:10]):
            # Get sample title
            sample_ep = novel_episodes[novel_id][0]
            title = sample_ep['episode_title'][:40]
            print(f"  {i+1:2}. Novel {novel_id:3}: {count:2} episodes - {title}...")
        
        # Check embedding presence by trying a similarity search
        print(f"\n🧮 Embedding Verification:")
        try:
            # This will fail if embeddings are not present
            search_results = collection.search(
                data=[[0.1] * 1024],  # Dummy vector of correct dimension
                anns_field="content_embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=1,
                output_fields=["episode_id"]
            )
            if search_results and len(search_results[0]) > 0:
                print("✅ Vector search successful - embeddings are present and functional")
            else:
                print("⚠️ Vector search returned no results")
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return False
            
        # Final assessment
        print(f"\n🎯 Final Assessment:")
        if (all_fields_present and 
            quality_percentage >= 80 and 
            total_episodes > 0 and 
            total_novels > 0):
            print("✅ VectorDB is PROPERLY configured with episode embeddings and metadata")
            print("✅ Data quality is good")
            print("✅ Embeddings are functional")
            return True
        else:
            print("❌ VectorDB has issues that need attention")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'milvus_client' in locals():
            milvus_client.disconnect()

if __name__ == "__main__":
    success = verify_vectordb_data()
    sys.exit(0 if success else 1)