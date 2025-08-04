#!/usr/bin/env python3
"""
Get specific episode IDs for Novel 25 for testing.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient

def get_novel25_episodes():
    """Get episode IDs for Novel 25."""
    print("🔍 Getting Episode IDs for Novel 25")
    print("=" * 50)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        if milvus_client.has_collection("episode_embeddings"):
            collection = milvus_client.get_collection("episode_embeddings")
            
            # Query to get Novel 25 episodes
            query_results = collection.query(
                expr="novel_id == 25",
                output_fields=["episode_id", "episode_number", "episode_title"],
                limit=20
            )
            
            print(f"📊 Found {len(query_results)} episodes for Novel 25")
            
            # Sort by episode_number
            episodes = sorted(query_results, key=lambda x: x['episode_number'])
            
            print("\n📚 Novel 25 Episodes:")
            print("-" * 40)
            
            episode_ids = []
            for ep in episodes:
                print(f"Episode {ep['episode_id']} (#{ep['episode_number']}): {ep['episode_title']}")
                episode_ids.append(ep['episode_id'])
            
            # Get first 10 episode IDs
            first_10_ids = episode_ids[:10]
            print(f"\n🎯 First 10 Episode IDs: {first_10_ids}")
            
            return first_10_ids
                
        else:
            print("❌ Collection 'episode_embeddings' does not exist")
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    get_novel25_episodes()