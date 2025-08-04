#!/usr/bin/env python3
"""
Check which novels have 10+ episodes for testing.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient

def check_novels_episodes():
    """Check novels with episode counts."""
    print("🔍 Checking Novels with Episode Counts")
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
            
            # Query to get novel_id and episode counts
            query_results = collection.query(
                expr="",  # Get all
                output_fields=["novel_id", "episode_id", "episode_number", "episode_title"],
                limit=500  # Get more results to analyze
            )
            
            print(f"📊 Total episodes found: {len(query_results)}")
            
            # Group by novel_id
            novels = {}
            for episode in query_results:
                novel_id = episode['novel_id']
                if novel_id not in novels:
                    novels[novel_id] = []
                novels[novel_id].append({
                    'episode_id': episode['episode_id'],
                    'episode_number': episode['episode_number'],
                    'episode_title': episode['episode_title']
                })
            
            # Sort and display novels with 10+ episodes
            print("\n📚 Novels with 10+ episodes:")
            print("-" * 40)
            
            novels_with_10plus = []
            for novel_id, episodes in novels.items():
                episode_count = len(episodes)
                if episode_count >= 10:
                    novels_with_10plus.append((novel_id, episode_count, episodes))
            
            # Sort by episode count (descending)
            novels_with_10plus.sort(key=lambda x: x[1], reverse=True)
            
            for novel_id, count, episodes in novels_with_10plus:
                print(f"Novel {novel_id}: {count} episodes")
                
                # Sort episodes by episode_number
                episodes.sort(key=lambda x: x['episode_number'])
                
                # Show first few episode titles to identify the novel
                print(f"  First episodes:")
                for i, ep in enumerate(episodes[:3]):
                    print(f"    {ep['episode_number']}. {ep['episode_title']}")
                
                if count > 3:
                    print(f"    ... and {count-3} more episodes")
                print()
                
        else:
            print("❌ Collection 'episode_embeddings' does not exist")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_novels_episodes()