#!/usr/bin/env python3
"""
Analyze Milvus data to understand novel distribution and episode counts.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from collections import defaultdict

def analyze_milvus_data():
    """Analyze novels and episodes from Milvus data."""
    print("🔍 Analyzing Milvus Episode Data")
    print("=" * 60)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        if milvus_client.has_collection("episode_embeddings"):
            collection = milvus_client.get_collection("episode_embeddings")
            
            # Query to get all episodes with novel info
            query_results = collection.query(
                expr="",  # Get all
                output_fields=["novel_id", "episode_id", "episode_number", "episode_title"],
                limit=500  # Get all episodes
            )
            
            total_episodes = len(query_results)
            print(f"📊 Total episodes in Milvus: {total_episodes}")
            
            # Group by novel_id
            novels = defaultdict(list)
            for episode in query_results:
                novel_id = episode['novel_id']
                novels[novel_id].append({
                    'episode_id': episode['episode_id'],
                    'episode_number': episode['episode_number'],
                    'episode_title': episode['episode_title']
                })
            
            total_novels = len(novels)
            print(f"📚 Total novels represented: {total_novels}")
            
            # Sort novels by episode count
            novels_by_count = []
            for novel_id, episodes in novels.items():
                episode_count = len(episodes)
                novels_by_count.append((novel_id, episode_count, episodes))
            
            novels_by_count.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 Novels ranked by episode count:")
            print("-" * 60)
            print(f"{'Rank':<6} {'Novel ID':<10} {'Episodes':<10} {'Sample Title':<30}")
            print("-" * 60)
            
            for rank, (novel_id, count, episodes) in enumerate(novels_by_count, 1):
                # Get first episode title as sample
                episodes.sort(key=lambda x: x['episode_number'])
                sample_title = episodes[0]['episode_title'][:30] if episodes else "N/A"
                print(f"{rank:<6} {novel_id:<10} {count:<10} {sample_title:<30}")
                
                if rank == 20:  # Show top 20
                    break
            
            # Check Novel 25 specifically
            print(f"\n🎯 Novel 25 Analysis:")
            print("-" * 40)
            
            novel25_found = False
            for novel_id, count, episodes in novels_by_count:
                if novel_id == 25:
                    novel25_found = True
                    episodes.sort(key=lambda x: x['episode_number'])
                    
                    # Find rank
                    rank = next(i for i, (nid, _, _) in enumerate(novels_by_count, 1) if nid == 25)
                    
                    print(f"- Novel 25 rank: #{rank} out of {total_novels} novels")
                    print(f"- Episode count: {count}")
                    print(f"- Episode range: #{episodes[0]['episode_number']} - #{episodes[-1]['episode_number']}")
                    print(f"- First episode: {episodes[0]['episode_title']}")
                    print(f"- Last episode: {episodes[-1]['episode_title']}")
                    break
            
            if not novel25_found:
                print("- Novel 25: Not found in Milvus data")
            
            # Distribution analysis
            print(f"\n📈 Episode Count Distribution:")
            print("-" * 40)
            
            distribution = defaultdict(int)
            for _, count, _ in novels_by_count:
                if count >= 20:
                    distribution["20+ episodes"] += 1
                elif count >= 15:
                    distribution["15-19 episodes"] += 1
                elif count >= 10:
                    distribution["10-14 episodes"] += 1
                elif count >= 5:
                    distribution["5-9 episodes"] += 1
                else:
                    distribution["1-4 episodes"] += 1
            
            for category, novel_count in distribution.items():
                percentage = (novel_count / total_novels) * 100
                print(f"- {category}: {novel_count} novels ({percentage:.1f}%)")
                
        else:
            print("❌ Collection 'episode_embeddings' does not exist")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_milvus_data()