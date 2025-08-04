#!/usr/bin/env python3
"""
Direct test of Milvus search to verify data is properly stored.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.embedding.factory import get_embedding_manager
from src.embedding.base import EmbeddingRequest

def test_direct_milvus_search():
    """Test direct Milvus search to verify data."""
    print("🔍 Testing Direct Milvus Search")
    print("=" * 50)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize components
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        embedding_manager = get_embedding_manager([config.embedding])
        print("✅ Embedding manager initialized")
        
        # Check collection exists and has data
        if milvus_client.has_collection("episode_embeddings"):
            print("✅ Collection 'episode_embeddings' exists")
            
            # Get collection stats
            entity_count = milvus_client.get_entity_count("episode_embeddings")
            print(f"📊 Collection has {entity_count} entities")
            
            if entity_count > 0:
                # Generate query embedding
                test_query = "주인공"
                embedding_request = EmbeddingRequest(
                    input=[test_query],
                    model=config.embedding.model,
                    encoding_format="float"
                )
                embedding_response = embedding_manager.generate_embeddings(embedding_request)
                query_vector = embedding_response.embeddings[0]
                print(f"✅ Generated query embedding: dimension={len(query_vector)}")
                
                # Search using collection object
                collection = milvus_client.get_collection("episode_embeddings")
                
                search_params = {
                    "metric_type": "L2",
                    "params": {"nlist": 1024}
                }
                
                results = collection.search(
                    data=[query_vector],
                    anns_field="content_embedding",
                    param=search_params,
                    limit=5,
                    output_fields=["episode_id", "novel_id", "episode_number", "episode_title"]
                )
                
                print(f"✅ Search completed: found {len(results[0])} results")
                
                # Display results
                for i, hit in enumerate(results[0]):
                    print(f"  {i+1}. Episode {hit.entity.get('episode_id')} (Novel {hit.entity.get('novel_id')})")
                    print(f"     Title: {hit.entity.get('episode_title', '')}")
                    print(f"     Score: {hit.distance:.4f}")
                    print()
                
            else:
                print("❌ Collection is empty")
        else:
            print("❌ Collection 'episode_embeddings' does not exist")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_milvus_search()