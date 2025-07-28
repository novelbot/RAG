"""
Episode-based RAG Usage Examples.

This script demonstrates how to use the episode-based RAG system
for processing novel episodes and performing targeted searches.
"""

import asyncio
from typing import List, Dict, Any
from datetime import date

# Import episode RAG components
from src.episode import (
    EpisodeRAGManager, EpisodeRAGConfig,
    EpisodeSearchRequest, EpisodeSortOrder,
    create_episode_rag_manager
)
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient
from src.core.config import get_config


async def main():
    """Main example function."""
    print("Episode-based RAG System Usage Examples")
    print("=" * 50)
    
    # Initialize core components (these would be dependency injected in a real app)
    config = get_config()
    
    # Create database manager
    db_manager = DatabaseManager(config.database)
    
    # Create embedding manager
    embedding_manager = EmbeddingManager([])  # Configure with your providers
    
    # Create Milvus client
    milvus_client = MilvusClient(config.milvus)
    
    # Create Episode RAG Manager
    rag_config = EpisodeRAGConfig(
        collection_name="novel_episodes",
        embedding_model="text-embedding-ada-002",
        processing_batch_size=50,
        vector_dimension=1536
    )
    
    episode_manager = await create_episode_rag_manager(
        database_manager=db_manager,
        embedding_manager=embedding_manager,
        milvus_client=milvus_client,
        config=rag_config,
        setup_collection=True
    )
    
    try:
        # Example 1: Process a novel's episodes
        print("\n1. Processing Novel Episodes")
        print("-" * 30)
        
        novel_id = 1  # Example novel ID
        result = await episode_manager.process_novel(novel_id)
        print(f"Processing result: {result}")
        
        # Example 2: Search within specific episodes
        print("\n2. Episode-Filtered Search")
        print("-" * 30)
        
        # Search only in episodes 1, 2, 5, and 10
        episode_ids = [1, 2, 5, 10]
        search_result = await episode_manager.search_episodes(
            query="What happened to the main character?",
            episode_ids=episode_ids,
            limit=5,
            sort_by_episode_number=True
        )
        
        print(f"Found {search_result.total_count} results:")
        for hit in search_result.hits:
            print(f"  Episode {hit.episode_number}: {hit.episode_title}")
            print(f"    Similarity: {hit.similarity_score:.3f}")
            print(f"    Content preview: {hit.content[:100] if hit.content else 'N/A'}...")
            print()
        
        # Example 3: Get structured context for LLM
        print("\n3. Episode Context for LLM")
        print("-" * 30)
        
        context_result = episode_manager.get_episode_context(
            episode_ids=[1, 2, 3],
            query="character development",
            max_context_length=5000
        )
        
        print(f"Context includes {context_result['episodes_included']} episodes")
        print(f"Total length: {context_result['total_length']} characters")
        print(f"Episode order: {context_result['episode_order']}")
        print(f"Context preview: {context_result['context'][:200]}...")
        
        # Example 4: Advanced search with filters
        print("\n4. Advanced Episode Search")
        print("-" * 30)
        
        advanced_request = EpisodeSearchRequest(
            query="romance subplot",
            novel_ids=[1],  # Search within specific novel
            limit=10,
            similarity_threshold=0.7,
            sort_order=EpisodeSortOrder.EPISODE_NUMBER,
            episode_num_from=5,  # Start from episode 5
            episode_num_to=15,   # End at episode 15
            include_content=True,
            include_metadata=True
        )
        
        advanced_result = await episode_manager.search_engine.search_async(advanced_request)
        
        print(f"Advanced search found {advanced_result.total_count} results")
        print(f"Search time: {advanced_result.search_time:.3f}s")
        print(f"Results sorted by: {advanced_result.sort_order.value}")
        
        # Example 5: API Usage Simulation
        print("\n5. API Usage Simulation")
        print("-" * 30)
        
        # Simulate API request payload
        api_request = {
            "query": "What challenges did the protagonist face?",
            "episode_ids": [1, 3, 5, 7, 9],
            "limit": 5,
            "sort_order": "episode_number",
            "include_content": True,
            "similarity_threshold": 0.75
        }
        
        print("API Request:")
        print(f"  Query: {api_request['query']}")
        print(f"  Episode IDs: {api_request['episode_ids']}")
        print(f"  Sort order: {api_request['sort_order']}")
        
        # This would be handled by the API endpoint
        api_result = await episode_manager.search_episodes(
            query=api_request['query'],
            episode_ids=api_request['episode_ids'],
            limit=api_request['limit'],
            sort_by_episode_number=api_request['sort_order'] == 'episode_number'
        )
        
        print("\nAPI Response structure:")
        print(f"  Total results: {api_result.total_count}")
        print(f"  Search time: {api_result.search_time:.3f}s")
        print(f"  Context ordered by episode: {api_result.metadata.get('context_ordered_by_episode', False)}")
        
        # Example 6: Health Check and Statistics
        print("\n6. System Health and Statistics")
        print("-" * 30)
        
        health_status = episode_manager.health_check()
        print(f"Overall status: {health_status['status']}")
        
        stats = episode_manager.get_manager_stats()
        print(f"Processed novels: {stats['manager_stats']['processed_novels']}")
        print(f"Processed episodes: {stats['manager_stats']['processed_episodes']}")
        print(f"Total searches: {stats['manager_stats']['total_searches']}")
        
        # Example 7: Batch Processing Multiple Novels
        print("\n7. Batch Processing")
        print("-" * 30)
        
        novel_ids = [1, 2, 3]  # Example novel IDs
        for novel_id in novel_ids:
            try:
                result = await episode_manager.process_novel(novel_id)
                print(f"Novel {novel_id}: {result['episodes_processed']} episodes processed")
            except Exception as e:
                print(f"Novel {novel_id}: Processing failed - {e}")
        
        print("\nBatch processing completed!")
        print(f"Final statistics: {episode_manager.get_manager_stats()['manager_stats']}")
        
    finally:
        # Clean up resources
        episode_manager.close()
        print("\nResources cleaned up successfully!")


def demonstrate_api_payloads():
    """Demonstrate API request/response formats."""
    print("\nAPI Payload Examples")
    print("=" * 20)
    
    # Episode search request
    search_request = {
        "query": "주인공이 어떻게 되었나요?",
        "episode_ids": [1, 2, 5, 10],
        "limit": 10,
        "similarity_threshold": 0.7,
        "sort_order": "episode_number",
        "include_content": True,
        "include_metadata": True
    }
    
    print("1. Episode Search Request:")
    import json
    print(json.dumps(search_request, indent=2, ensure_ascii=False))
    
    # Expected response structure
    search_response = {
        "query": "주인공이 어떻게 되었나요?",
        "hits": [
            {
                "episode_id": 1,
                "episode_number": 1,
                "episode_title": "프롤로그",
                "novel_id": 1,
                "similarity_score": 0.95,
                "distance": 0.05,
                "content": "Episode content...",
                "publication_date": "2024-01-15",
                "content_length": 1500,
                "metadata": {}
            }
        ],
        "total_count": 4,
        "search_time": 0.234,
        "sort_order": "episode_number",
        "metadata": {
            "episode_ids_filter": [1, 2, 5, 10],
            "context_ordered_by_episode": True
        },
        "user_id": "user123"
    }
    
    print("\n2. Episode Search Response:")
    print(json.dumps(search_response, indent=2, ensure_ascii=False))
    
    # RAG request
    rag_request = {
        "query": "주인공의 성격이 어떻게 변했나요?",
        "episode_ids": [1, 5, 10, 15],
        "max_context_episodes": 5,
        "max_context_length": 8000,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    print("\n3. Episode RAG Request:")
    print(json.dumps(rag_request, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Show API payload examples
    demonstrate_api_payloads()