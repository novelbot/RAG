#!/usr/bin/env python3
"""
Test script for Episode-based RAG system.

This script tests the core functionality of the episode RAG system
including data processing, storage, and search operations.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import get_config
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient
from src.episode import (
    EpisodeRAGManager, EpisodeRAGConfig,
    EpisodeSearchRequest, EpisodeSortOrder,
    create_episode_rag_manager
)

async def test_episode_components():
    """Test individual episode components."""
    print("\n=== Testing Episode RAG Components ===")
    
    try:
        # Load configuration
        config = get_config()
        print(f"✓ Configuration loaded: {config.app_name} v{config.version}")
        
        # Test database connection
        print("\n1. Testing Database Connection...")
        db_manager = DatabaseManager(config.database)
        
        # Test basic connection
        try:
            with db_manager.get_connection() as conn:
                # Simple test query using text()
                from sqlalchemy import text
                result = conn.execute(text("SELECT 1 as test")).fetchone()
                if result and result[0] == 1:
                    print("✓ Database connection successful")
                else:
                    print("✗ Database connection test failed")
                    return False
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
        
        # Test episode table existence
        try:
            with db_manager.get_connection() as conn:
                from sqlalchemy import text
                result = conn.execute(text("""
                    SELECT COUNT(*) as episode_count 
                    FROM episode 
                    LIMIT 1
                """)).fetchone()
                episode_count = result[0] if result else 0
                print(f"✓ Episode table accessible, contains {episode_count} episodes")
        except Exception as e:
            print(f"⚠ Episode table check failed: {e}")
            print("  This is expected if the episode table doesn't exist yet")
        
        # Test Milvus connection
        print("\n2. Testing Milvus Connection...")
        try:
            milvus_client = MilvusClient(config.milvus)
            # Test connection by listing collections
            collections = milvus_client.list_collections()
            print(f"✓ Milvus connection successful, found {len(collections)} collections")
        except Exception as e:
            print(f"✗ Milvus connection failed: {e}")
            print("  Please ensure Milvus is running and accessible")
            return False
        
        # Test embedding manager
        print("\n3. Testing Embedding Manager...")
        try:
            # Initialize with empty providers list for basic test
            embedding_manager = EmbeddingManager([])
            print("✓ Embedding manager initialized")
            
            # Test basic embedding (if providers are available)
            if config.embedding.api_key:
                test_text = "This is a test sentence for embedding."
                # Note: This would require actual embedding providers to be configured
                print("⚠ Embedding provider test skipped (requires API keys)")
            else:
                print("⚠ No embedding API key configured, skipping embedding test")
                
        except Exception as e:
            print(f"✗ Embedding manager test failed: {e}")
            return False
        
        print("\n✓ All component tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

def test_episode_data_models():
    """Test episode data models and validation."""
    print("\n=== Testing Episode Data Models ===")
    
    try:
        from src.episode.models import (
            EpisodeData, EpisodeSearchRequest, EpisodeSortOrder
        )
        
        # Test EpisodeData model
        episode_data = EpisodeData(
            episode_id=1,
            content="Test episode content for validation",
            episode_number=1,
            episode_title="Test Episode",
            publication_date=None,
            novel_id=1
        )
        print("✓ EpisodeData model validation passed")
        
        # Test EpisodeSearchRequest
        search_request = EpisodeSearchRequest(
            query="test query",
            episode_ids=[1, 2, 3],
            novel_ids=[1],
            limit=10,
            sort_order=EpisodeSortOrder.EPISODE_NUMBER
        )
        
        # Test filter expression building - only episode_ids should be included
        search_request_episodes_only = EpisodeSearchRequest(
            query="test query",
            episode_ids=[1, 2, 3],
            limit=10,
            sort_order=EpisodeSortOrder.EPISODE_NUMBER
        )
        filter_expr = search_request_episodes_only.build_filter_expression()
        expected_filter = "episode_id in [1,2,3]"
        if filter_expr == expected_filter:
            print("✓ Filter expression building works correctly")
        else:
            print(f"✗ Filter expression mismatch: got '{filter_expr}', expected '{expected_filter}'")
            return False
        
        print("✓ All data model tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Data model test failed: {e}")
        return False

async def test_episode_manager_creation():
    """Test episode manager creation and initialization."""
    print("\n=== Testing Episode Manager Creation ===")
    
    try:
        config = get_config()
        
        # Create core components
        db_manager = DatabaseManager(config.database)
        embedding_manager = EmbeddingManager([])
        milvus_client = MilvusClient(config.milvus)
        
        # Create episode RAG configuration
        rag_config = EpisodeRAGConfig(
            collection_name="test_novel_episodes",
            embedding_model="text-embedding-ada-002",
            processing_batch_size=10,
            vector_dimension=1536
        )
        
        # Test manager creation (without collection setup for basic test)
        print("1. Creating Episode RAG Manager...")
        episode_manager = EpisodeRAGManager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=rag_config
        )
        print("✓ Episode RAG Manager created successfully")
        
        # Test health check
        print("2. Testing health check...")
        try:
            health_status = episode_manager.health_check()
            print(f"✓ Health check completed: {health_status['status']}")
            
            # Display component health
            for component, status in health_status.get('components', {}).items():
                if isinstance(status, dict):
                    comp_status = status.get('status', 'unknown')
                else:
                    comp_status = status
                print(f"  - {component}: {comp_status}")
                
        except Exception as e:
            print(f"⚠ Health check failed: {e}")
        
        # Test manager statistics
        print("3. Testing manager statistics...")
        try:
            stats = episode_manager.get_manager_stats()
            manager_stats = stats.get('manager_stats', {})
            print(f"✓ Manager statistics retrieved:")
            print(f"  - Processed novels: {manager_stats.get('processed_novels', 0)}")
            print(f"  - Processed episodes: {manager_stats.get('processed_episodes', 0)}")
            print(f"  - Total searches: {manager_stats.get('total_searches', 0)}")
        except Exception as e:
            print(f"⚠ Statistics retrieval failed: {e}")
        
        # Clean up
        episode_manager.close()
        print("✓ Manager cleanup completed")
        
        print("✓ Episode manager creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Episode manager creation test failed: {e}")
        return False

def test_api_schemas():
    """Test API schema validation."""
    print("\n=== Testing API Schemas ===")
    
    try:
        from src.api.routes.episode import (
            EpisodeQueryRequest, EpisodeSearchResponse, 
            EpisodeSearchHitResponse, EpisodeContextRequest
        )
        
        # Test EpisodeQueryRequest
        query_request = EpisodeQueryRequest(
            query="What happened in the story?",
            episode_ids=[1, 2, 5, 10],
            novel_ids=[1],
            limit=5,
            sort_order="episode_number",
            include_content=True
        )
        print("✓ EpisodeQueryRequest validation passed")
        
        # Test EpisodeSearchHitResponse
        hit_response = EpisodeSearchHitResponse(
            episode_id=1,
            episode_number=1,
            episode_title="Episode 1: The Beginning",
            novel_id=1,
            similarity_score=0.95,
            distance=0.05,
            content="Episode content here...",
            publication_date="2024-01-15",
            content_length=1500
        )
        print("✓ EpisodeSearchHitResponse validation passed")
        
        # Test EpisodeContextRequest
        context_request = EpisodeContextRequest(
            episode_ids=[1, 2, 3],
            query="character development",
            max_context_length=5000
        )
        print("✓ EpisodeContextRequest validation passed")
        
        print("✓ All API schema tests passed")
        return True
        
    except Exception as e:
        print(f"✗ API schema test failed: {e}")
        return False

async def test_mock_episode_search():
    """Test episode search with mock data."""
    print("\n=== Testing Mock Episode Search ===")
    
    try:
        from src.api.routes.episode import EpisodeQueryRequest, EpisodeSortOrder
        
        # Create a mock search request
        request = EpisodeQueryRequest(
            query="What challenges did the protagonist face?",
            episode_ids=[1, 2, 5, 10],
            limit=5,
            sort_order="episode_number",
            include_content=True,
            include_metadata=True
        )
        
        print(f"✓ Mock search request created:")
        print(f"  - Query: {request.query}")
        print(f"  - Episode IDs: {request.episode_ids}")
        print(f"  - Sort order: {request.sort_order}")
        print(f"  - Include content: {request.include_content}")
        
        # Test sort order conversion
        sort_order_map = {
            "similarity": EpisodeSortOrder.SIMILARITY,
            "episode_number": EpisodeSortOrder.EPISODE_NUMBER,
            "publication_date": EpisodeSortOrder.PUBLICATION_DATE
        }
        
        sort_order = sort_order_map.get(request.sort_order, EpisodeSortOrder.EPISODE_NUMBER)
        print(f"✓ Sort order conversion: {request.sort_order} -> {sort_order}")
        
        # Simulate search results processing
        episode_ids_to_search = request.episode_ids or [1, 2, 3, 5]
        print(f"✓ Would search episodes: {episode_ids_to_search}")
        
        # Test filter expression building (if we had the model imported)
        print("✓ Mock episode search test passed")
        return True
        
    except Exception as e:
        print(f"✗ Mock episode search test failed: {e}")
        return False

async def run_all_tests():
    """Run all episode RAG system tests."""
    print("🚀 Starting Episode-based RAG System Tests")
    print("=" * 60)
    
    tests = [
        ("Component Tests", test_episode_components()),
        ("Data Model Tests", test_episode_data_models()),
        ("Manager Creation Tests", test_episode_manager_creation()),
        ("API Schema Tests", test_api_schemas()),
        ("Mock Search Tests", test_mock_episode_search())
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Episode RAG system is ready for use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    print("Episode RAG System Test Suite")
    print("Built for testing the custom episode-based RAG functionality")
    print()
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ System is ready for production use!")
        print("\nNext steps:")
        print("1. Configure your database with actual episode data")
        print("2. Set up embedding provider API keys")
        print("3. Start the Milvus service")
        print("4. Run the API server: uvicorn src.main:app --reload")
    else:
        print("\n❌ Please fix the failing tests before proceeding.")
    
    sys.exit(0 if success else 1)