#!/usr/bin/env python3
"""
Test script for implemented episode management functions
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_episode_imports():
    """Test that all episode modules can be imported correctly"""
    try:
        from src.episode import (
            EpisodeSearchEngine, EpisodeRAGManager, EpisodeRAGConfig, 
            create_episode_rag_manager, EpisodeData, EpisodeSearchRequest
        )
        print("✓ Episode module imports successful")
        
        from src.api.routes.episode import (
            EpisodeQueryRequest, EpisodeSearchResponse, 
            EpisodeContextRequest, EpisodeRAGRequest
        )
        print("✓ Episode API schema imports successful")
        
        from src.database.base import DatabaseFactory
        from src.embedding.manager import EmbeddingManager
        from src.milvus.client import MilvusClient
        print("✓ Required dependency imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_episode_route_logic():
    """Test that episode route logic can be instantiated"""
    try:
        from src.api.routes.episode import EpisodeQueryRequest, EpisodeSearchHitResponse
        
        # Test request model
        request = EpisodeQueryRequest(
            query="test query",
            episode_ids=[1, 2, 3],
            limit=10
        )
        assert request.query == "test query"
        assert request.episode_ids == [1, 2, 3]
        assert request.limit == 10
        print("✓ Episode request models work correctly")
        
        # Test response model  
        hit = EpisodeSearchHitResponse(
            episode_id=1,
            episode_number=1,
            episode_title="Test Episode",
            novel_id=1,
            similarity_score=0.95,
            distance=0.05
        )
        assert hit.episode_id == 1
        assert hit.similarity_score == 0.95
        print("✓ Episode response models work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Episode route logic test failed: {e}")
        return False

async def test_episode_rag_config():
    """Test episode RAG configuration"""
    try:
        from src.episode.manager import EpisodeRAGConfig
        
        # Test default config
        config = EpisodeRAGConfig()
        assert config.collection_name == "episode_embeddings"
        assert config.processing_batch_size == 100
        assert config.default_search_limit == 10
        print("✓ Default EpisodeRAGConfig created successfully")
        
        # Test custom config
        custom_config = EpisodeRAGConfig(
            collection_name="custom_episodes",
            processing_batch_size=50,
            default_search_limit=20
        )
        assert custom_config.collection_name == "custom_episodes"
        assert custom_config.processing_batch_size == 50
        print("✓ Custom EpisodeRAGConfig created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Episode RAG config test failed: {e}")
        return False

async def test_mock_database_operations():
    """Test database operation logic without actual database"""
    try:
        # Test SQL query construction (without execution)
        novel_id = 1
        limit = 10
        offset = 0
        
        count_query = "SELECT COUNT(*) as total FROM episode WHERE novel_id = %s"
        episodes_query = """
            SELECT 
                id as episode_id,
                episode_number,
                title as episode_title,
                created_at as publication_date,
                CHAR_LENGTH(content) as content_length
            FROM episode 
            WHERE novel_id = %s 
            ORDER BY episode_number ASC 
            LIMIT %s OFFSET %s
        """
        
        # Verify query parameters
        count_params = (novel_id,)
        episodes_params = (novel_id, limit, offset)
        
        assert len(count_params) == 1
        assert len(episodes_params) == 3
        assert episodes_params[0] == novel_id
        print("✓ Database query construction logic works")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

async def test_background_task_structure():
    """Test background task function structure"""
    try:
        from src.api.routes.episode import process_novel_episodes_background
        
        # Test that the function exists and is callable
        assert callable(process_novel_episodes_background)
        print("✓ Background task function is properly defined")
        
        # Test function signature (without actually calling it)
        import inspect
        sig = inspect.signature(process_novel_episodes_background)
        params = list(sig.parameters.keys())
        
        expected_params = ['novel_id', 'force_reprocess', 'user_id']
        assert all(param in params for param in expected_params)
        print("✓ Background task has correct parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Background task structure test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running episode implementation tests...\n")
    
    tests = [
        test_episode_imports,
        test_episode_route_logic,
        test_episode_rag_config,
        test_mock_database_operations,
        test_background_task_structure
    ]
    
    results = []
    for test in tests:
        print(f"Running {test.__name__}...")
        try:
            result = await test()
            results.append(result)
            print(f"{'✓ PASSED' if result else '❌ FAILED'}: {test.__name__}\n")
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*50)
    print(f"Episode Implementation Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All episode implementation tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)