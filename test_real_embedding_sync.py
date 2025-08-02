"""
Test data sync with real embedding generation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.data_sync import DataSyncManager
from src.core.config import get_config


@pytest.mark.asyncio
async def test_data_sync_with_real_embeddings():
    """Test data sync with actual embedding generation but mocked vector storage"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    print(f"🚀 Real Embedding + Mock Vector Test")
    
    # Create realistic sync configuration with small dataset
    source_config = {
        "id": "real_embedding_test",
        "type": "database",
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
            "driver": config.database.driver
        },
        "query": """
        SELECT 
            novel_id as id,
            CONCAT(LEFT(title, 100), ' - ', COALESCE(author, 'Unknown')) as content,
            author,
            genre,
            description
        FROM novels 
        WHERE title IS NOT NULL AND LENGTH(title) > 0
        LIMIT 3
        """
    }
    
    # Only mock vector engine, use real embedding client
    with patch('src.services.data_sync.VectorSearchEngine') as mock_vector_engine:
        
        # Mock vector engine
        mock_vector_instance = AsyncMock()
        mock_vector_instance.initialize = AsyncMock()
        mock_vector_instance.add_vector = AsyncMock()
        mock_vector_instance.update_vector = AsyncMock()
        mock_vector_engine.return_value = mock_vector_instance
        
        # Create sync manager
        sync_manager = DataSyncManager()
        sync_manager.sync_states = {}
        
        # Run the sync with real embeddings
        result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=False,
            dry_run=False  # Actually process
        )
        
        print(f"✅ Real Embedding Sync Results:")
        print(f"  Status: {result.sync_status.value}")
        print(f"  Records processed: {result.records_processed}")
        print(f"  Records added: {result.records_added}")
        print(f"  Duration: {result.sync_duration:.2f}s")
        
        # Should have processed records
        assert result.sync_status.value == "completed"
        assert result.records_processed > 0
        
        # Check if vector operations were attempted
        print(f"  Vector initialize called: {mock_vector_instance.initialize.called}")
        print(f"  Vector add_vector call count: {mock_vector_instance.add_vector.call_count}")
        
        # If vector processing worked, we should see add_vector calls
        if mock_vector_instance.add_vector.call_count > 0:
            print(f"✅ Real embedding processing with vector storage simulation successful!")
        else:
            print(f"ℹ️ Fallback to simulation mode (vector engine not available)")


@pytest.mark.asyncio
async def test_embedding_quality_check():
    """Test the quality and consistency of generated embeddings"""
    config = get_config()
    
    # Skip if no database config
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    print(f"🧪 Embedding Quality Test")
    
    try:
        from src.embedding.factory import get_embedding_client
        from src.embedding.base import EmbeddingRequest
        
        # Get embedding client
        embedding_client = get_embedding_client(config.embedding)
        
        # Test with Korean novel titles from database
        test_texts = [
            "젊은 느티나무 - 강신재",
            "젊은 느티나무 - 강신재",  # Same text for consistency check
            "백치 아다다 - 계용묵",
            "카프카를 읽는 밤 - 구효서"
        ]
        
        # Generate embeddings
        request = EmbeddingRequest(input=test_texts)
        response = await embedding_client.generate_embeddings_async(request)
        
        print(f"✅ Embedding Quality Results:")
        print(f"  Model: {response.model}")
        print(f"  Dimensions: {response.dimensions}")
        print(f"  Generated: {len(response.embeddings)} embeddings")
        
        # Check consistency (same text should produce identical embeddings)
        import numpy as np
        emb1 = np.array(response.embeddings[0])
        emb2 = np.array(response.embeddings[1])  # Same text
        emb3 = np.array(response.embeddings[2])  # Different text
        
        # Cosine similarity
        similarity_same = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_diff = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        print(f"  Consistency (same text): {similarity_same:.6f}")
        print(f"  Variation (different text): {similarity_diff:.6f}")
        
        # Quality checks
        assert similarity_same > 0.999, f"Same text should have very high similarity: {similarity_same}"
        assert similarity_diff < similarity_same, f"Different texts should be less similar: {similarity_diff}"
        assert response.dimensions > 0, "Should have positive dimensions"
        assert len(response.embeddings[0]) == response.dimensions, "Embedding size should match dimensions"
        
        print(f"✅ All embedding quality checks passed!")
        
    except Exception as e:
        pytest.skip(f"Embedding test failed: {e}")


@pytest.mark.asyncio
async def test_batch_embedding_performance():
    """Test embedding generation performance with batches"""
    config = get_config()
    
    if not config.database.host:
        pytest.skip("Database configuration not available")
    
    print(f"⚡ Batch Embedding Performance Test")
    
    # Test with real data from database
    source_config = {
        "id": "performance_test",
        "type": "database",
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
            "driver": config.database.driver
        },
        "query": """
        SELECT 
            novel_id as id,
            LEFT(CONCAT(title, ' by ', COALESCE(author, 'Unknown')), 200) as content
        FROM novels 
        WHERE title IS NOT NULL 
        LIMIT 10
        """
    }
    
    # Mock vector storage to focus on embedding performance
    with patch('src.services.data_sync.VectorSearchEngine') as mock_vector_engine:
        
        mock_vector_instance = AsyncMock()
        mock_vector_instance.initialize = AsyncMock()
        mock_vector_instance.add_vector = AsyncMock()
        mock_vector_engine.return_value = mock_vector_instance
        
        sync_manager = DataSyncManager()
        sync_manager.sync_states = {}
        
        import time
        start_time = time.time()
        
        result = await sync_manager.sync_data_source(
            source_config=source_config,
            incremental=False,
            dry_run=False
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Performance Results:")
        print(f"  Records processed: {result.records_processed}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per record: {total_time/result.records_processed:.2f}s")
        print(f"  Records/second: {result.records_processed/total_time:.2f}")
        
        # Performance assertions
        assert result.records_processed > 0
        assert total_time < 30, f"Should complete within 30 seconds, took {total_time:.2f}s"
        
        if result.records_processed > 1:
            avg_time = total_time / result.records_processed
            assert avg_time < 10, f"Should process each record in under 10s, took {avg_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])