#!/usr/bin/env python3
"""
Test script for collection metadata management functionality
"""

import pytest
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_metadata_imports():
    """Test that all metadata management modules can be imported correctly"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        from src.milvus.client import MilvusClient
        from src.core.config import get_config
        print("✓ Collection metadata imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_metadata_cache_structure():
    """Test metadata cache structure and operations"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create a mock search engine with metadata cache functionality
        config = SearchConfig()
        
        # Mock MilvusClient
        mock_client = Mock()
        mock_client.has_collection.return_value = True
        
        # Create search engine instance
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Test cache initialization
        assert not hasattr(search_engine, '_metadata_cache') or search_engine._metadata_cache == {}
        print("✓ Metadata cache properly initialized")
        
        # Test cache key validation
        cache_key = "test_collection_metadata"
        assert not search_engine._is_metadata_cache_valid(cache_key)
        print("✓ Cache validation works correctly")
        
        # Test fallback metadata structure
        fallback_metadata = search_engine._get_fallback_metadata("test_collection")
        expected_fields = [
            'collection_name', 'description', 'vector_dim', 'fields',
            'vector_fields', 'scalar_fields', 'primary_field', 'auto_id',
            'num_entities', 'is_empty', 'num_partitions', 'partition_names',
            'indexes', 'has_vector_index', 'index_types', 'metric_types',
            'metadata_version', 'last_updated', 'is_fallback'
        ]
        
        for field in expected_fields:
            assert field in fallback_metadata
            print(f"✓ Fallback metadata contains '{field}' field")
        
        assert fallback_metadata['is_fallback'] is True
        assert fallback_metadata['vector_dim'] == 768
        print("✓ Fallback metadata structure is correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata cache structure test failed: {e}")
        return False

async def test_schema_extraction_logic():
    """Test schema information extraction logic"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create mock collection with schema
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        
        # Mock schema with fields
        mock_schema = Mock()
        mock_schema.description = "Test collection description"
        mock_schema.auto_id = False
        
        # Create mock fields
        mock_vector_field = Mock()
        mock_vector_field.name = "embedding"
        mock_vector_field.dtype = "FLOAT_VECTOR"
        mock_vector_field.dim = 1536
        mock_vector_field.is_primary = False
        mock_vector_field.description = "Vector embeddings"
        
        mock_primary_field = Mock()
        mock_primary_field.name = "id"
        mock_primary_field.dtype = "VARCHAR"
        mock_primary_field.is_primary = True
        mock_primary_field.max_length = 100
        mock_primary_field.description = "Primary key"
        
        mock_schema.fields = [mock_primary_field, mock_vector_field]
        mock_collection.schema = mock_schema
        
        # Mock client and create search engine
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Test schema extraction
        schema_info = search_engine._extract_schema_info(mock_collection)
        
        # Verify extracted information
        assert schema_info['description'] == "Test collection description"
        assert schema_info['vector_dim'] == 1536
        assert schema_info['auto_id'] is False
        assert len(schema_info['fields']) == 2
        assert len(schema_info['vector_fields']) == 1
        assert len(schema_info['scalar_fields']) == 1
        
        # Check primary field identification
        assert schema_info['primary_field'] is not None
        assert schema_info['primary_field']['name'] == "id"
        assert schema_info['primary_field']['is_primary'] is True
        
        # Check vector field identification
        vector_field = schema_info['vector_fields'][0]
        assert vector_field['name'] == "embedding"
        assert vector_field['dimension'] == 1536
        
        print("✓ Schema extraction logic works correctly")
        print(f"✓ Extracted {len(schema_info['fields'])} fields")
        print(f"✓ Identified vector dimension: {schema_info['vector_dim']}")
        print(f"✓ Found primary field: {schema_info['primary_field']['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema extraction logic test failed: {e}")
        return False

async def test_collection_stats_extraction():
    """Test collection statistics extraction"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create mock collection with stats
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.num_entities = 1000
        mock_collection.is_empty = False
        
        # Mock partitions
        mock_partition1 = Mock()
        mock_partition1.name = "_default"
        mock_partition2 = Mock()
        mock_partition2.name = "partition_1"
        
        mock_collection.partitions = [mock_partition1, mock_partition2]
        
        # Create search engine
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Test stats extraction
        stats_info = search_engine._extract_collection_stats(mock_collection)
        
        # Verify extracted statistics
        assert stats_info['num_entities'] == 1000
        assert stats_info['is_empty'] is False
        assert stats_info['num_partitions'] == 2
        assert len(stats_info['partition_names']) == 2
        assert "_default" in stats_info['partition_names']
        assert "partition_1" in stats_info['partition_names']
        
        print("✓ Collection statistics extraction works correctly")
        print(f"✓ Found {stats_info['num_entities']} entities")
        print(f"✓ Found {stats_info['num_partitions']} partitions")
        print(f"✓ Collection is {'empty' if stats_info['is_empty'] else 'not empty'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection stats extraction test failed: {e}")
        return False

async def test_index_info_extraction():
    """Test index information extraction"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create mock collection with indexes
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        
        # Mock index
        mock_index = Mock()
        mock_index.field_name = "embedding"
        mock_index.params = {
            'index_type': 'IVF_FLAT',
            'metric_type': 'L2',
            'params': {'nlist': 1024}
        }
        
        mock_collection.indexes = [mock_index]
        
        # Create search engine
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Test index extraction
        index_info = search_engine._extract_index_info(mock_collection)
        
        # Verify extracted index information
        assert len(index_info['indexes']) == 1
        assert index_info['has_vector_index'] is True
        assert 'IVF_FLAT' in index_info['index_types']
        assert 'L2' in index_info['metric_types']
        
        # Check specific index details
        index = index_info['indexes'][0]
        assert index['field_name'] == "embedding"
        assert index['index_type'] == 'IVF_FLAT'
        assert index['metric_type'] == 'L2'
        assert index['params']['nlist'] == 1024
        
        print("✓ Index information extraction works correctly")
        print(f"✓ Found {len(index_info['indexes'])} indexes")
        print(f"✓ Index types: {index_info['index_types']}")
        print(f"✓ Metric types: {index_info['metric_types']}")
        print(f"✓ Has vector index: {index_info['has_vector_index']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Index info extraction test failed: {e}")
        return False

async def test_metadata_caching():
    """Test metadata caching functionality"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create search engine with mocked client
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Initialize cache
        search_engine._metadata_cache = {}
        search_engine._cache_timestamps = {}
        
        # Test cache operations
        cache_key = "test_collection_metadata"
        test_metadata = {
            'collection_name': 'test_collection',
            'vector_dim': 1536,
            'description': 'Test collection'
        }
        
        # Cache metadata
        search_engine._metadata_cache[cache_key] = test_metadata
        search_engine._cache_timestamps[cache_key] = time.time()
        
        # Test cache validity (should be valid immediately)
        assert search_engine._is_metadata_cache_valid(cache_key) is True
        print("✓ Fresh cache is valid")
        
        # Test expired cache
        search_engine._cache_timestamps[cache_key] = time.time() - 400  # 400 seconds ago
        assert search_engine._is_metadata_cache_valid(cache_key) is False
        print("✓ Expired cache is invalid")
        
        # Test cache clearing
        search_engine.clear_metadata_cache("test_collection")
        assert cache_key not in search_engine._metadata_cache
        assert cache_key not in search_engine._cache_timestamps
        print("✓ Single collection cache clearing works")
        
        # Test clearing all cache
        search_engine._metadata_cache[cache_key] = test_metadata
        search_engine._cache_timestamps[cache_key] = time.time()
        search_engine.clear_metadata_cache()
        assert len(search_engine._metadata_cache) == 0
        assert len(search_engine._cache_timestamps) == 0
        print("✓ Full cache clearing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata caching test failed: {e}")
        return False

async def test_schema_summary_generation():
    """Test schema summary generation for external use"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create search engine with mocked metadata
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Mock the _get_collection_metadata method
        test_metadata = {
            'collection_name': 'test_collection',
            'description': 'Test collection for schema summary',
            'vector_dim': 1536,
            'num_entities': 5000,
            'vector_fields': [{'name': 'embedding', 'dimension': 1536}],
            'scalar_fields': [{'name': 'id'}, {'name': 'text'}],
            'has_vector_index': True,
            'index_types': ['IVF_FLAT'],
            'partition_names': ['_default', 'partition_1']
        }
        
        # Mock the metadata retrieval
        search_engine._get_collection_metadata = Mock(return_value=test_metadata)
        
        # Test schema summary generation
        summary = search_engine.get_collection_schema_summary('test_collection')
        
        # Verify summary structure
        expected_fields = [
            'collection_name', 'description', 'vector_dimension',
            'total_entities', 'vector_fields', 'scalar_fields',
            'has_index', 'index_types', 'partitions'
        ]
        
        for field in expected_fields:
            assert field in summary
            print(f"✓ Schema summary contains '{field}' field")
        
        # Verify summary values
        assert summary['collection_name'] == 'test_collection'
        assert summary['vector_dimension'] == 1536
        assert summary['total_entities'] == 5000
        assert summary['vector_fields'] == 1
        assert summary['scalar_fields'] == 2
        assert summary['has_index'] is True
        assert 'IVF_FLAT' in summary['index_types']
        assert '_default' in summary['partitions']
        
        print("✓ Schema summary generation works correctly")
        print(f"✓ Collection: {summary['collection_name']}")
        print(f"✓ Vector dimension: {summary['vector_dimension']}")
        print(f"✓ Total entities: {summary['total_entities']}")
        print(f"✓ Vector fields: {summary['vector_fields']}")
        print(f"✓ Scalar fields: {summary['scalar_fields']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema summary generation test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling in metadata management"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        
        # Create search engine
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Test error handling with non-existent collection
        mock_client.has_collection.return_value = False
        
        # This should return fallback metadata instead of raising
        metadata = search_engine._get_collection_metadata('nonexistent_collection')
        
        assert metadata['is_fallback'] is True
        assert metadata['collection_name'] == 'nonexistent_collection'
        assert metadata['vector_dim'] == 768  # Default fallback
        print("✓ Error handling returns fallback metadata")
        
        # Test schema summary error handling
        search_engine._get_collection_metadata = Mock(side_effect=Exception("Test error"))
        
        summary = search_engine.get_collection_schema_summary('error_collection')
        assert 'error' in summary
        assert summary['collection_name'] == 'error_collection'
        assert summary['vector_dimension'] == 768  # Default fallback
        print("✓ Schema summary error handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def test_integration_with_get_collection():
    """Test integration of metadata management with _get_collection method"""
    try:
        from src.rag.vector_search_engine import VectorSearchEngine, SearchConfig
        from src.milvus.collection import MilvusCollection
        
        # Create search engine
        mock_client = Mock()
        config = SearchConfig()
        search_engine = VectorSearchEngine(mock_client, config)
        
        # Mock metadata retrieval
        test_metadata = {
            'collection_name': 'test_collection',
            'description': 'Integration test collection',
            'vector_dim': 1536,
            'num_entities': 1000
        }
        
        search_engine._get_collection_metadata = Mock(return_value=test_metadata)
        
        # Mock MilvusCollection to avoid actual instantiation
        with patch('src.rag.vector_search_engine.MilvusCollection') as MockMilvusCollection:
            mock_collection_instance = Mock()
            MockMilvusCollection.return_value = mock_collection_instance
            
            # Test _get_collection method
            result = search_engine._get_collection('test_collection')
            
            # Verify that metadata was retrieved and used
            search_engine._get_collection_metadata.assert_called_once_with('test_collection')
            
            # Verify MilvusCollection was created with correct parameters
            MockMilvusCollection.assert_called_once()
            call_args = MockMilvusCollection.call_args
            
            assert call_args[1]['client'] == mock_client
            assert call_args[1]['schema'] is not None
            
            assert result == mock_collection_instance
            print("✓ Integration with _get_collection works correctly")
            print("✓ Metadata is properly retrieved and used")
            print("✓ MilvusCollection is created with correct parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running collection metadata management tests...\n")
    
    tests = [
        test_metadata_imports,
        test_metadata_cache_structure,
        test_schema_extraction_logic,
        test_collection_stats_extraction,
        test_index_info_extraction,
        test_metadata_caching,
        test_schema_summary_generation,
        test_error_handling,
        test_integration_with_get_collection
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
    print(f"Collection Metadata Management Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All collection metadata management tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)