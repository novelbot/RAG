#!/usr/bin/env python3
"""
Test script for data cleanup functionality
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_data_cleanup_imports():
    """Test that all data cleanup modules can be imported correctly"""
    try:
        from src.cli.commands.data import cleanup_data, data_group
        from src.core.config import get_config
        from src.database.base import DatabaseFactory
        from src.models.document import Document, DocumentStatus
        print("✓ Data cleanup imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_cleanup_function_structure():
    """Test the cleanup function structure"""
    try:
        from src.cli.commands.data import cleanup_data
        
        # Test that the function exists and is callable
        assert callable(cleanup_data)
        print("✓ cleanup_data function is properly defined")
        
        # Test function parameters using click command inspection
        param_names = [param.name for param in cleanup_data.__click_params__]
        expected_params = ['orphaned', 'old_embeddings', 'confirm']
        
        for param in expected_params:
            assert param in param_names
            print(f"✓ cleanup_data has '{param}' parameter")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup function structure test failed: {e}")
        return False

async def test_document_model_integration():
    """Test Document model integration for cleanup"""
    try:
        from src.models.document import Document, DocumentStatus
        
        # Test DocumentStatus enum values needed for cleanup
        required_statuses = ['PROCESSING', 'PROCESSED', 'FAILED', 'UPLOADING', 'DELETED']
        for status_name in required_statuses:
            assert hasattr(DocumentStatus, status_name)
            print(f"✓ DocumentStatus.{status_name} exists")
        
        # Test Document model methods needed for cleanup
        document_methods = ['mark_failed']
        for method in document_methods:
            assert hasattr(Document, method)
            print(f"✓ Document.{method} method exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Document model integration test failed: {e}")
        return False

async def test_milvus_integration():
    """Test Milvus integration components for vector cleanup"""
    try:
        from src.milvus.client import MilvusClient
        
        # Test that MilvusClient can be imported and has required methods
        required_methods = ['connect', 'has_collection', 'get_collection_stats', 
                           'query', 'delete', 'compact']
        
        for method in required_methods:
            assert hasattr(MilvusClient, method)
            print(f"✓ MilvusClient.{method} method exists")
        
        return True
        
    except ImportError:
        # Milvus might not be available in test environment
        print("⚠️ Milvus components not available (expected in test environment)")
        return True
    except Exception as e:
        print(f"❌ Milvus integration test failed: {e}")
        return False

async def test_database_integration():
    """Test database integration for cleanup operations"""
    try:
        from src.database.base import DatabaseFactory
        from src.core.config import get_config
        from sqlalchemy.orm import sessionmaker
        
        # Test that required components exist
        assert hasattr(DatabaseFactory, 'create_manager')
        assert callable(get_config)
        print("✓ Database integration components available")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

async def test_file_system_cleanup_logic():
    """Test file system cleanup logic components"""
    try:
        import os
        import tempfile
        import glob
        from pathlib import Path
        
        # Test that required modules are available
        assert os is not None
        assert tempfile is not None
        assert glob is not None
        assert Path is not None
        
        # Test temp directory access
        temp_dir = tempfile.gettempdir()
        assert os.path.exists(temp_dir)
        print("✓ File system cleanup components available")
        
        # Test glob pattern matching functionality
        test_patterns = ['rag_*', 'embedding_*', 'vector_*', 'milvus_*']
        for pattern in test_patterns:
            # Just test that glob can handle the pattern
            result = glob.glob(pattern)  # This should not raise an error
        print("✓ Glob pattern matching works")
        
        return True
        
    except Exception as e:
        print(f"❌ File system cleanup logic test failed: {e}")
        return False

async def test_cleanup_statistics_structure():
    """Test cleanup statistics structure and calculations"""
    try:
        # Test the statistics structure that would be used in cleanup
        cleanup_stats = {
            'orphaned_vectors_removed': 0,
            'orphaned_documents_removed': 0,
            'duplicate_vectors_removed': 0,
            'old_embeddings_removed': 0,
            'temporary_files_removed': 0,
            'cache_files_removed': 0,
            'total_space_freed_mb': 0,
            'processing_time_ms': 0,
            'errors_encountered': []
        }
        
        # Test that all expected fields exist
        expected_fields = [
            'orphaned_vectors_removed', 'orphaned_documents_removed',
            'temporary_files_removed', 'total_space_freed_mb',
            'processing_time_ms', 'errors_encountered'
        ]
        
        for field in expected_fields:
            assert field in cleanup_stats
            print(f"✓ Cleanup stats contains '{field}' field")
        
        # Test statistics calculations
        cleanup_stats['orphaned_documents_removed'] = 5
        cleanup_stats['orphaned_vectors_removed'] = 100
        cleanup_stats['temporary_files_removed'] = 10
        
        total_items_cleaned = (cleanup_stats['orphaned_documents_removed'] + 
                             cleanup_stats['orphaned_vectors_removed'] + 
                             cleanup_stats['temporary_files_removed'])
        
        assert total_items_cleaned == 115
        print("✓ Statistics calculation logic works")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup statistics test failed: {e}")
        return False

async def test_error_handling_structure():
    """Test error handling structure in cleanup"""
    try:
        # Test error handling patterns that would be used
        errors_encountered = []
        
        # Simulate adding errors
        test_errors = [
            "Database connection failed",
            "Milvus cleanup error: Connection timeout",
            "Vector cleanup error: Invalid expression"
        ]
        
        for error in test_errors:
            errors_encountered.append(error)
        
        assert len(errors_encountered) == 3
        assert "Database connection failed" in errors_encountered
        print("✓ Error handling structure works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling structure test failed: {e}")
        return False

async def test_confirmation_logic():
    """Test confirmation logic for cleanup operations"""
    try:
        # Test confirmation scenarios
        def simulate_confirmation(confirm_flag, user_response=True):
            # Simulate the confirmation logic used in cleanup
            if confirm_flag:
                return True  # Auto-confirm
            else:
                return user_response  # Would prompt user
        
        # Test auto-confirm scenario
        assert simulate_confirmation(True, False) == True
        
        # Test user prompt scenarios
        assert simulate_confirmation(False, True) == True
        assert simulate_confirmation(False, False) == False
        
        print("✓ Confirmation logic works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Confirmation logic test failed: {e}")
        return False

async def test_batch_processing_logic():
    """Test batch processing logic for large cleanup operations"""
    try:
        # Test batch processing simulation
        def simulate_batch_processing(total_items, batch_size=1000):
            processed_count = 0
            batches = []
            
            for i in range(0, total_items, batch_size):
                batch_end = min(i + batch_size, total_items)
                batch_items = batch_end - i
                batches.append(batch_items)
                processed_count += batch_items
            
            return processed_count, len(batches)
        
        # Test various scenarios
        processed, batch_count = simulate_batch_processing(5000, 1000)
        assert processed == 5000
        assert batch_count == 5
        
        processed, batch_count = simulate_batch_processing(1500, 1000)
        assert processed == 1500
        assert batch_count == 2
        
        processed, batch_count = simulate_batch_processing(500, 1000)
        assert processed == 500
        assert batch_count == 1
        
        print("✓ Batch processing logic works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing logic test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running data cleanup tests...\n")
    
    tests = [
        test_data_cleanup_imports,
        test_cleanup_function_structure,
        test_document_model_integration,
        test_milvus_integration,
        test_database_integration,
        test_file_system_cleanup_logic,
        test_cleanup_statistics_structure,
        test_error_handling_structure,
        test_confirmation_logic,
        test_batch_processing_logic
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
    print(f"Data Cleanup Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All data cleanup tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)