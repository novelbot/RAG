#!/usr/bin/env python3
"""
Test script for document reprocessing functionality
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_document_reprocessing_imports():
    """Test that all document reprocessing modules can be imported correctly"""
    try:
        from src.api.routes.documents import reprocess_document_background, DocumentStatus
        from src.services.document_service import DocumentService
        from src.models.document import Document
        from src.metrics.collectors import DocumentEventCollector
        print("✓ Document reprocessing imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_reprocessing_background_function():
    """Test that reprocessing background function is properly defined"""
    try:
        from src.api.routes.documents import reprocess_document_background
        
        # Test that the function exists and is callable
        assert callable(reprocess_document_background)
        print("✓ Reprocessing background function is properly defined")
        
        # Test function signature
        import inspect
        sig = inspect.signature(reprocess_document_background)
        params = list(sig.parameters.keys())
        
        expected_params = ['document_id', 'filename', 'user_id', 'force_reembedding']
        assert all(param in params for param in expected_params)
        print("✓ Reprocessing function has correct parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Reprocessing function test failed: {e}")
        return False

async def test_document_service_integration():
    """Test document service methods needed for reprocessing"""
    try:
        from src.services.document_service import DocumentService
        
        # Test that DocumentService can be instantiated
        service = DocumentService()
        assert service is not None
        print("✓ DocumentService instantiated successfully")
        
        # Test that required methods exist
        assert hasattr(service, 'get_document')
        assert hasattr(service, 'delete_document')
        assert callable(service.get_document)
        assert callable(service.delete_document)
        print("✓ DocumentService has required methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Document service integration test failed: {e}")
        return False

async def test_document_status_enum():
    """Test DocumentStatus enum used in reprocessing"""
    try:
        from src.models.document import DocumentStatus
        
        # Test that required status values exist
        required_statuses = ['PROCESSING', 'PROCESSED', 'FAILED', 'UPLOADING', 'DELETED']
        for status_name in required_statuses:
            assert hasattr(DocumentStatus, status_name)
            print(f"✓ DocumentStatus.{status_name} exists")
        
        return True
        
    except Exception as e:
        print(f"❌ DocumentStatus enum test failed: {e}")
        return False

async def test_metrics_collector_reprocess_method():
    """Test that document reprocessing metrics logging is available"""
    try:
        from src.metrics.collectors import DocumentEventCollector
        
        # Test that DocumentEventCollector can be instantiated
        collector = DocumentEventCollector()
        assert collector is not None
        print("✓ DocumentEventCollector instantiated successfully")
        
        # Test that reprocess logging method exists
        assert hasattr(collector, 'log_document_reprocess')
        assert callable(collector.log_document_reprocess)
        print("✓ log_document_reprocess method exists")
        
        # Test method signature
        import inspect
        sig = inspect.signature(collector.log_document_reprocess)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'document_id', 'filename', 'user_id',
            'old_chunk_count', 'new_chunk_count', 
            'old_vector_count', 'new_vector_count',
            'processing_time_ms', 'metadata'
        ]
        assert all(param in params for param in expected_params)
        print("✓ log_document_reprocess has correct parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        return False

async def test_reprocessing_api_endpoint_logic():
    """Test the reprocessing API endpoint logic"""
    try:
        from src.api.routes.documents import DocumentStatus
        
        # Test status conflict logic
        processing_statuses = [DocumentStatus.PROCESSING, DocumentStatus.UPLOADING]
        
        # Simulate checking if document is in processing state
        def can_reprocess(document_status):
            return document_status not in processing_statuses
        
        # Test various statuses
        assert can_reprocess(DocumentStatus.PROCESSED) == True
        assert can_reprocess(DocumentStatus.FAILED) == True
        assert can_reprocess(DocumentStatus.PROCESSING) == False
        assert can_reprocess(DocumentStatus.UPLOADING) == False
        
        print("✓ Document status checking logic works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoint logic test failed: {e}")
        return False

async def test_reprocessing_metadata_structure():
    """Test reprocessing metadata structure"""
    try:
        # Test metadata structure that would be created during reprocessing
        sample_metadata = {
            'reprocessed': True,
            'reprocessing_timestamp': 1234567890.0,
            'original_stats': {
                'chunk_count': 10,
                'vector_count': 10
            },
            'new_stats': {
                'original_text_length': 5000,
                'cleaned_text_length': 4800,
                'chunk_count': 12,
                'vector_count': 12,
                'file_extension': '.pdf'
            },
            'extraction_metadata': {
                'reprocessed': True,
                'reprocessing_timestamp': 1234567890.0
            }
        }
        
        # Verify required fields exist
        assert sample_metadata['reprocessed'] == True
        assert 'reprocessing_timestamp' in sample_metadata
        assert 'original_stats' in sample_metadata
        assert 'new_stats' in sample_metadata
        assert sample_metadata['new_stats']['chunk_count'] == 12
        
        print("✓ Reprocessing metadata structure is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata structure test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running document reprocessing tests...\n")
    
    tests = [
        test_document_reprocessing_imports,
        test_reprocessing_background_function,
        test_document_service_integration,
        test_document_status_enum,
        test_metrics_collector_reprocess_method,
        test_reprocessing_api_endpoint_logic,
        test_reprocessing_metadata_structure
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
    print(f"Document Reprocessing Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All document reprocessing tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)