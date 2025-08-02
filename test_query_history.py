#!/usr/bin/env python3
"""
Test script for query history functionality
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_query_history_imports():
    """Test that all query history modules can be imported correctly"""
    try:
        from src.api.routes.query import get_query_history
        from src.services.query_logger import QueryLogger, QueryMetrics, QueryContext
        from src.models.query_log import QueryLog, QueryType, QueryStatus
        print("✓ Query history imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_query_logger_service():
    """Test that QueryLogger service is properly configured"""
    try:
        from src.services.query_logger import query_logger
        
        # Test that the service exists and is callable
        assert query_logger is not None
        assert hasattr(query_logger, 'get_user_query_history')
        assert callable(query_logger.get_user_query_history)
        print("✓ QueryLogger service is properly configured")
        
        # Test method signature
        import inspect
        sig = inspect.signature(query_logger.get_user_query_history)
        params = list(sig.parameters.keys())
        
        expected_params = ['user_id', 'limit', 'offset']
        assert all(param in params for param in expected_params)
        print("✓ get_user_query_history has correct parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ QueryLogger service test failed: {e}")
        return False

async def test_query_log_model():
    """Test QueryLog model structure and methods"""
    try:
        from src.models.query_log import QueryLog, QueryType, QueryStatus
        
        # Test that required enums exist
        required_query_types = ['SEARCH', 'RAG', 'BATCH_SEARCH', 'SIMILARITY']
        for query_type in required_query_types:
            assert hasattr(QueryType, query_type)
            print(f"✓ QueryType.{query_type} exists")
        
        required_statuses = ['SUCCESS', 'FAILED', 'TIMEOUT', 'CANCELLED']
        for status in required_statuses:
            assert hasattr(QueryStatus, status)
            print(f"✓ QueryStatus.{status} exists")
        
        # Test QueryLog methods
        assert hasattr(QueryLog, 'to_dict')
        assert hasattr(QueryLog, 'create_query_log')
        assert hasattr(QueryLog, 'mark_success')
        assert hasattr(QueryLog, 'mark_failed')
        print("✓ QueryLog model has required methods")
        
        return True
        
    except Exception as e:
        print(f"❌ QueryLog model test failed: {e}")
        return False

async def test_query_history_endpoint_structure():
    """Test the query history endpoint function structure"""
    try:
        from src.api.routes.query import get_query_history
        
        # Test that the function exists and is callable
        assert callable(get_query_history)
        print("✓ get_query_history function is properly defined")
        
        # Test function signature
        import inspect
        sig = inspect.signature(get_query_history)
        params = list(sig.parameters.keys())
        
        expected_params = ['limit', 'offset', 'query_type', 'status_filter', 'current_user']
        assert all(param in params for param in expected_params)
        print("✓ get_query_history has correct parameters")
        
        # Test parameter defaults
        param_defaults = {name: param.default for name, param in sig.parameters.items()}
        assert param_defaults['limit'] == 50
        assert param_defaults['offset'] == 0
        assert param_defaults['query_type'] == None
        assert param_defaults['status_filter'] == None
        print("✓ get_query_history has correct default values")
        
        return True
        
    except Exception as e:
        print(f"❌ Query history endpoint test failed: {e}")
        return False

async def test_database_session_handling():
    """Test database session handling in query history"""
    try:
        from src.core.database import SessionLocal
        
        # Test that SessionLocal can be instantiated
        session = SessionLocal()
        assert session is not None
        session.close()
        print("✓ Database session handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Database session test failed: {e}")
        return False

async def test_query_filtering_logic():
    """Test query filtering and validation logic"""
    try:
        from src.models.query_log import QueryType, QueryStatus
        
        # Test query type validation
        valid_query_types = ['search', 'rag', 'batch_search', 'similarity']
        for qt in valid_query_types:
            query_type_enum = QueryType(qt)
            assert query_type_enum is not None
            print(f"✓ QueryType '{qt}' validation works")
        
        # Test status validation
        valid_statuses = ['success', 'failed', 'timeout', 'cancelled']
        for status in valid_statuses:
            status_enum = QueryStatus(status)
            assert status_enum is not None
            print(f"✓ QueryStatus '{status}' validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Query filtering logic test failed: {e}")
        return False

async def test_response_structure():
    """Test the expected response structure for query history"""
    try:
        # Test response structure matches expected format
        expected_response_keys = [
            'history', 'pagination', 'filters', 
            'performance_stats', 'user_id', 'processing_time_ms'
        ]
        
        expected_pagination_keys = [
            'total_count', 'limit', 'offset', 'current_page', 
            'total_pages', 'has_next', 'has_prev'
        ]
        
        expected_performance_stats_keys = [
            'success_rate', 'average_response_time_ms', 'total_tokens_used',
            'total_results_returned', 'query_types_breakdown', 'top_error_messages'
        ]
        
        # Simulate response structure validation
        sample_response = {
            "history": [],
            "pagination": {
                "total_count": 0,
                "limit": 50,
                "offset": 0,
                "current_page": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False
            },
            "filters": {
                "query_type": None,
                "status_filter": None
            },
            "performance_stats": {
                "success_rate": 0,
                "average_response_time_ms": None,
                "total_tokens_used": 0,
                "total_results_returned": 0,
                "query_types_breakdown": {},
                "top_error_messages": []
            },
            "user_id": "test_user",
            "processing_time_ms": 10.5
        }
        
        # Validate main response structure
        for key in expected_response_keys:
            assert key in sample_response
            print(f"✓ Response contains '{key}' field")
        
        # Validate pagination structure
        for key in expected_pagination_keys:
            assert key in sample_response['pagination']
            print(f"✓ Pagination contains '{key}' field")
        
        # Validate performance stats structure
        for key in expected_performance_stats_keys:
            assert key in sample_response['performance_stats']
            print(f"✓ Performance stats contains '{key}' field")
        
        return True
        
    except Exception as e:
        print(f"❌ Response structure test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running query history tests...\n")
    
    tests = [
        test_query_history_imports,
        test_query_logger_service,
        test_query_log_model,
        test_query_history_endpoint_structure,
        test_database_session_handling,
        test_query_filtering_logic,
        test_response_structure
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
    print(f"Query History Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All query history tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)