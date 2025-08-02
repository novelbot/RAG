#!/usr/bin/env python3
"""
Integration test for query history API endpoint
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_query_history_api_integration():
    """Integration test for query history API endpoint"""
    try:
        from src.api.routes.query import get_query_history
        from src.auth.dependencies import MockUser
        
        # Create a mock user for testing
        mock_user = MockUser(
            id="test_user_123", 
            username="test_user", 
            email="test@example.com", 
            roles=["user"]
        )
        
        # Test basic functionality with fallback to mock data
        result = await get_query_history(
            limit=10,
            offset=0,
            current_user=mock_user
        )
        
        # Verify response structure
        assert isinstance(result, dict)
        assert "history" in result
        assert "pagination" in result
        assert "filters" in result
        assert "performance_stats" in result
        assert "user_id" in result
        assert "processing_time_ms" in result
        
        print("✓ Basic query history API call successful")
        
        # Test with filtering parameters
        result_filtered = await get_query_history(
            limit=5,
            offset=0,
            query_type="search",
            status_filter="success",
            current_user=mock_user
        )
        
        assert isinstance(result_filtered, dict)
        assert result_filtered["filters"]["query_type"] == "search"
        assert result_filtered["filters"]["status_filter"] == "success"
        
        print("✓ Query history API with filters successful")
        
        # Test pagination
        result_paginated = await get_query_history(
            limit=3,
            offset=5,
            current_user=mock_user
        )
        
        assert result_paginated["pagination"]["limit"] == 3
        assert result_paginated["pagination"]["offset"] == 5
        
        print("✓ Query history API pagination successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Query history API integration test failed: {e}")
        return False

async def test_query_history_error_handling():
    """Test error handling in query history API"""
    try:
        from src.api.routes.query import get_query_history
        from src.auth.dependencies import MockUser
        from fastapi import HTTPException
        
        mock_user = MockUser(
            id="test_user_456", 
            username="test_user", 
            email="test2@example.com", 
            roles=["user"]
        )
        
        # Test invalid limit (too high)
        try:
            await get_query_history(
                limit=150,  # Above max of 100
                offset=0,
                current_user=mock_user
            )
            assert False, "Should have raised HTTPException for invalid limit"
        except HTTPException as e:
            assert e.status_code == 400
            print("✓ Invalid limit properly rejected")
        
        # Test invalid query_type
        try:
            await get_query_history(
                limit=10,
                offset=0,
                query_type="invalid_type",
                current_user=mock_user
            )
            assert False, "Should have raised HTTPException for invalid query_type"
        except HTTPException as e:
            assert e.status_code == 400
            print("✓ Invalid query_type properly rejected")
        
        # Test invalid status_filter
        try:
            await get_query_history(
                limit=10,
                offset=0,
                status_filter="invalid_status",
                current_user=mock_user
            )
            assert False, "Should have raised HTTPException for invalid status_filter"
        except HTTPException as e:
            assert e.status_code == 400
            print("✓ Invalid status_filter properly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Query history error handling test failed: {e}")
        return False

async def test_performance_stats_calculation():
    """Test performance statistics calculation logic"""
    try:
        # Test the logic that would be used in performance stats calculation
        from collections import Counter
        
        # Simulate query logs for testing
        mock_query_logs = [
            {"status": "success", "response_time_ms": 100, "total_tokens": 50, "results_count": 5},
            {"status": "success", "response_time_ms": 200, "total_tokens": 75, "results_count": 3},
            {"status": "failed", "response_time_ms": 50, "total_tokens": None, "results_count": 0},
            {"status": "success", "response_time_ms": 150, "total_tokens": 60, "results_count": 4},
        ]
        
        # Calculate statistics
        successful_queries = [log for log in mock_query_logs if log["status"] == "success"]
        failed_queries = [log for log in mock_query_logs if log["status"] == "failed"]
        
        success_rate = len(successful_queries) / len(mock_query_logs)
        assert success_rate == 0.75
        
        response_times = [log["response_time_ms"] for log in mock_query_logs if log["response_time_ms"]]
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time == 125.0
        
        total_tokens_used = sum(log["total_tokens"] for log in mock_query_logs if log["total_tokens"])
        assert total_tokens_used == 185
        
        total_results_returned = sum(log["results_count"] for log in mock_query_logs if log["results_count"])
        assert total_results_returned == 12
        
        print("✓ Performance stats calculation logic works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance stats calculation test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("Running query history integration tests...\n")
    
    tests = [
        test_query_history_api_integration,
        test_query_history_error_handling,
        test_performance_stats_calculation
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
    print(f"Query History Integration Test Results:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All query history integration tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)