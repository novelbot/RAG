#!/usr/bin/env python3
"""
Test script for Episode RAG API endpoints.
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.app import create_app
import uvicorn

class EpisodeAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_episode_health(self):
        """Test episode health endpoint."""
        print("\n=== Testing Episode Health Endpoint ===")
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/episode/health")
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Response: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"✗ Failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    async def test_episode_search(self):
        """Test episode search endpoint."""
        print("\n=== Testing Episode Search Endpoint ===")
        
        # Test payload
        payload = {
            "query": "주인공이 어떤 도전에 직면했나요?",
            "episode_ids": [1, 2, 5, 10],
            "limit": 5,
            "sort_order": "episode_number",
            "include_content": True,
            "include_metadata": True
        }
        
        try:
            print(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/episode/search",
                json=payload
            )
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Found {data['total_count']} results in {data['search_time']:.3f}s")
                print(f"✓ Sort order: {data['sort_order']}")
                
                for i, hit in enumerate(data['hits'][:2]):  # Show first 2 results
                    print(f"  Result {i+1}:")
                    print(f"    Episode {hit['episode_number']}: {hit['episode_title']}")
                    print(f"    Similarity: {hit['similarity_score']:.3f}")
                    if hit.get('content'):
                        print(f"    Content preview: {hit['content'][:100]}...")
                
                return True
            else:
                print(f"✗ Failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    async def test_episode_context(self):
        """Test episode context endpoint."""
        print("\n=== Testing Episode Context Endpoint ===")
        
        payload = {
            "episode_ids": [1, 2, 3],
            "query": "character development",
            "max_context_length": 5000
        }
        
        try:
            print(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/episode/context",
                json=payload
            )
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Context includes {data['episodes_included']} episodes")
                print(f"✓ Total length: {data['total_length']} characters")
                print(f"✓ Episode order: {data['episode_order']}")
                print(f"✓ Truncated: {data['truncated']}")
                print(f"✓ Context preview: {data['context'][:200]}...")
                
                return True
            else:
                print(f"✗ Failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    async def test_episode_ask(self):
        """Test episode ask (RAG) endpoint."""
        print("\n=== Testing Episode Ask (RAG) Endpoint ===")
        
        payload = {
            "query": "주인공의 성격이 어떻게 변화했나요?",
            "episode_ids": [1, 5, 10, 15],
            "max_context_episodes": 4,
            "max_context_length": 8000,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            print(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/episode/ask",
                json=payload
            )
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Question: {data['question']}")
                print(f"✓ Answer preview: {data['answer'][:300]}...")
                print(f"✓ Context episodes used: {len(data['context_episodes'])}")
                
                processing_time = data['metadata'].get('processing_time_ms', 0)
                print(f"✓ Processing time: {processing_time}ms")
                
                return True
            else:
                print(f"✗ Failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    async def test_novel_episodes_list(self):
        """Test novel episodes listing endpoint."""
        print("\n=== Testing Novel Episodes List Endpoint ===")
        
        novel_id = 1
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/episode/novel/{novel_id}/episodes",
                params={"limit": 10, "offset": 0}
            )
            
            print(f"✓ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Novel {data['novel_id']} has {data['total_episodes']} episodes")
                print(f"✓ Showing {len(data['episodes'])} episodes:")
                
                for episode in data['episodes'][:3]:  # Show first 3
                    print(f"  Episode {episode['episode_number']}: {episode['episode_title']}")
                    print(f"    Length: {episode['content_length']} chars, Has embedding: {episode['has_embedding']}")
                
                return True
            else:
                print(f"✗ Failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting Episode API Tests")
        print("=" * 60)
        
        tests = [
            ("Episode Health", self.test_episode_health()),
            ("Episode Search", self.test_episode_search()),
            ("Episode Context", self.test_episode_context()),
            ("Episode Ask (RAG)", self.test_episode_ask()),
            ("Novel Episodes List", self.test_novel_episodes_list())
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                result = await test_coro
                results.append((test_name, result))
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 API TEST SUMMARY")
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
        
        return failed == 0
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

async def run_api_tests():
    """Run the API tests."""
    tester = EpisodeAPITester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n✅ All API tests passed! Episode endpoints are working correctly.")
        else:
            print("\n❌ Some API tests failed. Please check the issues above.")
        
        return success
        
    finally:
        await tester.close()

def start_test_server():
    """Start the test server."""
    print("🚀 Starting test server...")
    app = create_app()
    
    # Start server in background for testing
    import threading
    import time
    
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a bit for server to start
    time.sleep(3)
    print("✅ Test server started at http://127.0.0.1:8000")
    
    return server_thread

if __name__ == "__main__":
    print("Episode RAG API Test Suite")
    print("Testing the episode-based RAG API endpoints")
    print()
    
    # Start server
    server_thread = start_test_server()
    
    # Run tests
    try:
        success = asyncio.run(run_api_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)