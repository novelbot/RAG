#!/usr/bin/env python3
"""
Test episode search without LLM to isolate the issue.
"""

import requests
import json

# Server configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "demo_access_token"

def test_episode_search():
    """Test episode vector search without LLM."""
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("🔍 Testing Episode Vector Search Only")
    print("=" * 50)
    
    # Test episode search endpoint
    test_data = {
        "query": "주인공",
        "limit": 5,
        "include_content": True,
        "sort_order": "episode_number"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/episode/search",
            headers=headers,
            json=test_data,
            timeout=30
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search successful!")
            print(f"Results found: {len(result.get('results', []))}")
            
            # Show first result
            if result.get('results'):
                first_result = result['results'][0]
                print(f"First result:")
                print(f"  Episode ID: {first_result.get('episode_id')}")
                print(f"  Episode Number: {first_result.get('episode_number')}")
                print(f"  Title: {first_result.get('episode_title', '')[:50]}...")
                print(f"  Score: {first_result.get('score', 0):.3f}")
                print(f"  Content: {first_result.get('content', '')[:100]}...")
        else:
            print(f"❌ Search failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_episode_search()