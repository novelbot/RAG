#!/usr/bin/env python3
"""
Test episode chat API with real embedded data.
"""

import requests
import json
import time

# Server configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "demo_access_token"  # Use demo token for testing

def test_episode_chat():
    """Test episode chat functionality with real data."""
    
    # Headers for authenticated requests
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing Episode Chat API with Real Embedded Data")
    print("=" * 60)
    
    # Test 1: General episode question
    print("\n1️⃣ Test: General episode question")
    test_data = {
        "message": "주인공의 성격에 대해 알려주세요",
        "max_episodes": 5,
        "max_results": 5,
        "use_conversation_context": True,
        "include_episode_metadata": True,
        "response_format": "detailed"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/episode/chat",
            headers=headers,
            json=test_data,
            timeout=30
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['message'][:200]}...")
            print(f"Episodes used: {len(result['episode_sources'])}")
            print(f"Conversation ID: {result['conversation_id']}")
            
            # Store conversation ID for next test
            conversation_id = result['conversation_id']
        else:
            print(f"Error: {response.text}")
            return
            
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n" + "-" * 40)
    
    # Test 2: Specific novel filtering
    print("\n2️⃣ Test: Filter by specific novel")
    test_data = {
        "message": "이 소설의 첫 번째 에피소드는 어떤 내용인가요?",
        "conversation_id": conversation_id,
        "novel_ids": [1],  # Filter to novel 1
        "max_episodes": 3,
        "episode_sort_order": "episode_number",
        "use_conversation_context": True,
        "response_format": "detailed"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/episode/chat",
            headers=headers,
            json=test_data,
            timeout=30
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['message'][:200]}...")
            print(f"Episodes used: {len(result['episode_sources'])}")
            print(f"Novel IDs found: {list(set([ep['novel_id'] for ep in result['episode_sources']]))}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-" * 40)
    
    # Test 3: Episode number-based query
    print("\n3️⃣ Test: Query about specific episodes")
    test_data = {
        "message": "첫 몇 에피소드의 줄거리를 시간순으로 정리해주세요",
        "conversation_id": conversation_id,
        "max_episodes": 8,
        "episode_sort_order": "episode_number",
        "use_conversation_context": True,
        "include_episode_metadata": True,
        "response_format": "detailed"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/episode/chat",
            headers=headers,
            json=test_data,
            timeout=30
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['message'][:300]}...")
            print(f"Episodes used: {len(result['episode_sources'])}")
            
            # Show episode ordering
            episodes = result['episode_sources']
            if episodes:
                print("Episode order:")
                for ep in episodes[:5]:  # Show first 5
                    print(f"  - Novel {ep['novel_id']}, Episode {ep['episode_number']}: {ep['episode_title'][:30]}...")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-" * 40)
    
    # Test 4: Get conversation history
    print("\n4️⃣ Test: Get conversation history")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/episode/conversations/{conversation_id}",
            headers=headers,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Total messages: {result['total_messages']}")
            print(f"Episodes discussed: {result.get('episodes_discussed', [])}")
            print(f"Novels discussed: {result.get('novels_discussed', [])}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Episode Chat API testing completed!")

if __name__ == "__main__":
    test_episode_chat()