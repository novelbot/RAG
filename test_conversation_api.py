#!/usr/bin/env python3
"""
Test script for conversation API endpoints.
"""

import asyncio
import httpx
import json
from typing import Dict, Any

API_BASE = "http://localhost:8000/api/v1"


async def login(client: httpx.AsyncClient) -> str:
    """Login and get access token."""
    response = await client.post(
        f"{API_BASE}/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    return response.json()["access_token"]


async def test_conversation_history(token: str, conversation_id: str):
    """Test getting conversation history."""
    print("\n" + "="*80)
    print("Testing GET /api/v1/conversation/{conversation_id}")
    print("="*80)
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test getting conversation history
        response = await client.get(
            f"{API_BASE}/episode/conversation/{conversation_id}",
            headers=headers,
            params={"limit": 10}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Conversation ID: {data.get('conversation_id')}")
            print(f"User ID: {data.get('user_id')}")
            print(f"Created At: {data.get('created_at')}")
            print(f"Updated At: {data.get('updated_at')}")
            print(f"Message Count: {data.get('message_count')}")
            
            if data.get('messages'):
                print("\nMessages:")
                for i, msg in enumerate(data['messages'][:3], 1):
                    print(f"\n  Message {i}:")
                    print(f"    Role: {msg['role']}")
                    print(f"    Content: {msg['content'][:100]}...")
                    print(f"    Timestamp: {msg['timestamp']}")
                    
                if len(data['messages']) > 3:
                    print(f"\n  ... and {len(data['messages']) - 3} more messages")
            else:
                print("\nNo messages in conversation")
                
            return True
        else:
            print(f"Error: {response.text}")
            return False


async def test_generate_title(token: str, conversation_id: str):
    """Test generating conversation title."""
    print("\n" + "="*80)
    print("Testing POST /api/v1/conversation/{conversation_id}/generate-title")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test generating title
        response = await client.post(
            f"{API_BASE}/episode/conversation/{conversation_id}/generate-title",
            headers=headers
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Conversation ID: {data.get('conversation_id')}")
            print(f"Generated Title: {data.get('generated_title')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False


async def create_test_conversation(token: str) -> str:
    """Create a test conversation and return its ID."""
    print("\n" + "="*80)
    print("Creating test conversation")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Send a chat message to create a conversation
        response = await client.post(
            f"{API_BASE}/episode/chat",
            headers=headers,
            json={
                "message": "테스트 대화입니다. 현규와 민수가 나오는 에피소드에 대해 알려주세요.",
                "episode_ids": [],
                "novel_ids": [],
                "use_conversation_context": True
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            conversation_id = data.get('conversation_id')
            print(f"Created conversation: {conversation_id}")
            
            # Send another message to have some conversation history
            await client.post(
                f"{API_BASE}/episode/chat",
                headers=headers,
                json={
                    "message": "현규의 성격은 어떤가요?",
                    "conversation_id": conversation_id,
                    "use_conversation_context": True,
                    "episode_ids": [],
                    "novel_ids": []
                }
            )
            
            return conversation_id
        else:
            print(f"Failed to create conversation: {response.text}")
            return None


async def main():
    """Main test function."""
    try:
        # Login
        async with httpx.AsyncClient() as client:
            token = await login(client)
            print(f"Login successful. Token obtained.")
        
        # Create a test conversation
        conversation_id = await create_test_conversation(token)
        
        if conversation_id:
            # Wait a bit for conversation to be saved
            await asyncio.sleep(2)
            
            # Test getting conversation history
            history_success = await test_conversation_history(token, conversation_id)
            
            if history_success:
                # Test generating title
                await test_generate_title(token, conversation_id)
            
            # Test with non-existent conversation
            print("\n" + "="*80)
            print("Testing with non-existent conversation ID")
            print("="*80)
            await test_conversation_history(token, "non-existent-id-12345")
        else:
            print("Failed to create test conversation")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Conversation API Tests")
    print("Make sure the server is running: uv run rag-cli serve")
    asyncio.run(main())