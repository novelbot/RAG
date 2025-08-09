#!/usr/bin/env python3
"""
Test script for existing conversation API endpoints.
"""

import asyncio
import httpx
import json

API_BASE = "http://localhost:8000/api/v1"
EXISTING_CONVERSATION_ID = "58999a96-6567-4fa5-9434-2b945c098696"


async def login(client: httpx.AsyncClient) -> str:
    """Login and get access token."""
    response = await client.post(
        f"{API_BASE}/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    return response.json()["access_token"]


async def test_get_conversation():
    """Test getting existing conversation history."""
    print("\n" + "="*80)
    print(f"Testing GET /api/v1/episode/conversation/{EXISTING_CONVERSATION_ID}")
    print("="*80)
    
    async with httpx.AsyncClient() as client:
        token = await login(client)
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test getting conversation history
        response = await client.get(
            f"{API_BASE}/episode/conversation/{EXISTING_CONVERSATION_ID}",
            headers=headers,
            params={"limit": 5}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Successfully retrieved conversation!")
            print(f"Conversation ID: {data.get('conversation_id')}")
            print(f"User ID: {data.get('user_id')}")
            print(f"Created At: {data.get('created_at')}")
            print(f"Message Count: {data.get('message_count')}")
            
            if data.get('messages'):
                print("\nFirst few messages:")
                for i, msg in enumerate(data['messages'][:3], 1):
                    print(f"\n  Message {i}:")
                    print(f"    Role: {msg['role']}")
                    content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"    Content: {content_preview}")
        else:
            print(f"❌ Error: {response.text}")


async def test_generate_title():
    """Test generating title for existing conversation."""
    print("\n" + "="*80)
    print(f"Testing POST /api/v1/episode/conversation/{EXISTING_CONVERSATION_ID}/generate-title")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        token = await login(client)
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test generating title
        response = await client.post(
            f"{API_BASE}/episode/conversation/{EXISTING_CONVERSATION_ID}/generate-title",
            headers=headers
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Successfully generated title!")
            print(f"Conversation ID: {data.get('conversation_id')}")
            print(f"Generated Title: '{data.get('generated_title')}'")
        else:
            print(f"❌ Error: {response.text}")


async def main():
    """Main test function."""
    try:
        print(f"Testing with existing conversation: {EXISTING_CONVERSATION_ID}")
        
        # Test getting conversation history
        await test_get_conversation()
        
        # Test generating title
        await test_generate_title()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Conversation API Tests with Existing Conversation")
    print("Make sure the server is running: uv run rag-cli serve")
    asyncio.run(main())