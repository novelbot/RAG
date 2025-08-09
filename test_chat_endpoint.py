#!/usr/bin/env python3
"""Test script to compare /chat and /chat/stream endpoints."""

import asyncio
import httpx
import json
from typing import Dict, Any

async def test_chat_endpoints():
    """Test both chat and chat/stream endpoints."""
    
    # Login first
    async with httpx.AsyncClient() as client:
        # Login
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Test data
        test_data = {
            "message": "테스트 메시지입니다. 작품의 주요 등장인물에 대해 알려주세요.",
            "conversation_id": "test-conv-123",
            "use_conversation_context": False,
            "episode_ids": [],
            "novel_ids": [],
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini"
        }
        
        print("=" * 80)
        print("TESTING NON-STREAMING ENDPOINT: /api/v1/episode/chat")
        print("=" * 80)
        
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/episode/chat",
                json=test_data,
                headers=headers,
                timeout=30.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Response message: {result.get('message', '')[:200]}...")
                print(f"Conversation ID: {result.get('conversation_id')}")
                print(f"Response time: {result.get('response_time_ms')} ms")
            else:
                print(f"Error Response: {response.text}")
                
        except Exception as e:
            print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        
        print("\n" + "=" * 80)
        print("TESTING STREAMING ENDPOINT: /api/v1/episode/chat/stream")
        print("=" * 80)
        
        try:
            # For streaming, we'll just check if it starts properly
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=test_data,
                headers=headers,
                timeout=30.0
            ) as response:
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    print("Success! First 3 chunks:")
                    count = 0
                    async for line in response.aiter_lines():
                        if line and line.startswith("data: "):
                            count += 1
                            if count <= 3:
                                data = line[6:]  # Remove "data: " prefix
                                if data != "[DONE]":
                                    chunk = json.loads(data)
                                    print(f"  Chunk {count}: type={chunk.get('type')}, content_length={len(chunk.get('content', ''))}")
                            else:
                                break
                else:
                    content = await response.aread()
                    print(f"Error Response: {content.decode()}")
                    
        except Exception as e:
            print(f"Exception occurred: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_chat_endpoints())