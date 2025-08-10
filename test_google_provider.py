#!/usr/bin/env python3
"""Test script to test Google/Gemini provider with chat endpoints."""

import asyncio
import httpx
import json
from typing import Dict, Any

async def test_google_provider():
    """Test both chat and chat/stream endpoints with Google provider."""
    
    # Login first
    async with httpx.AsyncClient(timeout=60.0) as client:
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
        
        # Test data - note that provider is already set to Google in .env
        test_data = {
            "message": "테스트 메시지입니다. 작품의 주요 등장인물 한 명만 간단히 소개해주세요.",
            "conversation_id": "google-test-123",
            "use_conversation_context": False,
            "episode_ids": [],
            "novel_ids": [],
            "max_episodes": 3,
            "temperature": 0.7
        }
        
        print("=" * 80)
        print("TESTING WITH GOOGLE PROVIDER (from .env configuration)")
        print("Provider: Google, Model: gemini-2.0-flash")
        print("=" * 80)
        
        print("\n1. TESTING NON-STREAMING ENDPOINT: /api/v1/episode/chat")
        print("-" * 80)
        
        try:
            import time
            start_time = time.time()
            
            response = await client.post(
                "http://localhost:8000/api/v1/episode/chat",
                json=test_data,
                headers=headers
            )
            
            elapsed_time = (time.time() - start_time) * 1000
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"Response time: {elapsed_time:.2f} ms")
                print(f"Message preview: {result.get('message', '')[:200]}...")
                print(f"Conversation ID: {result.get('conversation_id')}")
            else:
                print(f"❌ Error Response: {response.text[:500]}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {type(e).__name__}: {str(e)}")
        
        print("\n2. TESTING STREAMING ENDPOINT: /api/v1/episode/chat/stream")
        print("-" * 80)
        
        try:
            import time
            start_time = time.time()
            chunk_count = 0
            content_chunks = []
            
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=test_data,
                headers=headers
            ) as response:
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ Success! Streaming response:")
                    
                    async for line in response.aiter_lines():
                        if line and line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data != "[DONE]":
                                try:
                                    chunk = json.loads(data)
                                    chunk_count += 1
                                    
                                    if chunk.get('type') == 'content':
                                        content_chunks.append(chunk.get('content', ''))
                                    
                                    # Print first few chunks info
                                    if chunk_count <= 3:
                                        print(f"  Chunk {chunk_count}: type={chunk.get('type')}, content_length={len(chunk.get('content', ''))}")
                                    
                                except json.JSONDecodeError:
                                    pass
                    
                    elapsed_time = (time.time() - start_time) * 1000
                    full_content = ''.join(content_chunks)
                    
                    print(f"\nTotal chunks received: {chunk_count}")
                    print(f"Response time: {elapsed_time:.2f} ms")
                    print(f"Full response preview: {full_content[:200]}...")
                    
                else:
                    content = await response.aread()
                    print(f"❌ Error Response: {content.decode()[:500]}")
                    
        except Exception as e:
            print(f"❌ Exception occurred: {type(e).__name__}: {str(e)}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_google_provider())