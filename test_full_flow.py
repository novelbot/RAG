import asyncio
import httpx
import json

async def test():
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
        
        # Send streaming request
        stream_data = {
            "message": "테스트 메시지입니다",
            "conversation_id": "test-conv-456",
            "use_conversation_context": False,
            "episode_ids": [],
            "novel_ids": []
        }
        
        print("1. Sending streaming request...")
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=stream_data,
            headers=headers,
            timeout=15.0
        ) as response:
            print(f"   Status: {response.status_code}")
            # Read some chunks
            count = 0
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    count += 1
                    if count >= 10:
                        break
        
        print(f"   Received {count} chunks")
        
        # Now fetch the debug prompt
        print("\n2. Fetching debug prompt...")
        debug_resp = await client.get(
            "http://localhost:8000/api/v1/episode/debug/prompt/test-conv-456",
            headers=headers
        )
        
        print(f"   Status: {debug_resp.status_code}")
        if debug_resp.status_code == 200:
            data = debug_resp.json()
            print(f"\n   ✅ Success\! Debug info:")
            print(f"   - Timestamp: {data.get('timestamp')}")
            print(f"   - Context messages: {data.get('context_messages')}")
            print(f"   - Episode sources: {data.get('episode_sources')}")
            print(f"   - Prompt length: {len(data.get('prompt', ''))}")
            print(f"\n   First 500 chars of prompt:")
            print("   " + "-"*50)
            print(data.get('prompt', '')[:500])
        else:
            print(f"   Error: {debug_resp.text}")

if __name__ == "__main__":
    asyncio.run(test())
