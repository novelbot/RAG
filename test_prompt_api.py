import asyncio
import httpx
import json

async def test_prompt_api():
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
        
        # First, send a streaming request to create a prompt
        stream_data = {
            "message": "테스트 메시지입니다",
            "conversation_id": "test-conv-123",
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
            timeout=10.0
        ) as response:
            print(f"   Response status: {response.status_code}")
            # Just read first chunk
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    break
        
        # Now try to fetch the debug prompt
        print("\n2. Fetching debug prompt...")
        debug_resp = await client.get(
            f"http://localhost:8000/api/v1/episode/debug/prompt/test-conv-123",
            headers=headers
        )
        
        print(f"   Debug API status: {debug_resp.status_code}")
        if debug_resp.status_code == 200:
            data = debug_resp.json()
            print(f"   Success\! Prompt length: {len(data.get('prompt', ''))}")
        else:
            print(f"   Error response: {debug_resp.text}")

if __name__ == "__main__":
    asyncio.run(test_prompt_api())
