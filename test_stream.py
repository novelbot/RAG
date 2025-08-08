import asyncio
import httpx
import json
import time

async def test_streaming():
    async with httpx.AsyncClient() as client:
        # 1. Login
        print("1. Logging in...")
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        print(f"✅ Login successful, token obtained")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # 2. Test streaming endpoint
        print("\n2. Testing streaming endpoint with OpenAI...")
        stream_data = {
            "message": "안녕하세요\! 간단한 스트리밍 테스트입니다. 1부터 5까지 세어주세요.",
            "episode_ids": [],
            "novel_ids": []
        }
        
        start_time = time.time()
        chunk_count = 0
        full_response = ""
        
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=stream_data,
            headers=headers,
            timeout=30.0
        ) as response:
            print(f"Response status: {response.status_code}")
            print("\n--- Streaming Response ---")
            
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content":
                            chunk_count += 1
                            content = data.get("content", "")
                            full_response += content
                            print(content, end="", flush=True)
                        elif data.get("type") == "done":
                            print("\n\n✅ Stream completed")
                            print(f"Total chunks: {chunk_count}")
                            print(f"Time taken: {time.time() - start_time:.2f}s")
                    except json.JSONDecodeError:
                        continue
        
        print("\n--- Full Response ---")
        print(full_response)
        
        # 3. Test non-streaming endpoint for comparison
        print("\n3. Testing non-streaming endpoint...")
        chat_data = {
            "message": "1부터 3까지 세어주세요.",
            "episode_ids": [],
            "novel_ids": []
        }
        
        start_time = time.time()
        chat_resp = await client.post(
            "http://localhost:8000/api/v1/episode/chat",
            json=chat_data,
            headers=headers,
            timeout=30.0
        )
        
        if chat_resp.status_code == 200:
            result = chat_resp.json()
            print(f"✅ Non-streaming response received")
            print(f"Time taken: {time.time() - start_time:.2f}s")
            print(f"Response: {result.get('response', '')[:200]}...")
        else:
            print(f"❌ Non-streaming failed: {chat_resp.status_code}")

if __name__ == "__main__":
    print("=== OpenAI Streaming Test ===\n")
    asyncio.run(test_streaming())
