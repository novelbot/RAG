import asyncio
import httpx
import json

async def test_streaming():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. 로그인
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # 2. 기본 설정으로 테스트 (환경변수의 OpenAI 사용)
        print("=" * 60)
        print("1. Testing with default config (should use OpenAI from .env)")
        print("=" * 60)
        
        request_data = {
            "message": "안녕하세요\! 1부터 3까지 세어주세요.",
            "episode_ids": [],
            "novel_ids": [],
            "temperature": 0.3,
            "max_tokens": 200
        }
        
        try:
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=request_data,
                headers=headers
            ) as response:
                print(f"Status: {response.status_code}")
                chunk_count = 0
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content":
                                chunk_count += 1
                                if chunk_count <= 3:
                                    print(f"  Chunk {chunk_count}: {data.get('content')[:50]}")
                            elif data.get("type") == "done":
                                print(f"✅ Streaming completed - {chunk_count} chunks received")
                            elif data.get("type") == "error":
                                print(f"❌ Error: {data.get('error')}")
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # 3. OpenAI provider 명시적 지정 테스트
        print("\n" + "=" * 60)
        print("2. Testing with explicit OpenAI provider override")
        print("=" * 60)
        
        request_data_override = {
            "message": "Hello from OpenAI\! Count from 1 to 3.",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "episode_ids": [],
            "novel_ids": [],
            "temperature": 0.3,
            "max_tokens": 200
        }
        
        try:
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=request_data_override,
                headers=headers
            ) as response:
                print(f"Status: {response.status_code}")
                chunk_count = 0
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content":
                                chunk_count += 1
                                if chunk_count <= 3:
                                    print(f"  Chunk {chunk_count}: {data.get('content')[:50]}")
                            elif data.get("type") == "done":
                                print(f"✅ Streaming completed - {chunk_count} chunks received")
                            elif data.get("type") == "error":
                                print(f"❌ Error: {data.get('error')}")
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming())
