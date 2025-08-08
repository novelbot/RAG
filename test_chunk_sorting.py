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
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test with specific novel/episode IDs to see chunk ordering
        stream_data = {
            "message": "테스트 메시지",
            "novel_ids": [88],  # Specific novel with chunks
            "episode_ids": [],
            "max_episodes": 15
        }
        
        print("Testing chunk ordering with novel_id=88...")
        
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=stream_data,
            headers=headers,
            timeout=10.0
        ) as response:
            print(f"Response status: {response.status_code}")
            chunks_received = 0
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks_received += 1
                    if chunks_received >= 3:
                        break
            print(f"Received {chunks_received} chunks")

if __name__ == "__main__":
    asyncio.run(test())
