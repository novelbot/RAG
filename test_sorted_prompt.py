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
        
        # Test streaming with a query that should return multiple episodes
        stream_data = {
            "message": "캐릭터들의 대화 내용을 알려줘",
            "episode_ids": [],
            "novel_ids": [],
            "max_episodes": 10  # Request more episodes to see sorting
        }
        
        print("Sending request to get episodes sorted by episode number...")
        
        # First, do a regular call to capture the prompt
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=stream_data,
            headers=headers,
            timeout=30.0
        ) as response:
            print(f"Response status: {response.status_code}")
            # Just consume the stream to let it save the debug prompt
            chunks_received = 0
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks_received += 1
                    if chunks_received >= 5:  # Just get a few chunks
                        break
        
        # Now fetch the debug prompt to see the sorting
        print("\nFetching debug prompt to check episode ordering...")
        
        # We need to get the conversation ID from somewhere
        # For testing, let's just check the server logs
        
if __name__ == "__main__":
    asyncio.run(test())
