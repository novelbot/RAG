import asyncio
import httpx
import json

async def test_conversation():
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
        
        # Test streaming with conversation context
        print("\n" + "="*80)
        print("TESTING STREAMING ENDPOINT WITH CONVERSATION CONTEXT")
        print("="*80)
        
        stream_data = {
            "message": "처음에 비누 냄새 나던 사람이 현규였나요?",
            "conversation_id": "58999a96-6567-4fa5-9434-2b945c098696",
            "use_conversation_context": True,
            "episode_ids": [],
            "novel_ids": []
        }
        
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=stream_data,
            headers=headers,
            timeout=60.0
        ) as response:
            print(f"Response status: {response.status_code}")
            print("\nFirst 10 chunks:")
            count = 0
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    count += 1
                    if count <= 10:
                        print(f"Chunk {count}: {line[:100]}...")
                    if count == 10:
                        break
        
        print("\nTest completed\!")

if __name__ == "__main__":
    asyncio.run(test_conversation())
