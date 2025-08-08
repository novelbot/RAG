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
        
        # Test streaming with conversation context
        stream_data = {
            "message": "처음에 비누 냄새 나던 사람이 현규였나요?",
            "conversation_id": "58999a96-6567-4fa5-9434-2b945c098696",
            "use_conversation_context": True,
            "episode_ids": [],
            "novel_ids": []
        }
        
        print("Sending request with conversation_id:", stream_data["conversation_id"])
        print("use_conversation_context:", stream_data["use_conversation_context"])
        
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=stream_data,
            headers=headers,
            timeout=10.0
        ) as response:
            print(f"Response status: {response.status_code}")
            # Just read first few chunks
            count = 0
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    count += 1
                    if count <= 3:
                        data = json.loads(line[6:])
                        if data.get("type") == "conversation_info":
                            print(f"Conversation info: {data}")
                    if count >= 5:
                        break

if __name__ == "__main__":
    asyncio.run(test())
