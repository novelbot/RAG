import asyncio
import httpx
import json
import traceback

async def test_openai():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Login
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        
        # Test with provider override
        request_data = {
            "message": "Hello from OpenAI test",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "episode_ids": [],
            "novel_ids": []
        }
        
        print("Testing OpenAI provider override...")
        try:
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=request_data,
                headers={"Authorization": f"Bearer {token}"}
            ) as response:
                print(f"Status: {response.status_code}")
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "error":
                            print(f"Error: {data.get('error')}")
                            break
                        elif data.get("type") == "content":
                            print(f"Content chunk received")
                            return True
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
            return False

asyncio.run(test_openai())
