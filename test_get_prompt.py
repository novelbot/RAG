import asyncio
import httpx

async def test():
    async with httpx.AsyncClient() as client:
        # Login
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Fetch the debug prompt
        print("Fetching debug prompt for test-conv-456...")
        debug_resp = await client.get(
            "http://localhost:8000/api/v1/episode/debug/prompt/test-conv-456",
            headers=headers
        )
        
        print(f"Status: {debug_resp.status_code}")
        if debug_resp.status_code == 200:
            data = debug_resp.json()
            print(f"\n✅ Success\! Debug info:")
            print(f"- Timestamp: {data.get('timestamp')}")
            print(f"- Context messages: {data.get('context_messages')}")
            print(f"- Episode sources: {data.get('episode_sources')}")
            print(f"- Prompt length: {len(data.get('prompt', ''))}")
            print(f"\nFirst 500 chars of prompt:")
            print("-"*50)
            print(data.get('prompt', '')[:500])
        else:
            print(f"Error: {debug_resp.text}")

asyncio.run(test())
