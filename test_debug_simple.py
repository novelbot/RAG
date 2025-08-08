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
        
        # Try to fetch debug prompt for non-existent conversation
        print("Testing debug endpoint...")
        debug_resp = await client.get(
            "http://localhost:8000/api/v1/episode/debug/prompt/test-123",
            headers=headers
        )
        
        print(f"Status: {debug_resp.status_code}")
        print(f"Response: {debug_resp.text}")

asyncio.run(test())
