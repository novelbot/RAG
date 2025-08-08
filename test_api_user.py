import asyncio
import httpx
import json

async def test_user_id():
    async with httpx.AsyncClient() as client:
        # Login
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        print(f"Token: {token[:20]}...")
        
        # Get current user info
        headers = {"Authorization": f"Bearer {token}"}
        user_resp = await client.get(
            "http://localhost:8000/api/v1/auth/me",
            headers=headers
        )
        
        user_data = user_resp.json()
        print(f"\nCurrent user data: {user_data}")
        print(f"User ID from API: {user_data.get('id')}")
        print(f"User ID type: {type(user_data.get('id'))}")

if __name__ == "__main__":
    asyncio.run(test_user_id())
