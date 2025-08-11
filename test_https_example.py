"""
Example test file demonstrating HTTP/HTTPS configuration support.
This test will work with both HTTP and HTTPS based on .env settings.
"""

import asyncio
import httpx
from src.utils.test_config import (
    get_server_url,
    get_auth_url,
    get_chat_url,
    get_stream_url,
    create_test_client,
    get_client_kwargs
)


async def test_health_check():
    """Test the health check endpoint."""
    async with create_test_client() as client:
        response = await client.get("/health")
        print(f"Health check response: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        return response.status_code == 200


async def test_authentication():
    """Test authentication endpoint."""
    async with create_test_client() as client:
        # Login
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        response = await client.post("/api/v1/auth/login", json=login_data)
        print(f"Login response: {response.status_code}")
        
        if response.status_code == 200:
            token = response.json().get("access_token")
            print(f"Successfully authenticated, token received: {token[:20]}...")
            return token
        else:
            print(f"Authentication failed: {response.text}")
            return None


async def test_chat_endpoint(token: str):
    """Test the chat endpoint with authentication."""
    async with create_test_client() as client:
        headers = {"Authorization": f"Bearer {token}"}
        
        chat_data = {
            "message": "HTTPS 테스트 메시지입니다.",
            "episode_ids": [],
            "novel_ids": []
        }
        
        response = await client.post(
            "/api/v1/episode/chat",
            json=chat_data,
            headers=headers
        )
        
        print(f"Chat response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Chat response message: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"Chat failed: {response.text}")
            return False


async def main():
    """Run all tests."""
    server_url = get_server_url()
    print(f"Testing server at: {server_url}")
    print("-" * 50)
    
    # Test health check
    print("1. Testing health check...")
    health_ok = await test_health_check()
    if not health_ok:
        print("Health check failed, server might not be running")
        return
    
    print("-" * 50)
    
    # Test authentication
    print("2. Testing authentication...")
    token = await test_authentication()
    if not token:
        print("Authentication failed")
        return
    
    print("-" * 50)
    
    # Test chat endpoint
    print("3. Testing chat endpoint...")
    chat_ok = await test_chat_endpoint(token)
    
    print("-" * 50)
    
    if health_ok and token and chat_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    print("RAG Server HTTP/HTTPS Test")
    print("=" * 50)
    asyncio.run(main())