"""Test Google Gemini streaming functionality."""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_google_streaming():
    """Test Google provider streaming endpoint."""
    
    # Check if Google API key is available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return
    
    print("✅ Google API key found")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # 1. Login
            print("\n1. Logging in...")
            login_resp = await client.post(
                "http://localhost:8000/api/v1/auth/login",
                json={"username": "admin", "password": "admin123"}
            )
            
            if login_resp.status_code != 200:
                print(f"❌ Login failed: {login_resp.status_code}")
                print(login_resp.text)
                return
            
            token = login_resp.json()["access_token"]
            print("✅ Login successful")
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # 2. Test streaming with Google provider
            print("\n2. Testing Google (Gemini) streaming...")
            stream_data = {
                "message": "안녕하세요! 짧게 인사해주세요.",
                "llm_provider": "google",
                "llm_model": "gemini-1.5-flash",  # Updated model name
                "temperature": 0.7,
                "max_tokens": 500,
                "episode_ids": [],
                "novel_ids": []
            }
            
            print(f"Request data: {json.dumps(stream_data, indent=2, ensure_ascii=False)}")
            
            # Start streaming
            print("\n3. Starting stream...")
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=stream_data,
                headers=headers,
                timeout=60.0
            ) as response:
                print(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"❌ Streaming failed: {error_text.decode()}")
                    return
                
                print("\n4. Receiving stream chunks:")
                print("-" * 40)
                
                full_response = ""
                chunk_count = 0
                
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        chunk_count += 1
                        try:
                            data = json.loads(line[6:])
                            
                            if data.get("type") == "content":
                                content = data.get("content", "")
                                full_response += content
                                print(content, end="", flush=True)
                            
                            elif data.get("type") == "error":
                                print(f"\n❌ Error: {data.get('error')}")
                                return
                            
                            elif data.get("type") == "usage":
                                print(f"\n\nUsage info: {data.get('usage')}")
                            
                        except json.JSONDecodeError as e:
                            print(f"\n⚠️ Failed to parse chunk {chunk_count}: {e}")
                            print(f"Raw line: {line[:100]}...")
                
                print("\n" + "-" * 40)
                print(f"\n✅ Stream completed successfully!")
                print(f"Total chunks received: {chunk_count}")
                print(f"Full response length: {len(full_response)} characters")
                
        except httpx.ConnectError:
            print("❌ Cannot connect to server. Make sure the server is running on port 8000.")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("Google Provider Streaming Test")
    print("=" * 50)
    
    asyncio.run(test_google_streaming())