import asyncio
import httpx
import json
import time

async def test_with_server():
    import subprocess
    
    # Start server
    server = subprocess.Popen(
        ["uv", "run", "rag-cli", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    await asyncio.sleep(5)
    
    try:
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
            
            # Send streaming request
            stream_data = {
                "message": "테스트 메시지입니다",
                "conversation_id": "test-conv-123",
                "use_conversation_context": False,
                "episode_ids": [],
                "novel_ids": []
            }
            
            print("Sending streaming request...")
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=stream_data,
                headers=headers,
                timeout=10.0
            ) as response:
                # Read some chunks
                count = 0
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        count += 1
                        if count >= 5:
                            break
            
            # Try to fetch debug prompt
            print("\nFetching debug prompt...")
            debug_resp = await client.get(
                f"http://localhost:8000/api/v1/episode/debug/prompt/test-conv-123",
                headers=headers
            )
            
            print(f"Debug API status: {debug_resp.status_code}")
            if debug_resp.status_code \!= 200:
                print(f"Error: {debug_resp.text}")
    
    finally:
        # Kill server
        server.terminate()
        await asyncio.sleep(1)
        
        # Check server output for errors
        print("\n=== Server Output (last 50 lines) ===")
        output = []
        for line in server.stdout:
            output.append(line)
            if len(output) > 50:
                break
        print(''.join(output[-50:]))

if __name__ == "__main__":
    asyncio.run(test_with_server())
