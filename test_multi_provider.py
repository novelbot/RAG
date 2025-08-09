import asyncio
import httpx
import json
import time

async def test_provider(provider_name, model_name, api_key=None):
    """Test a specific LLM provider"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Login first
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Prepare request
        request_data = {
            "message": f"안녕하세요\! {provider_name}를 사용중입니다. 1부터 3까지 세어주세요.",
            "llm_provider": provider_name.lower(),
            "llm_model": model_name,
            "temperature": 0.3,
            "max_tokens": 200,
            "max_episodes": 1,
            "episode_ids": [],
            "novel_ids": []
        }
        
        if api_key:
            request_data["llm_api_key"] = api_key
        
        print(f"\n{'='*60}")
        print(f"Testing {provider_name} with model {model_name}")
        print(f"{'='*60}")
        
        try:
            # Test streaming
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=request_data,
                headers=headers
            ) as response:
                if response.status_code == 200:
                    print(f"✅ {provider_name} streaming started")
                    chunk_count = 0
                    full_response = ""
                    
                    async for line in response.aiter_lines():
                        if line and line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content":
                                    chunk_count += 1
                                    content = data.get("content", "")
                                    full_response += content
                                    if chunk_count <= 5:  # Print first 5 chunks
                                        print(f"  Chunk {chunk_count}: {repr(content)}")
                                elif data.get("type") == "done":
                                    print(f"✅ Streaming completed - {chunk_count} chunks received")
                                    if full_response:
                                        print(f"Response preview: {full_response[:100]}...")
                                elif data.get("type") == "error":
                                    print(f"❌ Error: {data.get('error')}")
                                    return False
                            except json.JSONDecodeError:
                                continue
                    
                    return True
                else:
                    print(f"❌ HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False

async def main():
    print("🔍 Multi-Provider LLM Streaming Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # Default config test
        ("default", None, None),
        
        # OpenAI test
        ("openai", "gpt-4o-mini", None),
        
        # Ollama test (local)
        ("ollama", "llama3.2", None),
    ]
    
    results = []
    for provider, model, api_key in test_configs:
        if provider == "default":
            # Test with default configuration
            async with httpx.AsyncClient(timeout=30.0) as client:
                login_resp = await client.post(
                    "http://localhost:8000/api/v1/auth/login",
                    json={"username": "admin", "password": "admin123"}
                )
                token = login_resp.json()["access_token"]
                
                request_data = {
                    "message": "Default provider 테스트입니다. 안녕하세요\!",
                    "temperature": 0.3,
                    "max_tokens": 100,
                    "episode_ids": [],
                    "novel_ids": []
                }
                
                print(f"\n{'='*60}")
                print(f"Testing DEFAULT configuration")
                print(f"{'='*60}")
                
                try:
                    async with client.stream(
                        "POST",
                        "http://localhost:8000/api/v1/episode/chat/stream",
                        json=request_data,
                        headers={"Authorization": f"Bearer {token}"}
                    ) as response:
                        if response.status_code == 200:
                            print("✅ Default provider working")
                            results.append(("Default", True))
                        else:
                            print(f"❌ Default provider failed: HTTP {response.status_code}")
                            results.append(("Default", False))
                except Exception as e:
                    print(f"❌ Default provider error: {e}")
                    results.append(("Default", False))
        else:
            success = await test_provider(provider, model, api_key)
            results.append((provider, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print(f"{'='*60}")
    for provider, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{provider}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
