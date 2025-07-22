#!/usr/bin/env python3
"""
Direct test of /docs endpoint with detailed debugging
"""
import asyncio
import httpx
import subprocess
import time
import os
import signal

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
    'DEBUG': 'true',
    'DB_HOST': 'localhost',
    'DB_PORT': '3306',
    'DB_USER': 'mysql',
    'DB_PASSWORD': 'novelbotisbestie',
    'DB_NAME': 'ragdb',
    'MILVUS_HOST': 'localhost',
    'MILVUS_PORT': '19530',
    'LLM_PROVIDER': 'ollama',
    'LLM_MODEL': 'gemma3:27b-it-q8_0',
    'EMBEDDING_PROVIDER': 'ollama',
    'EMBEDDING_MODEL': 'jeffh/intfloat-multilingual-e5-large-instruct:f32',
    'SECRET_KEY': 'test-secret-key',
    'API_HOST': '127.0.0.1',
    'API_PORT': '8004'
})

async def test_docs_endpoint():
    """Test /docs endpoint with detailed analysis"""
    print("🧪 Testing /docs endpoint...")
    
    # Start server
    process = subprocess.Popen(
        ['uv', 'run', 'python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("   Waiting for server startup...")
    await asyncio.sleep(4)
    
    base_url = "http://127.0.0.1:8004"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test each endpoint with detailed info
        endpoints = [
            "/",
            "/health", 
            "/openapi.json",
            "/docs",
            "/redoc"
        ]
        
        for endpoint in endpoints:
            try:
                print(f"\n🔍 Testing {endpoint}...")
                response = await client.get(f"{base_url}{endpoint}")
                
                print(f"   Status Code: {response.status_code}")
                print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
                print(f"   Content-Length: {len(response.content)} bytes")
                
                if response.status_code != 200:
                    content_preview = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    print(f"   Response: {content_preview}")
                else:
                    print(f"   ✅ Success!")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Kill server
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    
    print("\n" + "=" * 50)
    print("If /docs shows 'Not Found', possible causes:")
    print("1. docs_url is disabled in FastAPI config")
    print("2. Middleware is blocking the request")
    print("3. Route registration issue")
    print("4. OpenAPI schema generation problem")

if __name__ == "__main__":
    asyncio.run(test_docs_endpoint())