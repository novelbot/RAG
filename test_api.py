#!/usr/bin/env python3
"""
Test API endpoints including /docs
"""
import asyncio
import httpx
import subprocess
import time
import os
from pathlib import Path

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
    'API_PORT': '8002'
})

async def test_api_endpoints():
    """Test various API endpoints"""
    print("🧪 Testing FastAPI Endpoints...")
    
    # Start server in background
    process = subprocess.Popen(
        ['uv', 'run', 'python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path.cwd()
    )
    
    # Wait for startup
    print("   Waiting for server startup...")
    await asyncio.sleep(4)
    
    base_url = "http://127.0.0.1:8002"
    
    # Test endpoints
    endpoints_to_test = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/docs", "API Documentation"),
        ("/redoc", "ReDoc Documentation"),  
        ("/openapi.json", "OpenAPI Schema"),
        ("/api/v1", "API Root"),
    ]
    
    results = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint, description in endpoints_to_test:
            try:
                print(f"   Testing {endpoint} ({description})...")
                response = await client.get(f"{base_url}{endpoint}")
                
                if response.status_code == 200:
                    print(f"   ✅ {endpoint} - OK ({len(response.content)} bytes)")
                    results[endpoint] = "✅ PASS"
                elif response.status_code == 404:
                    print(f"   ❌ {endpoint} - Not Found")
                    results[endpoint] = "❌ NOT_FOUND"
                elif response.status_code == 401:
                    print(f"   🔒 {endpoint} - Unauthorized (auth required)")
                    results[endpoint] = "🔒 AUTH_REQUIRED"
                else:
                    print(f"   ⚠️  {endpoint} - Status {response.status_code}")
                    results[endpoint] = f"⚠️  {response.status_code}"
                    
            except Exception as e:
                print(f"   ❌ {endpoint} - Error: {e}")
                results[endpoint] = f"❌ ERROR"
    
    # Terminate server
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 API ENDPOINT TEST RESULTS")
    print("=" * 50)
    
    for endpoint, result in results.items():
        print(f"{endpoint:<20} {result}")
    
    # Check if /docs is working
    docs_working = results.get("/docs") == "✅ PASS"
    
    if docs_working:
        print("\n🎉 FastAPI /docs is working correctly!")
    else:
        print("\n⚠️  FastAPI /docs is not accessible")
        print("This could be due to:")
        print("• Debug mode not enabled")
        print("• Route registration issues") 
        print("• Authentication middleware blocking access")
    
    return docs_working

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())