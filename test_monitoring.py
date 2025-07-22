#!/usr/bin/env python3
"""
Test the updated monitoring status endpoint
"""
import asyncio
import httpx
import subprocess
import time
import os
import json

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
    'DEBUG': 'true'
})

async def test_monitoring_status():
    """Test the updated /api/v1/monitoring/status endpoint"""
    print("🔧 Testing Updated Monitoring Status Endpoint...")
    
    # Start server
    process = subprocess.Popen(
        ['uv', 'run', 'python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("   Waiting for server startup...")
    await asyncio.sleep(4)
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            print("   Making request to /api/v1/monitoring/status...")
            response = await client.get("http://localhost:8000/api/v1/monitoring/status")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Status endpoint working!")
                print(f"   Response time: {data.get('response_time_ms', 'N/A')}ms")
                print(f"   Overall status: {data.get('overall_status', 'N/A')}")
                
                # Display services status
                services = data.get('services', {})
                print("\n📊 Services Status:")
                
                # API Server
                api_server = services.get('api_server', {})
                print(f"   API Server: {api_server.get('status', 'N/A')}")
                print(f"     Uptime: {api_server.get('uptime', 'N/A')}")
                print(f"     Environment: {api_server.get('environment', 'N/A')}")
                
                # Database
                database = services.get('database', {})
                print(f"   Database: {database.get('status', 'N/A')}")
                if database.get('status') == 'connected':
                    print(f"     Host: {database.get('host', 'N/A')}")
                    print(f"     Database: {database.get('database', 'N/A')}")
                    print(f"     Response time: {database.get('response_time_ms', 'N/A')}ms")
                elif 'error' in database:
                    print(f"     Error: {database.get('error', 'N/A')}")
                
                # Vector Database
                vector_db = services.get('vector_database', {})
                print(f"   Vector Database: {vector_db.get('status', 'N/A')}")
                if vector_db.get('status') == 'connected':
                    print(f"     Host: {vector_db.get('host', 'N/A')}")
                    print(f"     Collections: {vector_db.get('collection_count', 0)}")
                    print(f"     Version: {vector_db.get('version', 'N/A')}")
                elif 'error' in vector_db:
                    print(f"     Error: {vector_db.get('error', 'N/A')}")
                
                # LLM Providers
                llm_providers = services.get('llm_providers', {})
                print(f"   LLM Providers:")
                for provider, info in llm_providers.items():
                    status = info.get('status', 'N/A')
                    model = info.get('model', 'N/A')
                    print(f"     {provider}: {status}")
                    print(f"       Model: {model}")
                    if status == 'available':
                        print(f"       Latency: {info.get('latency_ms', 'N/A')}ms")
                    elif 'error' in info:
                        print(f"       Error: {info.get('error', 'N/A')}")
                
                # Embedding Providers
                embedding_providers = services.get('embedding_providers', {})
                print(f"   Embedding Providers:")
                for provider, info in embedding_providers.items():
                    status = info.get('status', 'N/A')
                    model = info.get('model', 'N/A')
                    print(f"     {provider}: {status}")
                    print(f"       Model: {model}")
                    if status == 'available':
                        print(f"       Dimension: {info.get('dimension', 'N/A')}")
                        print(f"       Latency: {info.get('latency_ms', 'N/A')}ms")
                    elif 'error' in info:
                        print(f"       Error: {info.get('error', 'N/A')}")
                
                # Configuration
                config = data.get('configuration', {})
                print(f"\n⚙️  Configuration:")
                print(f"   LLM Provider: {config.get('llm_provider', 'N/A')}")
                print(f"   LLM Model: {config.get('llm_model', 'N/A')}")
                print(f"   Embedding Provider: {config.get('embedding_provider', 'N/A')}")
                print(f"   Embedding Model: {config.get('embedding_model', 'N/A')}")
                
            elif response.status_code == 401:
                print("🔒 Authentication required for monitoring endpoint")
                print("   This is expected if authentication middleware is enabled")
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
    
    finally:
        # Kill server
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

if __name__ == "__main__":
    asyncio.run(test_monitoring_status())