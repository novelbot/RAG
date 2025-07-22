#!/usr/bin/env python3
"""
Test the monitoring status function directly without server
"""
import asyncio
import os
import sys
sys.path.append('.')

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
    'DEBUG': 'true',
    'DB_HOST': '35.216.85.254',
    'DB_PORT': '3306',
    'DB_NAME': 'novelbot',
    'DB_USER': 'rag-server',
    'DB_PASSWORD': 'rag901829!',
    'MILVUS_HOST': 'localhost',
    'MILVUS_PORT': '19530',
    'LLM_PROVIDER': 'ollama',
    'LLM_MODEL': 'gemma3:27b-it-q8_0',
    'EMBEDDING_PROVIDER': 'ollama',
    'EMBEDDING_MODEL': 'jeffh/intfloat-multilingual-e5-large-instruct:f32'
})

async def test_status_function():
    """Test the monitoring status function directly"""
    print("🔧 Testing Status Function Directly...")
    
    try:
        from src.api.routes.monitoring import get_service_status
        
        print("   Calling get_service_status()...")
        result = await get_service_status()
        
        print("✅ Function executed successfully!")
        
        # Display results
        services = result.get('services', {})
        print(f"\n📊 Results:")
        print(f"   Overall Status: {result.get('overall_status', 'N/A')}")
        print(f"   Response Time: {result.get('response_time_ms', 'N/A')}ms")
        
        # Database
        database = services.get('database', {})
        print(f"\n💾 Database:")
        print(f"   Status: {database.get('status', 'N/A')}")
        if database.get('status') == 'connected':
            print(f"   Host: {database.get('host', 'N/A')}")
            print(f"   Database: {database.get('database', 'N/A')}")
            print(f"   Response Time: {database.get('response_time_ms', 'N/A')}ms")
        elif 'error' in database:
            print(f"   Error: {database.get('error', 'N/A')}")
        
        # Vector Database
        vector_db = services.get('vector_database', {})
        print(f"\n🔍 Vector Database:")
        print(f"   Status: {vector_db.get('status', 'N/A')}")
        if vector_db.get('status') == 'connected':
            print(f"   Host: {vector_db.get('host', 'N/A')}")
            print(f"   Collections: {vector_db.get('collection_count', 0)}")
            print(f"   Collections List: {vector_db.get('collections', [])}")
            print(f"   Version: {vector_db.get('version', 'N/A')}")
        elif 'error' in vector_db:
            print(f"   Error: {vector_db.get('error', 'N/A')}")
        
        # LLM Providers
        llm_providers = services.get('llm_providers', {})
        print(f"\n🤖 LLM Providers:")
        for provider, info in llm_providers.items():
            status = info.get('status', 'N/A')
            print(f"   {provider}: {status}")
            print(f"     Model: {info.get('model', 'N/A')}")
            if status == 'available':
                print(f"     Latency: {info.get('latency_ms', 'N/A')}ms")
            elif 'error' in info:
                print(f"     Error: {info.get('error', 'N/A')}")
        
        # Embedding Providers
        embedding_providers = services.get('embedding_providers', {})
        print(f"\n📊 Embedding Providers:")
        for provider, info in embedding_providers.items():
            status = info.get('status', 'N/A')
            print(f"   {provider}: {status}")
            print(f"     Model: {info.get('model', 'N/A')}")
            if status == 'available':
                print(f"     Dimension: {info.get('dimension', 'N/A')}")
                print(f"     Latency: {info.get('latency_ms', 'N/A')}ms")
            elif 'error' in info:
                print(f"     Error: {info.get('error', 'N/A')}")
        
        # Configuration
        config = result.get('configuration', {})
        print(f"\n⚙️  Configuration:")
        print(f"   LLM Provider: {config.get('llm_provider', 'N/A')}")
        print(f"   LLM Model: {config.get('llm_model', 'N/A')}")
        print(f"   Embedding Provider: {config.get('embedding_provider', 'N/A')}")
        print(f"   Embedding Model: {config.get('embedding_model', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_status_function())