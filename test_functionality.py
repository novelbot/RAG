#!/usr/bin/env python3
"""
Comprehensive functionality test for RAG Server
Tests MySQL (alternative database), Milvus, Ollama embedding, and Ollama LLM
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_ollama_embedding():
    """Test Ollama embedding model"""
    print("\n🔧 Testing Ollama Embedding Model...")
    try:
        import ollama
        response = ollama.embeddings(
            model='jeffh/intfloat-multilingual-e5-large-instruct:f32',
            prompt='Test embedding functionality'
        )
        embedding = response["embedding"]
        print(f"✅ Embedding model working! Dimension: {len(embedding)}")
        return True, len(embedding)
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return False, None

async def test_ollama_llm():
    """Test Ollama LLM model"""
    print("\n🔧 Testing Ollama LLM Model...")
    try:
        import ollama
        response = ollama.chat(
            model='gemma3:27b-it-q8_0',
            messages=[
                {'role': 'user', 'content': 'Respond with exactly "LLM_TEST_SUCCESS"'}
            ]
        )
        answer = response["message"]["content"].strip()
        print(f"✅ LLM model working! Response: {answer}")
        return True, answer
    except Exception as e:
        print(f"❌ LLM model failed: {e}")
        return False, None

async def test_milvus_connection():
    """Test Milvus connection"""
    print("\n🔧 Testing Milvus Connection...")
    try:
        from pymilvus import connections, utility
        
        # Connect
        connections.connect(
            alias="test",
            host='localhost',
            port=19530
        )
        
        # Test connection
        version = utility.get_server_version()
        collections = utility.list_collections()
        
        print(f"✅ Milvus connection successful!")
        print(f"   Version: {version}")
        print(f"   Collections: {len(collections)} found")
        
        connections.disconnect("test")
        return True
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        return False

async def test_mysql_connection():
    """Test MySQL connection"""
    print("\n🔧 Testing MySQL Connection...")
    try:
        import pymysql
        connection = pymysql.connect(
            host='localhost',
            port=3306,
            user='mysql',
            password='novelbotisbestie',
            database='ragdb'
        )
        # Test basic query
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            
        connection.close()
        print("✅ MySQL connection successful!")
        return True
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return False

async def test_rag_pipeline():
    """Test basic RAG pipeline functionality"""
    print("\n🔧 Testing RAG Pipeline Components...")
    
    # Test embedding
    embedding_success, embedding_dim = await test_ollama_embedding()
    if not embedding_success:
        return False
    
    # Test LLM
    llm_success, llm_response = await test_ollama_llm()
    if not llm_success:
        return False
    
    # Test vector storage (Milvus)
    milvus_success = await test_milvus_connection()
    if not milvus_success:
        return False
        
    print("\n✅ All RAG pipeline components working!")
    return True

async def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing Configuration...")
    try:
        from core.config import get_config, ConfigManager
        
        # Load with environment variables
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        print(f"✅ Configuration loaded:")
        print(f"   App: {config.app_name} v{config.version}")
        print(f"   Environment: {config.environment}")
        print(f"   Database: {config.database.driver}://{config.database.host}:{config.database.port}")
        print(f"   Milvus: {config.milvus.host}:{config.milvus.port}")
        print(f"   LLM: {config.llm.provider} - {config.llm.model}")
        print(f"   Embedding: {config.embedding.provider} - {config.embedding.model}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all functionality tests"""
    print("🚀 Starting RAG Server Functionality Tests")
    print("=" * 50)
    
    # Test results
    results = {}
    
    # Test individual components
    results['mysql'] = await test_mysql_connection()
    results['milvus'] = await test_milvus_connection() 
    results['embedding'] = (await test_ollama_embedding())[0]
    results['llm'] = (await test_ollama_llm())[0]
    results['config'] = await test_configuration()
    results['rag_pipeline'] = await test_rag_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<20} {status}")
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All functionality tests PASSED!")
        print("\nYour RAG server setup is working correctly with:")
        print("• MySQL database")
        print("• Milvus vector database") 
        print("• Ollama embedding model (jeffh/intfloat-multilingual-e5-large-instruct:f32)")
        print("• Ollama LLM model (gemma3:27b-it-q8_0)")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ['APP_ENV'] = 'development'
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '3306'  
    os.environ['DB_USER'] = 'mysql'
    os.environ['DB_PASSWORD'] = 'novelbotisbestie'
    os.environ['DB_NAME'] = 'ragdb'
    os.environ['MILVUS_HOST'] = 'localhost'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['LLM_PROVIDER'] = 'ollama'
    os.environ['LLM_MODEL'] = 'gemma3:27b-it-q8_0'
    os.environ['EMBEDDING_PROVIDER'] = 'ollama'
    os.environ['EMBEDDING_MODEL'] = 'jeffh/intfloat-multilingual-e5-large-instruct:f32'
    
    asyncio.run(main())