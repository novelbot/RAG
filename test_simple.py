#!/usr/bin/env python3
"""
Simple test to verify core functionality
"""
import os
import asyncio

# Set environment variables
os.environ.update({
    'APP_ENV': 'development',
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
    'EMBEDDING_MODEL': 'jeffh/intfloat-multilingual-e5-large-instruct:f32'
})

async def test_database():
    """Test MySQL database connection"""
    print("🔧 Testing MySQL database...")
    try:
        import pymysql
        connection = pymysql.connect(
            host='localhost',
            port=3306,
            user='mysql',
            password='novelbotisbestie',
            database='ragdb'
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✅ MySQL connected! Version: {version[0]}")
        connection.close()
        return True
    except Exception as e:
        print(f"❌ MySQL failed: {e}")
        return False

async def test_milvus():
    """Test Milvus vector database"""
    print("🔧 Testing Milvus database...")
    try:
        from pymilvus import connections, utility
        connections.connect(host='localhost', port=19530)
        version = utility.get_server_version()
        print(f"✅ Milvus connected! Version: {version}")
        return True
    except Exception as e:
        print(f"❌ Milvus failed: {e}")
        return False

async def test_ollama():
    """Test Ollama models"""
    print("🔧 Testing Ollama models...")
    try:
        import ollama
        
        # Test embedding
        embed_resp = ollama.embeddings(
            model='jeffh/intfloat-multilingual-e5-large-instruct:f32',
            prompt='test embedding'
        )
        embed_dim = len(embed_resp["embedding"])
        print(f"✅ Embedding model working! Dimension: {embed_dim}")
        
        # Test LLM
        chat_resp = ollama.chat(
            model='gemma3:27b-it-q8_0',
            messages=[{'role': 'user', 'content': 'Say "OK" only'}]
        )
        response = chat_resp["message"]["content"]
        print(f"✅ LLM model working! Response: {response.strip()}")
        
        return True
    except Exception as e:
        print(f"❌ Ollama failed: {e}")
        return False

async def main():
    print("🚀 RAG Server Simple Test")
    print("=" * 40)
    
    results = []
    results.append(await test_database())
    results.append(await test_milvus()) 
    results.append(await test_ollama())
    
    print("\n" + "=" * 40)
    print("📊 RESULTS")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    services = ['MySQL Database', 'Milvus VectorDB', 'Ollama Models']
    for i, (service, result) in enumerate(zip(services, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{service:.<20} {status}")
    
    print(f"\nOverall: {passed}/{total} services working")
    
    if passed == total:
        print("\n🎉 All core services are working!")
        print("Your RAG server setup is ready to use:")
        print("• MySQL (port 3306) - ✅ ")
        print("• Milvus (port 19530) - ✅")
        print("• Ollama Embedding - ✅")
        print("• Ollama LLM - ✅")
        print("\nYou can now start the RAG server with: python main.py")
    else:
        print("\n⚠️  Some services need attention.")

if __name__ == "__main__":
    asyncio.run(main())