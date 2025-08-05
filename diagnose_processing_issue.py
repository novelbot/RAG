#!/usr/bin/env python3
"""
Diagnose why processing might not be working.
"""

import asyncio
import sys
import time
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.episode import EpisodeRAGManager, EpisodeRAGConfig, create_episode_rag_manager
from sqlalchemy import text

async def diagnose_issues():
    """Diagnose processing issues step by step."""
    print("🔍 Diagnosing Processing Issues")
    print("=" * 60)
    
    config = get_config()
    
    print("1. 📋 Configuration Check:")
    print(f"   Database: {config.database.host}:{config.database.port}")
    print(f"   Milvus: {config.milvus.host}:{config.milvus.port}")
    print(f"   Vector dimension: {config.rag.vector_dimension}")
    
    # Test database connection
    print("\n2. 🗄️ Database Connection Test:")
    try:
        db_manager = DatabaseManager(config.database)
        with db_manager.get_connection() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM episode WHERE novel_id = 15"))
            episode_count = result.scalar()
            print(f"   ✅ Database connected")
            print(f"   📊 Novel 15 has {episode_count} episodes")
    except Exception as e:
        print(f"   ❌ Database failed: {e}")
        return False
    
    # Test Milvus connection
    print("\n3. 📡 Milvus Connection Test:")
    try:
        milvus_client = MilvusClient(config.milvus)
        # Connect first
        milvus_client.connect()
        collections = milvus_client.list_collections()
        print(f"   ✅ Milvus connected")
        print(f"   📊 Collections: {collections}")
        
        # Check if episode_embeddings exists and get stats
        if "episode_embeddings" in collections:
            try:
                # Try to get collection info 
                from pymilvus import Collection
                collection = Collection("episode_embeddings")
                collection.load()
                count = collection.num_entities
                print(f"   📊 episode_embeddings has {count} entries")
            except Exception as ce:
                print(f"   ⚠️ Could not get collection stats: {ce}")
        else:
            print(f"   ⚠️ episode_embeddings collection doesn't exist")
            
    except Exception as e:
        print(f"   ❌ Milvus failed: {e}")
        return False
    
    # Test embedding provider
    print("\n4. 🤖 Embedding Provider Test:")
    try:
        embedding_manager = EmbeddingManager(config.embedding)
        from src.embedding.base import EmbeddingRequest
        
        # Test with simple text
        request = EmbeddingRequest(input=["테스트 텍스트"], encoding_format="float")
        response = embedding_manager.generate_embeddings(request)
        
        print(f"   ✅ Embedding generated")
        print(f"   📊 Dimension: {len(response.embeddings[0])}")
        
    except Exception as e:
        print(f"   ❌ Embedding failed: {e}")
        return False
    
    # Test episode RAG manager creation
    print("\n5. 🎯 Episode RAG Manager Test:")
    try:
        episode_config = EpisodeRAGConfig(
            collection_name="episode_embeddings_test",
            processing_batch_size=1,
            vector_dimension=config.rag.vector_dimension
        )
        
        episode_rag_manager = await create_episode_rag_manager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config,
            setup_collection=False
        )
        
        print(f"   ✅ RAG Manager created")
        
        # Test collection setup
        await episode_rag_manager.setup_collection(drop_existing=True)
        print(f"   ✅ Test collection setup successful")
        
        # Test processing a single novel
        print(f"   🔄 Testing single novel processing...")
        result = await episode_rag_manager.process_novel(15, force_reprocess=True)
        
        print(f"   ✅ Novel processing completed")
        print(f"   📊 Result: {result}")
        
        # Clean up test collection
        if milvus_client.has_collection("episode_embeddings_test"):
            milvus_client.drop_collection("episode_embeddings_test")
            print(f"   🧹 Test collection cleaned up")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RAG Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    result = asyncio.run(diagnose_issues())
    print(f"\n{'='*60}")
    if result:
        print("🎉 ALL TESTS PASSED - Processing should work!")
    else:
        print("❌ TESTS FAILED - Processing has issues")
    sys.exit(0 if result else 1)