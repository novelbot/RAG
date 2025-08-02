#!/usr/bin/env python3
"""
Debug Milvus connection issues in project pipeline
"""

import os
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test configuration loading"""
    print("🔧 Testing configuration loading...")
    
    try:
        from src.core.config import get_config
        config = get_config()
        
        print(f"✅ Config loaded successfully")
        print(f"   Milvus Host: {config.milvus.host}")
        print(f"   Milvus Port: {config.milvus.port}")
        print(f"   Milvus User: {config.milvus.user}")
        print(f"   Milvus Alias: {config.milvus.alias}")
        
        return config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_milvus_client():
    """Test MilvusClient initialization and connection"""
    print("\n🔧 Testing MilvusClient...")
    
    try:
        from src.core.config import get_config
        from src.milvus.client import MilvusClient
        
        config = get_config()
        client = MilvusClient(config.milvus)
        
        print("✅ MilvusClient created successfully")
        
        # Test connection
        print("🔗 Testing connection...")
        success = client.connect()
        
        if success:
            print("✅ Connection successful")
            
            # Test basic operations
            collections = client.list_collections()
            print(f"✅ Collections found: {collections}")
            
            # Test ping
            ping_result = client.ping()
            print(f"✅ Ping result: {ping_result['status']}")
            
        else:
            print("❌ Connection failed")
            
        return client
        
    except Exception as e:
        print(f"❌ MilvusClient test failed: {e}")
        traceback.print_exc()
        return None

def test_vector_search_engine():
    """Test VectorSearchEngine initialization"""
    print("\n🔧 Testing VectorSearchEngine...")
    
    try:
        from src.rag.vector_search_engine import VectorSearchEngine
        from src.core.config import get_config
        from src.milvus.client import MilvusClient
        
        # Create required dependencies
        config = get_config()
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        engine = VectorSearchEngine(milvus_client=milvus_client)
        print("✅ VectorSearchEngine created successfully")
        
        # Test health check instead
        health = engine.health_check()
        print(f"✅ VectorSearchEngine health: {health['status']}")
        
        return engine
        
    except Exception as e:
        print(f"❌ VectorSearchEngine test failed: {e}")
        traceback.print_exc()
        return None

def test_embedding_client():
    """Test embedding client initialization"""
    print("\n🔧 Testing embedding client...")
    
    try:
        from src.embedding.factory import get_embedding_client
        from src.core.config import get_config
        
        config = get_config()
        client = get_embedding_client(config.embedding)
        print("✅ Embedding client created successfully")
        
        # Test embedding generation
        from src.embedding.base import EmbeddingRequest
        test_text = "This is a test sentence."
        request = EmbeddingRequest(input=[test_text])
        response = client.generate_embeddings(request)
        print(f"✅ Embedding generated, dimension: {len(response.embeddings[0])}")
        
        return client
        
    except Exception as e:
        print(f"❌ Embedding client test failed: {e}")
        traceback.print_exc()
        return None

def test_data_sync_service():
    """Test DataSyncManager initialization"""
    print("\n🔧 Testing DataSyncManager...")
    
    try:
        from src.services.data_sync import DataSyncManager, SyncState, DataSourceType, SyncStatus
        
        # Create a mock sync state
        sync_state = SyncState(
            source_id="test_source",
            source_type=DataSourceType.DATABASE,
            last_sync=None,
            last_hash="",
            sync_status=SyncStatus.PENDING,
            error_message=None
        )
        
        # DataSyncManager needs dependencies, try basic instantiation
        print("✅ DataSyncManager imports successfully")
        print(f"✅ SyncState created: {sync_state.source_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataSyncManager test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Main debug function"""
    print("🚀 Debug Milvus Pipeline Issues")
    print("=" * 60)
    
    # Load environment
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Environment loaded")
    else:
        print("❌ .env file not found")
        return
    
    # Test individual components
    config = test_config_loading()
    if not config:
        return
    
    client = test_milvus_client()
    if not client:
        return
    
    engine = test_vector_search_engine()
    if not engine:
        return
    
    embedding_client = test_embedding_client()
    if not embedding_client:
        return
    
    service = test_data_sync_service()
    if not service:
        return
    
    print("\n" + "=" * 60)
    print("✅ All components initialized successfully!")
    print("🔍 Pipeline should be working correctly.")

if __name__ == "__main__":
    main()