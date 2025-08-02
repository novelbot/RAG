#!/usr/bin/env python3
"""
Simplified Episode RAG Integration Test
실제 RDB와 Milvus를 사용한 간단한 에피소드 RAG 테스트
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any

# Add project root to path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from src.core.config import get_config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.milvus.client import MilvusClient
from src.embedding.factory import get_embedding_client
from src.embedding.base import EmbeddingRequest


class SimpleEpisodeRAGTest:
    def __init__(self):
        self.config = get_config()
        self.db = None
        self.milvus_client = None
        self.embedding_client = None
        self.test_collection = "test_simple_episode_rag"
    
    def setup(self):
        """환경 설정"""
        print("🔧 Setting up simple Episode RAG test...")
        
        # Database connection
        db_config = self.config.database
        database_url = f"{db_config.driver}://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        engine = create_engine(database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.db = SessionLocal()
        print("✅ Database connected")
        
        # Milvus connection
        self.milvus_client = MilvusClient(self.config.milvus)
        self.milvus_client.connect()
        print("✅ Milvus connected")
        
        # Embedding client
        self.embedding_client = get_embedding_client(self.config.embedding)
        print("✅ Embedding client ready")
    
    def cleanup(self):
        """정리"""
        print("\n🧹 Cleanup...")
        if self.milvus_client and self.milvus_client.has_collection(self.test_collection):
            self.milvus_client.drop_collection(self.test_collection)
            print(f"✅ Dropped test collection: {self.test_collection}")
        
        if self.milvus_client:
            self.milvus_client.disconnect()
        if self.db:
            self.db.close()
    
    def test_rdb_data_fetch(self):
        """RDB에서 에피소드 데이터 가져오기 테스트"""
        print("\n📊 Testing RDB episode data fetch...")
        
        # Test connection
        result = self.db.execute(text("SELECT 1 as test"))
        if result.fetchone()[0] != 1:
            raise Exception("DB connection failed")
        print("✅ DB connection test passed")
        
        # Fetch episodes
        query = text("""
            SELECT 
                episode_id,
                episode_number, 
                episode_title,
                content,
                publication_date,
                novel_id
            FROM episode 
            WHERE content IS NOT NULL 
            AND LENGTH(content) > 100
            LIMIT 10
        """)
        
        result = self.db.execute(query)
        rows = result.fetchall()
        
        if not rows:
            raise Exception("No episodes found")
        
        # Convert to episode dictionaries
        episodes = []
        for row in rows:
            episode_data = {
                'episode_id': row.episode_id,
                'episode_number': row.episode_number,
                'episode_title': row.episode_title,
                'content': row.content,
                'publication_date': row.publication_date,
                'novel_id': row.novel_id
            }
            episodes.append(episode_data)
        
        print(f"✅ Found {len(episodes)} episodes")
        for ep in episodes:
            print(f"   Episode {ep['episode_number']}: {ep['episode_title'][:50]}...")
        
        return episodes
    
    def test_embedding_generation(self, episodes):
        """임베딩 생성 테스트"""
        print(f"\n🔄 Testing embedding generation for {len(episodes)} episodes...")
        
        episode_embeddings = []
        
        for episode in episodes:
            # Generate embedding
            request = EmbeddingRequest(input=[episode['content'][:1000]])  # First 1000 chars
            response = self.embedding_client.generate_embeddings(request)
            
            if not response.embeddings:
                raise Exception(f"No embedding generated for episode {episode['episode_id']}")
            
            embedding = response.embeddings[0]
            if len(embedding) != 1024:
                raise Exception(f"Wrong embedding dimension: {len(embedding)}")
            
            episode_embeddings.append({
                'episode_id': episode['episode_id'],
                'episode_number': episode['episode_number'], 
                'episode_title': episode['episode_title'],
                'content': episode['content'],
                'publication_date': episode['publication_date'],
                'novel_id': episode['novel_id'],
                'embedding': embedding
            })
        
        print(f"✅ Generated embeddings for all episodes (dim: 1024)")
        return episode_embeddings
    
    def test_milvus_storage(self, episode_embeddings):
        """Milvus 저장 테스트"""
        print(f"\n💾 Testing Milvus storage for {len(episode_embeddings)} episodes...")
        
        # Create collection if not exists
        collection = self.milvus_client.create_collection_if_not_exists(
            collection_name=self.test_collection,
            dim=1024
        )
        print(f"✅ Collection ready: {self.test_collection}")
        
        # Prepare data for insertion
        data = []
        for ep in episode_embeddings:
            data.append({
                "id": str(ep['episode_id']),
                "content": ep['content'][:1000],  # Limit content length
                "embedding": ep['embedding'],
                "metadata": {
                    "episode_number": ep['episode_number'],
                    "episode_title": ep['episode_title'],
                    "novel_id": ep['novel_id']
                },
                "access_tags": "test"
            })
        
        # Insert data
        collection.insert(data)
        collection.flush()
        
        # Verify insertion
        count = collection.num_entities
        if count != len(episode_embeddings):
            raise Exception(f"Storage failed: expected {len(episode_embeddings)}, got {count}")
        
        print(f"✅ Stored {count} episodes in Milvus")
        return collection
    
    def test_vector_search(self, collection, episode_embeddings):
        """벡터 검색 테스트"""
        print(f"\n🔍 Testing vector search...")
        
        # Use first episode's content as query
        query_text = episode_embeddings[0]['content'][:200]
        print(f"Query: {query_text}...")
        
        # Generate query embedding
        request = EmbeddingRequest(input=[query_text])
        response = self.embedding_client.generate_embeddings(request)
        query_embedding = response.embeddings[0]
        
        # Search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["content", "metadata"]
        )
        
        if not results or len(results[0]) == 0:
            raise Exception("No search results found")
        
        print(f"✅ Found {len(results[0])} search results")
        for i, hit in enumerate(results[0]):
            distance = hit.distance
            metadata = hit.entity.get("metadata", {})
            episode_num = metadata.get("episode_number", "?")
            episode_title = metadata.get("episode_title", "?")
            print(f"   {i+1}. Episode {episode_num}: {episode_title} (distance: {distance:.3f})")
        
        return results
    
    def run_test(self):
        """전체 테스트 실행"""
        print("🚀 Starting Simple Episode RAG Integration Test")
        print("=" * 70)
        
        try:
            # Setup
            self.setup()
            
            # Test RDB
            episodes = self.test_rdb_data_fetch()
            
            # Test embedding
            episode_embeddings = self.test_embedding_generation(episodes)
            
            # Test Milvus storage
            collection = self.test_milvus_storage(episode_embeddings)
            
            # Test search
            self.test_vector_search(collection, episode_embeddings)
            
            print("\n" + "=" * 70)
            print("🎉 ALL TESTS PASSED!")
            print("✅ Episode RAG system works with real RDB and Milvus")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.cleanup()


if __name__ == "__main__":
    test = SimpleEpisodeRAGTest()
    test.run_test()