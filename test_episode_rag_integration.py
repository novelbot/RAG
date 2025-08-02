#!/usr/bin/env python3
"""
Episode RAG Integration Test with Real RDB and Milvus
테스트 실제 RDB 및 Milvus 환경에서 에피소드 RAG 기능의 완전한 통합
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from src.core.config import get_config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.milvus.client import MilvusClient
from src.embedding.factory import get_embedding_client
from src.embedding.manager import EmbeddingManager, EmbeddingProviderConfig
from src.episode.models import EpisodeData, EpisodeSearchRequest, EpisodeSortOrder
from src.episode.vector_store import EpisodeVectorStore, EpisodeVectorStoreConfig
from src.episode.search_engine import EpisodeSearchEngine
from src.episode.processor import EpisodeEmbeddingProcessor, EpisodeProcessingConfig
from sqlalchemy import text

class EpisodeRAGIntegrationTest:
    """에피소드 RAG 통합 테스트 클래스"""
    
    def __init__(self):
        self.config = get_config()
        self.db = None
        self.milvus_client = None
        self.embedding_manager = None
        self.vector_store = None
        self.search_engine = None
        self.processor = None
        
        # Test collection name
        self.test_collection = "test_episode_rag_integration"
        
    async def setup(self):
        """테스트 환경 초기화"""
        print("🔧 Setting up Episode RAG integration test environment...")
        
        try:
            # Database connection
            db_config = self.config.database
            database_url = f"{db_config.driver}://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
            engine = create_engine(database_url, pool_pre_ping=True)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.db = SessionLocal()
            print("✅ Database connection established")
            
            # Milvus client
            self.milvus_client = MilvusClient(self.config.milvus)
            self.milvus_client.connect()
            print("✅ Milvus connection established")
            
            # Embedding manager
            embedding_client = get_embedding_client(self.config.embedding)
            provider_config = EmbeddingProviderConfig(
                provider=self.config.embedding.provider,
                config=self.config.embedding,
                instance=embedding_client
            )
            self.embedding_manager = EmbeddingManager([provider_config])
            print("✅ Embedding manager initialized")
            
            # Vector store
            vector_config = EpisodeVectorStoreConfig(
                collection_name=self.test_collection,
                vector_dimension=1024,  # E5 model dimension
                drop_existing=True
            )
            self.vector_store = EpisodeVectorStore(
                milvus_client=self.milvus_client,
                config=vector_config
            )
            await self.vector_store.initialize()
            print(f"✅ Vector store initialized with collection: {self.test_collection}")
            
            # Episode processor
            processing_config = EpisodeProcessingConfig(
                batch_size=10,
                enable_content_cleaning=True
            )
            self.processor = EpisodeEmbeddingProcessor(
                embedding_manager=self.embedding_manager,
                config=processing_config
            )
            print("✅ Episode processor initialized")
            
            # Search engine
            self.search_engine = EpisodeSearchEngine(
                milvus_client=self.milvus_client,
                embedding_manager=self.embedding_manager,
                vector_store=self.vector_store
            )
            print("✅ Episode search engine initialized")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def cleanup(self):
        """테스트 환경 정리"""
        print("\n🧹 Cleaning up test environment...")
        
        try:
            # Drop test collection
            if self.milvus_client and self.milvus_client.has_collection(self.test_collection):
                self.milvus_client.drop_collection(self.test_collection)
                print(f"✅ Dropped test collection: {self.test_collection}")
            
            # Close connections
            if self.milvus_client:
                self.milvus_client.disconnect()
                print("✅ Milvus connection closed")
            
            if self.db:
                self.db.close()
                print("✅ Database connection closed")
                
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    async def test_rdb_connection_and_data_fetch(self) -> List[EpisodeData]:
        """RDB 연결 및 에피소드 데이터 가져오기 테스트"""
        print("\n📊 Testing RDB connection and episode data fetch...")
        
        try:
            # Test basic connection
            result = self.db.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()
            if test_value[0] != 1:
                raise Exception("Database connection test failed")
            print("✅ Database connection test passed")
            
            # Fetch sample episodes from RDB
            query = text("""
                SELECT 
                    episode_id,
                    episode_number,
                    episode_title,
                    content,
                    publication_date,
                    novel_id
                FROM episodes 
                WHERE content IS NOT NULL 
                AND LENGTH(content) > 100
                ORDER BY episode_id 
                LIMIT 20
            """)
            
            result = self.db.execute(query)
            rows = result.fetchall()
            
            if not rows:
                raise Exception("No episode data found in RDB")
            
            # Convert to EpisodeData objects
            episodes = []
            for row in rows:
                episode_data = EpisodeData(
                    episode_id=row.episode_id,
                    episode_number=row.episode_number,
                    episode_title=row.episode_title,
                    content=row.content,
                    publication_date=row.publication_date,
                    novel_id=row.novel_id
                )
                episodes.append(episode_data)
            
            print(f"✅ Fetched {len(episodes)} episodes from RDB")
            print(f"   Sample episode: ID={episodes[0].episode_id}, Title='{episodes[0].episode_title}'")
            print(f"   Content length range: {min(ep.content_length for ep in episodes)}-{max(ep.content_length for ep in episodes)} chars")
            
            return episodes
            
        except Exception as e:
            print(f"❌ RDB data fetch failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def test_episode_processing_and_storage(self, episodes: List[EpisodeData]) -> None:
        """에피소드 처리 및 벡터 저장 테스트"""
        print(f"\n🔄 Testing episode processing and vector storage for {len(episodes)} episodes...")
        
        try:
            start_time = time.time()
            
            # Process episodes to generate embeddings
            processed_episodes = await self.processor.process_episodes_batch(episodes)
            
            processing_time = time.time() - start_time
            print(f"✅ Processed {len(processed_episodes)} episodes in {processing_time:.2f}s")
            
            # Verify embeddings were generated
            for episode in processed_episodes:
                if not episode.embedding:
                    raise Exception(f"No embedding generated for episode {episode.episode_id}")
                if len(episode.embedding) != 1024:
                    raise Exception(f"Invalid embedding dimension for episode {episode.episode_id}: {len(episode.embedding)}")
            
            print(f"✅ All episodes have valid embeddings (dimension: 1024)")
            
            # Store in vector database
            start_time = time.time()
            stored_count = await self.vector_store.store_episodes(processed_episodes)
            storage_time = time.time() - start_time
            
            print(f"✅ Stored {stored_count} episodes in vector database in {storage_time:.2f}s")
            
            # Verify storage
            collection_count = self.milvus_client.get_entity_count(self.test_collection)
            if collection_count != len(processed_episodes):
                raise Exception(f"Storage verification failed: expected {len(processed_episodes)}, got {collection_count}")
            
            print(f"✅ Storage verification passed: {collection_count} entities in collection")
            
        except Exception as e:
            print(f"❌ Episode processing/storage failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def test_episode_search_functionality(self, original_episodes: List[EpisodeData]) -> None:
        """에피소드 검색 기능 테스트"""
        print(f"\n🔍 Testing episode search functionality...")
        
        try:
            # Test 1: Basic similarity search
            print("Test 1: Basic similarity search")
            search_request = EpisodeSearchRequest(
                query="주인공의 모험",
                limit=5,
                include_content=True
            )
            
            start_time = time.time()
            search_result = await self.search_engine.search(search_request)
            search_time = time.time() - start_time
            
            print(f"✅ Basic search completed in {search_time:.3f}s")
            print(f"   Found {len(search_result.hits)} results")
            if search_result.hits:
                best_hit = search_result.hits[0]
                print(f"   Best match: Episode {best_hit.episode_number} (similarity: {best_hit.similarity_score:.3f})")
            
            # Test 2: Episode ID filtering
            print("\nTest 2: Episode ID filtering")
            if len(original_episodes) >= 3:
                target_episode_ids = [ep.episode_id for ep in original_episodes[:3]]
                search_request = EpisodeSearchRequest(
                    query="이야기",
                    episode_ids=target_episode_ids,
                    limit=10
                )
                
                search_result = await self.search_engine.search(search_request)
                print(f"✅ Filtered search completed")
                print(f"   Requested episodes: {target_episode_ids}")
                print(f"   Found {len(search_result.hits)} results")
                
                # Verify all results are from requested episodes
                found_ids = [hit.episode_id for hit in search_result.hits]
                if not all(id in target_episode_ids for id in found_ids):
                    raise Exception(f"Filter validation failed: found IDs {found_ids} not in {target_episode_ids}")
                print("✅ Episode ID filtering works correctly")
            
            # Test 3: Sort by episode number
            print("\nTest 3: Sort by episode number")
            search_request = EpisodeSearchRequest(
                query="캐릭터",
                limit=5,
                sort_order=EpisodeSortOrder.EPISODE_NUMBER
            )
            
            search_result = await self.search_engine.search(search_request)
            sorted_hits = search_result.get_sorted_hits()
            
            if len(sorted_hits) > 1:
                episode_numbers = [hit.episode_number for hit in sorted_hits]
                if episode_numbers != sorted(episode_numbers):
                    raise Exception(f"Sort by episode number failed: {episode_numbers}")
                print(f"✅ Sort by episode number works correctly: {episode_numbers}")
            
            # Test 4: Context generation
            print("\nTest 4: Context text generation")
            if search_result.hits:
                context_text = search_result.get_context_text()
                if not context_text or len(context_text) < 100:
                    raise Exception("Context text generation failed")
                print(f"✅ Context text generated ({len(context_text)} chars)")
                print(f"   Preview: {context_text[:200]}...")
            
        except Exception as e:
            print(f"❌ Episode search test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def test_performance_metrics(self, episodes: List[EpisodeData]) -> None:
        """성능 메트릭 테스트"""
        print(f"\n📈 Testing performance metrics...")
        
        try:
            # Embedding generation performance
            sample_texts = [ep.content[:500] for ep in episodes[:5]]  # First 500 chars
            
            start_time = time.time()
            for text in sample_texts:
                from src.embedding.base import EmbeddingRequest
                request = EmbeddingRequest(input=[text])
                response = await self.embedding_manager.generate_embeddings_async(request)
                if not response.embeddings:
                    raise Exception("Embedding generation failed")
            
            embedding_time = time.time() - start_time
            avg_embedding_time = embedding_time / len(sample_texts)
            
            print(f"✅ Embedding performance:")
            print(f"   {len(sample_texts)} texts processed in {embedding_time:.3f}s")
            print(f"   Average: {avg_embedding_time:.3f}s per text")
            
            # Search performance with multiple queries
            test_queries = [
                "주인공의 여행",
                "마법과 모험",
                "친구들과의 만남",
                "위험한 상황",
                "결말과 해결"
            ]
            
            search_times = []
            for query in test_queries:
                search_request = EpisodeSearchRequest(query=query, limit=3)
                
                start_time = time.time()
                await self.search_engine.search(search_request)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            print(f"✅ Search performance:")
            print(f"   {len(test_queries)} queries processed")
            print(f"   Average search time: {avg_search_time:.3f}s")
            print(f"   Range: {min(search_times):.3f}s - {max(search_times):.3f}s")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def run_integration_test(self):
        """전체 통합 테스트 실행"""
        print("🚀 Starting Episode RAG Integration Test")
        print("=" * 80)
        
        try:
            # Setup
            await self.setup()
            
            # Test RDB data fetch
            episodes = await self.test_rdb_connection_and_data_fetch()
            
            # Test processing and storage
            await self.test_episode_processing_and_storage(episodes)
            
            # Test search functionality
            await self.test_episode_search_functionality(episodes)
            
            # Test performance
            await self.test_performance_metrics(episodes)
            
            print("\n" + "=" * 80)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("🎉 Episode RAG system is fully functional with real RDB and Milvus")
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """메인 테스트 실행 함수"""
    test = EpisodeRAGIntegrationTest()
    await test.run_integration_test()


if __name__ == "__main__":
    asyncio.run(main())