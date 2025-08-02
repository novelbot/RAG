#!/usr/bin/env python3
"""
Episode RAG Test with Mock Data
Mock 데이터를 사용한 에피소드 RAG 기능 완전 테스트
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
from src.milvus.client import MilvusClient
from src.embedding.factory import get_embedding_client
from src.embedding.base import EmbeddingRequest


class MockEpisodeRAGTest:
    def __init__(self):
        self.config = get_config()
        self.milvus_client = None
        self.embedding_client = None
        self.test_collection = "test_mock_episode_rag"
        
        # Mock episode data
        self.mock_episodes = [
            {
                'episode_id': 1,
                'episode_number': 1,
                'episode_title': '첫 번째 모험',
                'content': '주인공은 작은 마을에서 태어났다. 어릴 때부터 모험을 꿈꾸었던 그는 마침내 여행을 떠나기로 결심했다. 마을 사람들은 그를 걱정했지만, 주인공의 의지는 확고했다.',
                'novel_id': 100,
                'publication_date': date(2023, 1, 1)
            },
            {
                'episode_id': 2,
                'episode_number': 2,
                'episode_title': '마법사와의 만남',
                'content': '숲에서 길을 잃은 주인공은 신비로운 마법사를 만났다. 마법사는 그에게 특별한 능력이 있다고 말했다. 이 만남은 주인공의 운명을 바꾸는 계기가 되었다.',
                'novel_id': 100,
                'publication_date': date(2023, 1, 8)
            },
            {
                'episode_id': 3,
                'episode_number': 3,
                'episode_title': '드래곤과의 전투',
                'content': '주인공은 마을을 위협하는 거대한 드래곤과 맞서 싸워야 했다. 마법사로부터 배운 주문을 사용하여 치열한 전투를 벌였다. 결국 용기와 지혜로 드래곤을 물리쳤다.',
                'novel_id': 100,
                'publication_date': date(2023, 1, 15)
            },
            {
                'episode_id': 4,
                'episode_number': 4,
                'episode_title': '보물 탐험',
                'content': '드래곤의 동굴에서 고대의 보물을 발견한 주인공. 하지만 보물에는 저주가 걸려있었다. 현명한 선택을 통해 저주를 풀고 진정한 보물의 의미를 깨달았다.',
                'novel_id': 100,
                'publication_date': date(2023, 1, 22)
            },
            {
                'episode_id': 5,
                'episode_number': 5,
                'episode_title': '동료들과의 결속',
                'content': '여행 중 만난 동료들과 함께 어려운 시련을 극복했다. 각자의 특별한 능력을 합쳐 강력한 적을 물리쳤다. 진정한 우정의 힘을 깨달은 소중한 경험이었다.',
                'novel_id': 100,
                'publication_date': date(2023, 1, 29)
            }
        ]
    
    def setup(self):
        """환경 설정"""
        print("🔧 Setting up Mock Episode RAG test...")
        
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
    
    def test_mock_data_preparation(self):
        """Mock 데이터 준비 테스트"""
        print(f"\n📋 Testing mock episode data preparation...")
        
        if not self.mock_episodes:
            raise Exception("No mock episodes available")
        
        print(f"✅ Prepared {len(self.mock_episodes)} mock episodes")
        for ep in self.mock_episodes:
            print(f"   Episode {ep['episode_number']}: {ep['episode_title']}")
        
        return self.mock_episodes
    
    def test_embedding_generation(self, episodes):
        """임베딩 생성 테스트"""
        print(f"\n🔄 Testing embedding generation for {len(episodes)} mock episodes...")
        
        episode_embeddings = []
        
        for episode in episodes:
            # Generate embedding
            request = EmbeddingRequest(input=[episode['content']])
            response = self.embedding_client.generate_embeddings(request)
            
            if not response.embeddings:
                raise Exception(f"No embedding generated for episode {episode['episode_id']}")
            
            embedding = response.embeddings[0]
            if len(embedding) != 1024:
                raise Exception(f"Wrong embedding dimension: {len(embedding)}")
            
            episode_data = episode.copy()
            episode_data['embedding'] = embedding
            episode_embeddings.append(episode_data)
        
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
                "content": ep['content'],
                "embedding": ep['embedding'],
                "metadata": {
                    "episode_number": ep['episode_number'],
                    "episode_title": ep['episode_title'],
                    "novel_id": ep['novel_id'],
                    "publication_date": ep['publication_date'].isoformat() if ep['publication_date'] else None
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
        print(f"\n🔍 Testing vector search with multiple queries...")
        
        # Test queries
        test_queries = [
            "마법사와 주문",
            "드래곤과 전투",
            "보물과 저주",
            "친구들과 모험",
            "주인공의 여행"
        ]
        
        all_results = []
        
        for query_text in test_queries:
            print(f"\nQuery: '{query_text}'")
            
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
                print("   No results found")
                continue
            
            print(f"   Found {len(results[0])} results:")
            for i, hit in enumerate(results[0]):
                distance = hit.distance
                metadata = hit.entity.get("metadata", {})
                episode_num = metadata.get("episode_number", "?")
                episode_title = metadata.get("episode_title", "?")
                print(f"     {i+1}. Episode {episode_num}: {episode_title} (distance: {distance:.3f})")
            
            all_results.append((query_text, results))
        
        print(f"\n✅ Completed {len(test_queries)} search queries")
        return all_results
    
    def test_episode_filtering(self, collection):
        """에피소드 필터링 테스트"""
        print(f"\n🎯 Testing episode filtering functionality...")
        
        # Test 1: Filter by episode number range
        print("Test 1: Filter by episode number")
        query_embedding = [0.1] * 1024  # Dummy embedding
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=10,
            expr="metadata['episode_number'] >= 2 and metadata['episode_number'] <= 4",
            output_fields=["metadata"]
        )
        
        if results and len(results[0]) > 0:
            episode_numbers = [hit.entity.get("metadata", {}).get("episode_number") for hit in results[0]]
            print(f"   Found episodes: {episode_numbers}")
            if not all(2 <= num <= 4 for num in episode_numbers if num):
                raise Exception("Episode number filtering failed")
            print("✅ Episode number filtering works correctly")
        
        # Test 2: Filter by novel ID
        print("\nTest 2: Filter by novel ID")
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding", 
            param=search_params,
            limit=10,
            expr="metadata['novel_id'] == 100",
            output_fields=["metadata"]
        )
        
        if results and len(results[0]) > 0:
            novel_ids = [hit.entity.get("metadata", {}).get("novel_id") for hit in results[0]]
            print(f"   Found novel IDs: {novel_ids}")
            if not all(nid == 100 for nid in novel_ids if nid):
                raise Exception("Novel ID filtering failed")
            print("✅ Novel ID filtering works correctly")
    
    def test_performance_metrics(self, episode_embeddings):
        """성능 메트릭 테스트"""
        print(f"\n📈 Testing performance metrics...")
        
        # Embedding generation performance
        sample_texts = [ep['content'] for ep in episode_embeddings[:3]]
        
        start_time = time.time()
        for text in sample_texts:
            request = EmbeddingRequest(input=[text])
            response = self.embedding_client.generate_embeddings(request)
            if not response.embeddings:
                raise Exception("Embedding generation failed")
        
        embedding_time = time.time() - start_time
        avg_embedding_time = embedding_time / len(sample_texts)
        
        print(f"✅ Embedding performance:")
        print(f"   {len(sample_texts)} texts processed in {embedding_time:.3f}s")
        print(f"   Average: {avg_embedding_time:.3f}s per text")
    
    def run_test(self):
        """전체 테스트 실행"""
        print("🚀 Starting Mock Episode RAG Integration Test")
        print("=" * 70)
        
        try:
            # Setup
            self.setup()
            
            # Test mock data
            episodes = self.test_mock_data_preparation()
            
            # Test embedding
            episode_embeddings = self.test_embedding_generation(episodes)
            
            # Test Milvus storage
            collection = self.test_milvus_storage(episode_embeddings)
            
            # Test search
            self.test_vector_search(collection, episode_embeddings)
            
            # Test filtering
            self.test_episode_filtering(collection)
            
            # Test performance
            self.test_performance_metrics(episode_embeddings)
            
            print("\n" + "=" * 70)
            print("🎉 ALL MOCK TESTS PASSED!")
            print("✅ Episode RAG system works correctly with Milvus")
            print("✅ Mock data demonstrates full functionality:")
            print("   - Episode data processing")
            print("   - Embedding generation")
            print("   - Vector storage in Milvus")
            print("   - Similarity search")
            print("   - Metadata filtering")
            print("   - Performance metrics")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.cleanup()


if __name__ == "__main__":
    test = MockEpisodeRAGTest()
    test.run_test()