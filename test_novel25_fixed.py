#!/usr/bin/env python3
"""
수정된 로직으로 Novel 25 (13화) 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.episode.manager import create_episode_rag_manager
from src.core.config import get_config
from src.database.base import DatabaseManager
from src.embedding.factory import get_embedding_manager
from src.embedding.types import EmbeddingConfig, EmbeddingProvider
from src.milvus.client import MilvusClient
import asyncio

async def test_novel25():
    """Novel 25로 수정된 로직 테스트"""
    try:
        # 설정 로드
        config = get_config()
        
        # 필요한 매니저들 초기화
        print("🚀 초기화 중...")
        db_manager = DatabaseManager(config.database)
        
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model="jeffh/intfloat-multilingual-e5-large-instruct:f32",
            base_url="http://localhost:11434"
        )
        embedding_manager = get_embedding_manager([embedding_config])
        
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        # EpisodeRAGManager 초기화 (컬렉션 자동 설정)
        episode_manager = await create_episode_rag_manager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            setup_collection=True  # 컬렉션 자동 설정
        )
        
        # Novel 25 처리 (이전에 실패했던 소설)
        novel_id = 25
        print(f"🎯 Novel {novel_id} 처리 시작...")
        
        result = await episode_manager.process_novel(
            novel_id=novel_id,
            force_reprocess=True
        )
        
        print(f"📊 처리 결과: {result}")
        
        if result.get('status') == 'success':
            print(f"✅ Novel {novel_id}: {result.get('episodes_processed', 0)}개 에피소드 처리 완료")
            return True
        else:
            print(f"❌ Novel {novel_id}: 처리 실패 - {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"💥 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Novel 25 (13화) 수정된 로직 테스트")
    print("   - 에피소드 단위 개별 처리")
    print("   - 2000토큰 초과시 자동 청킹")
    print()
    
    success = asyncio.run(test_novel25())
    
    if success:
        print("🎊 Novel 25 처리 성공!")
    else:
        print("💥 Novel 25 처리 실패!")
        sys.exit(1)