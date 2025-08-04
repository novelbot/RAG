#!/usr/bin/env python3
"""
수정된 로직으로 전체 소설 임베딩 재처리
- 에피소드 단위 개별 처리
- 긴 에피소드 자동 청킹
- 토큰 제한 초과 방지
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
from sqlalchemy import text
import asyncio

async def process_all_novels():
    """수정된 로직으로 전체 소설 처리"""
    try:
        # 설정 로드
        config = get_config()
        
        # 필요한 매니저들 초기화
        print("🚀 데이터베이스 매니저 초기화 중...")
        db_manager = DatabaseManager(config.database)
        
        print("🚀 임베딩 매니저 초기화 중...")
        # Ollama E5 임베딩 설정
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model="nomic-embed-text",  # 또는 사용 중인 모델
            base_url="http://localhost:11434"
        )
        embedding_manager = get_embedding_manager([embedding_config])
        
        print("🚀 Milvus 클라이언트 초기화 중...")
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        # EpisodeRAGManager 초기화
        print("🚀 EpisodeRAGManager 초기화 중...")
        episode_manager = await create_episode_rag_manager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            setup_collection=False  # 이미 컬렉션이 생성되어 있음
        )
        
        # 기존에 확인된 소설 ID 목록 (1~67)
        print("📚 처리할 소설 목록 준비 중...")
        novel_ids = list(range(1, 68))  # 1부터 67까지
        
        # 소설 정보를 튜플 형태로 준비 (기존 코드 호환을 위해)
        novels = [(novel_id, f"Novel {novel_id}", 0) for novel_id in novel_ids]
        total_novels = len(novels)
        
        print(f"📊 처리할 소설 수: {total_novels}개")
        print("=" * 50)
        
        success_count = 0
        failed_novels = []
        
        for i, novel in enumerate(novels, 1):
            novel_id = novel[0]  # novel.id
            novel_title = novel[1]  # novel.title
            episode_count = novel[2]  # episode count
            
            print(f"\n🎯 [{i}/{total_novels}] 처리 중: {novel_title} (ID: {novel_id}, 에피소드: {episode_count}개)")
            
            try:
                # 해당 소설의 에피소드 처리 (async 메서드 호출)
                result = await episode_manager.process_novel(
                    novel_id=novel_id,
                    force_reprocess=True  # 기존 데이터 재처리
                )
                
                if result.get('status') == 'success':
                    processed_count = result.get('episodes_processed', 0)
                    print(f"✅ {novel_title}: {processed_count}개 에피소드 처리 완료")
                    success_count += 1
                else:
                    error_msg = result.get('message', 'Unknown error')
                    print(f"❌ {novel_title}: 처리 실패 - {error_msg}")
                    failed_novels.append({'id': novel_id, 'title': novel_title, 'error': error_msg})
                
            except Exception as e:
                print(f"💥 {novel_title}: 예외 발생 - {e}")
                failed_novels.append({'id': novel_id, 'title': novel_title, 'error': str(e)})
            
            # 진행률 표시
            progress = (i / total_novels) * 100
            print(f"📊 전체 진행률: {progress:.1f}% ({success_count} 성공, {len(failed_novels)} 실패)")
        
        # 최종 결과 요약
        print("\n" + "=" * 50)
        print("🎉 전체 처리 완료!")
        print(f"📊 최종 결과:")
        print(f"   - 총 소설 수: {total_novels}")
        print(f"   - 성공: {success_count}")
        print(f"   - 실패: {len(failed_novels)}")
        print(f"   - 성공률: {(success_count/total_novels)*100:.1f}%")
        
        if failed_novels:
            print(f"\n❌ 실패한 소설들:")
            for novel in failed_novels:
                print(f"   - {novel['title']} (ID: {novel['id']}): {novel['error']}")
        
        print("=" * 50)
        
        return success_count == total_novels
        
    except Exception as e:
        print(f"💥 전체 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 수정된 로직으로 전체 소설 임베딩 처리 시작")
    print("   - 에피소드 단위 개별 처리")
    print("   - 2000토큰 초과시 자동 청킹")
    print("   - Milvus 벡터DB 저장")
    print()
    
    success = asyncio.run(process_all_novels())
    
    if success:
        print("🎊 모든 소설 처리 성공!")
    else:
        print("⚠️ 일부 소설 처리에 실패했습니다.")
        sys.exit(1)