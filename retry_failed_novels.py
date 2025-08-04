#!/usr/bin/env python3
"""
실패한 소설들만 재처리하는 도구
로그에서 확인된 실패한 소설들을 개별적으로 안전하게 재처리합니다.
"""

import asyncio
import time
from typing import List
from loguru import logger

from src.core.config import get_config
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient
from src.episode.manager import EpisodeRAGManager, EpisodeRAGConfig


async def retry_specific_novels(novel_ids: List[int]):
    """특정 소설들을 재처리합니다."""
    logger.info(f"🔄 {len(novel_ids)}개 소설 재처리 시작: {novel_ids}")
    
    config = get_config()
    
    # Initialize dependencies with more conservative settings
    db_manager = DatabaseManager(config.database)
    
    if config.embedding_providers:
        provider_configs = list(config.embedding_providers.values())
    else:
        provider_configs = [config.embedding]
        
    embedding_manager = EmbeddingManager(provider_configs)
    milvus_client = MilvusClient(config.milvus)
    
    # 매우 보수적인 설정
    episode_config = EpisodeRAGConfig(
        processing_batch_size=2,  # 매우 작은 배치
        vector_dimension=1024
    )
    
    episode_manager = EpisodeRAGManager(
        database_manager=db_manager,
        embedding_manager=embedding_manager,
        milvus_client=milvus_client,
        config=episode_config
    )
    
    # Connect to Milvus
    milvus_client.connect()
    
    success_count = 0
    failed_novels = []
    
    for i, novel_id in enumerate(novel_ids, 1):
        try:
            logger.info(f"📖 Novel {novel_id} 재처리 시작... ({i}/{len(novel_ids)})")
            
            # Provider 헬스체크
            primary_provider = list(embedding_manager.providers.values())[0] if embedding_manager.providers else None
            if primary_provider and hasattr(primary_provider, 'health_check'):
                health = primary_provider.health_check()
                if health.get('status') != 'healthy':
                    logger.warning(f"⚠️ Provider unhealthy for novel {novel_id}, waiting 10s...")
                    await asyncio.sleep(10)
                    
                    # 재확인
                    health = primary_provider.health_check()
                    if health.get('status') != 'healthy':
                        logger.error(f"❌ Provider still unhealthy, skipping novel {novel_id}")
                        failed_novels.append(novel_id)
                        continue
            
            # 소설 처리
            start_time = time.time()
            await episode_manager.process_novel(novel_id)
            processing_time = time.time() - start_time
            
            logger.success(f"✅ Novel {novel_id} 재처리 완료 ({processing_time:.1f}초)")
            success_count += 1
            
            # 소설 간 충분한 대기 시간
            if i < len(novel_ids):
                logger.info(f"다음 소설 처리까지 10초 대기...")
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ Novel {novel_id} 재처리 실패: {e}")
            failed_novels.append(novel_id)
            
            # 에러 발생시 더 긴 대기
            logger.info(f"에러 발생으로 15초 대기...")
            await asyncio.sleep(15)
    
    # 최종 결과
    logger.info(f"🎯 재처리 완료: {success_count}/{len(novel_ids)} 성공")
    
    if failed_novels:
        logger.warning(f"❌ 여전히 실패한 소설들: {failed_novels}")
    else:
        logger.success(f"🎉 모든 소설 재처리 성공!")
    
    return success_count, failed_novels


async def main():
    """메인 재처리 함수"""
    # 로그에서 확인된 실패한 소설들 (예시)
    failed_novel_ids = [
        78,  # Episodes 456, 460 등에서 실패
        # 다른 실패한 소설 IDs를 여기에 추가
    ]
    
    logger.info("🚀 실패한 소설들 재처리 시작")
    logger.info(f"📋 대상 소설: {failed_novel_ids}")
    
    if not failed_novel_ids:
        logger.info("재처리할 소설이 없습니다.")
        return
    
    success_count, still_failed = await retry_specific_novels(failed_novel_ids)
    
    logger.info("📊 최종 결과:")
    logger.info(f"  - 재처리 성공: {success_count}개")
    logger.info(f"  - 여전히 실패: {len(still_failed)}개")
    
    if still_failed:
        logger.info("💡 권장사항:")
        logger.info("  1. Ollama 서버 재시작")
        logger.info("  2. 시스템 리소스 확인")
        logger.info("  3. 수동으로 개별 에피소드 확인")


if __name__ == "__main__":
    asyncio.run(main())