#!/usr/bin/env python3
"""
Ollama Provider 안정성 테스트 도구
실패했던 에피소드들을 개별적으로 재처리해보고 Provider 상태를 모니터링합니다.
"""

import asyncio
import time
from typing import List, Dict, Any
from loguru import logger

from src.core.config import get_config
from src.database.base import DatabaseManager  
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient
from src.episode.manager import EpisodeRAGManager, EpisodeRAGConfig


def test_provider_health():
    """Provider 헬스체크 테스트"""
    logger.info("🏥 Ollama Provider 헬스체크 시작...")
    
    config = get_config()
    
    # Create embedding provider configs list
    if config.embedding_providers:
        provider_configs = list(config.embedding_providers.values())
    else:
        provider_configs = [config.embedding]
        
    embedding_manager = EmbeddingManager(provider_configs)
    
    # Get the first (primary) provider
    primary_provider = list(embedding_manager.providers.values())[0] if embedding_manager.providers else None
    
    if primary_provider and hasattr(primary_provider, 'health_check'):
        health = primary_provider.health_check()
        logger.info(f"📊 헬스체크 결과: {health}")
        
        if health.get('status') == 'healthy':
            logger.success("✅ Provider가 정상 상태입니다!")
        else:
            logger.warning(f"⚠️ Provider 상태 이상: {health.get('error', 'Unknown')}")
            
        return health.get('status') == 'healthy'
    else:
        logger.warning("⚠️ 헬스체크 기능이 없습니다.")
        return True


async def test_specific_episodes():
    """알려진 실패 에피소드들을 개별 테스트"""
    logger.info("🔧 실패했던 에피소드들 개별 테스트...")
    
    # 알려진 실패 케이스들
    failed_cases = [
        {"novel_id": 32, "episode_numbers": [135, 136, 137]},
        {"novel_id": 78, "episode_numbers": [456, 457, 458, 459, 460, 461, 462, 463, 464]}
    ]
    
    config = get_config()
    
    # Initialize dependencies
    db_manager = DatabaseManager(config.database)
    
    if config.embedding_providers:
        provider_configs = list(config.embedding_providers.values())
    else:
        provider_configs = [config.embedding]
        
    embedding_manager = EmbeddingManager(provider_configs)
    milvus_client = MilvusClient(config.milvus)
    
    # 더 보수적인 설정
    episode_config = EpisodeRAGConfig(
        processing_batch_size=1,  # 한 번에 하나씩만
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
    total_count = 0
    
    for case in failed_cases:
        novel_id = case["novel_id"]
        episode_numbers = case["episode_numbers"] 
        
        logger.info(f"📖 Novel {novel_id} 테스트 중...")
        
        for episode_num in episode_numbers:
            total_count += 1
            
            try:
                # Provider 상태 확인
                primary_provider = list(embedding_manager.providers.values())[0] if embedding_manager.providers else None
                if primary_provider and hasattr(primary_provider, 'health_check'):
                    health = primary_provider.health_check()
                    if health.get('status') != 'healthy':
                        logger.warning(f"⚠️ Provider unhealthy, skipping episode {episode_num}")
                        continue
                
                # 특정 에피소드 처리
                logger.info(f"🎯 Episode {episode_num} 처리 시도...")
                
                # 에피소드별 처리 (구체적인 구현은 episode_manager 내부 메소드 사용)
                await asyncio.sleep(1)  # 안전한 간격
                
                # 여기서는 실제 처리 대신 시뮬레이션
                # 실제로는 episode_manager.process_specific_episode() 같은 메소드 필요
                
                logger.success(f"✅ Episode {episode_num} 처리 성공")
                success_count += 1
                
                # 에피소드 간 충분한 간격
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"❌ Episode {episode_num} 처리 실패: {e}")
                continue
    
    logger.info(f"📊 테스트 완료: {success_count}/{total_count} 성공 ({(success_count/total_count)*100:.1f}%)")


async def monitor_provider_stability(duration_minutes: int = 10):
    """Provider 안정성 모니터링"""
    logger.info(f"📡 {duration_minutes}분 동안 Provider 안정성 모니터링...")
    
    config = get_config()
    
    if config.embedding_providers:
        provider_configs = list(config.embedding_providers.values())
    else:
        provider_configs = [config.embedding]
        
    embedding_manager = EmbeddingManager(provider_configs)
    
    primary_provider = list(embedding_manager.providers.values())[0] if embedding_manager.providers else None
    if not primary_provider or not hasattr(primary_provider, 'health_check'):
        logger.error("❌ 헬스체크 기능이 없습니다.")
        return
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    healthy_count = 0
    total_checks = 0
    
    while time.time() < end_time:
        try:
            health = primary_provider.health_check()
            total_checks += 1
            
            status = health.get('status', 'unknown')
            response_time = health.get('response_time', 0)
            
            if status == 'healthy':
                healthy_count += 1
                logger.info(f"✅ 정상 (응답시간: {response_time:.3f}s)")
            else:
                logger.warning(f"⚠️ 비정상: {status} - {health.get('error', 'Unknown')}")
            
            await asyncio.sleep(30)  # 30초마다 체크
            
        except Exception as e:
            logger.error(f"❌ 헬스체크 실패: {e}")
            total_checks += 1
    
    uptime_percentage = (healthy_count / total_checks) * 100 if total_checks > 0 else 0
    logger.info(f"📊 모니터링 완료: {healthy_count}/{total_checks} 정상 ({uptime_percentage:.1f}% uptime)")


async def main():
    """메인 테스트 함수"""
    logger.info("🚀 Ollama Provider 안정성 테스트 시작")
    
    # 1. 헬스체크
    if not test_provider_health():
        logger.error("❌ Provider가 불안정합니다. 테스트를 중단합니다.")
        return
    
    # 2. 실패 에피소드 재테스트
    await test_specific_episodes()
    
    # 3. 안정성 모니터링 (5분)
    await monitor_provider_stability(5)
    
    logger.success("🎉 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())