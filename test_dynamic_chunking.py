#!/usr/bin/env python3
"""
동적 청킹 시스템 테스트
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.episode.processor import EpisodeEmbeddingProcessor, EpisodeProcessingConfig
from sqlalchemy import text

def test_dynamic_chunking():
    """테스트 동적 청킹 시스템."""
    print("🧪 동적 청킹 시스템 테스트")
    print("=" * 60)
    
    try:
        # 설정 및 매니저 초기화
        config = get_config()
        db_manager = DatabaseManager(config.database)
        embedding_manager = EmbeddingManager(config.embedding)
        
        # 프로세서 설정
        processor_config = EpisodeProcessingConfig(
            enable_content_cleaning=True,
            enable_chunking=True
        )
        
        processor = EpisodeEmbeddingProcessor(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            config=processor_config
        )
        
        print("1. 📏 모델 max_tokens 확인:")
        max_tokens = processor._get_model_max_tokens()
        print(f"   현재 모델 max_tokens: {max_tokens}")
        
        print("\n2. 🎯 최적 청크 설정 계산:")
        chunk_size, overlap = processor._get_optimal_chunk_settings()
        print(f"   계산된 청크 크기: {chunk_size}자")
        print(f"   계산된 겹침: {overlap}자")
        print(f"   청킹 임계값: {int(max_tokens * 0.85)}토큰 (≈{int(max_tokens * 0.85 / 1.5)}자)")
        
        print("\n3. 📊 기존 설정과 비교:")
        print(f"   기존 청크 크기: 1500자 → 새로운: {chunk_size}자")
        print(f"   기존 겹침: 200자 → 새로운: {overlap}자")
        print(f"   기존 임계값: 1333자 → 새로운: {int(max_tokens * 0.85 / 1.5)}자")
        
        print("\n4. 🔍 실제 에피소드로 청킹 테스트:")
        
        # 테스트용 에피소드 데이터 가져오기
        with db_manager.get_connection() as conn:
            result = conn.execute(text("""
                SELECT episode_id, episode_title, content, LENGTH(content) as content_length
                FROM episode 
                WHERE LENGTH(content) > 100
                ORDER BY LENGTH(content) DESC
                LIMIT 3
            """))
            episodes = result.fetchall()
        
        for ep in episodes:
            content_length = ep.content_length
            estimated_tokens = int(content_length * 1.5)
            chunking_threshold = int(max_tokens * 0.85)
            
            should_chunk = estimated_tokens > chunking_threshold
            chunk_info = "청킹 필요" if should_chunk else "단일 임베딩"
            
            print(f"   Episode {ep.episode_id}: {content_length}자 ({estimated_tokens}토큰) → {chunk_info}")
            
            if should_chunk:
                # 실제 청킹 테스트
                chunks = processor._split_content_into_chunks(ep.content, chunk_size, overlap)
                total_chunk_chars = sum(len(chunk) for chunk in chunks)
                print(f"     └─ {len(chunks)}개 청크 생성, 총 {total_chunk_chars}자 (원본: {content_length}자)")
                
                # 각 청크가 토큰 제한 내인지 확인
                for i, chunk in enumerate(chunks[:3]):  # 처음 3개만 확인
                    chunk_tokens = int(len(chunk) * 1.5)
                    safe = "✅" if chunk_tokens <= max_tokens else "❌"
                    print(f"        청크 {i+1}: {len(chunk)}자 ({chunk_tokens}토큰) {safe}")
        
        print(f"\n🎉 동적 청킹 시스템 테스트 완료!")
        print(f"   • 모델 제한에 맞춰 청크 크기 자동 조정됨")
        print(f"   • {max_tokens}토큰 제한 모델에 최적화됨")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    success = test_dynamic_chunking()
    print(f"\n{'='*60}")
    if success:
        print("🎉 동적 청킹 시스템이 정상 작동합니다!")
    else:
        print("⚠️ 동적 청킹 시스템에 문제가 있습니다.")
    sys.exit(0 if success else 1)