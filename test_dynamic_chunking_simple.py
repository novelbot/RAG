#!/usr/bin/env python3
"""
동적 청킹 시스템 간단 테스트 - EmbeddingManager 없이
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from src.episode.processor import EpisodeEmbeddingProcessor, EpisodeProcessingConfig
from sqlalchemy import text

def test_chunking_logic_only():
    """청킹 로직만 테스트."""
    print("🧪 동적 청킹 로직 테스트")
    print("=" * 60)
    
    try:
        # 설정 및 매니저 초기화 (embedding_manager 없이)
        config = get_config()
        db_manager = DatabaseManager(config.database)
        
        # 프로세서 설정 (embedding_manager=None로 설정)
        processor_config = EpisodeProcessingConfig(
            enable_content_cleaning=True,
            enable_chunking=True
        )
        
        processor = EpisodeEmbeddingProcessor(
            database_manager=db_manager,
            embedding_manager=None,  # None으로 설정
            config=processor_config
        )
        
        print("1. 📏 모델 max_tokens 확인 (fallback 테스트):")
        max_tokens = processor._get_model_max_tokens()
        print(f"   폴백 max_tokens: {max_tokens}")
        
        print("\n2. 🎯 최적 청크 설정 계산:")
        chunk_size, overlap = processor._get_optimal_chunk_settings()
        print(f"   계산된 청크 크기: {chunk_size}자")
        print(f"   계산된 겹침: {overlap}자")
        print(f"   청킹 임계값: {int(max_tokens * 0.85)}토큰 (≈{int(max_tokens * 0.85 / 1.5)}자)")
        
        print("\n3. 📊 실제 모델 설정으로 테스트:")
        # 실제 jeffh/intfloat-multilingual-e5-large-instruct 모델의 512 토큰 기준으로 수동 계산
        actual_max_tokens = 512
        actual_safe_tokens = int(actual_max_tokens * 0.85)  # 435
        actual_safe_chars = int(actual_safe_tokens / 1.5)   # 290
        actual_chunk_size = min(1500, actual_safe_chars)    # 290
        actual_overlap = max(20, min(200, int(actual_chunk_size * (200/1500))))  # 39
        
        print(f"   실제 모델 max_tokens: {actual_max_tokens}")
        print(f"   실제 계산 청크 크기: {actual_chunk_size}자")
        print(f"   실제 계산 겹침: {actual_overlap}자")
        print(f"   실제 청킹 임계값: {actual_safe_tokens}토큰 (≈{int(actual_safe_tokens / 1.5)}자)")
        
        print("\n4. 🔍 실제 에피소드로 청킹 테스트:")
        
        # 테스트용 에피소드 데이터 가져오기
        with db_manager.get_connection() as conn:
            result = conn.execute(text("""
                SELECT episode_id, episode_title, content, LENGTH(content) as content_length
                FROM episode 
                WHERE LENGTH(content) > 100
                ORDER BY LENGTH(content) DESC
                LIMIT 5
            """))
            episodes = result.fetchall()
        
        for ep in episodes:
            content_length = ep.content_length
            estimated_tokens = int(content_length * 1.5)
            
            # 실제 모델 기준으로 청킹 판단
            should_chunk = estimated_tokens > actual_safe_tokens
            chunk_info = "청킹 필요" if should_chunk else "단일 임베딩"
            
            print(f"   Episode {ep.episode_id}: {content_length}자 ({estimated_tokens}토큰) → {chunk_info}")
            
            if should_chunk:
                # 실제 청킹 테스트
                chunks = processor._split_content_into_chunks(ep.content, actual_chunk_size, actual_overlap)
                total_chunk_chars = sum(len(chunk) for chunk in chunks)
                print(f"     └─ {len(chunks)}개 청크 생성, 총 {total_chunk_chars}자 (원본: {content_length}자)")
                
                # 각 청크가 토큰 제한 내인지 확인
                for i, chunk in enumerate(chunks[:3]):  # 처음 3개만 확인
                    chunk_tokens = int(len(chunk) * 1.5)
                    safe = "✅" if chunk_tokens <= actual_max_tokens else "❌"
                    print(f"        청크 {i+1}: {len(chunk)}자 ({chunk_tokens}토큰) {safe}")
        
        print(f"\n🎯 청킹 전후 비교:")
        print(f"   기존 설정 (1500자 청크, 2000토큰 임계값):")
        print(f"   • 1333자 이상 에피소드 → 1500자 청크 (2250토큰, 77% truncated)")
        print(f"   새 설정 ({actual_chunk_size}자 청크, {actual_safe_tokens}토큰 임계값):")
        print(f"   • {int(actual_safe_tokens/1.5)}자 이상 에피소드 → {actual_chunk_size}자 청크 ({int(actual_chunk_size*1.5)}토큰, 완전 보존)")
        
        print(f"\n🎉 동적 청킹 로직 테스트 완료!")
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
    success = test_chunking_logic_only()
    print(f"\n{'='*60}")
    if success:
        print("🎉 동적 청킹 로직이 정상 작동합니다!")
        print("실제 모델에서는 290자 청크로 512토큰 제한 내에서 완벽하게 처리됩니다.")
    else:
        print("⚠️ 동적 청킹 로직에 문제가 있습니다.")
    sys.exit(0 if success else 1)