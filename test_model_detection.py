#!/usr/bin/env python3
"""
모델 max_tokens 감지 테스트
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.episode.processor import EpisodeEmbeddingProcessor, EpisodeProcessingConfig

def test_model_detection():
    """모델 max_tokens 감지 테스트."""
    print("🔍 모델 max_tokens 감지 테스트")
    print("=" * 60)
    
    try:
        # 설정 및 매니저 초기화
        config = get_config()
        db_manager = DatabaseManager(config.database)
        
        print("1. 📋 설정 정보:")
        print(f"   임베딩 제공자: {config.embedding.provider}")
        print(f"   모델: {config.embedding.model}")
        print(f"   Base URL: {config.embedding.base_url}")
        
        # EmbeddingManager 초기화 (문제가 있는 부분)
        print("\n2. 🤖 EmbeddingManager 초기화 시도:")
        try:
            embedding_manager = EmbeddingManager(config.embedding)
            print("   ✅ EmbeddingManager 초기화 성공")
            
            print(f"   제공자 수: {len(embedding_manager.providers)}")
            for name, provider in embedding_manager.providers.items():
                print(f"   제공자: {name}, 타입: {type(provider).__name__}")
                
                # Provider 속성 확인
                if hasattr(provider, 'model'):
                    print(f"     모델: {provider.model}")
                if hasattr(provider, 'config'):
                    print(f"     설정: {provider.config}")
                    if hasattr(provider.config, 'model'):
                        print(f"     설정 모델: {provider.config.model}")
                if hasattr(provider, 'MODEL_SPECS'):
                    print(f"     MODEL_SPECS 있음: {list(provider.MODEL_SPECS.keys())}")
                    
        except Exception as e:
            print(f"   ❌ EmbeddingManager 초기화 실패: {e}")
            embedding_manager = None
        
        # 프로세서 테스트
        print("\n3. 🔧 프로세서로 모델 감지 테스트:")
        processor_config = EpisodeProcessingConfig()
        processor = EpisodeEmbeddingProcessor(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            config=processor_config
        )
        
        max_tokens = processor._get_model_max_tokens()
        print(f"   감지된 max_tokens: {max_tokens}")
        
        chunk_size, overlap = processor._get_optimal_chunk_settings()
        print(f"   계산된 청크 크기: {chunk_size}자")
        print(f"   계산된 겹침: {overlap}자")
        
        # 수동으로 Ollama MODEL_CONFIGS 확인
        print("\n4. 📊 Ollama MODEL_CONFIGS 직접 확인:")
        from src.embedding.providers.ollama import OllamaEmbeddingProvider
        model_name = config.embedding.model
        
        if model_name in OllamaEmbeddingProvider.MODEL_CONFIGS:
            specs = OllamaEmbeddingProvider.MODEL_CONFIGS[model_name]
            print(f"   모델 {model_name} 스펙:")
            for key, value in specs.items():
                print(f"     {key}: {value}")
        else:
            print(f"   모델 {model_name}이 MODEL_CONFIGS에 없음")
            print(f"   사용 가능한 모델: {list(OllamaEmbeddingProvider.MODEL_CONFIGS.keys())}")
        
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
    success = test_model_detection()
    print(f"\n{'='*60}")
    if success:
        print("🎉 모델 감지 테스트 완료!")
    else:
        print("⚠️ 모델 감지에 문제가 있습니다.")
    sys.exit(0 if success else 1)