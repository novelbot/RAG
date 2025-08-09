#!/usr/bin/env python3
"""
OpenAI 임베딩 모델의 max_tokens 제한 테스트
실제 API 호출 없이 설정값만 확인
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embedding.types import EmbeddingConfig, EmbeddingProvider
from src.embedding.langchain_embeddings import LangChainEmbeddingProvider
from src.embedding.manager import EmbeddingManager, EmbeddingProviderConfig
from src.episode.processor import EpisodeEmbeddingProcessor
from src.database.base import DatabaseManager
from src.core.config import DatabaseConfig, DatabaseType

def test_openai_max_tokens():
    """Test OpenAI embedding models max_tokens configuration"""
    
    print("=" * 80)
    print("OpenAI 임베딩 모델 max_tokens 제한 테스트")
    print("=" * 80)
    
    # Test models
    test_models = [
        "text-embedding-ada-002",
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ]
    
    for model in test_models:
        print(f"\n📊 테스트 모델: {model}")
        print("-" * 40)
        
        # Create config for OpenAI
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model=model,
            api_key="test-key-not-used"  # 실제 API 호출 안함
        )
        
        try:
            # Create provider (API 호출 없이 초기화만)
            provider = LangChainEmbeddingProvider(config)
            
            # Get model info
            model_info = provider.get_model_info()
            
            print(f"✅ Provider: {model_info.get('provider')}")
            print(f"✅ Model: {model_info.get('model')}")
            print(f"✅ Max Tokens: {model_info.get('max_tokens', 'NOT SET')}")
            print(f"✅ Dimensions: {model_info.get('dimensions')}")
            
            # Expected values
            expected_max_tokens = 8191
            actual_max_tokens = model_info.get('max_tokens', 0)
            
            if actual_max_tokens == expected_max_tokens:
                print(f"✅ Max tokens 올바르게 설정됨: {actual_max_tokens}")
            else:
                print(f"❌ Max tokens 불일치: 예상={expected_max_tokens}, 실제={actual_max_tokens}")
                
        except Exception as e:
            print(f"⚠️  Provider 초기화 중 오류 (API 키 없어서 정상): {e}")
            print("   max_tokens 확인을 위해 직접 접근...")
            
            # Direct check without provider initialization
            from src.embedding.langchain_embeddings import LangChainEmbeddingProvider
            max_tokens = LangChainEmbeddingProvider.MODEL_MAX_TOKENS.get(model, "NOT FOUND")
            print(f"   📏 MODEL_MAX_TOKENS['{model}'] = {max_tokens}")
    
    print("\n" + "=" * 80)
    print("청킹 임계값 계산 테스트")
    print("=" * 80)
    
    # Test chunking threshold calculation
    for model, max_tokens in [
        ("text-embedding-ada-002", 8191),
        ("ollama-model", 2048)
    ]:
        threshold = int(max_tokens * 0.85)
        char_limit = int(threshold / 1.5)  # Korean text estimation
        
        print(f"\n📊 {model}:")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   청킹 임계값 (85%): {threshold} tokens")
        print(f"   문자 수 제한 (한국어): ~{char_limit} 글자")
        print(f"   → {char_limit}자 이하: 단일 임베딩")
        print(f"   → {char_limit}자 초과: 자동 청킹")

def test_processor_integration():
    """Test processor's ability to get max_tokens from different providers"""
    
    print("\n" + "=" * 80)
    print("EpisodeProcessor와 통합 테스트")
    print("=" * 80)
    
    # Mock database config
    db_config = DatabaseConfig(
        database_type=DatabaseType.SQLITE,
        database="test.db"
    )
    
    # Create mock database manager
    db_manager = DatabaseManager(db_config)
    
    # Test with OpenAI provider
    openai_config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-3-small",
        api_key="test-key"
    )
    
    provider_config = EmbeddingProviderConfig(
        provider=EmbeddingProvider.OPENAI,
        config=openai_config
    )
    
    try:
        # Create embedding manager
        embedding_manager = EmbeddingManager([provider_config], enable_cache=False)
        
        # Create processor
        processor = EpisodeEmbeddingProcessor(db_manager, embedding_manager)
        
        # Get max tokens from processor
        max_tokens = processor._get_model_max_tokens()
        
        print(f"\n✅ Processor가 인식한 max_tokens: {max_tokens}")
        
        if max_tokens == 8191:
            print("✅ OpenAI 모델의 max_tokens가 올바르게 인식됨!")
        else:
            print(f"❌ 예상값(8191)과 다름: {max_tokens}")
            
    except Exception as e:
        print(f"⚠️  통합 테스트 중 오류 (API 키 없어서 정상): {e}")

if __name__ == "__main__":
    test_openai_max_tokens()
    test_processor_integration()
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)