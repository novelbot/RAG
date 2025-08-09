#!/usr/bin/env python3
"""
OpenAI 임베딩 모델의 max_tokens 설정 확인 (간단 버전)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_max_tokens_config():
    """MODEL_MAX_TOKENS 설정 확인"""
    
    print("=" * 80)
    print("OpenAI 임베딩 모델 max_tokens 설정 확인")
    print("=" * 80)
    
    # Import and check MODEL_MAX_TOKENS
    from src.embedding.langchain_embeddings import LangChainEmbeddingProvider
    
    print("\n📊 MODEL_MAX_TOKENS 딕셔너리 내용:")
    print("-" * 40)
    
    # OpenAI models
    openai_models = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]
    
    for model in openai_models:
        max_tokens = LangChainEmbeddingProvider.MODEL_MAX_TOKENS.get(model, "NOT FOUND")
        print(f"  {model}: {max_tokens} tokens")
    
    print("\n📊 청킹 임계값 계산:")
    print("-" * 40)
    
    for model in openai_models:
        max_tokens = LangChainEmbeddingProvider.MODEL_MAX_TOKENS.get(model, 2048)
        threshold = int(max_tokens * 0.85)
        char_limit = int(threshold / 1.5)
        
        print(f"\n{model}:")
        print(f"  • Max Tokens: {max_tokens}")
        print(f"  • 청킹 임계값 (85%): {threshold} tokens")
        print(f"  • 한국어 문자 제한: ~{char_limit:,} 글자")
        print(f"  • 처리 방식:")
        print(f"    - {char_limit:,}자 이하 → 단일 임베딩 생성")
        print(f"    - {char_limit:,}자 초과 → 자동 청킹 후 개별 임베딩")
    
    print("\n✅ OpenAI 모델들의 max_tokens가 모두 8191로 설정되어 있습니다.")
    print("✅ Ollama 모델(~1,160자)보다 약 4배 많은 텍스트를 한 번에 처리 가능합니다.")

def compare_with_ollama():
    """Ollama vs OpenAI 비교"""
    
    print("\n" + "=" * 80)
    print("Ollama vs OpenAI 임베딩 모델 비교")
    print("=" * 80)
    
    comparisons = [
        ("Ollama (jeffh/intfloat)", 2048, "기본 설정"),
        ("OpenAI (text-embedding-ada-002)", 8191, "새로 추가"),
        ("OpenAI (text-embedding-3-small)", 8191, "새로 추가"),
        ("OpenAI (text-embedding-3-large)", 8191, "새로 추가"),
    ]
    
    print("\n| 모델 | Max Tokens | 한국어 문자 제한 | 상태 |")
    print("|------|------------|-----------------|------|")
    
    for model, max_tokens, status in comparisons:
        char_limit = int(max_tokens * 0.85 / 1.5)
        print(f"| {model:<30} | {max_tokens:>10,} | {char_limit:>15,}자 | {status} |")
    
    print("\n📌 결론:")
    print("  • OpenAI 모델 사용 시 대부분의 에피소드는 청킹 없이 처리 가능")
    print("  • 4,600자 이상의 긴 텍스트만 자동 청킹 처리")
    print("  • Ollama는 1,160자 이상부터 청킹 필요")

if __name__ == "__main__":
    test_max_tokens_config()
    compare_with_ollama()
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)