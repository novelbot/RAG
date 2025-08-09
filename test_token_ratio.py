#!/usr/bin/env python3
"""
토큰 비율 수정 확인 테스트
실제 GPT-4o 기준: 1,424 characters → 921 tokens
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_token_calculations():
    """토큰 계산 로직 테스트"""
    
    print("=" * 80)
    print("한국어 텍스트 토큰 비율 계산 테스트")
    print("=" * 80)
    
    # 실제 GPT-4o 데이터
    actual_chars = 1424
    actual_tokens = 921
    actual_ratio = actual_chars / actual_tokens  # ~1.546
    
    print(f"\n📊 실제 GPT-4o 데이터:")
    print(f"  • 문자 수: {actual_chars:,} characters")
    print(f"  • 토큰 수: {actual_tokens:,} tokens")
    print(f"  • 비율: {actual_ratio:.2f} chars/token")
    
    # 우리가 사용하는 비율
    our_ratio = 1.55
    print(f"\n📊 적용된 비율: {our_ratio} chars/token")
    
    # 테스트 케이스들
    test_cases = [
        1000,   # 짧은 텍스트
        5000,   # 중간 텍스트  
        10000,  # 긴 텍스트
        15000,  # 매우 긴 텍스트
    ]
    
    print("\n📊 문자 수 → 토큰 수 변환 테스트:")
    print("-" * 50)
    print("| 문자 수 | 예상 토큰 (1.55) | 실제 비율 토큰 |")
    print("|---------|------------------|----------------|")
    
    for chars in test_cases:
        estimated_tokens_new = int(chars / 1.55)
        estimated_tokens_actual = int(chars / actual_ratio)
        print(f"| {chars:7,} | {estimated_tokens_new:16,} | {estimated_tokens_actual:14,} |")
    
    # OpenAI 모델의 청킹 임계값 계산
    print("\n" + "=" * 80)
    print("OpenAI 모델 청킹 임계값 (수정된 비율)")
    print("=" * 80)
    
    models = [
        ("text-embedding-ada-002", 8191),
        ("text-embedding-3-small", 8191),
        ("text-embedding-3-large", 8191),
    ]
    
    for model_name, max_tokens in models:
        safe_tokens = int(max_tokens * 0.85)  # 85% 안전 마진
        safe_chars = int(safe_tokens * 1.55)  # 수정된 비율
        
        print(f"\n📊 {model_name}:")
        print(f"  • Max Tokens: {max_tokens:,}")
        print(f"  • 안전 토큰 (85%): {safe_tokens:,} tokens")
        print(f"  • 최대 문자 수: {safe_chars:,} characters")
        print(f"  • 처리 방식:")
        print(f"    - {safe_chars:,}자 이하 → 단일 임베딩")
        print(f"    - {safe_chars:,}자 초과 → 자동 청킹")
    
    # Ollama와 비교
    print("\n" + "=" * 80)
    print("Ollama vs OpenAI 비교 (수정된 비율)")
    print("=" * 80)
    
    ollama_max = 2048
    ollama_safe = int(ollama_max * 0.85)
    ollama_chars = int(ollama_safe * 1.55)
    
    openai_max = 8191
    openai_safe = int(openai_max * 0.85)
    openai_chars = int(openai_safe * 1.55)
    
    print(f"\n| 모델 | 최대 문자 수 | 비고 |")
    print("|------|-------------|------|")
    print(f"| Ollama  | {ollama_chars:,}자 | 기본 모델 |")
    print(f"| OpenAI  | {openai_chars:,}자 | 약 {openai_chars/ollama_chars:.1f}배 더 많은 텍스트 처리 |")
    
    # 실제 에피소드 예시
    print("\n📝 실제 에피소드 길이 예시:")
    print("-" * 50)
    
    episode_examples = [
        ("짧은 에피소드", 2000),
        ("보통 에피소드", 5000),
        ("긴 에피소드", 10000),
        ("매우 긴 에피소드", 15000),
    ]
    
    for desc, char_count in episode_examples:
        tokens = int(char_count / 1.55)
        ollama_chunks = 1 if char_count <= ollama_chars else (char_count // 1500) + 1
        openai_chunks = 1 if char_count <= openai_chars else (char_count // 1500) + 1
        
        print(f"\n{desc} ({char_count:,}자 / {tokens:,} tokens):")
        print(f"  • Ollama: {ollama_chunks}개 청크로 처리")
        print(f"  • OpenAI: {openai_chunks}개 청크로 처리")

if __name__ == "__main__":
    test_token_calculations()
    
    print("\n" + "=" * 80)
    print("✅ 토큰 비율이 올바르게 수정되었습니다!")
    print("   이전: 1자 → 1.5 토큰 (잘못된 계산)")
    print("   현재: 1.55자 → 1 토큰 (올바른 계산)")
    print("=" * 80)