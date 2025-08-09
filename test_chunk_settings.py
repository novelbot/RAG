#!/usr/bin/env python3
"""
모델별 청크 크기 설정 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_chunk_settings():
    """각 모델의 청크 설정 계산"""
    
    print("=" * 80)
    print("모델별 청크 크기 설정 (수정 후)")
    print("=" * 80)
    
    # 테스트할 모델들과 max_tokens
    models = [
        ("Ollama (jeffh/intfloat)", 2048),
        ("OpenAI (text-embedding-ada-002)", 8191),
        ("OpenAI (text-embedding-3-small)", 8191),
        ("OpenAI (text-embedding-3-large)", 8191),
    ]
    
    print("\n| 모델 | Max Tokens | 청크 크기 | 오버랩 | 비고 |")
    print("|------|------------|-----------|--------|------|")
    
    for model_name, max_tokens in models:
        # 실제 계산 로직 (processor.py와 동일)
        safe_tokens = int(max_tokens * 0.85)
        safe_chars = int(safe_tokens * 1.55)
        
        # 새로운 제한: min 500, max 15000
        chunk_size = max(500, min(15000, safe_chars))
        
        # 오버랩 계산 (13.3%)
        calculated_overlap = int(chunk_size * 0.133)
        overlap = max(50, min(2000, calculated_overlap))
        
        # 이전 설정과 비교
        old_chunk = min(1500, safe_chars)
        old_overlap = 200
        
        change = "변경됨" if chunk_size != old_chunk else "동일"
        
        print(f"| {model_name:<30} | {max_tokens:>10,} | {chunk_size:>9,}자 | {overlap:>6}자 | {change} |")
    
    # 실제 에피소드 처리 예시
    print("\n" + "=" * 80)
    print("에피소드 길이별 청킹 비교")
    print("=" * 80)
    
    episode_lengths = [
        ("짧은 에피소드", 2000),
        ("보통 에피소드", 5000),
        ("긴 에피소드", 10000),
        ("매우 긴 에피소드", 15000),
        ("초장문 에피소드", 20000),
    ]
    
    # Ollama와 OpenAI 청크 설정
    ollama_chunk = 2697  # 계산된 값
    ollama_overlap = 358  # 13.3%
    
    openai_chunk = 10791  # 계산된 값
    openai_overlap = 1435  # 13.3%
    
    print("\n📊 Ollama (청크: 2,697자, 오버랩: 358자)")
    print("-" * 50)
    for desc, length in episode_lengths:
        if length <= ollama_chunk:
            chunks = 1
            print(f"{desc:15} ({length:6,}자): {chunks}개 청크 (청킹 불필요)")
        else:
            # 청킹 계산 (오버랩 고려)
            effective_chunk = ollama_chunk - ollama_overlap
            chunks = 1 + ((length - ollama_chunk) + effective_chunk - 1) // effective_chunk
            print(f"{desc:15} ({length:6,}자): {chunks}개 청크로 분할")
    
    print("\n📊 OpenAI (청크: 10,791자, 오버랩: 1,435자)")
    print("-" * 50)
    for desc, length in episode_lengths:
        if length <= openai_chunk:
            chunks = 1
            print(f"{desc:15} ({length:6,}자): {chunks}개 청크 (청킹 불필요)")
        else:
            # 청킹 계산 (오버랩 고려)
            effective_chunk = openai_chunk - openai_overlap
            chunks = 1 + ((length - openai_chunk) + effective_chunk - 1) // effective_chunk
            print(f"{desc:15} ({length:6,}자): {chunks}개 청크로 분할")
    
    # 이전 설정(1,500자 고정)과 비교
    print("\n" + "=" * 80)
    print("이전 설정(1,500자 고정) vs 현재 설정 비교")
    print("=" * 80)
    
    print("\n| 에피소드 길이 | 이전 (1,500자) | Ollama (2,697자) | OpenAI (10,791자) |")
    print("|---------------|----------------|------------------|-------------------|")
    
    for desc, length in episode_lengths:
        # 이전 설정 (1,500자 고정)
        old_chunks = (length + 1499) // 1500
        
        # Ollama
        if length <= ollama_chunk:
            ollama_chunks = 1
        else:
            effective = ollama_chunk - ollama_overlap
            ollama_chunks = 1 + ((length - ollama_chunk) + effective - 1) // effective
        
        # OpenAI
        if length <= openai_chunk:
            openai_chunks = 1
        else:
            effective = openai_chunk - openai_overlap
            openai_chunks = 1 + ((length - openai_chunk) + effective - 1) // effective
        
        print(f"| {desc:13} | {old_chunks:14}개 | {ollama_chunks:16}개 | {openai_chunks:17}개 |")
    
    # 효율성 개선 계산
    print("\n" + "=" * 80)
    print("효율성 개선 (청크 수 감소율)")
    print("=" * 80)
    
    total_old = sum((length + 1499) // 1500 for _, length in episode_lengths)
    
    # Ollama
    total_ollama = 0
    for _, length in episode_lengths:
        if length <= ollama_chunk:
            total_ollama += 1
        else:
            effective = ollama_chunk - ollama_overlap
            total_ollama += 1 + ((length - ollama_chunk) + effective - 1) // effective
    
    # OpenAI
    total_openai = 0
    for _, length in episode_lengths:
        if length <= openai_chunk:
            total_openai += 1
        else:
            effective = openai_chunk - openai_overlap
            total_openai += 1 + ((length - openai_chunk) + effective - 1) // effective
    
    ollama_reduction = (1 - total_ollama / total_old) * 100
    openai_reduction = (1 - total_openai / total_old) * 100
    
    print(f"\n전체 청크 수 (5개 에피소드 합계):")
    print(f"  • 이전 설정: {total_old}개")
    print(f"  • Ollama: {total_ollama}개 (▼ {ollama_reduction:.1f}% 감소)")
    print(f"  • OpenAI: {total_openai}개 (▼ {openai_reduction:.1f}% 감소)")

if __name__ == "__main__":
    test_chunk_settings()
    
    print("\n" + "=" * 80)
    print("✅ 청크 설정 개선 완료!")
    print("   • 1,500자 고정 제한 제거")
    print("   • 모델별 최적 청크 크기 사용")
    print("   • 동적 오버랩 계산 (13.3%)")
    print("   • 안전 제한 적용 (500~15,000자)")
    print("=" * 80)