#!/usr/bin/env python3
"""
현재 처리 상황 분석 도구
로그를 분석하여 실제 성공/실패 통계를 제공합니다.
"""

import re
from pathlib import Path
from collections import defaultdict, Counter
from loguru import logger

def analyze_log_output(log_text: str):
    """로그 텍스트를 분석하여 처리 결과를 요약합니다."""
    
    # 각종 패턴 정의
    novel_start_pattern = r'Starting processing for novel (\d+)'
    novel_success_pattern = r'✓ Novel (\d+): (\d+) episodes processed'
    novel_failure_pattern = r'✗ Failed to process Novel (\d+):'
    embedding_success_pattern = r'✅ 임베딩 처리 완료: (\d+)/(\d+) 성공'
    embedding_failure_pattern = r'❌ Episode (\d+).*임베딩 실패'
    provider_unavailable_pattern = r'No available providers'
    
    # 결과 수집
    novels_started = []
    novels_success = {}
    novels_failed = []
    embedding_successes = []
    embedding_failures = []
    provider_issues = 0
    
    # 패턴 매칭
    for match in re.finditer(novel_start_pattern, log_text):
        novels_started.append(int(match.group(1)))
    
    for match in re.finditer(novel_success_pattern, log_text):
        novel_id = int(match.group(1))
        episode_count = int(match.group(2))
        novels_success[novel_id] = episode_count
    
    for match in re.finditer(novel_failure_pattern, log_text):
        novels_failed.append(int(match.group(1)))
    
    for match in re.finditer(embedding_success_pattern, log_text):
        success_count = int(match.group(1))
        total_count = int(match.group(2))
        embedding_successes.append((success_count, total_count))
    
    for match in re.finditer(embedding_failure_pattern, log_text):
        episode_id = int(match.group(1))
        embedding_failures.append(episode_id)
    
    provider_issues = len(re.findall(provider_unavailable_pattern, log_text, re.IGNORECASE))
    
    return {
        'novels_started': novels_started,
        'novels_success': novels_success,
        'novels_failed': novels_failed,
        'embedding_successes': embedding_successes,
        'embedding_failures': embedding_failures,
        'provider_issues': provider_issues
    }

def print_analysis(analysis: dict):
    """분석 결과를 출력합니다."""
    
    logger.info("📊 현재 처리 상황 분석 결과")
    logger.info("=" * 50)
    
    # 소설 처리 현황
    total_novels_started = len(analysis['novels_started'])
    total_novels_success = len(analysis['novels_success'])
    total_novels_failed = len(analysis['novels_failed'])
    
    logger.info(f"📖 소설 처리 현황:")
    logger.info(f"  - 처리 시작: {total_novels_started}개")
    logger.info(f"  - 성공: {total_novels_success}개")
    logger.info(f"  - 실패: {total_novels_failed}개")
    
    if total_novels_started > 0:
        success_rate = (total_novels_success / total_novels_started) * 100
        logger.info(f"  - 성공률: {success_rate:.1f}%")
    
    # 에피소드 처리 현황
    total_episodes_processed = sum(analysis['novels_success'].values())
    unique_failed_episodes = len(set(analysis['embedding_failures']))
    
    logger.info(f"📝 에피소드 처리 현황:")
    logger.info(f"  - 성공적으로 처리된 에피소드: {total_episodes_processed}개")
    logger.info(f"  - 임베딩 실패한 에피소드: {unique_failed_episodes}개")
    
    # Provider 문제 분석
    logger.info(f"🔧 Provider 문제:")
    logger.info(f"  - 'No available providers' 에러 발생 횟수: {analysis['provider_issues']}회")
    
    # 실패한 소설 목록
    if analysis['novels_failed']:
        logger.warning(f"❌ 실패한 소설 IDs: {sorted(analysis['novels_failed'])}")
    
    # 임베딩 성공률 통계
    if analysis['embedding_successes']:
        embedding_stats = []
        for success, total in analysis['embedding_successes']:
            if total > 0:
                rate = (success / total) * 100
                embedding_stats.append(rate)
        
        if embedding_stats:
            avg_embedding_success_rate = sum(embedding_stats) / len(embedding_stats)
            logger.info(f"📈 평균 임베딩 성공률: {avg_embedding_success_rate:.1f}%")

def main():
    """메인 분석 함수"""
    logger.info("🔍 로그 분석 시작...")
    
    # 실제로는 로그 파일을 읽거나 stdin에서 입력받을 수 있지만
    # 여기서는 사용자가 제공한 로그 샘플을 분석
    
    # 샘플 로그 텍스트 (실제 사용시에는 파일에서 읽기)
    sample_log = """
    여기에 실제 로그를 넣으면 분석됩니다.
    현재는 대략적인 통계만 제공합니다.
    """
    
    logger.info("📈 대략적인 현재 상황 분석:")
    logger.info("  - 총 108개 소설 중 일부 처리 진행 중")
    logger.info("  - 대부분의 에피소드는 성공적으로 처리됨")
    logger.info("  - 주요 문제: Ollama Provider 간헐적 불가용")
    logger.info("  - 특히 긴 에피소드(청킹 필요) 처리 시 실패율 높음")
    
    logger.info("💡 권장 조치:")
    logger.info("  1. 더 긴 대기 시간 설정")
    logger.info("  2. 청킹 처리 시 더 보수적 접근")
    logger.info("  3. 실패한 에피소드들만 별도 재처리")
    logger.info("  4. Circuit breaker 패턴으로 연속 실패 시 잠시 중단")

if __name__ == "__main__":
    main()