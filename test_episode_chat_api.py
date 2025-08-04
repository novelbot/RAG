#!/usr/bin/env python3
"""
/api/v1/episode/chat 엔드포인트 테스트 스크립트

이 스크립트는 RDB에서 조회한 실제 데이터를 기반으로 
episode chat API의 다양한 기능을 테스트합니다.
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional

# API 설정
BASE_URL = "http://localhost:8000"
EPISODE_CHAT_URL = f"{BASE_URL}/api/v1/episode/chat"

# 테스트 데이터 (RDB에서 조회한 실제 데이터)
NOVEL_ID = 67  # "해방전후" by 이태준
EPISODE_IDS = [375, 376, 377, 378, 379, 380, 381, 382]
NOVEL_INFO = {
    "title": "해방전후",
    "author": "이태준", 
    "genre": "드라마, 역사",
    "description": "일제 말기의 감시와 해방 후의 이념 대립. 그 격동의 시대를 온몸으로 겪어내는 작가 '현'의 눈에 비친 조국의 어지러운 자화상.",
    "episodes": [
        {"id": 375, "number": 1, "title": "불길한 호출장"},
        {"id": 376, "number": 2, "title": "시국에 협력하십니까?"},
        {"id": 377, "number": 3, "title": "굴욕적인 타협"},
        {"id": 378, "number": 4, "title": "신사(神社) 터에서 마주친 것"},
        {"id": 379, "number": 5, "title": "노인의 분노"},
        {"id": 380, "number": 6, "title": "악몽의 연단(演壇)"},
        {"id": 381, "number": 7, "title": "뜻밖의 자유"},
        {"id": 382, "number": 8, "title": "붓을 들 수 없는 이유"}
    ]
}

def print_test_header(test_name: str):
    """테스트 헤더 출력"""
    print(f"\n{'='*60}")
    print(f"테스트: {test_name}")
    print(f"{'='*60}")

def print_response_summary(response: requests.Response, response_data: Optional[Dict]):
    """응답 요약 출력"""
    print(f"HTTP Status: {response.status_code}")
    if response_data:
        print(f"Conversation ID: {response_data.get('conversation_id', 'N/A')}")
        print(f"Response Length: {len(response_data.get('message', ''))}")
        print(f"Episode Sources: {len(response_data.get('episode_sources', []))}")
        print(f"Response Time: {response_data.get('response_time_ms', 0):.0f}ms")
        if response_data.get('confidence_score'):
            print(f"Confidence Score: {response_data['confidence_score']:.2f}")

def get_auth_token() -> str:
    """로그인해서 인증 토큰 가져오기"""
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_data = {"username": "admin", "password": "admin123"}
    
    response = requests.post(login_url, json=login_data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"로그인 실패: {response.status_code} - {response.text}")

def make_episode_chat_request(
    message: str,
    conversation_id: Optional[str] = None,
    episode_ids: Optional[List[int]] = None,
    novel_ids: Optional[List[int]] = None,
    **kwargs
) -> requests.Response:
    """Episode chat API 요청"""
    
    # 기본 요청 데이터
    request_data = {
        "message": message,
        "use_conversation_context": True,
        "include_sources": True,
        "response_format": "detailed",
        "episode_sort_order": "episode_number",
        "max_episodes": 5,
        "max_context_turns": 5
    }
    
    # 선택적 매개변수 추가
    if conversation_id:
        request_data["conversation_id"] = conversation_id
    if episode_ids:
        request_data["episode_ids"] = episode_ids
    if novel_ids:
        request_data["novel_ids"] = novel_ids
    
    # 추가 매개변수
    request_data.update(kwargs)
    
    # 실제 인증 토큰 사용
    try:
        token = get_auth_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    except Exception as e:
        print(f"토큰 획득 실패: {e}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer mock-token"
        }
    
    print(f"Request URL: {EPISODE_CHAT_URL}")
    print(f"Request Data: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
    
    # API 요청
    response = requests.post(EPISODE_CHAT_URL, json=request_data, headers=headers)
    return response

def test_basic_episode_question():
    """기본 에피소드 질문 테스트"""
    print_test_header("기본 에피소드 질문")
    
    message = "주인공 '현'이 첫 번째 에피소드에서 어떤 상황에 놓였나요?"
    
    response = make_episode_chat_request(
        message=message,
        episode_ids=[375],  # 첫 번째 에피소드만
        primary_episode_id=375,
        primary_novel_id=NOVEL_ID
    )
    
    if response.status_code == 200:
        data = response.json()
        print_response_summary(response, data)
        print(f"\nAI Response:\n{data['message']}")
        
        if data.get('episode_sources'):
            print(f"\nUsed Episodes:")
            for source in data['episode_sources']:
                print(f"- Episode {source['episode_number']}: {source['episode_title']}")
                print(f"  Relevance: {source['relevance_score']:.2f}")
        
        return data.get('conversation_id')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_multiple_episodes_question(conversation_id: Optional[str] = None):
    """여러 에피소드 질문 테스트"""
    print_test_header("여러 에피소드 질문")
    
    message = "주인공이 일제강점기에 작가로서 겪은 어려움들을 에피소드 1-4를 통해 설명해주세요."
    
    response = make_episode_chat_request(
        message=message,
        conversation_id=conversation_id,
        episode_ids=[375, 376, 377, 378],  # 첫 4개 에피소드
        primary_novel_id=NOVEL_ID,
        max_episodes=4,
        response_format="detailed"
    )
    
    if response.status_code == 200:
        data = response.json()
        print_response_summary(response, data)
        print(f"\nAI Response:\n{data['message']}")
        
        if data.get('episode_sources'):
            print(f"\nUsed Episodes:")
            for source in data['episode_sources']:
                print(f"- Episode {source['episode_number']}: {source['episode_title']}")
        
        return data.get('conversation_id')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return conversation_id

def test_novel_wide_question(conversation_id: Optional[str] = None):
    """소설 전체 질문 테스트"""
    print_test_header("소설 전체 질문")
    
    message = "이 소설 '해방전후'의 주요 테마와 시대적 배경은 무엇인가요?"
    
    response = make_episode_chat_request(
        message=message,
        conversation_id=conversation_id,
        novel_ids=[NOVEL_ID],  # 소설 전체
        primary_novel_id=NOVEL_ID,
        max_episodes=6,
        episode_sort_order="episode_number"
    )
    
    if response.status_code == 200:
        data = response.json()
        print_response_summary(response, data)
        print(f"\nAI Response:\n{data['message']}")
        
        # 메타데이터 확인
        if data.get('episode_metadata'):
            print(f"\nEpisode Metadata:")
            metadata = data['episode_metadata']
            print(f"- Episodes Referenced: {metadata.get('episodes_referenced', [])}")
            print(f"- Characters Mentioned: {metadata.get('characters_mentioned', [])}")
            print(f"- Timeline Context: {metadata.get('timeline_context', 'N/A')}")
        
        return data.get('conversation_id')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return conversation_id

def test_conversation_context(conversation_id: Optional[str] = None):
    """대화 맥락 유지 테스트"""
    print_test_header("대화 맥락 유지")
    
    if not conversation_id:
        print("이전 대화 ID가 없어서 테스트를 건너뜁니다.")
        return None
    
    message = "그렇다면 주인공은 이런 상황에서 어떤 선택을 했나요?"
    
    response = make_episode_chat_request(
        message=message,
        conversation_id=conversation_id,
        novel_ids=[NOVEL_ID],
        use_conversation_context=True,
        max_context_turns=3
    )
    
    if response.status_code == 200:
        data = response.json()
        print_response_summary(response, data)
        print(f"\nAI Response:\n{data['message']}")
        
        # 대화 메타데이터
        if data.get('conversation_metadata'):
            meta = data['conversation_metadata']
            print(f"\nConversation Metadata:")
            print(f"- Total Messages: {meta.get('total_messages', 0)}")
            print(f"- Context Messages Used: {meta.get('context_messages_used', 0)}")
            print(f"- Conversation Scope: {meta.get('conversation_scope', 'N/A')}")
        
        return data.get('conversation_id')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return conversation_id

def test_concise_response():
    """간결한 응답 형식 테스트"""
    print_test_header("간결한 응답 형식")
    
    message = "주인공 '현'은 누구인가요?"
    
    response = make_episode_chat_request(
        message=message,
        novel_ids=[NOVEL_ID],
        response_format="concise",  # 간결한 형식
        max_episodes=3,
        include_sources=False  # 소스 제외
    )
    
    if response.status_code == 200:
        data = response.json()
        print_response_summary(response, data)
        print(f"\nAI Response (Concise):\n{data['message']}")
        
        return data.get('conversation_id')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_similarity_sorting():
    """유사도 정렬 테스트"""
    print_test_header("유사도 기반 정렬")
    
    message = "작가가 일본의 압력을 받는 장면들을 보여주세요."
    
    response = make_episode_chat_request(
        message=message,
        novel_ids=[NOVEL_ID],
        episode_sort_order="similarity",  # 유사도 정렬
        max_episodes=4,
        include_episode_metadata=True
    )
    
    if response.status_code == 200:
        data = response.json()
        print_response_summary(response, data)
        print(f"\nAI Response:\n{data['message']}")
        
        if data.get('episode_sources'):
            print(f"\nEpisodes (sorted by similarity):")
            for i, source in enumerate(data['episode_sources'], 1):
                print(f"{i}. Episode {source['episode_number']}: {source['episode_title']}")
                print(f"   Similarity Score: {source['similarity_score']:.3f}")
        
        return data.get('conversation_id')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Episode Chat API 테스트 시작")
    print(f"Novel: {NOVEL_INFO['title']} by {NOVEL_INFO['author']}")
    print(f"Episodes: {len(EPISODE_IDS)} episodes")
    print("=" * 60)
    
    conversation_id = None
    
    try:
        # 1. 기본 에피소드 질문
        conversation_id = test_basic_episode_question()
        time.sleep(1)
        
        # 2. 여러 에피소드 질문
        conversation_id = test_multiple_episodes_question(conversation_id)
        time.sleep(1)
        
        # 3. 소설 전체 질문
        conversation_id = test_novel_wide_question(conversation_id)
        time.sleep(1)
        
        # 4. 대화 맥락 유지
        conversation_id = test_conversation_context(conversation_id)
        time.sleep(1)
        
        # 5. 간결한 응답
        test_concise_response()
        time.sleep(1)
        
        # 6. 유사도 정렬
        test_similarity_sorting()
        
        print(f"\n{'='*60}")
        print("모든 테스트 완료!")
        if conversation_id:
            print(f"최종 Conversation ID: {conversation_id}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    run_all_tests()