#!/usr/bin/env python3

import requests
import json

print("🚀 최종 Episode Chat API 테스트")
print("=" * 50)

# 새 토큰 받기
login_data = {'username': 'admin', 'password': 'admin123'}
login_response = requests.post('http://localhost:8000/api/v1/auth/login', json=login_data)

if login_response.status_code == 200:
    token_data = login_response.json()
    access_token = token_data['access_token']
    
    print("✅ 로그인 성공")
    print(f"Token Type: {token_data['token_type']}")
    print(f"Expires In: {token_data['expires_in']}초")
    
    # Episode Chat API 호출
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    
    chat_data = {
        "message": "주인공이 누구야?",
        "conversation_id": None,
        "episode_ids": [375, 376, 377, 378, 379, 380, 381, 382],
        "max_episodes": 5,
        "max_results": 5,
        "use_conversation_context": True,
        "max_context_turns": 5,
        "include_episode_metadata": True,
        "episode_sort_order": "episode_number",
        "include_sources": True,
        "response_format": "detailed"
    }
    
    print("\n📡 Episode Chat API 요청 중...")
    
    try:
        response = requests.post(
            'http://localhost:8000/api/v1/episode/chat',
            headers=headers,
            json=chat_data,
            timeout=60  # 더 긴 타임아웃
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("🎉 성공! API가 정상 작동합니다!")
            try:
                result = response.json()
                print(f"응답 타입: {type(result)}")
                if isinstance(result, dict):
                    print(f"응답 키들: {list(result.keys())}")
                    if 'response' in result:
                        print(f"AI 응답: {result['response'][:200]}...")
                    if 'sources' in result:
                        print(f"소스 개수: {len(result.get('sources', []))}")
            except:
                print("응답 파싱 실패, raw 응답:")
                print(response.text[:500])
                
        elif response.status_code == 401:
            print("❌ 인증 실패 - 토큰 문제")
            
        elif response.status_code == 500:
            print("❌ 서버 내부 오류")
            print(f"응답: {response.text}")
            
        else:
            print(f"❓ 예상치 못한 상태 코드: {response.status_code}")
            print(f"응답: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ 요청 타임아웃 (60초)")
    except requests.exceptions.RequestException as e:
        print(f"🔌 네트워크 오류: {e}")
        
else:
    print(f"❌ 로그인 실패: {login_response.text}")

print("\n" + "=" * 50)
print("테스트 완료")