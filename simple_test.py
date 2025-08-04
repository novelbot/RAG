#!/usr/bin/env python3

import requests

print("🔍 단계별 API 테스트")

# 로그인
login_response = requests.post('http://localhost:8000/api/v1/auth/login', 
                              json={'username': 'admin', 'password': 'admin123'})

if login_response.status_code == 200:
    access_token = login_response.json()['access_token']
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    
    print("✅ 로그인 성공")
    
    # 1. 가장 간단한 요청
    print("\n📡 테스트 1: 최소한의 요청")
    simple_data = {
        "message": "안녕",
        "episode_ids": [375]
    }
    
    response = requests.post('http://localhost:8000/api/v1/episode/chat', 
                           headers=headers, json=simple_data, timeout=30)
    print(f"Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
    else:
        print("✅ 간단한 요청 성공!")
        
    # 2. episode 없이 테스트
    print("\n📡 테스트 2: episode_ids 없이")
    no_episode_data = {
        "message": "안녕"
    }
    
    response2 = requests.post('http://localhost:8000/api/v1/episode/chat', 
                            headers=headers, json=no_episode_data, timeout=30)
    print(f"Status: {response2.status_code}")
    
    if response2.status_code != 200:
        print(f"Error: {response2.text}")
    else:
        print("✅ episode 없는 요청 성공!")
        
else:
    print("❌ 로그인 실패")