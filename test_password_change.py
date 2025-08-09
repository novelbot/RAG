#!/usr/bin/env python3
"""
비밀번호 변경 API 테스트 스크립트
"""

import httpx
import asyncio
import json


async def test_password_change():
    """비밀번호 변경 기능 테스트"""
    base_url = "http://localhost:8000/api/v1"
    
    async with httpx.AsyncClient() as client:
        print("=" * 60)
        print("비밀번호 변경 API 테스트")
        print("=" * 60)
        
        # 1. 로그인
        print("\n1. admin 계정으로 로그인...")
        login_response = await client.post(
            f"{base_url}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        if login_response.status_code != 200:
            print(f"❌ 로그인 실패: {login_response.text}")
            return
        
        token_data = login_response.json()
        access_token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        print("✅ 로그인 성공")
        
        # 2. 현재 사용자 정보 확인
        print("\n2. 현재 사용자 정보 확인...")
        me_response = await client.get(
            f"{base_url}/auth/me",
            headers=headers
        )
        
        if me_response.status_code == 200:
            user_info = me_response.json()
            print(f"✅ 사용자: {user_info['username']} (role: {user_info.get('roles', ['user'])})")
        else:
            print(f"❌ 사용자 정보 조회 실패: {me_response.text}")
        
        # 3. 비밀번호 변경 테스트
        print("\n3. 비밀번호 변경 테스트...")
        
        # 3-1. 잘못된 현재 비밀번호로 시도
        print("   3-1. 잘못된 현재 비밀번호로 시도...")
        change_response = await client.post(
            f"{base_url}/auth/change-password",
            json={
                "current_password": "wrong_password",
                "new_password": "newPassword123!",
                "confirm_password": "newPassword123!"
            },
            headers=headers
        )
        
        if change_response.status_code == 400:
            print(f"   ✅ 예상대로 실패: {change_response.json()['detail']}")
        else:
            print(f"   ❌ 예상치 못한 응답: {change_response.status_code}")
        
        # 3-2. 비밀번호 불일치로 시도
        print("   3-2. 새 비밀번호 확인 불일치로 시도...")
        try:
            change_response = await client.post(
                f"{base_url}/auth/change-password",
                json={
                    "current_password": "admin123",
                    "new_password": "newPassword123!",
                    "confirm_password": "differentPassword123!"
                },
                headers=headers
            )
            
            if change_response.status_code == 422:
                print(f"   ✅ 예상대로 검증 실패")
            else:
                print(f"   ❌ 예상치 못한 응답: {change_response.status_code}")
        except Exception as e:
            print(f"   ✅ 예상대로 검증 실패: {e}")
        
        # 3-3. 올바른 정보로 비밀번호 변경
        print("   3-3. 올바른 정보로 비밀번호 변경...")
        change_response = await client.post(
            f"{base_url}/auth/change-password",
            json={
                "current_password": "admin123",
                "new_password": "newAdminPass123!",
                "confirm_password": "newAdminPass123!"
            },
            headers=headers
        )
        
        if change_response.status_code == 200:
            result = change_response.json()
            print(f"   ✅ 비밀번호 변경 성공: {result['message']}")
        else:
            print(f"   ❌ 비밀번호 변경 실패: {change_response.text}")
            return
        
        # 4. 새 비밀번호로 로그인 테스트
        print("\n4. 새 비밀번호로 로그인 테스트...")
        
        # 4-1. 기존 비밀번호로 로그인 시도 (실패해야 함)
        print("   4-1. 기존 비밀번호로 로그인 시도...")
        login_response = await client.post(
            f"{base_url}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        if login_response.status_code == 401:
            print("   ✅ 예상대로 기존 비밀번호 로그인 실패")
        else:
            print("   ❌ 기존 비밀번호로 로그인이 되어버림!")
        
        # 4-2. 새 비밀번호로 로그인 시도 (성공해야 함)
        print("   4-2. 새 비밀번호로 로그인 시도...")
        login_response = await client.post(
            f"{base_url}/auth/login",
            json={"username": "admin", "password": "newAdminPass123!"}
        )
        
        if login_response.status_code == 200:
            print("   ✅ 새 비밀번호로 로그인 성공")
            new_token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {new_token}"}
        else:
            print("   ❌ 새 비밀번호로 로그인 실패")
            return
        
        # 5. 비밀번호 원복 (테스트 환경 유지)
        print("\n5. 테스트 환경 유지를 위해 비밀번호 원복...")
        change_response = await client.post(
            f"{base_url}/auth/change-password",
            json={
                "current_password": "newAdminPass123!",
                "new_password": "admin123",
                "confirm_password": "admin123"
            },
            headers=headers
        )
        
        if change_response.status_code == 200:
            print("   ✅ 비밀번호 원복 완료")
        else:
            print("   ⚠️  비밀번호 원복 실패 - 수동으로 원복 필요")
        
        # 6. Admin 권한으로 다른 사용자 비밀번호 리셋 테스트
        print("\n6. Admin 권한으로 다른 사용자 비밀번호 리셋...")
        
        # 6-1. user 계정 비밀번호 리셋
        print("   6-1. 'user' 계정 비밀번호 리셋...")
        reset_response = await client.post(
            f"{base_url}/auth/reset-password",
            json={
                "username": "user",
                "new_password": "resetUser123!"
            },
            headers=headers
        )
        
        if reset_response.status_code == 200:
            result = reset_response.json()
            print(f"   ✅ 비밀번호 리셋 성공: {result['message']}")
        else:
            print(f"   ❌ 비밀번호 리셋 실패: {reset_response.text}")
        
        # 6-2. 존재하지 않는 사용자 리셋 시도
        print("   6-2. 존재하지 않는 사용자 리셋 시도...")
        reset_response = await client.post(
            f"{base_url}/auth/reset-password",
            json={
                "username": "nonexistent_user",
                "new_password": "test123!"
            },
            headers=headers
        )
        
        if reset_response.status_code == 404:
            print(f"   ✅ 예상대로 실패: {reset_response.json()['detail']}")
        else:
            print(f"   ❌ 예상치 못한 응답: {reset_response.status_code}")
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_password_change())