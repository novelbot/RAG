import asyncio
import httpx
import json
import time

async def test_streaming():
    async with httpx.AsyncClient() as client:
        # 1. Login
        print("1. 로그인 중...")
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        print(f"✅ 로그인 성공")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # 2. Test streaming endpoint
        print("\n2. OpenAI 스트리밍 테스트...")
        print("=" * 50)
        
        test_cases = [
            {
                "message": "안녕하세요\! 1부터 5까지 천천히 세어주세요.",
                "description": "숫자 세기 테스트"
            },
            {
                "message": "한국의 수도는 어디인가요? 짧게 대답해주세요.",
                "description": "간단한 질문 테스트"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n테스트 {i}: {test['description']}")
            print(f"질문: {test['message']}")
            print("-" * 30)
            
            stream_data = {
                "message": test["message"],
                "episode_ids": [],
                "novel_ids": []
            }
            
            start_time = time.time()
            chunk_count = 0
            full_response = ""
            
            try:
                async with client.stream(
                    "POST",
                    "http://localhost:8000/api/v1/episode/chat/stream",
                    json=stream_data,
                    headers=headers,
                    timeout=30.0
                ) as response:
                    if response.status_code == 200:
                        print("스트리밍 응답: ", end="")
                        
                        async for line in response.aiter_lines():
                            if line and line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("type") == "content":
                                        chunk_count += 1
                                        content = data.get("content", "")
                                        full_response += content
                                        print(content, end="", flush=True)
                                    elif data.get("type") == "done":
                                        elapsed = time.time() - start_time
                                        print(f"\n\n✅ 스트리밍 완료")
                                        print(f"  - 청크 수: {chunk_count}")
                                        print(f"  - 소요 시간: {elapsed:.2f}초")
                                        print(f"  - 응답 길이: {len(full_response)}자")
                                except json.JSONDecodeError:
                                    continue
                    else:
                        print(f"❌ 에러: HTTP {response.status_code}")
                        
            except Exception as e:
                print(f"❌ 예외 발생: {e}")
        
        # 3. Test with conversation context
        print("\n\n3. 대화 컨텍스트 테스트...")
        print("=" * 50)
        
        # First message
        print("첫 번째 메시지: 내 이름은 김철수입니다.")
        first_msg = {
            "message": "내 이름은 김철수입니다. 기억해주세요.",
            "episode_ids": [],
            "novel_ids": []
        }
        
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v1/episode/chat/stream",
            json=first_msg,
            headers=headers,
            timeout=30.0
        ) as response:
            conversation_id = None
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "conversation_info":
                            conversation_id = data.get("conversation_id")
                            print(f"대화 ID: {conversation_id}")
                        elif data.get("type") == "content":
                            print(data.get("content", ""), end="", flush=True)
                    except:
                        continue
        
        if conversation_id:
            print("\n\n두 번째 메시지: 제 이름이 뭐라고 했죠?")
            second_msg = {
                "message": "제 이름이 뭐라고 했죠?",
                "conversation_id": conversation_id,
                "use_conversation_context": True,
                "episode_ids": [],
                "novel_ids": []
            }
            
            async with client.stream(
                "POST",
                "http://localhost:8000/api/v1/episode/chat/stream",
                json=second_msg,
                headers=headers,
                timeout=30.0
            ) as response:
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content":
                                print(data.get("content", ""), end="", flush=True)
                        except:
                            continue
            print("\n✅ 대화 컨텍스트 테스트 완료")

if __name__ == "__main__":
    print("=== OpenAI 스트리밍 종합 테스트 ===\n")
    asyncio.run(test_streaming())
    print("\n\n=== 모든 테스트 완료 ===")
