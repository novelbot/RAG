#!/usr/bin/env python3
"""
Test script for the continuous conversation system.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, Optional


class ConversationTester:
    """Test client for conversation system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.access_token: Optional[str] = None
    
    async def login(self, username: str = "testuser", password: str = "testpass") -> bool:
        """Login and get access token"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/login",
                    json={"username": username, "password": password}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.access_token = data.get("access_token")
                    print(f"✅ Logged in successfully as {username}")
                    return True
                else:
                    print(f"❌ Login failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Login error: {e}")
                return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        if not self.access_token:
            raise ValueError("Not authenticated - please login first")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def test_basic_query(self) -> bool:
        """Test basic query without conversation context"""
        print("\n🧪 Testing basic query without context...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query/ask",
                    headers=self._get_headers(),
                    json={
                        "query": "안녕하세요! 오늘 날씨는 어때요?",
                        "use_context": False
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Basic query successful")
                    print(f"   Question: {data['question']}")
                    print(f"   Answer: {data['answer'][:100]}...")
                    print(f"   Context used: {data['metadata']['conversation']['context_enabled']}")
                    return True
                else:
                    print(f"❌ Basic query failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Basic query error: {e}")
                return False
    
    async def test_create_conversation(self) -> bool:
        """Test creating a new conversation session"""
        print("\n🧪 Testing conversation session creation...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query/conversations",
                    headers=self._get_headers(),
                    params={"title": "테스트 대화 세션"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.session_id = data["conversation"]["session_id"]
                    print(f"✅ Conversation session created")
                    print(f"   Session ID: {self.session_id}")
                    print(f"   Title: {data['conversation']['title']}")
                    print(f"   Status: {data['conversation']['status']}")
                    return True
                else:
                    print(f"❌ Session creation failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Session creation error: {e}")
                return False
    
    async def test_context_conversation(self) -> bool:
        """Test conversation with context"""
        print("\n🧪 Testing conversation with context...")
        
        if not self.session_id:
            print("❌ No session ID available - creating new session")
            if not await self.test_create_conversation():
                return False
        
        # First question
        print("\n📝 First question:")
        success = await self._ask_with_context("제가 좋아하는 취미는 독서입니다.")
        if not success:
            return False
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Second question (referring to previous context)
        print("\n📝 Second question (with context):")
        success = await self._ask_with_context("그 취미와 관련된 추천을 해주세요.")
        if not success:
            return False
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Third question (continuing conversation)
        print("\n📝 Third question (continuing context):")
        success = await self._ask_with_context("특히 한국 소설 중에서 추천해주세요.")
        if not success:
            return False
        
        return True
    
    async def _ask_with_context(self, question: str) -> bool:
        """Ask a question with conversation context"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query/ask",
                    headers=self._get_headers(),
                    json={
                        "query": question,
                        "session_id": self.session_id,
                        "use_context": True,
                        "conversation_context": {
                            "max_context_turns": 5
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    conv_data = data['metadata']['conversation']
                    
                    print(f"✅ Contextual query successful")
                    print(f"   Question: {data['question']}")
                    print(f"   Answer: {data['answer'][:150]}...")
                    print(f"   Session ID: {conv_data['session_id']}")
                    print(f"   Context turns used: {conv_data['context_turns_used']}")
                    print(f"   Enhanced query used: {conv_data['enhanced_query_used']}")
                    return True
                else:
                    print(f"❌ Contextual query failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Contextual query error: {e}")
                return False
    
    async def test_get_conversation_history(self) -> bool:
        """Test retrieving conversation history"""
        print("\n🧪 Testing conversation history retrieval...")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/query/conversations/{self.session_id}",
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    data = response.json()
                    conversation = data["conversation"]
                    turns = conversation.get("conversation_turns", [])
                    
                    print(f"✅ Conversation history retrieved")
                    print(f"   Session ID: {conversation['session_id']}")
                    print(f"   Total turns: {len(turns)}")
                    print(f"   Session status: {conversation['status']}")
                    print(f"   Is active: {conversation['is_active']}")
                    
                    # Show conversation turns
                    for i, turn in enumerate(turns, 1):
                        print(f"   Turn {i}:")
                        print(f"     User: {turn['user_query'][:80]}...")
                        print(f"     Assistant: {turn['assistant_response'][:80]}...")
                    
                    return True
                else:
                    print(f"❌ History retrieval failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ History retrieval error: {e}")
                return False
    
    async def test_list_conversations(self) -> bool:
        """Test listing user conversations"""
        print("\n🧪 Testing conversation list...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/query/conversations",
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    data = response.json()
                    conversations = data["conversations"]
                    
                    print(f"✅ Conversation list retrieved")
                    print(f"   Total conversations: {data['total_count']}")
                    
                    for conv in conversations:
                        print(f"   - Session: {conv['session_id'][:8]}...")
                        print(f"     Title: {conv.get('title', 'No title')}")
                        print(f"     Turns: {conv['turn_count']}")
                        print(f"     Status: {conv['status']}")
                        print(f"     Last activity: {conv['last_activity_at']}")
                    
                    return True
                else:
                    print(f"❌ Conversation list failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Conversation list error: {e}")
                return False
    
    async def test_archive_conversation(self) -> bool:
        """Test archiving conversation"""
        print("\n🧪 Testing conversation archiving...")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{self.base_url}/query/conversations/{self.session_id}/archive",
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Conversation archived successfully")
                    print(f"   Message: {data['message']}")
                    return True
                else:
                    print(f"❌ Archive failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Archive error: {e}")
                return False
    
    async def run_all_tests(self) -> None:
        """Run all conversation tests"""
        print("🚀 Starting conversation system tests...")
        print("=" * 60)
        
        # Login first
        if not await self.login():
            print("❌ Cannot proceed without authentication")
            return
        
        test_results = []
        
        # Run tests
        test_results.append(("Basic Query", await self.test_basic_query()))
        test_results.append(("Create Conversation", await self.test_create_conversation()))
        test_results.append(("Context Conversation", await self.test_context_conversation()))
        test_results.append(("Get History", await self.test_get_conversation_history()))
        test_results.append(("List Conversations", await self.test_list_conversations()))
        test_results.append(("Archive Conversation", await self.test_archive_conversation()))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<25} {status}")
            if result:
                passed += 1
        
        print("=" * 60)
        print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️ Some tests failed - check implementation")


async def main():
    """Main test function"""
    tester = ConversationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())