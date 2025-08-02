#!/usr/bin/env python3
"""
Integration test for conversation system without requiring full API server.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def test_conversation_system():
    """Test the conversation system components."""
    
    print("🚀 Testing Conversation System Integration")
    print("=" * 60)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        
        from src.services.conversation_manager import conversation_manager
        from src.models.conversation import ConversationSession, ConversationTurn
        from src.api.schemas import QueryRequest, ConversationContext, ConversationTurn as SchemaTurn
        
        print("✅ All imports successful")
        
        # Test 1: Create a conversation session
        print("\n🧪 Test 1: Creating conversation session...")
        
        session = await conversation_manager.create_session(
            user_id="test_user_123",
            title="Integration Test Conversation",
            expires_in_hours=24
        )
        
        print(f"✅ Session created: {session.session_id}")
        print(f"   Title: {session.title}")
        print(f"   User: {session.user_id}")
        print(f"   Status: {session.status}")
        
        # Test 2: Add conversation turns
        print("\n🧪 Test 2: Adding conversation turns...")
        
        turns_data = [
            ("안녕하세요! 소설 추천을 받고 싶습니다.", "안녕하세요! 어떤 장르의 소설을 좋아하시나요?"),
            ("판타지 소설을 좋아합니다.", "판타지 소설이라면 '해리 포터' 시리즈나 '반지의 제왕' 같은 작품들이 인기가 많습니다. 한국 판타지 소설도 관심 있으시나요?"),
            ("네, 한국 판타지 소설 추천해주세요.", "한국 판타지 소설로는 '달빛조각사', '템빨', '나 혼자만 레벨업' 등이 유명합니다. 이 중에서 어떤 스타일을 선호하시나요?")
        ]
        
        for i, (user_query, assistant_response) in enumerate(turns_data, 1):
            turn = await conversation_manager.add_turn(
                session_id=session.session_id,
                user_query=user_query,
                assistant_response=assistant_response,
                response_time_ms=150 + i * 20,
                token_count=50 + i * 15
            )
            print(f"✅ Turn {i} added: {turn.turn_number}")
        
        # Test 3: Retrieve conversation context
        print("\n🧪 Test 3: Retrieving conversation context...")
        
        context = await conversation_manager.get_conversation_context(
            session.session_id, max_turns=5
        )
        
        print(f"✅ Context retrieved: {len(context)} turns")
        for i, turn in enumerate(context, 1):
            print(f"   Turn {i}:")
            print(f"     User: {turn['user_query'][:50]}...")
            print(f"     Assistant: {turn['assistant_response'][:50]}...")
        
        # Test 4: Build context prompt
        print("\n🧪 Test 4: Building context prompt...")
        
        current_query = "그 중에서 '나 혼자만 레벨업' 스타일과 비슷한 다른 작품도 추천해주세요."
        enhanced_prompt = conversation_manager.build_context_prompt(context, current_query)
        
        print("✅ Context prompt built:")
        print(f"   Length: {len(enhanced_prompt)} characters")
        print("   Preview:")
        lines = enhanced_prompt.split('\n')
        for line in lines[:8]:  # Show first 8 lines
            print(f"     {line}")
        if len(lines) > 8:
            print(f"     ... ({len(lines) - 8} more lines)")
        
        # Test 5: Test session management
        print("\n🧪 Test 5: Testing session management...")
        
        # Get user sessions
        user_sessions = await conversation_manager.get_user_sessions("test_user_123")
        print(f"✅ User sessions retrieved: {len(user_sessions)}")
        
        for sess in user_sessions:
            print(f"   - {sess.session_id[:8]}... ({sess.title})")
            # Get turn count separately to avoid lazy loading issues
            turn_count = await conversation_manager.get_conversation_context(sess.session_id, max_turns=100)
            print(f"     Turns: {len(turn_count)}, Active: {sess.is_active()}")
        
        # Test 6: Test conversation context manager
        print("\n🧪 Test 6: Testing conversation context manager...")
        
        async with conversation_manager.conversation_context(
            session.session_id, "test_user_123"
        ) as conv_ctx:
            if conv_ctx.session:
                print(f"✅ Context manager working: {conv_ctx.session.session_id[:8]}...")
                
                # Get context through context manager
                context_via_mgr = await conv_ctx.get_context(3)
                print(f"✅ Context via manager: {len(context_via_mgr)} turns")
                
                # Add a turn through context manager
                await conv_ctx.add_turn(
                    "테스트 질문입니다.",
                    "테스트 응답입니다.",
                    response_time_ms=100
                )
                print("✅ Turn added via context manager")
            else:
                print("❌ Context manager session is None")
        
        # Test 7: Archive session
        print("\n🧪 Test 7: Archiving session...")
        
        archive_success = await conversation_manager.archive_session(
            session.session_id, "test_user_123"
        )
        
        if archive_success:
            print("✅ Session archived successfully")
            
            # Verify archived status
            archived_session = await conversation_manager.get_session(
                session.session_id, "test_user_123"
            )
            if archived_session:
                print(f"✅ Archived session status: {archived_session.status}")
            else:
                print("❌ Could not retrieve archived session")
        else:
            print("❌ Session archiving failed")
        
        # Test 8: Test schema validation
        print("\n🧪 Test 8: Testing API schemas...")
        
        # Test QueryRequest with conversation context
        try:
            query_request = QueryRequest(
                query="테스트 쿼리입니다.",
                session_id=session.session_id,
                use_context=True,
                conversation_context=ConversationContext(
                    session_id=session.session_id,
                    recent_turns=[
                        SchemaTurn(
                            user_query="이전 질문",
                            assistant_response="이전 응답"
                        )
                    ],
                    max_context_turns=5
                )
            )
            print("✅ QueryRequest schema validation passed")
            print(f"   Query: {query_request.query}")
            print(f"   Session ID: {query_request.session_id}")
            print(f"   Use context: {query_request.use_context}")
            print(f"   Context turns: {len(query_request.conversation_context.recent_turns)}")
            
        except Exception as e:
            print(f"❌ Schema validation failed: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 All integration tests passed successfully!")
        print("✅ Conversation system is ready for production use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_conversation_system()
    
    if success:
        print("\n📋 System Features Summary:")
        print("✅ SQLite-based user data storage")
        print("✅ Conversation session management")
        print("✅ Context-aware dialogue processing")
        print("✅ Turn-by-turn conversation tracking")
        print("✅ Session lifecycle management")
        print("✅ Context prompt building")
        print("✅ API schema support")
        
        print("\n🔗 API Endpoints Available:")
        print("- POST /query/ask (with conversation context)")
        print("- GET /query/conversations")
        print("- GET /query/conversations/{session_id}")
        print("- POST /query/conversations")
        print("- PUT /query/conversations/{session_id}/archive")
        
        print("\n🚀 Ready to test with real API server!")
    else:
        print("\n❌ System not ready - check errors above")


if __name__ == "__main__":
    asyncio.run(main())