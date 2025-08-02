#!/usr/bin/env python3
"""
Setup SQLite database for user data and conversations.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_user_database():
    """Setup SQLite database for user data."""
    
    print("🚀 Setting up SQLite user database...")
    print("=" * 60)
    
    try:
        # Import will initialize the database
        from src.core.user_database import init_user_database, get_user_db_path, get_user_session
        from src.models.conversation import ConversationSession, ConversationTurn
        from src.models.query_log import QueryLog
        
        # Initialize database (creates tables)
        init_user_database()
        
        db_path = get_user_db_path()
        print(f"📊 SQLite database created at: {db_path}")
        
        # Test database connection
        from sqlalchemy import text
        with get_user_session() as db:
            # Test query to verify tables exist
            result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            tables = [row[0] for row in result]
            
        print(f"✅ Tables created: {tables}")
        
        # Verify specific tables
        expected_tables = ['conversation_sessions', 'conversation_turns', 'query_logs']
        missing_tables = [t for t in expected_tables if t not in tables]
        
        if missing_tables:
            print(f"⚠️ Missing tables: {missing_tables}")
        else:
            print("🎉 All required tables are present!")
        
        # Show table schemas
        print("\n📋 Table schemas:")
        with get_user_session() as db:
            for table in expected_tables:
                if table in tables:
                    schema = db.execute(text(f"PRAGMA table_info({table})")).fetchall()
                    print(f"\n  {table}:")
                    for col in schema:
                        col_name, col_type, not_null, default, pk = col[1], col[2], col[3], col[4], col[5]
                        constraints = []
                        if pk:
                            constraints.append("PRIMARY KEY")
                        if not_null:
                            constraints.append("NOT NULL")
                        if default:
                            constraints.append(f"DEFAULT {default}")
                        
                        constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                        print(f"    - {col_name}: {col_type}{constraint_str}")
        
        print("\n" + "=" * 60)
        print("✅ User database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up user database: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_operations():
    """Test basic database operations."""
    
    print("\n🧪 Testing database operations...")
    
    try:
        from src.services.conversation_manager import conversation_manager
        import asyncio
        
        async def run_tests():
            # Test creating a conversation session
            session = await conversation_manager.create_session(
                user_id="test_user",
                title="Test Conversation"
            )
            print(f"✅ Created test session: {session.session_id}")
            
            # Test adding a turn
            turn = await conversation_manager.add_turn(
                session_id=session.session_id,
                user_query="Hello, this is a test",
                assistant_response="Hello! This is a test response."
            )
            print(f"✅ Added test turn: {turn.turn_number}")
            
            # Test getting context
            context = await conversation_manager.get_conversation_context(session.session_id)
            print(f"✅ Retrieved context: {len(context)} turns")
            
            # Test getting user sessions
            sessions = await conversation_manager.get_user_sessions("test_user")
            print(f"✅ Retrieved user sessions: {len(sessions)}")
            
            return True
        
        # Run async tests
        result = asyncio.run(run_tests())
        
        if result:
            print("🎉 All database operations working correctly!")
        
        return result
        
    except Exception as e:
        print(f"❌ Database operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function."""
    
    # Setup database
    if setup_user_database():
        # Test operations
        test_database_operations()
    else:
        print("❌ Setup failed - skipping tests")


if __name__ == "__main__":
    main()