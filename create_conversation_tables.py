#!/usr/bin/env python3
"""
Database migration script to create conversation tables.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sqlalchemy import create_engine, text
from src.core.database import Base, init_database
from src.core.config import get_config

# Import models to register them with Base
from src.models.conversation import ConversationSession, ConversationTurn
from src.models.query_log import QueryLog


def create_tables():
    """Create conversation tables in the database."""
    
    print("🔄 Initializing database connection...")
    
    # Initialize database
    init_database()
    
    # Get config and create engine
    config = get_config()
    
    # Build database URL
    if config.database.password:
        database_url = f"{config.database.driver}://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.name}"
    else:
        database_url = f"{config.database.driver}://{config.database.user}@{config.database.host}:{config.database.port}/{config.database.name}"
    
    engine = create_engine(database_url, pool_pre_ping=True)
    
    print(f"📊 Connected to database: {config.database.host}:{config.database.port}/{config.database.name}")
    
    try:
        # Check existing tables
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """))
            existing_tables = [row[0] for row in result]
            
        print(f"📋 Existing tables: {existing_tables}")
        
        # Create all tables
        print("🏗️ Creating conversation tables...")
        Base.metadata.create_all(bind=engine)
        
        # Check tables after creation
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """))
            final_tables = [row[0] for row in result]
            
        new_tables = set(final_tables) - set(existing_tables)
        
        print("✅ Database migration completed!")
        print(f"📊 Total tables: {len(final_tables)}")
        
        if new_tables:
            print(f"🆕 New tables created: {list(new_tables)}")
        else:
            print("ℹ️ No new tables were created (already exist)")
        
        # Verify conversation tables
        conversation_tables = [t for t in final_tables if 'conversation' in t]
        if conversation_tables:
            print(f"🗣️ Conversation tables: {conversation_tables}")
        else:
            print("⚠️ Warning: No conversation tables found!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
    
    finally:
        engine.dispose()


def verify_tables():
    """Verify that conversation tables are created with correct structure."""
    
    print("\n🔍 Verifying table structure...")
    
    # Initialize database
    init_database()
    
    # Get config and create engine
    config = get_config()
    
    # Build database URL
    if config.database.password:
        database_url = f"{config.database.driver}://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.name}"
    else:
        database_url = f"{config.database.driver}://{config.database.user}@{config.database.host}:{config.database.port}/{config.database.name}"
    
    engine = create_engine(database_url, pool_pre_ping=True)
    
    try:
        with engine.connect() as conn:
            # Check conversation_sessions table
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'conversation_sessions'
                ORDER BY ordinal_position
            """))
            
            sessions_columns = result.fetchall()
            
            if sessions_columns:
                print("✅ conversation_sessions table structure:")
                for col_name, data_type, nullable in sessions_columns:
                    print(f"   - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
            else:
                print("❌ conversation_sessions table not found!")
            
            # Check conversation_turns table
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'conversation_turns'
                ORDER BY ordinal_position
            """))
            
            turns_columns = result.fetchall()
            
            if turns_columns:
                print("\n✅ conversation_turns table structure:")
                for col_name, data_type, nullable in turns_columns:
                    print(f"   - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
            else:
                print("❌ conversation_turns table not found!")
            
            return bool(sessions_columns and turns_columns)
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False
    
    finally:
        engine.dispose()


def main():
    """Main migration function."""
    
    print("🚀 Starting conversation system database migration...")
    print("=" * 60)
    
    # Create tables
    if create_tables():
        # Verify tables
        if verify_tables():
            print("\n🎉 Migration completed successfully!")
            print("✅ Conversation system is ready for testing")
        else:
            print("\n⚠️ Migration completed but verification failed")
            print("❌ Please check table structure manually")
    else:
        print("\n❌ Migration failed!")
        print("Please check database connection and permissions")
    
    print("=" * 60)


if __name__ == "__main__":
    main()