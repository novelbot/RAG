#!/usr/bin/env python3
"""
Inspect RDB schema to understand what tables exist.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from sqlalchemy import inspect, text

def inspect_rdb_schema():
    """Inspect RDB schema and show what tables exist."""
    print("🔍 Inspecting RDB Schema")
    print("=" * 50)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize database
        db_manager = DatabaseManager(config.database)
        print("✅ Connected to RDB")
        
        # Get inspector
        inspector = inspect(db_manager.engine)
        
        # Get all table names
        table_names = inspector.get_table_names()
        print(f"\n📊 Found {len(table_names)} tables:")
        
        for table_name in sorted(table_names):
            print(f"  - {table_name}")
        
        # Check if episode/novel related tables exist
        novel_tables = [t for t in table_names if 'novel' in t.lower() or 'episode' in t.lower()]
        if novel_tables:
            print(f"\n📚 Novel/Episode related tables:")
            for table in novel_tables:
                print(f"  - {table}")
                
                # Show columns for novel/episode tables
                columns = inspector.get_columns(table)
                print(f"    Columns ({len(columns)}):")
                for col in columns:
                    print(f"      - {col['name']}: {col['type']}")
                print()
        
        # Try to get sample data from any novel/episode table
        if novel_tables:
            print(f"📊 Sample data from {novel_tables[0]}:")
            with db_manager.get_connection() as conn:
                result = conn.execute(text(f"SELECT * FROM {novel_tables[0]} LIMIT 5"))
                rows = result.fetchall()
                if rows:
                    # Get column names
                    columns = result.keys()
                    print(f"    Columns: {list(columns)}")
                    for i, row in enumerate(rows):
                        print(f"    Row {i+1}: {dict(row._mapping)}")
                else:
                    print("    No data found")
        
        # Check for any tables with 'content' or similar fields
        content_tables = []
        for table_name in table_names:
            try:
                columns = inspector.get_columns(table_name)
                col_names = [col['name'].lower() for col in columns]
                if any(keyword in col_names for keyword in ['content', 'title', 'text', 'episode', 'novel']):
                    content_tables.append(table_name)
            except Exception as e:
                print(f"    Error inspecting {table_name}: {e}")
        
        if content_tables:
            print(f"\n📝 Tables with content-related fields:")
            for table in content_tables:
                print(f"  - {table}")
        
        print(f"\n🎯 Summary:")
        print(f"  - Total tables: {len(table_names)}")
        print(f"  - Novel/Episode tables: {len(novel_tables)}")
        print(f"  - Content tables: {len(content_tables)}")
        
        # Try to understand the data structure
        if content_tables and not novel_tables:
            print(f"\n💡 Suggestion: The project might be using generic 'documents' table")
            print("    instead of specific novel/episode tables.")
            
            # Check documents table structure
            if 'documents' in table_names:
                print(f"\n📄 Documents table structure:")
                columns = inspector.get_columns('documents')
                for col in columns:
                    print(f"    - {col['name']}: {col['type']}")
                
                # Sample documents data
                print(f"\n📊 Sample documents data:")
                with db_manager.get_connection() as conn:
                    result = conn.execute(text("SELECT id, filename, metadata FROM documents LIMIT 3"))
                    rows = result.fetchall()
                    for i, row in enumerate(rows):
                        print(f"    Row {i+1}: {dict(row._mapping)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    inspect_rdb_schema()