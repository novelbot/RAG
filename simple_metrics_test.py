#!/usr/bin/env python3
"""
Simple test script for metrics database functionality.
"""

import asyncio
import sys
import os
import random
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.metrics.database import MetricsDatabase


async def test_database_functionality():
    """Test the core database functionality."""
    print("🚀 Testing metrics database functionality...\n")
    
    # Initialize database
    print("📊 Initializing metrics database...")
    db = MetricsDatabase("test_metrics.db")
    await db.initialize()
    print("✅ Database initialized")
    
    # Test user sessions
    print("\n👥 Testing user sessions...")
    users = ["john.doe", "jane.smith", "mike.jones"]
    
    for user in users:
        session_id = await db.start_user_session(
            user_id=user,
            ip_address=f"192.168.1.{random.randint(10, 50)}",
            user_agent="Test Browser"
        )
        print(f"  Created session for {user} (ID: {session_id})")
    
    # Test document events
    print("\n📄 Testing document events...")
    documents = [
        ("doc_001", "report.pdf", 1024000),
        ("doc_002", "manual.docx", 512000),
        ("doc_003", "data.xlsx", 256000)
    ]
    
    for doc_id, filename, size in documents:
        user = random.choice(users)
        await db.log_document_event(
            document_id=doc_id,
            filename=filename,
            event_type="upload",
            user_id=user,
            file_size_bytes=size,
            processing_time_ms=random.randint(500, 2000)
        )
        print(f"  Logged document upload: {filename} by {user}")
    
    # Test queries
    print("\n🔍 Testing query logs...")
    queries = [
        "What is the revenue trend?",
        "How many employees do we have?",
        "Show me the financial summary"
    ]
    
    for _ in range(10):
        user = random.choice(users)
        query = random.choice(queries)
        success = random.random() > 0.1  # 90% success rate
        
        await db.log_query(
            user_id=user,
            query_text=query,
            response_time_ms=random.randint(200, 2000),
            success=success,
            error_message=None if success else "Query timeout",
            result_count=random.randint(1, 10) if success else 0,
            tokens_used=random.randint(50, 500)
        )
    
    print(f"  Logged 10 sample queries")
    
    # Test system events
    print("\n⚙️  Testing system events...")
    events = [
        ("user_login", "john.doe", "User logged in successfully"),
        ("document_uploaded", "jane.smith", "Uploaded new document"),
        ("config_changed", "admin", "Updated system configuration")
    ]
    
    for event_type, user, description in events:
        await db.log_system_event(
            event_type=event_type,
            user_id=user,
            description=description
        )
        print(f"  Logged system event: {description}")
    
    # Test performance metrics
    print("\n📈 Testing performance metrics...")
    for i in range(5):
        await db.log_performance_metrics(
            cpu_usage_percent=random.uniform(10, 80),
            memory_usage_percent=random.uniform(30, 90),
            storage_usage_percent=random.uniform(45, 85)
        )
    
    print("  Logged 5 performance metric samples")
    
    # Test data retrieval
    print("\n📊 Testing data retrieval...")
    
    # Test document count
    doc_count = await db.get_current_document_count()
    print(f"  Current document count: {doc_count}")
    
    # Test active users
    active_users = await db.get_active_users_count(minutes=60)
    print(f"  Active users (last hour): {active_users}")
    
    # Test query stats
    query_stats = await db.get_query_stats(days=1)
    print(f"  Query stats: {query_stats}")
    
    # Test recent events
    recent_events = await db.get_recent_system_events(limit=5)
    print(f"  Recent events: {len(recent_events)} events")
    for event in recent_events[:3]:
        print(f"    - {event['description']}")
    
    # Test daily aggregation
    print("\n📊 Testing daily aggregation...")
    await db.aggregate_daily_metrics()
    print("  Daily metrics aggregated")
    
    # Test daily metrics retrieval
    daily_metrics = await db.get_daily_metrics(days=7)
    print(f"  Daily metrics: {len(daily_metrics)} days of data")
    
    print("\n✅ All database tests completed successfully!")
    
    return db


async def main():
    """Main test function."""
    try:
        db = await test_database_functionality()
        
        print("\n💡 Database test completed!")
        print("📄 Test database created at: test_metrics.db")
        print("🎯 Next steps:")
        print("1. You can now integrate this into your FastAPI app")
        print("2. Start the WebUI to see the dashboard with real data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())