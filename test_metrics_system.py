#!/usr/bin/env python3
"""
Test script for the metrics collection system.
This script creates sample data and tests the complete flow.
"""

import asyncio
import sys
import os
import random
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.metrics.database import init_metrics_db
from src.metrics.collectors import (
    document_collector,
    query_collector,
    session_collector,
    performance_collector
)


async def create_sample_data():
    """Create sample data for testing the metrics system."""
    print("🏗️  Creating sample data for metrics system...")
    
    # Initialize database
    print("📊 Initializing metrics database...")
    metrics_db = await init_metrics_db("metrics.db")
    print("✅ Database initialized")
    
    # Create sample user sessions
    print("👥 Creating sample user sessions...")
    users = ["john.doe", "jane.smith", "mike.jones", "sarah.wilson", "admin"]
    session_ids = {}
    
    for user in users:
        session_id = await session_collector.start_session(
            user_id=user,
            ip_address=f"192.168.1.{random.randint(10, 50)}",
            user_agent="Mozilla/5.0 (Test Browser)"
        )
        session_ids[user] = session_id
        print(f"  Created session for {user} (ID: {session_id})")
        await asyncio.sleep(0.1)
    
    # Create sample document events
    print("📄 Creating sample document events...")
    documents = [
        ("doc_001", "financial_report_2024.pdf", 2048000),
        ("doc_002", "user_manual_v2.docx", 512000),
        ("doc_003", "quarterly_summary.xlsx", 128000),
        ("doc_004", "technical_specs.md", 64000),
        ("doc_005", "meeting_notes.txt", 32000)
    ]
    
    for doc_id, filename, size in documents:
        user = random.choice(users)
        await document_collector.log_document_upload(
            document_id=doc_id,
            filename=filename,
            user_id=user,
            file_size_bytes=size,
            processing_time_ms=random.randint(500, 3000),
            metadata={"category": "business", "department": "finance"}
        )
        print(f"  Uploaded document: {filename} by {user}")
        await asyncio.sleep(0.1)
    
    # Create sample queries for the last 7 days
    print("🔍 Creating sample query logs...")
    base_date = datetime.now() - timedelta(days=7)
    
    queries = [
        "What is the revenue trend for this quarter?",
        "How many employees do we have?",
        "What are the key performance indicators?",
        "Show me the financial summary",
        "Who is responsible for the project management?",
        "What is our market share?",
        "How do we improve customer satisfaction?",
        "What are the upcoming milestones?",
        "Show me the budget allocation",
        "What are the security protocols?"
    ]
    
    for day in range(7):
        current_date = base_date + timedelta(days=day)
        # More queries on recent days
        query_count = random.randint(5, 20) + (day * 2)
        
        for _ in range(query_count):
            user = random.choice(users)
            query = random.choice(queries)
            success = random.random() > 0.1  # 90% success rate
            response_time = random.randint(200, 2000)
            result_count = random.randint(1, 10) if success else 0
            
            # Create query at random time during the day
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)
            query_time = current_date.replace(
                hour=hours_offset, 
                minute=minutes_offset, 
                second=random.randint(0, 59)
            )
            
            # Manually insert with specific timestamp
            await metrics_db.log_query(
                user_id=user,
                query_text=query,
                response_time_ms=response_time,
                success=success,
                error_message=None if success else "Query timeout",
                result_count=result_count,
                tokens_used=random.randint(50, 500),
                ip_address=f"192.168.1.{random.randint(10, 50)}",
                user_agent="Test Browser"
            )
        
        print(f"  Created {query_count} queries for {current_date.strftime('%Y-%m-%d')}")
    
    # Create some system events
    print("⚙️  Creating sample system events...")
    system_events = [
        ("user_created", "admin", "Created new user account", {"username": "new.user"}),
        ("config_changed", "admin", "Updated LLM provider settings", {"provider": "openai"}),
        ("document_deleted", "mike.jones", "Deleted outdated document", {"filename": "old_report.pdf"}),
        ("user_login", "jane.smith", "User logged in", {"ip": "192.168.1.25"}),
        ("backup_completed", "system", "Daily backup completed successfully", {"size_mb": 1024})
    ]
    
    for event_type, user, description, details in system_events:
        await metrics_db.log_system_event(
            event_type=event_type,
            description=description,
            user_id=user if user != "system" else None,
            admin_user_id="admin" if event_type in ["user_created", "config_changed"] else None,
            details=details
        )
        print(f"  Created system event: {description}")
        await asyncio.sleep(0.1)
    
    # Create performance metrics samples
    print("📈 Creating sample performance metrics...")
    for i in range(24):  # Last 24 hours
        timestamp = datetime.now() - timedelta(hours=23-i)
        await metrics_db.log_performance_metrics(
            cpu_usage_percent=random.uniform(10, 80),
            memory_usage_percent=random.uniform(30, 90),
            storage_usage_percent=random.uniform(45, 85),
            active_connections=random.randint(5, 50),
            cache_hit_rate=random.uniform(0.7, 0.95),
            error_rate=random.uniform(0.01, 0.05)
        )
    
    print("  Created 24 performance metric samples")
    
    # Aggregate daily metrics
    print("📊 Aggregating daily metrics...")
    for day in range(7):
        target_date = (datetime.now() - timedelta(days=6-day)).date()
        await metrics_db.aggregate_daily_metrics(target_date)
        print(f"  Aggregated metrics for {target_date}")
    
    print("✅ Sample data creation completed!")
    return metrics_db


async def test_api_endpoints():
    """Test the API endpoint functionality."""
    print("\n🧪 Testing API endpoint functionality...")
    
    try:
        from src.metrics.database import get_metrics_db
        
        metrics_db = await get_metrics_db()
        
        # Test document count
        doc_count = await metrics_db.get_current_document_count()
        print(f"📄 Current document count: {doc_count}")
        
        # Test active users
        active_users = await metrics_db.get_active_users_count(minutes=1440)  # Last 24 hours
        print(f"👥 Active users (24h): {active_users}")
        
        # Test query stats
        query_stats = await metrics_db.get_query_stats(days=1)
        print(f"🔍 Today's query stats: {query_stats}")
        
        # Test daily trends
        trends = await metrics_db.get_daily_query_trends(days=7)
        print(f"📈 Daily trends (7 days): {len(trends)} data points")
        
        # Test recent events
        recent_events = await metrics_db.get_recent_system_events(limit=5)
        print(f"📝 Recent events: {len(recent_events)} events")
        for event in recent_events[:3]:
            print(f"  - {event['description']} ({event['timestamp']})")
        
        print("✅ API endpoint tests completed successfully!")
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("🚀 Starting metrics system test...\n")
    
    try:
        # Create sample data
        await create_sample_data()
        
        # Test API functionality
        await test_api_endpoints()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Next steps:")
        print("1. Start your FastAPI server: uv run main.py")
        print("2. Start your Streamlit WebUI: uv run run_webui.py")
        print("3. Visit the dashboard to see real data!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())