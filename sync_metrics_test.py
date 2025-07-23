#!/usr/bin/env python3
"""
Synchronous test for metrics database functionality.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path


def create_metrics_database():
    """Create and initialize the metrics database synchronously."""
    print("📊 Creating metrics database...")
    
    db_path = Path("test_metrics.db")
    if db_path.exists():
        db_path.unlink()  # Remove existing file
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Create tables
    print("🏗️  Creating database tables...")
    
    # Daily metrics table
    conn.execute("""
        CREATE TABLE daily_metrics (
            date DATE PRIMARY KEY,
            total_documents INTEGER DEFAULT 0,
            documents_added INTEGER DEFAULT 0,
            documents_deleted INTEGER DEFAULT 0,
            total_queries INTEGER DEFAULT 0,
            successful_queries INTEGER DEFAULT 0,
            failed_queries INTEGER DEFAULT 0,
            unique_users INTEGER DEFAULT 0,
            avg_query_time_ms REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Query logs table
    conn.execute("""
        CREATE TABLE query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            response_time_ms INTEGER DEFAULT 0,
            success BOOLEAN DEFAULT FALSE,
            error_message TEXT,
            result_count INTEGER DEFAULT 0,
            tokens_used INTEGER DEFAULT 0,
            ip_address TEXT,
            user_agent TEXT
        )
    """)
    
    # Document events table
    conn.execute("""
        CREATE TABLE document_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN ('upload', 'delete', 'update')),
            user_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size_bytes INTEGER DEFAULT 0,
            processing_time_ms INTEGER DEFAULT 0,
            metadata TEXT
        )
    """)
    
    # System events table
    conn.execute("""
        CREATE TABLE system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            user_id TEXT,
            admin_user_id TEXT,
            description TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # User sessions table
    conn.execute("""
        CREATE TABLE user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_end TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    print("🔗 Creating database indexes...")
    
    indexes = [
        "CREATE INDEX idx_query_logs_timestamp ON query_logs(timestamp)",
        "CREATE INDEX idx_query_logs_user_id ON query_logs(user_id)",
        "CREATE INDEX idx_document_events_timestamp ON document_events(timestamp)",
        "CREATE INDEX idx_system_events_timestamp ON system_events(timestamp)",
        "CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id)"
    ]
    
    for index_sql in indexes:
        conn.execute(index_sql)
    
    conn.commit()
    print("✅ Database created successfully")
    
    return conn


def insert_sample_data(conn):
    """Insert sample data into the database."""
    print("📝 Inserting sample data...")
    
    users = ["john.doe", "jane.smith", "mike.jones", "sarah.wilson", "admin"]
    
    # Insert user sessions
    print("  Creating user sessions...")
    for user in users:
        conn.execute("""
            INSERT INTO user_sessions (user_id, ip_address, user_agent, last_activity)
            VALUES (?, ?, ?, ?)
        """, (user, f"192.168.1.{random.randint(10, 50)}", "Test Browser", datetime.now()))
    
    # Insert document events
    print("  Creating document events...")
    documents = [
        ("doc_001", "financial_report_2024.pdf", 2048000),
        ("doc_002", "user_manual_v2.docx", 512000),
        ("doc_003", "quarterly_summary.xlsx", 128000),
        ("doc_004", "technical_specs.md", 64000),
        ("doc_005", "meeting_notes.txt", 32000)
    ]
    
    for doc_id, filename, size in documents:
        user = random.choice(users)
        conn.execute("""
            INSERT INTO document_events (document_id, filename, event_type, user_id, file_size_bytes, processing_time_ms)
            VALUES (?, ?, 'upload', ?, ?, ?)
        """, (doc_id, filename, user, size, random.randint(500, 3000)))
    
    # Insert queries for the last 7 days
    print("  Creating query logs...")
    queries = [
        "What is the revenue trend for this quarter?",
        "How many employees do we have?",
        "What are the key performance indicators?",
        "Show me the financial summary",
        "Who is responsible for the project management?"
    ]
    
    base_date = datetime.now() - timedelta(days=7)
    
    for day in range(7):
        current_date = base_date + timedelta(days=day)
        query_count = random.randint(5, 20) + (day * 2)  # More queries on recent days
        
        for _ in range(query_count):
            user = random.choice(users)
            query = random.choice(queries)
            success = random.random() > 0.1  # 90% success rate
            response_time = random.randint(200, 2000)
            result_count = random.randint(1, 10) if success else 0
            
            # Random time during the day
            query_time = current_date + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            conn.execute("""
                INSERT INTO query_logs (user_id, query_text, timestamp, response_time_ms, success, 
                                      error_message, result_count, tokens_used, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user, query, query_time, response_time,
                success, None if success else "Query timeout",
                result_count, random.randint(50, 500),
                f"192.168.1.{random.randint(10, 50)}", "Test Browser"
            ))
    
    # Insert system events
    print("  Creating system events...")
    system_events = [
        ("user_created", "admin", "admin", "Created new user account"),
        ("config_changed", "admin", "admin", "Updated LLM provider settings"),
        ("document_deleted", "mike.jones", None, "Deleted outdated document"),
        ("user_login", "jane.smith", None, "User logged in successfully"),
        ("backup_completed", None, None, "Daily backup completed successfully")
    ]
    
    for event_type, user, admin_user, description in system_events:
        conn.execute("""
            INSERT INTO system_events (event_type, user_id, admin_user_id, description)
            VALUES (?, ?, ?, ?)
        """, (event_type, user, admin_user, description))
    
    conn.commit()
    print("✅ Sample data inserted successfully")


def test_data_retrieval(conn):
    """Test data retrieval functions."""
    print("🧪 Testing data retrieval...")
    
    # Test document count
    cursor = conn.execute("""
        SELECT COALESCE(
            (SELECT SUM(CASE 
                WHEN event_type = 'upload' THEN 1 
                WHEN event_type = 'delete' THEN -1 
                ELSE 0 
            END) FROM document_events), 
            0
        )
    """)
    doc_count = cursor.fetchone()[0]
    print(f"  📄 Total documents: {doc_count}")
    
    # Test active users
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT user_id) 
        FROM user_sessions 
        WHERE last_activity >= datetime('now', '-30 minutes')
    """)
    active_users = cursor.fetchone()[0]
    print(f"  👥 Active users (30 min): {active_users}")
    
    # Test query stats
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_queries,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries,
            AVG(response_time_ms) as avg_response_time,
            COUNT(DISTINCT user_id) as unique_users
        FROM query_logs 
        WHERE timestamp >= datetime('now', '-1 days')
    """)
    
    result = cursor.fetchone()
    print(f"  🔍 Today's queries: {result[0]}")
    print(f"  ✅ Successful: {result[1]}")
    print(f"  ❌ Failed: {result[2]}")
    print(f"  ⏱️  Avg response time: {result[3]:.1f}ms")
    print(f"  🆔 Unique users: {result[4]}")
    
    # Test daily trends
    cursor = conn.execute("""
        SELECT 
            date(timestamp) as date,
            COUNT(*) as query_count,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
            AVG(response_time_ms) as avg_response_time
        FROM query_logs 
        WHERE timestamp >= datetime('now', '-7 days')
        GROUP BY date(timestamp)
        ORDER BY date ASC
    """)
    
    trends = cursor.fetchall()
    print(f"  📈 Daily trends ({len(trends)} days):")
    for trend in trends:
        print(f"    {trend[0]}: {trend[1]} queries, {trend[2]} successful")
    
    # Test recent system events
    cursor = conn.execute("""
        SELECT description, timestamp, user_id
        FROM system_events 
        ORDER BY timestamp DESC 
        LIMIT 5
    """)
    
    events = cursor.fetchall()
    print(f"  📝 Recent events:")
    for event in events:
        print(f"    - {event[0]} ({event[2] or 'system'})")
    
    print("✅ Data retrieval tests completed")


def main():
    """Main test function."""
    print("🚀 Starting synchronous metrics test...\n")
    
    try:
        # Create database
        conn = create_metrics_database()
        
        # Insert sample data
        insert_sample_data(conn)
        
        # Test data retrieval
        test_data_retrieval(conn)
        
        conn.close()
        
        print("\n✅ All tests completed successfully!")
        print("📄 Test database created: test_metrics.db")
        print("🎯 You can now copy the structure to your main application")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())