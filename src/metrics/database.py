"""
Database schema and connection management for metrics collection.
"""

import sqlite3
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
import json
import logging

logger = logging.getLogger(__name__)

class MetricsDatabase:
    """SQLite database for storing system metrics and analytics data."""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = Path(db_path)
        self._connection_pool = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize database with required tables."""
        async with self._get_connection() as conn:
            await self._create_tables(conn)
            await self._create_indexes(conn)
            logger.info(f"Metrics database initialized at {self.db_path}")
    
    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        """Get database connection with proper async handling."""
        async with self._lock:
            # Create connection in thread-safe manner
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                conn.close()
    
    async def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all required tables for metrics storage."""
        
        # Daily aggregated metrics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
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
        
        # Individual query logs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
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
        
        # Document events (upload, delete, update)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_events (
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
        
        # User session tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_end TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System events for recent activity feed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user_id TEXT,
                admin_user_id TEXT,
                description TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics sampling
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage_percent REAL DEFAULT 0.0,
                memory_usage_percent REAL DEFAULT 0.0,
                storage_usage_percent REAL DEFAULT 0.0,
                active_connections INTEGER DEFAULT 0,
                cache_hit_rate REAL DEFAULT 0.0,
                error_rate REAL DEFAULT 0.0
            )
        """)
    
    async def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for optimal query performance."""
        
        # Indexes for time-series queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_query_logs_user_id ON query_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_document_events_timestamp ON document_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_document_events_user_id ON document_events(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_start ON user_sessions(session_start)",
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    # Query logging methods
    async def log_query(
        self,
        user_id: str,
        query_text: str,
        response_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        result_count: int = 0,
        tokens_used: int = 0,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log a RAG query event."""
        async with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO query_logs (
                    user_id, query_text, response_time_ms, success,
                    error_message, result_count, tokens_used, ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, query_text, response_time_ms, success, 
                  error_message, result_count, tokens_used, ip_address, user_agent))
    
    # Document event methods
    async def log_document_event(
        self,
        document_id: str,
        filename: str,
        event_type: str,
        user_id: str,
        file_size_bytes: int = 0,
        processing_time_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a document upload/delete/update event."""
        metadata_json = json.dumps(metadata) if metadata else None
        async with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO document_events (
                    document_id, filename, event_type, user_id,
                    file_size_bytes, processing_time_ms, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (document_id, filename, event_type, user_id,
                  file_size_bytes, processing_time_ms, metadata_json))
    
    # User session methods
    async def start_user_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> int:
        """Start a new user session and return session ID."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO user_sessions (user_id, ip_address, user_agent)
                VALUES (?, ?, ?)
            """, (user_id, ip_address, user_agent))
            return cursor.lastrowid
    
    async def end_user_session(self, session_id: int) -> None:
        """End a user session."""
        async with self._get_connection() as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET session_end = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (session_id,))
    
    async def update_user_activity(self, user_id: str) -> None:
        """Update last activity timestamp for active sessions."""
        async with self._get_connection() as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET last_activity = CURRENT_TIMESTAMP 
                WHERE user_id = ? AND session_end IS NULL
            """, (user_id,))
    
    # System event methods
    async def log_system_event(
        self,
        event_type: str,
        description: str,
        user_id: Optional[str] = None,
        admin_user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a system event for recent activity display."""
        details_json = json.dumps(details) if details else None
        async with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO system_events (
                    event_type, user_id, admin_user_id, description, details
                ) VALUES (?, ?, ?, ?, ?)
            """, (event_type, user_id, admin_user_id, description, details_json))
    
    # Performance metrics methods
    async def log_performance_metrics(
        self,
        cpu_usage_percent: float,
        memory_usage_percent: float,
        storage_usage_percent: float,
        active_connections: int = 0,
        cache_hit_rate: float = 0.0,
        error_rate: float = 0.0
    ) -> None:
        """Log system performance metrics."""
        async with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO performance_metrics (
                    cpu_usage_percent, memory_usage_percent, storage_usage_percent,
                    active_connections, cache_hit_rate, error_rate
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (cpu_usage_percent, memory_usage_percent, storage_usage_percent,
                  active_connections, cache_hit_rate, error_rate))
    
    # Analytics and reporting methods
    async def get_daily_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily aggregated metrics for the last N days."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM daily_metrics 
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC
            """.format(days))
            return [dict(row) for row in cursor.fetchall()]
    
    async def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query logs."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM query_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    async def get_recent_system_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent system events for activity feed."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM system_events 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    async def get_active_users_count(self, minutes: int = 30) -> int:
        """Get count of users active in the last N minutes."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT user_id) 
                FROM user_sessions 
                WHERE last_activity >= datetime('now', '-{} minutes')
            """.format(minutes))
            result = cursor.fetchone()
            return result[0] if result else 0
    
    async def get_current_document_count(self) -> int:
        """Get current total document count."""
        async with self._get_connection() as conn:
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
            result = cursor.fetchone()
            return max(0, result[0] if result else 0)
    
    async def get_query_stats(self, days: int = 1) -> Dict[str, Any]:
        """Get query statistics for the last N days."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(DISTINCT user_id) as unique_users
                FROM query_logs 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            
            result = cursor.fetchone()
            if result:
                return {
                    'total_queries': result[0] or 0,
                    'successful_queries': result[1] or 0,
                    'failed_queries': result[2] or 0,
                    'avg_response_time_ms': round(result[3] or 0, 2),
                    'unique_users': result[4] or 0,
                    'success_rate': (result[1] or 0) / max(1, result[0] or 1)
                }
            return {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'avg_response_time_ms': 0.0,
                'unique_users': 0,
                'success_rate': 0.0
            }
    
    async def get_daily_query_trends(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily query count trends."""
        async with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    date(timestamp) as date,
                    COUNT(*) as query_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    AVG(response_time_ms) as avg_response_time
                FROM query_logs 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY date(timestamp)
                ORDER BY date ASC
            """.format(days))
            
            return [
                {
                    'date': row[0],
                    'query_count': row[1],
                    'successful_queries': row[2],
                    'avg_response_time_ms': round(row[3] or 0, 2)
                }
                for row in cursor.fetchall()
            ]
    
    async def aggregate_daily_metrics(self, target_date: Optional[date] = None) -> None:
        """Aggregate daily metrics from detailed logs."""
        if target_date is None:
            target_date = date.today()
        
        date_str = target_date.strftime('%Y-%m-%d')
        
        async with self._get_connection() as conn:
            # Get query statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_queries,
                    AVG(response_time_ms) as avg_response_time,
                    COUNT(DISTINCT user_id) as unique_users
                FROM query_logs 
                WHERE date(timestamp) = ?
            """, (date_str,))
            
            query_stats = cursor.fetchone()
            
            # Get document statistics
            cursor = conn.execute("""
                SELECT 
                    SUM(CASE WHEN event_type = 'upload' THEN 1 ELSE 0 END) as docs_added,
                    SUM(CASE WHEN event_type = 'delete' THEN 1 ELSE 0 END) as docs_deleted
                FROM document_events 
                WHERE date(timestamp) = ?
            """, (date_str,))
            
            doc_stats = cursor.fetchone()
            
            # Get total document count up to this date
            total_docs = await self.get_current_document_count()
            
            # Insert or update daily metrics
            conn.execute("""
                INSERT OR REPLACE INTO daily_metrics (
                    date, total_documents, documents_added, documents_deleted,
                    total_queries, successful_queries, failed_queries,
                    unique_users, avg_query_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                total_docs,
                doc_stats[0] or 0,
                doc_stats[1] or 0,
                query_stats[0] or 0,
                query_stats[1] or 0,
                query_stats[2] or 0,
                query_stats[4] or 0,
                query_stats[3] or 0.0
            ))


# Global metrics database instance
_metrics_db: Optional[MetricsDatabase] = None

async def get_metrics_db() -> MetricsDatabase:
    """Get or create the global metrics database instance."""
    global _metrics_db
    if _metrics_db is None:
        _metrics_db = MetricsDatabase()
        await _metrics_db.initialize()
    return _metrics_db

async def init_metrics_db(db_path: str = "metrics.db") -> MetricsDatabase:
    """Initialize the metrics database."""
    global _metrics_db
    _metrics_db = MetricsDatabase(db_path)
    await _metrics_db.initialize()
    return _metrics_db