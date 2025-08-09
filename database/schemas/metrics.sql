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
        );
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
        );
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
        );
CREATE TABLE system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            user_id TEXT,
            admin_user_id TEXT,
            description TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_end TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
CREATE INDEX idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX idx_document_events_timestamp ON document_events(timestamp);
CREATE INDEX idx_system_events_timestamp ON system_events(timestamp);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE TABLE performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage_percent REAL DEFAULT 0.0,
                memory_usage_percent REAL DEFAULT 0.0,
                storage_usage_percent REAL DEFAULT 0.0,
                active_connections INTEGER DEFAULT 0,
                cache_hit_rate REAL DEFAULT 0.0,
                error_rate REAL DEFAULT 0.0
            );
CREATE INDEX idx_document_events_user_id ON document_events(user_id);
CREATE INDEX idx_system_events_type ON system_events(event_type);
CREATE INDEX idx_user_sessions_start ON user_sessions(session_start);
CREATE INDEX idx_performance_timestamp ON performance_metrics(timestamp);
