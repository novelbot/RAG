CREATE TABLE conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE(conversation_id)
                );
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );
CREATE INDEX idx_conversation_id 
                ON messages(conversation_id)
            ;
CREATE INDEX idx_timestamp 
                ON messages(timestamp)
            ;
