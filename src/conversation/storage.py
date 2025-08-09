"""
Conversation storage using SQLite for persistence.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    conversation_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': json.dumps(self.metadata) if self.metadata else '{}'
        }


class ConversationStorage:
    """SQLite-based conversation storage."""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        """
        Initialize conversation storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE(conversation_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            ''')
            
            # Create indexes for faster queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_id 
                ON messages(conversation_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON messages(timestamp)
            ''')
            
            conn.commit()
    
    async def create_conversation(self, conversation_id: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Unique conversation ID
            user_id: User ID
            metadata: Optional metadata
        """
        def _create():
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute('''
                    INSERT OR IGNORE INTO conversations (conversation_id, user_id, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (conversation_id, user_id, now, now, json.dumps(metadata) if metadata else '{}'))
                conn.commit()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _create)
    
    async def add_message(self, message: ConversationMessage) -> None:
        """
        Add a message to a conversation.
        
        Args:
            message: Message to add
        """
        def _add():
            with sqlite3.connect(self.db_path) as conn:
                # Add message
                conn.execute('''
                    INSERT INTO messages (conversation_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    json.dumps(message.metadata) if message.metadata else '{}'
                ))
                
                # Update conversation updated_at
                conn.execute('''
                    UPDATE conversations 
                    SET updated_at = ? 
                    WHERE conversation_id = ?
                ''', (message.timestamp, message.conversation_id))
                
                conn.commit()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _add)
    
    async def get_messages(self, conversation_id: str, limit: int = 10) -> List[ConversationMessage]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to retrieve (most recent)
            
        Returns:
            List of messages
        """
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (conversation_id, limit))
                
                messages = []
                for row in cursor:
                    messages.append(ConversationMessage(
                        conversation_id=row['conversation_id'],
                        role=row['role'],
                        content=row['content'],
                        timestamp=row['timestamp'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    ))
                
                # Return in chronological order
                return list(reversed(messages))
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _get)
    
    async def get_conversation_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation information.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation info or None if not found
        """
        def _get():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM conversations 
                    WHERE conversation_id = ?
                ''', (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'conversation_id': row['conversation_id'],
                        'user_id': row['user_id'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    }
                return None
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _get)
    
    async def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: Conversation ID to delete
        """
        def _delete():
            with sqlite3.connect(self.db_path) as conn:
                # Delete messages first (foreign key constraint)
                conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
                # Delete conversation
                conn.execute('DELETE FROM conversations WHERE conversation_id = ?', (conversation_id,))
                conn.commit()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _delete)
    
    async def update_conversation_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update conversation metadata.
        
        Args:
            conversation_id: Conversation ID
            metadata: New metadata dictionary
        """
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute('''
                    UPDATE conversations 
                    SET metadata = ?, updated_at = ?
                    WHERE conversation_id = ?
                ''', (json.dumps(metadata), now, conversation_id))
                conn.commit()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _update)
    
    async def list_conversations(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List conversations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation info
        """
        def _list():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT c.*, COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                    WHERE c.user_id = ?
                    GROUP BY c.conversation_id
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                ''', (user_id, limit))
                
                conversations = []
                for row in cursor:
                    conversations.append({
                        'conversation_id': row['conversation_id'],
                        'user_id': row['user_id'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    })
                
                return conversations
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _list)
    
    def close(self):
        """Close the executor."""
        self.executor.shutdown(wait=True)


# Global conversation storage instance
conversation_storage = ConversationStorage()