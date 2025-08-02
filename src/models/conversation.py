"""
Conversation session models for managing dialogue context.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.core.user_database import UserBase as Base


class ConversationStatus(Enum):
    """Conversation session status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class ConversationSession(Base):
    """Conversation session model for tracking continuous dialogues"""
    
    __tablename__ = "conversation_sessions"
    
    id = Column(Integer, primary_key=True)
    
    # Session identification
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Session metadata
    title = Column(String(200), nullable=True)  # Auto-generated from first query
    status = Column(String(20), nullable=False, default=ConversationStatus.ACTIVE.value)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # Optional expiration
    
    # Session configuration
    max_turns = Column(Integer, default=50, nullable=False)  # Maximum conversation turns
    context_window = Column(Integer, default=5, nullable=False)  # Number of recent turns to include
    
    # Session metadata
    session_metadata = Column(JSON, nullable=True)
    
    # Relationships
    turns = relationship("ConversationTurn", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"ConversationSession(id={self.id}, session_id='{self.session_id}', user_id='{self.user_id}', status='{self.status}')"
    
    @classmethod
    def create_session(cls, user_id: str, title: Optional[str] = None, 
                      expires_in_hours: int = 24, **kwargs) -> "ConversationSession":
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        return cls(
            session_id=session_id,
            user_id=user_id,
            title=title,
            expires_at=expires_at,
            **kwargs
        )
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if session is active and not expired"""
        return (self.status == ConversationStatus.ACTIVE.value and 
                not self.is_expired())
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()
    
    def get_recent_turns(self, limit: Optional[int] = None) -> List["ConversationTurn"]:
        """Get recent conversation turns"""
        limit = limit or self.context_window
        return sorted(self.turns, key=lambda t: t.created_at, reverse=True)[:limit]
    
    def add_turn(self, user_query: str, assistant_response: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> "ConversationTurn":
        """Add a new conversation turn"""
        turn = ConversationTurn.create_turn(
            session=self,
            user_query=user_query,
            assistant_response=assistant_response,
            metadata=metadata
        )
        self.turns.append(turn)
        self.update_activity()
        return turn
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_turns": self.max_turns,
            "context_window": self.context_window,
            "turn_count": len(self.turns),
            "is_active": self.is_active(),
            "is_expired": self.is_expired()
        }


class ConversationTurn(Base):
    """Individual conversation turn model"""
    
    __tablename__ = "conversation_turns"
    
    id = Column(Integer, primary_key=True)
    
    # Session reference
    session_id = Column(Integer, ForeignKey("conversation_sessions.id"), nullable=False)
    session = relationship("ConversationSession", back_populates="turns")
    
    # Turn data
    turn_number = Column(Integer, nullable=False)  # Sequential number within session
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    # Turn metadata
    turn_metadata = Column(JSON, nullable=True)  # Search results, sources, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"ConversationTurn(id={self.id}, session_id={self.session_id}, turn_number={self.turn_number})"
    
    @classmethod
    def create_turn(cls, session: ConversationSession, user_query: str, 
                   assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> "ConversationTurn":
        """Create a new conversation turn"""
        # Calculate turn number
        turn_number = len(session.turns) + 1
        
        return cls(
            session=session,
            turn_number=turn_number,
            user_query=user_query,
            assistant_response=assistant_response,
            turn_metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "response_time_ms": self.response_time_ms,
            "token_count": self.token_count,
            "turn_metadata": self.turn_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def to_context_format(self) -> Dict[str, str]:
        """Convert turn to format suitable for conversation context"""
        return {
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "timestamp": self.created_at.isoformat() if self.created_at else None
        }