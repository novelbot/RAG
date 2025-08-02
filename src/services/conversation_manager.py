"""
Conversation management service for handling continuous dialogue sessions.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..core.user_database import get_user_session
from ..models.conversation import ConversationSession, ConversationTurn, ConversationStatus
from ..core.logging import LoggerMixin


class ConversationManager(LoggerMixin):
    """
    Service for managing conversation sessions and context.
    
    Provides functionality for:
    - Creating and managing conversation sessions
    - Adding conversation turns
    - Retrieving conversation context
    - Session cleanup and archiving
    """
    
    def __init__(self):
        """Initialize the conversation manager"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
    
    async def create_session(self, user_id: str, title: Optional[str] = None, 
                           expires_in_hours: int = 24,
                           max_turns: int = 50,
                           context_window: int = 5) -> ConversationSession:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            title: Optional session title
            expires_in_hours: Session expiration time in hours
            max_turns: Maximum turns allowed in session
            context_window: Number of recent turns to include in context
            
        Returns:
            Created conversation session
        """
        db = get_user_session()
        try:
            session = ConversationSession.create_session(
                user_id=user_id,
                title=title,
                expires_in_hours=expires_in_hours,
                max_turns=max_turns,
                context_window=context_window
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            self._logger.info(f"Created conversation session {session.session_id} for user {user_id}")
            return session
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to create conversation session: {e}")
            raise
        finally:
            db.close()
    
    async def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[ConversationSession]:
        """
        Get conversation session by ID.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID for authorization
            
        Returns:
            Conversation session or None if not found
        """
        db = get_user_session()
        try:
            query = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            )
            
            if user_id:
                query = query.filter(ConversationSession.user_id == user_id)
            
            session = query.first()
            return session
            
        except Exception as e:
            self._logger.error(f"Failed to get conversation session: {e}")
            raise
        finally:
            db.close()
    
    async def get_or_create_session(self, session_id: Optional[str], user_id: str,
                                  auto_create: bool = True) -> Optional[ConversationSession]:
        """
        Get existing session or create new one if not found.
        
        Args:
            session_id: Optional session identifier
            user_id: User identifier
            auto_create: Whether to create new session if not found
            
        Returns:
            Conversation session or None
        """
        if session_id:
            session = await self.get_session(session_id, user_id)
            if session and session.is_active():
                return session
            elif session and session.is_expired():
                self._logger.info(f"Session {session_id} is expired")
        
        if auto_create:
            return await self.create_session(user_id)
        
        return None
    
    async def add_turn(self, session_id: str, user_query: str, assistant_response: str,
                      response_time_ms: Optional[int] = None,
                      token_count: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """
        Add a conversation turn to a session.
        
        Args:
            session_id: Session identifier
            user_query: User's query
            assistant_response: Assistant's response
            response_time_ms: Response time in milliseconds
            token_count: Number of tokens used
            metadata: Additional metadata
            
        Returns:
            Created conversation turn
        """
        db = get_user_session()
        try:
            session = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            ).first()
            
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if not session.is_active():
                raise ValueError(f"Session {session_id} is not active")
            
            turn = session.add_turn(user_query, assistant_response, metadata)
            
            # Update performance metrics
            if response_time_ms is not None:
                turn.response_time_ms = response_time_ms
            if token_count is not None:
                turn.token_count = token_count
            
            db.commit()
            db.refresh(turn)
            
            self._logger.debug(f"Added turn {turn.turn_number} to session {session_id}")
            return turn
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to add conversation turn: {e}")
            raise
        finally:
            db.close()
    
    async def get_conversation_context(self, session_id: str, max_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns to include
            
        Returns:
            List of conversation turns in context format
        """
        db = get_user_session()
        try:
            session = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            ).first()
            
            if not session:
                return []
            
            # Get recent turns ordered by creation time (most recent first)
            recent_turns = db.query(ConversationTurn).filter(
                ConversationTurn.session_id == session.id
            ).order_by(desc(ConversationTurn.created_at)).limit(max_turns).all()
            
            # Reverse to get chronological order (oldest first)
            recent_turns.reverse()
            
            return [turn.to_context_format() for turn in recent_turns]
            
        except Exception as e:
            self._logger.error(f"Failed to get conversation context: {e}")
            return []
        finally:
            db.close()
    
    async def get_user_sessions(self, user_id: str, limit: int = 20, 
                              include_inactive: bool = False) -> List[ConversationSession]:
        """
        Get conversation sessions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            include_inactive: Whether to include inactive sessions
            
        Returns:
            List of conversation sessions
        """
        db = get_user_session()
        try:
            query = db.query(ConversationSession).filter(
                ConversationSession.user_id == user_id
            )
            
            if not include_inactive:
                query = query.filter(
                    ConversationSession.status == ConversationStatus.ACTIVE.value
                )
            
            sessions = query.order_by(desc(ConversationSession.last_activity_at))\
                          .limit(limit).all()
            
            return sessions
            
        except Exception as e:
            self._logger.error(f"Failed to get user sessions: {e}")
            return []
        finally:
            db.close()
    
    async def archive_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """
        Archive a conversation session.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID for authorization
            
        Returns:
            True if session was archived, False otherwise
        """
        db = get_user_session()
        try:
            query = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            )
            
            if user_id:
                query = query.filter(ConversationSession.user_id == user_id)
            
            session = query.first()
            if not session:
                return False
            
            session.status = ConversationStatus.ARCHIVED.value
            db.commit()
            
            self._logger.info(f"Archived conversation session {session_id}")
            return True
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to archive conversation session: {e}")
            return False
        finally:
            db.close()
    
    async def cleanup_expired_sessions(self, batch_size: int = 100) -> int:
        """
        Clean up expired conversation sessions.
        
        Args:
            batch_size: Number of sessions to process in one batch
            
        Returns:
            Number of sessions cleaned up
        """
        db = get_user_session()
        try:
            expired_sessions = db.query(ConversationSession).filter(
                and_(
                    ConversationSession.expires_at < datetime.utcnow(),
                    ConversationSession.status == ConversationStatus.ACTIVE.value
                )
            ).limit(batch_size).all()
            
            count = 0
            for session in expired_sessions:
                session.status = ConversationStatus.INACTIVE.value
                count += 1
            
            db.commit()
            
            if count > 0:
                self._logger.info(f"Cleaned up {count} expired conversation sessions")
            
            return count
            
        except Exception as e:
            db.rollback()
            self._logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
        finally:
            db.close()
    
    def build_context_prompt(self, conversation_context: List[Dict[str, Any]], 
                           current_query: str) -> str:
        """
        Build a context-aware prompt from conversation history.
        
        Args:
            conversation_context: List of previous conversation turns
            current_query: Current user query
            
        Returns:
            Context-enhanced prompt
        """
        if not conversation_context:
            return current_query
        
        context_parts = ["Previous conversation:"]
        
        for turn in conversation_context:
            context_parts.append(f"User: {turn['user_query']}")
            context_parts.append(f"Assistant: {turn['assistant_response']}")
        
        context_parts.append(f"\nCurrent question: {current_query}")
        
        return "\n".join(context_parts)
    
    @asynccontextmanager
    async def conversation_context(self, session_id: Optional[str], user_id: str,
                                 auto_create: bool = True):
        """
        Context manager for conversation operations.
        
        Usage:
            async with conversation_manager.conversation_context(session_id, user_id) as ctx:
                response = await process_query_with_context(ctx.session, query)
                await ctx.add_turn(query, response)
        """
        session = await self.get_or_create_session(session_id, user_id, auto_create)
        
        class ConversationContext:
            def __init__(self, session: ConversationSession, manager: ConversationManager):
                self.session = session
                self.manager = manager
            
            async def add_turn(self, user_query: str, assistant_response: str, **kwargs):
                if self.session:
                    return await self.manager.add_turn(
                        self.session.session_id, user_query, assistant_response, **kwargs
                    )
                return None
            
            async def get_context(self, max_turns: int = 5):
                if self.session:
                    return await self.manager.get_conversation_context(
                        self.session.session_id, max_turns
                    )
                return []
        
        try:
            yield ConversationContext(session, self)
        except Exception as e:
            self._logger.error(f"Error in conversation context: {e}")
            raise


# Global conversation manager instance
conversation_manager = ConversationManager()