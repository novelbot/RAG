"""
Authentication dependencies for FastAPI routes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Dict, Any, Optional
import sqlite3
from pathlib import Path

from .schemas import UserResponse
from .jwt_manager import JWTManager
from .exceptions import TokenExpiredError, InvalidTokenError, TokenBlacklistedError

security = HTTPBearer()
jwt_manager = JWTManager()


# Simple User class for SQLite auth
class SimpleUser:
    """Simple user class for SQLite authentication without SQLAlchemy complexity."""
    
    def __init__(self, user_id: int, username: str, email: str, role: str, is_active: bool = True):
        self.id = user_id
        self.username = username
        self.email = email
        self.is_active = is_active
        self.is_superuser = (role == 'admin')
        self.is_verified = True
        self.full_name = username
        self.hashed_password = ""
        self._role_name = role
        self.roles = []
        
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return self._role_name == role_name
    
    def is_locked(self) -> bool:
        """Check if user is locked (always False for SQLite users)."""
        return False


async def get_current_user(
    token: str = Depends(security)
) -> SimpleUser:
    """
    Get current authenticated user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        SimpleUser: Current authenticated user
        
    Raises:
        HTTPException: 401 if token is invalid
    """
    # JWT Manager 토큰 검증
    try:
        token_payload = jwt_manager.validate_token(token.credentials, token_type="access")
        
        # SQLite auth.db에서 사용자 조회
        auth_db_path = Path("auth.db")
        with sqlite3.connect(auth_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, username, email, role, is_active FROM users WHERE id = ?",
                (token_payload.user_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Create SimpleUser object
            user = SimpleUser(
                user_id=row['id'],
                username=row['username'],
                email=row['email'],
                role=row['role'],
                is_active=bool(row['is_active'])
            )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Inactive user",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return user
        
    except (TokenExpiredError, InvalidTokenError, TokenBlacklistedError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in get_current_user: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(current_user: SimpleUser = Depends(get_current_user)) -> SimpleUser:
    """
    Get current active user.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        SimpleUser: Current active user
        
    Raises:
        HTTPException: 400 if user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_admin_user(current_user: SimpleUser = Depends(get_current_user)) -> SimpleUser:
    """
    Get current user with admin role.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        SimpleUser: Current admin user
        
    Raises:
        HTTPException: 403 if user doesn't have admin role
    """
    if not current_user.has_role("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user