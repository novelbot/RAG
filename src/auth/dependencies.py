"""
Authentication dependencies for FastAPI routes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Dict, Any
import asyncio

# from .models import User  # Commented out to avoid import issues
from .schemas import UserResponse

security = HTTPBearer()


class MockUser:
    """Mock user class for development"""
    def __init__(self, id: str, username: str, email: str, roles: list):
        self.id = id
        self.username = username
        self.email = email
        self.roles = roles
        self.is_active = True


async def get_current_user(token: str = Depends(security)) -> MockUser:
    """
    Get current authenticated user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: 401 if token is invalid
    """
    # Simulate async token validation
    await asyncio.sleep(0.05)
    
    # TODO: Implement actual JWT validation and user lookup
    # For now, return a mock user for development
    if token.credentials == "demo_access_token":
        return MockUser(
            id="demo_user_id",
            username="demo",
            email="demo@example.com",
            roles=["user"]
        )
    elif token.credentials == "admin_token":
        return MockUser(
            id="admin_user_id", 
            username="admin",
            email="admin@example.com",
            roles=["admin", "user"]
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(current_user: MockUser = Depends(get_current_user)) -> MockUser:
    """
    Get current active user.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: 400 if user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_admin_user(current_user: MockUser = Depends(get_current_user)) -> MockUser:
    """
    Get current user with admin role.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User: Current admin user
        
    Raises:
        HTTPException: 403 if user doesn't have admin role
    """
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user