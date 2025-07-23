"""
User Manager for handling user CRUD operations
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import or_

from .models import User, Role, UserRole
from ..core.logging import get_logger

logger = get_logger(__name__)


class UserManager:
    """Manages user operations including creation, authentication, and role management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        role: Optional[str] = None,
        department: Optional[str] = None,
        full_name: Optional[str] = None,
        is_active: bool = True,
        is_verified: bool = False
    ) -> User:
        """Create a new user with optional role assignment."""
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            is_active=is_active,
            is_verified=is_verified
        )
        user.set_password(password)
        
        self.db.add(user)
        self.db.flush()  # Get the user ID
        
        # Assign role if provided
        if role:
            self.assign_role(user.id, role)
        
        self.db.commit()
        return user
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_username_or_email(self, identifier: str) -> Optional[User]:
        """Get user by username or email."""
        return self.db.query(User).filter(
            or_(User.username == identifier, User.email == identifier)
        ).first()
    
    def get_all_users(self, limit: int = 50, offset: int = 0) -> List[User]:
        """Get all users with pagination."""
        return self.db.query(User).offset(offset).limit(limit).all()
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user by ID."""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Handle password separately
        if 'password' in kwargs:
            password = kwargs.pop('password')
            user.set_password(password)
        
        # Update other fields
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        self.db.commit()
        return user
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user by ID."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        self.db.delete(user)
        self.db.commit()
        return True
    
    def authenticate_user(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        user = self.get_user_by_username_or_email(username_or_email)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if user.is_locked():
            return None
        
        if user.verify_password(password):
            user.reset_failed_login()
            user.update_last_login()
            self.db.commit()
            return user
        else:
            user.increment_failed_login()
            self.db.commit()
            return None
    
    def assign_role(self, user_id: int, role_name: str) -> bool:
        """Assign a role to a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        role = self.db.query(Role).filter(Role.name == role_name).first()
        if not role:
            return False
        
        # Check if user already has this role
        existing = self.db.query(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.role_id == role.id
        ).first()
        
        if existing:
            return True  # Already has role
        
        user_role = UserRole(user_id=user_id, role_id=role.id)
        self.db.add(user_role)
        self.db.commit()
        return True
    
    def remove_role(self, user_id: int, role_name: str) -> bool:
        """Remove a role from a user."""
        role = self.db.query(Role).filter(Role.name == role_name).first()
        if not role:
            return False
        
        user_role = self.db.query(UserRole).filter(
            UserRole.user_id == user_id,
            UserRole.role_id == role.id
        ).first()
        
        if user_role:
            self.db.delete(user_role)
            self.db.commit()
        
        return True
    
    def get_user_roles(self, user_id: int) -> List[str]:
        """Get all roles for a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            return []
        
        return [ur.role.name for ur in user.user_roles]
    
    def has_role(self, user_id: int, role_name: str) -> bool:
        """Check if user has a specific role."""
        return role_name in self.get_user_roles(user_id)
    
    def activate_user(self, user_id: int) -> bool:
        """Activate a user account."""
        return self.update_user(user_id, is_active=True) is not None
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account."""
        return self.update_user(user_id, is_active=False) is not None
    
    def verify_user(self, user_id: int) -> bool:
        """Mark a user as verified."""
        return self.update_user(user_id, is_verified=True) is not None
    
    def lock_user(self, user_id: int, duration_minutes: int = 30) -> bool:
        """Lock a user account for a specific duration."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.lock_account(duration_minutes)
        self.db.commit()
        return True
    
    def unlock_user(self, user_id: int) -> bool:
        """Unlock a user account."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.unlock_account()
        self.db.commit()
        return True