"""
Authentication and Authorization Models.

This module defines SQLAlchemy models for user management, roles, and permissions
with support for many-to-many relationships and password hashing.
"""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Boolean, DateTime, Text, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext

from src.core.database import Base
from src.core.mixins import TimestampMixin

# Forward reference for Group model
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.access_control.group_manager import Group


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base, TimestampMixin):
    """
    User model for authentication and authorization.
    
    Supports username/email login, password hashing, and role-based access control.
    """
    
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # User profile information
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Profile metadata
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failed_login_attempts: Mapped[int] = mapped_column(default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Additional user information
    avatar_url: Mapped[Optional[str]] = mapped_column(String(255))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    timezone: Mapped[Optional[str]] = mapped_column(String(50), default="UTC")
    
    # Relationships
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Convenience relationship to access roles directly
    roles: Mapped[List["Role"]] = relationship(
        "Role",
        secondary="user_roles",
        back_populates="users",
        viewonly=True
    )
    
    # Convenience relationship to access groups directly
    groups: Mapped[List["Group"]] = relationship(
        "Group",
        secondary="group_users",
        back_populates="users",
        viewonly=True
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_user_username_email", "username", "email"),
        Index("idx_user_active", "is_active"),
        Index("idx_user_last_login", "last_login"),
    )
    
    def set_password(self, password: str) -> None:
        """Hash and set user password."""
        self.hashed_password = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        return pwd_context.verify(password, self.hashed_password)
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock user account for specified duration."""
        self.locked_until = datetime.now(timezone.utc) + datetime.timedelta(minutes=duration_minutes)
    
    def unlock_account(self) -> None:
        """Unlock user account and reset failed login attempts."""
        self.locked_until = None
        self.failed_login_attempts = 0
    
    def increment_failed_login(self) -> None:
        """Increment failed login attempts."""
        self.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            self.lock_account()
    
    def reset_failed_login(self) -> None:
        """Reset failed login attempts on successful login."""
        self.failed_login_attempts = 0
    
    def update_last_login(self) -> None:
        """Update last login timestamp."""
        self.last_login = datetime.now(timezone.utc)
    
    def get_permissions(self) -> List[str]:
        """Get all permissions for this user through their roles."""
        permissions = set()
        for user_role in self.user_roles:
            for role_permission in user_role.role.role_permissions:
                permissions.add(role_permission.permission.name)
        return list(permissions)
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if user has a specific permission."""
        return permission_name in self.get_permissions()
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return any(ur.role.name == role_name for ur in self.user_roles)
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class Role(Base, TimestampMixin):
    """
    Role model for role-based access control.
    
    Supports hierarchical roles and permission assignments.
    """
    
    __tablename__ = "roles"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Hierarchical role support
    parent_role_id: Mapped[Optional[int]] = mapped_column(ForeignKey("roles.id"))
    parent_role: Mapped[Optional["Role"]] = relationship(
        "Role",
        remote_side=[id],
        back_populates="child_roles"
    )
    child_roles: Mapped[List["Role"]] = relationship(
        "Role",
        back_populates="parent_role"
    )
    
    # Role metadata
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    priority: Mapped[int] = mapped_column(default=0, nullable=False)  # Higher priority = more important
    
    # Relationships
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="role",
        cascade="all, delete-orphan"
    )
    
    role_permissions: Mapped[List["RolePermission"]] = relationship(
        "RolePermission",
        back_populates="role",
        cascade="all, delete-orphan"
    )
    
    # Convenience relationship to access users directly
    users: Mapped[List["User"]] = relationship(
        "User",
        secondary="user_roles",
        back_populates="roles",
        viewonly=True
    )
    
    # Convenience relationship to access permissions directly
    permissions: Mapped[List["Permission"]] = relationship(
        "Permission",
        secondary="role_permissions",
        back_populates="roles",
        viewonly=True
    )
    
    # Convenience relationship to access groups directly
    groups: Mapped[List["Group"]] = relationship(
        "Group",
        secondary="group_roles",
        back_populates="roles",
        viewonly=True
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_role_name", "name"),
        Index("idx_role_default", "is_default"),
        Index("idx_role_priority", "priority"),
    )
    
    def get_all_permissions(self) -> List[str]:
        """Get all permissions for this role, including inherited permissions."""
        permissions = set()
        
        # Add direct permissions
        for rp in self.role_permissions:
            permissions.add(rp.permission.name)
        
        # Add inherited permissions from parent roles
        if self.parent_role:
            permissions.update(self.parent_role.get_all_permissions())
        
        return list(permissions)
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if role has a specific permission."""
        return permission_name in self.get_all_permissions()
    
    def __repr__(self) -> str:
        return f"<Role(id={self.id}, name='{self.name}')>"


class Permission(Base, TimestampMixin):
    """
    Permission model for fine-grained access control.
    
    Supports hierarchical permissions and resource-based access control.
    """
    
    __tablename__ = "permissions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Permission categorization
    resource: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # e.g., "users", "documents", "models"
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)    # e.g., "read", "write", "delete"
    
    # Permission metadata
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Relationships
    role_permissions: Mapped[List["RolePermission"]] = relationship(
        "RolePermission",
        back_populates="permission",
        cascade="all, delete-orphan"
    )
    
    # Convenience relationship to access roles directly
    roles: Mapped[List["Role"]] = relationship(
        "Role",
        secondary="role_permissions",
        back_populates="permissions",
        viewonly=True
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_permission_name", "name"),
        Index("idx_permission_resource_action", "resource", "action"),
        UniqueConstraint("resource", "action", name="uq_permission_resource_action"),
    )
    
    def __repr__(self) -> str:
        return f"<Permission(id={self.id}, name='{self.name}', resource='{self.resource}', action='{self.action}')>"


class UserRole(Base, TimestampMixin):
    """
    Many-to-many association table between Users and Roles.
    
    Includes additional metadata for role assignments.
    """
    
    __tablename__ = "user_roles"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
    
    # Assignment metadata
    assigned_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="user_roles", foreign_keys=[user_id])
    role: Mapped["Role"] = relationship("Role", back_populates="user_roles")
    assigned_by_user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[assigned_by])
    
    # Unique constraint to prevent duplicate assignments
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_user_role"),
        Index("idx_user_role_user", "user_id"),
        Index("idx_user_role_role", "role_id"),
        Index("idx_user_role_expires", "expires_at"),
    )
    
    def is_expired(self) -> bool:
        """Check if role assignment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def __repr__(self) -> str:
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"


class RolePermission(Base, TimestampMixin):
    """
    Many-to-many association table between Roles and Permissions.
    
    Includes additional metadata for permission assignments.
    """
    
    __tablename__ = "role_permissions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
    permission_id: Mapped[int] = mapped_column(ForeignKey("permissions.id"), nullable=False)
    
    # Assignment metadata
    assigned_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    role: Mapped["Role"] = relationship("Role", back_populates="role_permissions")
    permission: Mapped["Permission"] = relationship("Permission", back_populates="role_permissions")
    assigned_by_user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[assigned_by])
    
    # Unique constraint to prevent duplicate assignments
    __table_args__ = (
        UniqueConstraint("role_id", "permission_id", name="uq_role_permission"),
        Index("idx_role_permission_role", "role_id"),
        Index("idx_role_permission_permission", "permission_id"),
    )
    
    def __repr__(self) -> str:
        return f"<RolePermission(role_id={self.role_id}, permission_id={self.permission_id})>"