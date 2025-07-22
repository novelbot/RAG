"""
Role-Based Access Control (RBAC) Manager.

This module provides comprehensive RBAC functionality including permission checking,
role management, and access control enforcement.
"""

from typing import List, Optional, Dict, Any, Set
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from functools import wraps
from datetime import datetime, timezone

from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from .models import User, Role, Permission, UserRole, RolePermission
from .exceptions import (
    InsufficientPermissionsError, AuthorizationError, 
    UserNotFoundError, RoleNotFoundError, PermissionNotFoundError
)


class RBACManager(LoggerMixin):
    """
    Role-Based Access Control Manager.
    
    Provides comprehensive RBAC functionality including:
    - Permission checking and enforcement
    - Role management and assignment
    - Hierarchical role support
    - Permission caching for performance
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize RBAC Manager."""
        self.db_manager = db_manager
        
        # Permission cache for performance
        self.permission_cache: Dict[int, Set[str]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[int, datetime] = {}
        
        self.logger.info("RBAC Manager initialized successfully")
    
    def _get_session(self) -> Session:
        """Get database session."""
        return self.db_manager.get_session()
    
    def _is_cache_valid(self, user_id: int) -> bool:
        """Check if permission cache is valid for user."""
        if user_id not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[user_id]
        return (datetime.now(timezone.utc) - cache_time).total_seconds() < self.cache_ttl
    
    def _update_cache(self, user_id: int, permissions: Set[str]) -> None:
        """Update permission cache for user."""
        self.permission_cache[user_id] = permissions
        self.cache_timestamps[user_id] = datetime.now(timezone.utc)
    
    def _clear_cache(self, user_id: Optional[int] = None) -> None:
        """Clear permission cache."""
        if user_id:
            self.permission_cache.pop(user_id, None)
            self.cache_timestamps.pop(user_id, None)
        else:
            self.permission_cache.clear()
            self.cache_timestamps.clear()
    
    def get_user_permissions(self, user_id: int, use_cache: bool = True) -> Set[str]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User ID
            use_cache: Whether to use cached permissions
            
        Returns:
            Set of permission names
        """
        # Check cache first
        if use_cache and self._is_cache_valid(user_id):
            return self.permission_cache.get(user_id, set())
        
        with self._get_session() as session:
            # Get user with roles and permissions
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise UserNotFoundError(f"User with ID {user_id} not found")
            
            permissions = set()
            
            # Get permissions from all user roles
            for user_role in user.user_roles:
                # Skip expired roles
                if user_role.is_expired():
                    continue
                
                # Get all permissions for this role (including inherited)
                role_permissions = self._get_role_permissions(session, user_role.role)
                permissions.update(role_permissions)
            
            # Update cache
            if use_cache:
                self._update_cache(user_id, permissions)
            
            return permissions
    
    def _get_role_permissions(self, session: Session, role: Role) -> Set[str]:
        """Get all permissions for a role, including inherited permissions."""
        permissions = set()
        
        # Add direct permissions
        for role_permission in role.role_permissions:
            permissions.add(role_permission.permission.name)
        
        # Add inherited permissions from parent roles
        if role.parent_role:
            parent_permissions = self._get_role_permissions(session, role.parent_role)
            permissions.update(parent_permissions)
        
        return permissions
    
    def has_permission(self, user_id: int, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission name
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            user_permissions = self.get_user_permissions(user_id)
            return permission in user_permissions
        except UserNotFoundError:
            return False
    
    def has_any_permission(self, user_id: int, permissions: List[str]) -> bool:
        """
        Check if user has any of the specified permissions.
        
        Args:
            user_id: User ID
            permissions: List of permission names
            
        Returns:
            True if user has at least one permission, False otherwise
        """
        try:
            user_permissions = self.get_user_permissions(user_id)
            return any(perm in user_permissions for perm in permissions)
        except UserNotFoundError:
            return False
    
    def has_all_permissions(self, user_id: int, permissions: List[str]) -> bool:
        """
        Check if user has all specified permissions.
        
        Args:
            user_id: User ID
            permissions: List of permission names
            
        Returns:
            True if user has all permissions, False otherwise
        """
        try:
            user_permissions = self.get_user_permissions(user_id)
            return all(perm in user_permissions for perm in permissions)
        except UserNotFoundError:
            return False
    
    def has_role(self, user_id: int, role_name: str) -> bool:
        """
        Check if user has a specific role.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            True if user has role, False otherwise
        """
        with self._get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            return any(
                ur.role.name == role_name and not ur.is_expired()
                for ur in user.user_roles
            )
    
    def get_user_roles(self, user_id: int) -> List[str]:
        """
        Get all roles for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of role names
        """
        with self._get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise UserNotFoundError(f"User with ID {user_id} not found")
            
            return [
                ur.role.name
                for ur in user.user_roles
                if not ur.is_expired()
            ]
    
    def assign_role(self, user_id: int, role_name: str, assigned_by: Optional[int] = None, 
                   expires_at: Optional[datetime] = None) -> bool:
        """
        Assign a role to a user.
        
        Args:
            user_id: User ID
            role_name: Role name
            assigned_by: User ID who assigned the role
            expires_at: Optional expiration time
            
        Returns:
            True if role was assigned, False if already assigned
        """
        with self._get_session() as session:
            # Get user and role
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise UserNotFoundError(f"User with ID {user_id} not found")
            
            role = session.query(Role).filter(Role.name == role_name).first()
            if not role:
                raise RoleNotFoundError(f"Role '{role_name}' not found")
            
            # Check if user already has this role
            existing_assignment = session.query(UserRole).filter(
                and_(UserRole.user_id == user_id, UserRole.role_id == role.id)
            ).first()
            
            if existing_assignment:
                # Update expiration if provided
                if expires_at:
                    existing_assignment.expires_at = expires_at
                    session.commit()
                    self.logger.info(f"Updated role assignment expiration for user {user_id}, role {role_name}")
                return False
            
            # Create new role assignment
            user_role = UserRole(
                user_id=user_id,
                role_id=role.id,
                assigned_by=assigned_by,
                expires_at=expires_at
            )
            
            session.add(user_role)
            session.commit()
            
            # Clear cache for this user
            self._clear_cache(user_id)
            
            self.logger.info(f"Assigned role '{role_name}' to user {user_id}")
            return True
    
    def revoke_role(self, user_id: int, role_name: str) -> bool:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            True if role was revoked, False if not assigned
        """
        with self._get_session() as session:
            # Get role
            role = session.query(Role).filter(Role.name == role_name).first()
            if not role:
                raise RoleNotFoundError(f"Role '{role_name}' not found")
            
            # Find and remove role assignment
            user_role = session.query(UserRole).filter(
                and_(UserRole.user_id == user_id, UserRole.role_id == role.id)
            ).first()
            
            if not user_role:
                return False
            
            session.delete(user_role)
            session.commit()
            
            # Clear cache for this user
            self._clear_cache(user_id)
            
            self.logger.info(f"Revoked role '{role_name}' from user {user_id}")
            return True
    
    def create_role(self, name: str, description: Optional[str] = None, 
                   parent_role_name: Optional[str] = None, 
                   permissions: Optional[List[str]] = None) -> Role:
        """
        Create a new role.
        
        Args:
            name: Role name
            description: Role description
            parent_role_name: Parent role name for hierarchy
            permissions: List of permission names to assign
            
        Returns:
            Created role
        """
        with self._get_session() as session:
            # Check if role already exists
            existing_role = session.query(Role).filter(Role.name == name).first()
            if existing_role:
                raise ValueError(f"Role '{name}' already exists")
            
            # Get parent role if specified
            parent_role = None
            if parent_role_name:
                parent_role = session.query(Role).filter(Role.name == parent_role_name).first()
                if not parent_role:
                    raise RoleNotFoundError(f"Parent role '{parent_role_name}' not found")
            
            # Create role
            role = Role(
                name=name,
                description=description,
                parent_role_id=parent_role.id if parent_role else None
            )
            
            session.add(role)
            session.flush()  # Get the role ID
            
            # Assign permissions if provided
            if permissions:
                for perm_name in permissions:
                    permission = session.query(Permission).filter(Permission.name == perm_name).first()
                    if permission:
                        role_permission = RolePermission(
                            role_id=role.id,
                            permission_id=permission.id
                        )
                        session.add(role_permission)
            
            session.commit()
            
            # Clear all caches since role hierarchy might have changed
            self._clear_cache()
            
            self.logger.info(f"Created role '{name}' with ID {role.id}")
            return role
    
    def create_permission(self, name: str, resource: str, action: str, 
                         description: Optional[str] = None) -> Permission:
        """
        Create a new permission.
        
        Args:
            name: Permission name
            resource: Resource type
            action: Action type
            description: Permission description
            
        Returns:
            Created permission
        """
        with self._get_session() as session:
            # Check if permission already exists
            existing_permission = session.query(Permission).filter(
                or_(
                    Permission.name == name,
                    and_(Permission.resource == resource, Permission.action == action)
                )
            ).first()
            
            if existing_permission:
                raise ValueError(f"Permission '{name}' or '{resource}:{action}' already exists")
            
            # Create permission
            permission = Permission(
                name=name,
                resource=resource,
                action=action,
                description=description
            )
            
            session.add(permission)
            session.commit()
            
            self.logger.info(f"Created permission '{name}' with ID {permission.id}")
            return permission
    
    def assign_permission_to_role(self, role_name: str, permission_name: str, 
                                 assigned_by: Optional[int] = None) -> bool:
        """
        Assign a permission to a role.
        
        Args:
            role_name: Role name
            permission_name: Permission name
            assigned_by: User ID who assigned the permission
            
        Returns:
            True if permission was assigned, False if already assigned
        """
        with self._get_session() as session:
            # Get role and permission
            role = session.query(Role).filter(Role.name == role_name).first()
            if not role:
                raise RoleNotFoundError(f"Role '{role_name}' not found")
            
            permission = session.query(Permission).filter(Permission.name == permission_name).first()
            if not permission:
                raise PermissionNotFoundError(f"Permission '{permission_name}' not found")
            
            # Check if role already has this permission
            existing_assignment = session.query(RolePermission).filter(
                and_(RolePermission.role_id == role.id, RolePermission.permission_id == permission.id)
            ).first()
            
            if existing_assignment:
                return False
            
            # Create permission assignment
            role_permission = RolePermission(
                role_id=role.id,
                permission_id=permission.id,
                assigned_by=assigned_by
            )
            
            session.add(role_permission)
            session.commit()
            
            # Clear all caches since role permissions changed
            self._clear_cache()
            
            self.logger.info(f"Assigned permission '{permission_name}' to role '{role_name}'")
            return True
    
    def revoke_permission_from_role(self, role_name: str, permission_name: str) -> bool:
        """
        Revoke a permission from a role.
        
        Args:
            role_name: Role name
            permission_name: Permission name
            
        Returns:
            True if permission was revoked, False if not assigned
        """
        with self._get_session() as session:
            # Get role and permission
            role = session.query(Role).filter(Role.name == role_name).first()
            if not role:
                raise RoleNotFoundError(f"Role '{role_name}' not found")
            
            permission = session.query(Permission).filter(Permission.name == permission_name).first()
            if not permission:
                raise PermissionNotFoundError(f"Permission '{permission_name}' not found")
            
            # Find and remove permission assignment
            role_permission = session.query(RolePermission).filter(
                and_(RolePermission.role_id == role.id, RolePermission.permission_id == permission.id)
            ).first()
            
            if not role_permission:
                return False
            
            session.delete(role_permission)
            session.commit()
            
            # Clear all caches since role permissions changed
            self._clear_cache()
            
            self.logger.info(f"Revoked permission '{permission_name}' from role '{role_name}'")
            return True
    
    def check_access(self, user_id: int, resource: str, action: str) -> bool:
        """
        Check if user has access to perform an action on a resource.
        
        Args:
            user_id: User ID
            resource: Resource type
            action: Action type
            
        Returns:
            True if user has access, False otherwise
        """
        # Check for specific permission
        specific_permission = f"{resource}:{action}"
        if self.has_permission(user_id, specific_permission):
            return True
        
        # Check for wildcard permissions
        wildcard_resource = f"{resource}:*"
        wildcard_action = f"*:{action}"
        wildcard_all = "*:*"
        
        return self.has_any_permission(user_id, [wildcard_resource, wildcard_action, wildcard_all])
    
    def enforce_permission(self, user_id: int, permission: str) -> None:
        """
        Enforce that a user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission name
            
        Raises:
            InsufficientPermissionsError: If user lacks permission
        """
        if not self.has_permission(user_id, permission):
            raise InsufficientPermissionsError(f"User {user_id} lacks permission: {permission}")
    
    def enforce_access(self, user_id: int, resource: str, action: str) -> None:
        """
        Enforce that a user has access to perform an action on a resource.
        
        Args:
            user_id: User ID
            resource: Resource type
            action: Action type
            
        Raises:
            InsufficientPermissionsError: If user lacks access
        """
        if not self.check_access(user_id, resource, action):
            raise InsufficientPermissionsError(
                f"User {user_id} lacks access to perform '{action}' on '{resource}'"
            )


def require_permission(permission: str):
    """
    Decorator to require a specific permission.
    
    Args:
        permission: Permission name required
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependency injection
            # The actual implementation would get the current user from the request
            # For now, this is a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: str):
    """
    Decorator to require a specific role.
    
    Args:
        role: Role name required
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependency injection
            # The actual implementation would get the current user from the request
            # For now, this is a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_access(resource: str, action: str):
    """
    Decorator to require access to a resource.
    
    Args:
        resource: Resource type
        action: Action type
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependency injection
            # The actual implementation would get the current user from the request
            # For now, this is a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator