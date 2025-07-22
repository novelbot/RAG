"""
Group-Based Access Control Manager.

This module provides group management system with role-based permissions
and hierarchical group membership handling.
"""

from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import threading
import json

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

from src.core.database import Base
from src.core.mixins import TimestampMixin
from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from src.auth.models import User, Role
from src.auth.rbac import RBACManager as AuthRBACManager
from .exceptions import GroupNotFoundError, AccessControlError


class GroupType(Enum):
    """Group types for different organizational structures."""
    DEPARTMENT = "department"
    TEAM = "team"
    PROJECT = "project"
    SECURITY = "security"
    TEMPORARY = "temporary"


class MembershipStatus(Enum):
    """Membership status in a group."""
    ACTIVE = "active"
    PENDING = "pending"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class GroupRole(Enum):
    """Roles within a group."""
    MEMBER = "member"
    MODERATOR = "moderator"
    ADMIN = "admin"
    OWNER = "owner"


# Association table for group-user relationships
group_users = Table(
    'group_users',
    Base.metadata,
    Column('id', Integer, primary_key=True),
    Column('group_id', Integer, ForeignKey('groups.id'), nullable=False),
    Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
    Column('group_role', String(50), default=GroupRole.MEMBER.value, nullable=False),
    Column('status', String(50), default=MembershipStatus.ACTIVE.value, nullable=False),
    Column('joined_at', DateTime(timezone=True), default=func.now(), nullable=False),
    Column('expires_at', DateTime(timezone=True)),
    Column('added_by', Integer, ForeignKey('users.id')),
    Column('created_at', DateTime(timezone=True), default=func.now(), nullable=False),
    Column('updated_at', DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
)


# Association table for group-role relationships
group_roles = Table(
    'group_roles',
    Base.metadata,
    Column('id', Integer, primary_key=True),
    Column('group_id', Integer, ForeignKey('groups.id'), nullable=False),
    Column('role_id', Integer, ForeignKey('roles.id'), nullable=False),
    Column('assigned_by', Integer, ForeignKey('users.id')),
    Column('assigned_at', DateTime(timezone=True), default=func.now(), nullable=False),
    Column('expires_at', DateTime(timezone=True)),
    Column('created_at', DateTime(timezone=True), default=func.now(), nullable=False),
    Column('updated_at', DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
)


class Group(Base, TimestampMixin):
    """
    Group model for organizational access control.
    
    Supports hierarchical groups, role-based permissions, and membership management.
    """
    
    __tablename__ = "groups"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    group_type = Column(String(50), default=GroupType.TEAM.value, nullable=False)
    
    # Hierarchical group support
    parent_group_id = Column(Integer, ForeignKey('groups.id'))
    parent_group = relationship("Group", remote_side=[id], back_populates="child_groups")
    child_groups = relationship("Group", back_populates="parent_group")
    
    # Group settings
    is_active = Column(Boolean, default=True, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    auto_join = Column(Boolean, default=False, nullable=False)
    max_members = Column(Integer)
    
    # Metadata
    metadata = Column(Text)  # JSON field for additional metadata
    
    # Group ownership
    owner_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    owner = relationship("User", foreign_keys=[owner_id])
    
    # Relationships
    users = relationship("User", secondary=group_users, back_populates="groups")
    roles = relationship("Role", secondary=group_roles, back_populates="groups")
    
    def __repr__(self) -> str:
        return f"<Group(id={self.id}, name='{self.name}', type='{self.group_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "group_type": self.group_type,
            "parent_group_id": self.parent_group_id,
            "is_active": self.is_active,
            "is_public": self.is_public,
            "auto_join": self.auto_join,
            "max_members": self.max_members,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": json.loads(self.metadata) if self.metadata else {}
        }


@dataclass
class GroupMembership:
    """Group membership information."""
    group_id: int
    user_id: int
    group_role: GroupRole
    status: MembershipStatus
    joined_at: datetime
    expires_at: Optional[datetime] = None
    added_by: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if membership has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if membership is active."""
        return self.status == MembershipStatus.ACTIVE and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "group_id": self.group_id,
            "user_id": self.user_id,
            "group_role": self.group_role.value,
            "status": self.status.value,
            "joined_at": self.joined_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "added_by": self.added_by,
            "is_expired": self.is_expired(),
            "is_active": self.is_active()
        }


@dataclass
class GroupPermissions:
    """Group permissions aggregation."""
    group_id: int
    permissions: Set[str]
    roles: Set[str]
    inherited_permissions: Set[str]
    effective_permissions: Set[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "group_id": self.group_id,
            "permissions": list(self.permissions),
            "roles": list(self.roles),
            "inherited_permissions": list(self.inherited_permissions),
            "effective_permissions": list(self.effective_permissions)
        }


class GroupManager(LoggerMixin):
    """
    Group-Based Access Control Manager.
    
    Manages groups, memberships, and hierarchical permission inheritance.
    """
    
    def __init__(self, db_manager: DatabaseManager, auth_rbac_manager: AuthRBACManager):
        """
        Initialize group manager.
        
        Args:
            db_manager: Database manager
            auth_rbac_manager: Authentication RBAC manager
        """
        self.db_manager = db_manager
        self.auth_rbac_manager = auth_rbac_manager
        
        # Cache for group permissions
        self.group_permissions_cache: Dict[int, GroupPermissions] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[int, datetime] = {}
        self._lock = threading.Lock()
        
        self.logger.info("Group Manager initialized successfully")
    
    def _get_session(self) -> Session:
        """Get database session."""
        return self.db_manager.get_session()
    
    def _is_cache_valid(self, group_id: int) -> bool:
        """Check if cached group permissions are still valid."""
        if group_id not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[group_id]
        elapsed = (datetime.utcnow() - cache_time).total_seconds()
        return elapsed < self.cache_ttl
    
    def _clear_cache(self, group_id: Optional[int] = None) -> None:
        """Clear group permissions cache."""
        with self._lock:
            if group_id:
                self.group_permissions_cache.pop(group_id, None)
                self.cache_timestamps.pop(group_id, None)
            else:
                self.group_permissions_cache.clear()
                self.cache_timestamps.clear()
    
    def create_group(self, 
                    name: str,
                    description: Optional[str] = None,
                    group_type: GroupType = GroupType.TEAM,
                    owner_id: int = None,
                    parent_group_id: Optional[int] = None,
                    is_public: bool = False,
                    auto_join: bool = False,
                    max_members: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Group:
        """
        Create a new group.
        
        Args:
            name: Group name
            description: Group description
            group_type: Type of group
            owner_id: Owner user ID
            parent_group_id: Parent group ID
            is_public: Whether group is public
            auto_join: Whether users can auto-join
            max_members: Maximum number of members
            metadata: Additional metadata
            
        Returns:
            Created group
            
        Raises:
            AccessControlError: If group creation fails
        """
        try:
            with self._get_session() as session:
                # Check if group name already exists
                existing_group = session.query(Group).filter(Group.name == name).first()
                if existing_group:
                    raise AccessControlError(f"Group with name '{name}' already exists")
                
                # Validate parent group if specified
                if parent_group_id:
                    parent_group = session.query(Group).filter(Group.id == parent_group_id).first()
                    if not parent_group:
                        raise GroupNotFoundError(f"Parent group with ID {parent_group_id} not found")
                
                # Create group
                group = Group(
                    name=name,
                    description=description,
                    group_type=group_type.value,
                    owner_id=owner_id,
                    parent_group_id=parent_group_id,
                    is_public=is_public,
                    auto_join=auto_join,
                    max_members=max_members,
                    metadata=json.dumps(metadata) if metadata else None
                )
                
                session.add(group)
                session.commit()
                
                # Automatically add owner as admin member
                if owner_id:
                    self.add_user_to_group(
                        group_id=group.id,
                        user_id=owner_id,
                        group_role=GroupRole.OWNER,
                        added_by=owner_id
                    )
                
                self.logger.info(f"Created group: {name} (ID: {group.id})")
                return group
                
        except Exception as e:
            self.logger.error(f"Failed to create group '{name}': {e}")
            raise AccessControlError(f"Group creation failed: {e}")
    
    def get_group(self, group_id: int) -> Optional[Group]:
        """
        Get group by ID.
        
        Args:
            group_id: Group ID
            
        Returns:
            Group or None if not found
        """
        try:
            with self._get_session() as session:
                return session.query(Group).filter(Group.id == group_id).first()
        except Exception as e:
            self.logger.error(f"Failed to get group {group_id}: {e}")
            return None
    
    def get_group_by_name(self, name: str) -> Optional[Group]:
        """
        Get group by name.
        
        Args:
            name: Group name
            
        Returns:
            Group or None if not found
        """
        try:
            with self._get_session() as session:
                return session.query(Group).filter(Group.name == name).first()
        except Exception as e:
            self.logger.error(f"Failed to get group '{name}': {e}")
            return None
    
    def add_user_to_group(self, 
                         group_id: int,
                         user_id: int,
                         group_role: GroupRole = GroupRole.MEMBER,
                         added_by: Optional[int] = None,
                         expires_at: Optional[datetime] = None) -> bool:
        """
        Add user to group.
        
        Args:
            group_id: Group ID
            user_id: User ID
            group_role: Role in group
            added_by: User who added the member
            expires_at: Membership expiration
            
        Returns:
            True if user was added successfully
            
        Raises:
            GroupNotFoundError: If group not found
            AccessControlError: If operation fails
        """
        try:
            with self._get_session() as session:
                # Verify group exists
                group = session.query(Group).filter(Group.id == group_id).first()
                if not group:
                    raise GroupNotFoundError(f"Group with ID {group_id} not found")
                
                # Verify user exists
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    raise AccessControlError(f"User with ID {user_id} not found")
                
                # Check if user is already a member
                existing_membership = session.execute(
                    group_users.select().where(
                        (group_users.c.group_id == group_id) &
                        (group_users.c.user_id == user_id)
                    )
                ).first()
                
                if existing_membership:
                    # Update existing membership
                    session.execute(
                        group_users.update().where(
                            (group_users.c.group_id == group_id) &
                            (group_users.c.user_id == user_id)
                        ).values(
                            group_role=group_role.value,
                            status=MembershipStatus.ACTIVE.value,
                            expires_at=expires_at,
                            updated_at=func.now()
                        )
                    )
                    self.logger.info(f"Updated membership for user {user_id} in group {group_id}")
                else:
                    # Check group capacity
                    if group.max_members:
                        current_members = session.execute(
                            group_users.select().where(
                                (group_users.c.group_id == group_id) &
                                (group_users.c.status == MembershipStatus.ACTIVE.value)
                            )
                        ).rowcount
                        
                        if current_members >= group.max_members:
                            raise AccessControlError(f"Group {group_id} has reached maximum capacity")
                    
                    # Add new membership
                    session.execute(
                        group_users.insert().values(
                            group_id=group_id,
                            user_id=user_id,
                            group_role=group_role.value,
                            status=MembershipStatus.ACTIVE.value,
                            added_by=added_by,
                            expires_at=expires_at
                        )
                    )
                    self.logger.info(f"Added user {user_id} to group {group_id} as {group_role.value}")
                
                session.commit()
                
                # Clear cache for this group
                self._clear_cache(group_id)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add user {user_id} to group {group_id}: {e}")
            raise AccessControlError(f"Failed to add user to group: {e}")
    
    def remove_user_from_group(self, group_id: int, user_id: int) -> bool:
        """
        Remove user from group.
        
        Args:
            group_id: Group ID
            user_id: User ID
            
        Returns:
            True if user was removed successfully
        """
        try:
            with self._get_session() as session:
                # Remove membership
                result = session.execute(
                    group_users.delete().where(
                        (group_users.c.group_id == group_id) &
                        (group_users.c.user_id == user_id)
                    )
                )
                
                if result.rowcount > 0:
                    session.commit()
                    
                    # Clear cache for this group
                    self._clear_cache(group_id)
                    
                    self.logger.info(f"Removed user {user_id} from group {group_id}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove user {user_id} from group {group_id}: {e}")
            return False
    
    def get_user_groups(self, user_id: int, include_inactive: bool = False) -> List[GroupMembership]:
        """
        Get all groups for a user.
        
        Args:
            user_id: User ID
            include_inactive: Include inactive memberships
            
        Returns:
            List of group memberships
        """
        try:
            with self._get_session() as session:
                query = session.execute(
                    group_users.select().where(group_users.c.user_id == user_id)
                )
                
                memberships = []
                for row in query:
                    membership = GroupMembership(
                        group_id=row.group_id,
                        user_id=row.user_id,
                        group_role=GroupRole(row.group_role),
                        status=MembershipStatus(row.status),
                        joined_at=row.joined_at,
                        expires_at=row.expires_at,
                        added_by=row.added_by
                    )
                    
                    if include_inactive or membership.is_active():
                        memberships.append(membership)
                
                return memberships
                
        except Exception as e:
            self.logger.error(f"Failed to get groups for user {user_id}: {e}")
            return []
    
    def get_group_members(self, group_id: int, include_inactive: bool = False) -> List[GroupMembership]:
        """
        Get all members of a group.
        
        Args:
            group_id: Group ID
            include_inactive: Include inactive memberships
            
        Returns:
            List of group memberships
        """
        try:
            with self._get_session() as session:
                query = session.execute(
                    group_users.select().where(group_users.c.group_id == group_id)
                )
                
                memberships = []
                for row in query:
                    membership = GroupMembership(
                        group_id=row.group_id,
                        user_id=row.user_id,
                        group_role=GroupRole(row.group_role),
                        status=MembershipStatus(row.status),
                        joined_at=row.joined_at,
                        expires_at=row.expires_at,
                        added_by=row.added_by
                    )
                    
                    if include_inactive or membership.is_active():
                        memberships.append(membership)
                
                return memberships
                
        except Exception as e:
            self.logger.error(f"Failed to get members for group {group_id}: {e}")
            return []
    
    def get_group_permissions(self, group_id: int, use_cache: bool = True) -> GroupPermissions:
        """
        Get all permissions for a group.
        
        Args:
            group_id: Group ID
            use_cache: Whether to use cached permissions
            
        Returns:
            Group permissions
        """
        try:
            # Check cache first
            if use_cache and self._is_cache_valid(group_id):
                return self.group_permissions_cache[group_id]
            
            with self._get_session() as session:
                group = session.query(Group).filter(Group.id == group_id).first()
                if not group:
                    raise GroupNotFoundError(f"Group with ID {group_id} not found")
                
                permissions = set()
                roles = set()
                inherited_permissions = set()
                
                # Get direct role permissions
                group_role_query = session.execute(
                    group_roles.select().where(group_roles.c.group_id == group_id)
                )
                
                for row in group_role_query:
                    role = session.query(Role).filter(Role.id == row.role_id).first()
                    if role:
                        roles.add(role.name)
                        role_permissions = self.auth_rbac_manager.get_user_permissions(row.role_id)
                        permissions.update(role_permissions)
                
                # Get inherited permissions from parent groups
                if group.parent_group_id:
                    parent_permissions = self.get_group_permissions(group.parent_group_id, use_cache)
                    inherited_permissions = parent_permissions.effective_permissions
                
                # Combine all permissions
                effective_permissions = permissions | inherited_permissions
                
                # Create group permissions object
                group_permissions = GroupPermissions(
                    group_id=group_id,
                    permissions=permissions,
                    roles=roles,
                    inherited_permissions=inherited_permissions,
                    effective_permissions=effective_permissions
                )
                
                # Cache the result
                if use_cache:
                    with self._lock:
                        self.group_permissions_cache[group_id] = group_permissions
                        self.cache_timestamps[group_id] = datetime.utcnow()
                
                return group_permissions
                
        except Exception as e:
            self.logger.error(f"Failed to get permissions for group {group_id}: {e}")
            raise AccessControlError(f"Failed to get group permissions: {e}")
    
    def assign_role_to_group(self, 
                           group_id: int,
                           role_id: int,
                           assigned_by: Optional[int] = None,
                           expires_at: Optional[datetime] = None) -> bool:
        """
        Assign role to group.
        
        Args:
            group_id: Group ID
            role_id: Role ID
            assigned_by: User who assigned the role
            expires_at: Role assignment expiration
            
        Returns:
            True if role was assigned successfully
        """
        try:
            with self._get_session() as session:
                # Verify group and role exist
                group = session.query(Group).filter(Group.id == group_id).first()
                if not group:
                    raise GroupNotFoundError(f"Group with ID {group_id} not found")
                
                role = session.query(Role).filter(Role.id == role_id).first()
                if not role:
                    raise AccessControlError(f"Role with ID {role_id} not found")
                
                # Check if role is already assigned
                existing_assignment = session.execute(
                    group_roles.select().where(
                        (group_roles.c.group_id == group_id) &
                        (group_roles.c.role_id == role_id)
                    )
                ).first()
                
                if existing_assignment:
                    # Update existing assignment
                    session.execute(
                        group_roles.update().where(
                            (group_roles.c.group_id == group_id) &
                            (group_roles.c.role_id == role_id)
                        ).values(
                            expires_at=expires_at,
                            updated_at=func.now()
                        )
                    )
                    self.logger.info(f"Updated role assignment for group {group_id}")
                else:
                    # Add new assignment
                    session.execute(
                        group_roles.insert().values(
                            group_id=group_id,
                            role_id=role_id,
                            assigned_by=assigned_by,
                            expires_at=expires_at
                        )
                    )
                    self.logger.info(f"Assigned role {role_id} to group {group_id}")
                
                session.commit()
                
                # Clear cache for this group
                self._clear_cache(group_id)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to assign role {role_id} to group {group_id}: {e}")
            raise AccessControlError(f"Failed to assign role to group: {e}")
    
    def revoke_role_from_group(self, group_id: int, role_id: int) -> bool:
        """
        Revoke role from group.
        
        Args:
            group_id: Group ID
            role_id: Role ID
            
        Returns:
            True if role was revoked successfully
        """
        try:
            with self._get_session() as session:
                result = session.execute(
                    group_roles.delete().where(
                        (group_roles.c.group_id == group_id) &
                        (group_roles.c.role_id == role_id)
                    )
                )
                
                if result.rowcount > 0:
                    session.commit()
                    
                    # Clear cache for this group
                    self._clear_cache(group_id)
                    
                    self.logger.info(f"Revoked role {role_id} from group {group_id}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to revoke role {role_id} from group {group_id}: {e}")
            return False
    
    def get_user_effective_permissions(self, user_id: int) -> Set[str]:
        """
        Get effective permissions for a user including group permissions.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of effective permissions
        """
        try:
            # Get user's direct permissions
            user_permissions = self.auth_rbac_manager.get_user_permissions(user_id)
            
            # Get permissions from all user's groups
            user_groups = self.get_user_groups(user_id)
            
            for membership in user_groups:
                if membership.is_active():
                    group_permissions = self.get_group_permissions(membership.group_id)
                    user_permissions.update(group_permissions.effective_permissions)
            
            return user_permissions
            
        except Exception as e:
            self.logger.error(f"Failed to get effective permissions for user {user_id}: {e}")
            return set()
    
    def get_group_statistics(self) -> Dict[str, Any]:
        """Get group system statistics."""
        try:
            with self._get_session() as session:
                total_groups = session.query(Group).count()
                active_groups = session.query(Group).filter(Group.is_active == True).count()
                
                # Count by group type
                group_types = {}
                for group_type in GroupType:
                    count = session.query(Group).filter(Group.group_type == group_type.value).count()
                    group_types[group_type.value] = count
                
                # Count memberships
                total_memberships = session.execute(group_users.select()).rowcount
                active_memberships = session.execute(
                    group_users.select().where(group_users.c.status == MembershipStatus.ACTIVE.value)
                ).rowcount
                
                return {
                    "total_groups": total_groups,
                    "active_groups": active_groups,
                    "group_types": group_types,
                    "total_memberships": total_memberships,
                    "active_memberships": active_memberships,
                    "cached_permissions": len(self.group_permissions_cache),
                    "cache_ttl_seconds": self.cache_ttl
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get group statistics: {e}")
            return {"error": str(e)}