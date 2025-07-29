"""
Milvus Row-Level RBAC Manager.

This module provides integration between the authentication system and
Milvus row-level access control for fine-grained vector database security.
"""

from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.core.logging import LoggerMixin
from src.core.database import get_db
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection
from src.milvus.rbac import RBACManager, UserContext, AccessRule, Permission, AccessScope
from src.auth.models import User, Role, Permission as AuthPermission
from src.auth.rbac import RBACManager as AuthRBACManager
from .exceptions import MilvusRBACError, InsufficientPermissionsError

@dataclass
class MilvusUserContext:
    """Extended user context for Milvus operations."""
    user_id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool
    roles: List[str]
    permissions: List[str]
    group_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_rbac_context(self) -> UserContext:
        """Convert to Milvus RBAC UserContext."""
        # Map auth permissions to Milvus permissions
        milvus_permissions = []
        
        # Admin permissions
        if self.is_superuser or "admin" in self.roles:
            milvus_permissions.extend([
                Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN
            ])
        else:
            # Standard permissions mapping
            if any(perm in self.permissions for perm in ["read", "query", "search"]):
                milvus_permissions.append(Permission.READ)
            if any(perm in self.permissions for perm in ["write", "create", "update"]):
                milvus_permissions.append(Permission.WRITE)
            if any(perm in self.permissions for perm in ["delete", "remove"]):
                milvus_permissions.append(Permission.DELETE)
        
        return UserContext(
            user_id=str(self.user_id),
            group_ids=self.group_ids,
            permissions=milvus_permissions,
            is_admin=self.is_superuser,
            additional_metadata=self.metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "roles": self.roles,
            "permissions": self.permissions,
            "group_ids": self.group_ids,
            "metadata": self.metadata
        }


class MilvusRBACManager(LoggerMixin):
    """
    Milvus Row-Level RBAC Manager.
    
    Integrates authentication system with Milvus RBAC for fine-grained
    vector database access control.
    """
    
    def __init__(self, 
                 milvus_client: MilvusClient,
                 db_manager,
                 auth_rbac_manager: AuthRBACManager):
        """
        Initialize Milvus RBAC Manager.
        
        Args:
            milvus_client: Milvus client instance
            db_manager: Database manager
            auth_rbac_manager: Authentication RBAC manager
        """
        self.milvus_client = milvus_client
        self.db_manager = db_manager
        self.auth_rbac_manager = auth_rbac_manager
        
        # Initialize Milvus RBAC manager
        self.rbac_manager = RBACManager(client=milvus_client)
        
        # Cache for user contexts
        self._user_context_cache: Dict[int, MilvusUserContext] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[int, datetime] = {}
        
        self.logger.info("Milvus RBAC Manager initialized successfully")
    
    def create_user_context(self, user_id: int, refresh_cache: bool = False) -> MilvusUserContext:
        """
        Create Milvus user context from authentication system.
        
        Args:
            user_id: User ID
            refresh_cache: Force refresh of cached context
            
        Returns:
            Milvus user context
            
        Raises:
            MilvusRBACError: If user context creation fails
        """
        try:
            # Check cache first
            if not refresh_cache and self._is_cache_valid(user_id):
                return self._user_context_cache[user_id]
            
            # Get user from database using proper session context manager
            with self.db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    raise MilvusRBACError(f"User with ID {user_id} not found")
                
                if not user.is_active:
                    raise MilvusRBACError(f"User {user_id} is inactive")
                
                # Get user permissions and roles
                user_permissions = self.auth_rbac_manager.get_user_permissions(user_id)
                user_roles = self.auth_rbac_manager.get_user_roles(user_id)
                
                # Create Milvus user context
                context = MilvusUserContext(
                    user_id=user.id,
                    username=user.username,
                    email=user.email,
                    is_active=user.is_active,
                    is_superuser=user.is_superuser,
                    roles=user_roles,
                    permissions=list(user_permissions),
                    group_ids=self._get_user_groups(user_id),
                    metadata={
                        "full_name": user.full_name,
                        "timezone": getattr(user, 'timezone', None),
                        "last_login": user.last_login.isoformat() if user.last_login else None,
                        "created_at": user.created_at.isoformat()
                    }
                )
                
                # Cache the context
                self._user_context_cache[user_id] = context
                self._cache_timestamps[user_id] = datetime.now(timezone.utc)
                
                self.logger.debug(f"Created Milvus user context for user {user_id}")
                return context
                
        except Exception as e:
            self.logger.error(f"Failed to create user context for user {user_id}: {e}")
            raise MilvusRBACError(f"User context creation failed: {e}")
    
    def _get_user_groups(self, user_id: int) -> List[str]:
        """Get user group IDs (placeholder - extend based on your group system)."""
        # This would be implemented based on your group management system
        # For now, return empty list
        return []
    
    def _is_cache_valid(self, user_id: int) -> bool:
        """Check if cached user context is still valid."""
        if user_id not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[user_id]
        elapsed = (datetime.now(timezone.utc) - cache_time).total_seconds()
        return elapsed < self._cache_ttl
    
    def create_access_rule(self, 
                          rule_id: str,
                          name: str,
                          scope: AccessScope,
                          required_permissions: List[str],
                          allowed_users: Optional[List[int]] = None,
                          allowed_groups: Optional[List[str]] = None,
                          denied_users: Optional[List[int]] = None,
                          denied_groups: Optional[List[str]] = None,
                          metadata_conditions: Optional[Dict[str, Any]] = None) -> AccessRule:
        """
        Create custom access rule for Milvus collections.
        
        Args:
            rule_id: Unique rule identifier
            name: Human-readable rule name
            scope: Access scope (public, private, group, shared)
            required_permissions: Required permission names
            allowed_users: List of allowed user IDs
            allowed_groups: List of allowed group IDs
            denied_users: List of denied user IDs
            denied_groups: List of denied group IDs
            metadata_conditions: Additional metadata conditions
            
        Returns:
            Created access rule
        """
        try:
            # Map permission names to Milvus permissions
            milvus_permissions = []
            for perm in required_permissions:
                if perm in ["read", "query", "search"]:
                    milvus_permissions.append(Permission.READ)
                elif perm in ["write", "create", "update", "insert"]:
                    milvus_permissions.append(Permission.WRITE)
                elif perm in ["delete", "remove"]:
                    milvus_permissions.append(Permission.DELETE)
                elif perm == "admin":
                    milvus_permissions.append(Permission.ADMIN)
            
            # Create access rule
            rule = AccessRule(
                rule_id=rule_id,
                name=name,
                scope=scope,
                required_permissions=milvus_permissions,
                allowed_users=[str(uid) for uid in (allowed_users or [])],
                allowed_groups=allowed_groups or [],
                denied_users=[str(uid) for uid in (denied_users or [])],
                denied_groups=denied_groups or [],
                metadata_conditions=metadata_conditions or {}
            )
            
            # Add rule to RBAC manager
            self.rbac_manager.add_access_rule(rule)
            
            self.logger.info(f"Created access rule: {name} ({rule_id})")
            return rule
            
        except Exception as e:
            self.logger.error(f"Failed to create access rule {rule_id}: {e}")
            raise MilvusRBACError(f"Access rule creation failed: {e}")
    
    def search_with_access_control(self,
                                  collection: MilvusCollection,
                                  user_id: int,
                                  query_vectors: List[List[float]],
                                  limit: int = 10,
                                  search_params: Optional[Dict[str, Any]] = None,
                                  additional_filters: Optional[str] = None,
                                  output_fields: Optional[List[str]] = None,
                                  required_permission: str = "read") -> Any:
        """
        Perform vector search with access control.
        
        Args:
            collection: Milvus collection to search
            user_id: User ID performing the search
            query_vectors: Query vectors for similarity search
            limit: Maximum number of results
            search_params: Search parameters
            additional_filters: Additional filter conditions
            output_fields: Fields to return
            required_permission: Required permission name
            
        Returns:
            Search results with access control applied
            
        Raises:
            InsufficientPermissionsError: If user lacks required permission
            MilvusRBACError: If search fails
        """
        try:
            # Create user context
            user_context = self.create_user_context(user_id)
            rbac_context = user_context.to_rbac_context()
            
            # Map permission name to Milvus permission
            milvus_permission = Permission.READ
            if required_permission in ["write", "create", "update"]:
                milvus_permission = Permission.WRITE
            elif required_permission in ["delete", "remove"]:
                milvus_permission = Permission.DELETE
            elif required_permission == "admin":
                milvus_permission = Permission.ADMIN
            
            # Perform search with RBAC
            result = self.rbac_manager.search_with_rbac(
                collection=collection,
                user_context=rbac_context,
                query_vectors=query_vectors,
                limit=limit,
                search_params=search_params,
                additional_filters=additional_filters,
                output_fields=output_fields,
                required_permission=milvus_permission
            )
            
            self.logger.info(f"Access-controlled search completed for user {user_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Access-controlled search failed for user {user_id}: {e}")
            raise MilvusRBACError(f"Search with access control failed: {e}")
    
    def query_with_access_control(self,
                                 collection: MilvusCollection,
                                 user_id: int,
                                 base_expr: Optional[str] = None,
                                 output_fields: Optional[List[str]] = None,
                                 limit: Optional[int] = None,
                                 required_permission: str = "read") -> List[Dict[str, Any]]:
        """
        Perform query with access control.
        
        Args:
            collection: Milvus collection to query
            user_id: User ID performing the query
            base_expr: Base query expression
            output_fields: Fields to return
            limit: Maximum number of results
            required_permission: Required permission name
            
        Returns:
            Query results with access control applied
            
        Raises:
            InsufficientPermissionsError: If user lacks required permission
            MilvusRBACError: If query fails
        """
        try:
            # Create user context
            user_context = self.create_user_context(user_id)
            rbac_context = user_context.to_rbac_context()
            
            # Map permission name to Milvus permission
            milvus_permission = Permission.READ
            if required_permission in ["write", "create", "update"]:
                milvus_permission = Permission.WRITE
            elif required_permission in ["delete", "remove"]:
                milvus_permission = Permission.DELETE
            elif required_permission == "admin":
                milvus_permission = Permission.ADMIN
            
            # Perform query with RBAC
            results = self.rbac_manager.query_with_rbac(
                collection=collection,
                user_context=rbac_context,
                base_expr=base_expr,
                output_fields=output_fields,
                limit=limit,
                required_permission=milvus_permission
            )
            
            self.logger.info(f"Access-controlled query completed for user {user_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Access-controlled query failed for user {user_id}: {e}")
            raise MilvusRBACError(f"Query with access control failed: {e}")
    
    def can_access_resource(self,
                           user_id: int,
                           resource_metadata: Dict[str, Any],
                           required_permission: str = "read") -> bool:
        """
        Check if user can access a specific resource.
        
        Args:
            user_id: User ID
            resource_metadata: Resource metadata to check
            required_permission: Required permission name
            
        Returns:
            True if user can access resource
        """
        try:
            # Create user context
            user_context = self.create_user_context(user_id)
            rbac_context = user_context.to_rbac_context()
            
            # Map permission name to Milvus permission
            milvus_permission = Permission.READ
            if required_permission in ["write", "create", "update"]:
                milvus_permission = Permission.WRITE
            elif required_permission in ["delete", "remove"]:
                milvus_permission = Permission.DELETE
            elif required_permission == "admin":
                milvus_permission = Permission.ADMIN
            
            # Check access
            return self.rbac_manager.can_access_entity(
                user_context=rbac_context,
                entity_metadata=resource_metadata,
                required_permission=milvus_permission
            )
            
        except Exception as e:
            self.logger.error(f"Access check failed for user {user_id}: {e}")
            return False
    
    def get_user_access_filter(self,
                              user_id: int,
                              required_permission: str = "read",
                              additional_filters: Optional[str] = None) -> str:
        """
        Get access filter expression for user.
        
        Args:
            user_id: User ID
            required_permission: Required permission name
            additional_filters: Additional filter conditions
            
        Returns:
            Boolean expression for access filtering
        """
        try:
            # Create user context
            user_context = self.create_user_context(user_id)
            rbac_context = user_context.to_rbac_context()
            
            # Map permission name to Milvus permission
            milvus_permission = Permission.READ
            if required_permission in ["write", "create", "update"]:
                milvus_permission = Permission.WRITE
            elif required_permission in ["delete", "remove"]:
                milvus_permission = Permission.DELETE
            elif required_permission == "admin":
                milvus_permission = Permission.ADMIN
            
            # Build access filter
            return self.rbac_manager.build_access_filter(
                user_context=rbac_context,
                required_permission=milvus_permission,
                additional_filters=additional_filters
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build access filter for user {user_id}: {e}")
            raise MilvusRBACError(f"Access filter generation failed: {e}")
    
    def validate_collection_rbac(self, collection: MilvusCollection) -> Dict[str, Any]:
        """
        Validate RBAC setup for a collection.
        
        Args:
            collection: Collection to validate
            
        Returns:
            Validation results
        """
        try:
            return self.rbac_manager.validate_rbac_setup(collection)
        except Exception as e:
            self.logger.error(f"RBAC validation failed: {e}")
            raise MilvusRBACError(f"RBAC validation failed: {e}")
    
    def get_rbac_statistics(self) -> Dict[str, Any]:
        """Get RBAC system statistics."""
        try:
            rbac_stats = self.rbac_manager.get_rbac_stats()
            rbac_stats.update({
                "cached_user_contexts": len(self._user_context_cache),
                "cache_ttl_seconds": self._cache_ttl
            })
            return rbac_stats
        except Exception as e:
            self.logger.error(f"Failed to get RBAC statistics: {e}")
            return {"error": str(e)}
    
    def clear_user_cache(self, user_id: Optional[int] = None) -> None:
        """
        Clear user context cache.
        
        Args:
            user_id: Specific user ID to clear, or None for all
        """
        if user_id:
            self._user_context_cache.pop(user_id, None)
            self._cache_timestamps.pop(user_id, None)
            self.rbac_manager.clear_user_cache(str(user_id))
            self.logger.debug(f"Cleared cache for user {user_id}")
        else:
            self._user_context_cache.clear()
            self._cache_timestamps.clear()
            self.rbac_manager.clear_user_cache()
            self.logger.info("Cleared all user caches")
    
    def assign_collection_permissions(self,
                                    collection_name: str,
                                    user_id: int,
                                    permissions: List[str],
                                    scope: AccessScope = AccessScope.PRIVATE) -> None:
        """
        Assign permissions to a user for a specific collection.
        
        Args:
            collection_name: Name of the collection
            user_id: User ID
            permissions: List of permission names
            scope: Access scope
        """
        try:
            rule_id = f"user_{user_id}_{collection_name}"
            rule_name = f"User {user_id} access to {collection_name}"
            
            self.create_access_rule(
                rule_id=rule_id,
                name=rule_name,
                scope=scope,
                required_permissions=permissions,
                allowed_users=[user_id],
                metadata_conditions={"collection": collection_name}
            )
            
            self.logger.info(f"Assigned permissions {permissions} to user {user_id} for collection {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to assign collection permissions: {e}")
            raise MilvusRBACError(f"Permission assignment failed: {e}")
    
    def revoke_collection_permissions(self, collection_name: str, user_id: int) -> None:
        """
        Revoke user permissions for a specific collection.
        
        Args:
            collection_name: Name of the collection
            user_id: User ID
        """
        try:
            rule_id = f"user_{user_id}_{collection_name}"
            self.rbac_manager.remove_access_rule(rule_id)
            
            self.logger.info(f"Revoked permissions for user {user_id} from collection {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to revoke collection permissions: {e}")
            raise MilvusRBACError(f"Permission revocation failed: {e}")