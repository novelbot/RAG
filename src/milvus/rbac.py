"""
Row-Level Role-Based Access Control (RBAC) for Milvus vector database.
"""

import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json
import threading

from pymilvus import MilvusException
from loguru import logger

from src.core.exceptions import MilvusError, RBACError, PermissionError
from src.core.logging import LoggerMixin
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection, SearchResult


class Permission(Enum):
    """Permission levels for RBAC."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class AccessScope(Enum):
    """Access scope for data filtering."""
    PUBLIC = "public"           # Accessible by all users
    PRIVATE = "private"         # Accessible only by owner
    GROUP = "group"             # Accessible by group members
    SHARED = "shared"           # Accessible by shared users


@dataclass
class UserContext:
    """User context for RBAC operations."""
    user_id: str
    group_ids: List[str] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    is_admin: bool = False
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return self.is_admin or permission in self.permissions
    
    def is_member_of_group(self, group_id: str) -> bool:
        """Check if user is member of specific group."""
        return group_id in self.group_ids


@dataclass
class AccessRule:
    """Access rule definition for RBAC."""
    rule_id: str
    name: str
    scope: AccessScope
    required_permissions: List[Permission]
    allowed_users: List[str] = field(default_factory=list)
    allowed_groups: List[str] = field(default_factory=list)
    denied_users: List[str] = field(default_factory=list)
    denied_groups: List[str] = field(default_factory=list)
    metadata_conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


class RBACManager(LoggerMixin):
    """
    Row-Level Role-Based Access Control Manager for Milvus.
    
    Based on Context7 documentation for PyMilvus metadata filtering:
    - Uses boolean expressions with metadata fields for access control
    - Implements user_id, group_ids, and permissions filtering
    - Supports complex access rules and inheritance
    """
    
    def __init__(self, 
                 client: MilvusClient,
                 default_rules: Optional[List[AccessRule]] = None):
        """
        Initialize RBAC Manager.
        
        Args:
            client: Milvus client instance
            default_rules: Default access rules to apply
        """
        self.client = client
        self._access_rules: Dict[str, AccessRule] = {}
        self._user_cache: Dict[str, UserContext] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_timestamps: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        # Initialize default rules
        if default_rules:
            for rule in default_rules:
                self.add_access_rule(rule)
        else:
            self._create_default_rules()
    
    def _create_default_rules(self) -> None:
        """Create default access rules."""
        # Public read access
        public_read = AccessRule(
            rule_id="public_read",
            name="Public Read Access",
            scope=AccessScope.PUBLIC,
            required_permissions=[Permission.READ]
        )
        
        # Private access (owner only)
        private_access = AccessRule(
            rule_id="private_access",
            name="Private User Access",
            scope=AccessScope.PRIVATE,
            required_permissions=[Permission.READ, Permission.WRITE, Permission.DELETE]
        )
        
        # Group access
        group_access = AccessRule(
            rule_id="group_access",
            name="Group Member Access",
            scope=AccessScope.GROUP,
            required_permissions=[Permission.READ]
        )
        
        # Admin access
        admin_access = AccessRule(
            rule_id="admin_access",
            name="Administrator Access",
            scope=AccessScope.PUBLIC,
            required_permissions=[Permission.ADMIN]
        )
        
        self.add_access_rule(public_read)
        self.add_access_rule(private_access)
        self.add_access_rule(group_access)
        self.add_access_rule(admin_access)
    
    def add_access_rule(self, rule: AccessRule) -> None:
        """Add new access rule."""
        with self._lock:
            self._access_rules[rule.rule_id] = rule
            self.logger.info(f"Added access rule: {rule.name}")
    
    def remove_access_rule(self, rule_id: str) -> None:
        """Remove access rule."""
        with self._lock:
            if rule_id in self._access_rules:
                rule = self._access_rules.pop(rule_id)
                self.logger.info(f"Removed access rule: {rule.name}")
    
    def get_access_rule(self, rule_id: str) -> Optional[AccessRule]:
        """Get access rule by ID."""
        with self._lock:
            return self._access_rules.get(rule_id)
    
    def list_access_rules(self) -> List[AccessRule]:
        """List all access rules."""
        with self._lock:
            return list(self._access_rules.values())
    
    def create_user_context(self, 
                          user_id: str,
                          group_ids: Optional[List[str]] = None,
                          permissions: Optional[List[Permission]] = None,
                          is_admin: bool = False,
                          **kwargs) -> UserContext:
        """Create user context for RBAC operations."""
        context = UserContext(
            user_id=user_id,
            group_ids=group_ids or [],
            permissions=permissions or [],
            is_admin=is_admin,
            additional_metadata=kwargs
        )
        
        # Cache user context
        with self._lock:
            self._user_cache[user_id] = context
            self._cache_timestamps[user_id] = datetime.now(timezone.utc)
        
        self.logger.debug(f"Created user context for: {user_id}")
        return context
    
    def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Get cached user context."""
        with self._lock:
            # Check cache validity
            if user_id in self._cache_timestamps:
                cache_time = self._cache_timestamps[user_id]
                elapsed = (datetime.now(timezone.utc) - cache_time).total_seconds()
                
                if elapsed > self._cache_ttl:
                    # Cache expired
                    self._user_cache.pop(user_id, None)
                    self._cache_timestamps.pop(user_id, None)
                    return None
            
            return self._user_cache.get(user_id)
    
    def build_access_filter(self, 
                           user_context: UserContext,
                           required_permission: Permission = Permission.READ,
                           additional_filters: Optional[str] = None) -> str:
        """
        Build Milvus boolean expression for access control filtering.
        
        Based on Context7 PyMilvus search expressions:
        - Uses metadata fields: user_id, group_ids, permissions
        - Supports complex boolean conditions
        - Implements hierarchical access control
        
        Args:
            user_context: User context for access control
            required_permission: Required permission level
            additional_filters: Additional filter conditions
            
        Returns:
            Boolean expression string for Milvus filtering
        """
        conditions = []
        
        # Admin access - bypass all restrictions
        if user_context.is_admin:
            admin_condition = 'JSON_CONTAINS(permissions, \'{"admin": true}\')'
            conditions.append(f"({admin_condition} OR user_id == '{user_context.user_id}')")
        else:
            # Apply permission-based filtering
            permission_conditions = []
            
            # 1. Public access - accessible by all users with required permission
            if user_context.has_permission(required_permission):
                public_condition = 'JSON_CONTAINS(permissions, "{\\"scope\\": \\"public\\"}")'
                permission_conditions.append(public_condition)
            
            # 2. Private access - accessible only by owner
            private_condition = f"user_id == '{user_context.user_id}'"
            permission_conditions.append(private_condition)
            
            # 3. Group access - accessible by group members
            if user_context.group_ids:
                group_conditions = []
                for group_id in user_context.group_ids:
                    group_condition = f'JSON_CONTAINS(group_ids, "\\"{group_id}\\"")'
                    group_conditions.append(group_condition)
                
                if group_conditions:
                    group_access = f"({' OR '.join(group_conditions)})"
                    permission_conditions.append(group_access)
            
            # 4. Shared access - explicitly shared with user
            shared_condition = f'JSON_CONTAINS(permissions, "{{\\"shared_users\\": [\\"{user_context.user_id}\\"]}}")'
            permission_conditions.append(shared_condition)
            
            # Combine permission conditions
            if permission_conditions:
                conditions.append(f"({' OR '.join(permission_conditions)})")
        
        # Apply access rules
        rule_conditions = self._build_rule_conditions(user_context, required_permission)
        if rule_conditions:
            conditions.extend(rule_conditions)
        
        # Add additional filters
        if additional_filters:
            conditions.append(f"({additional_filters})")
        
        # Combine all conditions
        if not conditions:
            # No access allowed
            return "user_id == 'NO_ACCESS'"
        
        final_expression = " AND ".join(conditions)
        self.logger.debug(f"Built access filter for {user_context.user_id}: {final_expression}")
        
        return final_expression
    
    def _build_rule_conditions(self, 
                             user_context: UserContext,
                             required_permission: Permission) -> List[str]:
        """Build conditions based on access rules."""
        conditions = []
        
        with self._lock:
            for rule in self._access_rules.values():
                if not rule.is_active:
                    continue
                
                # Check if rule applies to this permission
                if required_permission not in rule.required_permissions and not user_context.is_admin:
                    continue
                
                rule_conditions = []
                
                # Check allowed users
                if rule.allowed_users and user_context.user_id in rule.allowed_users:
                    rule_conditions.append("1 == 1")  # Always true condition
                
                # Check allowed groups
                if rule.allowed_groups:
                    for group_id in user_context.group_ids:
                        if group_id in rule.allowed_groups:
                            rule_conditions.append("1 == 1")
                            break
                
                # Check denied users (override)
                if rule.denied_users and user_context.user_id in rule.denied_users:
                    rule_conditions = ["1 == 0"]  # Always false condition
                
                # Check denied groups (override)
                if rule.denied_groups:
                    for group_id in user_context.group_ids:
                        if group_id in rule.denied_groups:
                            rule_conditions = ["1 == 0"]
                            break
                
                # Apply metadata conditions
                if rule.metadata_conditions:
                    for field, value in rule.metadata_conditions.items():
                        if isinstance(value, str):
                            rule_conditions.append(f'{field} == "{value}"')
                        elif isinstance(value, (int, float)):
                            rule_conditions.append(f'{field} == {value}')
                        elif isinstance(value, bool):
                            rule_conditions.append(f'{field} == {str(value).lower()}')
                
                if rule_conditions:
                    combined_rule = " AND ".join(rule_conditions)
                    conditions.append(f"({combined_rule})")
        
        return conditions
    
    def search_with_rbac(self,
                        collection: MilvusCollection,
                        user_context: UserContext,
                        query_vectors: List[List[float]],
                        limit: int = 10,
                        search_params: Optional[Dict[str, Any]] = None,
                        additional_filters: Optional[str] = None,
                        output_fields: Optional[List[str]] = None,
                        required_permission: Permission = Permission.READ) -> SearchResult:
        """
        Perform vector search with RBAC filtering.
        
        Args:
            collection: Milvus collection to search
            user_context: User context for access control
            query_vectors: Query vectors for similarity search
            limit: Maximum number of results
            search_params: Search parameters
            additional_filters: Additional filter conditions
            output_fields: Fields to return
            required_permission: Required permission level
            
        Returns:
            SearchResult with access-controlled results
        """
        try:
            # Check if user has required permission
            if not user_context.has_permission(required_permission) and not user_context.is_admin:
                raise PermissionError(f"User {user_context.user_id} lacks {required_permission.value} permission")
            
            # Build access control filter
            access_filter = self.build_access_filter(
                user_context=user_context,
                required_permission=required_permission,
                additional_filters=additional_filters
            )
            
            # Perform search with access control
            start_time = time.time()
            result = collection.search(
                query_vectors=query_vectors,
                limit=limit,
                search_params=search_params,
                expr=access_filter,
                output_fields=output_fields
            )
            
            # Log access
            search_time = time.time() - start_time
            self.logger.info(f"RBAC search completed for {user_context.user_id}: "
                          f"{len(result.hits)} results in {search_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"RBAC search failed for {user_context.user_id}: {e}")
            raise RBACError(f"Access-controlled search failed: {e}")
    
    def query_with_rbac(self,
                       collection: MilvusCollection,
                       user_context: UserContext,
                       base_expr: Optional[str] = None,
                       output_fields: Optional[List[str]] = None,
                       limit: Optional[int] = None,
                       required_permission: Permission = Permission.READ) -> List[Dict[str, Any]]:
        """
        Perform query with RBAC filtering.
        
        Args:
            collection: Milvus collection to query
            user_context: User context for access control
            base_expr: Base query expression
            output_fields: Fields to return
            limit: Maximum number of results
            required_permission: Required permission level
            
        Returns:
            Query results with access control applied
        """
        try:
            # Check if user has required permission
            if not user_context.has_permission(required_permission) and not user_context.is_admin:
                raise PermissionError(f"User {user_context.user_id} lacks {required_permission.value} permission")
            
            # Build access control filter
            access_filter = self.build_access_filter(
                user_context=user_context,
                required_permission=required_permission,
                additional_filters=base_expr
            )
            
            # Perform query with access control
            results = collection.query(
                expr=access_filter,
                output_fields=output_fields,
                limit=limit
            )
            
            self.logger.info(f"RBAC query completed for {user_context.user_id}: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"RBAC query failed for {user_context.user_id}: {e}")
            raise RBACError(f"Access-controlled query failed: {e}")
    
    def can_access_entity(self,
                         user_context: UserContext,
                         entity_metadata: Dict[str, Any],
                         required_permission: Permission = Permission.READ) -> bool:
        """
        Check if user can access specific entity.
        
        Args:
            user_context: User context for access control
            entity_metadata: Entity metadata to check
            required_permission: Required permission level
            
        Returns:
            True if user can access entity
        """
        try:
            # Admin always has access
            if user_context.is_admin:
                return True
            
            # Check if user has required permission
            if not user_context.has_permission(required_permission):
                return False
            
            # Check owner access
            entity_user_id = entity_metadata.get("user_id")
            if entity_user_id == user_context.user_id:
                return True
            
            # Check group access
            entity_groups = entity_metadata.get("group_ids", [])
            if any(group in user_context.group_ids for group in entity_groups):
                return True
            
            # Check public access
            permissions = entity_metadata.get("permissions", {})
            if permissions.get("scope") == "public":
                return True
            
            # Check shared access
            shared_users = permissions.get("shared_users", [])
            if user_context.user_id in shared_users:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Access check failed: {e}")
            return False
    
    def get_rbac_stats(self) -> Dict[str, Any]:
        """Get RBAC system statistics."""
        with self._lock:
            return {
                "total_rules": len(self._access_rules),
                "active_rules": len([r for r in self._access_rules.values() if r.is_active]),
                "cached_users": len(self._user_cache),
                "cache_ttl": self._cache_ttl,
                "rules_by_scope": {
                    scope.value: len([r for r in self._access_rules.values() 
                                   if r.scope == scope])
                    for scope in AccessScope
                }
            }
    
    def validate_rbac_setup(self, collection: MilvusCollection) -> Dict[str, Any]:
        """
        Validate RBAC setup for collection.
        
        Args:
            collection: Collection to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            "status": "unknown",
            "required_fields": [],
            "missing_fields": [],
            "recommendations": []
        }
        
        try:
            # Check collection schema for RBAC fields
            schema_info = collection.schema.get_schema_info()
            field_names = [field["name"] for field in schema_info["fields"]]
            
            # Required RBAC fields
            required_fields = ["user_id", "group_ids", "permissions"]
            missing_fields = [field for field in required_fields if field not in field_names]
            
            validation_results["required_fields"] = required_fields
            validation_results["missing_fields"] = missing_fields
            
            if not missing_fields:
                validation_results["status"] = "valid"
            else:
                validation_results["status"] = "invalid"
                validation_results["recommendations"].append(
                    f"Add missing RBAC fields to collection schema: {missing_fields}"
                )
            
            # Check access rules
            if not self._access_rules:
                validation_results["recommendations"].append(
                    "No access rules defined. Consider adding default rules."
                )
            
            self.logger.info(f"RBAC validation completed: {validation_results['status']}")
            return validation_results
            
        except Exception as e:
            validation_results["status"] = "error"
            validation_results["error"] = str(e)
            self.logger.error(f"RBAC validation failed: {e}")
            return validation_results
    
    def clear_user_cache(self, user_id: Optional[str] = None) -> None:
        """Clear user cache."""
        with self._lock:
            if user_id:
                self._user_cache.pop(user_id, None)
                self._cache_timestamps.pop(user_id, None)
                self.logger.debug(f"Cleared cache for user: {user_id}")
            else:
                self._user_cache.clear()
                self._cache_timestamps.clear()
                self.logger.info("Cleared all user cache")


def create_rbac_manager(client: MilvusClient, 
                       custom_rules: Optional[List[AccessRule]] = None) -> RBACManager:
    """
    Create RBAC Manager with optional custom rules.
    
    Args:
        client: Milvus client instance
        custom_rules: Custom access rules
        
    Returns:
        Configured RBAC Manager
    """
    return RBACManager(client=client, default_rules=custom_rules)


def create_default_user_context(user_id: str, 
                               is_admin: bool = False,
                               groups: Optional[List[str]] = None) -> UserContext:
    """
    Create default user context with basic permissions.
    
    Args:
        user_id: User identifier
        is_admin: Whether user has admin privileges
        groups: User's group memberships
        
    Returns:
        User context with default permissions
    """
    permissions = [Permission.READ]
    if is_admin:
        permissions.extend([Permission.WRITE, Permission.DELETE, Permission.ADMIN])
    
    return UserContext(
        user_id=user_id,
        group_ids=groups or [],
        permissions=permissions,
        is_admin=is_admin
    )