"""
Permission Inheritance System.

This module provides hierarchical permission inheritance for folders,
collections, and nested resources with support for overrides and denial rules.
"""

import json
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading

from src.core.logging import LoggerMixin
from .exceptions import PermissionInheritanceError, AccessControlError


class InheritanceType(Enum):
    """Permission inheritance types."""
    INHERIT = "inherit"           # Inherit from parent
    OVERRIDE = "override"         # Override parent permissions
    DENY = "deny"                # Explicit denial (highest priority)
    GRANT = "grant"              # Explicit grant


class ResourceType(Enum):
    """Resource types for permission inheritance."""
    FOLDER = "folder"
    COLLECTION = "collection"
    DOCUMENT = "document"
    TABLE = "table"
    ROW = "row"
    FIELD = "field"


@dataclass
class PermissionRule:
    """Individual permission rule."""
    resource_id: str
    resource_type: ResourceType
    permission: str
    inheritance_type: InheritanceType
    user_id: Optional[int] = None
    group_id: Optional[str] = None
    role_name: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[int] = None
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if permission rule has expired."""
        return self.expires_at is not None and datetime.now(timezone.utc) > self.expires_at
    
    def matches_subject(self, user_id: int, group_ids: List[str], roles: List[str]) -> bool:
        """Check if rule matches the subject (user, group, or role)."""
        if self.user_id and self.user_id == user_id:
            return True
        if self.group_id and self.group_id in group_ids:
            return True
        if self.role_name and self.role_name in roles:
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "permission": self.permission,
            "inheritance_type": self.inheritance_type.value,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "role_name": self.role_name,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "conditions": self.conditions
        }


@dataclass
class ResourceNode:
    """Node in the resource hierarchy."""
    resource_id: str
    resource_type: ResourceType
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    permissions: List[PermissionRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "permissions": [p.to_dict() for p in self.permissions],
            "metadata": self.metadata
        }


@dataclass
class InheritanceContext:
    """Context for permission inheritance resolution."""
    user_id: int
    group_ids: List[str]
    roles: List[str]
    resource_path: List[str]
    requested_permission: str
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionResult:
    """Result of permission resolution."""
    granted: bool
    inheritance_chain: List[str]
    applied_rules: List[PermissionRule]
    denial_reason: Optional[str] = None
    effective_permissions: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "granted": self.granted,
            "inheritance_chain": self.inheritance_chain,
            "applied_rules": [r.to_dict() for r in self.applied_rules],
            "denial_reason": self.denial_reason,
            "effective_permissions": list(self.effective_permissions)
        }


class PermissionInheritanceManager(LoggerMixin):
    """
    Permission Inheritance Manager.
    
    Manages hierarchical permission inheritance for folders, collections,
    and nested resources with support for overrides and denial rules.
    """
    
    def __init__(self):
        """Initialize permission inheritance manager."""
        self.resource_hierarchy: Dict[str, ResourceNode] = {}
        self.permission_cache: Dict[str, PermissionResult] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        
        # Initialize root resource
        self._create_root_resource()
        
        self.logger.info("Permission Inheritance Manager initialized successfully")
    
    def _create_root_resource(self) -> None:
        """Create root resource for the hierarchy."""
        root_node = ResourceNode(
            resource_id="root",
            resource_type=ResourceType.FOLDER,
            metadata={"name": "Root", "description": "Root of resource hierarchy"}
        )
        self.resource_hierarchy["root"] = root_node
    
    def create_resource(self, 
                       resource_id: str,
                       resource_type: ResourceType,
                       parent_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> ResourceNode:
        """
        Create a new resource in the hierarchy.
        
        Args:
            resource_id: Unique resource identifier
            resource_type: Type of resource
            parent_id: Parent resource ID (None for root level)
            metadata: Additional resource metadata
            
        Returns:
            Created resource node
            
        Raises:
            PermissionInheritanceError: If resource creation fails
        """
        try:
            with self._lock:
                # Check if resource already exists
                if resource_id in self.resource_hierarchy:
                    raise PermissionInheritanceError(f"Resource {resource_id} already exists")
                
                # Validate parent exists
                if parent_id and parent_id not in self.resource_hierarchy:
                    raise PermissionInheritanceError(f"Parent resource {parent_id} not found")
                
                # Create resource node
                node = ResourceNode(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    parent_id=parent_id or "root",
                    metadata=metadata or {}
                )
                
                # Add to hierarchy
                self.resource_hierarchy[resource_id] = node
                
                # Update parent's children
                if parent_id:
                    parent_node = self.resource_hierarchy[parent_id]
                    parent_node.children.append(resource_id)
                else:
                    # Add to root
                    self.resource_hierarchy["root"].children.append(resource_id)
                
                # Clear cache as hierarchy changed
                self._clear_cache()
                
                self.logger.info(f"Created resource: {resource_id} ({resource_type.value})")
                return node
                
        except Exception as e:
            self.logger.error(f"Failed to create resource {resource_id}: {e}")
            raise PermissionInheritanceError(f"Resource creation failed: {e}")
    
    def remove_resource(self, resource_id: str) -> None:
        """
        Remove a resource from the hierarchy.
        
        Args:
            resource_id: Resource ID to remove
            
        Raises:
            PermissionInheritanceError: If resource removal fails
        """
        try:
            with self._lock:
                if resource_id not in self.resource_hierarchy:
                    raise PermissionInheritanceError(f"Resource {resource_id} not found")
                
                if resource_id == "root":
                    raise PermissionInheritanceError("Cannot remove root resource")
                
                node = self.resource_hierarchy[resource_id]
                
                # Remove from parent's children
                if node.parent_id and node.parent_id in self.resource_hierarchy:
                    parent_node = self.resource_hierarchy[node.parent_id]
                    if resource_id in parent_node.children:
                        parent_node.children.remove(resource_id)
                
                # Remove children (recursively)
                for child_id in node.children.copy():
                    self.remove_resource(child_id)
                
                # Remove from hierarchy
                del self.resource_hierarchy[resource_id]
                
                # Clear cache
                self._clear_cache()
                
                self.logger.info(f"Removed resource: {resource_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to remove resource {resource_id}: {e}")
            raise PermissionInheritanceError(f"Resource removal failed: {e}")
    
    def add_permission_rule(self, 
                           resource_id: str,
                           permission: str,
                           inheritance_type: InheritanceType,
                           user_id: Optional[int] = None,
                           group_id: Optional[str] = None,
                           role_name: Optional[str] = None,
                           created_by: Optional[int] = None,
                           expires_at: Optional[datetime] = None,
                           conditions: Optional[Dict[str, Any]] = None) -> PermissionRule:
        """
        Add permission rule to a resource.
        
        Args:
            resource_id: Resource ID
            permission: Permission name
            inheritance_type: Type of inheritance
            user_id: User ID (if user-specific)
            group_id: Group ID (if group-specific)
            role_name: Role name (if role-specific)
            created_by: User who created the rule
            expires_at: Expiration timestamp
            conditions: Additional conditions
            
        Returns:
            Created permission rule
            
        Raises:
            PermissionInheritanceError: If rule creation fails
        """
        try:
            with self._lock:
                if resource_id not in self.resource_hierarchy:
                    raise PermissionInheritanceError(f"Resource {resource_id} not found")
                
                # Validate subject (at least one must be specified)
                if not any([user_id, group_id, role_name]):
                    raise PermissionInheritanceError("At least one subject (user, group, or role) must be specified")
                
                node = self.resource_hierarchy[resource_id]
                
                # Create permission rule
                rule = PermissionRule(
                    resource_id=resource_id,
                    resource_type=node.resource_type,
                    permission=permission,
                    inheritance_type=inheritance_type,
                    user_id=user_id,
                    group_id=group_id,
                    role_name=role_name,
                    created_by=created_by,
                    expires_at=expires_at,
                    conditions=conditions or {}
                )
                
                # Add to resource
                node.permissions.append(rule)
                
                # Clear cache
                self._clear_cache()
                
                self.logger.info(f"Added permission rule: {permission} for {resource_id}")
                return rule
                
        except Exception as e:
            self.logger.error(f"Failed to add permission rule: {e}")
            raise PermissionInheritanceError(f"Permission rule creation failed: {e}")
    
    def remove_permission_rule(self, 
                              resource_id: str,
                              permission: str,
                              user_id: Optional[int] = None,
                              group_id: Optional[str] = None,
                              role_name: Optional[str] = None) -> bool:
        """
        Remove permission rule from a resource.
        
        Args:
            resource_id: Resource ID
            permission: Permission name
            user_id: User ID (if user-specific)
            group_id: Group ID (if group-specific)
            role_name: Role name (if role-specific)
            
        Returns:
            True if rule was removed
        """
        try:
            with self._lock:
                if resource_id not in self.resource_hierarchy:
                    return False
                
                node = self.resource_hierarchy[resource_id]
                
                # Find matching rules
                rules_to_remove = []
                for rule in node.permissions:
                    if (rule.permission == permission and
                        rule.user_id == user_id and
                        rule.group_id == group_id and
                        rule.role_name == role_name):
                        rules_to_remove.append(rule)
                
                # Remove rules
                for rule in rules_to_remove:
                    node.permissions.remove(rule)
                
                if rules_to_remove:
                    # Clear cache
                    self._clear_cache()
                    
                    self.logger.info(f"Removed {len(rules_to_remove)} permission rules for {resource_id}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove permission rule: {e}")
            return False
    
    def resolve_permissions(self, context: InheritanceContext) -> PermissionResult:
        """
        Resolve permissions for a user on a resource.
        
        Args:
            context: Inheritance context
            
        Returns:
            Permission resolution result
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(context)
            
            # Check cache
            if self._is_cache_valid(cache_key):
                return self.permission_cache[cache_key]
            
            # Get resource path
            resource_path = self._get_resource_path(context.resource_path[-1])
            
            # Resolve permissions along the path
            result = self._resolve_permissions_recursive(context, resource_path)
            
            # Cache result
            with self._lock:
                self.permission_cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resolve permissions: {e}")
            raise PermissionInheritanceError(f"Permission resolution failed: {e}")
    
    def _resolve_permissions_recursive(self, 
                                     context: InheritanceContext,
                                     resource_path: List[str]) -> PermissionResult:
        """Recursively resolve permissions along the resource path."""
        try:
            applied_rules = []
            inheritance_chain = []
            effective_permissions = set()
            denial_reason = None
            
            # Process from root to target resource
            for resource_id in resource_path:
                if resource_id not in self.resource_hierarchy:
                    continue
                
                node = self.resource_hierarchy[resource_id]
                inheritance_chain.append(resource_id)
                
                # Get applicable rules for this resource
                applicable_rules = self._get_applicable_rules(node, context)
                
                # Process rules by priority (DENY > OVERRIDE > GRANT > INHERIT)
                for rule in sorted(applicable_rules, key=lambda r: self._get_rule_priority(r.inheritance_type)):
                    if rule.is_expired():
                        continue
                    
                    applied_rules.append(rule)
                    
                    # Handle different inheritance types
                    if rule.inheritance_type == InheritanceType.DENY:
                        # Denial overrides everything
                        denial_reason = f"Explicitly denied by rule on {resource_id}"
                        return PermissionResult(
                            granted=False,
                            inheritance_chain=inheritance_chain,
                            applied_rules=applied_rules,
                            denial_reason=denial_reason,
                            effective_permissions=set()
                        )
                    
                    elif rule.inheritance_type == InheritanceType.OVERRIDE:
                        # Override clears previous permissions and sets new ones
                        effective_permissions.clear()
                        effective_permissions.add(rule.permission)
                    
                    elif rule.inheritance_type == InheritanceType.GRANT:
                        # Grant adds to existing permissions
                        effective_permissions.add(rule.permission)
                    
                    elif rule.inheritance_type == InheritanceType.INHERIT:
                        # Inherit from parent (already handled by path traversal)
                        effective_permissions.add(rule.permission)
            
            # Check if requested permission is granted
            granted = (context.requested_permission in effective_permissions or
                      "*" in effective_permissions)
            
            return PermissionResult(
                granted=granted,
                inheritance_chain=inheritance_chain,
                applied_rules=applied_rules,
                effective_permissions=effective_permissions
            )
            
        except Exception as e:
            self.logger.error(f"Failed to resolve permissions recursively: {e}")
            raise PermissionInheritanceError(f"Recursive permission resolution failed: {e}")
    
    def _get_applicable_rules(self, node: ResourceNode, context: InheritanceContext) -> List[PermissionRule]:
        """Get applicable permission rules for a node and context."""
        applicable_rules = []
        
        for rule in node.permissions:
            # Check if rule applies to the requested permission
            if rule.permission != context.requested_permission and rule.permission != "*":
                continue
            
            # Check if rule matches the subject
            if rule.matches_subject(context.user_id, context.group_ids, context.roles):
                # Check additional conditions
                if self._check_rule_conditions(rule, context):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_rule_conditions(self, rule: PermissionRule, context: InheritanceContext) -> bool:
        """Check if rule conditions are met."""
        if not rule.conditions:
            return True
        
        # Check time-based conditions
        if "time_range" in rule.conditions:
            time_range = rule.conditions["time_range"]
            current_time = datetime.now(timezone.utc).time()
            
            start_time = datetime.strptime(time_range["start"], "%H:%M").time()
            end_time = datetime.strptime(time_range["end"], "%H:%M").time()
            
            if not (start_time <= current_time <= end_time):
                return False
        
        # Check IP-based conditions
        if "ip_range" in rule.conditions:
            client_ip = context.additional_context.get("client_ip")
            if not client_ip or not self._check_ip_range(client_ip, rule.conditions["ip_range"]):
                return False
        
        # Check custom conditions
        if "custom" in rule.conditions:
            custom_conditions = rule.conditions["custom"]
            for key, value in custom_conditions.items():
                if context.additional_context.get(key) != value:
                    return False
        
        return True
    
    def _check_ip_range(self, client_ip: str, ip_range: str) -> bool:
        """Check if client IP is within allowed range."""
        try:
            from ipaddress import ip_address, ip_network
            
            client = ip_address(client_ip)
            network = ip_network(ip_range, strict=False)
            
            return client in network
            
        except Exception:
            return False
    
    def _get_rule_priority(self, inheritance_type: InheritanceType) -> int:
        """Get rule priority for sorting (lower number = higher priority)."""
        priorities = {
            InheritanceType.DENY: 0,
            InheritanceType.OVERRIDE: 1,
            InheritanceType.GRANT: 2,
            InheritanceType.INHERIT: 3
        }
        return priorities.get(inheritance_type, 999)
    
    def _get_resource_path(self, resource_id: str) -> List[str]:
        """Get full path from root to resource."""
        if resource_id not in self.resource_hierarchy:
            return []
        
        path = []
        current_id = resource_id
        
        while current_id and current_id in self.resource_hierarchy:
            path.append(current_id)
            node = self.resource_hierarchy[current_id]
            current_id = node.parent_id
            
            # Prevent infinite loops
            if current_id == "root":
                path.append("root")
                break
        
        return list(reversed(path))
    
    def _generate_cache_key(self, context: InheritanceContext) -> str:
        """Generate cache key for permission resolution."""
        key_parts = [
            str(context.user_id),
            ",".join(context.group_ids),
            ",".join(context.roles),
            ",".join(context.resource_path),
            context.requested_permission,
            json.dumps(context.additional_context, sort_keys=True)
        ]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        elapsed = (datetime.now(timezone.utc) - cache_time).total_seconds()
        return elapsed < self.cache_ttl
    
    def _clear_cache(self) -> None:
        """Clear permission cache."""
        with self._lock:
            self.permission_cache.clear()
            self.cache_timestamps.clear()
    
    def get_resource_permissions(self, resource_id: str, user_id: int, 
                               group_ids: List[str], roles: List[str]) -> List[str]:
        """
        Get all permissions for a user on a resource.
        
        Args:
            resource_id: Resource ID
            user_id: User ID
            group_ids: User's group IDs
            roles: User's roles
            
        Returns:
            List of granted permissions
        """
        try:
            if resource_id not in self.resource_hierarchy:
                return []
            
            # Get all possible permissions
            all_permissions = set()
            resource_path = self._get_resource_path(resource_id)
            
            for path_resource_id in resource_path:
                if path_resource_id in self.resource_hierarchy:
                    node = self.resource_hierarchy[path_resource_id]
                    for rule in node.permissions:
                        if rule.matches_subject(user_id, group_ids, roles):
                            all_permissions.add(rule.permission)
            
            # Resolve each permission
            granted_permissions = []
            for permission in all_permissions:
                context = InheritanceContext(
                    user_id=user_id,
                    group_ids=group_ids,
                    roles=roles,
                    resource_path=[resource_id],
                    requested_permission=permission
                )
                
                result = self.resolve_permissions(context)
                if result.granted:
                    granted_permissions.append(permission)
            
            return granted_permissions
            
        except Exception as e:
            self.logger.error(f"Failed to get resource permissions: {e}")
            return []
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get hierarchy statistics."""
        with self._lock:
            total_resources = len(self.resource_hierarchy)
            total_rules = sum(len(node.permissions) for node in self.resource_hierarchy.values())
            
            resource_types = {}
            for node in self.resource_hierarchy.values():
                resource_type = node.resource_type.value
                resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
            
            return {
                "total_resources": total_resources,
                "total_permission_rules": total_rules,
                "resource_types": resource_types,
                "cached_results": len(self.permission_cache),
                "cache_ttl_seconds": self.cache_ttl
            }
    
    def export_hierarchy(self) -> Dict[str, Any]:
        """Export entire hierarchy to dictionary."""
        with self._lock:
            return {
                "resources": {
                    resource_id: node.to_dict() 
                    for resource_id, node in self.resource_hierarchy.items()
                },
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }
    
    def import_hierarchy(self, hierarchy_data: Dict[str, Any]) -> None:
        """Import hierarchy from dictionary."""
        try:
            with self._lock:
                # Clear existing hierarchy
                self.resource_hierarchy.clear()
                self._clear_cache()
                
                # Import resources
                resources_data = hierarchy_data.get("resources", {})
                for resource_id, resource_data in resources_data.items():
                    # Create resource node
                    node = ResourceNode(
                        resource_id=resource_data["resource_id"],
                        resource_type=ResourceType(resource_data["resource_type"]),
                        parent_id=resource_data.get("parent_id"),
                        children=resource_data.get("children", []),
                        metadata=resource_data.get("metadata", {})
                    )
                    
                    # Import permission rules
                    for rule_data in resource_data.get("permissions", []):
                        rule = PermissionRule(
                            resource_id=rule_data["resource_id"],
                            resource_type=ResourceType(rule_data["resource_type"]),
                            permission=rule_data["permission"],
                            inheritance_type=InheritanceType(rule_data["inheritance_type"]),
                            user_id=rule_data.get("user_id"),
                            group_id=rule_data.get("group_id"),
                            role_name=rule_data.get("role_name"),
                            created_at=datetime.fromisoformat(rule_data["created_at"]),
                            created_by=rule_data.get("created_by"),
                            expires_at=datetime.fromisoformat(rule_data["expires_at"]) if rule_data.get("expires_at") else None,
                            conditions=rule_data.get("conditions", {})
                        )
                        node.permissions.append(rule)
                    
                    self.resource_hierarchy[resource_id] = node
                
                self.logger.info(f"Imported hierarchy with {len(self.resource_hierarchy)} resources")
                
        except Exception as e:
            self.logger.error(f"Failed to import hierarchy: {e}")
            raise PermissionInheritanceError(f"Hierarchy import failed: {e}")