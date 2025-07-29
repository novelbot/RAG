"""
Access Control Manager.

This module provides a comprehensive access control system integrating
all components: Milvus RBAC, metadata filtering, permission inheritance,
group management, and audit logging.
"""

from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from src.milvus.client import MilvusClient
from src.milvus.collection import MilvusCollection
from src.auth.rbac import RBACManager as AuthRBACManager
from .milvus_rbac import MilvusRBACManager
from .metadata_filter import MetadataFilter, FilterGroup
from .permission_inheritance import (
    PermissionInheritanceManager, InheritanceContext, ResourceType
)
from .group_manager import GroupManager
from .audit_logger import AuditLogger, AuditEventType, AuditSeverity, AuditResult, AuditContext
from .exceptions import AccessControlError, InsufficientPermissionsError


@dataclass
class AccessRequest:
    """Access request for comprehensive authorization."""
    user_id: int
    resource_type: str
    resource_id: str
    action: str
    context: Optional[Dict[str, Any]] = None
    client_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "context": self.context,
            "client_info": self.client_info
        }


@dataclass
class AccessResponse:
    """Response from access control evaluation."""
    granted: bool
    reason: Optional[str] = None
    required_permissions: List[str] = field(default_factory=list)
    applied_filters: List[str] = field(default_factory=list)
    audit_event_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "granted": self.granted,
            "reason": self.reason,
            "required_permissions": self.required_permissions,
            "applied_filters": self.applied_filters,
            "audit_event_id": self.audit_event_id,
            "metadata": self.metadata
        }


class AccessControlManager(LoggerMixin):
    """
    Comprehensive Access Control Manager.
    
    Integrates all access control components to provide unified
    authorization, auditing, and security management.
    """
    
    def __init__(self, 
                 milvus_client: MilvusClient,
                 db_manager: DatabaseManager,
                 auth_rbac_manager: AuthRBACManager):
        """
        Initialize Access Control Manager.
        
        Args:
            milvus_client: Milvus client instance
            db_manager: Database manager
            auth_rbac_manager: Authentication RBAC manager
        """
        self.milvus_client = milvus_client
        self.db_manager = db_manager
        self.auth_rbac_manager = auth_rbac_manager
        
        # Initialize all components
        self.milvus_rbac = MilvusRBACManager(milvus_client, db_manager, auth_rbac_manager)
        self.metadata_filter = MetadataFilter()
        self.permission_inheritance = PermissionInheritanceManager()
        self.group_manager = GroupManager(db_manager, auth_rbac_manager)
        self.audit_logger = AuditLogger(db_manager)
        
        # Configuration
        self.enable_strict_mode = True
        self.enable_audit_logging = True
        self.default_deny = True
        
        self.logger.info("Access Control Manager initialized successfully")
    
    def check_access(self, request: AccessRequest) -> AccessResponse:
        """
        Comprehensive access control check.
        
        Args:
            request: Access request
            
        Returns:
            Access response with decision and metadata
        """
        try:
            # Create audit context
            audit_context = AuditContext(
                user_id=request.user_id,
                client_info=request.client_info,
                additional_data=request.context
            )
            
            # Get user context
            user_context = self.milvus_rbac.create_user_context(request.user_id)
            
            # Check basic permissions
            has_permission = self.auth_rbac_manager.has_permission(
                request.user_id, 
                f"{request.resource_type}:{request.action}"
            )
            
            if not has_permission:
                # Check group permissions
                effective_permissions = self.group_manager.get_user_effective_permissions(request.user_id)
                has_permission = f"{request.resource_type}:{request.action}" in effective_permissions
            
            # Check permission inheritance
            inheritance_context = InheritanceContext(
                user_id=request.user_id,
                group_ids=[str(gid) for gid in user_context.group_ids],
                roles=user_context.roles,
                resource_path=[request.resource_id],
                requested_permission=request.action,
                additional_context=request.context or {}
            )
            
            inheritance_result = self.permission_inheritance.resolve_permissions(inheritance_context)
            
            # Final decision
            granted = has_permission and inheritance_result.granted
            
            # Log audit event
            audit_event_id = None
            if self.enable_audit_logging:
                audit_log = self.audit_logger.log_event(
                    event_type=AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED,
                    message=f"Access {'granted' if granted else 'denied'} for {request.action} on {request.resource_type}:{request.resource_id}",
                    context=audit_context,
                    severity=AuditSeverity.MEDIUM,
                    result=AuditResult.SUCCESS if granted else AuditResult.FAILURE,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    metadata=request.to_dict()
                )
                if audit_log:
                    audit_event_id = audit_log.id
            
            response = AccessResponse(
                granted=granted,
                reason=inheritance_result.denial_reason if not granted else None,
                required_permissions=[f"{request.resource_type}:{request.action}"],
                applied_filters=inheritance_result.inheritance_chain,
                audit_event_id=audit_event_id,
                metadata={
                    "user_context": user_context.to_dict(),
                    "inheritance_result": inheritance_result.to_dict()
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Access control check failed: {e}")
            
            # Log audit event for error
            if self.enable_audit_logging:
                self.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    message=f"Access control check failed: {e}",
                    context=AuditContext(user_id=request.user_id),
                    severity=AuditSeverity.HIGH,
                    result=AuditResult.FAILURE,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id
                )
            
            # Return deny on error if strict mode enabled
            return AccessResponse(
                granted=not self.enable_strict_mode,
                reason=f"Access control error: {e}",
                metadata={"error": str(e)}
            )
    
    def search_with_access_control(self,
                                  collection: MilvusCollection,
                                  user_id: int,
                                  query_vectors: List[List[float]],
                                  limit: int = 10,
                                  search_params: Optional[Dict[str, Any]] = None,
                                  additional_filters: Optional[str] = None,
                                  output_fields: Optional[List[str]] = None) -> Any:
        """
        Perform vector search with comprehensive access control.
        
        Args:
            collection: Milvus collection to search
            user_id: User ID performing the search
            query_vectors: Query vectors for similarity search
            limit: Maximum number of results
            search_params: Search parameters
            additional_filters: Additional filter conditions
            output_fields: Fields to return
            
        Returns:
            Search results with access control applied
        """
        try:
            # Log search attempt
            if self.enable_audit_logging:
                self.audit_logger.log_event(
                    event_type=AuditEventType.RESOURCE_ACCESSED,
                    message=f"Vector search initiated on collection {collection.collection_name}",
                    context=AuditContext(user_id=user_id),
                    severity=AuditSeverity.LOW,
                    result=AuditResult.SUCCESS,
                    resource_type="collection",
                    resource_id=collection.collection_name
                )
            
            # Perform search with RBAC
            results = self.milvus_rbac.search_with_access_control(
                collection=collection,
                user_id=user_id,
                query_vectors=query_vectors,
                limit=limit,
                search_params=search_params,
                additional_filters=additional_filters,
                output_fields=output_fields
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Access-controlled search failed: {e}")
            
            # Log audit event
            if self.enable_audit_logging:
                self.audit_logger.log_event(
                    event_type=AuditEventType.ACCESS_DENIED,
                    message=f"Vector search failed: {e}",
                    context=AuditContext(user_id=user_id),
                    severity=AuditSeverity.MEDIUM,
                    result=AuditResult.FAILURE,
                    resource_type="collection",
                    resource_id=collection.collection_name
                )
            
            raise AccessControlError(f"Access-controlled search failed: {e}")
    
    def get_user_access_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive access summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User access summary
        """
        try:
            # Get user context
            user_context = self.milvus_rbac.create_user_context(user_id)
            
            # Get group memberships
            group_memberships = self.group_manager.get_user_groups(user_id)
            
            # Get effective permissions
            effective_permissions = self.group_manager.get_user_effective_permissions(user_id)
            
            # Get recent audit events
            recent_events = self.audit_logger.search_logs(
                user_id=user_id,
                limit=10
            )
            
            return {
                "user_id": user_id,
                "user_context": user_context.to_dict(),
                "group_memberships": [membership.to_dict() for membership in group_memberships],
                "effective_permissions": list(effective_permissions),
                "recent_audit_events": [event.to_dict() for event in recent_events],
                "summary_generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user access summary: {e}")
            return {"error": str(e)}
    
    def validate_system_security(self) -> Dict[str, Any]:
        """
        Validate overall system security configuration.
        
        Returns:
            Security validation results
        """
        try:
            results = {
                "overall_status": "unknown",
                "components": {},
                "recommendations": [],
                "critical_issues": [],
                "warnings": []
            }
            
            # Validate Milvus RBAC
            milvus_stats = self.milvus_rbac.get_rbac_statistics()
            results["components"]["milvus_rbac"] = {
                "status": "healthy" if milvus_stats.get("total_resources", 0) > 0 else "warning",
                "statistics": milvus_stats
            }
            
            # Validate group management
            group_stats = self.group_manager.get_group_statistics()
            results["components"]["group_management"] = {
                "status": "healthy" if group_stats.get("total_groups", 0) > 0 else "warning",
                "statistics": group_stats
            }
            
            # Validate permission inheritance
            inheritance_stats = self.permission_inheritance.get_hierarchy_statistics()
            results["components"]["permission_inheritance"] = {
                "status": "healthy" if inheritance_stats.get("total_resources", 0) > 0 else "warning",
                "statistics": inheritance_stats
            }
            
            # Validate audit logging
            audit_health = self.audit_logger.get_system_health()
            results["components"]["audit_logging"] = {
                "status": "healthy" if audit_health.get("total_logs", 0) > 0 else "warning",
                "health": audit_health
            }
            
            # Overall status
            component_statuses = [comp["status"] for comp in results["components"].values()]
            if all(status == "healthy" for status in component_statuses):
                results["overall_status"] = "healthy"
            elif "error" in component_statuses:
                results["overall_status"] = "error"
            else:
                results["overall_status"] = "warning"
            
            # Generate recommendations
            if results["overall_status"] != "healthy":
                results["recommendations"].append("Review component configurations")
                results["recommendations"].append("Ensure proper initialization of all components")
            
            return results
            
        except Exception as e:
            self.logger.error(f"System security validation failed: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "recommendations": ["Check system logs for detailed error information"]
            }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            return {
                "milvus_rbac": self.milvus_rbac.get_rbac_statistics(),
                "metadata_filter": self.metadata_filter.get_filter_statistics(),
                "permission_inheritance": self.permission_inheritance.get_hierarchy_statistics(),
                "group_management": self.group_manager.get_group_statistics(),
                "audit_logging": self.audit_logger.get_system_health(),
                "configuration": {
                    "strict_mode": self.enable_strict_mode,
                    "audit_logging": self.enable_audit_logging,
                    "default_deny": self.default_deny
                },
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system statistics: {e}")
            return {"error": str(e)}
    
    def configure_system(self, config: Dict[str, Any]) -> None:
        """
        Configure access control system.
        
        Args:
            config: Configuration dictionary
        """
        try:
            if "strict_mode" in config:
                self.enable_strict_mode = config["strict_mode"]
            
            if "audit_logging" in config:
                self.enable_audit_logging = config["audit_logging"]
            
            if "default_deny" in config:
                self.default_deny = config["default_deny"]
            
            # Log configuration change
            if self.enable_audit_logging:
                self.audit_logger.log_event(
                    event_type=AuditEventType.CONFIGURATION_CHANGED,
                    message="Access control system configuration updated",
                    context=AuditContext(additional_data=config),
                    severity=AuditSeverity.MEDIUM,
                    result=AuditResult.SUCCESS,
                    metadata=config
                )
            
            self.logger.info("Access control system configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure system: {e}")
            raise AccessControlError(f"System configuration failed: {e}")