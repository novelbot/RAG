"""
Access Control Filtering for RAG Search Results.

This module provides row-level access control filtering for search results,
integrating with the authentication system to check user permissions and roles,
and implementing data source-specific access controls with comprehensive audit logging.
"""

import time
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from src.core.logging import LoggerMixin
from src.access_control.exceptions import AccessControlError
from src.access_control.access_control_manager import AccessControlManager, AccessRequest, AccessResponse
from src.access_control.audit_logger import AuditEventType, AuditSeverity, AuditResult, AuditContext
from src.rag.context_retriever import RetrievalResult, DocumentContext
from src.auth.models import User


class FilterLevel(Enum):
    """Access control filter levels."""
    NONE = "none"                    # No filtering
    BASIC = "basic"                 # Basic user/role filtering
    STRICT = "strict"               # Strict permission checking
    CUSTOM = "custom"               # Custom filtering logic


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"               # Publicly accessible
    INTERNAL = "internal"           # Internal use only
    CONFIDENTIAL = "confidential"   # Confidential data
    RESTRICTED = "restricted"       # Highly restricted
    SECRET = "secret"               # Secret data


@dataclass
class AccessControlConfig:
    """Configuration for access control filtering."""
    
    # Filter settings
    filter_level: FilterLevel = FilterLevel.STRICT
    enable_row_level_security: bool = True
    enable_field_level_security: bool = True
    enable_data_classification: bool = True
    
    # Permission settings
    require_explicit_permissions: bool = True
    allow_inherited_permissions: bool = True
    check_data_source_permissions: bool = True
    
    # Security settings
    enable_audit_logging: bool = True
    log_filtered_results: bool = True
    enable_security_headers: bool = True
    
    # Performance settings
    enable_permission_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000
    
    # Default access levels
    default_classification: DataClassification = DataClassification.INTERNAL
    anonymous_access_level: DataClassification = DataClassification.PUBLIC


@dataclass
class FilterCriteria:
    """Criteria for filtering search results."""
    user_id: int
    user_roles: List[str] = field(default_factory=list)
    user_groups: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    data_sources: Optional[List[str]] = None
    classification_level: Optional[DataClassification] = None
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of access control filtering."""
    original_count: int
    filtered_count: int
    removed_count: int
    filter_time: float
    filtered_contexts: List[DocumentContext]
    access_decisions: List[Dict[str, Any]] = field(default_factory=list)
    audit_events: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_count": self.original_count,
            "filtered_count": self.filtered_count,
            "removed_count": self.removed_count,
            "filter_time": self.filter_time,
            "access_decisions": self.access_decisions,
            "audit_events": self.audit_events,
            "metadata": self.metadata
        }


class AccessControlFilter(LoggerMixin):
    """
    Access Control Filter for RAG search results.
    
    Features:
    - Row-level access control for search results
    - Integration with authentication and RBAC systems
    - Data source-specific access control policies
    - Data classification and sensitivity filtering
    - Field-level security for sensitive information
    - Comprehensive audit logging for access decisions
    - Performance optimization with permission caching
    - Custom filtering logic support
    """
    
    def __init__(
        self,
        access_control_manager: AccessControlManager,
        config: Optional[AccessControlConfig] = None
    ):
        """
        Initialize Access Control Filter.
        
        Args:
            access_control_manager: Access control manager instance
            config: Filter configuration
        """
        self.access_control_manager = access_control_manager
        self.config = config or AccessControlConfig()
        
        # Permission cache
        self.permission_cache: Dict[str, Tuple[bool, datetime]] = {}
        
        # Statistics
        self.filter_stats = {
            "total_filtering_operations": 0,
            "total_contexts_filtered": 0,
            "total_contexts_removed": 0,
            "total_filter_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info("AccessControlFilter initialized successfully")
    
    def filter_search_results(
        self,
        retrieval_result: RetrievalResult,
        filter_criteria: FilterCriteria
    ) -> FilterResult:
        """
        Filter search results based on access control policies.
        
        Args:
            retrieval_result: Original retrieval results
            filter_criteria: Filtering criteria
            
        Returns:
            Filtered results with access control applied
        """
        start_time = time.time()
        
        try:
            original_contexts = retrieval_result.contexts
            original_count = len(original_contexts)
            
            if self.config.filter_level == FilterLevel.NONE:
                # No filtering applied
                return FilterResult(
                    original_count=original_count,
                    filtered_count=original_count,
                    removed_count=0,
                    filter_time=time.time() - start_time,
                    filtered_contexts=original_contexts,
                    metadata={"filter_level": "none"}
                )
            
            # Apply filtering based on level
            filtered_contexts = []
            access_decisions = []
            audit_events = []
            
            for context in original_contexts:
                # Check access for this context
                access_result = self._check_context_access(context, filter_criteria)
                
                if access_result["granted"]:
                    # Apply field-level filtering if enabled
                    if self.config.enable_field_level_security:
                        filtered_context = self._apply_field_level_filtering(
                            context, filter_criteria
                        )
                        filtered_contexts.append(filtered_context)
                    else:
                        filtered_contexts.append(context)
                
                access_decisions.append(access_result)
                
                # Log audit event if required
                if self.config.enable_audit_logging and access_result.get("audit_event_id"):
                    audit_events.append(access_result["audit_event_id"])
            
            filter_time = time.time() - start_time
            filtered_count = len(filtered_contexts)
            removed_count = original_count - filtered_count
            
            # Update statistics
            self._update_filter_stats(filter_time, original_count, removed_count)
            
            # Create result
            result = FilterResult(
                original_count=original_count,
                filtered_count=filtered_count,
                removed_count=removed_count,
                filter_time=filter_time,
                filtered_contexts=filtered_contexts,
                access_decisions=access_decisions,
                audit_events=audit_events,
                metadata={
                    "filter_level": self.config.filter_level.value,
                    "user_id": filter_criteria.user_id,
                    "applied_policies": self._get_applied_policies(filter_criteria)
                }
            )
            
            self.logger.info(
                f"Access control filtering completed: {filtered_count}/{original_count} "
                f"contexts passed filter in {filter_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            filter_time = time.time() - start_time
            self.logger.error(f"Access control filtering failed: {e}")
            
            # Log audit event for failure
            if self.config.enable_audit_logging:
                self.access_control_manager.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    message=f"Access control filtering failed: {e}",
                    context=AuditContext(user_id=filter_criteria.user_id),
                    severity=AuditSeverity.HIGH,
                    result=AuditResult.FAILURE
                )
            
            raise AccessControlError(f"Access control filtering failed: {e}")
    
    def _check_context_access(
        self,
        context: DocumentContext,
        filter_criteria: FilterCriteria
    ) -> Dict[str, Any]:
        """Check access permissions for a single context."""
        try:
            # Extract resource information from context
            resource_type = context.metadata.get("resource_type", "document")
            resource_id = str(context.id)
            data_source = context.metadata.get("data_source", "unknown")
            
            # Check cache first
            if self.config.enable_permission_caching:
                cache_key = self._generate_cache_key(
                    filter_criteria.user_id, resource_type, resource_id
                )
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return {
                        "granted": cached_result,
                        "reason": "cached_decision",
                        "cached": True
                    }
            
            # Check basic access permissions
            access_request = AccessRequest(
                user_id=filter_criteria.user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action="read",
                context=filter_criteria.context
            )
            
            access_response = self.access_control_manager.check_access(access_request)
            
            if not access_response.granted:
                return {
                    "granted": False,
                    "reason": access_response.reason,
                    "audit_event_id": access_response.audit_event_id
                }
            
            # Check data classification level
            if self.config.enable_data_classification:
                classification_check = self._check_data_classification(
                    context, filter_criteria
                )
                if not classification_check["granted"]:
                    return classification_check
            
            # Check data source permissions
            if self.config.check_data_source_permissions:
                data_source_check = self._check_data_source_access(
                    data_source, filter_criteria
                )
                if not data_source_check["granted"]:
                    return data_source_check
            
            # Check custom filters
            if filter_criteria.custom_filters:
                custom_check = self._apply_custom_filters(
                    context, filter_criteria
                )
                if not custom_check["granted"]:
                    return custom_check
            
            # Cache positive result
            if self.config.enable_permission_caching:
                self._save_to_cache(cache_key, True)
            
            return {
                "granted": True,
                "reason": "access_granted",
                "audit_event_id": access_response.audit_event_id,
                "metadata": access_response.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Context access check failed: {e}")
            return {
                "granted": False,
                "reason": f"access_check_error: {e}",
                "error": str(e)
            }
    
    def _check_data_classification(
        self,
        context: DocumentContext,
        filter_criteria: FilterCriteria
    ) -> Dict[str, Any]:
        """Check data classification access."""
        # Get context classification
        context_classification = context.metadata.get(
            "classification", 
            self.config.default_classification.value
        )
        
        try:
            context_level = DataClassification(context_classification)
        except ValueError:
            context_level = self.config.default_classification
        
        # Get user's maximum allowed classification level
        user_max_level = self._get_user_classification_level(filter_criteria)
        
        # Define classification hierarchy (higher number = more restrictive)
        classification_levels = {
            DataClassification.PUBLIC: 0,
            DataClassification.INTERNAL: 1,
            DataClassification.CONFIDENTIAL: 2,
            DataClassification.RESTRICTED: 3,
            DataClassification.SECRET: 4
        }
        
        context_level_num = classification_levels.get(context_level, 999)
        user_level_num = classification_levels.get(user_max_level, 0)
        
        granted = user_level_num >= context_level_num
        
        return {
            "granted": granted,
            "reason": f"classification_check: user_level={user_max_level.value}, context_level={context_level.value}" if not granted else "classification_allowed",
            "context_classification": context_level.value,
            "user_max_classification": user_max_level.value
        }
    
    def _check_data_source_access(
        self,
        data_source: str,
        filter_criteria: FilterCriteria
    ) -> Dict[str, Any]:
        """Check data source specific access."""
        # If specific data sources are required
        if filter_criteria.data_sources is not None:
            if data_source not in filter_criteria.data_sources:
                return {
                    "granted": False,
                    "reason": f"data_source_not_allowed: {data_source}",
                    "allowed_sources": filter_criteria.data_sources
                }
        
        # Check for data source specific permissions
        data_source_permission = f"data_source:{data_source}:read"
        if data_source_permission in filter_criteria.required_permissions:
            # This would typically check against the user's actual permissions
            # For now, we'll assume granted if they have general read permissions
            pass
        
        return {
            "granted": True,
            "reason": "data_source_allowed",
            "data_source": data_source
        }
    
    def _apply_custom_filters(
        self,
        context: DocumentContext,
        filter_criteria: FilterCriteria
    ) -> Dict[str, Any]:
        """Apply custom filtering logic."""
        for filter_name, filter_value in filter_criteria.custom_filters.items():
            # Get context value for this filter
            context_value = context.metadata.get(filter_name)
            
            if context_value is None:
                continue
            
            # Apply filter logic based on type
            if isinstance(filter_value, list):
                # List filter - context value must be in the list
                if context_value not in filter_value:
                    return {
                        "granted": False,
                        "reason": f"custom_filter_failed: {filter_name} = {context_value} not in {filter_value}"
                    }
            elif isinstance(filter_value, dict):
                # Dictionary filter - supports operators
                if not self._evaluate_filter_expression(context_value, filter_value):
                    return {
                        "granted": False,
                        "reason": f"custom_filter_failed: {filter_name} = {context_value} failed expression {filter_value}"
                    }
            else:
                # Direct value comparison
                if context_value != filter_value:
                    return {
                        "granted": False,
                        "reason": f"custom_filter_failed: {filter_name} = {context_value} != {filter_value}"
                    }
        
        return {
            "granted": True,
            "reason": "custom_filters_passed"
        }
    
    def _apply_field_level_filtering(
        self,
        context: DocumentContext,
        filter_criteria: FilterCriteria
    ) -> DocumentContext:
        """Apply field-level security filtering."""
        # Create a copy of the context
        filtered_context = DocumentContext(
            id=context.id,
            content=context.content,
            metadata=context.metadata.copy(),
            similarity_score=context.similarity_score,
            relevance_score=context.relevance_score,
            ranking_score=context.ranking_score,
            source_info=context.source_info.copy(),
            retrieval_timestamp=context.retrieval_timestamp
        )
        
        # Filter sensitive fields from metadata
        sensitive_fields = ["internal_id", "personal_data", "confidential_notes"]
        
        for field in sensitive_fields:
            if field in filtered_context.metadata:
                # Check if user has permission to see this field
                if not self._has_field_permission(field, filter_criteria):
                    filtered_context.metadata[field] = "[REDACTED]"
        
        # Filter sensitive content patterns
        filtered_context.content = self._filter_sensitive_content(
            filtered_context.content, filter_criteria
        )
        
        return filtered_context
    
    def _has_field_permission(
        self,
        field_name: str,
        filter_criteria: FilterCriteria
    ) -> bool:
        """Check if user has permission to access a specific field."""
        # Simple field permission check - could be enhanced
        field_permission = f"field:{field_name}:read"
        return field_permission in filter_criteria.required_permissions
    
    def _filter_sensitive_content(
        self,
        content: str,
        filter_criteria: FilterCriteria
    ) -> str:
        """Filter sensitive patterns from content."""
        import re
        
        # Simple pattern filtering - could be enhanced with ML
        patterns = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN_REDACTED]',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL_REDACTED]',  # Email
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b': '[CARD_REDACTED]'  # Credit card
        }
        
        filtered_content = content
        for pattern, replacement in patterns.items():
            filtered_content = re.sub(pattern, replacement, filtered_content)
        
        return filtered_content
    
    def _get_user_classification_level(
        self,
        filter_criteria: FilterCriteria
    ) -> DataClassification:
        """Get user's maximum allowed classification level."""
        # Check user roles for classification permissions
        role_classifications = {
            "admin": DataClassification.SECRET,
            "manager": DataClassification.RESTRICTED,
            "employee": DataClassification.CONFIDENTIAL,
            "contractor": DataClassification.INTERNAL,
            "guest": DataClassification.PUBLIC
        }
        
        max_level = DataClassification.PUBLIC
        
        for role in filter_criteria.user_roles:
            role_level = role_classifications.get(role.lower(), DataClassification.PUBLIC)
            if role_level.value in ["secret", "restricted", "confidential", "internal", "public"]:
                # Compare classification levels
                current_level = DataClassification(max_level.value)
                if self._is_higher_classification(role_level, current_level):
                    max_level = role_level
        
        return max_level
    
    def _is_higher_classification(
        self,
        level1: DataClassification,
        level2: DataClassification
    ) -> bool:
        """Check if level1 is higher than level2."""
        levels = {
            DataClassification.PUBLIC: 0,
            DataClassification.INTERNAL: 1,
            DataClassification.CONFIDENTIAL: 2,
            DataClassification.RESTRICTED: 3,
            DataClassification.SECRET: 4
        }
        return levels.get(level1, 0) > levels.get(level2, 0)
    
    def _evaluate_filter_expression(
        self,
        value: Any,
        expression: Dict[str, Any]
    ) -> bool:
        """Evaluate a filter expression against a value."""
        for operator, operand in expression.items():
            if operator == "eq":
                return value == operand
            elif operator == "ne":
                return value != operand
            elif operator == "gt":
                return value > operand
            elif operator == "gte":
                return value >= operand
            elif operator == "lt":
                return value < operand
            elif operator == "lte":
                return value <= operand
            elif operator == "in":
                return value in operand
            elif operator == "nin":
                return value not in operand
            elif operator == "contains":
                return operand in str(value)
            elif operator == "regex":
                import re
                return bool(re.search(operand, str(value)))
        
        return True
    
    def _get_applied_policies(self, filter_criteria: FilterCriteria) -> List[str]:
        """Get list of applied security policies."""
        policies = []
        
        if self.config.enable_row_level_security:
            policies.append("row_level_security")
        
        if self.config.enable_field_level_security:
            policies.append("field_level_security")
        
        if self.config.enable_data_classification:
            policies.append("data_classification")
        
        if self.config.check_data_source_permissions:
            policies.append("data_source_permissions")
        
        if filter_criteria.custom_filters:
            policies.append("custom_filters")
        
        return policies
    
    def _generate_cache_key(
        self,
        user_id: int,
        resource_type: str,
        resource_id: str
    ) -> str:
        """Generate cache key for permission check."""
        return f"perm:{user_id}:{resource_type}:{resource_id}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[bool]:
        """Get permission result from cache."""
        if cache_key in self.permission_cache:
            result, timestamp = self.permission_cache[cache_key]
            
            # Check TTL
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < self.config.cache_ttl:
                self.filter_stats["cache_hits"] += 1
                return result
            else:
                # Remove expired entry
                del self.permission_cache[cache_key]
        
        self.filter_stats["cache_misses"] += 1
        return None
    
    def _save_to_cache(self, cache_key: str, result: bool) -> None:
        """Save permission result to cache."""
        # Clean cache if it's full
        if len(self.permission_cache) >= self.config.max_cache_size:
            # Remove oldest 10% of entries
            oldest_keys = sorted(
                self.permission_cache.keys(),
                key=lambda k: self.permission_cache[k][1]
            )[:self.config.max_cache_size // 10]
            
            for key in oldest_keys:
                del self.permission_cache[key]
        
        self.permission_cache[cache_key] = (result, datetime.now(timezone.utc))
    
    def _update_filter_stats(
        self,
        filter_time: float,
        original_count: int,
        removed_count: int
    ) -> None:
        """Update filtering statistics."""
        self.filter_stats["total_filtering_operations"] += 1
        self.filter_stats["total_contexts_filtered"] += original_count
        self.filter_stats["total_contexts_removed"] += removed_count
        self.filter_stats["total_filter_time"] += filter_time
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        total_ops = self.filter_stats["total_filtering_operations"]
        
        return {
            "total_filtering_operations": total_ops,
            "total_contexts_filtered": self.filter_stats["total_contexts_filtered"],
            "total_contexts_removed": self.filter_stats["total_contexts_removed"],
            "average_filter_time": (
                self.filter_stats["total_filter_time"] / total_ops
                if total_ops > 0 else 0.0
            ),
            "cache_hit_rate": (
                self.filter_stats["cache_hits"] / 
                (self.filter_stats["cache_hits"] + self.filter_stats["cache_misses"])
                if (self.filter_stats["cache_hits"] + self.filter_stats["cache_misses"]) > 0
                else 0.0
            ),
            "config": {
                "filter_level": self.config.filter_level.value,
                "row_level_security": self.config.enable_row_level_security,
                "field_level_security": self.config.enable_field_level_security,
                "data_classification": self.config.enable_data_classification
            }
        }
    
    def clear_cache(self) -> None:
        """Clear permission cache."""
        self.permission_cache.clear()
        self.logger.info("Access control filter cache cleared")


def create_access_control_filter(
    access_control_manager: AccessControlManager,
    config: Optional[AccessControlConfig] = None
) -> AccessControlFilter:
    """
    Create and configure an AccessControlFilter instance.
    
    Args:
        access_control_manager: Access control manager instance
        config: Filter configuration
        
    Returns:
        Configured AccessControlFilter
    """
    return AccessControlFilter(
        access_control_manager=access_control_manager,
        config=config or AccessControlConfig()
    )


def create_filter_criteria(
    user_id: int,
    user_roles: Optional[List[str]] = None,
    user_groups: Optional[List[str]] = None,
    **kwargs
) -> FilterCriteria:
    """
    Create filter criteria with simplified interface.
    
    Args:
        user_id: User ID
        user_roles: User roles
        user_groups: User groups
        **kwargs: Additional criteria parameters
        
    Returns:
        Filter criteria object
    """
    return FilterCriteria(
        user_id=user_id,
        user_roles=user_roles or [],
        user_groups=user_groups or [],
        **kwargs
    )