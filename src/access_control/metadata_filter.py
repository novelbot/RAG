"""
Metadata-Based Filtering System.

This module provides metadata-based filtering for search results based on
user permissions, content attributes, and hierarchical access control.
"""

import json
from typing import Dict, List, Any, Optional, Set, Union, Callable
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field

from src.core.logging import LoggerMixin
from .exceptions import MetadataFilterError, InsufficientPermissionsError


class FilterOperation(Enum):
    """Filter operation types."""
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class FilterLogic(Enum):
    """Filter logic operators."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class FilterCondition:
    """Individual filter condition."""
    field: str
    operation: FilterOperation
    value: Any = None
    case_sensitive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field": self.field,
            "operation": self.operation.value,
            "value": self.value,
            "case_sensitive": self.case_sensitive
        }


@dataclass
class FilterGroup:
    """Group of filter conditions with logic operator."""
    conditions: List[Union[FilterCondition, 'FilterGroup']]
    logic: FilterLogic = FilterLogic.AND
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "logic": self.logic.value,
            "conditions": [
                cond.to_dict() if isinstance(cond, FilterCondition) else cond.to_dict()
                for cond in self.conditions
            ]
        }


@dataclass
class FilterResult:
    """Result of filtering operation."""
    passed: bool
    filtered_data: Any = None
    filter_reason: Optional[str] = None
    matched_conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "passed": self.passed,
            "filter_reason": self.filter_reason,
            "matched_conditions": self.matched_conditions
        }


class MetadataFilter(LoggerMixin):
    """
    Metadata-based filtering system for access control.
    
    Provides flexible filtering based on user permissions, content attributes,
    and hierarchical access control rules.
    """
    
    def __init__(self):
        """Initialize metadata filter."""
        self.custom_filters: Dict[str, Callable] = {}
        self.field_transformers: Dict[str, Callable] = {}
        self.permission_resolvers: Dict[str, Callable] = {}
        
        # Register default field transformers
        self._register_default_transformers()
        
        self.logger.info("Metadata filter initialized successfully")
    
    def _register_default_transformers(self) -> None:
        """Register default field transformers."""
        def normalize_string(value: str) -> str:
            """Normalize string for comparison."""
            return value.lower().strip() if isinstance(value, str) else str(value)
        
        def parse_timestamp(value: Union[str, datetime]) -> datetime:
            """Parse timestamp from various formats."""
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return datetime.fromtimestamp(value)
        
        def parse_json_array(value: Union[str, List]) -> List:
            """Parse JSON array from string."""
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.split(",")
            return [value]
        
        self.field_transformers.update({
            "string": normalize_string,
            "timestamp": parse_timestamp,
            "json_array": parse_json_array
        })
    
    def register_custom_filter(self, name: str, filter_func: Callable) -> None:
        """
        Register custom filter function.
        
        Args:
            name: Filter name
            filter_func: Filter function that takes (data, criteria) and returns bool
        """
        self.custom_filters[name] = filter_func
        self.logger.debug(f"Registered custom filter: {name}")
    
    def register_field_transformer(self, field_type: str, transformer_func: Callable) -> None:
        """
        Register field transformer function.
        
        Args:
            field_type: Field type name
            transformer_func: Transformer function
        """
        self.field_transformers[field_type] = transformer_func
        self.logger.debug(f"Registered field transformer: {field_type}")
    
    def register_permission_resolver(self, permission: str, resolver_func: Callable) -> None:
        """
        Register permission resolver function.
        
        Args:
            permission: Permission name
            resolver_func: Resolver function that takes (user_context, metadata) and returns bool
        """
        self.permission_resolvers[permission] = resolver_func
        self.logger.debug(f"Registered permission resolver: {permission}")
    
    def create_user_permission_filter(self, user_id: int, permissions: List[str]) -> FilterGroup:
        """
        Create filter group for user permissions.
        
        Args:
            user_id: User ID
            permissions: List of user permissions
            
        Returns:
            Filter group for user permissions
        """
        conditions = []
        
        # Owner access
        conditions.append(FilterCondition(
            field="user_id",
            operation=FilterOperation.EQUALS,
            value=user_id
        ))
        
        # Public access
        conditions.append(FilterCondition(
            field="scope",
            operation=FilterOperation.EQUALS,
            value="public"
        ))
        
        # Permission-based access
        for permission in permissions:
            conditions.append(FilterCondition(
                field="required_permissions",
                operation=FilterOperation.CONTAINS,
                value=permission
            ))
        
        return FilterGroup(conditions=conditions, logic=FilterLogic.OR)
    
    def create_group_access_filter(self, group_ids: List[str]) -> FilterGroup:
        """
        Create filter group for group access.
        
        Args:
            group_ids: List of user's group IDs
            
        Returns:
            Filter group for group access
        """
        conditions = []
        
        for group_id in group_ids:
            conditions.append(FilterCondition(
                field="group_ids",
                operation=FilterOperation.CONTAINS,
                value=group_id
            ))
        
        return FilterGroup(conditions=conditions, logic=FilterLogic.OR)
    
    def create_content_classification_filter(self, 
                                           user_clearance: str,
                                           allowed_classifications: List[str]) -> FilterGroup:
        """
        Create filter for content classification.
        
        Args:
            user_clearance: User's security clearance level
            allowed_classifications: List of allowed content classifications
            
        Returns:
            Filter group for content classification
        """
        conditions = []
        
        # Classification level filter
        conditions.append(FilterCondition(
            field="classification",
            operation=FilterOperation.IN,
            value=allowed_classifications
        ))
        
        # Clearance level filter
        clearance_levels = ["public", "internal", "confidential", "restricted", "secret"]
        user_level_index = clearance_levels.index(user_clearance) if user_clearance in clearance_levels else 0
        allowed_levels = clearance_levels[:user_level_index + 1]
        
        conditions.append(FilterCondition(
            field="clearance_level",
            operation=FilterOperation.IN,
            value=allowed_levels
        ))
        
        return FilterGroup(conditions=conditions, logic=FilterLogic.AND)
    
    def create_temporal_filter(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             access_window: Optional[int] = None) -> FilterGroup:
        """
        Create temporal access filter.
        
        Args:
            start_date: Start date for access window
            end_date: End date for access window
            access_window: Access window in days from now
            
        Returns:
            Filter group for temporal access
        """
        conditions = []
        
        if start_date:
            conditions.append(FilterCondition(
                field="created_at",
                operation=FilterOperation.GREATER_THAN_EQUAL,
                value=start_date
            ))
        
        if end_date:
            conditions.append(FilterCondition(
                field="created_at",
                operation=FilterOperation.LESS_THAN_EQUAL,
                value=end_date
            ))
        
        if access_window:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=access_window)
            conditions.append(FilterCondition(
                field="last_accessed",
                operation=FilterOperation.GREATER_THAN_EQUAL,
                value=cutoff_date
            ))
        
        return FilterGroup(conditions=conditions, logic=FilterLogic.AND)
    
    def evaluate_condition(self, data: Dict[str, Any], condition: FilterCondition) -> bool:
        """
        Evaluate single filter condition.
        
        Args:
            data: Data to evaluate
            condition: Filter condition
            
        Returns:
            True if condition is met
        """
        try:
            # Get field value
            field_value = self._get_nested_value(data, condition.field)
            
            # Apply field transformer if available
            if condition.field in self.field_transformers:
                field_value = self.field_transformers[condition.field](field_value)
            
            # Handle null checks
            if condition.operation == FilterOperation.IS_NULL:
                return field_value is None
            elif condition.operation == FilterOperation.IS_NOT_NULL:
                return field_value is not None
            
            # Skip evaluation if field is None and operation requires value
            if field_value is None:
                return False
            
            # Evaluate operation
            return self._evaluate_operation(field_value, condition.operation, condition.value, condition.case_sensitive)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition.field}: {e}")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        try:
            keys = field_path.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and key.isdigit():
                    index = int(key)
                    value = value[index] if 0 <= index < len(value) else None
                else:
                    return None
            
            return value
            
        except (KeyError, IndexError, TypeError):
            return None
    
    def _evaluate_operation(self, field_value: Any, operation: FilterOperation, 
                          compare_value: Any, case_sensitive: bool) -> bool:
        """Evaluate comparison operation."""
        try:
            # String operations
            if isinstance(field_value, str) and isinstance(compare_value, str):
                if not case_sensitive:
                    field_value = field_value.lower()
                    compare_value = compare_value.lower()
            
            # Equality operations
            if operation == FilterOperation.EQUALS:
                return field_value == compare_value
            elif operation == FilterOperation.NOT_EQUALS:
                return field_value != compare_value
            
            # Comparison operations
            elif operation == FilterOperation.GREATER_THAN:
                return field_value > compare_value
            elif operation == FilterOperation.GREATER_THAN_EQUAL:
                return field_value >= compare_value
            elif operation == FilterOperation.LESS_THAN:
                return field_value < compare_value
            elif operation == FilterOperation.LESS_THAN_EQUAL:
                return field_value <= compare_value
            
            # Membership operations
            elif operation == FilterOperation.IN:
                return field_value in compare_value
            elif operation == FilterOperation.NOT_IN:
                return field_value not in compare_value
            
            # String operations
            elif operation == FilterOperation.CONTAINS:
                return compare_value in field_value
            elif operation == FilterOperation.NOT_CONTAINS:
                return compare_value not in field_value
            elif operation == FilterOperation.STARTS_WITH:
                return field_value.startswith(compare_value)
            elif operation == FilterOperation.ENDS_WITH:
                return field_value.endswith(compare_value)
            
            # Regex operation
            elif operation == FilterOperation.REGEX:
                import re
                pattern = re.compile(compare_value, re.IGNORECASE if not case_sensitive else 0)
                return bool(pattern.search(field_value))
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating operation {operation.value}: {e}")
            return False
    
    def evaluate_filter_group(self, data: Dict[str, Any], filter_group: FilterGroup) -> bool:
        """
        Evaluate filter group.
        
        Args:
            data: Data to evaluate
            filter_group: Filter group
            
        Returns:
            True if filter group passes
        """
        try:
            results = []
            
            for condition in filter_group.conditions:
                if isinstance(condition, FilterCondition):
                    result = self.evaluate_condition(data, condition)
                elif isinstance(condition, FilterGroup):
                    result = self.evaluate_filter_group(data, condition)
                else:
                    result = False
                
                results.append(result)
            
            # Apply logic operator
            if filter_group.logic == FilterLogic.AND:
                return all(results)
            elif filter_group.logic == FilterLogic.OR:
                return any(results)
            elif filter_group.logic == FilterLogic.NOT:
                return not all(results)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating filter group: {e}")
            return False
    
    def filter_search_results(self, 
                            results: List[Dict[str, Any]],
                            filter_groups: List[FilterGroup],
                            user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Filter search results based on metadata.
        
        Args:
            results: List of search results
            filter_groups: List of filter groups to apply
            user_context: User context for permission-based filtering
            
        Returns:
            Filtered search results
        """
        try:
            filtered_results = []
            
            for result in results:
                # Check all filter groups
                passed = True
                for filter_group in filter_groups:
                    if not self.evaluate_filter_group(result, filter_group):
                        passed = False
                        break
                
                # Apply custom permission resolvers
                if passed and user_context:
                    for permission, resolver in self.permission_resolvers.items():
                        if not resolver(user_context, result):
                            passed = False
                            break
                
                if passed:
                    filtered_results.append(result)
            
            self.logger.debug(f"Filtered {len(results)} results down to {len(filtered_results)}")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error filtering search results: {e}")
            raise MetadataFilterError(f"Result filtering failed: {e}")
    
    def create_milvus_expression(self, filter_groups: List[FilterGroup]) -> str:
        """
        Create Milvus boolean expression from filter groups.
        
        Args:
            filter_groups: List of filter groups
            
        Returns:
            Milvus boolean expression string
        """
        try:
            expressions = []
            
            for filter_group in filter_groups:
                expr = self._convert_filter_group_to_expression(filter_group)
                if expr:
                    expressions.append(f"({expr})")
            
            if not expressions:
                return "1 == 1"  # Allow all if no filters
            
            return " AND ".join(expressions)
            
        except Exception as e:
            self.logger.error(f"Error creating Milvus expression: {e}")
            raise MetadataFilterError(f"Expression creation failed: {e}")
    
    def _convert_filter_group_to_expression(self, filter_group: FilterGroup) -> str:
        """Convert filter group to Milvus expression."""
        try:
            expressions = []
            
            for condition in filter_group.conditions:
                if isinstance(condition, FilterCondition):
                    expr = self._convert_condition_to_expression(condition)
                    if expr:
                        expressions.append(expr)
                elif isinstance(condition, FilterGroup):
                    expr = self._convert_filter_group_to_expression(condition)
                    if expr:
                        expressions.append(f"({expr})")
            
            if not expressions:
                return ""
            
            # Join with logic operator
            if filter_group.logic == FilterLogic.AND:
                return " AND ".join(expressions)
            elif filter_group.logic == FilterLogic.OR:
                return " OR ".join(expressions)
            elif filter_group.logic == FilterLogic.NOT:
                return f"NOT ({' AND '.join(expressions)})"
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error converting filter group to expression: {e}")
            return ""
    
    def _convert_condition_to_expression(self, condition: FilterCondition) -> str:
        """Convert filter condition to Milvus expression."""
        try:
            field = condition.field
            operation = condition.operation
            value = condition.value
            
            # Handle string values
            if isinstance(value, str):
                value = f'"{value}"'
            elif isinstance(value, list):
                formatted_values = []
                for v in value:
                    if isinstance(v, str):
                        formatted_values.append(f'"{v}"')
                    else:
                        formatted_values.append(str(v))
                value = f"[{', '.join(formatted_values)}]"
            
            # Convert operations
            if operation == FilterOperation.EQUALS:
                return f"{field} == {value}"
            elif operation == FilterOperation.NOT_EQUALS:
                return f"{field} != {value}"
            elif operation == FilterOperation.GREATER_THAN:
                return f"{field} > {value}"
            elif operation == FilterOperation.GREATER_THAN_EQUAL:
                return f"{field} >= {value}"
            elif operation == FilterOperation.LESS_THAN:
                return f"{field} < {value}"
            elif operation == FilterOperation.LESS_THAN_EQUAL:
                return f"{field} <= {value}"
            elif operation == FilterOperation.IN:
                return f"{field} in {value}"
            elif operation == FilterOperation.NOT_IN:
                return f"{field} not in {value}"
            elif operation == FilterOperation.CONTAINS:
                return f'JSON_CONTAINS({field}, {value})'
            elif operation == FilterOperation.IS_NULL:
                return f"{field} == null"
            elif operation == FilterOperation.IS_NOT_NULL:
                return f"{field} != null"
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error converting condition to expression: {e}")
            return ""
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filter system statistics."""
        return {
            "custom_filters": len(self.custom_filters),
            "field_transformers": len(self.field_transformers),
            "permission_resolvers": len(self.permission_resolvers),
            "registered_filters": list(self.custom_filters.keys()),
            "registered_transformers": list(self.field_transformers.keys()),
            "registered_resolvers": list(self.permission_resolvers.keys())
        }