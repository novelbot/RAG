"""
Access Control Exceptions.

Custom exceptions for access control operations including permissions,
resources, groups, and audit logging.
"""

from src.core.exceptions import BaseCustomException


class AccessControlError(BaseCustomException):
    """Base exception for access control errors."""
    pass


class InsufficientPermissionsError(AccessControlError):
    """Raised when a user lacks required permissions."""
    pass


class ResourceNotFoundError(AccessControlError):
    """Raised when a requested resource is not found."""
    pass


class GroupNotFoundError(AccessControlError):
    """Raised when a requested group is not found."""
    pass


class PermissionInheritanceError(AccessControlError):
    """Raised when permission inheritance fails."""
    pass


class AuditLoggingError(AccessControlError):
    """Raised when audit logging fails."""
    pass


class MetadataFilterError(AccessControlError):
    """Raised when metadata filtering fails."""
    pass


class MilvusRBACError(AccessControlError):
    """Raised when Milvus RBAC operations fail."""
    pass