"""
Authentication and Authorization System for RAG Server.

This module provides comprehensive authentication and authorization capabilities
including JWT token management, role-based access control (RBAC), and user management.
"""

# Temporarily commenting out complex imports to avoid dependency issues
# from .jwt_manager import JWTManager
# from .models import User, Role, Permission, UserRole, RolePermission
# from .rbac import RBACManager
# from .middleware import AuthMiddleware
# from .schemas import (
#     UserCreate, UserUpdate, UserResponse,
#     RoleCreate, RoleUpdate, RoleResponse,
#     PermissionCreate, PermissionUpdate, PermissionResponse,
#     TokenResponse, LoginRequest, RefreshTokenRequest
# )
from .exceptions import (
    AuthenticationError, AuthorizationError, TokenExpiredError,
    InvalidTokenError, InsufficientPermissionsError
)

__all__ = [
    # Core managers
    "JWTManager",
    "RBACManager",
    "AuthMiddleware",
    
    # Models
    "User", "Role", "Permission", "UserRole", "RolePermission",
    
    # Schemas
    "UserCreate", "UserUpdate", "UserResponse",
    "RoleCreate", "RoleUpdate", "RoleResponse", 
    "PermissionCreate", "PermissionUpdate", "PermissionResponse",
    "TokenResponse", "LoginRequest", "RefreshTokenRequest",
    
    # Exceptions
    "AuthenticationError", "AuthorizationError", "TokenExpiredError",
    "InvalidTokenError", "InsufficientPermissionsError"
]