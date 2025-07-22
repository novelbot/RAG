"""
Authentication and Authorization Schemas.

This module defines Pydantic schemas for API requests and responses
related to authentication, authorization, and user management.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(from_attributes=True)


# User schemas
class UserBase(BaseSchema):
    """Base user schema with common fields."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    is_active: bool = Field(True, description="Whether user is active")
    timezone: Optional[str] = Field("UTC", description="User timezone")
    bio: Optional[str] = Field(None, description="User biography")
    avatar_url: Optional[str] = Field(None, description="Avatar URL")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, description="Password")
    is_superuser: bool = Field(False, description="Whether user is a superuser")
    is_verified: bool = Field(False, description="Whether user is verified")
    
    @validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseSchema):
    """Schema for updating user information."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    is_verified: Optional[bool] = None
    timezone: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    
    @validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if v is not None:
            if len(v) < 8:
                raise ValueError("Password must be at least 8 characters long")
            if not any(c.isupper() for c in v):
                raise ValueError("Password must contain at least one uppercase letter")
            if not any(c.islower() for c in v):
                raise ValueError("Password must contain at least one lowercase letter")
            if not any(c.isdigit() for c in v):
                raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(UserBase):
    """Schema for user response."""
    id: int
    is_superuser: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")


class UserSummary(BaseSchema):
    """Schema for user summary (minimal information)."""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    avatar_url: Optional[str] = None


# Role schemas
class RoleBase(BaseSchema):
    """Base role schema with common fields."""
    name: str = Field(..., min_length=2, max_length=50, description="Role name")
    description: Optional[str] = Field(None, max_length=255, description="Role description")
    is_default: bool = Field(False, description="Whether this is a default role")
    priority: int = Field(0, description="Role priority (higher = more important)")


class RoleCreate(RoleBase):
    """Schema for creating a new role."""
    parent_role_id: Optional[int] = Field(None, description="Parent role ID for hierarchy")
    permission_ids: List[int] = Field(default_factory=list, description="Permission IDs to assign")


class RoleUpdate(BaseSchema):
    """Schema for updating role information."""
    name: Optional[str] = Field(None, min_length=2, max_length=50)
    description: Optional[str] = Field(None, max_length=255)
    is_default: Optional[bool] = None
    priority: Optional[int] = None
    parent_role_id: Optional[int] = None
    permission_ids: Optional[List[int]] = None


class RoleResponse(RoleBase):
    """Schema for role response."""
    id: int
    parent_role_id: Optional[int] = None
    is_system: bool
    created_at: datetime
    updated_at: datetime
    permissions: List[str] = Field(default_factory=list, description="Role permissions")
    user_count: int = Field(0, description="Number of users with this role")


class RoleSummary(BaseSchema):
    """Schema for role summary (minimal information)."""
    id: int
    name: str
    description: Optional[str] = None
    priority: int


# Permission schemas
class PermissionBase(BaseSchema):
    """Base permission schema with common fields."""
    name: str = Field(..., min_length=2, max_length=100, description="Permission name")
    description: Optional[str] = Field(None, max_length=255, description="Permission description")
    resource: str = Field(..., min_length=2, max_length=50, description="Resource type")
    action: str = Field(..., min_length=2, max_length=50, description="Action type")


class PermissionCreate(PermissionBase):
    """Schema for creating a new permission."""
    pass


class PermissionUpdate(BaseSchema):
    """Schema for updating permission information."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=255)
    resource: Optional[str] = Field(None, min_length=2, max_length=50)
    action: Optional[str] = Field(None, min_length=2, max_length=50)


class PermissionResponse(PermissionBase):
    """Schema for permission response."""
    id: int
    is_system: bool
    created_at: datetime
    updated_at: datetime
    role_count: int = Field(0, description="Number of roles with this permission")


class PermissionSummary(BaseSchema):
    """Schema for permission summary (minimal information)."""
    id: int
    name: str
    resource: str
    action: str


# Authentication schemas
class LoginRequest(BaseSchema):
    """Schema for login request."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(False, description="Remember login")


class TokenResponse(BaseSchema):
    """Schema for token response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserSummary


class RefreshTokenRequest(BaseSchema):
    """Schema for refresh token request."""
    refresh_token: str = Field(..., description="Refresh token")


class ChangePasswordRequest(BaseSchema):
    """Schema for password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class ResetPasswordRequest(BaseSchema):
    """Schema for password reset request."""
    email: str = Field(..., description="Email address")


class ResetPasswordConfirm(BaseSchema):
    """Schema for password reset confirmation."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# Role assignment schemas
class UserRoleAssignment(BaseSchema):
    """Schema for user role assignment."""
    user_id: int = Field(..., description="User ID")
    role_id: int = Field(..., description="Role ID")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")


class RolePermissionAssignment(BaseSchema):
    """Schema for role permission assignment."""
    role_id: int = Field(..., description="Role ID")
    permission_id: int = Field(..., description="Permission ID")


# Response schemas
class MessageResponse(BaseSchema):
    """Schema for simple message response."""
    message: str = Field(..., description="Response message")


class ErrorResponse(BaseSchema):
    """Schema for error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")


# Pagination schemas
class PaginationParams(BaseSchema):
    """Schema for pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")
    sort: Optional[str] = Field(None, description="Sort field")
    order: Optional[str] = Field("asc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseSchema):
    """Schema for paginated response."""
    items: List[dict]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool