"""
JWT Token Management System.

This module provides comprehensive JWT token generation, validation, and management
with support for access tokens, refresh tokens, and token blacklisting.
"""

import jwt
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

from src.core.config import get_config
from src.core.logging import LoggerMixin
from .exceptions import (
    TokenExpiredError, InvalidTokenError, TokenBlacklistedError
)


@dataclass
class TokenPayload:
    """Token payload structure."""
    
    user_id: int
    username: str
    email: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    token_type: str = "access"  # access or refresh
    jti: Optional[str] = None  # JWT ID for blacklisting
    iat: Optional[int] = None  # issued at
    exp: Optional[int] = None  # expiration time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "token_type": self.token_type,
            "jti": self.jti,
            "iat": self.iat,
            "exp": self.exp
        }


@dataclass
class TokenResponse:
    """Token response structure."""
    
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in
        }


class JWTManager(LoggerMixin):
    """
    JWT Token Manager for handling authentication tokens.
    
    Provides secure token generation, validation, and management with support
    for access tokens, refresh tokens, and token blacklisting.
    """
    
    def __init__(self):
        """Initialize JWT Manager."""
        self.settings = get_config()
        
        # JWT Configuration
        self.secret_key = getattr(self.settings, 'auth', None) and getattr(self.settings.auth, 'secret_key', None) or 'default-secret-key-change-in-production'
        self.algorithm = getattr(self.settings, 'JWT_ALGORITHM', 'HS256')
        self.access_token_expire_minutes = getattr(self.settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 15)
        self.refresh_token_expire_days = getattr(self.settings, 'REFRESH_TOKEN_EXPIRE_DAYS', 30)
        
        # Token blacklist (in production, use Redis or database)
        self.token_blacklist: set[str] = set()
        
        self.logger.info("JWT Manager initialized successfully")
    
    def generate_access_token(self, user_data: Dict[str, Any]) -> str:
        """
        Generate an access token for a user.
        
        Args:
            user_data: User information including id, username, email, roles, permissions
            
        Returns:
            JWT access token string
        """
        now = datetime.now(timezone.utc)
        expiration = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = TokenPayload(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            token_type="access",
            jti=f"access_{user_data['id']}_{int(time.time())}",
            iat=int(now.timestamp()),
            exp=int(expiration.timestamp())
        )
        
        token = jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm
        )
        
        self.logger.debug(f"Generated access token for user {user_data['username']}")
        return token
    
    def generate_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """
        Generate a refresh token for a user.
        
        Args:
            user_data: User information including id, username, email
            
        Returns:
            JWT refresh token string
        """
        now = datetime.now(timezone.utc)
        expiration = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = TokenPayload(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=[],  # Refresh tokens don't need roles/permissions
            permissions=[],
            token_type="refresh",
            jti=f"refresh_{user_data['id']}_{int(time.time())}",
            iat=int(now.timestamp()),
            exp=int(expiration.timestamp())
        )
        
        token = jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm
        )
        
        self.logger.debug(f"Generated refresh token for user {user_data['username']}")
        return token
    
    def generate_token_pair(self, user_data: Dict[str, Any]) -> TokenResponse:
        """
        Generate both access and refresh tokens for a user.
        
        Args:
            user_data: User information
            
        Returns:
            TokenResponse with access and refresh tokens
        """
        access_token = self.generate_access_token(user_data)
        refresh_token = self.generate_refresh_token(user_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60
        )
    
    def validate_token(self, token: str, token_type: str = "access") -> TokenPayload:
        """
        Validate a JWT token and return its payload.
        
        Args:
            token: JWT token string
            token_type: Expected token type (access or refresh)
            
        Returns:
            TokenPayload with decoded information
            
        Raises:
            InvalidTokenError: If token is invalid or malformed
            TokenExpiredError: If token has expired
            TokenBlacklistedError: If token has been blacklisted
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and jti in self.token_blacklist:
                raise TokenBlacklistedError("Token has been blacklisted")
            
            # Validate token type
            if payload.get("token_type") != token_type:
                raise InvalidTokenError(f"Expected {token_type} token, got {payload.get('token_type')}")
            
            # Create TokenPayload object
            token_payload = TokenPayload(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                token_type=payload["token_type"],
                jti=payload.get("jti"),
                iat=payload.get("iat"),
                exp=payload.get("exp")
            )
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            raise InvalidTokenError(f"Invalid token: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error validating token: {e}")
            raise InvalidTokenError(f"Token validation failed: {e}")
    
    def refresh_access_token(self, refresh_token: str, user_data: Dict[str, Any]) -> str:
        """
        Generate a new access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            user_data: Updated user information
            
        Returns:
            New access token string
            
        Raises:
            InvalidTokenError: If refresh token is invalid
            TokenExpiredError: If refresh token has expired
        """
        # Validate refresh token
        token_payload = self.validate_token(refresh_token, token_type="refresh")
        
        # Verify user matches
        if token_payload.user_id != user_data["id"]:
            raise InvalidTokenError("Refresh token does not match user")
        
        # Generate new access token
        new_access_token = self.generate_access_token(user_data)
        
        self.logger.debug(f"Refreshed access token for user {user_data['username']}")
        return new_access_token
    
    def blacklist_token(self, token: str) -> None:
        """
        Add a token to the blacklist.
        
        Args:
            token: JWT token to blacklist
        """
        try:
            # Decode token to get JTI
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Don't verify expiration for blacklisting
            )
            
            jti = payload.get("jti")
            if jti:
                self.token_blacklist.add(jti)
                self.logger.debug(f"Blacklisted token with JTI: {jti}")
            
        except jwt.InvalidTokenError:
            # If token is invalid, no need to blacklist
            pass
    
    def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token is blacklisted, False otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            return jti in self.token_blacklist if jti else False
            
        except jwt.InvalidTokenError:
            return False
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Get token information without validating expiration.
        
        Args:
            token: JWT token
            
        Returns:
            Dictionary with token information
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            return {
                "user_id": payload.get("user_id"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "token_type": payload.get("token_type"),
                "issued_at": datetime.fromtimestamp(payload["iat"], timezone.utc) if payload.get("iat") else None,
                "expires_at": datetime.fromtimestamp(payload["exp"], timezone.utc) if payload.get("exp") else None,
                "is_expired": payload.get("exp", 0) < time.time(),
                "is_blacklisted": self.is_token_blacklisted(token)
            }
            
        except jwt.InvalidTokenError:
            return {"error": "Invalid token"}
    
    def cleanup_expired_tokens(self) -> None:
        """
        Clean up expired tokens from blacklist.
        
        In production, this would be handled by Redis TTL or database cleanup job.
        """
        # This is a simplified implementation
        # In production, use Redis with TTL or database cleanup
        self.logger.debug("Token cleanup would be handled by Redis TTL or database cleanup")