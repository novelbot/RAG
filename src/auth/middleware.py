"""
Authentication Middleware for FastAPI.

This module provides middleware for JWT authentication, user context injection,
and permission enforcement in FastAPI applications.
"""

from typing import Optional, Dict, Any, Callable
from fastapi import HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import time

from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from .jwt_manager import JWTManager, TokenPayload
from .rbac import RBACManager
from .models import User
from .exceptions import (
    TokenExpiredError, InvalidTokenError, TokenBlacklistedError,
    AuthenticationError, InsufficientPermissionsError
)


class AuthMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """
    Authentication middleware for FastAPI applications.
    
    Handles JWT token validation, user context injection, and request logging.
    """
    
    def __init__(self, app, jwt_manager: JWTManager, rbac_manager: RBACManager, 
                 db_manager: DatabaseManager):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            jwt_manager: JWT token manager
            rbac_manager: RBAC manager
            db_manager: Database manager
        """
        super().__init__(app)
        self.jwt_manager = jwt_manager
        self.rbac_manager = rbac_manager
        self.db_manager = db_manager
        
        # Paths that don't require authentication
        self.public_paths = {
            "/", "/health", "/docs", "/redoc", "/openapi.json",
            "/auth/login", "/auth/register", "/auth/refresh",
            "/auth/reset-password", "/auth/verify-email"
        }
        
        # Paths that require authentication
        self.protected_paths = {
            "/auth/me", "/auth/logout", "/auth/change-password",
            "/users/", "/roles/", "/permissions/",
            "/query", "/chat", "/embeddings", "/extract"
        }
        
        self.logger.info("Authentication middleware initialized")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through authentication middleware.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        # Check if path requires authentication
        if not self._requires_authentication(request):
            response = await call_next(request)
            self._log_request(request, response, time.time() - start_time)
            return response
        
        try:
            # Extract and validate token
            token = self._extract_token(request)
            if not token:
                return self._unauthorized_response("Missing authentication token")
            
            # Validate token and get user context
            token_payload = self.jwt_manager.validate_token(token)
            user_context = await self._get_user_context(token_payload)
            
            # Inject user context into request
            request.state.user = user_context
            request.state.token_payload = token_payload
            
            # Process request
            response = await call_next(request)
            
            # Log successful request
            self._log_request(request, response, time.time() - start_time, user_context)
            
            return response
            
        except TokenExpiredError:
            return self._unauthorized_response("Token has expired")
        except InvalidTokenError as e:
            return self._unauthorized_response(f"Invalid token: {e}")
        except TokenBlacklistedError:
            return self._unauthorized_response("Token has been revoked")
        except AuthenticationError as e:
            return self._unauthorized_response(str(e))
        except Exception as e:
            self.logger.error(f"Authentication middleware error: {e}")
            return self._server_error_response("Authentication failed")
    
    def _requires_authentication(self, request: Request) -> bool:
        """
        Check if request path requires authentication.
        
        Args:
            request: HTTP request
            
        Returns:
            True if authentication required, False otherwise
        """
        path = request.url.path
        
        # Check public paths
        if path in self.public_paths:
            return False
        
        # Check protected paths
        for protected_path in self.protected_paths:
            if path.startswith(protected_path):
                return True
        
        # Default to requiring authentication for unknown paths
        return True
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from request.
        
        Args:
            request: HTTP request
            
        Returns:
            JWT token string or None
        """
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        # Check query parameter (for WebSocket or special cases)
        token = request.query_params.get("token")
        if token:
            return token
        
        # Check cookie (if using cookie-based auth)
        token = request.cookies.get("access_token")
        if token:
            return token
        
        return None
    
    async def _get_user_context(self, token_payload: TokenPayload) -> Dict[str, Any]:
        """
        Get user context from token payload.
        
        Args:
            token_payload: Decoded token payload
            
        Returns:
            User context dictionary
        """
        # Get user from database
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.id == token_payload.user_id).first()
            if not user:
                raise AuthenticationError("User not found")
            
            if not user.is_active:
                raise AuthenticationError("User account is disabled")
            
            if user.is_locked():
                raise AuthenticationError("User account is locked")
            
            # Get fresh permissions (in case they changed since token was issued)
            user_permissions = self.rbac_manager.get_user_permissions(user.id)
            user_roles = self.rbac_manager.get_user_roles(user.id)
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_superuser": user.is_superuser,
                "is_verified": user.is_verified,
                "roles": user_roles,
                "permissions": list(user_permissions),
                "last_login": user.last_login,
                "timezone": user.timezone
            }
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """
        Create unauthorized response.
        
        Args:
            message: Error message
            
        Returns:
            JSON response with 401 status
        """
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Unauthorized", "message": message}
        )
    
    def _server_error_response(self, message: str) -> JSONResponse:
        """
        Create server error response.
        
        Args:
            message: Error message
            
        Returns:
            JSON response with 500 status
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal Server Error", "message": message}
        )
    
    def _log_request(self, request: Request, response: Response, 
                    duration: float, user_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log request details.
        
        Args:
            request: HTTP request
            response: HTTP response
            duration: Request duration in seconds
            user_context: User context if authenticated
        """
        user_info = "anonymous"
        if user_context:
            user_info = f"user:{user_context['username']}({user_context['id']})"
        
        self.logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - "
            f"{duration:.3f}s - {user_info}"
        )


class BearerTokenAuth(HTTPBearer):
    """
    Bearer token authentication for FastAPI dependency injection.
    """
    
    def __init__(self, jwt_manager: JWTManager, rbac_manager: RBACManager, 
                 db_manager: DatabaseManager, auto_error: bool = True):
        """
        Initialize bearer token authentication.
        
        Args:
            jwt_manager: JWT token manager
            rbac_manager: RBAC manager
            db_manager: Database manager
            auto_error: Whether to raise HTTPException on auth failure
        """
        super().__init__(auto_error=auto_error)
        self.jwt_manager = jwt_manager
        self.rbac_manager = rbac_manager
        self.db_manager = db_manager
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        Authenticate request and return user context.
        
        Args:
            request: HTTP request
            
        Returns:
            User context dictionary
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            credentials: HTTPAuthorizationCredentials = await super().__call__(request)
            
            # Validate token
            token_payload = self.jwt_manager.validate_token(credentials.credentials)
            
            # Get user context
            with self.db_manager.get_session() as session:
                user = session.query(User).filter(User.id == token_payload.user_id).first()
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not found"
                    )
                
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is disabled"
                    )
                
                if user.is_locked():
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is locked"
                    )
                
                # Get fresh permissions
                user_permissions = self.rbac_manager.get_user_permissions(user.id)
                user_roles = self.rbac_manager.get_user_roles(user.id)
                
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "is_superuser": user.is_superuser,
                    "is_verified": user.is_verified,
                    "roles": user_roles,
                    "permissions": list(user_permissions),
                    "last_login": user.last_login,
                    "timezone": user.timezone,
                    "token_payload": token_payload
                }
                
        except TokenExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {e}"
            )
        except TokenBlacklistedError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )


class RequirePermission:
    """
    Dependency class to require specific permissions.
    """
    
    def __init__(self, permission: str):
        """
        Initialize permission requirement.
        
        Args:
            permission: Required permission name
        """
        self.permission = permission
    
    def __call__(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if user has required permission.
        
        Args:
            user_context: User context from authentication
            
        Returns:
            User context if authorized
            
        Raises:
            HTTPException: If user lacks permission
        """
        if self.permission not in user_context.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {self.permission} required"
            )
        
        return user_context


class RequireRole:
    """
    Dependency class to require specific roles.
    """
    
    def __init__(self, role: str):
        """
        Initialize role requirement.
        
        Args:
            role: Required role name
        """
        self.role = role
    
    def __call__(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if user has required role.
        
        Args:
            user_context: User context from authentication
            
        Returns:
            User context if authorized
            
        Raises:
            HTTPException: If user lacks role
        """
        if self.role not in user_context.get("roles", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {self.role} role required"
            )
        
        return user_context


class RequireAccess:
    """
    Dependency class to require access to resources.
    """
    
    def __init__(self, resource: str, action: str):
        """
        Initialize access requirement.
        
        Args:
            resource: Resource type
            action: Action type
        """
        self.resource = resource
        self.action = action
    
    def __call__(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if user has required access.
        
        Args:
            user_context: User context from authentication
            
        Returns:
            User context if authorized
            
        Raises:
            HTTPException: If user lacks access
        """
        permissions = user_context.get("permissions", [])
        
        # Check for specific permission
        specific_permission = f"{self.resource}:{self.action}"
        if specific_permission in permissions:
            return user_context
        
        # Check for wildcard permissions
        wildcard_resource = f"{self.resource}:*"
        wildcard_action = f"*:{self.action}"
        wildcard_all = "*:*"
        
        if any(perm in permissions for perm in [wildcard_resource, wildcard_action, wildcard_all]):
            return user_context
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions: {self.action} access to {self.resource} required"
        )


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware, LoggerMixin):
    """
    Rate limiting middleware for API endpoints.
    """
    
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
        
        self.logger.info(f"Rate limiting middleware initialized: {max_requests} requests per {window_seconds}s")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        # Get client identifier
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(client_id, current_time)
        
        # Check rate limit
        if len(self.requests.get(client_id, [])) >= self.max_requests:
            self.logger.warning(f"Rate limit exceeded for client {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded", "message": "Too many requests"}
            )
        
        # Record request
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.max_requests - len(self.requests[client_id]))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Args:
            request: HTTP request
            
        Returns:
            Client identifier
        """
        # Try to get authenticated user ID
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user['id']}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host}"
    
    def _clean_old_requests(self, client_id: str, current_time: float) -> None:
        """
        Clean old requests from tracking.
        
        Args:
            client_id: Client identifier
            current_time: Current timestamp
        """
        if client_id in self.requests:
            cutoff_time = current_time - self.window_seconds
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[client_id]:
                del self.requests[client_id]