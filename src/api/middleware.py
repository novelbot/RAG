"""
API Middleware for FastAPI application.
"""

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
import time
import json
from typing import Optional, Set
from urllib.parse import urlparse


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Simple authentication middleware for API endpoints.
    """
    
    def __init__(self, app, exempt_paths: Optional[Set[str]] = None):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            exempt_paths: Set of paths that don't require authentication
        """
        super().__init__(app)
        self.exempt_paths = exempt_paths or {
            "/",
            "/health",
            "/docs", 
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/monitoring/health",
            "/api/v1/monitoring/health/simple"
        }
    
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
        if self._is_exempt_path(request.url.path):
            response = await call_next(request)
            self._add_timing_header(response, start_time)
            return response
        
        # Check for authentication token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return self._unauthorized_response("Missing or invalid authorization header")
        
        token = auth_header.split(" ")[1]
        
        # Simple token validation (replace with actual JWT validation)
        if not self._validate_token(token):
            return self._unauthorized_response("Invalid authentication token")
        
        # Process request
        response = await call_next(request)
        self._add_timing_header(response, start_time)
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if path is exempt from authentication.
        
        Args:
            path: Request path
            
        Returns:
            True if path is exempt, False otherwise
        """
        return path in self.exempt_paths
    
    def _validate_token(self, token: str) -> bool:
        """
        Validate authentication token.
        
        Args:
            token: JWT token
            
        Returns:
            True if token is valid, False otherwise
        """
        # TODO: Implement actual JWT validation
        # For now, accept specific demo tokens
        valid_tokens = {
            "demo_access_token",
            "admin_token",
            "test_token"
        }
        return token in valid_tokens
    
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
            content={
                "error": "Unauthorized",
                "message": message,
                "timestamp": time.time()
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _add_timing_header(self, response: Response, start_time: float) -> None:
        """
        Add request timing header to response.
        
        Args:
            response: HTTP response
            start_time: Request start time
        """
        duration = time.time() - start_time
        response.headers["X-Process-Time"] = f"{duration:.4f}"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging API requests and responses.
    """
    
    def __init__(self, app):
        """
        Initialize request logging middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through logging middleware.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        # Extract request details
        method = request.method
        url = str(request.url)
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request
        self._log_request(
            method=method,
            url=url,
            status_code=response.status_code,
            duration=duration,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    def _log_request(self, method: str, url: str, status_code: int, 
                    duration: float, client_ip: str, user_agent: str) -> None:
        """
        Log request details.
        
        Args:
            method: HTTP method
            url: Request URL
            status_code: Response status code
            duration: Request duration in seconds
            client_ip: Client IP address
            user_agent: User agent string
        """
        # TODO: Use proper logging framework
        print(f"[API] {method} {url} - {status_code} - {duration:.4f}s - {client_ip} - {user_agent}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
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
        self.request_counts = {}  # client_ip -> list of timestamps
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through rate limiting middleware.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(client_ip, current_time)
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            return self._rate_limit_response()
        
        # Record request
        self._record_request(client_ip, current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_ip, current_time)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, client_ip: str, current_time: float) -> None:
        """
        Remove old request timestamps outside the window.
        
        Args:
            client_ip: Client IP address
            current_time: Current timestamp
        """
        if client_ip in self.request_counts:
            cutoff_time = current_time - self.window_seconds
            self.request_counts[client_ip] = [
                timestamp for timestamp in self.request_counts[client_ip]
                if timestamp > cutoff_time
            ]
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if rate limited, False otherwise
        """
        if client_ip not in self.request_counts:
            return False
        
        return len(self.request_counts[client_ip]) >= self.max_requests
    
    def _record_request(self, client_ip: str, current_time: float) -> None:
        """
        Record request timestamp for client.
        
        Args:
            client_ip: Client IP address
            current_time: Current timestamp
        """
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        self.request_counts[client_ip].append(current_time)
    
    def _rate_limit_response(self) -> JSONResponse:
        """
        Create rate limit exceeded response.
        
        Returns:
            JSON response with 429 status
        """
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "timestamp": time.time()
            }
        )
    
    def _add_rate_limit_headers(self, response: Response, client_ip: str, current_time: float) -> None:
        """
        Add rate limit headers to response.
        
        Args:
            response: HTTP response
            client_ip: Client IP address
            current_time: Current timestamp
        """
        remaining = max(0, self.max_requests - len(self.request_counts.get(client_ip, [])))
        reset_time = int(current_time + self.window_seconds)
        
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware for additional control.
    """
    
    def __init__(self, app, allowed_origins: Optional[list] = None):
        """
        Initialize CORS middleware.
        
        Args:
            app: FastAPI application
            allowed_origins: List of allowed origins
        """
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through CORS middleware.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        origin = request.headers.get("Origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, origin)
            return response
        
        # Process request
        response = await call_next(request)
        self._add_cors_headers(response, origin)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: Optional[str]) -> None:
        """
        Add CORS headers to response.
        
        Args:
            response: HTTP response
            origin: Request origin
        """
        if origin and (self.allowed_origins == ["*"] or origin in self.allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif self.allowed_origins == ["*"]:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"