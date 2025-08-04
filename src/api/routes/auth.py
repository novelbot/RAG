"""
Authentication API routes for user login, logout, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import asyncio

from ...auth.dependencies import MockUser
from ...auth.sqlite_auth import auth_manager
from ...auth.jwt_manager import JWTManager
from ..schemas import LoginRequest, TokenResponse, UserResponse, RegisterRequest, RegisterResponse, RefreshTokenRequest
from ...metrics.collectors import session_collector

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
jwt_manager = JWTManager()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, http_request: Request) -> TokenResponse:
    """
    Authenticate user and return JWT token.
    
    Args:
        request: Login credentials containing username and password
        http_request: FastAPI request object for extracting IP and user agent
        
    Returns:
        TokenResponse: JWT access token and refresh token
        
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    # SQLite 기반 실제 인증
    user_data = auth_manager.authenticate(request.username, request.password)
    
    if user_data:
        # JWT Manager를 사용하여 올바른 토큰 쌍 생성
        token_response = jwt_manager.generate_token_pair(user_data)
        
        # Log successful login event
        try:
            # Extract client information
            ip_address = http_request.headers.get("X-Forwarded-For", 
                                                 http_request.headers.get("X-Real-IP", 
                                                 getattr(http_request.client, "host", "unknown") if http_request.client else "unknown"))
            user_agent = http_request.headers.get("user-agent", "")
            
            # Start user session and log login
            await session_collector.start_session(
                user_id=str(user_data['id']),
                ip_address=ip_address,
                user_agent=user_agent
            )
        except Exception as e:
            # Don't fail login if logging fails
            print(f"Failed to log login event: {e}")
        
        return TokenResponse(
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            token_type=token_response.token_type,
            expires_in=token_response.expires_in
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest) -> RegisterResponse:
    """
    Register a new user.
    
    Args:
        request: Registration data containing username, password, email, and role
        
    Returns:
        RegisterResponse: Registration result with user info if successful
        
    Raises:
        HTTPException: 400 if username already exists or registration fails
    """
    # 새 사용자 생성
    user_data = auth_manager.create_user(
        username=request.username,
        password=request.password,
        email=request.email or f"{request.username}@example.com",
        role=request.role or "user"
    )
    
    if user_data:
        return RegisterResponse(
            message="User registered successfully",
            user_id=str(user_data["id"]),
            username=user_data["username"]
        )
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Username already exists or registration failed"
    )


@router.post("/logout")
async def logout(token: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:
    """
    Logout user and invalidate token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Dict: Success message
    """
    # Get user info before invalidating token
    try:
        token_payload = jwt_manager.validate_token(token.credentials, token_type="access")
        user_id = str(token_payload.user_id)
        
        # Log logout event
        await session_collector.end_session(user_id)
        
        # Blacklist the token
        jwt_manager.blacklist_token(token.credentials)
        
    except Exception as e:
        # Don't fail logout if logging fails or token is invalid
        print(f"Failed to log logout event or invalidate token: {e}")
    
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """
    Refresh JWT access token using refresh token.
    
    Args:
        request: RefreshTokenRequest containing the refresh token
        
    Returns:
        TokenResponse: New access token and refresh token
        
    Raises:
        HTTPException: 401 if refresh token is invalid
    """
    # Simulate async token refresh process
    await asyncio.sleep(0.1)
    
    try:
        # Verify the refresh token using JWT Manager
        token_payload = jwt_manager.validate_token(request.refresh_token, token_type="refresh")
        
        # Get user data from the database using token payload
        user_data = {
            'id': token_payload.user_id,
            'username': token_payload.username,
            'email': token_payload.email,
            'roles': token_payload.roles,
            'permissions': token_payload.permissions
        }
        
        # Blacklist the old refresh token (token rotation for security)
        jwt_manager.blacklist_token(request.refresh_token)
        
        # Generate new token pair
        token_response = jwt_manager.generate_token_pair(user_data)
        
        return TokenResponse(
            access_token=token_response.access_token,
            refresh_token=token_response.refresh_token,
            token_type=token_response.token_type,
            expires_in=token_response.expires_in
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """
    Get current authenticated user information.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        UserResponse: Current user information
        
    Raises:
        HTTPException: 401 if token is invalid
    """
    # JWT Manager로 토큰 검증
    try:
        token_payload = jwt_manager.validate_token(token.credentials, token_type="access")
        
        return UserResponse(
            id=str(token_payload.user_id),
            username=token_payload.username,
            email=token_payload.email,
            roles=token_payload.roles if token_payload.roles else ["user"],
            is_active=True
        )
        
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


