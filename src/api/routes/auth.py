"""
Authentication API routes for user login, logout, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import asyncio

from ...auth.dependencies import MockUser
from ...auth.sqlite_auth import auth_manager
from ..schemas import LoginRequest, TokenResponse, UserResponse, RegisterRequest, RegisterResponse
from ...metrics.collectors import session_collector

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()


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
        token = auth_manager.create_token(user_data)
        
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
            access_token=token,
            refresh_token=token,  # 간단히 같은 토큰 사용
            token_type="bearer",
            expires_in=86400  # 24시간
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
        session_data = auth_manager.verify_token(token.credentials)
        if session_data:
            user_id = str(session_data.get('user_id'))
            # Log logout event
            await session_collector.end_session(user_id)
    except Exception as e:
        # Don't fail logout if logging fails
        print(f"Failed to log logout event: {e}")
    
    # 실제 토큰 무효화
    auth_manager.logout(token.credentials)
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str) -> TokenResponse:
    """
    Refresh JWT access token using refresh token.
    
    Args:
        refresh_token: Valid refresh token
        
    Returns:
        TokenResponse: New access token and refresh token
        
    Raises:
        HTTPException: 401 if refresh token is invalid
    """
    # Simulate async token refresh process
    await asyncio.sleep(0.1)
    
    # TODO: Implement actual token refresh logic
    if refresh_token == "demo_refresh_token":
        return TokenResponse(
            access_token="new_access_token",
            refresh_token="new_refresh_token",
            token_type="bearer",
            expires_in=3600
        )
    
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
    # 실제 토큰 검증
    session_data = auth_manager.verify_token(token.credentials)
    
    if session_data:
        return UserResponse(
            id=str(session_data["user_id"]),
            username=session_data["username"],
            email=session_data["email"],
            roles=[session_data["role"]],
            is_active=True
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


