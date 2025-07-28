"""
Authentication module for the RAG Server Web UI
Handles JWT-based authentication and session management
"""

import streamlit as st
import requests
import jwt
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
import hashlib
import hmac

@dataclass
class UserInfo:
    username: str
    role: str
    user_id: str
    email: str = ""

class AuthManager:
    """Manages authentication for the Streamlit application"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session_key = "auth_session"
        self.token_key = "jwt_token"
        self.user_info_key = "user_info"
    
    def is_authenticated(self) -> bool:
        """Check if the user is currently authenticated"""
        if self.session_key not in st.session_state:
            return False
            
        # Check if token exists and is valid
        token = st.session_state.get(self.token_key)
        if not token:
            return False
            
        try:
            # Decode JWT without verification (we'll verify with the server)
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Check expiration
            exp = payload.get('exp')
            if exp and datetime.now(timezone.utc).timestamp() > exp:
                self.logout()
                return False
                
            return True
        except jwt.InvalidTokenError:
            self.logout()
            return False
    
    def login_page(self):
        """Display the login page"""
        st.title("🤖 RAG Server Login")
        
        # 탭으로 로그인/회원가입 구분
        tab1, tab2 = st.tabs(["로그인", "회원가입"])
        
        with tab1:
            self._show_login_form()
        
        with tab2:
            self._show_register_form()
    
    def _show_login_form(self):
        """Display the login form"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.markdown("### Please sign in")
                
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                login_button = st.form_submit_button("Login", use_container_width=True)
                
                if login_button:
                    if not username or not password:
                        st.error("Please enter both username and password")
                        return
                    
                    # Attempt authentication
                    success, token, user_info = self._authenticate(username, password)
                    
                    if success:
                        # Store authentication data in session state
                        st.session_state[self.session_key] = True
                        st.session_state[self.token_key] = token
                        st.session_state[self.user_info_key] = user_info
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            # Demo credentials info (remove in production)
            with st.expander("Demo Credentials", expanded=False):
                st.info("""
                **Demo Admin Account:**
                - Username: `admin`
                - Password: `admin123`
                
                **Demo User Account:**
                - Username: `user`
                - Password: `user123`
                
                *Note: These are demo credentials for testing purposes.*
                """)
    
    def _show_register_form(self):
        """Display the registration form"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("register_form"):
                st.markdown("### Create a new account")
                
                username = st.text_input(
                    "Username", 
                    placeholder="Choose a username (3-50 characters)",
                    help="Username must be 3-50 characters long"
                )
                
                email = st.text_input(
                    "Email (optional)", 
                    placeholder="Enter your email address"
                )
                
                password = st.text_input(
                    "Password", 
                    type="password",
                    placeholder="Enter a secure password (minimum 6 characters)",
                    help="Password must be at least 6 characters long"
                )
                
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password", 
                    placeholder="Re-enter your password"
                )
                
                role = st.selectbox(
                    "Role",
                    options=["user", "manager"],
                    index=0,
                    help="Select your account role (admin accounts must be created by an administrator)"
                )
                
                register_button = st.form_submit_button("Create Account", use_container_width=True)
                
                if register_button:
                    # Validation
                    if not username:
                        st.error("Username is required")
                        return
                    
                    if len(username) < 3 or len(username) > 50:
                        st.error("Username must be 3-50 characters long")
                        return
                    
                    if not password:
                        st.error("Password is required")
                        return
                    
                    if len(password) < 6:
                        st.error("Password must be at least 6 characters long")
                        return
                    
                    if password != confirm_password:
                        st.error("Passwords do not match")
                        return
                    
                    # 회원가입 시도
                    success, message = self._register(username, password, email, role)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.info("You can now login with your new account using the Login tab")
                    else:
                        st.error(f"❌ {message}")
    
    def _register(self, username: str, password: str, email: str = "", role: str = "user") -> tuple[bool, str]:
        """
        Register a new user with the RAG server API
        Returns: (success, message)
        """
        try:
            from webui.api_client import get_api_client
            api_client = get_api_client()
            
            result = api_client.register(
                username=username,
                password=password,
                email=email,
                role=role
            )
            
            return True, result.get("message", "Account created successfully")
            
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                return False, "Username already exists. Please choose a different username."
            elif "400" in error_msg:
                return False, "Registration failed. Please check your input and try again."
            else:
                return False, f"Registration failed: {error_msg}"
    
    def _authenticate(self, username: str, password: str) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Authenticate user with the RAG server API
        Returns: (success, jwt_token, user_info)
        """
        try:
            # 실제 백엔드 API 호출
            response = requests.post(
                f"{self.api_base_url}/api/v1/auth/login",
                json={"username": username, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                token = data.get("access_token")
                
                # 토큰으로 사용자 정보 가져오기
                user_info_response = requests.get(
                    f"{self.api_base_url}/api/v1/auth/me",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10
                )
                
                if user_info_response.status_code == 200:
                    user_info = user_info_response.json()
                    return True, token, user_info
                else:
                    return False, None, None
            else:
                return False, None, None
                
        except requests.RequestException as e:
            if hasattr(st, 'error'):
                st.error(f"Authentication error: {str(e)}")
            return False, None, None
    
    def _demo_authenticate(self, username: str, password: str) -> bool:
        """Demo authentication - replace with real API call"""
        demo_users = {
            "admin": {"password": "admin123", "role": "admin"},
            "user": {"password": "user123", "role": "user"},
            "manager": {"password": "manager123", "role": "manager"}
        }
        
        user_data = demo_users.get(username)
        if user_data and user_data["password"] == password:
            return True
        return False
    
    def _get_demo_user_info(self, username: str) -> Dict[str, Any]:
        """Get demo user information"""
        demo_users = {
            "admin": {
                "username": "admin", 
                "role": "admin", 
                "user_id": "admin_001",
                "email": "admin@ragserver.local"
            },
            "user": {
                "username": "user", 
                "role": "user", 
                "user_id": "user_001",
                "email": "user@ragserver.local"
            },
            "manager": {
                "username": "manager", 
                "role": "manager", 
                "user_id": "manager_001",
                "email": "manager@ragserver.local"
            }
        }
        return demo_users.get(username, {})
    
    def _generate_mock_jwt(self, user_info: Dict[str, Any]) -> str:
        """Generate a mock JWT token for demo purposes"""
        payload = {
            "username": user_info["username"],
            "role": user_info["role"],
            "user_id": user_info["user_id"],
            "iat": datetime.now(timezone.utc).timestamp(),
            "exp": (datetime.now(timezone.utc) + timedelta(hours=8)).timestamp()
        }
        
        # For demo purposes, use a simple encoding (not secure for production)
        secret = "demo_secret_key"
        return jwt.encode(payload, secret, algorithm="HS256")
    
    def logout(self):
        """Logout the user and clear session data"""
        keys_to_remove = [self.session_key, self.token_key, self.user_info_key]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get the current user's information"""
        return st.session_state.get(self.user_info_key, {})
    
    def get_token(self) -> Optional[str]:
        """Get the current JWT token"""
        return st.session_state.get(self.token_key)
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests"""
        token = self.get_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

def require_auth(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        auth_manager = AuthManager()
        if not auth_manager.is_authenticated():
            auth_manager.login_page()
            return
        return func(*args, **kwargs)
    return wrapper

def require_role(required_role: str):
    """Decorator to require a specific role"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            auth_manager = AuthManager()
            if not auth_manager.is_authenticated():
                auth_manager.login_page()
                return
            
            user_info = auth_manager.get_user_info()
            user_role = user_info.get("role", "user")
            
            # Simple role hierarchy: admin > manager > user
            role_hierarchy = {"user": 1, "manager": 2, "admin": 3}
            
            if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 999):
                st.error(f"Access denied. {required_role.title()} role required.")
                return
                
            return func(*args, **kwargs)
        return wrapper
    return decorator