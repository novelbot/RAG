"""
Authentication module for the RAG Server Web UI
Handles JWT-based authentication and session management
"""

import streamlit as st
import requests
import jwt
import json
from datetime import datetime, timedelta
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
            if exp and datetime.utcnow().timestamp() > exp:
                self.logout()
                return False
                
            return True
        except jwt.InvalidTokenError:
            self.logout()
            return False
    
    def login_page(self):
        """Display the login page"""
        st.title("🤖 RAG Server Login")
        
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
    
    def _authenticate(self, username: str, password: str) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Authenticate user with the RAG server API
        Returns: (success, jwt_token, user_info)
        """
        try:
            # For demo purposes, we'll use mock authentication
            # In production, this should call the actual RAG server API
            if self._demo_authenticate(username, password):
                # Generate a mock JWT token
                user_info = self._get_demo_user_info(username)
                token = self._generate_mock_jwt(user_info)
                return True, token, user_info
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
            "iat": datetime.utcnow().timestamp(),
            "exp": (datetime.utcnow() + timedelta(hours=8)).timestamp()
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