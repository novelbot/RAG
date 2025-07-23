"""
RAG Server Web UI - Main Application
A Streamlit-based web interface for the RAG server system.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to import the RAG server modules
sys.path.append(str(Path(__file__).parent.parent))

# Import authentication and page modules
from webui.auth import AuthManager, require_auth
from webui.pages import _dashboard as dashboard, _documents as documents, _query as query, _admin as admin, _settings as settings, _simple_query as simple_query

def main():
    """Main application entry point"""
    
    # Configure page settings
    st.set_page_config(
        page_title="RAG Server",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide Streamlit's default navigation menu
    st.markdown("""
    <style>
        /* Hide the auto-generated navigation links completely */
        [data-testid="stSidebar"] nav[data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Alternative selectors for different Streamlit versions */
        .stSidebar nav,
        .stSidebar ul,
        section[data-testid="stSidebar"] nav,
        section[data-testid="stSidebar"] ul {
            display: none !important;
        }
        
        /* Hide any navigation list items */
        [data-testid="stSidebar"] ul li {
            display: none !important;
        }
        
        /* Hide specific navigation elements */
        .css-1544g2n,
        .css-1d391kg,
        .stSelectbox[data-baseweb="select"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize authentication manager
    auth_manager = AuthManager()
    
    # Initialize session state for navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    
    # Authentication check
    if not auth_manager.is_authenticated():
        auth_manager.login_page()
        return
    
    # Main application layout
    with st.sidebar:
        st.title("🤖 RAG Server")
        st.markdown("---")
        
        # User info
        user_info = auth_manager.get_user_info()
        st.write(f"👤 {user_info.get('username', 'User')}")
        st.write(f"🔑 {user_info.get('role', 'user').title()}")
        st.markdown("---")
        
        # Navigation menu
        pages = {
            "📊 Dashboard": "dashboard",
            "📄 Documents": "documents", 
            "🔍 Query": "query",
            "⚙️ Settings": "settings"
        }
        
        # Add admin page for admin users
        if user_info.get("role") == "admin":
            pages["👥 Admin"] = "admin"
            
        # Navigation buttons
        for page_name, page_key in pages.items():
            if st.button(
                page_name, 
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_key else "secondary"
            ):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            auth_manager.logout()
            st.rerun()
    
    # Main content area
    current_page = st.session_state.current_page
    
    # Route to appropriate page
    if current_page == "dashboard":
        dashboard.show()
    elif current_page == "documents":
        documents.show()
    elif current_page == "query":
        query.show()
    elif current_page == "admin":
        if user_info.get("role") == "admin":
            admin.show()
        else:
            st.error("Access denied. Admin privileges required.")
    elif current_page == "settings":
        settings.show()
    else:
        st.error("Page not found")

if __name__ == "__main__":
    main()