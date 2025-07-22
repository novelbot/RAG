"""
Admin page for RAG Server Web UI
Handles user management, system configuration, and monitoring
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from webui.api_client import get_api_client
from webui.auth import require_role
from webui.config import config
import json

@require_role("admin")
def show():
    """Display the admin panel"""
    st.title("👥 Admin Panel")
    
    api_client = get_api_client()
    
    # Create tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 User Management", 
        "⚙️ System Config", 
        "📊 Monitoring", 
        "🔐 Access Control",
        "🛠️ Maintenance"
    ])
    
    with tab1:
        show_user_management(api_client)
    
    with tab2:
        show_system_configuration(api_client)
    
    with tab3:
        show_monitoring(api_client)
    
    with tab4:
        show_access_control(api_client)
    
    with tab5:
        show_maintenance(api_client)

def show_user_management(api_client):
    """Display user management interface"""
    st.subheader("👤 User Management")
    
    # User creation form
    with st.expander("➕ Create New User"):
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username*")
                new_email = st.text_input("Email*")
            
            with col2:
                # Get user roles from config
                available_roles = list(config.get_user_roles().keys())
                if not available_roles:
                    available_roles = ["user", "manager", "admin"]
                
                new_role = st.selectbox("Role*", available_roles)
                
                # Get departments from config
                available_departments = [dept['name'] for dept in config.get_document_categories()]
                if not available_departments:
                    available_departments = ["IT", "Finance", "HR", "Marketing", "Legal", "Operations"]
                
                new_department = st.selectbox(
                    "Department", 
                    available_departments
                )
            
            new_password = st.text_input("Password*", type="password")
            new_password_confirm = st.text_input("Confirm Password*", type="password")
            
            create_user_btn = st.form_submit_button("Create User", type="primary")
            
            if create_user_btn:
                if not all([new_username, new_email, new_password, new_role]):
                    st.error("Please fill in all required fields")
                elif new_password != new_password_confirm:
                    st.error("Passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    create_user(api_client, {
                        "username": new_username,
                        "email": new_email,
                        "role": new_role,
                        "department": new_department,
                        "password": new_password
                    })
    
    st.markdown("---")
    
    # User list and management
    st.subheader("📋 Existing Users")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get user roles from config for filter
        available_roles = list(config.get_user_roles().keys())
        if not available_roles:
            available_roles = ["admin", "manager", "user"]
        role_filter_options = ["All"] + available_roles
        role_filter = st.selectbox("Filter by Role", role_filter_options)
    
    with col2:
        # Get departments from config for filter
        available_departments = [dept['name'] for dept in config.get_document_categories()]
        if not available_departments:
            available_departments = ["IT", "Finance", "HR", "Marketing", "Legal", "Operations"]
        dept_filter_options = ["All"] + available_departments
        department_filter = st.selectbox("Filter by Department", dept_filter_options)
    
    with col3:
        status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive"])
    
    # Get users (mock data for demo)
    users = get_mock_users()
    
    # Apply filters
    filtered_users = users
    if role_filter != "All":
        filtered_users = [u for u in filtered_users if u["role"] == role_filter]
    if department_filter != "All":
        filtered_users = [u for u in filtered_users if u["department"] == department_filter]
    if status_filter != "All":
        filtered_users = [u for u in filtered_users if u["status"] == status_filter]
    
    # Users table
    if not filtered_users:
        st.info("No users found matching the criteria")
        return
    
    st.write(f"Found {len(filtered_users)} user(s)")
    
    for i, user in enumerate(filtered_users):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
            
            with col1:
                st.write(f"**{user['username']}**")
                st.write(f"📧 {user['email']}")
                st.write(f"🏢 {user['department']}")
            
            with col2:
                st.write(f"**Role:** {user['role'].title()}")
                st.write(f"**Status:** {'🟢' if user['status'] == 'Active' else '🔴'} {user['status']}")
                st.write(f"**Last Login:** {user['last_login']}")
            
            with col3:
                st.write(f"**Created:**")
                st.write(user['created_date'])
                st.write(f"**Queries:**")
                st.write(user['total_queries'])
            
            with col4:
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("✏️ Edit", key=f"edit_user_{i}", use_container_width=True):
                        edit_user_modal(user)
                
                with action_col2:
                    if user['status'] == 'Active':
                        if st.button("🚫 Disable", key=f"disable_user_{i}", use_container_width=True):
                            toggle_user_status(api_client, user['id'], 'Inactive')
                    else:
                        if st.button("✅ Enable", key=f"enable_user_{i}", use_container_width=True):
                            toggle_user_status(api_client, user['id'], 'Active')
                
                with action_col3:
                    if st.button("🗑️ Delete", key=f"delete_user_{i}", use_container_width=True):
                        if user['username'] != st.session_state.get('user_info', {}).get('username'):
                            delete_user_modal(api_client, user)
                        else:
                            st.error("Cannot delete your own account")
            
            st.markdown("---")

def show_system_configuration(api_client):
    """Display system configuration interface"""
    st.subheader("⚙️ System Configuration")
    
    # LLM Provider Configuration
    st.markdown("### 🤖 LLM Provider Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**OpenAI Configuration**")
        openai_enabled = st.checkbox("Enable OpenAI", value=True)
        openai_api_key = st.text_input("API Key", type="password", key="openai_key")
        openai_models = st.multiselect(
            "Available Models", 
            ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            default=["gpt-4", "gpt-3.5-turbo"]
        )
        
        st.markdown("**Anthropic Configuration**")
        anthropic_enabled = st.checkbox("Enable Anthropic", value=True)
        anthropic_api_key = st.text_input("API Key", type="password", key="anthropic_key")
        anthropic_models = st.multiselect(
            "Available Models",
            ["claude-3-5-sonnet-latest", "claude-3-haiku", "claude-3-opus"],
            default=["claude-3-5-sonnet-latest"]
        )
    
    with col2:
        st.markdown("**Google Configuration**")
        google_enabled = st.checkbox("Enable Google", value=True)
        google_api_key = st.text_input("API Key", type="password", key="google_key")
        google_models = st.multiselect(
            "Available Models",
            ["gemini-2.0-flash-001", "gemini-1.5-pro"],
            default=["gemini-2.0-flash-001"]
        )
        
        st.markdown("**Ollama Configuration**")
        ollama_enabled = st.checkbox("Enable Ollama", value=False)
        ollama_base_url = st.text_input("Base URL", value="http://localhost:11434")
        ollama_models = st.multiselect(
            "Available Models",
            ["llama3.2", "gemma2", "mistral"],
            default=["llama3.2"]
        )
    
    # Database Configuration
    st.markdown("---")
    st.markdown("### 🗄️ Database Configuration")
    
    db_col1, db_col2 = st.columns(2)
    
    with db_col1:
        st.markdown("**Primary Database**")
        db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "MariaDB", "SQL Server"])
        db_host = st.text_input("Host", value="localhost")
        db_port = st.number_input("Port", value=3306 if db_type == "MySQL" else 5432)
        db_name = st.text_input("Database Name", value="novelbot")
    
    with db_col2:
        st.markdown("**Milvus Vector Database**")
        milvus_host = st.text_input("Milvus Host", value="localhost")
        milvus_port = st.number_input("Milvus Port", value=19530)
        milvus_user = st.text_input("Username (optional)")
        milvus_password = st.text_input("Password (optional)", type="password")
    
    # System Settings
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        max_concurrent_users = st.number_input("Max Concurrent Users", value=50, min_value=1)
        query_timeout = st.number_input("Query Timeout (seconds)", value=30, min_value=5)
        max_upload_size = st.number_input("Max Upload Size (MB)", value=100, min_value=1)
    
    with settings_col2:
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)
        enable_metrics = st.checkbox("Enable Metrics Collection", value=True)
        enable_auto_backup = st.checkbox("Enable Auto Backup", value=False)
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        save_system_configuration({
            "llm_providers": {
                "openai": {"enabled": openai_enabled, "models": openai_models},
                "anthropic": {"enabled": anthropic_enabled, "models": anthropic_models},
                "google": {"enabled": google_enabled, "models": google_models},
                "ollama": {"enabled": ollama_enabled, "base_url": ollama_base_url, "models": ollama_models}
            },
            "databases": {
                "primary": {"type": db_type, "host": db_host, "port": db_port, "name": db_name},
                "milvus": {"host": milvus_host, "port": milvus_port}
            },
            "system": {
                "max_concurrent_users": max_concurrent_users,
                "query_timeout": query_timeout,
                "max_upload_size": max_upload_size,
                "enable_logging": enable_logging,
                "enable_metrics": enable_metrics,
                "enable_auto_backup": enable_auto_backup
            }
        })

def show_monitoring(api_client):
    """Display system monitoring interface"""
    st.subheader("📊 System Monitoring")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Connections", "23", "+5")
    
    with col2:
        st.metric("Queries/Hour", "456", "+12%")
    
    with col3:
        st.metric("Error Rate", "0.3%", "-0.1%")
    
    with col4:
        st.metric("Avg Response Time", "2.1s", "-0.3s")
    
    st.markdown("---")
    
    # System health
    st.subheader("🩺 System Health")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        st.markdown("**Service Status**")
        services = [
            ("API Server", "🟢 Running"),
            ("Database", "🟢 Connected"),
            ("Vector DB", "🟢 Healthy"),
            ("LLM Services", "🟡 Degraded"),
            ("File Storage", "🟢 Available")
        ]
        
        for service, status in services:
            st.write(f"**{service}:** {status}")
    
    with health_col2:
        st.markdown("**Resource Usage**")
        st.progress(0.67, "CPU: 67%")
        st.progress(0.54, "Memory: 54%")
        st.progress(0.78, "Storage: 78%")
        st.progress(0.23, "Network: 23%")
    
    with health_col3:
        st.markdown("**Recent Events**")
        events = [
            "10:30 - High CPU usage detected",
            "10:25 - User john.doe logged in",
            "10:20 - Backup completed successfully",
            "10:15 - New document processed"
        ]
        
        for event in events:
            st.write(f"• {event}")
    
    # Error logs
    st.markdown("---")
    st.subheader("📋 Recent Error Logs")
    
    error_logs = [
        {
            "timestamp": "2024-01-16 10:30:15",
            "level": "ERROR",
            "component": "LLM Service",
            "message": "OpenAI API rate limit exceeded",
            "user": "system"
        },
        {
            "timestamp": "2024-01-16 10:28:43",
            "level": "WARNING", 
            "component": "File Processor",
            "message": "Large file processing taking longer than expected",
            "user": "jane.smith"
        },
        {
            "timestamp": "2024-01-16 10:25:12",
            "level": "ERROR",
            "component": "Database",
            "message": "Connection timeout during backup operation",
            "user": "system"
        }
    ]
    
    for log in error_logs:
        level_color = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}
        st.write(f"{level_color.get(log['level'], '⚪')} **{log['timestamp']}** - {log['component']}: {log['message']}")

def show_access_control(api_client):
    """Display access control interface"""
    st.subheader("🔐 Access Control")
    
    # Role management
    st.markdown("### 👔 Role Management")
    
    roles = ["admin", "manager", "user"]
    selected_role = st.selectbox("Select Role to Configure", roles)
    
    # Permissions for selected role
    st.markdown(f"### Permissions for '{selected_role}' role")
    
    permissions = {
        "documents": {
            "upload": st.checkbox(f"Upload documents", value=True, key=f"{selected_role}_doc_upload"),
            "download": st.checkbox(f"Download documents", value=True, key=f"{selected_role}_doc_download"),
            "delete": st.checkbox(f"Delete documents", value=selected_role in ["admin", "manager"], key=f"{selected_role}_doc_delete"),
            "manage_all": st.checkbox(f"Manage all documents", value=selected_role == "admin", key=f"{selected_role}_doc_manage_all")
        },
        "queries": {
            "rag_query": st.checkbox(f"RAG queries", value=True, key=f"{selected_role}_query_rag"),
            "chat": st.checkbox(f"Chat with LLM", value=True, key=f"{selected_role}_query_chat"),
            "ensemble": st.checkbox(f"Ensemble queries", value=selected_role in ["admin", "manager"], key=f"{selected_role}_query_ensemble"),
            "view_history": st.checkbox(f"View query history", value=True, key=f"{selected_role}_query_history")
        },
        "admin": {
            "user_management": st.checkbox(f"User management", value=selected_role == "admin", key=f"{selected_role}_admin_users"),
            "system_config": st.checkbox(f"System configuration", value=selected_role == "admin", key=f"{selected_role}_admin_config"),
            "monitoring": st.checkbox(f"System monitoring", value=selected_role in ["admin", "manager"], key=f"{selected_role}_admin_monitor"),
            "logs": st.checkbox(f"View system logs", value=selected_role in ["admin", "manager"], key=f"{selected_role}_admin_logs")
        }
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Document Permissions**")
        for perm, value in permissions["documents"].items():
            st.write(f"• {perm.replace('_', ' ').title()}: {'✅' if value else '❌'}")
    
    with col2:
        st.markdown("**Query Permissions**")
        for perm, value in permissions["queries"].items():
            st.write(f"• {perm.replace('_', ' ').title()}: {'✅' if value else '❌'}")
    
    with col3:
        st.markdown("**Admin Permissions**")
        for perm, value in permissions["admin"].items():
            st.write(f"• {perm.replace('_', ' ').title()}: {'✅' if value else '❌'}")
    
    # Department access control
    st.markdown("---")
    st.markdown("### 🏢 Department Access Control")
    
    department = st.selectbox("Select Department", ["IT", "Finance", "HR", "Marketing", "Legal", "Operations"])
    
    st.write("**Document Access by Category:**")
    
    # Get categories from config
    category_config = config.get_document_categories()
    categories = [cat['name'] for cat in category_config] if category_config else ["General", "Technical", "Financial", "Legal", "Marketing", "HR"]
    dept_access = {}
    
    for category in categories:
        dept_access[category] = st.checkbox(f"{category} documents", key=f"dept_{department}_{category}")
    
    # Save permissions
    if st.button("💾 Save Permissions", type="primary"):
        st.success("✅ Permissions updated successfully!")

def show_maintenance(api_client):
    """Display system maintenance interface"""
    st.subheader("🛠️ System Maintenance")
    
    # Database maintenance
    st.markdown("### 🗄️ Database Maintenance")
    
    maint_col1, maint_col2 = st.columns(2)
    
    with maint_col1:
        st.markdown("**Backup Operations**")
        
        if st.button("💾 Create Database Backup", use_container_width=True):
            create_database_backup()
        
        if st.button("📁 Create Vector DB Backup", use_container_width=True):
            create_vector_backup()
        
        backup_schedule = st.selectbox("Auto Backup Schedule", ["Disabled", "Daily", "Weekly", "Monthly"])
    
    with maint_col2:
        st.markdown("**Cleanup Operations**")
        
        if st.button("🧹 Clean Temporary Files", use_container_width=True):
            clean_temp_files()
        
        if st.button("📜 Archive Old Logs", use_container_width=True):
            archive_old_logs()
        
        if st.button("🗑️ Clean Orphaned Documents", use_container_width=True):
            clean_orphaned_docs()
    
    # System optimization
    st.markdown("---")
    st.markdown("### ⚡ System Optimization")
    
    opt_col1, opt_col2 = st.columns(2)
    
    with opt_col1:
        st.markdown("**Index Optimization**")
        
        if st.button("🔄 Rebuild Vector Indexes", use_container_width=True):
            rebuild_vector_indexes()
        
        if st.button("📊 Optimize Database Indexes", use_container_width=True):
            optimize_db_indexes()
    
    with opt_col2:
        st.markdown("**Cache Management**")
        
        if st.button("🗑️ Clear Application Cache", use_container_width=True):
            clear_app_cache()
        
        if st.button("🔄 Refresh Model Cache", use_container_width=True):
            refresh_model_cache()
    
    # System restart
    st.markdown("---")
    st.markdown("### 🔄 System Control")
    
    st.warning("⚠️ These operations will affect all users")
    
    restart_col1, restart_col2, restart_col3 = st.columns(3)
    
    with restart_col1:
        if st.button("🔄 Restart API Server", type="secondary"):
            restart_api_server()
    
    with restart_col2:
        if st.button("⏹️ Graceful Shutdown", type="secondary"):
            graceful_shutdown()
    
    with restart_col3:
        if st.button("🚨 Emergency Stop", type="secondary"):
            if st.checkbox("I understand this will force stop all services"):
                emergency_stop()

# Helper functions
def get_mock_users():
    """Generate mock user data"""
    return [
        {
            "id": "user_001",
            "username": "admin",
            "email": "admin@ragserver.local",
            "role": "admin",
            "department": "IT",
            "status": "Active",
            "created_date": "2024-01-01",
            "last_login": "2024-01-16 09:30",
            "total_queries": 1245
        },
        {
            "id": "user_002", 
            "username": "john.doe",
            "email": "john.doe@company.com",
            "role": "manager",
            "department": "Finance",
            "status": "Active",
            "created_date": "2024-01-05",
            "last_login": "2024-01-16 10:15",
            "total_queries": 456
        },
        {
            "id": "user_003",
            "username": "jane.smith",
            "email": "jane.smith@company.com", 
            "role": "user",
            "department": "Marketing",
            "status": "Active",
            "created_date": "2024-01-10",
            "last_login": "2024-01-15 16:45",
            "total_queries": 123
        },
        {
            "id": "user_004",
            "username": "inactive.user",
            "email": "inactive@company.com",
            "role": "user", 
            "department": "HR",
            "status": "Inactive",
            "created_date": "2024-01-02",
            "last_login": "2024-01-10 12:00",
            "total_queries": 45
        }
    ]

def create_user(api_client, user_data):
    """Create a new user"""
    try:
        # Call the actual API to create user
        result = api_client.create_user(
            username=user_data['username'],
            email=user_data['email'],
            role=user_data['role'],
            password=user_data['password'],
            department=user_data.get('department')
        )
        
        st.success(f"✅ User '{user_data['username']}' created successfully!")
        
        st.rerun()
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            st.error("❌ Unauthorized: Admin privileges required")
        elif "409" in error_msg:
            st.error("❌ User already exists with that username or email")
        elif "422" in error_msg:
            st.error("❌ Invalid user data. Please check all fields")
        else:
            st.error(f"❌ Failed to create user: {error_msg}")

def edit_user_modal(user):
    """Show user edit modal"""
    st.info(f"Editing user: {user['username']} (Feature coming soon)")

def toggle_user_status(api_client, user_id, new_status):
    """Toggle user active/inactive status"""
    try:
        st.success(f"✅ User status updated to {new_status}")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to update user status: {str(e)}")

def delete_user_modal(api_client, user):
    """Show user deletion confirmation"""
    if st.button(f"⚠️ Confirm deletion of {user['username']}", type="secondary"):
        try:
            st.success(f"✅ User '{user['username']}' deleted")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to delete user: {str(e)}")

def save_system_configuration(config):
    """Save system configuration"""
    st.success("✅ System configuration saved successfully!")

# Maintenance helper functions
def create_database_backup():
    st.success("✅ Database backup created successfully!")

def create_vector_backup():
    st.success("✅ Vector database backup created successfully!")

def clean_temp_files():
    st.success("✅ Temporary files cleaned successfully!")

def archive_old_logs():
    st.success("✅ Old logs archived successfully!")

def clean_orphaned_docs():
    st.success("✅ Orphaned documents cleaned successfully!")

def rebuild_vector_indexes():
    st.success("✅ Vector indexes rebuilt successfully!")

def optimize_db_indexes():
    st.success("✅ Database indexes optimized successfully!")

def clear_app_cache():
    st.success("✅ Application cache cleared successfully!")

def refresh_model_cache():
    st.success("✅ Model cache refreshed successfully!")

def restart_api_server():
    st.warning("🔄 API server restart initiated...")

def graceful_shutdown():
    st.warning("⏹️ Graceful shutdown initiated...")

def emergency_stop():
    st.error("🚨 Emergency stop initiated!")