"""
Settings page for RAG Server Web UI
Handles user preferences, theme settings, and personal configuration
"""

import streamlit as st
import json
from datetime import datetime
from webui.api_client import get_api_client
from webui.auth import require_auth
from webui.config import config

@require_auth
def show():
    """Display the settings page"""
    st.title("⚙️ Settings")
    
    api_client = get_api_client()
    user_info = st.session_state.get("user_info", {})
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profile", 
        "🎨 Appearance", 
        "🔔 Notifications", 
        "🔧 Preferences"
    ])
    
    with tab1:
        show_profile_settings(api_client, user_info)
    
    with tab2:
        show_appearance_settings()
    
    with tab3:
        show_notification_settings()
    
    with tab4:
        show_preference_settings()

def show_profile_settings(api_client, user_info):
    """Display profile settings"""
    st.subheader("👤 Profile Settings")
    
    # Current user information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Profile")
        st.write(f"**Username:** {user_info.get('username', 'N/A')}")
        st.write(f"**Role:** {user_info.get('role', 'user').title()}")
        st.write(f"**Email:** {user_info.get('email', 'N/A')}")
        st.write(f"**User ID:** {user_info.get('user_id', 'N/A')}")
    
    with col2:
        st.markdown("### Profile Picture")
        # Profile picture placeholder
        st.image("https://via.placeholder.com/150/cccccc/969696?text=Profile", width=150)
        uploaded_image = st.file_uploader(
            "Upload new profile picture",
            type=['png', 'jpg', 'jpeg'],
            help="Maximum size: 2MB"
        )
        
        if uploaded_image:
            st.success("Profile picture uploaded! (Feature coming soon)")
    
    st.markdown("---")
    
    # Profile update form
    st.markdown("### Update Profile Information")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_email = st.text_input("Email", value=user_info.get('email', ''))
            new_first_name = st.text_input("First Name", value=user_info.get('first_name', ''))
        
        with col2:
            new_phone = st.text_input("Phone Number", value=user_info.get('phone', ''))
            new_last_name = st.text_input("Last Name", value=user_info.get('last_name', ''))
        
        new_bio = st.text_area("Bio", value=user_info.get('bio', ''), height=100)
        
        # Department and location (if allowed to change)
        if user_info.get('role') in ['admin', 'manager']:
            col3, col4 = st.columns(2)
            
            with col3:
                # Get departments from config
                available_departments = [dept['name'] for dept in config.get_document_categories()]
                if not available_departments:
                    available_departments = ["IT", "Finance", "HR", "Marketing", "Legal", "Operations"]
                
                current_dept = user_info.get('department', available_departments[0])
                dept_index = available_departments.index(current_dept) if current_dept in available_departments else 0
                
                new_department = st.selectbox(
                    "Department",
                    available_departments,
                    index=dept_index
                )
            
            with col4:
                new_location = st.text_input("Location", value=user_info.get('location', ''))
        
        update_profile_btn = st.form_submit_button("💾 Update Profile", type="primary")
        
        if update_profile_btn:
            update_profile_data = {
                "email": new_email,
                "first_name": new_first_name,
                "last_name": new_last_name,
                "phone": new_phone,
                "bio": new_bio
            }
            
            if user_info.get('role') in ['admin', 'manager']:
                update_profile_data.update({
                    "department": new_department,
                    "location": new_location
                })
            
            update_user_profile(api_client, update_profile_data)
    
    # Password change section
    st.markdown("---")
    st.markdown("### Change Password")
    
    with st.form("password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        change_password_btn = st.form_submit_button("🔐 Change Password", type="secondary")
        
        if change_password_btn:
            if not all([current_password, new_password, confirm_password]):
                st.error("Please fill in all password fields")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long")
            else:
                change_user_password(api_client, current_password, new_password)

def show_appearance_settings():
    """Display appearance settings"""
    st.subheader("🎨 Appearance Settings")
    
    # Theme settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Theme")
        
        # Get current theme from session state
        current_theme = st.session_state.get("theme", "light")
        
        theme_options = {
            "Light": "light",
            "Dark": "dark",
            "Auto": "auto"
        }
        
        selected_theme = st.radio(
            "Choose theme",
            options=list(theme_options.keys()),
            index=list(theme_options.values()).index(current_theme)
        )
        
        if st.button("Apply Theme"):
            st.session_state["theme"] = theme_options[selected_theme]
            st.success(f"Theme changed to {selected_theme}")
            # Note: Streamlit doesn't support runtime theme switching natively
            st.info("Theme will be applied on next page refresh")
    
    with col2:
        st.markdown("### Display Options")
        
        # Sidebar preferences
        sidebar_collapsed = st.checkbox(
            "Collapse sidebar by default",
            value=st.session_state.get("sidebar_collapsed", False)
        )
        
        # Page width
        page_width = st.selectbox(
            "Page width",
            ["Wide", "Centered", "Full"],
            index=["Wide", "Centered", "Full"].index(st.session_state.get("page_width", "Wide"))
        )
        
        # Font size
        font_size = st.selectbox(
            "Font size",
            ["Small", "Medium", "Large"],
            index=["Small", "Medium", "Large"].index(st.session_state.get("font_size", "Medium"))
        )
        
        if st.button("Save Display Settings"):
            st.session_state.update({
                "sidebar_collapsed": sidebar_collapsed,
                "page_width": page_width,
                "font_size": font_size
            })
            st.success("Display settings saved!")
    
    st.markdown("---")
    
    # Dashboard customization
    st.markdown("### 📊 Dashboard Customization")
    
    st.write("Select which widgets to show on your dashboard:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_system_status = st.checkbox("System Status", value=True)
        show_recent_queries = st.checkbox("Recent Queries", value=True)
        show_document_stats = st.checkbox("Document Statistics", value=True)
    
    with col2:
        show_query_history = st.checkbox("Query History", value=True)
        show_performance_metrics = st.checkbox("Performance Metrics", value=False)
        show_activity_feed = st.checkbox("Activity Feed", value=True)
    
    with col3:
        show_quick_actions = st.checkbox("Quick Actions", value=True)
        show_resource_usage = st.checkbox("Resource Usage", value=False)
        show_health_status = st.checkbox("Health Status", value=True)
    
    if st.button("Save Dashboard Layout"):
        dashboard_config = {
            "show_system_status": show_system_status,
            "show_recent_queries": show_recent_queries,
            "show_document_stats": show_document_stats,
            "show_query_history": show_query_history,
            "show_performance_metrics": show_performance_metrics,
            "show_activity_feed": show_activity_feed,
            "show_quick_actions": show_quick_actions,
            "show_resource_usage": show_resource_usage,
            "show_health_status": show_health_status
        }
        st.session_state["dashboard_config"] = dashboard_config
        st.success("Dashboard layout saved!")

def show_notification_settings():
    """Display notification settings"""
    st.subheader("🔔 Notification Settings")
    
    # Email notifications
    st.markdown("### 📧 Email Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**System Notifications**")
        email_system_alerts = st.checkbox("System alerts and errors", value=True)
        email_maintenance = st.checkbox("Scheduled maintenance", value=True)
        email_security = st.checkbox("Security alerts", value=True)
        
        st.markdown("**Activity Notifications**")
        email_query_complete = st.checkbox("Query completion", value=False)
        email_document_processed = st.checkbox("Document processing complete", value=True)
        email_weekly_summary = st.checkbox("Weekly activity summary", value=False)
    
    with col2:
        st.markdown("**Admin Notifications** (Admin only)")
        admin_user = st.session_state.get("user_info", {}).get("role") == "admin"
        
        email_new_users = st.checkbox(
            "New user registrations", 
            value=True, 
            disabled=not admin_user,
            help="Available for admin users only"
        )
        email_system_errors = st.checkbox(
            "Critical system errors", 
            value=True,
            disabled=not admin_user,
            help="Available for admin users only"
        )
        email_usage_reports = st.checkbox(
            "Daily usage reports", 
            value=False,
            disabled=not admin_user,
            help="Available for admin users only"
        )
    
    # In-app notifications
    st.markdown("---")
    st.markdown("### 🔔 In-App Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Real-time Notifications**")
        inapp_query_updates = st.checkbox("Query progress updates", value=True)
        inapp_system_status = st.checkbox("System status changes", value=True)
        inapp_document_updates = st.checkbox("Document processing updates", value=True)
    
    with col2:
        st.markdown("**Notification Display**")
        notification_duration = st.selectbox(
            "Notification display duration",
            ["3 seconds", "5 seconds", "10 seconds", "Until dismissed"],
            index=1
        )
        
        notification_position = st.selectbox(
            "Notification position",
            ["Top right", "Top center", "Top left", "Bottom right"],
            index=0
        )
        
        notification_sound = st.checkbox("Play notification sound", value=False)
    
    # Save notification settings
    if st.button("💾 Save Notification Settings", type="primary"):
        notification_config = {
            "email": {
                "system_alerts": email_system_alerts,
                "maintenance": email_maintenance,
                "security": email_security,
                "query_complete": email_query_complete,
                "document_processed": email_document_processed,
                "weekly_summary": email_weekly_summary,
                "new_users": email_new_users if admin_user else False,
                "system_errors": email_system_errors if admin_user else False,
                "usage_reports": email_usage_reports if admin_user else False
            },
            "inapp": {
                "query_updates": inapp_query_updates,
                "system_status": inapp_system_status,
                "document_updates": inapp_document_updates,
                "duration": notification_duration,
                "position": notification_position,
                "sound": notification_sound
            }
        }
        st.session_state["notification_config"] = notification_config
        st.success("Notification settings saved!")

def show_preference_settings():
    """Display user preferences"""
    st.subheader("🔧 User Preferences")
    
    # Query preferences
    st.markdown("### 🔍 Query Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Default Query Settings**")
        
        # Get available LLM providers from config
        available_providers = list(config.get_enabled_llm_providers().keys())
        if not available_providers:
            available_providers = ["openai", "anthropic", "google", "ollama"]
        
        default_llm_provider = st.selectbox(
            "Default LLM Provider",
            available_providers,
            index=0
        )
        
        # Get models for selected provider
        provider_config = config.get_enabled_llm_providers().get(default_llm_provider, {})
        available_models = provider_config.get('models', [])
        if not available_models:
            available_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet-latest", "gemini-2.0-flash-001"]
        
        default_model = st.selectbox(
            "Default Model",
            available_models,
            index=0
        )
        
        default_k_value = st.slider("Default number of retrieved documents", 1, 20, 5)
        default_temperature = st.slider("Default temperature", 0.0, 1.0, 0.7, 0.1)
    
    with col2:
        st.markdown("**Query History**")
        
        save_query_history = st.checkbox("Save query history", value=True)
        max_history_items = st.number_input("Max history items", 10, 1000, 100)
        auto_delete_history = st.selectbox(
            "Auto-delete history after",
            ["Never", "30 days", "90 days", "1 year"],
            index=2
        )
        
        export_format = st.selectbox(
            "Default export format",
            ["JSON", "CSV", "TXT", "Markdown"],
            index=0
        )
    
    # Document preferences
    st.markdown("---")
    st.markdown("### 📄 Document Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Settings**")
        
        # Get document categories from config
        available_categories = [cat['name'] for cat in config.get_document_categories()]
        if not available_categories:
            available_categories = ["General", "Technical", "Financial", "Legal", "Marketing", "HR"]
        
        default_category = st.selectbox(
            "Default document category",
            available_categories,
            index=0
        )
        
        # Get access levels from config
        available_access_levels = [level['name'] for level in config.get_access_levels()]
        if not available_access_levels:
            available_access_levels = ["Public", "Internal", "Restricted", "Confidential"]
        
        default_access_level = st.selectbox(
            "Default access level",
            available_access_levels,
            index=min(1, len(available_access_levels) - 1)
        )
        
        auto_tag_documents = st.checkbox("Enable automatic tagging", value=True)
    
    with col2:
        st.markdown("**Processing Settings**")
        
        notify_processing_complete = st.checkbox("Notify when processing completes", value=True)
        auto_retry_failed = st.checkbox("Auto-retry failed processing", value=False)
        
        max_retry_attempts = st.number_input(
            "Max retry attempts",
            1, 5, 3,
            disabled=not auto_retry_failed
        )
    
    # Language and regional settings
    st.markdown("---")
    st.markdown("### 🌍 Language & Regional Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        language = st.selectbox(
            "Interface Language",
            ["English", "한국어", "日本語", "Español", "Français"],
            index=0
        )
        
        date_format = st.selectbox(
            "Date Format",
            ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD", "DD MMM YYYY"],
            index=2
        )
    
    with col2:
        timezone = st.selectbox(
            "Timezone",
            ["UTC", "Asia/Seoul", "America/New_York", "Europe/London", "Asia/Tokyo"],
            index=1
        )
        
        time_format = st.selectbox(
            "Time Format",
            ["12 Hour (AM/PM)", "24 Hour"],
            index=1
        )
    
    # Advanced preferences
    st.markdown("---")
    st.markdown("### 🔧 Advanced Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance Settings**")
        
        enable_caching = st.checkbox("Enable response caching", value=True)
        cache_duration = st.selectbox(
            "Cache duration",
            ["5 minutes", "15 minutes", "30 minutes", "1 hour"],
            index=2,
            disabled=not enable_caching
        )
        
        concurrent_requests = st.number_input("Max concurrent requests", 1, 10, 3)
    
    with col2:
        st.markdown("**Privacy Settings**")
        
        analytics_tracking = st.checkbox("Enable usage analytics", value=True)
        error_reporting = st.checkbox("Enable error reporting", value=True)
        
        data_retention = st.selectbox(
            "Personal data retention",
            ["3 months", "6 months", "1 year", "2 years"],
            index=2
        )
    
    # Save all preferences
    if st.button("💾 Save All Preferences", type="primary", use_container_width=True):
        preferences_config = {
            "query": {
                "default_llm_provider": default_llm_provider,
                "default_model": default_model,
                "default_k_value": default_k_value,
                "default_temperature": default_temperature,
                "save_history": save_query_history,
                "max_history_items": max_history_items,
                "auto_delete_history": auto_delete_history,
                "export_format": export_format
            },
            "documents": {
                "default_category": default_category,
                "default_access_level": default_access_level,
                "auto_tag": auto_tag_documents,
                "notify_complete": notify_processing_complete,
                "auto_retry": auto_retry_failed,
                "max_retry_attempts": max_retry_attempts
            },
            "regional": {
                "language": language,
                "timezone": timezone,
                "date_format": date_format,
                "time_format": time_format
            },
            "advanced": {
                "enable_caching": enable_caching,
                "cache_duration": cache_duration,
                "concurrent_requests": concurrent_requests,
                "analytics_tracking": analytics_tracking,
                "error_reporting": error_reporting,
                "data_retention": data_retention
            }
        }
        
        st.session_state["user_preferences"] = preferences_config
        st.success("All preferences saved successfully!")
        
        # Show export/import options
        st.markdown("---")
        st.markdown("### 📦 Export/Import Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Settings", use_container_width=True):
                export_user_settings()
        
        with col2:
            imported_settings = st.file_uploader(
                "📥 Import Settings",
                type=['json'],
                help="Upload a previously exported settings file"
            )
            
            if imported_settings:
                import_user_settings(imported_settings)

def update_user_profile(api_client, profile_data):
    """Update user profile information"""
    try:
        # Mock profile update
        st.success("✅ Profile updated successfully!")
        
        # Update session state with new info
        user_info = st.session_state.get("user_info", {})
        user_info.update(profile_data)
        st.session_state["user_info"] = user_info
        
    except Exception as e:
        st.error(f"❌ Failed to update profile: {str(e)}")

def change_user_password(api_client, current_password, new_password):
    """Change user password"""
    try:
        # Mock password change
        st.success("✅ Password changed successfully!")
        st.info("You will need to log in again with your new password")
        
    except Exception as e:
        st.error(f"❌ Failed to change password: {str(e)}")

def export_user_settings():
    """Export user settings to JSON"""
    settings_data = {
        "theme": st.session_state.get("theme", "light"),
        "dashboard_config": st.session_state.get("dashboard_config", {}),
        "notification_config": st.session_state.get("notification_config", {}),
        "user_preferences": st.session_state.get("user_preferences", {}),
        "export_timestamp": datetime.now().isoformat()
    }
    
    settings_json = json.dumps(settings_data, indent=2)
    
    st.download_button(
        label="💾 Download Settings File",
        data=settings_json,
        file_name=f"rag_server_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("Settings exported successfully!")

def import_user_settings(settings_file):
    """Import user settings from JSON file"""
    try:
        settings_data = json.load(settings_file)
        
        # Validate and import settings
        if "theme" in settings_data:
            st.session_state["theme"] = settings_data["theme"]
        
        if "dashboard_config" in settings_data:
            st.session_state["dashboard_config"] = settings_data["dashboard_config"]
        
        if "notification_config" in settings_data:
            st.session_state["notification_config"] = settings_data["notification_config"]
        
        if "user_preferences" in settings_data:
            st.session_state["user_preferences"] = settings_data["user_preferences"]
        
        st.success("✅ Settings imported successfully!")
        st.info("Some settings will take effect after page refresh")
        
    except Exception as e:
        st.error(f"❌ Failed to import settings: {str(e)}")