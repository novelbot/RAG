"""
Configuration Management Page
Allows administrators to modify system settings dynamically
"""

import streamlit as st
import yaml
from typing import Dict, Any, List
import json

from webui.config import config
from webui.api_client import get_api_client

def render_config_manager():
    """Render the configuration management interface"""
    
    st.title("⚙️ Configuration Management")
    
    # Check if user is admin
    if not _is_admin_user():
        st.error("🚫 Access Denied: Administrator privileges required")
        return
    
    # Configuration tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎨 Application", "🔌 API & Auth", "📤 Upload", 
        "🤖 LLM Providers", "🔧 Advanced"
    ])
    
    with tab1:
        _render_app_settings()
    
    with tab2:
        _render_api_auth_settings()
    
    with tab3:
        _render_upload_settings()
    
    with tab4:
        _render_llm_provider_settings()
    
    with tab5:
        _render_advanced_settings()

def _is_admin_user() -> bool:
    """Check if current user has admin privileges"""
    if "user_info" not in st.session_state:
        return False
    
    user_role = st.session_state.user_info.get("role", "")
    return user_role == "admin"

def _render_app_settings():
    """Render application settings"""
    st.subheader("📱 Application Settings")
    
    with st.form("app_settings"):
        # App Title and Icon
        col1, col2 = st.columns([2, 1])
        
        with col1:
            app_title = st.text_input(
                "Application Title",
                value=config.get("app.title", "RAG Server"),
                help="Title displayed in the browser and header"
            )
        
        with col2:
            app_icon = st.text_input(
                "App Icon",
                value=config.get("app.icon", "🤖"),
                help="Emoji used as the app icon"
            )
        
        # Theme and Version
        col3, col4 = st.columns(2)
        
        with col3:
            app_theme = st.selectbox(
                "Default Theme",
                options=["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(config.get("app.theme", "light")),
                help="Default theme for new users"
            )
        
        with col4:
            app_version = st.text_input(
                "Version",
                value=config.get("app.version", "1.0.0"),
                help="Application version"
            )
        
        # UI Settings
        st.subheader("🖼️ UI Settings")
        
        col5, col6 = st.columns(2)
        
        with col5:
            items_per_page = st.number_input(
                "Items Per Page",
                min_value=5,
                max_value=100,
                value=config.get("ui.items_per_page", 20),
                help="Number of items to show per page in lists"
            )
        
        with col6:
            max_query_history = st.number_input(
                "Max Query History",
                min_value=10,
                max_value=1000,
                value=config.get("ui.max_query_history", 100),
                help="Maximum number of queries to keep in history"
            )
        
        # Feature flags
        st.subheader("🎛️ Feature Flags")
        
        col7, col8 = st.columns(2)
        
        with col7:
            enable_dark_mode = st.checkbox(
                "Enable Dark Mode Toggle",
                value=config.get("ui.enable_dark_mode", True),
                help="Allow users to switch between light/dark themes"
            )
            
            enable_file_preview = st.checkbox(
                "Enable File Preview",
                value=config.get("ui.enable_file_preview", True),
                help="Show file previews in document manager"
            )
        
        with col8:
            enable_query_history = st.checkbox(
                "Enable Query History",
                value=config.get("ui.enable_query_history", True),
                help="Save and display user query history"
            )
            
            show_advanced_options = st.checkbox(
                "Show Advanced Options",
                value=config.get("ui.show_advanced_options", False),
                help="Show advanced options in query interface"
            )
        
        submitted = st.form_submit_button("💾 Save App Settings")
        
        if submitted:
            # Update configuration
            config.set("app.title", app_title)
            config.set("app.icon", app_icon)
            config.set("app.theme", app_theme)
            config.set("app.version", app_version)
            config.set("ui.items_per_page", items_per_page)
            config.set("ui.max_query_history", max_query_history)
            config.set("ui.enable_dark_mode", enable_dark_mode)
            config.set("ui.enable_file_preview", enable_file_preview)
            config.set("ui.enable_query_history", enable_query_history)
            config.set("ui.show_advanced_options", show_advanced_options)
            
            if config.save():
                st.success("✅ Application settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")

def _render_api_auth_settings():
    """Render API and authentication settings"""
    st.subheader("🔌 API Settings")
    
    with st.form("api_auth_settings"):
        # API Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            api_base_url = st.text_input(
                "API Base URL",
                value=config.get("api.base_url", "http://localhost:8000"),
                help="Base URL for the RAG API server"
            )
            
            api_timeout = st.number_input(
                "API Timeout (seconds)",
                min_value=5,
                max_value=300,
                value=config.get("api.timeout", 30),
                help="Timeout for API requests"
            )
        
        with col2:
            retry_attempts = st.number_input(
                "Retry Attempts",
                min_value=0,
                max_value=10,
                value=config.get("api.retry_attempts", 3),
                help="Number of retry attempts for failed API calls"
            )
            
            retry_delay = st.number_input(
                "Retry Delay (seconds)",
                min_value=0.1,
                max_value=10.0,
                value=config.get("api.retry_delay", 1.0),
                step=0.1,
                help="Delay between retry attempts"
            )
        
        # Authentication Configuration
        st.subheader("🔐 Authentication Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            session_timeout = st.number_input(
                "Session Timeout (seconds)",
                min_value=300,
                max_value=86400,  # 24 hours
                value=config.get("auth.session_timeout", 3600),
                help="How long user sessions last"
            )
        
        with col4:
            enable_demo_users = st.checkbox(
                "Enable Demo Users",
                value=config.get("auth.enable_demo_users", True),
                help="Allow login with demo accounts"
            )
        
        # JWT Secret (masked for security)
        jwt_secret = st.text_input(
            "JWT Secret Key",
            value="*" * 20,  # Masked
            type="password",
            help="Secret key for JWT token generation (leave blank to keep current)"
        )
        
        submitted = st.form_submit_button("💾 Save API & Auth Settings")
        
        if submitted:
            # Update configuration
            config.set("api.base_url", api_base_url)
            config.set("api.timeout", api_timeout)
            config.set("api.retry_attempts", retry_attempts)
            config.set("api.retry_delay", retry_delay)
            config.set("auth.session_timeout", session_timeout)
            config.set("auth.enable_demo_users", enable_demo_users)
            
            # Only update JWT secret if a new value is provided
            if jwt_secret and jwt_secret != "*" * 20:
                config.set("auth.jwt_secret_key", jwt_secret)
            
            if config.save():
                st.success("✅ API & Auth settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")

def _render_upload_settings():
    """Render upload settings"""
    st.subheader("📤 File Upload Settings")
    
    with st.form("upload_settings"):
        # File size limit
        max_file_size = st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=1000,
            value=config.get("upload.max_file_size_mb", 100),
            help="Maximum size for uploaded files"
        )
        
        # Allowed file types
        current_types = config.get("upload.allowed_file_types", ["txt", "pdf", "docx", "xlsx", "md"])
        
        st.subheader("📋 Allowed File Types")
        
        # Common file types
        col1, col2, col3 = st.columns(3)
        
        file_types = {}
        
        with col1:
            file_types["txt"] = st.checkbox("📄 Text (.txt)", value="txt" in current_types)
            file_types["md"] = st.checkbox("📝 Markdown (.md)", value="md" in current_types)
            file_types["pdf"] = st.checkbox("📕 PDF (.pdf)", value="pdf" in current_types)
            file_types["docx"] = st.checkbox("📘 Word (.docx)", value="docx" in current_types)
        
        with col2:
            file_types["xlsx"] = st.checkbox("📊 Excel (.xlsx)", value="xlsx" in current_types)
            file_types["pptx"] = st.checkbox("📈 PowerPoint (.pptx)", value="pptx" in current_types)
            file_types["csv"] = st.checkbox("📋 CSV (.csv)", value="csv" in current_types)
            file_types["json"] = st.checkbox("🔗 JSON (.json)", value="json" in current_types)
        
        with col3:
            file_types["xml"] = st.checkbox("🏷️ XML (.xml)", value="xml" in current_types)
            file_types["html"] = st.checkbox("🌐 HTML (.html)", value="html" in current_types)
            file_types["rtf"] = st.checkbox("📄 RTF (.rtf)", value="rtf" in current_types)
            file_types["odt"] = st.checkbox("📄 ODT (.odt)", value="odt" in current_types)
        
        # Custom file types
        custom_types = st.text_input(
            "Custom File Types",
            value=",".join([t for t in current_types if t not in file_types]),
            help="Additional file extensions (comma-separated, e.g., py,js,cpp)"
        )
        
        submitted = st.form_submit_button("💾 Save Upload Settings")
        
        if submitted:
            # Build allowed types list
            allowed_types = [ext for ext, enabled in file_types.items() if enabled]
            
            if custom_types:
                custom_list = [t.strip().lower() for t in custom_types.split(",") if t.strip()]
                allowed_types.extend(custom_list)
            
            # Update configuration
            config.set("upload.max_file_size_mb", max_file_size)
            config.set("upload.allowed_file_types", allowed_types)
            
            if config.save():
                st.success("✅ Upload settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")

def _render_llm_provider_settings():
    """Render LLM provider settings"""
    st.subheader("🤖 LLM Provider Configuration")
    
    providers = config.get_llm_providers()
    
    for provider_id, provider_config in providers.items():
        with st.expander(f"{provider_config.get('name', provider_id)} Settings", expanded=True):
            with st.form(f"provider_{provider_id}"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    enabled = st.checkbox(
                        "Enabled",
                        value=provider_config.get("enabled", True),
                        key=f"enabled_{provider_id}"
                    )
                
                with col2:
                    provider_name = st.text_input(
                        "Display Name",
                        value=provider_config.get("name", provider_id),
                        key=f"name_{provider_id}"
                    )
                
                # Models configuration
                st.subheader("Available Models")
                
                current_models = provider_config.get("models", [])
                default_model = provider_config.get("default_model", "")
                
                # Display current models with edit capability
                updated_models = []
                for i, model in enumerate(current_models):
                    col_model, col_default, col_remove = st.columns([3, 1, 1])
                    
                    with col_model:
                        model_name = st.text_input(
                            f"Model {i+1}",
                            value=model,
                            key=f"model_{provider_id}_{i}"
                        )
                        if model_name:
                            updated_models.append(model_name)
                    
                    with col_default:
                        is_default = st.checkbox(
                            "Default",
                            value=(model == default_model),
                            key=f"default_{provider_id}_{i}"
                        )
                        if is_default:
                            default_model = model_name
                    
                    with col_remove:
                        if st.button("🗑️", key=f"remove_{provider_id}_{i}"):
                            # Model will be removed by not including it
                            pass
                
                # Add new model
                new_model = st.text_input(
                    "Add New Model",
                    key=f"new_model_{provider_id}",
                    placeholder="Enter model name..."
                )
                
                if new_model:
                    updated_models.append(new_model)
                
                submitted = st.form_submit_button(f"💾 Save {provider_config.get('name', provider_id)} Settings")
                
                if submitted:
                    # Update provider configuration
                    config.set(f"llm_providers.{provider_id}.enabled", enabled)
                    config.set(f"llm_providers.{provider_id}.name", provider_name)
                    config.set(f"llm_providers.{provider_id}.models", updated_models)
                    config.set(f"llm_providers.{provider_id}.default_model", default_model)
                    
                    if config.save():
                        st.success(f"✅ {provider_name} settings saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")

def _render_advanced_settings():
    """Render advanced settings"""
    st.subheader("🔧 Advanced Configuration")
    
    # Document Categories Management
    st.subheader("📁 Document Categories")
    _render_category_management()
    
    st.divider()
    
    # Access Levels Management
    st.subheader("🔐 Access Levels")
    _render_access_level_management()
    
    st.divider()
    
    # Query Defaults
    st.subheader("🎯 Query Defaults")
    _render_query_defaults()
    
    st.divider()
    
    # Debug Settings
    st.subheader("🐛 Debug Settings")
    _render_debug_settings()

def _render_category_management():
    """Render document category management"""
    categories = config.get_document_categories()
    
    # Display existing categories
    for i, category in enumerate(categories):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                name = st.text_input(
                    "Name",
                    value=category.get("name", ""),
                    key=f"cat_name_{i}"
                )
            
            with col2:
                description = st.text_input(
                    "Description",
                    value=category.get("description", ""),
                    key=f"cat_desc_{i}"
                )
            
            with col3:
                color = st.color_picker(
                    "Color",
                    value=category.get("color", "#2196F3"),
                    key=f"cat_color_{i}"
                )
            
            with col4:
                if st.button("🗑️", key=f"cat_remove_{i}"):
                    # Remove category
                    categories.pop(i)
                    config.set("document_categories", categories)
                    config.save()
                    st.rerun()
            
            # Update category if changed
            categories[i] = {
                "name": name,
                "description": description,
                "color": color
            }
    
    # Add new category
    st.subheader("➕ Add New Category")
    with st.form("add_category"):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            new_name = st.text_input("Category Name")
        
        with col2:
            new_desc = st.text_input("Description")
        
        with col3:
            new_color = st.color_picker("Color", value="#2196F3")
        
        if st.form_submit_button("➕ Add Category"):
            if new_name:
                categories.append({
                    "name": new_name,
                    "description": new_desc,
                    "color": new_color
                })
                config.set("document_categories", categories)
                if config.save():
                    st.success("✅ Category added successfully!")
                    st.rerun()

def _render_access_level_management():
    """Render access level management"""
    levels = config.get_access_levels()
    
    # Display existing levels
    for i, level in enumerate(levels):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
            
            with col1:
                name = st.text_input(
                    "Name",
                    value=level.get("name", ""),
                    key=f"level_name_{i}"
                )
            
            with col2:
                description = st.text_input(
                    "Description",
                    value=level.get("description", ""),
                    key=f"level_desc_{i}"
                )
            
            with col3:
                level_num = st.number_input(
                    "Level",
                    min_value=0,
                    value=level.get("level", 0),
                    key=f"level_num_{i}"
                )
            
            with col4:
                color = st.color_picker(
                    "Color",
                    value=level.get("color", "#4CAF50"),
                    key=f"level_color_{i}"
                )
            
            with col5:
                if st.button("🗑️", key=f"level_remove_{i}"):
                    # Remove level
                    levels.pop(i)
                    config.set("access_levels", levels)
                    config.save()
                    st.rerun()
            
            # Update level if changed
            levels[i] = {
                "name": name,
                "description": description,
                "level": level_num,
                "color": color
            }
    
    # Add new access level
    st.subheader("➕ Add New Access Level")
    with st.form("add_access_level"):
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            new_name = st.text_input("Level Name")
        
        with col2:
            new_desc = st.text_input("Description")
        
        with col3:
            new_level = st.number_input("Level Number", min_value=0)
        
        with col4:
            new_color = st.color_picker("Color", value="#4CAF50")
        
        if st.form_submit_button("➕ Add Access Level"):
            if new_name:
                levels.append({
                    "name": new_name,
                    "description": new_desc,
                    "level": new_level,
                    "color": new_color
                })
                config.set("access_levels", levels)
                if config.save():
                    st.success("✅ Access level added successfully!")
                    st.rerun()

def _render_query_defaults():
    """Render query default settings"""
    with st.form("query_defaults"):
        col1, col2 = st.columns(2)
        
        with col1:
            k = st.number_input(
                "Default K (Top-K Results)",
                min_value=1,
                max_value=20,
                value=config.get("query_defaults.k", 5),
                help="Number of top results to retrieve"
            )
            
            temperature = st.slider(
                "Default Temperature",
                min_value=0.0,
                max_value=2.0,
                value=config.get("query_defaults.temperature", 0.7),
                step=0.1,
                help="Controls randomness in LLM responses"
            )
        
        with col2:
            max_tokens = st.number_input(
                "Default Max Tokens",
                min_value=100,
                max_value=4000,
                value=config.get("query_defaults.max_tokens", 1000),
                help="Maximum tokens for LLM responses"
            )
            
            similarity_threshold = st.slider(
                "Default Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config.get("query_defaults.similarity_threshold", 0.7),
                step=0.05,
                help="Minimum similarity score for results"
            )
        
        if st.form_submit_button("💾 Save Query Defaults"):
            config.set("query_defaults.k", k)
            config.set("query_defaults.temperature", temperature)
            config.set("query_defaults.max_tokens", max_tokens)
            config.set("query_defaults.similarity_threshold", similarity_threshold)
            
            if config.save():
                st.success("✅ Query defaults saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")

def _render_debug_settings():
    """Render debug settings"""
    with st.form("debug_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            debug_enabled = st.checkbox(
                "Enable Debug Mode",
                value=config.get("debug.enabled", False),
                help="Enable debug logging and features"
            )
            
            log_api_calls = st.checkbox(
                "Log API Calls",
                value=config.get("debug.log_api_calls", False),
                help="Log all API requests and responses"
            )
        
        with col2:
            log_level = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("debug.log_level", "INFO")),
                help="Minimum log level to display"
            )
            
            show_query_time = st.checkbox(
                "Show Query Time",
                value=config.get("debug.show_query_time", True),
                help="Display query execution time in UI"
            )
        
        if st.form_submit_button("💾 Save Debug Settings"):
            config.set("debug.enabled", debug_enabled)
            config.set("debug.log_level", log_level)
            config.set("debug.log_api_calls", log_api_calls)
            config.set("debug.show_query_time", show_query_time)
            
            if config.save():
                st.success("✅ Debug settings saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save settings")

if __name__ == "__main__":
    render_config_manager()