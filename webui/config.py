"""
Configuration module for RAG Server Web UI
Handles application settings from YAML file and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WebUIConfig:
    """Configuration class for the web UI"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_file = config_file or (Path(__file__).parent / "settings.yaml")
        self.settings = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file and environment variables"""
        try:
            # Load YAML settings
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.settings = yaml.safe_load(f) or {}
                logger.info(f"Loaded settings from {self.config_file}")
            else:
                logger.warning(f"Settings file not found: {self.config_file}. Using defaults.")
                self.settings = self._get_default_settings()
        except Exception as e:
            logger.error(f"Error loading settings: {e}. Using defaults.")
            self.settings = self._get_default_settings()
        
        # Override with environment variables
        self._apply_env_overrides()
        
        # Set attributes for backward compatibility
        self._set_legacy_attributes()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings if YAML file is not available"""
        return {
            "app": {
                "title": "RAG Server",
                "icon": "🤖",
                "theme": "light",
                "version": "1.0.0"
            },
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "auth": {
                "session_timeout": 3600,
                "jwt_secret_key": "demo_secret_key",
                "enable_demo_users": True,
                "demo_users": {
                    "admin": {
                        "password": "admin123",
                        "role": "admin",
                        "email": "admin@ragserver.local",
                        "department": "IT"
                    }
                }
            },
            "upload": {
                "max_file_size_mb": 100,
                "allowed_file_types": ["txt", "pdf", "docx", "xlsx", "md"]
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Load from main .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # API settings - use main server config when possible
        api_host = os.getenv("API_HOST", "0.0.0.0")
        api_port = os.getenv("API_PORT", "8000")
        self.settings.setdefault("api", {})["base_url"] = os.getenv("RAG_API_BASE_URL", f"http://{api_host}:{api_port}")
        
        if os.getenv("RAG_API_TIMEOUT"):
            self.settings.setdefault("api", {})["timeout"] = int(os.getenv("RAG_API_TIMEOUT"))
        
        # App settings
        if os.getenv("RAG_APP_TITLE"):
            self.settings.setdefault("app", {})["title"] = os.getenv("RAG_APP_TITLE")
        if os.getenv("RAG_APP_ICON"):
            self.settings.setdefault("app", {})["icon"] = os.getenv("RAG_APP_ICON")
        if os.getenv("RAG_THEME"):
            self.settings.setdefault("app", {})["theme"] = os.getenv("RAG_THEME")
        
        # Auth settings - use main server secret key
        main_secret = os.getenv("SECRET_KEY")
        jwt_secret = os.getenv("JWT_SECRET_KEY", main_secret)
        if jwt_secret:
            self.settings.setdefault("auth", {})["jwt_secret_key"] = jwt_secret
            
        if os.getenv("SESSION_TIMEOUT"):
            self.settings.setdefault("auth", {})["session_timeout"] = int(os.getenv("SESSION_TIMEOUT"))
        if os.getenv("ENABLE_DEMO_USERS"):
            self.settings.setdefault("auth", {})["enable_demo_users"] = os.getenv("ENABLE_DEMO_USERS").lower() == "true"
        
        # Upload settings
        if os.getenv("MAX_UPLOAD_SIZE_MB"):
            self.settings.setdefault("upload", {})["max_file_size_mb"] = int(os.getenv("MAX_UPLOAD_SIZE_MB"))
        if os.getenv("ALLOWED_FILE_TYPES"):
            self.settings.setdefault("upload", {})["allowed_file_types"] = os.getenv("ALLOWED_FILE_TYPES").split(",")
        
        # Debug settings - use main server debug flag
        main_debug = os.getenv("DEBUG")
        if main_debug:
            self.settings.setdefault("debug", {})["enabled"] = main_debug.lower() == "true"
        if os.getenv("LOG_LEVEL"):
            self.settings.setdefault("debug", {})["log_level"] = os.getenv("LOG_LEVEL")
        
        # Sync LLM providers from main config
        self._sync_llm_providers_from_env()
    
    def _sync_llm_providers_from_env(self):
        """Sync LLM providers from environment variables"""
        # Get main LLM provider from .env
        main_provider = os.getenv("LLM_PROVIDER", "ollama")
        main_model = os.getenv("LLM_MODEL", "gemma3:27b-it-q8_0")
        
        # Check if we have YAML config for LLM providers
        if "llm_providers" not in self.settings:
            self.settings["llm_providers"] = {}
        
        providers = self.settings["llm_providers"]
        
        # Update enabled status based on available API keys
        if "openai" in providers:
            providers["openai"]["enabled"] = bool(os.getenv("OPENAI_API_KEY"))
            if main_provider == "openai":
                providers["openai"]["default_model"] = main_model
        
        if "anthropic" in providers:
            providers["anthropic"]["enabled"] = bool(os.getenv("ANTHROPIC_API_KEY"))
            if main_provider == "anthropic":
                providers["anthropic"]["default_model"] = main_model
        
        if "google" in providers:
            providers["google"]["enabled"] = bool(os.getenv("GOOGLE_API_KEY"))
            if main_provider == "google":
                providers["google"]["default_model"] = main_model
        
        if "ollama" in providers:
            providers["ollama"]["enabled"] = True  # Ollama doesn't need API key
            if main_provider == "ollama":
                providers["ollama"]["default_model"] = main_model
                # Add current model to ollama models list if not present
                if main_model not in providers["ollama"].get("models", []):
                    providers["ollama"].setdefault("models", []).append(main_model)
    
    def _set_legacy_attributes(self):
        """Set legacy attributes for backward compatibility"""
        # API Configuration
        self.API_BASE_URL = self.get("api.base_url", "http://localhost:8000")
        self.API_TIMEOUT = self.get("api.timeout", 30)
        
        # Authentication Configuration
        self.JWT_SECRET_KEY = self.get("auth.jwt_secret_key", "demo_secret_key")
        self.SESSION_TIMEOUT = self.get("auth.session_timeout", 3600)
        
        # UI Configuration
        self.APP_TITLE = self.get("app.title", "RAG Server")
        self.APP_ICON = self.get("app.icon", "🤖")
        self.THEME = self.get("app.theme", "light")
        
        # Upload Configuration
        self.MAX_UPLOAD_SIZE_MB = self.get("upload.max_file_size_mb", 100)
        self.ALLOWED_FILE_TYPES = self.get("upload.allowed_file_types", ["txt", "pdf", "docx", "xlsx", "md"])
        
        # Demo Configuration
        self.ENABLE_DEMO_USERS = self.get("auth.enable_demo_users", True)
        self.DEMO_MODE = self.get("auth.enable_demo_users", True)  # Alias for backward compatibility
        
        # Debug Configuration
        self.DEBUG = self.get("debug.enabled", False)
        self.LOG_LEVEL = self.get("debug.log_level", "INFO")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = key.split('.')
        settings = self.settings
        
        for k in keys[:-1]:
            if k not in settings:
                settings[k] = {}
            settings = settings[k]
        
        settings[keys[-1]] = value
    
    def save(self):
        """Save current settings to YAML file"""
        try:
            # Add metadata
            self.settings['_metadata'] = {
                'last_updated': datetime.now().isoformat(),
                'version': self.get('app.version', '1.0.0')
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.settings, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Settings saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def reload(self):
        """Reload settings from file"""
        self.load_config()
        
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit-specific configuration"""
        return {
            "page_title": self.APP_TITLE,
            "page_icon": self.APP_ICON,
            "layout": "wide",
            "initial_sidebar_state": "expanded"
        }
    
    def get_demo_users(self) -> Dict[str, Dict[str, str]]:
        """Get demo user credentials"""
        if not self.ENABLE_DEMO_USERS:
            return {}
        
        return self.get("auth.demo_users", {})
    
    def get_llm_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get LLM provider configurations"""
        return self.get("llm_providers", {})
    
    def get_enabled_llm_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get only enabled LLM providers"""
        providers = self.get_llm_providers()
        return {k: v for k, v in providers.items() if v.get("enabled", True)}
    
    def get_document_categories(self) -> List[Dict[str, Any]]:
        """Get document categories"""
        return self.get("document_categories", [])
    
    def get_access_levels(self) -> List[Dict[str, Any]]:
        """Get access levels"""
        return self.get("access_levels", [])
    
    def get_user_roles(self) -> Dict[str, Dict[str, Any]]:
        """Get user roles"""
        return self.get("user_roles", {})
    
    def get_query_defaults(self) -> Dict[str, Any]:
        """Get default query parameters"""
        return self.get("query_defaults", {
            "k": 5,
            "temperature": 0.7,
            "max_tokens": 1000,
            "similarity_threshold": 0.7
        })
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get default API headers"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"RAG-WebUI/{self.get('app.version', '1.0.0')}"
        }
        
        # Add auth token if available
        if hasattr(st.session_state, "jwt_token") and st.session_state.jwt_token:
            headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
        
        return headers
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags"""
        return self.get("features", {})

# Global configuration instance
config = WebUIConfig()

# Backward compatibility functions for existing code
def get_default_query_params():
    """Get default query parameters (backward compatibility)"""
    return config.get_query_defaults()

def get_llm_providers():
    """Get LLM providers (backward compatibility)"""
    return config.get_enabled_llm_providers()

def get_document_categories():
    """Get document categories (backward compatibility)"""
    categories = config.get_document_categories()
    if categories:
        return [cat.get("name", cat) if isinstance(cat, dict) else cat for cat in categories]
    return ["General", "Technical", "Financial", "Legal", "Marketing", "HR", "Operations"]

def get_access_levels():
    """Get access levels (backward compatibility)"""
    levels = config.get_access_levels()
    if levels:
        return [level.get("name", level) if isinstance(level, dict) else level for level in levels]
    return ["Public", "Internal", "Restricted", "Confidential"]

def get_user_roles():
    """Get user roles (backward compatibility)"""
    return config.get_user_roles()

# Legacy constants for backward compatibility (deprecated - use functions above)
DEFAULT_QUERY_PARAMS = get_default_query_params()
LLM_PROVIDERS = get_llm_providers()
DOCUMENT_CATEGORIES = get_document_categories()
ACCESS_LEVELS = get_access_levels()
USER_ROLES = get_user_roles()