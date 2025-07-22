"""
Configuration management system for the RAG server.
Supports loading from YAML/JSON files and environment variables.
"""

import os
from typing import Dict, Optional
from dataclasses import field
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Load environment variables from .env file
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 3306
    name: str = "novelbot"
    user: str = "root"
    password: str = ""
    driver: str = "mysql"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    @field_validator('driver')
    @classmethod
    def validate_driver(cls, v):
        allowed_drivers = ['postgresql', 'mysql', 'oracle', 'mssql', 'mariadb']
        if v not in allowed_drivers:
            raise ValueError(f"Driver must be one of: {allowed_drivers}")
        return v


class MilvusConfig(BaseModel):
    """Milvus vector database configuration"""
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    secure: bool = False
    alias: str = "default"
    db_name: str = "default"
    collection_name: str = "rag_vectors"
    index_type: str = "IVF_FLAT"
    metric_type: str = "IP"
    nlist: int = 1024
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMConfig(BaseModel):
    """LLM provider configuration"""
    provider: str = "ollama"
    model: str = "gemma3:27b-it-q8_0"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed_providers = ['openai', 'anthropic', 'google', 'ollama']
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    provider: str = "ollama"
    model: str = "jeffh/intfloat-multilingual-e5-large-instruct:f32"
    api_key: str = ""
    base_url: Optional[str] = None
    dimension: int = 1024
    batch_size: int = 100
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed_providers = ['openai', 'huggingface', 'sentence-transformers', 'ollama']
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v


class AuthConfig(BaseModel):
    """Authentication and authorization configuration"""
    secret_key: str = "your-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    enable_rbac: bool = True
    enable_row_level_security: bool = True


class APIConfig(BaseModel):
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit: int = 100


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time} | {level} | {message}"
    file_path: str = "logs/app.log"
    max_size: str = "10 MB"
    rotation: str = "daily"
    retention: str = "7 days"


class DataSourceConfig(BaseModel):
    """Data source configuration"""
    rdb_connections: Dict[str, DatabaseConfig] = field(default_factory=dict)
    file_paths: list = field(default_factory=list)
    sync_interval: int = 3600  # seconds
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RAGConfig(BaseModel):
    """RAG system configuration"""
    mode: str = "single"  # "single" or "multi"
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    rerank_enabled: bool = True
    ensemble_models: list = field(default_factory=list)
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v not in ['single', 'multi']:
            raise ValueError("Mode must be 'single' or 'multi'")
        return v


class AppConfig(BaseModel):
    """Main application configuration"""
    app_name: str = "RAG Server"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    milvus: MilvusConfig = MilvusConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    auth: AuthConfig = AuthConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    rag: RAGConfig = RAGConfig()


class ConfigManager:
    """Configuration manager for loading and managing app settings from environment variables"""
    
    def __init__(self):
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from environment variables with sensible defaults"""
        if self._config is None:
            self._config = AppConfig()
            self._override_from_env()
        return self._config
    
    def _override_from_env(self) -> None:
        """Override configuration values from environment variables"""
        if not self._config:
            return
        
        # Database overrides
        if os.getenv('DB_HOST'):
            self._config.database.host = os.getenv('DB_HOST', self._config.database.host)
        if os.getenv('DB_PORT'):
            self._config.database.port = int(os.getenv('DB_PORT', str(self._config.database.port)))
        if os.getenv('DB_NAME'):
            self._config.database.name = os.getenv('DB_NAME', self._config.database.name)
        if os.getenv('DB_USER'):
            self._config.database.user = os.getenv('DB_USER', self._config.database.user)
        if os.getenv('DB_PASSWORD'):
            self._config.database.password = os.getenv('DB_PASSWORD', self._config.database.password)
        
        # Milvus overrides
        if os.getenv('MILVUS_HOST'):
            self._config.milvus.host = os.getenv('MILVUS_HOST', self._config.milvus.host)
        if os.getenv('MILVUS_PORT'):
            self._config.milvus.port = int(os.getenv('MILVUS_PORT', str(self._config.milvus.port)))
        if os.getenv('MILVUS_USER'):
            self._config.milvus.user = os.getenv('MILVUS_USER', self._config.milvus.user)
        if os.getenv('MILVUS_PASSWORD'):
            self._config.milvus.password = os.getenv('MILVUS_PASSWORD', self._config.milvus.password)
        
        # LLM overrides
        if os.getenv('LLM_PROVIDER'):
            self._config.llm.provider = os.getenv('LLM_PROVIDER', self._config.llm.provider)
        if os.getenv('LLM_MODEL'):
            self._config.llm.model = os.getenv('LLM_MODEL', self._config.llm.model)
        if os.getenv('LLM_API_KEY'):
            self._config.llm.api_key = os.getenv('LLM_API_KEY', self._config.llm.api_key)
        if os.getenv('OPENAI_API_KEY'):
            self._config.llm.api_key = os.getenv('OPENAI_API_KEY', self._config.llm.api_key)
        if os.getenv('ANTHROPIC_API_KEY'):
            self._config.llm.api_key = os.getenv('ANTHROPIC_API_KEY', self._config.llm.api_key)
        if os.getenv('GOOGLE_API_KEY'):
            self._config.llm.api_key = os.getenv('GOOGLE_API_KEY', self._config.llm.api_key)
        
        # Embedding overrides
        if os.getenv('EMBEDDING_PROVIDER'):
            self._config.embedding.provider = os.getenv('EMBEDDING_PROVIDER', self._config.embedding.provider)
        if os.getenv('EMBEDDING_MODEL'):
            self._config.embedding.model = os.getenv('EMBEDDING_MODEL', self._config.embedding.model)
        if os.getenv('EMBEDDING_API_KEY'):
            self._config.embedding.api_key = os.getenv('EMBEDDING_API_KEY', self._config.embedding.api_key)
        
        # Auth overrides
        if os.getenv('SECRET_KEY'):
            self._config.auth.secret_key = os.getenv('SECRET_KEY', self._config.auth.secret_key)
        
        # API overrides
        if os.getenv('API_HOST'):
            self._config.api.host = os.getenv('API_HOST', self._config.api.host)
        if os.getenv('API_PORT'):
            self._config.api.port = int(os.getenv('API_PORT', str(self._config.api.port)))
        
        # App overrides
        if os.getenv('APP_ENV'):
            self._config.environment = os.getenv('APP_ENV', self._config.environment)
        if os.getenv('DEBUG'):
            debug_value = os.getenv('DEBUG', 'false')
            self._config.debug = debug_value.lower() == 'true'
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.load_config()
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from environment variables"""
        self._config = None
        return self.load_config()


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config_manager.get_config()


def reload_config() -> AppConfig:
    """Reload the global configuration"""
    return config_manager.reload_config()