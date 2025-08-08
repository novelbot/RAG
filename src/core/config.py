"""
Configuration management system for the RAG server.
Supports loading from YAML/JSON files and environment variables.
"""

import os
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import field
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from src.embedding.types import EmbeddingProvider, EmbeddingConfig

# Load environment variables from .env file
load_dotenv()


class DatabaseType(Enum):
    """Supported database types"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    MSSQL = "mssql"
    MARIADB = "mariadb"
    SQLITE = "sqlite"


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    database_type: DatabaseType = DatabaseType.MYSQL
    host: str = "localhost"
    port: int = 3306
    database: str = "novelbot"  # renamed from name to database for consistency
    user: str = "root"          # Keep as user for consistency with existing code
    password: str = ""
    driver: str = "mysql+pymysql"  # Add driver field
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    # Backward compatibility properties
    @property
    def name(self) -> str:
        """Backward compatibility for name attribute"""
        return self.database


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
    timeout: Optional[float] = 60.0
    max_retries: int = 3
    custom_headers: Optional[Dict[str, str]] = None
    stream: bool = True  # Add streaming support
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed_providers = ['openai', 'anthropic', 'google', 'ollama']
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v


# EmbeddingConfig is now imported from src.embedding.base


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
    vector_dimension: int = 1024  # Default dimension for embeddings
    
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
    embedding: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig(
        provider=EmbeddingProvider.OLLAMA,
        model="nomic-embed-text"
    ))
    auth: AuthConfig = AuthConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    rag: RAGConfig = RAGConfig()
    access_control: Optional[Any] = None
    
    # RDB connections for multiple databases
    rdb_connections: Dict[str, DatabaseConfig] = {}
    
    # Embedding providers configuration  
    embedding_providers: Dict[str, EmbeddingConfig] = {}


class ConfigManager:
    """Configuration manager for loading and managing app settings from environment variables"""
    
    def __init__(self):
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from environment variables with sensible defaults"""
        if self._config is None:
            self._config = AppConfig()
            self._override_from_env()
            self._setup_embedding_providers()
            self._setup_rdb_connections()
            self._setup_access_control()
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
            self._config.database.database = os.getenv('DB_NAME', self._config.database.database)
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
        
        # Embedding overrides (Context7 MCP pattern)
        if os.getenv('EMBEDDING_PROVIDER'):
            provider_str = os.getenv('EMBEDDING_PROVIDER')
            try:
                self._config.embedding.provider = EmbeddingProvider(provider_str.lower())
                # Set provider-specific base URLs
                if self._config.embedding.provider == EmbeddingProvider.OLLAMA:
                    self._config.embedding.base_url = "http://localhost:11434"
                elif self._config.embedding.provider in [EmbeddingProvider.GOOGLE, EmbeddingProvider.OPENAI]:
                    self._config.embedding.base_url = None
            except ValueError:
                print(f"Warning: Unknown embedding provider in environment: {provider_str}")
        if os.getenv('EMBEDDING_MODEL'):
            self._config.embedding.model = os.getenv('EMBEDDING_MODEL', self._config.embedding.model)
        if os.getenv('EMBEDDING_API_KEY'):
            self._config.embedding.api_key = os.getenv('EMBEDDING_API_KEY', self._config.embedding.api_key)
        
        # RAG overrides - prioritize .env variables, fallback to dynamic detection
        if os.getenv('VECTOR_DIMENSION'):
            self._config.rag.vector_dimension = int(os.getenv('VECTOR_DIMENSION', str(self._config.rag.vector_dimension)))
        else:
            # Try to dynamically detect vector dimension from embedding provider
            try:
                if self._config.embedding.provider == EmbeddingProvider.GOOGLE:
                    from src.embedding.providers.google import GoogleEmbeddingProvider
                    provider = GoogleEmbeddingProvider(self._config.embedding)
                    detected_dim = provider._detect_model_dimensions(self._config.embedding.model)
                    self._config.rag.vector_dimension = detected_dim
                elif self._config.embedding.provider == EmbeddingProvider.OPENAI:
                    # OpenAI model dimension mapping
                    if "text-embedding-3-small" in self._config.embedding.model:
                        self._config.rag.vector_dimension = 1536
                    elif "text-embedding-3-large" in self._config.embedding.model:
                        self._config.rag.vector_dimension = 3072
                    else:
                        self._config.rag.vector_dimension = 1536  # default
                elif self._config.embedding.provider == EmbeddingProvider.OLLAMA:
                    # Ollama model dimension mapping
                    if "nomic-embed-text" in self._config.embedding.model:
                        self._config.rag.vector_dimension = 768
                    else:
                        self._config.rag.vector_dimension = 1024  # default
            except Exception as e:
                # Fallback to default if dynamic detection fails
                print(f"Warning: Failed to detect vector dimension: {e}")
                self._config.rag.vector_dimension = 1024  # safe default
        if os.getenv('RAG_RETRIEVAL_K'):
            self._config.rag.retrieval_k = int(os.getenv('RAG_RETRIEVAL_K', str(self._config.rag.retrieval_k)))
        if os.getenv('RAG_SIMILARITY_THRESHOLD'):
            self._config.rag.similarity_threshold = float(os.getenv('RAG_SIMILARITY_THRESHOLD', str(self._config.rag.similarity_threshold)))
        
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
    
    def _setup_embedding_providers(self) -> None:
        """Setup embedding providers from main embedding config."""
        if not self._config:
            return
        
        # Set up default embedding provider from main config
        if self._config.embedding:
            self._config.embedding_providers = {
                "default": self._config.embedding
            }
        
        # Add additional providers if configured via environment
        # This allows for multiple embedding providers to be configured
        # Example: EMBEDDING_PROVIDER_OPENAI_API_KEY, EMBEDDING_PROVIDER_GOOGLE_API_KEY
        provider_names = ["openai", "google", "ollama", "huggingface"]
        for provider in provider_names:
            env_key = f"EMBEDDING_PROVIDER_{provider.upper()}_API_KEY"
            if os.getenv(env_key):
                # Convert string to EmbeddingProvider enum (Context7 MCP pattern)
                try:
                    provider_enum = EmbeddingProvider(provider)
                except ValueError:
                    continue  # Skip unsupported providers
                
                # Get dimension value with proper fallback
                dimension_env_key = f"EMBEDDING_PROVIDER_{provider.upper()}_DIMENSION"
                dimension_value = os.getenv(dimension_env_key)
                if dimension_value is None:
                    dimensions = self._config.embedding.dimensions
                else:
                    try:
                        dimensions = int(dimension_value)
                    except ValueError:
                        dimensions = self._config.embedding.dimensions
                
                provider_config = EmbeddingConfig(
                    provider=provider_enum,
                    api_key=os.getenv(env_key),
                    model=os.getenv(f"EMBEDDING_PROVIDER_{provider.upper()}_MODEL", self._config.embedding.model),
                    dimensions=dimensions
                )
                self._config.embedding_providers[provider] = provider_config
    
    def _setup_rdb_connections(self) -> None:
        """Setup RDB connections from environment variables."""
        if not self._config:
            return
        
        # Set up default RDB connection if database environment variables are present
        if os.getenv('DB_HOST'):
            self._config.rdb_connections = {
                "default": DatabaseConfig(
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=int(os.getenv('DB_PORT', 3306)),
                    database=os.getenv('DB_NAME', 'ragdb'),
                    user=os.getenv('DB_USER', 'mysql'),
                    password=os.getenv('DB_PASSWORD', ''),
                    driver=os.getenv('DB_DRIVER', 'mysql+pymysql')
                )
            }
    
    def _setup_access_control(self) -> None:
        """Setup access control configuration with lazy import"""
        if not self._config:
            return
        
        try:
            from src.rag.access_control_filter import AccessControlConfig
            self._config.access_control = AccessControlConfig()
        except ImportError:
            # If access control is not available, keep it as None
            self._config.access_control = None
    
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