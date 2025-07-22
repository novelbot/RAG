"""
Multi-database driver implementation for SQLAlchemy.
Supports MySQL, PostgreSQL, Oracle, SQL Server, and MariaDB.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, ConfigurationError
from src.core.logging import LoggerMixin


class DatabaseDriver(ABC, LoggerMixin):
    """Abstract base class for database drivers."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate database configuration."""
        pass
    
    @abstractmethod
    def build_connection_url(self) -> str:
        """Build database connection URL."""
        pass
    
    @abstractmethod
    def get_engine_options(self) -> Dict[str, Any]:
        """Get database-specific engine options."""
        pass
    
    @abstractmethod
    def get_health_check_query(self) -> str:
        """Get database-specific health check query."""
        pass
    
    @abstractmethod
    def get_version_query(self) -> str:
        """Get database version query."""
        pass
    
    def test_connection(self, engine: Engine) -> bool:
        """Test database connection with specific query."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text(self.get_health_check_query()))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


class PostgreSQLDriver(DatabaseDriver):
    """PostgreSQL database driver."""
    
    def _validate_config(self) -> None:
        """Validate PostgreSQL configuration."""
        if not self.config.user:
            raise ConfigurationError("PostgreSQL user is required")
        if not self.config.name:
            raise ConfigurationError("PostgreSQL database name is required")
    
    def build_connection_url(self) -> str:
        """Build PostgreSQL connection URL."""
        if self.config.password:
            return f"postgresql+psycopg2://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}"
        else:
            return f"postgresql+psycopg2://{self.config.user}@{self.config.host}:{self.config.port}/{self.config.name}"
    
    def get_engine_options(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific engine options."""
        return {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'connect_args': {
                'connect_timeout': 10,
                'application_name': 'rag_server'
            }
        }
    
    def get_health_check_query(self) -> str:
        """Get PostgreSQL health check query."""
        return "SELECT 1"
    
    def get_version_query(self) -> str:
        """Get PostgreSQL version query."""
        return "SELECT version()"


class MySQLDriver(DatabaseDriver):
    """MySQL database driver."""
    
    def _validate_config(self) -> None:
        """Validate MySQL configuration."""
        if not self.config.user:
            raise ConfigurationError("MySQL user is required")
        if not self.config.name:
            raise ConfigurationError("MySQL database name is required")
    
    def build_connection_url(self) -> str:
        """Build MySQL connection URL."""
        if self.config.password:
            return f"mysql+pymysql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}"
        else:
            return f"mysql+pymysql://{self.config.user}@{self.config.host}:{self.config.port}/{self.config.name}"
    
    def get_engine_options(self) -> Dict[str, Any]:
        """Get MySQL-specific engine options."""
        return {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': 3600,  # MySQL closes idle connections after 8 hours
            'pool_pre_ping': True,
            'connect_args': {
                'connect_timeout': 10,
                'charset': 'utf8mb4',
                'use_unicode': True,
                'autocommit': False
            }
        }
    
    def get_health_check_query(self) -> str:
        """Get MySQL health check query."""
        return "SELECT 1"
    
    def get_version_query(self) -> str:
        """Get MySQL version query."""
        return "SELECT VERSION()"


class MariaDBDriver(DatabaseDriver):
    """MariaDB database driver (similar to MySQL)."""
    
    def _validate_config(self) -> None:
        """Validate MariaDB configuration."""
        if not self.config.user:
            raise ConfigurationError("MariaDB user is required")
        if not self.config.name:
            raise ConfigurationError("MariaDB database name is required")
    
    def build_connection_url(self) -> str:
        """Build MariaDB connection URL."""
        if self.config.password:
            return f"mysql+pymysql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}"
        else:
            return f"mysql+pymysql://{self.config.user}@{self.config.host}:{self.config.port}/{self.config.name}"
    
    def get_engine_options(self) -> Dict[str, Any]:
        """Get MariaDB-specific engine options."""
        return {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'connect_args': {
                'connect_timeout': 10,
                'charset': 'utf8mb4',
                'use_unicode': True,
                'autocommit': False
            }
        }
    
    def get_health_check_query(self) -> str:
        """Get MariaDB health check query."""
        return "SELECT 1"
    
    def get_version_query(self) -> str:
        """Get MariaDB version query."""
        return "SELECT VERSION()"


class OracleDriver(DatabaseDriver):
    """Oracle database driver."""
    
    def _validate_config(self) -> None:
        """Validate Oracle configuration."""
        if not self.config.user:
            raise ConfigurationError("Oracle user is required")
        if not self.config.name:
            raise ConfigurationError("Oracle database name/service is required")
    
    def build_connection_url(self) -> str:
        """Build Oracle connection URL."""
        if self.config.password:
            return f"oracle+cx_oracle://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}"
        else:
            return f"oracle+cx_oracle://{self.config.user}@{self.config.host}:{self.config.port}/{self.config.name}"
    
    def get_engine_options(self) -> Dict[str, Any]:
        """Get Oracle-specific engine options."""
        return {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'connect_args': {
                'threaded': True,
                'events': True
            }
        }
    
    def get_health_check_query(self) -> str:
        """Get Oracle health check query."""
        return "SELECT 1 FROM DUAL"
    
    def get_version_query(self) -> str:
        """Get Oracle version query."""
        return "SELECT * FROM V$VERSION WHERE BANNER LIKE 'Oracle%'"


class SQLServerDriver(DatabaseDriver):
    """SQL Server database driver."""
    
    def _validate_config(self) -> None:
        """Validate SQL Server configuration."""
        if not self.config.user:
            raise ConfigurationError("SQL Server user is required")
        if not self.config.name:
            raise ConfigurationError("SQL Server database name is required")
    
    def build_connection_url(self) -> str:
        """Build SQL Server connection URL."""
        if self.config.password:
            return f"mssql+pyodbc://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.name}?driver=ODBC+Driver+17+for+SQL+Server"
        else:
            return f"mssql+pyodbc://{self.config.user}@{self.config.host}:{self.config.port}/{self.config.name}?driver=ODBC+Driver+17+for+SQL+Server"
    
    def get_engine_options(self) -> Dict[str, Any]:
        """Get SQL Server-specific engine options."""
        return {
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_timeout': self.config.pool_timeout,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'connect_args': {
                'timeout': 10,
                'autocommit': False
            }
        }
    
    def get_health_check_query(self) -> str:
        """Get SQL Server health check query."""
        return "SELECT 1"
    
    def get_version_query(self) -> str:
        """Get SQL Server version query."""
        return "SELECT @@VERSION"


class DatabaseDriverFactory:
    """Factory for creating database drivers."""
    
    _drivers = {
        'postgresql': PostgreSQLDriver,
        'mysql': MySQLDriver,
        'mariadb': MariaDBDriver,
        'oracle': OracleDriver,
        'mssql': SQLServerDriver,
        'sqlserver': SQLServerDriver,
    }
    
    @classmethod
    def create_driver(cls, config: DatabaseConfig) -> DatabaseDriver:
        """
        Create database driver based on configuration.
        
        Args:
            config: Database configuration
            
        Returns:
            DatabaseDriver instance
            
        Raises:
            ConfigurationError: If driver is not supported
        """
        driver_name = config.driver.lower()
        
        if driver_name not in cls._drivers:
            raise ConfigurationError(f"Unsupported database driver: {driver_name}")
        
        driver_class = cls._drivers[driver_name]
        return driver_class(config)
    
    @classmethod
    def get_supported_drivers(cls) -> List[str]:
        """Get list of supported database drivers."""
        return list(cls._drivers.keys())
    
    @classmethod
    def is_driver_supported(cls, driver_name: str) -> bool:
        """Check if driver is supported."""
        return driver_name.lower() in cls._drivers


class MultiDatabaseManager:
    """Manager for multiple database connections."""
    
    def __init__(self):
        self._connections: Dict[str, DatabaseDriver] = {}
        self._engines: Dict[str, Engine] = {}
    
    def add_database(self, name: str, config: DatabaseConfig) -> None:
        """
        Add database connection.
        
        Args:
            name: Database connection name
            config: Database configuration
        """
        try:
            # Create driver
            driver = DatabaseDriverFactory.create_driver(config)
            
            # Create engine
            connection_url = driver.build_connection_url()
            engine_options = driver.get_engine_options()
            engine = create_engine(connection_url, **engine_options)
            
            # Store connection
            self._connections[name] = driver
            self._engines[name] = engine
            
            logger.info(f"Added database connection: {name} ({config.driver})")
            
        except Exception as e:
            logger.error(f"Failed to add database {name}: {e}")
            raise DatabaseError(f"Failed to add database {name}: {e}")
    
    def get_engine(self, name: str) -> Engine:
        """
        Get database engine by name.
        
        Args:
            name: Database connection name
            
        Returns:
            SQLAlchemy engine
        """
        if name not in self._engines:
            raise DatabaseError(f"Database connection not found: {name}")
        
        return self._engines[name]
    
    def get_driver(self, name: str) -> DatabaseDriver:
        """
        Get database driver by name.
        
        Args:
            name: Database connection name
            
        Returns:
            Database driver
        """
        if name not in self._connections:
            raise DatabaseError(f"Database connection not found: {name}")
        
        return self._connections[name]
    
    def test_connection(self, name: str) -> bool:
        """
        Test database connection.
        
        Args:
            name: Database connection name
            
        Returns:
            bool: True if connection is healthy
        """
        try:
            engine = self.get_engine(name)
            driver = self.get_driver(name)
            return driver.test_connection(engine)
        except Exception as e:
            logger.error(f"Connection test failed for {name}: {e}")
            return False
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test all database connections.
        
        Returns:
            Dict with connection names and their status
        """
        results = {}
        for name in self._connections.keys():
            results[name] = self.test_connection(name)
        return results
    
    def remove_database(self, name: str) -> None:
        """
        Remove database connection.
        
        Args:
            name: Database connection name
        """
        if name in self._engines:
            self._engines[name].dispose()
            del self._engines[name]
        
        if name in self._connections:
            del self._connections[name]
        
        logger.info(f"Removed database connection: {name}")
    
    def close_all(self) -> None:
        """Close all database connections."""
        for name in list(self._connections.keys()):
            self.remove_database(name)
    
    def get_connection_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all connections.
        
        Returns:
            Dict with connection information
        """
        info = {}
        for name, driver in self._connections.items():
            info[name] = {
                'driver': driver.config.driver,
                'host': driver.config.host,
                'port': driver.config.port,
                'database': driver.config.name,
                'user': driver.config.user,
                'pool_size': driver.config.pool_size,
                'max_overflow': driver.config.max_overflow,
                'connected': self.test_connection(name)
            }
        return info