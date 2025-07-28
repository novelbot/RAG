"""
RDB Configuration Validator - Validates RDB pipeline configurations and connections.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.core.logging import LoggerMixin
from src.core.config import DatabaseConfig, get_config
from src.extraction.base import ExtractionConfig, ExtractionMode, DataFormat
from src.extraction.factory import RDBExtractorFactory
from src.milvus.client import MilvusClient
from src.embedding.manager import EmbeddingManager


class ValidationSeverity(Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""
    
    component: str
    check_name: str
    severity: ValidationSeverity
    status: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "component": self.component,
            "check_name": self.check_name,
            "severity": self.severity.value,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    
    overall_status: bool
    validation_id: str
    timestamp: datetime
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def critical_errors(self) -> List[ValidationResult]:
        """Get critical error results."""
        return [r for r in self.results if r.severity == ValidationSeverity.CRITICAL and not r.status]
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get error results."""
        return [r for r in self.results if r.severity == ValidationSeverity.ERROR and not r.status]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get warning results."""
        return [r for r in self.results if r.severity == ValidationSeverity.WARNING and not r.status]
    
    @property
    def summary(self) -> Dict[str, int]:
        """Get validation summary counts."""
        return {
            "total_checks": len(self.results),
            "passed": len([r for r in self.results if r.status]),
            "failed": len([r for r in self.results if not r.status]),
            "critical_errors": len(self.critical_errors),
            "errors": len(self.errors),
            "warnings": len(self.warnings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overall_status": self.overall_status,
            "validation_id": self.validation_id,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results]
        }


class RDBConfigValidator(LoggerMixin):
    """
    Validator for RDB pipeline configurations and system health.
    
    This class performs comprehensive validation of:
    - Database connections and configurations
    - Vector database (Milvus) connectivity
    - Embedding service availability
    - Pipeline configuration validity
    - System resource requirements
    """
    
    def __init__(self):
        """Initialize RDB configuration validator."""
        self.validation_id = f"rdb_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.results: List[ValidationResult] = []
        
        self.logger.info(f"Initialized RDB configuration validator {self.validation_id}")
    
    async def validate_complete_system(self) -> ValidationReport:
        """
        Perform complete system validation.
        
        Returns:
            Complete validation report
        """
        self.logger.info("Starting complete RDB system validation")
        
        # Clear previous results
        self.results = []
        
        # Validate configuration
        await self._validate_application_config()
        
        # Validate RDB connections
        await self._validate_rdb_connections()
        
        # Validate vector database
        await self._validate_vector_database()
        
        # Validate embedding services
        await self._validate_embedding_services()
        
        # Validate system resources
        await self._validate_system_resources()
        
        # Determine overall status
        critical_errors = [r for r in self.results if r.severity == ValidationSeverity.CRITICAL and not r.status]
        errors = [r for r in self.results if r.severity == ValidationSeverity.ERROR and not r.status]
        overall_status = len(critical_errors) == 0 and len(errors) == 0
        
        report = ValidationReport(
            overall_status=overall_status,
            validation_id=self.validation_id,
            timestamp=datetime.now(timezone.utc),
            results=self.results.copy()
        )
        
        self.logger.info(f"System validation completed: {report.summary}")
        return report
    
    async def validate_rdb_connection(self, database_config: DatabaseConfig) -> ValidationReport:
        """
        Validate a specific RDB connection.
        
        Args:
            database_config: Database configuration to validate
            
        Returns:
            Validation report for the connection
        """
        self.results = []
        
        # Test basic connection
        await self._test_database_connection(database_config)
        
        # Test table discovery
        await self._test_table_discovery(database_config)
        
        # Test data extraction
        await self._test_data_extraction(database_config)
        
        overall_status = not any(r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] and not r.status for r in self.results)
        
        return ValidationReport(
            overall_status=overall_status,
            validation_id=self.validation_id,
            timestamp=datetime.now(timezone.utc),
            results=self.results.copy()
        )
    
    async def _validate_application_config(self):
        """Validate application configuration."""
        try:
            config = get_config()
            
            # Check if config is loaded
            self._add_result(
                "application_config",
                "config_loaded",
                ValidationSeverity.CRITICAL,
                config is not None,
                "Application configuration loaded successfully" if config else "Failed to load application configuration"
            )
            
            if not config:
                return
            
            # Check required configuration sections
            required_sections = ["milvus", "embedding_providers"]
            for section in required_sections:
                has_section = hasattr(config, section) and getattr(config, section) is not None
                self._add_result(
                    "application_config",
                    f"{section}_config",
                    ValidationSeverity.ERROR,
                    has_section,
                    f"{section} configuration present" if has_section else f"{section} configuration missing"
                )
            
            # Check RDB connections
            has_rdb_connections = hasattr(config, 'rdb_connections') and config.rdb_connections
            self._add_result(
                "application_config",
                "rdb_connections",
                ValidationSeverity.WARNING,
                bool(has_rdb_connections),
                f"RDB connections configured: {len(config.rdb_connections) if has_rdb_connections else 0}"
            )
            
        except Exception as e:
            self._add_result(
                "application_config",
                "config_validation",
                ValidationSeverity.CRITICAL,
                False,
                f"Configuration validation failed: {e}"
            )
    
    async def _validate_rdb_connections(self):
        """Validate all configured RDB connections."""
        try:
            config = get_config()
            
            if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
                self._add_result(
                    "rdb_connections",
                    "connections_configured",
                    ValidationSeverity.WARNING,
                    False,
                    "No RDB connections configured"
                )
                return
            
            for db_name, db_config in config.rdb_connections.items():
                await self._test_database_connection(db_config, db_name)
                
        except Exception as e:
            self._add_result(
                "rdb_connections",
                "validation_error",
                ValidationSeverity.ERROR,
                False,
                f"RDB connections validation failed: {e}"
            )
    
    async def _test_database_connection(self, database_config: DatabaseConfig, db_name: Optional[str] = None):
        """Test a specific database connection."""
        component_name = f"rdb_connection_{db_name}" if db_name else "rdb_connection"
        
        try:
            # Create extraction config
            extraction_config = ExtractionConfig(
                database_config=database_config,
                mode=ExtractionMode.FULL,
                batch_size=1,
                timeout=30
            )
            
            # Create extractor
            extractor = RDBExtractorFactory.create(extraction_config)
            
            try:
                # Test connection
                connection_valid = extractor.validate_connection()
                self._add_result(
                    component_name,
                    "connection_test",
                    ValidationSeverity.ERROR,
                    connection_valid,
                    f"Database connection successful" if connection_valid else f"Database connection failed",
                    {"host": database_config.host, "database": database_config.database}
                )
                
            finally:
                extractor.close()
                
        except Exception as e:
            self._add_result(
                component_name,
                "connection_test",
                ValidationSeverity.ERROR,
                False,
                f"Database connection test failed: {e}",
                {"host": database_config.host, "database": database_config.database}
            )
    
    async def _test_table_discovery(self, database_config: DatabaseConfig, db_name: Optional[str] = None):
        """Test table discovery capability."""
        component_name = f"rdb_discovery_{db_name}" if db_name else "rdb_discovery"
        
        try:
            extraction_config = ExtractionConfig(
                database_config=database_config,
                mode=ExtractionMode.FULL,
                timeout=30
            )
            
            extractor = RDBExtractorFactory.create(extraction_config)
            
            try:
                tables = extractor.discover_tables()
                has_tables = len(tables) > 0
                
                self._add_result(
                    component_name,
                    "table_discovery",
                    ValidationSeverity.WARNING,
                    has_tables,
                    f"Discovered {len(tables)} tables" if has_tables else "No tables found in database",
                    {"table_count": len(tables), "tables": tables[:10]}  # First 10 tables
                )
                
            finally:
                extractor.close()
                
        except Exception as e:
            self._add_result(
                component_name,
                "table_discovery",
                ValidationSeverity.WARNING,
                False,
                f"Table discovery failed: {e}"
            )
    
    async def _test_data_extraction(self, database_config: DatabaseConfig, db_name: Optional[str] = None):
        """Test data extraction capability."""
        component_name = f"rdb_extraction_{db_name}" if db_name else "rdb_extraction"
        
        try:
            extraction_config = ExtractionConfig(
                database_config=database_config,
                mode=ExtractionMode.FULL,
                batch_size=5,  # Small batch for testing
                max_rows=5,    # Limit rows for testing
                timeout=30
            )
            
            extractor = RDBExtractorFactory.create(extraction_config)
            
            try:
                # Get tables
                tables = extractor.discover_tables()
                
                if not tables:
                    self._add_result(
                        component_name,
                        "data_extraction",
                        ValidationSeverity.INFO,
                        True,
                        "No tables available for extraction testing"
                    )
                    return
                
                # Test extraction on first table
                test_table = tables[0]
                result = extractor.extract_table_data(test_table)
                
                has_data = len(result.data) > 0
                self._add_result(
                    component_name,
                    "data_extraction",
                    ValidationSeverity.INFO,
                    True,
                    f"Successfully extracted {len(result.data)} rows from table '{test_table}'",
                    {"test_table": test_table, "row_count": len(result.data)}
                )
                
            finally:
                extractor.close()
                
        except Exception as e:
            self._add_result(
                component_name,
                "data_extraction",
                ValidationSeverity.WARNING,
                False,
                f"Data extraction test failed: {e}"
            )
    
    async def _validate_vector_database(self):
        """Validate vector database (Milvus) connectivity."""
        try:
            config = get_config()
            
            if not hasattr(config, 'milvus') or not config.milvus:
                self._add_result(
                    "vector_database",
                    "milvus_config",
                    ValidationSeverity.ERROR,
                    False,
                    "Milvus configuration not found"
                )
                return
            
            # Test Milvus connection
            try:
                milvus_client = MilvusClient(config.milvus)
                ping_result = milvus_client.ping()
                
                is_healthy = ping_result.get("status") == "healthy"
                self._add_result(
                    "vector_database",
                    "milvus_connection",
                    ValidationSeverity.ERROR,
                    is_healthy,
                    f"Milvus connection {'successful' if is_healthy else 'failed'}",
                    ping_result
                )
                
            except Exception as e:
                self._add_result(
                    "vector_database",
                    "milvus_connection",
                    ValidationSeverity.ERROR,
                    False,
                    f"Milvus connection failed: {e}"
                )
                
        except Exception as e:
            self._add_result(
                "vector_database",
                "milvus_validation",
                ValidationSeverity.ERROR,
                False,
                f"Vector database validation failed: {e}"
            )
    
    async def _validate_embedding_services(self):
        """Validate embedding service availability."""
        try:
            config = get_config()
            
            if not hasattr(config, 'embedding_providers') or not config.embedding_providers:
                self._add_result(
                    "embedding_services",
                    "providers_config",
                    ValidationSeverity.ERROR,
                    False,
                    "No embedding providers configured"
                )
                return
            
            # Test embedding configuration availability (simplified)
            try:
                # For now, just test if embedding configuration is properly set up
                has_embedding_config = config.embedding and config.embedding.provider
                
                # Test specific provider configuration
                if has_embedding_config:
                    if config.embedding.provider == "ollama":
                        # Test Ollama connectivity (simplified)
                        try:
                            import requests
                            ollama_host = config.embedding.base_url or "http://localhost:11434"
                            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                            ollama_available = response.status_code == 200
                        except:
                            ollama_available = False
                        
                        self._add_result(
                            "embedding_services",
                            "ollama_connectivity",
                            ValidationSeverity.ERROR,
                            ollama_available,
                            f"Ollama service {'available' if ollama_available else 'unavailable'} at {ollama_host}",
                            {"provider": "ollama", "host": ollama_host}
                        )
                    else:
                        # For other providers, just check configuration
                        has_api_key = bool(config.embedding.api_key)
                        self._add_result(
                            "embedding_services",
                            "provider_config",
                            ValidationSeverity.WARNING,
                            has_api_key,
                            f"Embedding provider '{config.embedding.provider}' configured with{'out' if not has_api_key else ''} API key",
                            {"provider": config.embedding.provider, "has_api_key": has_api_key}
                        )
                else:
                    self._add_result(
                        "embedding_services",
                        "config_check",
                        ValidationSeverity.ERROR,
                        False,
                        "No embedding configuration found"
                    )
                
            except Exception as e:
                self._add_result(
                    "embedding_services",
                    "provider_health",
                    ValidationSeverity.ERROR,
                    False,
                    f"Embedding service validation failed: {e}"
                )
                
        except Exception as e:
            self._add_result(
                "embedding_services",
                "validation_error",
                ValidationSeverity.ERROR,
                False,
                f"Embedding services validation failed: {e}"
            )
    
    async def _validate_system_resources(self):
        """Validate system resource requirements."""
        try:
            import psutil
            import sys
            
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            memory_ok = available_gb >= 2.0  # Minimum 2GB available
            self._add_result(
                "system_resources",
                "memory_check",
                ValidationSeverity.WARNING,
                memory_ok,
                f"Available memory: {available_gb:.1f}GB",
                {"available_gb": available_gb, "total_gb": memory.total / (1024**3)}
            )
            
            # Check disk space
            disk = psutil.disk_usage('/')
            available_disk_gb = disk.free / (1024**3)
            
            disk_ok = available_disk_gb >= 5.0  # Minimum 5GB available
            self._add_result(
                "system_resources",
                "disk_check",
                ValidationSeverity.WARNING,
                disk_ok,
                f"Available disk space: {available_disk_gb:.1f}GB",
                {"available_gb": available_disk_gb, "total_gb": disk.total / (1024**3)}
            )
            
            # Check Python version
            python_version = sys.version_info
            python_ok = python_version >= (3, 8)
            self._add_result(
                "system_resources",
                "python_version",
                ValidationSeverity.ERROR,
                python_ok,
                f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}",
                {"version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"}
            )
            
        except ImportError:
            self._add_result(
                "system_resources",
                "psutil_missing",
                ValidationSeverity.WARNING,
                False,
                "psutil not available for system resource checking"
            )
        except Exception as e:
            self._add_result(
                "system_resources",
                "resource_check",
                ValidationSeverity.WARNING,
                False,
                f"System resource validation failed: {e}"
            )
    
    def _add_result(
        self,
        component: str,
        check_name: str,
        severity: ValidationSeverity,
        status: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add a validation result."""
        result = ValidationResult(
            component=component,
            check_name=check_name,
            severity=severity,
            status=status,
            message=message,
            details=details
        )
        
        self.results.append(result)
        
        # Log result
        log_level = "info" if status else ("error" if severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] else "warning")
        getattr(self.logger, log_level)(f"[{component}:{check_name}] {message}")


# Convenience function for quick validation
async def validate_rdb_system() -> ValidationReport:
    """
    Quick system validation function.
    
    Returns:
        Complete validation report
    """
    validator = RDBConfigValidator()
    return await validator.validate_complete_system()


# Convenience function for testing a specific database
async def test_database_connection(database_config: DatabaseConfig) -> ValidationReport:
    """
    Test a specific database connection.
    
    Args:
        database_config: Database configuration to test
        
    Returns:
        Validation report for the connection
    """
    validator = RDBConfigValidator()
    return await validator.validate_rdb_connection(database_config)