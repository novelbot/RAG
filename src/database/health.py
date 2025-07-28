"""
Database health check and validation system with comprehensive monitoring.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from sqlalchemy import text, select, exc
from sqlalchemy.engine import Engine
from sqlalchemy.sql import literal_column
from loguru import logger

from src.core.config import DatabaseConfig
from src.core.exceptions import DatabaseError, HealthCheckError
from src.core.logging import LoggerMixin
from src.database.pool import AdvancedConnectionPool
from src.database.drivers import DatabaseDriver


class HealthStatus(Enum):
    """Health check status types."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    status: HealthStatus
    response_time: float
    timestamp: datetime
    message: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a database validation operation."""
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class DatabaseHealthChecker(LoggerMixin):
    """Comprehensive database health checking and validation system."""
    
    def __init__(self, pool: AdvancedConnectionPool, config: DatabaseConfig):
        self.pool = pool
        self.config = config
        self.driver = pool._driver
        self._health_history: List[HealthCheckResult] = []
        self._lock = threading.Lock()
        self._max_history = 100
        
    def ping_connection(self, timeout: float = 5.0) -> HealthCheckResult:
        """
        Perform a simple ping test to check basic connectivity.
        
        Args:
            timeout: Maximum time to wait for response
            
        Returns:
            HealthCheckResult with ping status
        """
        start_time = time.time()
        
        try:
            with self.pool.get_connection() as conn:
                # Use driver-specific health check query
                health_query = self.driver.get_health_check_query()
                
                # Execute with timeout
                result = conn.execute(text(health_query))
                expected_value = 1 if "SELECT 1" in health_query else None
                
                # Validate result
                if expected_value is not None:
                    actual_value = result.scalar()
                    if actual_value != expected_value:
                        raise HealthCheckError(f"Unexpected ping result: {actual_value}")
                
                response_time = time.time() - start_time
                
                if response_time > timeout:
                    status = HealthStatus.DEGRADED
                    message = f"Slow response: {response_time:.3f}s"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Ping successful: {response_time:.3f}s"
                
                return HealthCheckResult(
                    status=status,
                    response_time=response_time,
                    timestamp=datetime.now(timezone.utc),
                    message=message,
                    details={
                        'query': health_query,
                        'timeout': timeout,
                        'result': str(result.scalar() if result.returns_rows else 'No result')
                    }
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Ping failed: {e}")
            
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=datetime.now(timezone.utc),
                message="Ping failed",
                error=str(e),
                details={
                    'query': self.driver.get_health_check_query(),
                    'timeout': timeout,
                    'exception_type': type(e).__name__
                }
            )
    
    def validate_connection_pool(self) -> ValidationResult:
        """
        Validate connection pool configuration and status.
        
        Returns:
            ValidationResult with pool validation details
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Check pool configuration
            if self.config.pool_size <= 0:
                result.checks_failed.append("Invalid pool size")
                result.errors.append(f"Pool size must be positive: {self.config.pool_size}")
                result.is_valid = False
            else:
                result.checks_passed.append("Pool size configuration valid")
            
            if self.config.max_overflow < 0:
                result.checks_failed.append("Invalid max overflow")
                result.errors.append(f"Max overflow must be non-negative: {self.config.max_overflow}")
                result.is_valid = False
            else:
                result.checks_passed.append("Max overflow configuration valid")
            
            if self.config.pool_timeout <= 0:
                result.checks_failed.append("Invalid pool timeout")
                result.errors.append(f"Pool timeout must be positive: {self.config.pool_timeout}")
                result.is_valid = False
            else:
                result.checks_passed.append("Pool timeout configuration valid")
            
            # Check pool status
            pool_status = self.pool.get_pool_status()
            if 'error' in pool_status:
                result.checks_failed.append("Pool status check failed")
                result.errors.append(f"Pool status error: {pool_status['error']}")
                result.is_valid = False
            else:
                result.checks_passed.append("Pool status accessible")
                
                # Check for pool exhaustion
                total_connections = pool_status.get('total_connections', 0)
                active_connections = pool_status.get('active_connections', 0)
                
                if total_connections > 0:
                    usage_ratio = active_connections / total_connections
                    if usage_ratio > 0.9:
                        result.warnings.append(f"High pool usage: {usage_ratio:.1%}")
                    elif usage_ratio > 0.8:
                        result.warnings.append(f"Moderate pool usage: {usage_ratio:.1%}")
                
                # Check for connection errors
                connection_errors = pool_status.get('connection_errors', 0)
                if connection_errors > 0:
                    result.warnings.append(f"Connection errors detected: {connection_errors}")
                
                result.details['pool_status'] = pool_status
            
        except Exception as e:
            result.checks_failed.append("Pool validation exception")
            result.errors.append(f"Pool validation failed: {e}")
            result.is_valid = False
            
        return result
    
    def validate_database_schema(self, 
                                expected_tables: Optional[List[str]] = None,
                                check_permissions: bool = True) -> ValidationResult:
        """
        Validate database schema and permissions.
        
        Args:
            expected_tables: List of tables that should exist
            check_permissions: Whether to check user permissions
            
        Returns:
            ValidationResult with schema validation details
        """
        result = ValidationResult(is_valid=True)
        
        try:
            with self.pool.get_connection() as conn:
                # Check database version
                try:
                    version_query = self.driver.get_version_query()
                    version_result = conn.execute(text(version_query))
                    version = version_result.scalar()
                    result.checks_passed.append("Database version accessible")
                    result.details['database_version'] = str(version)
                except Exception as e:
                    result.warnings.append(f"Could not retrieve database version: {e}")
                
                # Check table existence
                if expected_tables:
                    try:
                        # Get existing tables using driver-specific query
                        if self.config.driver.lower() == 'postgresql':
                            tables_query = """
                                SELECT tablename FROM pg_tables 
                                WHERE schemaname = 'public'
                            """
                        elif self.config.driver.lower() in ['mysql', 'mariadb']:
                            tables_query = "SHOW TABLES"
                        elif self.config.driver.lower() == 'oracle':
                            tables_query = """
                                SELECT table_name FROM user_tables
                            """
                        elif self.config.driver.lower() in ['mssql', 'sqlserver']:
                            tables_query = """
                                SELECT name FROM sys.tables
                            """
                        else:
                            tables_query = None
                        
                        if tables_query:
                            existing_tables = set()
                            table_result = conn.execute(text(tables_query))
                            for row in table_result:
                                existing_tables.add(row[0].lower())
                            
                            missing_tables = []
                            for table in expected_tables:
                                if table.lower() not in existing_tables:
                                    missing_tables.append(table)
                            
                            if missing_tables:
                                result.checks_failed.append("Missing required tables")
                                result.errors.append(f"Missing tables: {missing_tables}")
                                result.is_valid = False
                            else:
                                result.checks_passed.append("All required tables exist")
                            
                            result.details['existing_tables'] = list(existing_tables)
                            result.details['missing_tables'] = missing_tables
                        
                    except Exception as e:
                        result.warnings.append(f"Could not check table existence: {e}")
                
                # Check basic permissions
                if check_permissions:
                    try:
                        # Test SELECT permission
                        conn.execute(text(self.driver.get_health_check_query()))
                        result.checks_passed.append("SELECT permission verified")
                        
                        # Test CREATE permission (create temporary table)
                        if self.config.driver.lower() == 'postgresql':
                            create_test = "CREATE TEMP TABLE health_check_test (id INTEGER)"
                            drop_test = "DROP TABLE health_check_test"
                        elif self.config.driver.lower() in ['mysql', 'mariadb']:
                            create_test = "CREATE TEMPORARY TABLE health_check_test (id INTEGER)"
                            drop_test = "DROP TABLE health_check_test"
                        elif self.config.driver.lower() == 'oracle':
                            create_test = "CREATE GLOBAL TEMPORARY TABLE health_check_test (id NUMBER)"
                            drop_test = "DROP TABLE health_check_test"
                        elif self.config.driver.lower() in ['mssql', 'sqlserver']:
                            create_test = "CREATE TABLE #health_check_test (id INTEGER)"
                            drop_test = "DROP TABLE #health_check_test"
                        else:
                            create_test = None
                            drop_test = None
                        
                        if create_test and drop_test:
                            conn.execute(text(create_test))
                            conn.execute(text(drop_test))
                            result.checks_passed.append("CREATE permission verified")
                        
                    except Exception as e:
                        result.warnings.append(f"Permission check failed: {e}")
                
        except Exception as e:
            result.checks_failed.append("Schema validation exception")
            result.errors.append(f"Schema validation failed: {e}")
            result.is_valid = False
            
        return result
    
    def validate_connection_limits(self) -> ValidationResult:
        """
        Validate connection limits and resource usage.
        
        Returns:
            ValidationResult with connection limit validation
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Test multiple connections
            test_connections = min(self.config.pool_size, 5)
            connections = []
            
            try:
                for i in range(test_connections):
                    conn = self.pool.get_connection()
                    connections.append(conn)
                    
                result.checks_passed.append(f"Successfully created {test_connections} connections")
                
                # Test concurrent access
                for conn in connections:
                    with conn:
                        conn.execute(text(self.driver.get_health_check_query()))
                        
                result.checks_passed.append("Concurrent connection access successful")
                
            except Exception as e:
                result.checks_failed.append("Connection limit test failed")
                result.errors.append(f"Connection limit error: {e}")
                result.is_valid = False
                
            finally:
                # Clean up connections
                for conn in connections:
                    try:
                        conn.close()
                    except:
                        pass
            
            # Check pool metrics
            pool_status = self.pool.get_pool_status()
            if 'error' not in pool_status:
                hit_ratio = pool_status.get('hit_ratio', 0)
                if hit_ratio < 0.8:
                    result.warnings.append(f"Low pool hit ratio: {hit_ratio:.1%}")
                
                connection_errors = pool_status.get('connection_errors', 0)
                if connection_errors > 0:
                    result.warnings.append(f"Connection errors: {connection_errors}")
                
                result.details['pool_metrics'] = pool_status
            
        except Exception as e:
            result.checks_failed.append("Connection limit validation exception")
            result.errors.append(f"Connection limit validation failed: {e}")
            result.is_valid = False
            
        return result
    
    def perform_comprehensive_health_check(self,
                                         expected_tables: Optional[List[str]] = None,
                                         ping_timeout: float = 5.0) -> Dict[str, Any]:
        """
        Perform comprehensive health check including all validation types.
        
        Args:
            expected_tables: List of expected tables to validate
            ping_timeout: Timeout for ping operations
            
        Returns:
            Dictionary with comprehensive health check results
        """
        start_time = time.time()
        
        # Perform individual checks
        ping_result = self.ping_connection(timeout=ping_timeout)
        pool_validation = self.validate_connection_pool()
        schema_validation = self.validate_database_schema(expected_tables)
        connection_validation = self.validate_connection_limits()
        
        # Determine overall health status
        overall_status = HealthStatus.HEALTHY
        
        if ping_result.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif (not pool_validation.is_valid or 
              not schema_validation.is_valid or 
              not connection_validation.is_valid):
            overall_status = HealthStatus.UNHEALTHY
        elif (ping_result.status == HealthStatus.DEGRADED or
              pool_validation.warnings or 
              schema_validation.warnings or 
              connection_validation.warnings):
            overall_status = HealthStatus.DEGRADED
        
        # Compile results
        total_time = time.time() - start_time
        
        comprehensive_result = {
            'overall_status': overall_status.value,
            'total_check_time': total_time,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'database_info': {
                'driver': self.config.driver,
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.name,
                'pool_size': self.config.pool_size,
                'max_overflow': self.config.max_overflow
            },
            'checks': {
                'ping': {
                    'status': ping_result.status.value,
                    'response_time': ping_result.response_time,
                    'message': ping_result.message,
                    'error': ping_result.error,
                    'details': ping_result.details
                },
                'pool_validation': {
                    'is_valid': pool_validation.is_valid,
                    'checks_passed': pool_validation.checks_passed,
                    'checks_failed': pool_validation.checks_failed,
                    'warnings': pool_validation.warnings,
                    'errors': pool_validation.errors,
                    'details': pool_validation.details
                },
                'schema_validation': {
                    'is_valid': schema_validation.is_valid,
                    'checks_passed': schema_validation.checks_passed,
                    'checks_failed': schema_validation.checks_failed,
                    'warnings': schema_validation.warnings,
                    'errors': schema_validation.errors,
                    'details': schema_validation.details
                },
                'connection_validation': {
                    'is_valid': connection_validation.is_valid,
                    'checks_passed': connection_validation.checks_passed,
                    'checks_failed': connection_validation.checks_failed,
                    'warnings': connection_validation.warnings,
                    'errors': connection_validation.errors,
                    'details': connection_validation.details
                }
            }
        }
        
        # Store in history
        self._store_health_result(ping_result)
        
        return comprehensive_result
    
    def _store_health_result(self, result: HealthCheckResult) -> None:
        """Store health check result in history."""
        with self._lock:
            self._health_history.append(result)
            if len(self._health_history) > self._max_history:
                self._health_history.pop(0)
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent health check history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent health check results
        """
        with self._lock:
            recent_results = self._health_history[-limit:]
            return [
                {
                    'status': result.status.value,
                    'response_time': result.response_time,
                    'timestamp': result.timestamp.isoformat(),
                    'message': result.message,
                    'error': result.error,
                    'details': result.details
                }
                for result in recent_results
            ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of health check history.
        
        Returns:
            Dictionary with health summary statistics
        """
        with self._lock:
            if not self._health_history:
                return {
                    'total_checks': 0,
                    'avg_response_time': 0,
                    'success_rate': 0,
                    'status_distribution': {}
                }
            
            total_checks = len(self._health_history)
            successful_checks = sum(1 for r in self._health_history 
                                  if r.status == HealthStatus.HEALTHY)
            avg_response_time = sum(r.response_time for r in self._health_history) / total_checks
            
            status_distribution = {}
            for status in HealthStatus:
                count = sum(1 for r in self._health_history if r.status == status)
                status_distribution[status.value] = count
            
            return {
                'total_checks': total_checks,
                'avg_response_time': avg_response_time,
                'success_rate': successful_checks / total_checks,
                'status_distribution': status_distribution,
                'last_check': self._health_history[-1].timestamp.isoformat(),
                'oldest_check': self._health_history[0].timestamp.isoformat()
            }


class HealthCheckManager(LoggerMixin):
    """Manager for multiple database health checkers."""
    
    def __init__(self):
        self._checkers: Dict[str, DatabaseHealthChecker] = {}
        self._lock = threading.Lock()
    
    def add_checker(self, name: str, checker: DatabaseHealthChecker) -> None:
        """Add a health checker."""
        with self._lock:
            self._checkers[name] = checker
            self.logger.info(f"Added health checker: {name}")
    
    def remove_checker(self, name: str) -> None:
        """Remove a health checker."""
        with self._lock:
            if name in self._checkers:
                del self._checkers[name]
                self.logger.info(f"Removed health checker: {name}")
    
    def get_checker(self, name: str) -> Optional[DatabaseHealthChecker]:
        """Get a health checker by name."""
        with self._lock:
            return self._checkers.get(name)
    
    def check_all_databases(self, 
                          expected_tables: Optional[Dict[str, List[str]]] = None,
                          timeout: float = 30.0) -> Dict[str, Any]:
        """
        Perform health checks on all registered databases.
        
        Args:
            expected_tables: Dictionary mapping database names to expected tables
            timeout: Maximum time for all checks
            
        Returns:
            Dictionary with results for all databases
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self._checkers)) as executor:
            # Submit all health checks
            future_to_name = {}
            for name, checker in self._checkers.items():
                tables = expected_tables.get(name) if expected_tables else None
                future = executor.submit(
                    checker.perform_comprehensive_health_check,
                    expected_tables=tables
                )
                future_to_name[future] = name
            
            # Collect results with timeout
            for future in as_completed(future_to_name, timeout=timeout):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    results[name] = {
                        'overall_status': HealthStatus.UNHEALTHY.value,
                        'error': str(e),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
        
        return results
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Get summary report for all databases.
        
        Returns:
            Dictionary with overall health summary
        """
        with self._lock:
            if not self._checkers:
                return {
                    'total_databases': 0,
                    'healthy_databases': 0,
                    'unhealthy_databases': 0,
                    'databases': {}
                }
            
            database_summaries = {}
            healthy_count = 0
            
            for name, checker in self._checkers.items():
                try:
                    # Quick ping to determine status
                    ping_result = checker.ping_connection(timeout=5.0)
                    if ping_result.status == HealthStatus.HEALTHY:
                        healthy_count += 1
                    
                    database_summaries[name] = {
                        'status': ping_result.status.value,
                        'response_time': ping_result.response_time,
                        'last_check': ping_result.timestamp.isoformat(),
                        'summary': checker.get_health_summary()
                    }
                except Exception as e:
                    database_summaries[name] = {
                        'status': HealthStatus.UNHEALTHY.value,
                        'error': str(e),
                        'last_check': datetime.now(timezone.utc).isoformat()
                    }
            
            return {
                'total_databases': len(self._checkers),
                'healthy_databases': healthy_count,
                'unhealthy_databases': len(self._checkers) - healthy_count,
                'databases': database_summaries
            }