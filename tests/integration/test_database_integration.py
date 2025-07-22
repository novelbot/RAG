"""
Integration tests for Database layer.
"""
import pytest
import os
from unittest.mock import patch
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from src.core.config import ConfigManager
from src.database.base import DatabaseManager, DatabaseFactory
from src.database.pool import AdvancedConnectionPool, PoolMetrics
from src.database.health import DatabaseHealthChecker, HealthCheckManager


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseEngineIntegration:
    """Integration tests for DatabaseEngine."""
    
    @pytest.fixture
    def real_config_manager(self, test_config_dict):
        """Real config manager for testing."""
        with patch.object(ConfigManager, 'get_database_config', return_value=test_config_dict["database"]):
            config_manager = ConfigManager()
            yield config_manager

    @pytest.fixture
    def test_engine_config(self):
        """Test engine configuration for integration tests."""
        return EngineConfig(
            host=os.getenv("TEST_DB_HOST", "localhost"),
            port=int(os.getenv("TEST_DB_PORT", "5432")),
            database=os.getenv("TEST_DB_NAME", "test_ragdb"),
            username=os.getenv("TEST_DB_USER", "test_user"),
            password=os.getenv("TEST_DB_PASSWORD", "test_password"),
            driver="postgresql",
            pool_size=5,
            max_overflow=2,
            pool_timeout=10
        )

    def test_engine_initialization_and_connection(self, real_config_manager, skip_if_no_database):
        """Test engine initialization and connection to real database."""
        skip_if_no_database()
        
        engine = DatabaseEngine(real_config_manager)
        
        try:
            engine.initialize()
            
            # Test connection
            assert engine.test_connection() is True
            
            # Test engine info
            info = engine.get_engine_info()
            assert "driver" in info
            assert "url" in info
            assert "pool_size" in info
            
        finally:
            engine.close()

    def test_engine_connection_failure(self, test_engine_config):
        """Test engine connection failure with invalid config."""
        # Use invalid configuration
        invalid_config = EngineConfig(
            host="invalid_host",
            port=9999,
            database="invalid_db",
            username="invalid_user",
            password="invalid_password",
            driver="postgresql"
        )
        
        with patch.object(ConfigManager, 'get_database_config', return_value=invalid_config.to_dict()):
            config_manager = ConfigManager()
            engine = DatabaseEngine(config_manager)
            
            engine.initialize()
            
            # Connection should fail
            assert engine.test_connection() is False

    def test_engine_multiple_connections(self, real_config_manager, skip_if_no_database):
        """Test multiple simultaneous connections."""
        skip_if_no_database()
        
        engine = DatabaseEngine(real_config_manager)
        
        try:
            engine.initialize()
            
            # Create multiple connections
            connections = []
            for _ in range(3):
                conn = engine.engine.connect()
                connections.append(conn)
                
                # Execute simple query
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
            
            # Close all connections
            for conn in connections:
                conn.close()
                
        finally:
            engine.close()


@pytest.mark.integration
@pytest.mark.database
class TestConnectionPoolIntegration:
    """Integration tests for ConnectionPool."""
    
    @pytest.fixture
    def database_engine(self, real_config_manager, skip_if_no_database):
        """Real database engine for testing."""
        skip_if_no_database()
        
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        yield engine
        
        engine.close()

    def test_pool_connection_management(self, database_engine):
        """Test connection pool management."""
        pool_config = PoolConfig(
            pool_size=3,
            max_overflow=1,
            pool_timeout=5
        )
        
        pool = ConnectionPool(database_engine, pool_config)
        
        # Test getting and returning connections
        connections = []
        
        try:
            # Get multiple connections
            for i in range(3):
                with pool.get_connection() as conn:
                    result = conn.execute(text("SELECT 1"))
                    assert result.scalar() == 1
                    connections.append(conn)
            
            # Test pool status
            status = pool.get_pool_status()
            assert status["size"] >= 3
            assert "utilization" in status
            
        finally:
            pool.close()

    def test_pool_query_execution(self, database_engine):
        """Test query execution through pool."""
        pool = ConnectionPool(database_engine)
        
        try:
            # Execute simple query
            result = pool.execute_query(text("SELECT 1 as test_value"))
            assert result.scalar() == 1
            
            # Execute query with parameters
            result = pool.execute_query(
                text("SELECT :value as param_value"),
                {"value": 42}
            )
            assert result.scalar() == 42
            
        finally:
            pool.close()

    def test_pool_transaction_execution(self, database_engine):
        """Test transaction execution through pool."""
        pool = ConnectionPool(database_engine)
        
        try:
            def test_transaction(conn):
                # Execute queries within transaction
                result1 = conn.execute(text("SELECT 1"))
                result2 = conn.execute(text("SELECT 2"))
                return result1.scalar() + result2.scalar()
            
            result = pool.execute_transaction(test_transaction)
            assert result == 3
            
        finally:
            pool.close()

    def test_pool_health_check(self, database_engine):
        """Test pool health checking."""
        pool = ConnectionPool(database_engine)
        
        try:
            # Pool should be healthy with working database
            assert pool.is_healthy() is True
            
            # Get detailed health status
            status = pool.get_pool_status()
            assert status["size"] > 0
            
        finally:
            pool.close()

    def test_pool_metrics_collection(self, database_engine):
        """Test pool metrics collection."""
        pool = ConnectionPool(database_engine)
        
        try:
            # Execute some operations to generate metrics
            for _ in range(5):
                with pool.get_connection() as conn:
                    conn.execute(text("SELECT 1"))
            
            metrics = pool.get_pool_metrics()
            assert metrics["total_connections"] == 5
            assert metrics["successful_connections"] == 5
            assert metrics["failed_connections"] == 0
            assert metrics["success_rate"] == 1.0
            
        finally:
            pool.close()


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseHealthMonitorIntegration:
    """Integration tests for DatabaseHealthMonitor."""
    
    @pytest.fixture
    def connection_pool(self, real_config_manager, skip_if_no_database):
        """Real connection pool for testing."""
        skip_if_no_database()
        
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        pool = ConnectionPool(engine)
        
        yield pool
        
        pool.close()
        engine.close()

    def test_health_monitor_all_checks(self, connection_pool):
        """Test all health monitor checks."""
        monitor = DatabaseHealthMonitor(
            pool=connection_pool,
            check_interval=10,
            check_timeout=5.0
        )
        
        # Perform all health checks
        checks = monitor.perform_all_checks()
        
        assert len(checks) == 3  # connection, pool, query_performance
        
        # All checks should be healthy with working database
        for check in checks:
            assert check.status.value in ["healthy", "degraded"]  # Allow degraded for slower systems

    def test_health_monitor_connection_check(self, connection_pool):
        """Test connection health check."""
        monitor = DatabaseHealthMonitor(connection_pool)
        
        result = monitor.check_connection()
        
        assert result["status"] == "healthy"
        assert "latency" in result
        assert result["latency"] > 0

    def test_health_monitor_pool_check(self, connection_pool):
        """Test pool health check."""
        monitor = DatabaseHealthMonitor(connection_pool)
        
        result = monitor.check_pool_health()
        
        assert result["status"] in ["healthy", "degraded"]
        assert "pool_usage" in result
        assert result["pool_usage"] >= 0

    def test_health_monitor_query_performance(self, connection_pool):
        """Test query performance check."""
        monitor = DatabaseHealthMonitor(connection_pool)
        
        result = monitor.check_query_performance()
        
        assert result["status"] in ["healthy", "degraded"]
        assert "query_latency" in result
        assert result["query_latency"] > 0

    def test_health_monitor_report_generation(self, connection_pool):
        """Test health report generation."""
        monitor = DatabaseHealthMonitor(connection_pool)
        
        report = monitor.get_health_report()
        
        assert "overall_status" in report
        assert "checks" in report
        assert "timestamp" in report
        assert len(report["checks"]) == 3
        
        # Overall status should be healthy or degraded
        assert report["overall_status"] in ["healthy", "degraded"]

    def test_health_monitor_history_tracking(self, connection_pool):
        """Test health history tracking."""
        monitor = DatabaseHealthMonitor(connection_pool, history_size=5)
        
        # Perform multiple checks to build history
        for _ in range(3):
            monitor.perform_all_checks()
        
        history = monitor.get_health_history()
        assert len(history) == 9  # 3 checks * 3 iterations
        
        # Get summary
        summary = monitor.get_health_summary()
        assert summary["total_checks"] == 9
        assert summary["healthy_count"] >= 0
        assert summary["health_percentage"] >= 0

    def test_health_monitor_is_healthy(self, connection_pool):
        """Test overall health status."""
        monitor = DatabaseHealthMonitor(connection_pool)
        
        # With working database, should be healthy
        assert monitor.is_healthy() is True


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseLayerIntegration:
    """Integration tests for complete database layer."""
    
    def test_full_database_stack(self, real_config_manager, skip_if_no_database):
        """Test complete database stack integration."""
        skip_if_no_database()
        
        # Initialize engine
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        # Create connection pool
        pool_config = PoolConfig(pool_size=3, max_overflow=1)
        pool = ConnectionPool(engine, pool_config)
        
        # Create health monitor
        monitor = DatabaseHealthMonitor(pool)
        
        try:
            # Test engine functionality
            assert engine.test_connection() is True
            
            # Test pool functionality
            with pool.get_connection() as conn:
                result = conn.execute(text("SELECT 'integration_test' as test"))
                assert result.scalar() == "integration_test"
            
            # Test health monitoring
            assert monitor.is_healthy() is True
            
            # Test pool metrics
            metrics = pool.get_pool_metrics()
            assert metrics["total_connections"] >= 1
            
            # Test health report
            report = monitor.get_health_report()
            assert report["overall_status"] in ["healthy", "degraded"]
            
        finally:
            pool.close()
            engine.close()

    def test_database_error_handling(self, real_config_manager, skip_if_no_database):
        """Test database error handling and recovery."""
        skip_if_no_database()
        
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        pool = ConnectionPool(engine)
        monitor = DatabaseHealthMonitor(pool)
        
        try:
            # Test with invalid query
            with pytest.raises(Exception):  # Should raise DatabaseError
                pool.execute_query(text("SELECT FROM invalid_syntax"))
            
            # Pool should still be healthy after query error
            assert monitor.check_connection()["status"] == "healthy"
            
            # Test recovery with valid query
            result = pool.execute_query(text("SELECT 1"))
            assert result.scalar() == 1
            
        finally:
            pool.close()
            engine.close()

    def test_concurrent_database_operations(self, real_config_manager, skip_if_no_database):
        """Test concurrent database operations."""
        skip_if_no_database()
        
        import threading
        import time
        
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        pool = ConnectionPool(engine, PoolConfig(pool_size=5))
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(3):
                    with pool.get_connection() as conn:
                        result = conn.execute(text("SELECT :worker_id + :iteration"),
                                            {"worker_id": worker_id, "iteration": i})
                        results.append(result.scalar())
                        time.sleep(0.1)  # Simulate work
            except Exception as e:
                errors.append(e)
        
        try:
            # Start multiple worker threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check results
            assert len(errors) == 0  # No errors should occur
            assert len(results) == 9  # 3 workers * 3 iterations
            
        finally:
            pool.close()
            engine.close()


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
class TestDatabasePerformanceIntegration:
    """Performance integration tests for database layer."""
    
    def test_connection_pool_performance(self, real_config_manager, skip_if_no_database):
        """Test connection pool performance under load."""
        skip_if_no_database()
        
        import time
        
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        pool = ConnectionPool(engine, PoolConfig(pool_size=10))
        
        try:
            start_time = time.time()
            
            # Execute many queries
            for _ in range(100):
                with pool.get_connection() as conn:
                    result = conn.execute(text("SELECT 1"))
                    assert result.scalar() == 1
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete reasonably quickly (adjust threshold as needed)
            assert duration < 10.0  # 10 seconds for 100 queries
            
            # Check pool efficiency
            metrics = pool.get_pool_metrics()
            assert metrics["success_rate"] == 1.0
            
        finally:
            pool.close()
            engine.close()

    def test_health_monitoring_performance(self, real_config_manager, skip_if_no_database):
        """Test health monitoring performance."""
        skip_if_no_database()
        
        import time
        
        engine = DatabaseEngine(real_config_manager)
        engine.initialize()
        
        pool = ConnectionPool(engine)
        monitor = DatabaseHealthMonitor(pool)
        
        try:
            start_time = time.time()
            
            # Perform multiple health checks
            for _ in range(10):
                monitor.perform_all_checks()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Health checks should be fast
            assert duration < 5.0  # 5 seconds for 10 full health checks
            
            # Check history was maintained
            history = monitor.get_health_history()
            assert len(history) == 30  # 10 iterations * 3 checks each
            
        finally:
            pool.close()
            engine.close()