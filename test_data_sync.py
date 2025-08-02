"""
Comprehensive tests for data synchronization functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import json

from src.services.data_sync import (
    DataSyncManager, 
    DataSourceAdapter, 
    DatabaseSourceAdapter,
    FileSystemSourceAdapter,
    SyncState,
    SyncStatus,
    DataSourceType, 
    DataRecord
)
from src.models.sync_state import SyncStateModel, SyncLogModel, SyncStatusDB, DataSourceTypeDB


class TestDataRecord:
    """Test DataRecord functionality"""
    
    def test_data_record_creation(self):
        """Test creating a data record"""
        record = DataRecord(
            id="test_1",
            content="Test content",
            metadata={"key": "value"}
        )
        
        assert record.id == "test_1"
        assert record.content == "Test content"
        assert record.metadata == {"key": "value"}
        assert record.hash is not None
    
    def test_data_record_hash_generation(self):
        """Test hash generation for data records"""
        record1 = DataRecord(id="1", content="content")
        record2 = DataRecord(id="1", content="content")  # Same content
        record3 = DataRecord(id="1", content="different")  # Different content
        
        assert record1.hash == record2.hash
        assert record1.hash != record3.hash


class TestSyncState:
    """Test SyncState functionality"""
    
    def test_sync_state_creation(self):
        """Test creating sync state"""
        state = SyncState(
            source_id="test_source",
            source_type=DataSourceType.DATABASE
        )
        
        assert state.source_id == "test_source"
        assert state.source_type == DataSourceType.DATABASE
        assert state.sync_status == SyncStatus.PENDING
        assert state.records_processed == 0
    
    def test_sync_state_to_dict(self):
        """Test converting sync state to dictionary"""
        state = SyncState(
            source_id="test_source",
            source_type=DataSourceType.DATABASE,
            records_processed=100,
            last_sync=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        result = state.to_dict()
        
        assert result["source_id"] == "test_source"
        assert result["source_type"] == "database"
        assert result["records_processed"] == 100
        assert result["last_sync"] == "2024-01-01T12:00:00"


class TestFileSystemSourceAdapter:
    """Test file system data source adapter"""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            file1 = temp_path / "test1.txt"
            file1.write_text("Content of file 1")
            
            file2 = temp_path / "test2.txt"
            file2.write_text("Content of file 2")
            
            yield [str(file1), str(file2)]
    
    @pytest.mark.asyncio
    async def test_file_system_adapter_get_records(self, temp_files):
        """Test getting records from file system"""
        config = {
            "id": "test_fs",
            "type": "file_system",
            "paths": temp_files
        }
        
        adapter = FileSystemSourceAdapter(config)
        records = await adapter.get_records()
        
        assert len(records) == 2
        assert records[0].content == "Content of file 1" or records[0].content == "Content of file 2"
        assert all(record.metadata.get("file_size") > 0 for record in records)
    
    @pytest.mark.asyncio
    async def test_file_system_adapter_incremental(self, temp_files):
        """Test incremental sync with file system adapter"""
        config = {
            "id": "test_fs",
            "type": "file_system",
            "paths": temp_files
        }
        
        adapter = FileSystemSourceAdapter(config)
        
        # Future date - should return no records
        future_date = datetime.now() + timedelta(days=1)
        records = await adapter.get_records(since=future_date)
        
        assert len(records) == 0
    
    @pytest.mark.asyncio
    async def test_file_system_adapter_test_connection(self, temp_files):
        """Test file system connection test"""
        config = {
            "id": "test_fs",
            "type": "file_system",
            "paths": temp_files
        }
        
        adapter = FileSystemSourceAdapter(config)
        result = await adapter.test_connection()
        
        assert result is True


class TestDatabaseSourceAdapter:
    """Test database source adapter"""
    
    @pytest.mark.asyncio
    async def test_database_adapter_connection_failure(self):
        """Test database adapter with invalid connection"""
        config = {
            "id": "test_db",
            "type": "database",
            "config": {
                "host": "nonexistent",
                "port": 9999,
                "database": "test",
                "user": "test",
                "password": "test",
                "driver": "mysql+pymysql"
            }
        }
        
        adapter = DatabaseSourceAdapter(config)
        result = await adapter.test_connection()
        
        assert result is False


class TestDataSyncManager:
    """Test DataSyncManager functionality"""
    
    @pytest.fixture
    def sync_manager(self):
        """Create sync manager for testing"""
        with patch('src.services.data_sync.get_config') as mock_config:
            mock_config.return_value.embedding = MagicMock()
            manager = DataSyncManager()
            # Clear sync states for clean test
            manager.sync_states = {}
            return manager
    
    def test_sync_manager_initialization(self, sync_manager):
        """Test sync manager initialization"""
        assert isinstance(sync_manager.sync_states, dict)
        assert isinstance(sync_manager.active_syncs, set)
    
    @pytest.mark.asyncio
    async def test_sync_data_source_already_running(self, sync_manager):
        """Test error when trying to sync source that's already running"""
        source_config = {"id": "test_source", "type": "database"}
        
        # Add to active syncs
        sync_manager.active_syncs.add("test_source")
        
        with pytest.raises(ValueError, match="Sync already in progress"):
            await sync_manager.sync_data_source(source_config)
    
    @pytest.mark.asyncio
    async def test_sync_data_source_file_system(self, sync_manager, temp_files):
        """Test syncing file system data source"""
        source_config = {
            "id": "test_fs",
            "type": "file_system",
            "paths": temp_files
        }
        
        # Mock dependencies
        with patch('src.services.data_sync.VectorSearchEngine') as mock_vector, \
             patch('src.services.data_sync.get_embedding_client') as mock_embedding:
            
            mock_vector_instance = AsyncMock()
            mock_vector.return_value = mock_vector_instance
            
            mock_embedding_client = AsyncMock()
            mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_embedding.return_value = mock_embedding_client
            
            result = await sync_manager.sync_data_source(source_config, dry_run=True)
            
            assert result.source_id == "test_fs"
            assert result.sync_status == SyncStatus.COMPLETED
            assert result.records_processed == 2
    
    @pytest.mark.asyncio
    async def test_sync_all_sources_dry_run(self, sync_manager):
        """Test syncing all sources in dry run mode"""
        # Mock file system for demo sources
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value="test content"), \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_stat.return_value.st_size = 100
            mock_stat.return_value.st_mtime = datetime.now().timestamp()
            
            results = await sync_manager.sync_all_sources(dry_run=True)
            
            assert len(results) > 0
            assert all(isinstance(state, SyncState) for state in results.values())
    
    def test_get_sync_status(self, sync_manager):
        """Test getting sync status"""
        # Add a test sync state
        test_state = SyncState(
            source_id="test_source",
            source_type=DataSourceType.DATABASE
        )
        sync_manager.sync_states["test_source"] = test_state
        
        # Get specific source status
        status = sync_manager.get_sync_status("test_source")
        assert len(status) == 1
        assert "test_source" in status
        
        # Get all statuses
        all_status = sync_manager.get_sync_status()
        assert len(all_status) == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, sync_manager):
        """Test cleanup of old data"""
        result = await sync_manager.cleanup_old_data(30)
        
        # Should return count of cleaned records
        assert isinstance(result, int)
        assert result >= 0


class TestSyncStateModel:
    """Test database sync state model"""
    
    def test_sync_state_model_creation(self):
        """Test creating sync state model"""
        state = SyncStateModel.create_from_config(
            source_id="test_source",
            source_type=DataSourceTypeDB.DATABASE,
            config={"host": "localhost"}
        )
        
        assert state.source_id == "test_source"
        assert state.source_type == DataSourceTypeDB.DATABASE
        assert state.sync_status == SyncStatusDB.PENDING
        assert state.source_config == {"host": "localhost"}
    
    def test_sync_state_model_to_dict(self):
        """Test converting model to dictionary"""
        state = SyncStateModel(
            source_id="test_source",
            source_type=DataSourceTypeDB.DATABASE,
            records_processed=100
        )
        
        result = state.to_dict()
        
        assert result["source_id"] == "test_source"
        assert result["source_type"] == "database"
        assert result["records_processed"] == 100
    
    def test_sync_state_model_update_methods(self):
        """Test sync state update methods"""
        state = SyncStateModel(
            source_id="test_source",
            source_type=DataSourceTypeDB.DATABASE
        )
        
        # Test sync start
        state.update_sync_start()
        assert state.sync_status == SyncStatusDB.RUNNING
        assert state.error_message is None
        
        # Test sync completion
        state.update_sync_completed(
            records_processed=100,
            records_added=50,
            records_updated=30,
            sync_duration=10.5
        )
        assert state.sync_status == SyncStatusDB.COMPLETED
        assert state.records_processed == 100
        assert state.records_added == 50
        assert state.records_updated == 30
        assert state.sync_duration == 10.5
        
        # Test sync failure
        state.update_sync_failed("Test error")
        assert state.sync_status == SyncStatusDB.FAILED
        assert state.error_message == "Test error"


class TestSyncLogModel:
    """Test sync log model"""
    
    def test_sync_log_creation(self):
        """Test creating sync log"""
        log = SyncLogModel.create_log_entry("test_source", "incremental")
        
        assert log.source_id == "test_source"
        assert log.sync_type == "incremental"
        assert log.sync_status == SyncStatusDB.RUNNING
        assert log.sync_started_at is not None
    
    def test_sync_log_completion(self):
        """Test completing sync log"""
        log = SyncLogModel.create_log_entry("test_source")
        
        log.complete_sync(
            records_processed=100,
            records_added=50,
            memory_usage_mb=256.5,
            cpu_usage_percent=45.2
        )
        
        assert log.sync_status == SyncStatusDB.COMPLETED
        assert log.records_processed == 100
        assert log.records_added == 50
        assert log.memory_usage_mb == 256.5
        assert log.cpu_usage_percent == 45.2
        assert log.sync_completed_at is not None
        assert log.sync_duration > 0
    
    def test_sync_log_failure(self):
        """Test failing sync log"""
        log = SyncLogModel.create_log_entry("test_source")
        
        log.fail_sync("Test error", {"detail": "error details"})
        
        assert log.sync_status == SyncStatusDB.FAILED
        assert log.error_message == "Test error"
        assert log.error_details == {"detail": "error details"}
        assert log.sync_completed_at is not None


@pytest.fixture
def temp_files():
    """Fixture for creating temporary test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        file1 = temp_path / "test1.txt"
        file1.write_text("Content of file 1")
        
        file2 = temp_path / "test2.txt"
        file2.write_text("Content of file 2")
        
        yield [str(file1), str(file2)]


# Integration tests
class TestDataSyncIntegration:
    """Integration tests for data synchronization"""
    
    @pytest.mark.asyncio
    async def test_full_sync_workflow(self, temp_files):
        """Test complete sync workflow"""
        with patch('src.services.data_sync.get_config') as mock_config, \
             patch('src.services.data_sync.VectorSearchEngine') as mock_vector, \
             patch('src.services.data_sync.get_embedding_client') as mock_embedding:
            
            # Mock configuration
            mock_config.return_value.embedding = MagicMock()
            
            # Mock vector engine
            mock_vector_instance = AsyncMock()
            mock_vector.return_value = mock_vector_instance
            
            # Mock embedding client
            mock_embedding_client = AsyncMock()
            mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_embedding.return_value = mock_embedding_client
            
            # Create sync manager
            sync_manager = DataSyncManager()
            sync_manager.sync_states = {}
            
            # Create source config
            source_config = {
                "id": "test_integration",
                "type": "file_system",
                "paths": temp_files
            }
            
            # Run sync
            result = await sync_manager.sync_data_source(source_config, dry_run=False)
            
            # Verify results
            assert result.source_id == "test_integration"
            assert result.sync_status == SyncStatus.COMPLETED
            assert result.records_processed == 2
            assert result.sync_duration > 0
            
            # Verify embedding client was called
            mock_embedding_client.get_embeddings.assert_called_once()
            
            # Verify vector operations
            assert mock_vector_instance.initialize.called
            assert mock_vector_instance.add_vector.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])