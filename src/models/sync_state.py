"""
Sync state tracking models for data synchronization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from src.core.database import Base


class SyncStatusDB(Enum):
    """Database sync status enumeration"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSourceTypeDB(Enum):
    """Database data source type enumeration"""
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    API = "api"
    WEBHOOK = "webhook"


class SyncStateModel(Base):
    """Database model for tracking sync states"""
    
    __tablename__ = "sync_states"
    
    id = Column(Integer, primary_key=True)
    source_id = Column(String(255), nullable=False, unique=True, index=True)
    source_type = Column(SQLEnum(DataSourceTypeDB), nullable=False)
    
    # Sync tracking
    last_sync = Column(DateTime, nullable=True)
    last_hash = Column(String(255), nullable=True)
    sync_status = Column(SQLEnum(SyncStatusDB), nullable=False, default=SyncStatusDB.PENDING)
    error_message = Column(Text, nullable=True)
    
    # Statistics
    records_processed = Column(Integer, default=0)
    records_added = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_deleted = Column(Integer, default=0)
    sync_duration = Column(Float, default=0.0)
    
    # Metadata
    source_config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"SyncState(source_id='{self.source_id}', status='{self.sync_status.value}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sync state to dictionary format"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type.value if self.source_type else None,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_hash": self.last_hash,
            "sync_status": self.sync_status.value if self.sync_status else None,
            "error_message": self.error_message,
            "records_processed": self.records_processed,
            "records_added": self.records_added,
            "records_updated": self.records_updated,
            "records_deleted": self.records_deleted,
            "sync_duration": self.sync_duration,
            "source_config": self.source_config or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def create_from_config(cls, source_id: str, source_type: DataSourceTypeDB, 
                          config: Dict[str, Any]) -> "SyncStateModel":
        """Create sync state from source configuration"""
        return cls(
            source_id=source_id,
            source_type=source_type,
            source_config=config,
            sync_status=SyncStatusDB.PENDING
        )
    
    def update_sync_start(self):
        """Update state when sync starts"""
        self.sync_status = SyncStatusDB.RUNNING
        self.error_message = None
        self.updated_at = datetime.utcnow()
    
    def update_sync_completed(self, records_processed: int = 0, records_added: int = 0,
                             records_updated: int = 0, records_deleted: int = 0,
                             sync_duration: float = 0.0, last_hash: Optional[str] = None):
        """Update state when sync completes successfully"""
        self.sync_status = SyncStatusDB.COMPLETED
        self.last_sync = datetime.utcnow()
        self.records_processed = records_processed
        self.records_added = records_added
        self.records_updated = records_updated
        self.records_deleted = records_deleted
        self.sync_duration = sync_duration
        if last_hash:
            self.last_hash = last_hash
        self.error_message = None
        self.updated_at = datetime.utcnow()
    
    def update_sync_failed(self, error_message: str):
        """Update state when sync fails"""
        self.sync_status = SyncStatusDB.FAILED
        self.error_message = error_message
        self.updated_at = datetime.utcnow()


class SyncLogModel(Base):
    """Model for logging sync operations"""
    
    __tablename__ = "sync_logs"
    
    id = Column(Integer, primary_key=True)
    source_id = Column(String(255), nullable=False, index=True)
    sync_started_at = Column(DateTime, nullable=False)
    sync_completed_at = Column(DateTime, nullable=True)
    
    # Results
    sync_status = Column(SQLEnum(SyncStatusDB), nullable=False)
    records_processed = Column(Integer, default=0)
    records_added = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_deleted = Column(Integer, default=0)
    sync_type = Column(String(50), nullable=False)  # 'full' or 'incremental'
    
    # Error details
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Performance metrics
    sync_duration = Column(Float, default=0.0)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    def __repr__(self) -> str:
        return f"SyncLog(source_id='{self.source_id}', status='{self.sync_status.value}', duration={self.sync_duration}s)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sync log to dictionary format"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "sync_started_at": self.sync_started_at.isoformat() if self.sync_started_at else None,
            "sync_completed_at": self.sync_completed_at.isoformat() if self.sync_completed_at else None,
            "sync_status": self.sync_status.value,
            "records_processed": self.records_processed,
            "records_added": self.records_added,
            "records_updated": self.records_updated,
            "records_deleted": self.records_deleted,
            "sync_type": self.sync_type,
            "error_message": self.error_message,
            "error_details": self.error_details or {},
            "sync_duration": self.sync_duration,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }
    
    @classmethod
    def create_log_entry(cls, source_id: str, sync_type: str = "incremental") -> "SyncLogModel":
        """Create new sync log entry"""
        return cls(
            source_id=source_id,
            sync_started_at=datetime.utcnow(),
            sync_status=SyncStatusDB.RUNNING,
            sync_type=sync_type
        )
    
    def complete_sync(self, records_processed: int = 0, records_added: int = 0,
                     records_updated: int = 0, records_deleted: int = 0,
                     memory_usage_mb: Optional[float] = None,
                     cpu_usage_percent: Optional[float] = None):
        """Complete sync log entry"""
        self.sync_completed_at = datetime.utcnow()
        self.sync_status = SyncStatusDB.COMPLETED
        self.records_processed = records_processed
        self.records_added = records_added
        self.records_updated = records_updated
        self.records_deleted = records_deleted
        
        if self.sync_started_at:
            self.sync_duration = (self.sync_completed_at - self.sync_started_at).total_seconds()
        
        if memory_usage_mb is not None:
            self.memory_usage_mb = memory_usage_mb
        if cpu_usage_percent is not None:
            self.cpu_usage_percent = cpu_usage_percent
    
    def fail_sync(self, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """Fail sync log entry"""
        self.sync_completed_at = datetime.utcnow()
        self.sync_status = SyncStatusDB.FAILED
        self.error_message = error_message
        self.error_details = error_details
        
        if self.sync_started_at:
            self.sync_duration = (self.sync_completed_at - self.sync_started_at).total_seconds()