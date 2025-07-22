"""
Base RDB Data Extractor - Abstract base class for database data extraction.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from sqlalchemy import text, Table, MetaData, inspect, func
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import DatabaseConfig
from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from src.database.introspection import DatabaseIntrospector
from .exceptions import (
    ExtractionError, ExtractionTimeoutError, ExtractionValidationError,
    ExtractionConnectionError, ExtractionQueryError, ExtractionConfigurationError
)


class ExtractionMode(Enum):
    """Data extraction modes."""
    FULL = "full"
    INCREMENTAL = "incremental"
    CUSTOM = "custom"


class DataFormat(Enum):
    """Output data formats."""
    DICT = "dict"
    JSON = "json"
    DATAFRAME = "dataframe"
    RAW = "raw"


@dataclass
class ExtractionConfig:
    """Configuration for RDB data extraction."""
    
    # Database connection (reuse existing config)
    database_config: DatabaseConfig
    
    # Extraction settings
    mode: ExtractionMode = ExtractionMode.FULL
    batch_size: int = 1000
    max_rows: Optional[int] = None
    timeout: int = 300  # seconds
    
    # Table filtering
    include_tables: Optional[List[str]] = None
    exclude_tables: Optional[List[str]] = None
    table_patterns: Optional[List[str]] = None
    
    # Incremental sync settings
    incremental_column: Optional[str] = None  # timestamp/id column
    incremental_value: Optional[Any] = None  # last sync value
    use_checksum: bool = False
    checksum_columns: Optional[List[str]] = None
    
    # Data validation
    validate_data: bool = True
    max_null_percentage: float = 0.95
    min_row_count: int = 0
    
    # Output format
    output_format: DataFormat = DataFormat.DICT
    include_metadata: bool = True
    
    # Performance settings
    parallel_tables: int = 1
    connection_pool_size: int = 5
    
    # Error handling
    continue_on_error: bool = False
    max_retries: int = 3
    retry_delay: int = 1
    
    def validate(self) -> None:
        """Validate extraction configuration."""
        if self.batch_size <= 0:
            raise ExtractionConfigurationError("batch_size must be positive")
        
        if self.max_rows is not None and self.max_rows <= 0:
            raise ExtractionConfigurationError("max_rows must be positive")
        
        if self.timeout <= 0:
            raise ExtractionConfigurationError("timeout must be positive")
        
        if self.mode == ExtractionMode.INCREMENTAL and not self.incremental_column:
            raise ExtractionConfigurationError("incremental_column required for incremental mode")
        
        if not 0 <= self.max_null_percentage <= 1:
            raise ExtractionConfigurationError("max_null_percentage must be between 0 and 1")


@dataclass
class TableMetadata:
    """Metadata information for a database table."""
    
    name: str
    schema: Optional[str] = None
    columns: List[Dict[str, Any]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    row_count: Optional[int] = None
    table_size: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "schema": self.schema,
            "columns": self.columns,
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
            "indexes": self.indexes,
            "row_count": self.row_count,
            "table_size": self.table_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class ExtractionStats:
    """Statistics for extraction operations."""
    
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_rows: int = 0
    processed_rows: int = 0
    failed_rows: int = 0
    tables_processed: int = 0
    tables_failed: int = 0
    total_time: Optional[float] = None
    rows_per_second: Optional[float] = None
    
    def complete(self) -> None:
        """Mark extraction as complete and calculate final stats."""
        self.end_time = datetime.utcnow()
        self.total_time = (self.end_time - self.start_time).total_seconds()
        
        if self.total_time > 0:
            self.rows_per_second = self.processed_rows / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "failed_rows": self.failed_rows,
            "tables_processed": self.tables_processed,
            "tables_failed": self.tables_failed,
            "total_time": self.total_time,
            "rows_per_second": self.rows_per_second
        }


@dataclass
class ExtractionResult:
    """Result of data extraction operation."""
    
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Optional[TableMetadata] = None
    stats: Optional[ExtractionStats] = None
    errors: List[str] = field(default_factory=list)
    extraction_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "data": self.data,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "stats": self.stats.to_dict() if self.stats else None,
            "errors": self.errors,
            "extraction_id": self.extraction_id,
            "timestamp": self.timestamp.isoformat()
        }
    
    def is_successful(self) -> bool:
        """Check if extraction was successful."""
        return len(self.errors) == 0 and len(self.data) > 0


@dataclass
class IncrementalState:
    """State tracking for incremental extraction."""
    
    table_name: str
    last_sync_time: datetime
    last_sync_value: Any
    checksum: Optional[str] = None
    row_count: int = 0
    
    def generate_checksum(self, data: List[Dict[str, Any]]) -> str:
        """Generate checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "table_name": self.table_name,
            "last_sync_time": self.last_sync_time.isoformat(),
            "last_sync_value": self.last_sync_value,
            "checksum": self.checksum,
            "row_count": self.row_count
        }


class BaseRDBExtractor(ABC, LoggerMixin):
    """
    Abstract base class for RDB data extractors.
    
    Provides common functionality for extracting data from relational databases
    with support for incremental sync, data validation, and error handling.
    """
    
    def __init__(self, config: ExtractionConfig):
        """
        Initialize RDB extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config
        self.config.validate()
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config.database_config)
        self.introspector = DatabaseIntrospector(self.db_manager)
        
        # State tracking
        self.extraction_stats = ExtractionStats()
        self.incremental_states: Dict[str, IncrementalState] = {}
        
        # Initialize engine and metadata
        self._engine: Optional[Engine] = None
        self._metadata: Optional[MetaData] = None
        
        self.logger.info(f"Initialized RDB extractor for {config.database_config.host}")
    
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        if self._engine is None:
            self._engine = self.db_manager.get_engine()
        return self._engine
    
    @property
    def metadata(self) -> MetaData:
        """Get SQLAlchemy metadata."""
        if self._metadata is None:
            self._metadata = MetaData()
        return self._metadata
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def extract_table_data(
        self, 
        table_name: str, 
        schema: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract data from a specific table.
        
        Args:
            table_name: Name of the table to extract from
            schema: Database schema name
            filters: Filter conditions
            order_by: Column to order by
            
        Returns:
            Extraction result with data and metadata
        """
        pass
    
    @abstractmethod
    async def extract_table_data_async(
        self, 
        table_name: str, 
        schema: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract data from a specific table asynchronously.
        
        Args:
            table_name: Name of the table to extract from
            schema: Database schema name
            filters: Filter conditions
            order_by: Column to order by
            
        Returns:
            Extraction result with data and metadata
        """
        pass
    
    @abstractmethod
    def extract_tables_batch(
        self, 
        table_names: List[str],
        schema: Optional[str] = None
    ) -> List[ExtractionResult]:
        """
        Extract data from multiple tables in batches.
        
        Args:
            table_names: List of table names to extract from
            schema: Database schema name
            
        Returns:
            List of extraction results
        """
        pass
    
    @abstractmethod
    def extract_incremental(
        self, 
        table_name: str,
        schema: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract data using incremental sync.
        
        Args:
            table_name: Name of the table to extract from
            schema: Database schema name
            
        Returns:
            Extraction result with incremental data
        """
        pass
    
    @abstractmethod
    def execute_query(
        self, 
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            Extraction result
        """
        pass
    
    # Concrete helper methods
    
    def get_table_metadata(self, table_name: str, schema: Optional[str] = None) -> TableMetadata:
        """
        Get metadata for a specific table.
        
        Args:
            table_name: Name of the table
            schema: Database schema name
            
        Returns:
            Table metadata
        """
        try:
            # Use existing introspector
            table_info = self.introspector.get_table_info(table_name, schema)
            
            # Convert to our metadata format
            metadata = TableMetadata(
                name=table_name,
                schema=schema,
                columns=table_info.get("columns", []),
                primary_keys=table_info.get("primary_keys", []),
                foreign_keys=table_info.get("foreign_keys", []),
                indexes=table_info.get("indexes", []),
                row_count=self.get_row_count(table_name, schema)
            )
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for table {table_name}: {e}")
            raise ExtractionError(f"Failed to get table metadata: {e}")
    
    def discover_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Discover all tables in the database.
        
        Args:
            schema: Database schema name
            
        Returns:
            List of table names
        """
        try:
            tables = self.introspector.get_tables(schema)
            
            # Apply filtering
            if self.config.include_tables:
                tables = [t for t in tables if t in self.config.include_tables]
            
            if self.config.exclude_tables:
                tables = [t for t in tables if t not in self.config.exclude_tables]
            
            # Apply pattern matching
            if self.config.table_patterns:
                import fnmatch
                filtered_tables = []
                for table in tables:
                    for pattern in self.config.table_patterns:
                        if fnmatch.fnmatch(table, pattern):
                            filtered_tables.append(table)
                            break
                tables = filtered_tables
            
            self.logger.info(f"Discovered {len(tables)} tables")
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to discover tables: {e}")
            raise ExtractionError(f"Failed to discover tables: {e}")
    
    def get_row_count(self, table_name: str, schema: Optional[str] = None) -> int:
        """
        Get the total number of rows in a table.
        
        Args:
            table_name: Name of the table
            schema: Database schema name
            
        Returns:
            Number of rows
        """
        try:
            with self.engine.connect() as conn:
                # Build table reference
                table_ref = f"{schema}.{table_name}" if schema else table_name
                
                # Execute count query
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_ref}"))
                row_count = result.scalar()
                
                return row_count or 0
                
        except Exception as e:
            self.logger.error(f"Failed to get row count for {table_name}: {e}")
            raise ExtractionQueryError(f"Failed to get row count: {e}")
    
    def validate_connection(self) -> bool:
        """
        Validate database connection.
        
        Returns:
            True if connection is valid
        """
        try:
            health_result = self.db_manager.health_check()
            return health_result.get("status") == "healthy"
            
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    def validate_data(self, data: List[Dict[str, Any]], table_name: str) -> List[str]:
        """
        Validate extracted data quality.
        
        Args:
            data: Extracted data
            table_name: Name of the table
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if not data:
            errors.append(f"No data extracted from table {table_name}")
            return errors
        
        # Check minimum row count
        if len(data) < self.config.min_row_count:
            errors.append(f"Table {table_name} has {len(data)} rows, minimum required: {self.config.min_row_count}")
        
        # Check null percentage
        if self.config.max_null_percentage < 1.0:
            for column in data[0].keys():
                null_count = sum(1 for row in data if row.get(column) is None)
                null_percentage = null_count / len(data)
                
                if null_percentage > self.config.max_null_percentage:
                    errors.append(f"Column {column} has {null_percentage:.2%} null values, maximum allowed: {self.config.max_null_percentage:.2%}")
        
        return errors
    
    def create_extraction_id(self, table_name: str) -> str:
        """
        Create unique extraction ID.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Unique extraction ID
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{table_name}_{timestamp}_{hash(time.time()) % 10000}"
    
    def update_incremental_state(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """
        Update incremental sync state.
        
        Args:
            table_name: Name of the table
            data: Extracted data
        """
        if not data or not self.config.incremental_column:
            return
        
        # Get the latest value from the incremental column
        latest_value = max(row.get(self.config.incremental_column) for row in data)
        
        # Create or update incremental state
        state = IncrementalState(
            table_name=table_name,
            last_sync_time=datetime.utcnow(),
            last_sync_value=latest_value,
            row_count=len(data)
        )
        
        # Generate checksum if enabled
        if self.config.use_checksum:
            state.checksum = state.generate_checksum(data)
        
        self.incremental_states[table_name] = state
        self.logger.info(f"Updated incremental state for {table_name}: {latest_value}")
    
    def get_incremental_state(self, table_name: str) -> Optional[IncrementalState]:
        """
        Get incremental sync state for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Incremental state or None
        """
        return self.incremental_states.get(table_name)
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            if self.db_manager:
                self.db_manager.close()
            
            self.logger.info("RDB extractor closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing RDB extractor: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()