"""
Generic RDB data extractor implementation.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from sqlalchemy import text, Table, MetaData, inspect, func
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from .base import BaseRDBExtractor, ExtractionResult, ExtractionConfig, TableMetadata
from .exceptions import (
    ExtractionError, ExtractionTimeoutError, ExtractionQueryError,
    ExtractionConnectionError, ExtractionValidationError
)


class GenericRDBExtractor(BaseRDBExtractor):
    """
    Generic RDB data extractor that works with any SQLAlchemy-supported database.
    
    This implementation provides basic data extraction functionality that should
    work across different database types without requiring database-specific optimizations.
    """
    
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
        extraction_id = self.create_extraction_id(table_name)
        start_time = time.time()
        
        try:
            # Get table metadata
            metadata = self.get_table_metadata(table_name, schema)
            
            # Build table reference
            table_ref = f"{schema}.{table_name}" if schema else table_name
            
            # Build query
            query = f"SELECT * FROM {table_ref}"
            
            # Add filters
            if filters:
                filter_conditions = []
                for column, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f"{column} = '{value}'")
                    else:
                        filter_conditions.append(f"{column} = {value}")
                
                if filter_conditions:
                    query += " WHERE " + " AND ".join(filter_conditions)
            
            # Add ordering
            if order_by:
                query += f" ORDER BY {order_by}"
            
            # Add limit if specified
            if self.config.max_rows:
                query += f" LIMIT {self.config.max_rows}"
            
            # Execute query in batches
            data = []
            offset = 0
            
            with self.engine.connect() as conn:
                while True:
                    # Build batch query with MySQL-compatible syntax
                    batch_query = f"{query} LIMIT {offset}, {self.config.batch_size}"
                    
                    # Execute batch
                    result = conn.execute(text(batch_query))
                    batch_data = [dict(row._mapping) for row in result]
                    
                    if not batch_data:
                        break
                    
                    data.extend(batch_data)
                    offset += self.config.batch_size
                    
                    # Check if we've reached max rows
                    if self.config.max_rows and len(data) >= self.config.max_rows:
                        data = data[:self.config.max_rows]
                        break
                    
                    # Log progress
                    if offset % (self.config.batch_size * 10) == 0:
                        self.logger.info(f"Extracted {len(data)} rows from {table_name}")
            
            # Validate data if enabled
            errors = []
            if self.config.validate_data:
                errors = self.validate_data(data, table_name)
            
            # Update incremental state if needed
            if self.config.mode.value == "incremental":
                self.update_incremental_state(table_name, data)
            
            # Update metadata with actual row count
            metadata.row_count = len(data)
            
            # Create result
            result = ExtractionResult(
                data=data,
                metadata=metadata,
                errors=errors,
                extraction_id=extraction_id,
                timestamp=datetime.utcnow()
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"Extracted {len(data)} rows from {table_name} in {execution_time:.2f}s")
            
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error extracting from {table_name}: {e}")
            raise ExtractionQueryError(f"Database error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error extracting from {table_name}: {e}")
            raise ExtractionError(f"Unexpected error: {e}")
    
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
        # For now, run the sync version in a thread pool
        # In a real implementation, this would use async SQLAlchemy
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.extract_table_data, 
            table_name, 
            schema, 
            filters, 
            order_by
        )
    
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
        results = []
        
        for table_name in table_names:
            try:
                result = self.extract_table_data(table_name, schema)
                results.append(result)
                
                # Update stats
                self.extraction_stats.tables_processed += 1
                self.extraction_stats.processed_rows += len(result.data)
                
            except Exception as e:
                self.logger.error(f"Failed to extract from table {table_name}: {e}")
                
                # Create error result
                error_result = ExtractionResult(
                    data=[],
                    metadata=None,
                    errors=[str(e)],
                    extraction_id=self.create_extraction_id(table_name),
                    timestamp=datetime.utcnow()
                )
                results.append(error_result)
                
                # Update stats
                self.extraction_stats.tables_failed += 1
                
                # Continue or stop based on configuration
                if not self.config.continue_on_error:
                    break
        
        return results
    
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
        if not self.config.incremental_column:
            raise ExtractionError("Incremental column not configured")
        
        # Get existing state
        state = self.get_incremental_state(table_name)
        
        # Build incremental filter
        filters = {}
        if state and state.last_sync_value:
            filters[self.config.incremental_column] = f"> {state.last_sync_value}"
        elif self.config.incremental_value:
            filters[self.config.incremental_column] = f"> {self.config.incremental_value}"
        
        # Extract data with incremental filter
        return self.extract_table_data(
            table_name=table_name,
            schema=schema,
            filters=filters,
            order_by=self.config.incremental_column
        )
    
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
        extraction_id = self.create_extraction_id("custom_query")
        start_time = time.time()
        
        try:
            data = []
            
            with self.engine.connect() as conn:
                if parameters:
                    result = conn.execute(text(query), parameters)
                else:
                    result = conn.execute(text(query))
                
                data = [dict(row._mapping) for row in result]
            
            # Validate data if enabled
            errors = []
            if self.config.validate_data and data:
                # Basic validation for custom queries
                if len(data) < self.config.min_row_count:
                    errors.append(f"Query returned {len(data)} rows, minimum required: {self.config.min_row_count}")
            
            # Create result
            result = ExtractionResult(
                data=data,
                metadata=None,  # No metadata for custom queries
                errors=errors,
                extraction_id=extraction_id,
                timestamp=datetime.utcnow()
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"Executed custom query and got {len(data)} rows in {execution_time:.2f}s")
            
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error executing query: {e}")
            raise ExtractionQueryError(f"Database error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {e}")
            raise ExtractionError(f"Unexpected error: {e}")
    
    def extract_all_tables(self, schema: Optional[str] = None) -> List[ExtractionResult]:
        """
        Extract data from all discoverable tables.
        
        Args:
            schema: Database schema name
            
        Returns:
            List of extraction results
        """
        try:
            # Discover tables
            tables = self.discover_tables(schema)
            
            if not tables:
                self.logger.warning(f"No tables found in schema {schema}")
                return []
            
            # Extract from all tables
            self.logger.info(f"Starting extraction from {len(tables)} tables")
            results = self.extract_tables_batch(tables, schema)
            
            # Complete stats
            self.extraction_stats.complete()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to extract from all tables: {e}")
            raise ExtractionError(f"Failed to extract from all tables: {e}")
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of extraction operations.
        
        Returns:
            Extraction summary
        """
        return {
            "config": {
                "mode": self.config.mode.value,
                "batch_size": self.config.batch_size,
                "max_rows": self.config.max_rows,
                "database_type": self.config.database_config.database_type.value
            },
            "stats": self.extraction_stats.to_dict(),
            "incremental_states": {
                table: state.to_dict() 
                for table, state in self.incremental_states.items()
            }
        }