"""
RDB to Document Adapter - Converts RDB data to pipeline-compatible Document objects.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field

from src.core.logging import LoggerMixin
from src.extraction.base import BaseRDBExtractor, ExtractionResult, ExtractionConfig
from src.extraction.factory import RDBExtractorFactory
from src.pipeline.pipeline import Document
from src.core.config import DatabaseConfig


@dataclass
class RDBAdapterConfig:
    """Configuration for RDB to Document adapter."""
    
    # Content formatting
    content_format: str = "structured"  # "structured", "json", "plain"
    include_table_name: bool = True
    include_column_names: bool = True
    field_separator: str = "\n"
    key_value_separator: str = ": "
    
    # Document metadata
    include_extraction_metadata: bool = True
    include_table_metadata: bool = True
    include_row_index: bool = True
    
    # Content filtering
    exclude_null_values: bool = True
    exclude_empty_strings: bool = True
    exclude_columns: List[str] = field(default_factory=list)
    
    # Document ID generation
    id_prefix: str = "rdb"
    include_table_in_id: bool = True
    include_timestamp_in_id: bool = False
    
    # Processing options
    max_content_length: Optional[int] = None
    truncate_long_content: bool = True
    batch_size: int = 1000


class RDBDocumentAdapter(LoggerMixin):
    """
    Adapter for converting RDB extraction results to Document objects.
    
    This adapter handles the conversion of database rows to Document objects
    that can be processed by the vector pipeline, with configurable formatting
    and metadata handling.
    """
    
    def __init__(self, config: RDBAdapterConfig):
        """
        Initialize RDB document adapter.
        
        Args:
            config: Adapter configuration
        """
        self.config = config
        self.logger.info("Initialized RDB document adapter")
    
    def convert_extraction_result(
        self, 
        result: ExtractionResult, 
        table_name: str,
        database_name: Optional[str] = None
    ) -> List[Document]:
        """
        Convert an extraction result to Document objects.
        
        Args:
            result: Extraction result from RDB extractor
            table_name: Name of the source table
            database_name: Name of the source database
            
        Returns:
            List of Document objects
        """
        if not result.data:
            self.logger.warning(f"No data in extraction result for table {table_name}")
            return []
        
        documents = []
        timestamp = datetime.utcnow().isoformat()
        
        for i, row_data in enumerate(result.data):
            try:
                # Generate document ID
                doc_id = self._generate_document_id(table_name, i, database_name)
                
                # Generate content
                content = self._format_row_content(row_data, table_name)
                
                # Prepare metadata
                metadata = self._prepare_metadata(
                    row_data, table_name, database_name, result, i, timestamp
                )
                
                # Create document
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    source_path=f"{database_name or 'unknown'}/{table_name}" if database_name else table_name
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.error(f"Error converting row {i} from table {table_name}: {e}")
                continue
        
        self.logger.info(f"Converted {len(documents)} rows from table {table_name} to documents")
        return documents
    
    def convert_multiple_tables(
        self,
        extractor: BaseRDBExtractor,
        table_names: List[str],
        database_name: Optional[str] = None
    ) -> List[Document]:
        """
        Convert multiple tables to Document objects.
        
        Args:
            extractor: RDB extractor instance
            table_names: List of table names to convert
            database_name: Name of the source database
            
        Returns:
            List of Document objects from all tables
        """
        all_documents = []
        
        for table_name in table_names:
            try:
                self.logger.info(f"Extracting and converting table: {table_name}")
                
                # Extract table data
                result = extractor.extract_table_data(table_name)
                
                # Convert to documents
                documents = self.convert_extraction_result(result, table_name, database_name)
                all_documents.extend(documents)
                
            except Exception as e:
                self.logger.error(f"Error processing table {table_name}: {e}")
                continue
        
        self.logger.info(f"Converted {len(all_documents)} total documents from {len(table_names)} tables")
        return all_documents
    
    async def convert_multiple_tables_async(
        self,
        extractor: BaseRDBExtractor,
        table_names: List[str],
        database_name: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[Document]:
        """
        Convert multiple tables to Document objects asynchronously.
        
        Args:
            extractor: RDB extractor instance
            table_names: List of table names to convert
            database_name: Name of the source database
            max_concurrent: Maximum concurrent table extractions
            
        Returns:
            List of Document objects from all tables
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_and_convert_table(table_name: str) -> List[Document]:
            async with semaphore:
                try:
                    self.logger.info(f"Extracting and converting table: {table_name}")
                    
                    # Extract table data asynchronously
                    result = await extractor.extract_table_data_async(table_name)
                    
                    # Convert to documents
                    documents = self.convert_extraction_result(result, table_name, database_name)
                    return documents
                    
                except Exception as e:
                    self.logger.error(f"Error processing table {table_name}: {e}")
                    return []
        
        # Process all tables concurrently
        tasks = [extract_and_convert_table(table_name) for table_name in table_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_documents = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
        
        self.logger.info(f"Converted {len(all_documents)} total documents from {len(table_names)} tables")
        return all_documents
    
    def _generate_document_id(
        self, 
        table_name: str, 
        row_index: int, 
        database_name: Optional[str] = None
    ) -> str:
        """Generate a unique document ID."""
        parts = [self.config.id_prefix]
        
        if database_name and self.config.include_table_in_id:
            parts.append(database_name)
        
        if self.config.include_table_in_id:
            parts.append(table_name)
        
        if self.config.include_row_index:
            parts.append(str(row_index))
        
        if self.config.include_timestamp_in_id:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)
        
        return "_".join(parts)
    
    def _format_row_content(self, row_data: Dict[str, Any], table_name: str) -> str:
        """Format row data into document content."""
        # Filter out excluded columns and values
        filtered_data = {}
        for key, value in row_data.items():
            if key in self.config.exclude_columns:
                continue
            if self.config.exclude_null_values and value is None:
                continue
            if self.config.exclude_empty_strings and value == "":
                continue
            filtered_data[key] = value
        
        if self.config.content_format == "json":
            import json
            content = json.dumps(filtered_data, ensure_ascii=False, default=str)
        
        elif self.config.content_format == "plain":
            # Simple concatenation of values
            values = [str(v) for v in filtered_data.values() if v is not None]
            content = " ".join(values)
        
        else:  # structured format (default)
            content_parts = []
            
            # Add table name if configured
            if self.config.include_table_name:
                content_parts.append(f"Table: {table_name}")
            
            # Add field-value pairs
            if self.config.include_column_names:
                for key, value in filtered_data.items():
                    if value is not None:
                        content_parts.append(f"{key}{self.config.key_value_separator}{value}")
            else:
                # Just values
                for value in filtered_data.values():
                    if value is not None:
                        content_parts.append(str(value))
            
            content = self.config.field_separator.join(content_parts)
        
        # Apply content length limits
        if self.config.max_content_length and len(content) > self.config.max_content_length:
            if self.config.truncate_long_content:
                content = content[:self.config.max_content_length] + "..."
                self.logger.warning(f"Content truncated to {self.config.max_content_length} characters")
            else:
                self.logger.warning(f"Content length {len(content)} exceeds limit {self.config.max_content_length}")
        
        return content
    
    def _prepare_metadata(
        self,
        row_data: Dict[str, Any],
        table_name: str,
        database_name: Optional[str],
        extraction_result: ExtractionResult,
        row_index: int,
        timestamp: str
    ) -> Dict[str, Any]:
        """Prepare metadata for the document."""
        metadata = {
            "source_type": "database",
            "table_name": table_name,
            "row_index": row_index,
            "ingested_at": timestamp,
            "adapter_config": {
                "content_format": self.config.content_format,
                "include_table_name": self.config.include_table_name,
                "include_column_names": self.config.include_column_names
            }
        }
        
        # Add database name if provided
        if database_name:
            metadata["database_name"] = database_name
        
        # Add original row data for reference
        metadata["row_data"] = row_data
        
        # Add extraction metadata if configured
        if self.config.include_extraction_metadata:
            metadata["extraction"] = {
                "extraction_id": extraction_result.extraction_id,
                "timestamp": extraction_result.timestamp.isoformat() if extraction_result.timestamp else None,
                "errors": extraction_result.errors
            }
        
        # Add table metadata if configured and available
        if self.config.include_table_metadata and extraction_result.metadata:
            # Convert TableInfo to dict using dataclass fields
            import dataclasses
            if dataclasses.is_dataclass(extraction_result.metadata):
                metadata["table_metadata"] = dataclasses.asdict(extraction_result.metadata)
            else:
                # Fallback for non-dataclass objects
                metadata["table_metadata"] = str(extraction_result.metadata)
        
        return metadata


class RDBPipelineConnector(LoggerMixin):
    """
    High-level connector for integrating RDB extraction with vector pipeline.
    
    This class provides a simplified interface for extracting data from RDB
    and converting it to documents ready for pipeline processing.
    """
    
    def __init__(
        self,
        extraction_config: ExtractionConfig,
        adapter_config: Optional[RDBAdapterConfig] = None
    ):
        """
        Initialize RDB pipeline connector.
        
        Args:
            extraction_config: Configuration for RDB extraction
            adapter_config: Configuration for document adapter
        """
        self.extraction_config = extraction_config
        self.adapter_config = adapter_config or RDBAdapterConfig()
        
        # Create extractor and adapter
        self.extractor = RDBExtractorFactory.create(extraction_config)
        self.adapter = RDBDocumentAdapter(self.adapter_config)
        
        self.logger.info("Initialized RDB pipeline connector")
    
    def extract_and_convert_all_tables(
        self,
        database_name: Optional[str] = None
    ) -> List[Document]:
        """
        Extract and convert all available tables.
        
        Args:
            database_name: Name of the source database for metadata
            
        Returns:
            List of Document objects from all tables
        """
        try:
            # Discover tables
            table_names = self.extractor.discover_tables()
            
            if not table_names:
                self.logger.warning("No tables found in database")
                return []
            
            self.logger.info(f"Found {len(table_names)} tables to process")
            
            # Extract and convert all tables
            documents = self.adapter.convert_multiple_tables(
                self.extractor, table_names, database_name
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error extracting and converting tables: {e}")
            raise
    
    async def extract_and_convert_all_tables_async(
        self,
        database_name: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[Document]:
        """
        Extract and convert all available tables asynchronously.
        
        Args:
            database_name: Name of the source database for metadata
            max_concurrent: Maximum concurrent table extractions
            
        Returns:
            List of Document objects from all tables
        """
        try:
            # Discover tables
            table_names = self.extractor.discover_tables()
            
            if not table_names:
                self.logger.warning("No tables found in database")
                return []
            
            self.logger.info(f"Found {len(table_names)} tables to process")
            
            # Extract and convert all tables asynchronously
            documents = await self.adapter.convert_multiple_tables_async(
                self.extractor, table_names, database_name, max_concurrent
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error extracting and converting tables: {e}")
            raise
    
    def extract_and_convert_tables(
        self,
        table_names: List[str],
        database_name: Optional[str] = None
    ) -> List[Document]:
        """
        Extract and convert specific tables.
        
        Args:
            table_names: List of table names to process
            database_name: Name of the source database for metadata
            
        Returns:
            List of Document objects from specified tables
        """
        try:
            documents = self.adapter.convert_multiple_tables(
                self.extractor, table_names, database_name
            )
            return documents
            
        except Exception as e:
            self.logger.error(f"Error extracting and converting specified tables: {e}")
            raise
    
    def close(self):
        """Close the RDB extractor and cleanup resources."""
        try:
            if self.extractor:
                self.extractor.close()
            self.logger.info("RDB pipeline connector closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing RDB pipeline connector: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()