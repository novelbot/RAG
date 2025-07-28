"""
RDB Vector Pipeline - Complete integration of RDB extraction with vector pipeline.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from src.core.logging import LoggerMixin
from src.core.config import DatabaseConfig, get_config
from src.extraction.base import ExtractionConfig, ExtractionMode, DataFormat
from src.extraction.factory import RDBExtractorFactory
from src.pipeline.pipeline import VectorPipeline, PipelineConfig, Document, ProcessingResult
from src.pipeline.rdb_adapter import RDBPipelineConnector, RDBAdapterConfig
from src.milvus.client import MilvusClient
from src.embedding.manager import EmbeddingManager
from src.text_processing.text_cleaner import TextCleaner
from src.text_processing.text_splitter import TextSplitter
from src.text_processing.metadata_manager import MetadataManager
from src.access_control.access_control_manager import AccessControlManager


@dataclass
class RDBPipelineConfig:
    """Configuration for RDB vector pipeline."""
    
    # Database extraction settings
    database_name: str
    database_config: DatabaseConfig
    extraction_mode: ExtractionMode = ExtractionMode.FULL
    
    # Table filtering
    include_tables: Optional[List[str]] = None
    exclude_tables: Optional[List[str]] = None
    table_patterns: Optional[List[str]] = None
    
    # RDB extraction settings
    extraction_batch_size: int = 1000
    max_rows_per_table: Optional[int] = None
    extraction_timeout: int = 300
    
    # Document adapter settings
    adapter_config: RDBAdapterConfig = field(default_factory=RDBAdapterConfig)
    
    # Vector pipeline settings
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Processing settings
    max_concurrent_tables: int = 3
    max_concurrent_documents: int = 100
    collection_name: str = "rdb_documents"
    
    # Error handling
    continue_on_table_error: bool = True
    continue_on_pipeline_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Monitoring and logging
    enable_detailed_logging: bool = True
    save_intermediate_results: bool = False
    intermediate_results_path: Optional[str] = None


@dataclass
class RDBPipelineResult:
    """Result of RDB vector pipeline processing."""
    
    pipeline_id: str
    database_name: str
    total_tables: int
    processed_tables: int
    failed_tables: int
    total_documents: int
    successful_documents: int
    failed_documents: int
    processing_time: float
    start_time: datetime
    end_time: datetime
    
    # Detailed results
    table_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_result: Optional[ProcessingResult] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def table_success_rate(self) -> float:
        """Calculate table processing success rate."""
        if self.total_tables == 0:
            return 100.0
        return (self.processed_tables / self.total_tables) * 100.0
    
    @property
    def document_success_rate(self) -> float:
        """Calculate document processing success rate."""
        if self.total_documents == 0:
            return 100.0
        return (self.successful_documents / self.total_documents) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "pipeline_id": self.pipeline_id,
            "database_name": self.database_name,
            "total_tables": self.total_tables,
            "processed_tables": self.processed_tables,
            "failed_tables": self.failed_tables,
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "processing_time": self.processing_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "table_success_rate": self.table_success_rate,
            "document_success_rate": self.document_success_rate,
            "table_results": self.table_results,
            "pipeline_result": self.pipeline_result.to_dict() if self.pipeline_result else None,
            "errors": self.errors
        }


class RDBVectorPipeline(LoggerMixin):
    """
    Complete RDB to Vector Pipeline.
    
    This class orchestrates the entire process of extracting data from RDB,
    converting it to documents, and processing through the vector pipeline
    to generate embeddings and store in vector database.
    """
    
    def __init__(self, config: RDBPipelineConfig):
        """
        Initialize RDB vector pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.pipeline_id = f"rdb_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Initialized RDB vector pipeline {self.pipeline_id}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        app_config = get_config()
        
        # Create extraction config
        self.extraction_config = ExtractionConfig(
            database_config=self.config.database_config,
            mode=self.config.extraction_mode,
            batch_size=self.config.extraction_batch_size,
            max_rows=self.config.max_rows_per_table,
            timeout=self.config.extraction_timeout,
            include_tables=self.config.include_tables,
            exclude_tables=self.config.exclude_tables,
            table_patterns=self.config.table_patterns,
            output_format=DataFormat.DICT,
            include_metadata=True,
            continue_on_error=self.config.continue_on_table_error,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        
        # Initialize RDB connector
        self.rdb_connector = RDBPipelineConnector(
            extraction_config=self.extraction_config,
            adapter_config=self.config.adapter_config
        )
        
        # Initialize vector pipeline components
        self.milvus_client = MilvusClient(app_config.milvus)
        # Explicitly connect to Milvus server
        self.milvus_client.connect()
        
        # Create collection if it doesn't exist using Context7 MCP implementation
        try:
            self.milvus_client.create_collection_if_not_exists(
                collection_name=self.config.collection_name,
                dim=1024,  # Default embedding dimension
                description=f"Auto-generated collection for RDB data from {self.config.database_name}"
            )
            self.logger.info(f"Collection {self.config.collection_name} is ready")
        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {e}")
            if not self.config.continue_on_pipeline_error:
                raise
        
        # Convert embedding providers config to EmbeddingProviderConfig list
        from src.embedding.manager import EmbeddingProviderConfig
        provider_configs = []
        if isinstance(app_config.embedding_providers, dict):
            for name, embedding_config in app_config.embedding_providers.items():
                provider_config = EmbeddingProviderConfig(
                    provider=embedding_config.provider,
                    config=embedding_config,
                    priority=1,
                    enabled=True
                )
                provider_configs.append(provider_config)
        else:
            # If it's already a list, use it directly
            provider_configs = app_config.embedding_providers
        
        self.embedding_manager = EmbeddingManager(provider_configs)
        self.text_cleaner = TextCleaner()
        self.text_splitter = TextSplitter()
        self.metadata_manager = MetadataManager()
        
        # Initialize access control if configured
        self.access_control_manager = None
        if app_config.access_control:
            try:
                from src.database.base import DatabaseManager
                from src.auth.rbac import RBACManager as AuthRBACManager
                
                # Create database manager for access control
                db_manager = DatabaseManager(app_config.database)
                
                # Create auth RBAC manager
                auth_rbac_manager = AuthRBACManager(db_manager)
                
                # Initialize access control manager
                self.access_control_manager = AccessControlManager(
                    milvus_client=self.milvus_client,
                    db_manager=db_manager,
                    auth_rbac_manager=auth_rbac_manager
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize access control: {e}")
        
        # Initialize vector pipeline
        self.vector_pipeline = VectorPipeline(
            config=self.config.pipeline_config,
            milvus_client=self.milvus_client,
            embedding_manager=self.embedding_manager,
            text_cleaner=self.text_cleaner,
            text_splitter=self.text_splitter,
            metadata_manager=self.metadata_manager,
            access_control_manager=self.access_control_manager,
            collection_name=self.config.collection_name
        )
    
    async def process_all_tables(self) -> RDBPipelineResult:
        """
        Process all tables from the configured database.
        
        Returns:
            Complete pipeline processing result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting RDB vector pipeline processing for database: {self.config.database_name}")
            
            # Extract and convert documents from RDB
            documents = await self._extract_rdb_documents()
            
            if not documents:
                self.logger.warning("No documents extracted from RDB")
                return self._create_empty_result(start_time)
            
            # Process documents through vector pipeline
            pipeline_result = await self._process_through_vector_pipeline(documents)
            
            # Create final result
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            result = RDBPipelineResult(
                pipeline_id=self.pipeline_id,
                database_name=self.config.database_name,
                total_tables=len(self.table_results),
                processed_tables=sum(1 for r in self.table_results.values() if r.get("success", False)),
                failed_tables=sum(1 for r in self.table_results.values() if not r.get("success", False)),
                total_documents=len(documents),
                successful_documents=pipeline_result.successful_documents,
                failed_documents=pipeline_result.failed_documents,
                processing_time=processing_time,
                start_time=start_time,
                end_time=end_time,
                table_results=self.table_results,
                pipeline_result=pipeline_result,
                errors=self.processing_errors
            )
            
            self.logger.info(
                f"RDB vector pipeline completed: "
                f"{result.processed_tables}/{result.total_tables} tables, "
                f"{result.successful_documents}/{result.total_documents} documents processed"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"RDB vector pipeline failed: {e}")
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            # Return error result
            result = RDBPipelineResult(
                pipeline_id=self.pipeline_id,
                database_name=self.config.database_name,
                total_tables=0,
                processed_tables=0,
                failed_tables=0,
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                processing_time=processing_time,
                start_time=start_time,
                end_time=end_time,
                errors=[{"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}]
            )
            
            return result
    
    async def process_specific_tables(self, table_names: List[str]) -> RDBPipelineResult:
        """
        Process specific tables from the configured database.
        
        Args:
            table_names: List of table names to process
            
        Returns:
            Complete pipeline processing result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting RDB vector pipeline processing for tables: {table_names}")
            
            # Extract and convert documents from specific tables
            documents = await self._extract_specific_tables(table_names)
            
            if not documents:
                self.logger.warning("No documents extracted from specified tables")
                return self._create_empty_result(start_time, table_names)
            
            # Process documents through vector pipeline
            pipeline_result = await self._process_through_vector_pipeline(documents)
            
            # Create final result
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            result = RDBPipelineResult(
                pipeline_id=self.pipeline_id,
                database_name=self.config.database_name,
                total_tables=len(table_names),
                processed_tables=sum(1 for r in self.table_results.values() if r.get("success", False)),
                failed_tables=sum(1 for r in self.table_results.values() if not r.get("success", False)),
                total_documents=len(documents),
                successful_documents=pipeline_result.successful_documents,
                failed_documents=pipeline_result.failed_documents,
                processing_time=processing_time,
                start_time=start_time,
                end_time=end_time,
                table_results=self.table_results,
                pipeline_result=pipeline_result,
                errors=self.processing_errors
            )
            
            self.logger.info(
                f"RDB vector pipeline completed: "
                f"{result.processed_tables}/{result.total_tables} tables, "
                f"{result.successful_documents}/{result.total_documents} documents processed"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"RDB vector pipeline failed: {e}")
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            # Return error result
            result = RDBPipelineResult(
                pipeline_id=self.pipeline_id,
                database_name=self.config.database_name,
                total_tables=len(table_names),
                processed_tables=0,
                failed_tables=len(table_names),
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                processing_time=processing_time,
                start_time=start_time,
                end_time=end_time,
                errors=[{"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}]
            )
            
            return result
    
    async def _extract_rdb_documents(self) -> List[Document]:
        """Extract and convert all RDB tables to documents."""
        self.table_results = {}
        self.processing_errors = []
        
        try:
            # Extract and convert all tables
            documents = await self.rdb_connector.extract_and_convert_all_tables_async(
                database_name=self.config.database_name,
                max_concurrent=self.config.max_concurrent_tables
            )
            
            # Track table results (simplified - in real implementation you'd track per table)
            # For now, we'll assume success if we got documents
            if documents:
                self.table_results["all_tables"] = {
                    "success": True,
                    "document_count": len(documents),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error extracting RDB documents: {e}")
            self.processing_errors.append({
                "stage": "rdb_extraction",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            if not self.config.continue_on_table_error:
                raise
            
            return []
    
    async def _extract_specific_tables(self, table_names: List[str]) -> List[Document]:
        """Extract and convert specific RDB tables to documents."""
        self.table_results = {}
        self.processing_errors = []
        
        all_documents = []
        
        for table_name in table_names:
            try:
                self.logger.info(f"Processing table: {table_name}")
                
                # Extract table data
                result = self.rdb_connector.extractor.extract_table_data(table_name)
                
                # Convert to documents
                documents = self.rdb_connector.adapter.convert_extraction_result(
                    result, table_name, self.config.database_name
                )
                
                all_documents.extend(documents)
                
                # Track success
                self.table_results[table_name] = {
                    "success": True,
                    "document_count": len(documents),
                    "row_count": len(result.data),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self.logger.info(f"Successfully processed table {table_name}: {len(documents)} documents")
                
            except Exception as e:
                self.logger.error(f"Error processing table {table_name}: {e}")
                
                self.table_results[table_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self.processing_errors.append({
                    "stage": "rdb_extraction",
                    "table": table_name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                if not self.config.continue_on_table_error:
                    raise
        
        return all_documents
    
    async def _process_through_vector_pipeline(self, documents: List[Document]) -> ProcessingResult:
        """Process documents through the vector pipeline."""
        try:
            self.logger.info(f"Processing {len(documents)} documents through vector pipeline")
            
            # Initialize vector pipeline
            await self.vector_pipeline.initialize()
            
            # Process documents
            result = await self.vector_pipeline.process_documents(documents)
            
            # Shutdown pipeline
            await self.vector_pipeline.shutdown()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing through vector pipeline: {e}")
            
            self.processing_errors.append({
                "stage": "vector_pipeline",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Try to shutdown pipeline even on error
            try:
                await self.vector_pipeline.shutdown()
            except Exception:
                pass
            
            if not self.config.continue_on_pipeline_error:
                raise
            
            # Return empty result on error
            from src.pipeline.pipeline import ProcessingResult
            return ProcessingResult(
                pipeline_id=self.pipeline_id,
                total_documents=len(documents),
                successful_documents=0,
                failed_documents=len(documents),
                processing_time=0.0,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                errors=[{"error": str(e), "type": type(e).__name__, "timestamp": datetime.now(timezone.utc).isoformat()}]
            )
    
    def _create_empty_result(self, start_time: datetime, table_names: Optional[List[str]] = None) -> RDBPipelineResult:
        """Create an empty result for cases with no data."""
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()
        
        return RDBPipelineResult(
            pipeline_id=self.pipeline_id,
            database_name=self.config.database_name,
            total_tables=len(table_names) if table_names else 0,
            processed_tables=0,
            failed_tables=0,
            total_documents=0,
            successful_documents=0,
            failed_documents=0,
            processing_time=processing_time,
            start_time=start_time,
            end_time=end_time
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipeline components."""
        health_status = {
            "pipeline_healthy": True,
            "components": {},
            "overall_status": "healthy"
        }
        
        # Check RDB connection
        try:
            rdb_healthy = self.rdb_connector.extractor.validate_connection()
            health_status["components"]["rdb_extractor"] = {
                "healthy": rdb_healthy,
                "database": self.config.database_name
            }
            if not rdb_healthy:
                health_status["pipeline_healthy"] = False
        except Exception as e:
            health_status["components"]["rdb_extractor"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["pipeline_healthy"] = False
        
        # Check vector pipeline
        try:
            pipeline_health = await self.vector_pipeline.health_check()
            health_status["components"]["vector_pipeline"] = pipeline_health
            if not pipeline_health.get("pipeline_healthy", False):
                health_status["pipeline_healthy"] = False
        except Exception as e:
            health_status["components"]["vector_pipeline"] = {
                "healthy": False,
                "error": str(e)
            }
            health_status["pipeline_healthy"] = False
        
        # Set overall status
        if not health_status["pipeline_healthy"]:
            health_status["overall_status"] = "unhealthy"
        elif any(not comp.get("healthy", True) for comp in health_status["components"].values()):
            health_status["overall_status"] = "degraded"
        
        return health_status
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "pipeline_id": self.pipeline_id,
            "database_name": self.config.database_name,
            "collection_name": self.config.collection_name,
            "extraction_mode": self.config.extraction_mode.value,
            "table_filters": {
                "include_tables": self.config.include_tables,
                "exclude_tables": self.config.exclude_tables,
                "table_patterns": self.config.table_patterns
            },
            "pipeline_config": {
                "max_concurrent_tables": self.config.max_concurrent_tables,
                "max_concurrent_documents": self.config.max_concurrent_documents,
                "continue_on_error": self.config.continue_on_table_error
            }
        }
    
    def close(self):
        """Close all pipeline components and cleanup resources."""
        try:
            if hasattr(self, 'rdb_connector'):
                self.rdb_connector.close()
            
            # Note: vector_pipeline cleanup is handled in its own shutdown method
            
            self.logger.info("RDB vector pipeline closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing RDB vector pipeline: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Factory function for easy pipeline creation
def create_rdb_vector_pipeline(
    database_name: str,
    database_config: DatabaseConfig,
    collection_name: str = "rdb_documents",
    include_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
    **kwargs
) -> RDBVectorPipeline:
    """
    Factory function to create RDB vector pipeline with common configurations.
    
    Args:
        database_name: Name of the database
        database_config: Database connection configuration
        collection_name: Target Milvus collection name
        include_tables: Tables to include (None for all)
        exclude_tables: Tables to exclude
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RDB vector pipeline
    """
    config = RDBPipelineConfig(
        database_name=database_name,
        database_config=database_config,
        collection_name=collection_name,
        include_tables=include_tables,
        exclude_tables=exclude_tables,
        **kwargs
    )
    
    return RDBVectorPipeline(config)