"""
Vector Processing Pipeline - Main orchestration framework.
"""

import asyncio
import time
import yaml
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import uuid

from src.core.logging import LoggerMixin
from src.core.exceptions import PipelineError, ConfigurationError
from src.milvus.client import MilvusClient
from src.embedding.manager import EmbeddingManager
from src.text_processing.text_cleaner import TextCleaner
from src.text_processing.text_splitter import TextSplitter
from src.text_processing.metadata_manager import MetadataManager
from src.access_control.access_control_manager import AccessControlManager

from .stages import (
    PipelineStage, ProcessingStageType, StageConfig, ProcessingData, 
    ProcessingContext, ParallelPipelineStage, create_stage_config
)
from .batch_processor import BatchProcessor, BatchConfig, BatchProcessingStrategy
from .monitoring import PipelineMetrics, PipelineStatus, AlertSeverity


class PipelineMode(Enum):
    """Pipeline execution modes."""
    BATCH = "batch"  # Process all documents in batches
    STREAMING = "streaming"  # Process documents as they arrive
    HYBRID = "hybrid"  # Mix of batch and streaming


@dataclass
class Document:
    """Document input for pipeline processing."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    user_id: Optional[str] = None
    access_tags: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for the vector pipeline."""
    # Pipeline settings
    mode: PipelineMode = PipelineMode.BATCH
    enable_parallel_processing: bool = True
    max_concurrent_documents: int = 100
    checkpoint_interval: int = 100  # Save state every N documents
    
    # Stage configurations
    text_cleaning_config: StageConfig = field(default_factory=lambda: create_stage_config())
    text_splitting_config: StageConfig = field(default_factory=lambda: create_stage_config())
    embedding_config: StageConfig = field(default_factory=lambda: create_stage_config(batch_size=50))
    metadata_config: StageConfig = field(default_factory=lambda: create_stage_config())
    access_control_config: StageConfig = field(default_factory=lambda: create_stage_config())
    vector_storage_config: StageConfig = field(default_factory=lambda: create_stage_config(batch_size=100))
    
    # Batch processing
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 30.0
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Feature flags
    enable_caching: bool = True
    enable_access_control: bool = True
    enable_metadata_enrichment: bool = True
    enable_progress_tracking: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert nested dictionaries to appropriate config objects
            # This is a simplified implementation - in practice you'd want more sophisticated config parsing
            return cls(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Failed to load pipeline config: {e}")


@dataclass
class ProcessingResult:
    """Result of pipeline processing."""
    pipeline_id: str
    total_documents: int
    successful_documents: int
    failed_documents: int
    processing_time: float
    start_time: datetime
    end_time: datetime
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_data: Optional[Dict[str, Any]] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_documents == 0:
            return 100.0
        return (self.successful_documents / self.total_documents) * 100.0


# Pipeline stage implementations
class DocumentValidationStage(PipelineStage):
    """Stage for validating input documents."""
    
    async def _initialize_stage(self) -> None:
        """Initialize document validation stage."""
        self.max_content_length = self.config.custom_params.get("max_content_length", 1000000)  # 1MB
        self.required_fields = self.config.custom_params.get("required_fields", ["id", "content"])
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Validate document data."""
        doc = data.content
        
        # Check required fields
        for field in self.required_fields:
            if not hasattr(doc, field) or not getattr(doc, field):
                raise PipelineError(f"Document missing required field: {field}")
        
        # Check content length
        if len(doc.content) > self.max_content_length:
            raise PipelineError(f"Document content too long: {len(doc.content)} > {self.max_content_length}")
        
        # Validate document ID format
        if not doc.id or len(doc.id) < 1:
            raise PipelineError("Invalid document ID")
        
        return data


class TextCleaningStage(PipelineStage):
    """Stage for cleaning text content."""
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig, text_cleaner: TextCleaner):
        super().__init__(stage_type, config)
        self.text_cleaner = text_cleaner
    
    async def _initialize_stage(self) -> None:
        """Initialize text cleaning stage."""
        pass  # TextCleaner is already initialized
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Clean document text."""
        doc = data.content
        
        # Clean the text content
        cleaned_content = self.text_cleaner.clean_text(doc.content)
        
        # Update document
        doc.content = cleaned_content
        data.content = doc
        
        # Add cleaning metadata
        data.update_context(
            text_cleaning_applied=True,
            original_length=len(doc.content),
            cleaned_length=len(cleaned_content)
        )
        
        return data


class TextSplittingStage(PipelineStage):
    """Stage for splitting text into chunks."""
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig, text_splitter: TextSplitter):
        super().__init__(stage_type, config)
        self.text_splitter = text_splitter
    
    async def _initialize_stage(self) -> None:
        """Initialize text splitting stage."""
        pass  # TextSplitter is already initialized
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Split document into chunks."""
        doc = data.content
        
        # Split the text
        chunks = self.text_splitter.split_text(doc.content)
        
        # Create chunk documents
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc.id}_chunk_{i}",
                content=chunk,
                metadata={**doc.metadata, "chunk_index": i, "parent_document_id": doc.id},
                source_path=doc.source_path,
                user_id=doc.user_id,
                access_tags=doc.access_tags.copy()
            )
            chunk_docs.append(chunk_doc)
        
        # Update processing data with chunks
        data.content = chunk_docs
        data.update_context(
            text_splitting_applied=True,
            original_document_id=doc.id,
            chunk_count=len(chunks)
        )
        
        return data


class EmbeddingGenerationStage(ParallelPipelineStage):
    """Stage for generating embeddings."""
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig, embedding_manager: EmbeddingManager):
        super().__init__(stage_type, config)
        self.embedding_manager = embedding_manager
    
    async def _initialize_stage(self) -> None:
        """Initialize embedding generation stage."""
        pass  # EmbeddingManager is already initialized
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Generate embeddings for document chunks."""
        from src.embedding.base import EmbeddingRequest
        
        chunks = data.content if isinstance(data.content, list) else [data.content]
        
        # Prepare embedding requests
        texts = [chunk.content for chunk in chunks]
        request = EmbeddingRequest(input=texts)
        
        # Generate embeddings
        response = await self.embedding_manager.generate_embeddings_async(request)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, response.embeddings):
            chunk.metadata["embedding"] = embedding
            chunk.metadata["embedding_model"] = response.model
            chunk.metadata["embedding_dimensions"] = response.dimensions
        
        data.content = chunks
        data.update_context(
            embedding_generation_applied=True,
            embedding_model=response.model,
            embedding_dimensions=response.dimensions,
            embedding_provider=response.metadata.get("provider"),
            embedding_cost=response.metadata.get("cost", 0.0)
        )
        
        return data


class MetadataEnrichmentStage(PipelineStage):
    """Stage for enriching metadata."""
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig, metadata_manager: MetadataManager):
        super().__init__(stage_type, config)
        self.metadata_manager = metadata_manager
    
    async def _initialize_stage(self) -> None:
        """Initialize metadata enrichment stage."""
        pass  # MetadataManager is already initialized
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Enrich document metadata."""
        chunks = data.content if isinstance(data.content, list) else [data.content]
        
        for chunk in chunks:
            # Enrich metadata using metadata manager
            enriched_metadata = self.metadata_manager.enrich_metadata(
                chunk.content, 
                chunk.metadata,
                source_path=chunk.source_path
            )
            chunk.metadata.update(enriched_metadata)
            
            # Add processing metadata
            chunk.metadata.update({
                "processed_at": datetime.utcnow().isoformat(),
                "processing_pipeline_id": data.context.batch_id,
                "document_id": data.context.document_id
            })
        
        data.content = chunks
        data.update_context(metadata_enrichment_applied=True)
        
        return data


class AccessControlTaggingStage(PipelineStage):
    """Stage for applying access control tags."""
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig, access_control_manager: AccessControlManager):
        super().__init__(stage_type, config)
        self.access_control_manager = access_control_manager
    
    async def _initialize_stage(self) -> None:
        """Initialize access control tagging stage."""
        pass  # AccessControlManager is already initialized
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Apply access control tags."""
        chunks = data.content if isinstance(data.content, list) else [data.content]
        
        for chunk in chunks:
            # Generate access control tags based on content and user
            access_tags = await self._generate_access_tags(chunk, data.context)
            chunk.access_tags.extend(access_tags)
            chunk.metadata["access_control_tags"] = chunk.access_tags
        
        data.content = chunks
        data.update_context(access_control_tagging_applied=True)
        
        return data
    
    async def _generate_access_tags(self, chunk: Document, context: ProcessingContext) -> List[str]:
        """Generate access control tags for a chunk."""
        tags = []
        
        # Add user-based tags
        if context.user_id:
            tags.append(f"user:{context.user_id}")
        
        # Add source-based tags
        if chunk.source_path:
            tags.append(f"source:{Path(chunk.source_path).stem}")
        
        # Add content-based tags (simplified example)
        if "confidential" in chunk.content.lower():
            tags.append("classification:confidential")
        else:
            tags.append("classification:public")
        
        return tags


class VectorStorageStage(ParallelPipelineStage):
    """Stage for storing vectors in Milvus."""
    
    def __init__(self, stage_type: ProcessingStageType, config: StageConfig, milvus_client: MilvusClient, collection_name: str):
        super().__init__(stage_type, config)
        self.milvus_client = milvus_client
        self.collection_name = collection_name
    
    async def _initialize_stage(self) -> None:
        """Initialize vector storage stage."""
        # Ensure collection exists and is loaded
        if not self.milvus_client.has_collection(self.collection_name):
            raise PipelineError(f"Collection {self.collection_name} does not exist")
    
    async def _process_data(self, data: ProcessingData) -> ProcessingData:
        """Store vectors in Milvus."""
        from pymilvus import Collection
        
        chunks = data.content if isinstance(data.content, list) else [data.content]
        
        # Prepare data for insertion
        entities = []
        for chunk in chunks:
            if "embedding" not in chunk.metadata:
                raise PipelineError(f"Chunk {chunk.id} has no embedding")
            
            entity = {
                "id": chunk.id,
                "content": chunk.content,
                "embedding": chunk.metadata["embedding"],
                "metadata": chunk.metadata,
                "access_tags": chunk.access_tags
            }
            entities.append(entity)
        
        # Insert into Milvus
        collection = Collection(self.collection_name, using=self.milvus_client.alias)
        
        # Prepare data in the format expected by Milvus
        data_to_insert = [
            [entity["id"] for entity in entities],  # IDs
            [entity["content"] for entity in entities],  # Content
            [entity["embedding"] for entity in entities],  # Embeddings
            [entity["metadata"] for entity in entities],  # Metadata
            [entity["access_tags"] for entity in entities],  # Access tags
        ]
        
        insert_result = collection.insert(data_to_insert)
        
        # Update context with storage information
        data.update_context(
            vector_storage_applied=True,
            collection_name=self.collection_name,
            inserted_count=len(entities),
            milvus_ids=insert_result.primary_keys
        )
        
        return data


class VectorPipeline(LoggerMixin):
    """
    Main Vector Processing Pipeline.
    
    Orchestrates the complete document-to-vector workflow with:
    - Modular, configurable stages
    - Batch processing optimization
    - Comprehensive monitoring
    - Error handling and recovery
    - Progress tracking
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        milvus_client: MilvusClient,
        embedding_manager: EmbeddingManager,
        text_cleaner: TextCleaner,
        text_splitter: TextSplitter,
        metadata_manager: MetadataManager,
        access_control_manager: Optional[AccessControlManager] = None,
        collection_name: str = "documents"
    ):
        """
        Initialize vector pipeline.
        
        Args:
            config: Pipeline configuration
            milvus_client: Milvus client for vector storage
            embedding_manager: Embedding generation manager
            text_cleaner: Text cleaning component
            text_splitter: Text splitting component
            metadata_manager: Metadata enrichment component
            access_control_manager: Access control component (optional)
            collection_name: Milvus collection name for storage
        """
        self.config = config
        self.milvus_client = milvus_client
        self.embedding_manager = embedding_manager
        self.text_cleaner = text_cleaner
        self.text_splitter = text_splitter
        self.metadata_manager = metadata_manager
        self.access_control_manager = access_control_manager
        self.collection_name = collection_name
        
        # Pipeline state
        self.pipeline_id = str(uuid.uuid4())
        self.is_initialized = False
        self.is_running = False
        self.status = PipelineStatus.INITIALIZING
        
        # Processing components
        self.stages: List[PipelineStage] = []
        self.batch_processor: Optional[BatchProcessor] = None
        self.metrics: Optional[PipelineMetrics] = None
        
        # Progress tracking
        self.processed_documents = 0
        self.failed_documents = 0
        self.current_batch_id: Optional[str] = None
        
    async def initialize(self) -> None:
        """Initialize the pipeline and all its components."""
        if self.is_initialized:
            return
        
        try:
            self.logger.info(f"Initializing pipeline {self.pipeline_id}")
            
            # Initialize monitoring
            if self.config.enable_monitoring:
                self.metrics = PipelineMetrics(
                    monitoring_interval=self.config.monitoring_interval
                )
                await self.metrics.start_monitoring()
            
            # Initialize batch processor
            self.batch_processor = BatchProcessor(self.config.batch_config)
            await self.batch_processor.start()
            
            # Create and initialize pipeline stages
            await self._create_pipeline_stages()
            
            # Initialize all stages
            for stage in self.stages:
                await stage.initialize()
                if self.metrics:
                    self.metrics.update_component_health(
                        f"stage_{stage.stage_type.value}", 
                        True, 
                        stage_type=stage.stage_type.value
                    )
            
            self.is_initialized = True
            self.status = PipelineStatus.RUNNING
            self.logger.info(f"Pipeline {self.pipeline_id} initialized successfully")
            
        except Exception as e:
            self.status = PipelineStatus.ERROR
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise PipelineError(f"Pipeline initialization failed: {e}")
    
    async def _create_pipeline_stages(self) -> None:
        """Create and configure pipeline stages."""
        self.stages = []
        
        # Document validation stage
        validation_stage = DocumentValidationStage(
            ProcessingStageType.DOCUMENT_VALIDATION,
            self.config.text_cleaning_config  # Reusing config for simplicity
        )
        self.stages.append(validation_stage)
        
        # Text cleaning stage
        cleaning_stage = TextCleaningStage(
            ProcessingStageType.TEXT_CLEANING,
            self.config.text_cleaning_config,
            self.text_cleaner
        )
        self.stages.append(cleaning_stage)
        
        # Text splitting stage
        splitting_stage = TextSplittingStage(
            ProcessingStageType.TEXT_SPLITTING,
            self.config.text_splitting_config,
            self.text_splitter
        )
        self.stages.append(splitting_stage)
        
        # Embedding generation stage
        embedding_stage = EmbeddingGenerationStage(
            ProcessingStageType.EMBEDDING_GENERATION,
            self.config.embedding_config,
            self.embedding_manager
        )
        self.stages.append(embedding_stage)
        
        # Metadata enrichment stage (if enabled)
        if self.config.enable_metadata_enrichment:
            metadata_stage = MetadataEnrichmentStage(
                ProcessingStageType.METADATA_ENRICHMENT,
                self.config.metadata_config,
                self.metadata_manager
            )
            self.stages.append(metadata_stage)
        
        # Access control tagging stage (if enabled and manager provided)
        if self.config.enable_access_control and self.access_control_manager:
            access_control_stage = AccessControlTaggingStage(
                ProcessingStageType.ACCESS_CONTROL_TAGGING,
                self.config.access_control_config,
                self.access_control_manager
            )
            self.stages.append(access_control_stage)
        
        # Vector storage stage
        storage_stage = VectorStorageStage(
            ProcessingStageType.VECTOR_STORAGE,
            self.config.vector_storage_config,
            self.milvus_client,
            self.collection_name
        )
        self.stages.append(storage_stage)
    
    async def process_documents(self, documents: List[Document]) -> ProcessingResult:
        """
        Process a list of documents through the pipeline.
        
        Args:
            documents: List of documents to process
            
        Returns:
            ProcessingResult: Results of the processing
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        batch_id = str(uuid.uuid4())
        self.current_batch_id = batch_id
        
        try:
            self.logger.info(f"Starting document processing: {len(documents)} documents, batch {batch_id}")
            
            # Process documents based on mode
            if self.config.mode == PipelineMode.BATCH:
                result = await self._process_batch_mode(documents, batch_id)
            elif self.config.mode == PipelineMode.STREAMING:
                result = await self._process_streaming_mode(documents, batch_id)
            else:  # HYBRID
                result = await self._process_hybrid_mode(documents, batch_id)
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create processing result
            processing_result = ProcessingResult(
                pipeline_id=self.pipeline_id,
                total_documents=len(documents),
                successful_documents=result["successful"],
                failed_documents=result["failed"],
                processing_time=processing_time,
                start_time=start_time,
                end_time=end_time,
                errors=result.get("errors", []),
                metrics=self.get_metrics() if self.metrics else {}
            )
            
            self.logger.info(
                f"Document processing completed: "
                f"{processing_result.successful_documents}/{processing_result.total_documents} successful "
                f"in {processing_time:.2f}s"
            )
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            self.status = PipelineStatus.ERROR
            raise PipelineError(f"Document processing failed: {e}")
        finally:
            self.current_batch_id = None
    
    async def _process_batch_mode(self, documents: List[Document], batch_id: str) -> Dict[str, Any]:
        """Process documents in batch mode."""
        successful = 0
        failed = 0
        errors = []
        
        # Create processing data objects
        processing_items = []
        for doc in documents:
            context = ProcessingContext(
                document_id=doc.id,
                batch_id=batch_id,
                user_id=doc.user_id,
                source_path=doc.source_path,
                metadata=doc.metadata.copy(),
                access_control_tags=doc.access_tags.copy()
            )
            processing_data = ProcessingData(content=doc, context=context)
            processing_items.append(processing_data)
        
        # Process through each stage
        current_items = processing_items
        
        for stage in self.stages:
            try:
                self.logger.debug(f"Processing {len(current_items)} items through {stage.stage_type.value}")
                
                if isinstance(stage, ParallelPipelineStage) and len(current_items) > 1:
                    # Use parallel processing for compatible stages
                    processed_items = await stage.process_batch(current_items)
                else:
                    # Process sequentially
                    processed_items = []
                    for item in current_items:
                        try:
                            processed_item = await stage.process(item)
                            processed_items.append(processed_item)
                        except Exception as e:
                            if not self.config.continue_on_error:
                                raise
                            errors.append({
                                "document_id": item.context.document_id,
                                "stage": stage.stage_type.value,
                                "error": str(e)
                            })
                            failed += 1
                
                # Flatten results if needed (for text splitting stage)
                flattened_items = []
                for item in processed_items:
                    if isinstance(item.content, list):
                        # Handle chunks from text splitting
                        for chunk in item.content:
                            chunk_context = ProcessingContext(
                                document_id=chunk.id,
                                batch_id=batch_id,
                                user_id=item.context.user_id,
                                source_path=item.context.source_path,
                                metadata=item.context.metadata.copy(),
                                access_control_tags=item.context.access_control_tags.copy()
                            )
                            chunk_data = ProcessingData(content=chunk, context=chunk_context)
                            chunk_data.stage_outputs = item.stage_outputs.copy()
                            flattened_items.append(chunk_data)
                    else:
                        flattened_items.append(item)
                
                current_items = flattened_items
                
                # Update metrics
                if self.metrics:
                    self.metrics.update_component_health(
                        f"stage_{stage.stage_type.value}",
                        True,
                        processed_items=len(processed_items)
                    )
                
            except Exception as e:
                self.logger.error(f"Stage {stage.stage_type.value} failed: {e}")
                if self.metrics:
                    self.metrics.update_component_health(
                        f"stage_{stage.stage_type.value}",
                        False,
                        error=str(e)
                    )
                raise
        
        successful = len(current_items)
        
        return {
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "processed_items": current_items
        }
    
    async def _process_streaming_mode(self, documents: List[Document], batch_id: str) -> Dict[str, Any]:
        """Process documents in streaming mode (one by one)."""
        successful = 0
        failed = 0
        errors = []
        
        for doc in documents:
            try:
                # Process single document
                single_doc_result = await self._process_batch_mode([doc], batch_id)
                successful += single_doc_result["successful"]
                failed += single_doc_result["failed"]
                errors.extend(single_doc_result["errors"])
                
                # Update metrics
                if self.metrics:
                    self.metrics.record_document_processed(
                        processing_time=0.1,  # Placeholder - would need actual timing
                        success=single_doc_result["successful"] > 0
                    )
                
            except Exception as e:
                failed += 1
                errors.append({
                    "document_id": doc.id,
                    "error": str(e)
                })
                if self.metrics:
                    self.metrics.record_document_processed(
                        processing_time=0.1,
                        success=False
                    )
        
        return {
            "successful": successful,
            "failed": failed,
            "errors": errors
        }
    
    async def _process_hybrid_mode(self, documents: List[Document], batch_id: str) -> Dict[str, Any]:
        """Process documents in hybrid mode (optimized batching)."""
        # For now, just use batch mode - could be enhanced with more sophisticated logic
        return await self._process_batch_mode(documents, batch_id)
    
    async def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        if not self.is_running:
            return
        
        self.logger.info(f"Shutting down pipeline {self.pipeline_id}")
        self.status = PipelineStatus.STOPPING
        
        try:
            # Cleanup stages
            for stage in self.stages:
                await stage.cleanup()
            
            # Stop batch processor
            if self.batch_processor:
                await self.batch_processor.stop()
            
            # Stop monitoring
            if self.metrics:
                await self.metrics.stop_monitoring()
            
            self.status = PipelineStatus.STOPPED
            self.is_running = False
            self.logger.info(f"Pipeline {self.pipeline_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during pipeline shutdown: {e}")
            self.status = PipelineStatus.ERROR
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "current_batch_id": self.current_batch_id,
            "stage_count": len(self.stages),
            "collection_name": self.collection_name
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        metrics = {
            "pipeline": self.get_status(),
            "batch_processor": self.batch_processor.get_metrics() if self.batch_processor else {},
            "stages": [stage.get_metrics() for stage in self.stages]
        }
        
        if self.metrics:
            metrics.update({
                "monitoring": self.metrics.get_current_status(),
                "performance": self.metrics.get_performance_summary(),
                "resources": self.metrics.get_resource_summary(),
                "costs": self.metrics.get_cost_summary(),
                "alerts": self.metrics.get_alerts(),
                "component_health": self.metrics.get_component_health_summary()
            })
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "pipeline_healthy": True,
            "components": {},
            "overall_status": "healthy"
        }
        
        # Check pipeline status
        health_status["components"]["pipeline"] = {
            "healthy": self.status in [PipelineStatus.RUNNING, PipelineStatus.PAUSED],
            "status": self.status.value,
            "initialized": self.is_initialized
        }
        
        # Check stages
        for stage in self.stages:
            stage_health = stage.get_health_status()
            health_status["components"][f"stage_{stage.stage_type.value}"] = stage_health
            if not stage_health["healthy"]:
                health_status["pipeline_healthy"] = False
        
        # Check batch processor
        if self.batch_processor:
            batch_metrics = self.batch_processor.get_metrics()
            health_status["components"]["batch_processor"] = {
                "healthy": batch_metrics["is_running"],
                "queue_size": batch_metrics["queue_size"],
                "active_jobs": batch_metrics["active_jobs"]
            }
        
        # Check external dependencies
        try:
            milvus_ping = self.milvus_client.ping()
            health_status["components"]["milvus"] = {
                "healthy": milvus_ping["status"] == "healthy",
                "response_time": milvus_ping["response_time"]
            }
        except Exception as e:
            health_status["components"]["milvus"] = {
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