"""
Metadata Management for Text Processing and Chunking.

This module provides comprehensive metadata management capabilities for tracking
document source, chunk position, creation timestamps, and content characteristics
throughout the text processing pipeline.
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum

from src.core.logging import LoggerMixin
from .exceptions import MetadataError, InvalidConfigurationError


class MetadataScope(Enum):
    """Scope of metadata application."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    BOTH = "both"


class MetadataType(Enum):
    """Types of metadata."""
    SOURCE = "source"
    PROCESSING = "processing"
    CONTENT = "content"
    TEMPORAL = "temporal"
    TECHNICAL = "technical"
    CUSTOM = "custom"


@dataclass
class ChunkMetadata:
    """Comprehensive metadata for text chunks."""
    
    # Core identification
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_index: Optional[int] = None
    parent_document_id: Optional[str] = None
    
    # Source information
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    source_encoding: Optional[str] = None
    
    # Position and structure
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    
    # Content characteristics
    content_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    language: Optional[str] = None
    content_hash: Optional[str] = None
    
    # Processing information
    chunking_strategy: Optional[str] = None
    chunking_parameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    processing_timestamp: Optional[datetime] = None
    processing_version: Optional[str] = None
    
    # Quality metrics
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    coherence_score: Optional[float] = None
    completeness_score: Optional[float] = None
    
    # Relationships
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    related_chunks: List[str] = field(default_factory=list)
    
    # Custom metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set processing timestamp if not provided."""
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        result = asdict(self)
        
        # Handle datetime serialization
        if self.processing_timestamp:
            result['processing_timestamp'] = self.processing_timestamp.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary with proper deserialization."""
        # Handle datetime deserialization
        if 'processing_timestamp' in data and isinstance(data['processing_timestamp'], str):
            data['processing_timestamp'] = datetime.fromisoformat(data['processing_timestamp'])
        
        return cls(**data)
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate and set content hash."""
        self.content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return self.content_hash
    
    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag if present."""
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def set_custom_field(self, key: str, value: Any) -> None:
        """Set a custom metadata field."""
        self.custom_fields[key] = value
    
    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """Get a custom metadata field."""
        return self.custom_fields.get(key, default)


@dataclass
class MetadataConfig:
    """Configuration for metadata management."""
    
    # Basic settings
    preserve_source_metadata: bool = True
    generate_chunk_ids: bool = True
    calculate_content_hashes: bool = True
    track_relationships: bool = True
    
    # Content analysis
    analyze_content_characteristics: bool = True
    detect_language: bool = False
    calculate_quality_metrics: bool = False
    
    # Processing tracking
    track_processing_steps: bool = True
    include_processing_timestamp: bool = True
    include_chunking_parameters: bool = True
    
    # Inheritance rules
    inherit_custom_fields: bool = True
    inherit_tags: bool = True
    custom_inheritance_rules: Dict[str, bool] = field(default_factory=dict)
    
    # Validation
    validate_metadata: bool = True
    required_fields: List[str] = field(default_factory=list)
    
    # Serialization
    metadata_format: str = "json"  # "json", "yaml", "pickle"
    pretty_print: bool = True
    
    # Custom extractors
    custom_extractors: Dict[str, Callable] = field(default_factory=dict)


class MetadataManager(LoggerMixin):
    """
    Comprehensive metadata management system for text processing.
    
    Manages document and chunk metadata throughout the text processing pipeline,
    including inheritance, enrichment, validation, and serialization.
    """
    
    def __init__(self, config: Optional[MetadataConfig] = None):
        """
        Initialize metadata manager.
        
        Args:
            config: Metadata management configuration
        """
        self.config = config or MetadataConfig()
        
        # Metadata storage
        self._document_metadata: Dict[str, Dict[str, Any]] = {}
        self._chunk_metadata: Dict[str, ChunkMetadata] = {}
        
        # Relationship tracking
        self._chunk_relationships: Dict[str, List[str]] = {}
        
        self.logger.info("Metadata manager initialized")
    
    def create_document_metadata(self, 
                                document_id: str,
                                source_info: Optional[Dict[str, Any]] = None,
                                **kwargs) -> Dict[str, Any]:
        """
        Create metadata for a document.
        
        Args:
            document_id: Unique document identifier
            source_info: Source information
            **kwargs: Additional metadata fields
            
        Returns:
            Document metadata dictionary
        """
        try:
            metadata = {
                "document_id": document_id,
                "creation_timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_version": "1.0",
                **kwargs
            }
            
            if source_info:
                metadata.update(source_info)
            
            # Apply custom extractors
            for extractor_name, extractor_func in self.config.custom_extractors.items():
                try:
                    extracted_data = extractor_func(metadata)
                    if extracted_data:
                        metadata[extractor_name] = extracted_data
                except Exception as e:
                    self.logger.warning(f"Custom extractor '{extractor_name}' failed: {e}")
            
            # Validate if configured
            if self.config.validate_metadata:
                self._validate_metadata(metadata, MetadataScope.DOCUMENT)
            
            # Store document metadata
            self._document_metadata[document_id] = metadata
            
            self.logger.debug(f"Created document metadata for: {document_id}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create document metadata: {e}")
            raise MetadataError(f"Failed to create document metadata: {e}")
    
    def create_chunk_metadata(self,
                             content: str,
                             chunk_index: int,
                             parent_document_id: Optional[str] = None,
                             parent_metadata: Optional[Dict[str, Any]] = None,
                             processing_info: Optional[Dict[str, Any]] = None,
                             **kwargs) -> ChunkMetadata:
        """
        Create comprehensive metadata for a text chunk.
        
        Args:
            content: Chunk content
            chunk_index: Index of the chunk
            parent_document_id: ID of parent document
            parent_metadata: Parent document metadata
            processing_info: Processing information
            **kwargs: Additional metadata fields
            
        Returns:
            ChunkMetadata object
        """
        try:
            # Initialize chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_index=chunk_index,
                parent_document_id=parent_document_id,
                content_length=len(content)
            )
            
            # Calculate content characteristics
            if self.config.analyze_content_characteristics:
                self._analyze_content_characteristics(content, chunk_metadata)
            
            # Calculate content hash
            if self.config.calculate_content_hashes:
                chunk_metadata.calculate_content_hash(content)
            
            # Inherit from parent metadata
            if parent_metadata and self.config.preserve_source_metadata:
                self._inherit_metadata(chunk_metadata, parent_metadata)
            
            # Add processing information
            if processing_info and self.config.track_processing_steps:
                self._add_processing_info(chunk_metadata, processing_info)
            
            # Apply custom fields
            for key, value in kwargs.items():
                chunk_metadata.set_custom_field(key, value)
            
            # Apply custom extractors
            for extractor_name, extractor_func in self.config.custom_extractors.items():
                try:
                    extracted_data = extractor_func({
                        "content": content,
                        "metadata": chunk_metadata.to_dict()
                    })
                    if extracted_data:
                        chunk_metadata.set_custom_field(extractor_name, extracted_data)
                except Exception as e:
                    self.logger.warning(f"Custom extractor '{extractor_name}' failed: {e}")
            
            # Validate if configured
            if self.config.validate_metadata:
                self._validate_chunk_metadata(chunk_metadata)
            
            # Store chunk metadata
            self._chunk_metadata[chunk_metadata.chunk_id] = chunk_metadata
            
            self.logger.debug(f"Created chunk metadata: {chunk_metadata.chunk_id}")
            return chunk_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create chunk metadata: {e}")
            raise MetadataError(f"Failed to create chunk metadata: {e}")
    
    def link_chunks(self, chunk_ids: List[str]) -> None:
        """
        Link consecutive chunks to track relationships.
        
        Args:
            chunk_ids: List of chunk IDs in sequence order
        """
        try:
            if not self.config.track_relationships or len(chunk_ids) < 2:
                return
            
            for i, chunk_id in enumerate(chunk_ids):
                if chunk_id not in self._chunk_metadata:
                    continue
                
                chunk_metadata = self._chunk_metadata[chunk_id]
                
                # Set previous chunk
                if i > 0:
                    chunk_metadata.previous_chunk_id = chunk_ids[i - 1]
                
                # Set next chunk
                if i < len(chunk_ids) - 1:
                    chunk_metadata.next_chunk_id = chunk_ids[i + 1]
                
                # Store relationships
                self._chunk_relationships[chunk_id] = []
                if chunk_metadata.previous_chunk_id:
                    self._chunk_relationships[chunk_id].append(chunk_metadata.previous_chunk_id)
                if chunk_metadata.next_chunk_id:
                    self._chunk_relationships[chunk_id].append(chunk_metadata.next_chunk_id)
            
            self.logger.debug(f"Linked {len(chunk_ids)} chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to link chunks: {e}")
            raise MetadataError(f"Failed to link chunks: {e}")
    
    def calculate_overlaps(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Calculate overlap information between consecutive chunks.
        
        Args:
            chunks: List of chunks with content and metadata
        """
        try:
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get('metadata', {}).get('chunk_id')
                if not chunk_id or chunk_id not in self._chunk_metadata:
                    continue
                
                chunk_metadata = self._chunk_metadata[chunk_id]
                content = chunk.get('content', '')
                
                # Calculate overlap with previous chunk
                if i > 0:
                    prev_content = chunks[i - 1].get('content', '')
                    overlap = self._calculate_text_overlap(prev_content, content)
                    chunk_metadata.overlap_with_previous = overlap
                
                # Calculate overlap with next chunk
                if i < len(chunks) - 1:
                    next_content = chunks[i + 1].get('content', '')
                    overlap = self._calculate_text_overlap(content, next_content)
                    chunk_metadata.overlap_with_next = overlap
            
            self.logger.debug(f"Calculated overlaps for {len(chunks)} chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overlaps: {e}")
    
    def enrich_metadata(self, chunk_id: str, enrichment_data: Dict[str, Any]) -> bool:
        """
        Enrich existing chunk metadata with additional data.
        
        Args:
            chunk_id: Chunk identifier
            enrichment_data: Additional metadata to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if chunk_id not in self._chunk_metadata:
                return False
            
            chunk_metadata = self._chunk_metadata[chunk_id]
            
            # Add enrichment data to custom fields
            for key, value in enrichment_data.items():
                if hasattr(chunk_metadata, key):
                    setattr(chunk_metadata, key, value)
                else:
                    chunk_metadata.set_custom_field(key, value)
            
            self.logger.debug(f"Enriched metadata for chunk: {chunk_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enrich metadata for {chunk_id}: {e}")
            return False
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get chunk metadata by ID."""
        return self._chunk_metadata.get(chunk_id)
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        return self._document_metadata.get(document_id)
    
    def get_related_chunks(self, chunk_id: str, depth: int = 1) -> List[str]:
        """
        Get related chunks up to specified depth.
        
        Args:
            chunk_id: Starting chunk ID
            depth: Relationship depth (1 = immediate neighbors)
            
        Returns:
            List of related chunk IDs
        """
        related = set()
        to_process = {chunk_id}
        
        for _ in range(depth):
            new_chunks = set()
            for cid in to_process:
                if cid in self._chunk_relationships:
                    new_chunks.update(self._chunk_relationships[cid])
            
            related.update(new_chunks)
            to_process = new_chunks - related
            
            if not to_process:
                break
        
        related.discard(chunk_id)  # Remove the starting chunk
        return list(related)
    
    def export_metadata(self, 
                       output_path: Union[str, Path],
                       include_chunks: bool = True,
                       include_documents: bool = True,
                       format_type: Optional[str] = None) -> None:
        """
        Export metadata to file.
        
        Args:
            output_path: Output file path
            include_chunks: Whether to include chunk metadata
            include_documents: Whether to include document metadata
            format_type: Export format ('json', 'yaml', 'pickle')
        """
        try:
            output_path = Path(output_path)
            format_type = format_type or self.config.metadata_format
            
            # Prepare export data
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }
            
            if include_documents:
                export_data["documents"] = self._document_metadata
            
            if include_chunks:
                export_data["chunks"] = {
                    chunk_id: metadata.to_dict()
                    for chunk_id, metadata in self._chunk_metadata.items()
                }
                export_data["relationships"] = self._chunk_relationships
            
            # Export based on format
            if format_type == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2 if self.config.pretty_print else None)
            
            elif format_type == "yaml":
                import yaml
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            
            elif format_type == "pickle":
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(export_data, f)
            
            else:
                raise InvalidConfigurationError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Exported metadata to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metadata: {e}")
            raise MetadataError(f"Failed to export metadata: {e}")
    
    def import_metadata(self, input_path: Union[str, Path], format_type: Optional[str] = None) -> None:
        """
        Import metadata from file.
        
        Args:
            input_path: Input file path
            format_type: Import format ('json', 'yaml', 'pickle')
        """
        try:
            input_path = Path(input_path)
            
            if not input_path.exists():
                raise MetadataError(f"Metadata file does not exist: {input_path}")
            
            format_type = format_type or input_path.suffix.lstrip('.')
            
            # Import based on format
            if format_type == "json":
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            elif format_type == "yaml":
                import yaml
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            
            elif format_type == "pickle":
                import pickle
                with open(input_path, 'rb') as f:
                    data = pickle.load(f)
            
            else:
                raise InvalidConfigurationError(f"Unsupported import format: {format_type}")
            
            # Load documents
            if "documents" in data:
                self._document_metadata.update(data["documents"])
            
            # Load chunks
            if "chunks" in data:
                for chunk_id, chunk_data in data["chunks"].items():
                    self._chunk_metadata[chunk_id] = ChunkMetadata.from_dict(chunk_data)
            
            # Load relationships
            if "relationships" in data:
                self._chunk_relationships.update(data["relationships"])
            
            self.logger.info(f"Imported metadata from: {input_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import metadata: {e}")
            raise MetadataError(f"Failed to import metadata: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata statistics."""
        return {
            "total_documents": len(self._document_metadata),
            "total_chunks": len(self._chunk_metadata),
            "total_relationships": len(self._chunk_relationships),
            "avg_chunks_per_document": len(self._chunk_metadata) / max(len(self._document_metadata), 1),
            "chunks_with_relationships": sum(1 for cid in self._chunk_metadata if cid in self._chunk_relationships),
            "config": {
                "preserve_source_metadata": self.config.preserve_source_metadata,
                "track_relationships": self.config.track_relationships,
                "analyze_content_characteristics": self.config.analyze_content_characteristics
            }
        }
    
    def _analyze_content_characteristics(self, content: str, metadata: ChunkMetadata) -> None:
        """Analyze content characteristics and update metadata."""
        # Basic counts
        metadata.word_count = len(content.split())
        metadata.sentence_count = len([s for s in content.split('.') if s.strip()])
        metadata.paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Language detection (if configured)
        if self.config.detect_language:
            try:
                import langdetect
                metadata.language = langdetect.detect(content)
            except Exception:
                pass  # Language detection is optional
    
    def _inherit_metadata(self, chunk_metadata: ChunkMetadata, parent_metadata: Dict[str, Any]) -> None:
        """Inherit metadata from parent document."""
        # Inherit source information
        chunk_metadata.source_file = parent_metadata.get('source_file')
        chunk_metadata.source_url = parent_metadata.get('source_url')
        chunk_metadata.source_type = parent_metadata.get('source_type')
        chunk_metadata.source_encoding = parent_metadata.get('source_encoding')
        
        # Inherit custom fields if configured
        if self.config.inherit_custom_fields:
            for key, value in parent_metadata.items():
                if key.startswith('custom_') or key in self.config.custom_inheritance_rules:
                    chunk_metadata.set_custom_field(key, value)
        
        # Inherit tags if configured
        if self.config.inherit_tags and 'tags' in parent_metadata:
            for tag in parent_metadata['tags']:
                chunk_metadata.add_tag(tag)
    
    def _add_processing_info(self, chunk_metadata: ChunkMetadata, processing_info: Dict[str, Any]) -> None:
        """Add processing information to chunk metadata."""
        chunk_metadata.chunking_strategy = processing_info.get('strategy')
        
        if self.config.include_chunking_parameters:
            chunk_metadata.chunking_parameters = processing_info.get('parameters', {})
        
        if self.config.track_processing_steps:
            steps = processing_info.get('preprocessing_steps', [])
            chunk_metadata.preprocessing_steps.extend(steps)
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> int:
        """Calculate character overlap between two texts."""
        if not text1 or not text2:
            return 0
        
        # Find the longest common suffix of text1 and prefix of text2
        max_overlap = min(len(text1), len(text2))
        
        for i in range(max_overlap, 0, -1):
            if text1[-i:] == text2[:i]:
                return i
        
        return 0
    
    def _validate_metadata(self, metadata: Dict[str, Any], scope: MetadataScope) -> None:
        """Validate metadata against configuration requirements."""
        if not self.config.required_fields:
            return
        
        missing_fields = []
        for field in self.config.required_fields:
            if field not in metadata or metadata[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise MetadataError(f"Missing required metadata fields: {missing_fields}")
    
    def _validate_chunk_metadata(self, metadata: ChunkMetadata) -> None:
        """Validate chunk metadata."""
        if not metadata.chunk_id:
            raise MetadataError("Chunk metadata must have a chunk_id")
        
        if metadata.content_length < 0:
            raise MetadataError("Content length cannot be negative")
    
    def clear_metadata(self, document_id: Optional[str] = None) -> None:
        """
        Clear metadata storage.
        
        Args:
            document_id: If provided, clear only this document's metadata
        """
        if document_id:
            # Clear specific document
            if document_id in self._document_metadata:
                del self._document_metadata[document_id]
            
            # Clear related chunks
            chunk_ids_to_remove = [
                cid for cid, metadata in self._chunk_metadata.items()
                if metadata.parent_document_id == document_id
            ]
            
            for chunk_id in chunk_ids_to_remove:
                if chunk_id in self._chunk_metadata:
                    del self._chunk_metadata[chunk_id]
                if chunk_id in self._chunk_relationships:
                    del self._chunk_relationships[chunk_id]
            
            self.logger.info(f"Cleared metadata for document: {document_id}")
        else:
            # Clear all metadata
            self._document_metadata.clear()
            self._chunk_metadata.clear()
            self._chunk_relationships.clear()
            self.logger.info("Cleared all metadata")