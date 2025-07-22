"""
Text Splitting System with LangChain Integration.

This module provides intelligent text chunking capabilities using LangChain
text splitters, supporting multiple chunking strategies with configurable
parameters and metadata preservation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
import time

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter,
        NLTKTextSplitter,
        SpacyTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter,
        PythonCodeTextSplitter,
        MarkdownTextSplitter,
        LatexTextSplitter,
        SentenceTransformersTokenTextSplitter
    )
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Fallback imports for type hints
    Document = object
    RecursiveCharacterTextSplitter = object

from src.core.logging import LoggerMixin
from .exceptions import ChunkingError, InvalidConfigurationError
from .text_cleaner import TextCleaner, CleaningConfig


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""
    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    TOKEN = "token"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    LATEX = "latex"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    NLTK = "nltk"
    SPACY = "spacy"


class OverlapStrategy(Enum):
    """Strategies for handling chunk overlap."""
    FIXED = "fixed"  # Fixed number of characters/tokens
    PERCENTAGE = "percentage"  # Percentage of chunk size
    SENTENCE = "sentence"  # Sentence-based overlap
    NONE = "none"  # No overlap


@dataclass
class ChunkingConfig:
    """Configuration for text chunking operations."""
    
    # Basic chunking parameters
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER
    chunk_size: int = 1000
    chunk_overlap: int = 200
    overlap_strategy: OverlapStrategy = OverlapStrategy.FIXED
    
    # Advanced parameters
    length_function: Optional[str] = None  # 'len', 'tiktoken', 'transformers'
    model_name: Optional[str] = None  # For token-based splitting
    encoding_name: Optional[str] = "cl100k_base"  # For tiktoken
    separators: Optional[List[str]] = None  # Custom separators
    keep_separator: bool = False
    add_start_index: bool = True
    strip_whitespace: bool = True
    
    # Document type specific
    language: Optional[str] = None  # For code splitting
    headers_to_split_on: Optional[List[tuple]] = None  # For HTML/Markdown
    
    # Quality control
    min_chunk_size: int = 50
    max_chunk_size: Optional[int] = None
    discard_short_chunks: bool = True
    merge_small_chunks: bool = True
    
    # Processing options
    preserve_metadata: bool = True
    add_chunk_index: bool = True
    add_source_info: bool = True
    batch_size: int = 1000
    
    # Text cleaning integration
    clean_text_before_chunking: bool = True
    cleaning_config: Optional[CleaningConfig] = None


@dataclass
class ChunkResult:
    """Result of a text chunking operation."""
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks)
    
    def get_total_length(self) -> int:
        """Get total length of all chunks."""
        return sum(len(chunk.get('content', '')) for chunk in self.chunks)
    
    def get_average_chunk_size(self) -> float:
        """Get average chunk size."""
        if not self.chunks:
            return 0.0
        return self.get_total_length() / len(self.chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "chunk_count": self.get_chunk_count(),
            "total_length": self.get_total_length(),
            "average_chunk_size": self.get_average_chunk_size(),
            "chunks": self.chunks,
            "metadata": self.metadata,
            "statistics": self.statistics,
            "errors": self.errors,
            "warnings": self.warnings
        }


class BaseChunker(ABC, LoggerMixin):
    """Abstract base class for text chunkers."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the chunker with configuration."""
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to preserve
            
        Returns:
            ChunkResult with chunks and metadata
        """
        pass
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> ChunkResult:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            
        Returns:
            ChunkResult with all chunks
        """
        all_chunks = []
        all_errors = []
        all_warnings = []
        processing_stats = []
        
        for i, doc in enumerate(documents):
            try:
                content = doc.get('content', '')
                doc_metadata = doc.get('metadata', {})
                doc_metadata['source_document_index'] = i
                
                result = self.chunk_text(content, doc_metadata)
                all_chunks.extend(result.chunks)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                processing_stats.append(result.statistics)
                
            except Exception as e:
                error_msg = f"Failed to chunk document {i}: {e}"
                all_errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Combine statistics
        combined_stats = self._combine_statistics(processing_stats)
        
        return ChunkResult(
            chunks=all_chunks,
            metadata={"total_documents": len(documents)},
            statistics=combined_stats,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.chunk_size <= 0:
            raise InvalidConfigurationError("chunk_size must be positive")
        
        if self.config.chunk_overlap < 0:
            raise InvalidConfigurationError("chunk_overlap cannot be negative")
        
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise InvalidConfigurationError("chunk_overlap must be smaller than chunk_size")
        
        if self.config.min_chunk_size <= 0:
            raise InvalidConfigurationError("min_chunk_size must be positive")
    
    def _combine_statistics(self, stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine statistics from multiple operations."""
        if not stats_list:
            return {}
        
        combined = {
            "total_processing_time": sum(s.get("processing_time", 0) for s in stats_list),
            "total_input_length": sum(s.get("input_length", 0) for s in stats_list),
            "total_output_chunks": sum(s.get("chunk_count", 0) for s in stats_list),
            "average_chunk_size": sum(s.get("average_chunk_size", 0) for s in stats_list) / len(stats_list),
            "documents_processed": len(stats_list)
        }
        
        return combined


class LangChainChunker(BaseChunker):
    """LangChain-based text chunker with multiple strategies."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize LangChain chunker."""
        super().__init__(config)
        
        if not LANGCHAIN_AVAILABLE:
            raise ChunkingError("LangChain text splitters not available. Install with: pip install langchain-text-splitters")
        
        self.text_cleaner = None
        if config.clean_text_before_chunking:
            cleaning_config = config.cleaning_config or CleaningConfig()
            self.text_cleaner = TextCleaner(cleaning_config)
        
        self._splitter = self._create_splitter()
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Chunk text using LangChain splitters."""
        start_time = time.time()
        
        try:
            # Initialize result
            result = ChunkResult(
                chunks=[],
                metadata=metadata or {},
                statistics={"input_length": len(text)}
            )
            
            # Validate input
            if not text or not text.strip():
                result.warnings.append("Input text is empty or contains only whitespace")
                return result
            
            # Clean text if configured
            if self.text_cleaner:
                cleaning_result = self.text_cleaner.clean_text(text)
                text = cleaning_result.cleaned_text
                result.metadata["text_cleaning"] = cleaning_result.to_dict()
                
                if cleaning_result.warnings:
                    result.warnings.extend(cleaning_result.warnings)
            
            # Create LangChain documents for processing
            documents = [Document(page_content=text, metadata=metadata or {})]
            
            # Split documents
            split_docs = self._splitter.split_documents(documents)
            
            # Convert to our format
            chunks = []
            for i, doc in enumerate(split_docs):
                chunk_metadata = doc.metadata.copy() if self.config.preserve_metadata else {}
                
                if self.config.add_chunk_index:
                    chunk_metadata['chunk_index'] = i
                
                if self.config.add_source_info:
                    chunk_metadata['chunk_source'] = 'langchain_splitter'
                    chunk_metadata['chunking_strategy'] = self.config.strategy.value
                
                chunk = {
                    'content': doc.page_content,
                    'metadata': chunk_metadata,
                    'length': len(doc.page_content)
                }
                
                chunks.append(chunk)
            
            # Apply quality control
            chunks = self._apply_quality_control(chunks, result)
            
            # Update result
            result.chunks = chunks
            result.statistics.update({
                "chunk_count": len(chunks),
                "average_chunk_size": sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
                "processing_time": time.time() - start_time,
                "splitter_type": type(self._splitter).__name__
            })
            
            self.logger.debug(f"Text chunked: {len(text)} chars -> {len(chunks)} chunks")
            return result
            
        except Exception as e:
            self.logger.error(f"Text chunking failed: {e}")
            raise ChunkingError(f"Text chunking failed: {e}")
    
    def _create_splitter(self):
        """Create the appropriate LangChain text splitter."""
        strategy = self.config.strategy
        
        try:
            if strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
                return self._create_recursive_character_splitter()
            elif strategy == ChunkingStrategy.CHARACTER:
                return self._create_character_splitter()
            elif strategy == ChunkingStrategy.TOKEN:
                return self._create_token_splitter()
            elif strategy == ChunkingStrategy.SENTENCE:
                return self._create_sentence_splitter()
            elif strategy == ChunkingStrategy.MARKDOWN:
                return self._create_markdown_splitter()
            elif strategy == ChunkingStrategy.HTML:
                return self._create_html_splitter()
            elif strategy == ChunkingStrategy.CODE:
                return self._create_code_splitter()
            elif strategy == ChunkingStrategy.LATEX:
                return self._create_latex_splitter()
            elif strategy == ChunkingStrategy.NLTK:
                return self._create_nltk_splitter()
            elif strategy == ChunkingStrategy.SPACY:
                return self._create_spacy_splitter()
            else:
                raise InvalidConfigurationError(f"Unsupported chunking strategy: {strategy}")
                
        except Exception as e:
            raise ChunkingError(f"Failed to create text splitter: {e}")
    
    def _create_recursive_character_splitter(self):
        """Create RecursiveCharacterTextSplitter."""
        kwargs = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "add_start_index": self.config.add_start_index,
            "strip_whitespace": self.config.strip_whitespace,
            "keep_separator": self.config.keep_separator
        }
        
        if self.config.separators:
            kwargs["separators"] = self.config.separators
        
        if self.config.length_function == "tiktoken" and self.config.model_name:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name=self.config.model_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.length_function == "tiktoken" and self.config.encoding_name:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.config.encoding_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        return RecursiveCharacterTextSplitter(**kwargs)
    
    def _create_character_splitter(self):
        """Create CharacterTextSplitter."""
        separator = self.config.separators[0] if self.config.separators else "\n\n"
        
        kwargs = {
            "separator": separator,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "add_start_index": self.config.add_start_index,
            "strip_whitespace": self.config.strip_whitespace,
            "keep_separator": self.config.keep_separator
        }
        
        if self.config.length_function == "tiktoken" and self.config.encoding_name:
            return CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.config.encoding_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        return CharacterTextSplitter(**kwargs)
    
    def _create_token_splitter(self):
        """Create TokenTextSplitter."""
        return TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _create_sentence_splitter(self):
        """Create sentence-based splitter using SentenceTransformers."""
        if self.config.model_name:
            return SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.config.chunk_overlap,
                model_name=self.config.model_name
            )
        else:
            # Fallback to NLTK sentence splitting
            return self._create_nltk_splitter()
    
    def _create_markdown_splitter(self):
        """Create MarkdownTextSplitter."""
        if self.config.headers_to_split_on:
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.config.headers_to_split_on
            )
            return header_splitter
        else:
            return MarkdownTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
    
    def _create_html_splitter(self):
        """Create HTMLHeaderTextSplitter."""
        if self.config.headers_to_split_on:
            return HTMLHeaderTextSplitter(
                headers_to_split_on=self.config.headers_to_split_on
            )
        else:
            # Fallback to recursive character splitter
            return self._create_recursive_character_splitter()
    
    def _create_code_splitter(self):
        """Create PythonCodeTextSplitter."""
        # Note: LangChain has language-specific splitters
        return PythonCodeTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _create_latex_splitter(self):
        """Create LatexTextSplitter."""
        return LatexTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _create_nltk_splitter(self):
        """Create NLTKTextSplitter."""
        return NLTKTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _create_spacy_splitter(self):
        """Create SpacyTextSplitter."""
        return SpacyTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _apply_quality_control(self, chunks: List[Dict[str, Any]], result: ChunkResult) -> List[Dict[str, Any]]:
        """Apply quality control measures to chunks."""
        if not chunks:
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            content_length = len(content.strip())
            
            # Check minimum size
            if content_length < self.config.min_chunk_size:
                if self.config.discard_short_chunks:
                    result.warnings.append(f"Discarded chunk with length {content_length} (below minimum {self.config.min_chunk_size})")
                    continue
            
            # Check maximum size
            if self.config.max_chunk_size and content_length > self.config.max_chunk_size:
                result.warnings.append(f"Chunk exceeds maximum size: {content_length} > {self.config.max_chunk_size}")
                # Could implement splitting here if needed
            
            filtered_chunks.append(chunk)
        
        # Merge small chunks if configured
        if self.config.merge_small_chunks:
            filtered_chunks = self._merge_small_chunks(filtered_chunks, result)
        
        return filtered_chunks
    
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]], result: ChunkResult) -> List[Dict[str, Any]]:
        """Merge consecutive small chunks."""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            content_length = len(chunk['content'].strip())
            
            if content_length < self.config.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk.copy()
                else:
                    # Merge with current chunk
                    current_chunk['content'] += '\n' + chunk['content']
                    current_chunk['length'] = len(current_chunk['content'])
                    
                    # Merge metadata
                    if 'chunk_index' in chunk['metadata']:
                        current_chunk['metadata']['merged_chunk_indices'] = current_chunk['metadata'].get('merged_chunk_indices', [])
                        current_chunk['metadata']['merged_chunk_indices'].append(chunk['metadata']['chunk_index'])
            else:
                # Add any pending merged chunk
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                
                # Add current chunk
                merged_chunks.append(chunk)
        
        # Add final merged chunk if exists
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
        
        if len(merged_chunks) != len(chunks):
            result.warnings.append(f"Merged {len(chunks) - len(merged_chunks)} small chunks")
        
        return merged_chunks


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size text chunker without LangChain dependency."""
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Chunk text into fixed-size pieces."""
        start_time = time.time()
        
        try:
            result = ChunkResult(
                chunks=[],
                metadata=metadata or {},
                statistics={"input_length": len(text)}
            )
            
            if not text or not text.strip():
                result.warnings.append("Input text is empty or contains only whitespace")
                return result
            
            chunk_size = self.config.chunk_size
            overlap = self.config.chunk_overlap
            chunks = []
            
            start = 0
            chunk_index = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk_content = text[start:end]
                
                if len(chunk_content.strip()) >= self.config.min_chunk_size:
                    chunk_metadata = (metadata or {}).copy() if self.config.preserve_metadata else {}
                    
                    if self.config.add_chunk_index:
                        chunk_metadata['chunk_index'] = chunk_index
                    
                    if self.config.add_source_info:
                        chunk_metadata['chunk_source'] = 'fixed_size_chunker'
                        chunk_metadata['start_index'] = start
                        chunk_metadata['end_index'] = end
                    
                    chunk = {
                        'content': chunk_content,
                        'metadata': chunk_metadata,
                        'length': len(chunk_content)
                    }
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                start = end - overlap
                
                # Prevent infinite loop
                if start >= end:
                    break
            
            result.chunks = chunks
            result.statistics.update({
                "chunk_count": len(chunks),
                "average_chunk_size": sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
                "processing_time": time.time() - start_time,
                "splitter_type": "FixedSizeChunker"
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fixed-size chunking failed: {e}")
            raise ChunkingError(f"Fixed-size chunking failed: {e}")


class SemanticChunker(BaseChunker):
    """Semantic chunker (placeholder for future implementation)."""
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Chunk text semantically (currently falls back to recursive character splitting)."""
        # For now, fallback to LangChain implementation
        fallback_config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        if LANGCHAIN_AVAILABLE:
            chunker = LangChainChunker(fallback_config)
        else:
            chunker = FixedSizeChunker(fallback_config)
        
        return chunker.chunk_text(text, metadata)


class SentenceBasedChunker(BaseChunker):
    """Sentence-based chunker using simple sentence detection."""
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Chunk text by sentences."""
        start_time = time.time()
        
        try:
            result = ChunkResult(
                chunks=[],
                metadata=metadata or {},
                statistics={"input_length": len(text)}
            )
            
            if not text or not text.strip():
                result.warnings.append("Input text is empty or contains only whitespace")
                return result
            
            # Simple sentence splitting
            import re
            sentences = re.split(r'[.!?]+\s+', text)
            
            chunks = []
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed chunk size
                if current_chunk and len(current_chunk + " " + sentence) > self.config.chunk_size:
                    # Save current chunk
                    if len(current_chunk.strip()) >= self.config.min_chunk_size:
                        chunk_metadata = (metadata or {}).copy() if self.config.preserve_metadata else {}
                        
                        if self.config.add_chunk_index:
                            chunk_metadata['chunk_index'] = chunk_index
                        
                        if self.config.add_source_info:
                            chunk_metadata['chunk_source'] = 'sentence_based_chunker'
                        
                        chunk = {
                            'content': current_chunk.strip(),
                            'metadata': chunk_metadata,
                            'length': len(current_chunk.strip())
                        }
                        
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # Start new chunk with overlap
                    if self.config.chunk_overlap > 0:
                        # Keep last few sentences for overlap
                        overlap_sentences = current_chunk.split('. ')[-2:]  # Keep last 2 sentences
                        current_chunk = '. '.join(overlap_sentences) + ". " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Add final chunk
            if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
                chunk_metadata = (metadata or {}).copy() if self.config.preserve_metadata else {}
                
                if self.config.add_chunk_index:
                    chunk_metadata['chunk_index'] = chunk_index
                
                if self.config.add_source_info:
                    chunk_metadata['chunk_source'] = 'sentence_based_chunker'
                
                chunk = {
                    'content': current_chunk.strip(),
                    'metadata': chunk_metadata,
                    'length': len(current_chunk.strip())
                }
                
                chunks.append(chunk)
            
            result.chunks = chunks
            result.statistics.update({
                "chunk_count": len(chunks),
                "average_chunk_size": sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
                "processing_time": time.time() - start_time,
                "splitter_type": "SentenceBasedChunker"
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sentence-based chunking failed: {e}")
            raise ChunkingError(f"Sentence-based chunking failed: {e}")


class TextSplitter(LoggerMixin):
    """
    Main text splitter interface with multiple chunking strategies.
    
    Provides a unified interface for different chunking approaches
    including LangChain integration and custom implementations.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize text splitter.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self.chunker = self._create_chunker()
        
        self.logger.info(f"Text splitter initialized with strategy: {self.config.strategy.value}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """
        Chunk text using configured strategy.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to preserve
            
        Returns:
            ChunkResult with chunks and metadata
        """
        return self.chunker.chunk_text(text, metadata)
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> ChunkResult:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            
        Returns:
            ChunkResult with all chunks
        """
        return self.chunker.chunk_documents(documents)
    
    def chunk_file(self, file_path: Union[str, Path], **kwargs) -> ChunkResult:
        """
        Chunk text from a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional metadata
            
        Returns:
            ChunkResult with chunks and metadata
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise ChunkingError(f"File does not exist: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Prepare metadata
            metadata = {
                "source_file": str(file_path),
                "file_size": file_path.stat().st_size,
                **kwargs
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            self.logger.error(f"File chunking failed for {file_path}: {e}")
            raise ChunkingError(f"File chunking failed for {file_path}: {e}")
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """Get information about the current chunker."""
        return {
            "strategy": self.config.strategy.value,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "chunker_type": type(self.chunker).__name__,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "config": {
                "length_function": self.config.length_function,
                "model_name": self.config.model_name,
                "preserve_metadata": self.config.preserve_metadata,
                "clean_text_before_chunking": self.config.clean_text_before_chunking
            }
        }
    
    def _create_chunker(self) -> BaseChunker:
        """Create the appropriate chunker based on strategy."""
        strategy = self.config.strategy
        
        # Strategies that require LangChain
        langchain_strategies = {
            ChunkingStrategy.RECURSIVE_CHARACTER,
            ChunkingStrategy.CHARACTER,
            ChunkingStrategy.TOKEN,
            ChunkingStrategy.MARKDOWN,
            ChunkingStrategy.HTML,
            ChunkingStrategy.CODE,
            ChunkingStrategy.LATEX,
            ChunkingStrategy.NLTK,
            ChunkingStrategy.SPACY
        }
        
        if strategy in langchain_strategies:
            if LANGCHAIN_AVAILABLE:
                return LangChainChunker(self.config)
            else:
                self.logger.warning(f"LangChain not available, falling back to fixed-size chunking for strategy: {strategy}")
                return FixedSizeChunker(self.config)
        
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(self.config)
        
        elif strategy == ChunkingStrategy.SENTENCE:
            if LANGCHAIN_AVAILABLE:
                return LangChainChunker(self.config)
            else:
                return SentenceBasedChunker(self.config)
        
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(self.config)
        
        else:
            raise InvalidConfigurationError(f"Unsupported chunking strategy: {strategy}")