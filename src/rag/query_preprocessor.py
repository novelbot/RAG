"""
Query Preprocessing and Embedding System for RAG.

This module provides comprehensive query preprocessing capabilities including:
- Text normalization and cleaning
- Query validation and metadata extraction  
- Embedding generation using HuggingFace models
- Support for different query types and formats
- Caching for improved performance
"""

import re
import time
import json
import hashlib
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.core.logging import LoggerMixin
from src.core.exceptions import EmbeddingError, ConfigurationError
from src.embedding.manager import EmbeddingManager
from src.embedding.base import EmbeddingRequest
from src.text_processing.text_cleaner import TextCleaner, CleaningConfig, CleaningMode


class QueryType(Enum):
    """Types of queries that can be processed."""
    SIMPLE_TEXT = "simple_text"
    STRUCTURED = "structured"
    SEARCH = "search"
    QUESTION_ANSWER = "question_answer"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    MULTILINGUAL = "multilingual"


class QueryIntentType(Enum):
    """Intent classification for queries."""
    SEARCH = "search"
    QUESTION = "question"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    INSTRUCTION = "instruction"
    SUMMARIZATION = "summarization"
    UNKNOWN = "unknown"


@dataclass
class QueryMetadata:
    """Metadata extracted from query analysis."""
    query_type: QueryType
    intent: QueryIntentType
    language: Optional[str] = None
    complexity_score: float = 0.0
    temporal_info: Optional[Dict[str, Any]] = None
    location_info: Optional[Dict[str, Any]] = None
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0
    preprocessing_time: float = 0.0


@dataclass 
class QueryResult:
    """Result of query preprocessing and embedding generation."""
    original_query: str
    processed_query: str
    embedding: Optional[np.ndarray] = None
    metadata: QueryMetadata = field(default_factory=lambda: QueryMetadata(
        query_type=QueryType.SIMPLE_TEXT,
        intent=QueryIntentType.UNKNOWN
    ))
    processing_time: float = 0.0
    validation_passed: bool = True
    error_message: Optional[str] = None
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "original_query": self.original_query,
            "processed_query": self.processed_query,
            "embedding_shape": self.embedding.shape if self.embedding is not None else None,
            "metadata": {
                "query_type": self.metadata.query_type.value,
                "intent": self.metadata.intent.value,
                "language": self.metadata.language,
                "complexity_score": self.metadata.complexity_score,
                "entities": self.metadata.entities,
                "keywords": self.metadata.keywords,
                "confidence": self.metadata.confidence,
                "preprocessing_time": self.metadata.preprocessing_time
            },
            "processing_time": self.processing_time,
            "validation_passed": self.validation_passed,
            "error_message": self.error_message,
            "cache_hit": self.cache_hit
        }


@dataclass
class QueryPreprocessorConfig:
    """Configuration for query preprocessing."""
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    use_sentence_transformers: bool = True
    max_token_length: int = 512
    embedding_dimensions: Optional[int] = None
    normalize_embeddings: bool = True
    
    # Text preprocessing settings
    cleaning_mode: CleaningMode = CleaningMode.STANDARD
    normalize_case: bool = False
    remove_stop_words: bool = False
    expand_contractions: bool = True
    handle_emojis: bool = True
    preserve_entity_markers: bool = True
    
    # Query analysis settings
    enable_intent_detection: bool = True
    enable_language_detection: bool = True
    enable_entity_extraction: bool = True
    enable_keyword_extraction: bool = True
    min_query_length: int = 1
    max_query_length: int = 2048
    
    # Caching settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 10000
    
    # Performance settings
    batch_size: int = 32
    async_processing: bool = True


class QueryPreprocessor(LoggerMixin):
    """
    Comprehensive query preprocessing and embedding system for RAG.
    
    Features:
    - Multi-format query support (text, structured, JSON)
    - Intelligent text cleaning and normalization
    - HuggingFace/Sentence Transformers integration
    - Query type and intent detection
    - Language detection and entity extraction
    - Memory-based caching with TTL
    - Async processing support
    - Comprehensive validation and error handling
    """
    
    def __init__(
        self, 
        config: Optional[QueryPreprocessorConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        text_cleaner: Optional[TextCleaner] = None
    ):
        """
        Initialize the query preprocessor.
        
        Args:
            config: Preprocessing configuration
            embedding_manager: External embedding manager (optional)
            text_cleaner: External text cleaner (optional)
        """
        self.config = config or QueryPreprocessorConfig()
        
        # Initialize components
        self.embedding_manager = embedding_manager
        self.text_cleaner = text_cleaner or self._create_text_cleaner()
        
        # Initialize embedding model
        self.sentence_transformer = None
        if self.config.use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer(self.config.embedding_model)
                self.logger.info(f"Loaded SentenceTransformer model: {self.config.embedding_model}")
            except Exception as e:
                self.logger.warning(f"Failed to load SentenceTransformer: {e}")
                self.sentence_transformer = None
        
        # Cache for processed queries
        self.cache: Dict[str, Tuple[QueryResult, datetime]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Query patterns for intent detection
        self._initialize_patterns()
        
        self.logger.info("QueryPreprocessor initialized successfully")
    
    def process(self, query: Union[str, Dict[str, Any]], **kwargs) -> QueryResult:
        """
        Process a query through the complete preprocessing pipeline.
        
        Args:
            query: Query to process (string or structured dict)
            **kwargs: Override configuration parameters
            
        Returns:
            QueryResult with processed query and embeddings
        """
        start_time = time.time()
        
        try:
            # Convert to string if needed
            query_text = self._normalize_query_input(query)
            
            # Check cache first
            if self.config.enable_cache:
                cache_key = self._generate_cache_key(query_text, kwargs)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.cache_hits += 1
                    cached_result.cache_hit = True
                    return cached_result
                self.cache_misses += 1
            
            # Create result object
            result = QueryResult(
                original_query=query_text,
                processed_query=query_text
            )
            
            # Step 1: Validate query
            if not self._validate_query(query_text, result):
                return result
            
            # Step 2: Clean and normalize text
            cleaned_text = self._clean_and_normalize(query_text, result)
            result.processed_query = cleaned_text
            
            # Step 3: Extract metadata
            metadata_start = time.time()
            result.metadata = self._extract_metadata(cleaned_text)
            result.metadata.preprocessing_time = time.time() - metadata_start
            
            # Step 4: Generate embedding
            embedding = self._generate_embedding(cleaned_text)
            result.embedding = embedding
            
            # Final processing time
            result.processing_time = time.time() - start_time
            
            # Cache the result
            if self.config.enable_cache:
                self._save_to_cache(cache_key, result)
            
            self.logger.debug(
                f"Query processed successfully in {result.processing_time:.3f}s: "
                f"'{query_text[:50]}...' -> {result.metadata.query_type.value}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return QueryResult(
                original_query=str(query),
                processed_query="",
                processing_time=time.time() - start_time,
                validation_passed=False,
                error_message=str(e)
            )
    
    async def process_async(self, query: Union[str, Dict[str, Any]], **kwargs) -> QueryResult:
        """
        Process a query asynchronously.
        
        Args:
            query: Query to process
            **kwargs: Override configuration parameters
            
        Returns:
            QueryResult with processed query and embeddings
        """
        if not self.config.async_processing:
            return self.process(query, **kwargs)
        
        # Run in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, query, **kwargs)
    
    def process_batch(self, queries: List[Union[str, Dict[str, Any]]], **kwargs) -> List[QueryResult]:
        """
        Process multiple queries in a batch.
        
        Args:
            queries: List of queries to process
            **kwargs: Override configuration parameters
            
        Returns:
            List of QueryResult objects
        """
        results = []
        
        # Process in batches for memory efficiency
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = [self.process(query, **kwargs) for query in batch]
            results.extend(batch_results)
        
        self.logger.info(f"Processed batch of {len(queries)} queries")
        return results
    
    async def process_batch_async(
        self, 
        queries: List[Union[str, Dict[str, Any]]], 
        **kwargs
    ) -> List[QueryResult]:
        """
        Process multiple queries asynchronously.
        
        Args:
            queries: List of queries to process
            **kwargs: Override configuration parameters
            
        Returns:
            List of QueryResult objects
        """
        if not self.config.async_processing:
            return self.process_batch(queries, **kwargs)
        
        # Process all queries concurrently
        tasks = [self.process_async(query, **kwargs) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = QueryResult(
                    original_query=str(queries[i]),
                    processed_query="",
                    validation_passed=False,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        self.logger.info(f"Processed batch of {len(queries)} queries asynchronously")
        return processed_results
    
    def _normalize_query_input(self, query: Union[str, Dict[str, Any]]) -> str:
        """Normalize different query input formats to string."""
        if isinstance(query, str):
            return query
        elif isinstance(query, dict):
            # Handle structured queries
            if 'query' in query:
                return query['query']
            elif 'text' in query:
                return query['text']
            elif 'question' in query:
                return query['question']
            else:
                # Convert dict to JSON string
                return json.dumps(query, ensure_ascii=False)
        else:
            return str(query)
    
    def _validate_query(self, query: str, result: QueryResult) -> bool:
        """Validate query meets basic requirements."""
        if not query or not query.strip():
            result.validation_passed = False
            result.error_message = "Query is empty or contains only whitespace"
            return False
        
        query_length = len(query.strip())
        
        if query_length < self.config.min_query_length:
            result.validation_passed = False
            result.error_message = f"Query too short (min: {self.config.min_query_length})"
            return False
        
        if query_length > self.config.max_query_length:
            result.validation_passed = False
            result.error_message = f"Query too long (max: {self.config.max_query_length})"
            return False
        
        return True
    
    def _clean_and_normalize(self, query: str, result: QueryResult) -> str:
        """Clean and normalize query text."""
        try:
            # Use text cleaner for basic cleaning
            cleaning_result = self.text_cleaner.clean_text(
                query,
                mode=self.config.cleaning_mode,
                normalize_case="lower" if self.config.normalize_case else None,
                preserve_structure=True
            )
            
            cleaned = cleaning_result.cleaned_text
            
            # Additional query-specific processing
            if self.config.expand_contractions:
                cleaned = self._expand_contractions(cleaned)
            
            if self.config.handle_emojis:
                cleaned = self._handle_emojis(cleaned)
            
            # Remove excessive whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Text cleaning failed, using original: {e}")
            return query.strip()
    
    def _extract_metadata(self, query: str) -> QueryMetadata:
        """Extract comprehensive metadata from query."""
        metadata = QueryMetadata(
            query_type=QueryType.SIMPLE_TEXT,
            intent=QueryIntentType.UNKNOWN
        )
        
        # Detect query type
        metadata.query_type = self._detect_query_type(query)
        
        # Detect intent
        if self.config.enable_intent_detection:
            metadata.intent = self._detect_intent(query)
        
        # Detect language
        if self.config.enable_language_detection:
            metadata.language = self._detect_language(query)
        
        # Extract entities
        if self.config.enable_entity_extraction:
            metadata.entities = self._extract_entities(query)
        
        # Extract keywords
        if self.config.enable_keyword_extraction:
            metadata.keywords = self._extract_keywords(query)
        
        # Calculate complexity score
        metadata.complexity_score = self._calculate_complexity(query)
        
        # Extract temporal information
        metadata.temporal_info = self._extract_temporal_info(query)
        
        # Extract location information
        metadata.location_info = self._extract_location_info(query)
        
        return metadata
    
    def _generate_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for the processed query."""
        try:
            # Try SentenceTransformers first
            if self.sentence_transformer:
                embedding = self.sentence_transformer.encode(
                    query,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True
                )
                return embedding
            
            # Fallback to embedding manager
            elif self.embedding_manager:
                request = EmbeddingRequest(
                    input=[query],
                    model=self.config.embedding_model,
                    dimensions=self.config.embedding_dimensions,
                    normalize=self.config.normalize_embeddings,
                    metadata={"is_query": True}
                )
                
                response = self.embedding_manager.generate_embeddings(request)
                if response.embeddings:
                    return np.array(response.embeddings[0])
            
            else:
                self.logger.warning("No embedding provider available")
                return None
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query."""
        query_lower = query.lower()
        
        # Check for structured markers
        if any(marker in query for marker in ['{', '}', ':', '"']):
            return QueryType.STRUCTURED
        
        # Check for question patterns
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(query_lower.startswith(word) for word in question_words) or query.endswith('?'):
            return QueryType.QUESTION_ANSWER
        
        # Check for search patterns
        search_patterns = ['find', 'search', 'look for', 'get', 'retrieve']
        if any(pattern in query_lower for pattern in search_patterns):
            return QueryType.SEARCH
        
        # Check for multilingual content
        if self._has_non_latin_chars(query):
            return QueryType.MULTILINGUAL
        
        # Default to simple text
        return QueryType.SIMPLE_TEXT
    
    def _detect_intent(self, query: str) -> QueryIntentType:
        """Detect query intent using pattern matching."""
        query_lower = query.lower()
        
        # Question intent
        if any(query_lower.startswith(word) for word in self.question_patterns) or query.endswith('?'):
            return QueryIntentType.QUESTION
        
        # Comparison intent
        if any(word in query_lower for word in self.comparison_patterns):
            return QueryIntentType.COMPARISON
        
        # Definition intent
        if any(pattern in query_lower for pattern in self.definition_patterns):
            return QueryIntentType.DEFINITION
        
        # Explanation intent
        if any(pattern in query_lower for pattern in self.explanation_patterns):
            return QueryIntentType.EXPLANATION
        
        # Instruction intent
        if any(query_lower.startswith(word) for word in self.instruction_patterns):
            return QueryIntentType.INSTRUCTION
        
        # Summarization intent
        if any(pattern in query_lower for pattern in self.summarization_patterns):
            return QueryIntentType.SUMMARIZATION
        
        # Default to search
        return QueryIntentType.SEARCH
    
    def _detect_language(self, query: str) -> Optional[str]:
        """Simple language detection based on character patterns."""
        # Basic language detection - could be enhanced with langdetect library
        if re.search(r'[가-힣]', query):
            return 'ko'
        elif re.search(r'[一-龯]', query):
            return 'zh'
        elif re.search(r'[あ-ん]|[ア-ン]|[一-龯]', query):
            return 'ja'
        elif re.search(r'[а-я]', query):
            return 'ru'
        elif re.search(r'[α-ω]', query):
            return 'el'
        elif re.search(r'[א-ת]', query):
            return 'he'
        elif re.search(r'[ا-ي]', query):
            return 'ar'
        else:
            return 'en'  # Default to English
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        # Simple regex-based entity extraction
        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        entities.extend(re.findall(url_pattern, query))
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities.extend(re.findall(email_pattern, query))
        
        # Dates (simple patterns)
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b'   # MM-DD-YYYY
        ]
        for pattern in date_patterns:
            entities.extend(re.findall(pattern, query))
        
        # Numbers
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        entities.extend(re.findall(number_pattern, query))
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = query.lower().split()
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'how', 'when', 'where', 'why',
            'who', 'which'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        score = 0.0
        
        # Length factor
        length_score = min(len(query) / 100, 1.0) * 0.3
        score += length_score
        
        # Word count factor
        word_count = len(query.split())
        word_score = min(word_count / 20, 1.0) * 0.2
        score += word_score
        
        # Special characters factor
        special_chars = len(re.findall(r'[^\w\s]', query))
        special_score = min(special_chars / 10, 1.0) * 0.2
        score += special_score
        
        # Question complexity
        if '?' in query:
            score += 0.1
        
        # Multiple sentences
        sentence_count = len(re.split(r'[.!?]+', query))
        if sentence_count > 1:
            score += min(sentence_count / 5, 1.0) * 0.2
        
        return min(score, 1.0)
    
    def _extract_temporal_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal information from query."""
        temporal_patterns = {
            'today': r'\btoday\b',
            'yesterday': r'\byesterday\b',
            'tomorrow': r'\btomorrow\b',
            'last_week': r'\blast\s+week\b',
            'next_week': r'\bnext\s+week\b',
            'last_month': r'\blast\s+month\b',
            'next_month': r'\bnext\s+month\b',
            'last_year': r'\blast\s+year\b',
            'next_year': r'\bnext\s+year\b',
            'recent': r'\brecent\b|\brecently\b',
            'current': r'\bcurrent\b|\bnow\b'
        }
        
        found_temporal = {}
        query_lower = query.lower()
        
        for temporal_type, pattern in temporal_patterns.items():
            if re.search(pattern, query_lower):
                found_temporal[temporal_type] = True
        
        return found_temporal if found_temporal else None
    
    def _extract_location_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract location information from query."""
        location_patterns = {
            'near_me': r'\bnear\s+me\b|\bnearby\b',
            'city': r'\bin\s+[A-Z][a-z]+\b',
            'country': r'\bin\s+[A-Z][a-z]+\b',
            'address': r'\b\d+\s+[A-Za-z\s]+\b'
        }
        
        found_locations = {}
        query_lower = query.lower()
        
        for location_type, pattern in location_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                found_locations[location_type] = matches
        
        return found_locations if found_locations else None
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _handle_emojis(self, text: str) -> str:
        """Handle emojis in text."""
        # Simple emoji removal - could be enhanced to convert to text descriptions
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    def _has_non_latin_chars(self, text: str) -> bool:
        """Check if text contains non-Latin characters."""
        return bool(re.search(r'[^\x00-\x7F]', text))
    
    def _create_text_cleaner(self) -> TextCleaner:
        """Create default text cleaner configuration."""
        cleaning_config = CleaningConfig(
            mode=self.config.cleaning_mode,
            preserve_structure=True,
            normalize_whitespace=True,
            remove_extra_newlines=True,
            convert_html_entities=True
        )
        return TextCleaner(cleaning_config)
    
    def _initialize_patterns(self) -> None:
        """Initialize patterns for intent detection."""
        self.question_patterns = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',
            'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
            'is', 'are', 'was', 'were'
        ]
        
        self.comparison_patterns = [
            'vs', 'versus', 'compare', 'difference', 'better', 'worse',
            'more', 'less', 'than', 'between'
        ]
        
        self.definition_patterns = [
            'what is', 'what are', 'define', 'definition', 'meaning',
            'explain', 'describe'
        ]
        
        self.explanation_patterns = [
            'how does', 'why does', 'explain', 'describe', 'detail',
            'elaborate', 'clarify'
        ]
        
        self.instruction_patterns = [
            'how to', 'tell me', 'show me', 'help me', 'teach me',
            'guide me', 'instruct', 'demonstrate'
        ]
        
        self.summarization_patterns = [
            'summarize', 'summary', 'overview', 'brief', 'outline',
            'key points', 'main points'
        ]
    
    def _generate_cache_key(self, query: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for query and parameters."""
        key_data = {
            'query': query,
            'config': {
                'embedding_model': self.config.embedding_model,
                'normalize_embeddings': self.config.normalize_embeddings,
                'cleaning_mode': self.config.cleaning_mode.value
            },
            'kwargs': kwargs
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[QueryResult]:
        """Get result from cache if not expired."""
        if cache_key not in self.cache:
            return None
        
        result, timestamp = self.cache[cache_key]
        
        # Check if expired
        if (datetime.now(timezone.utc) - timestamp).total_seconds() > self.config.cache_ttl:
            del self.cache[cache_key]
            return None
        
        return result
    
    def _save_to_cache(self, cache_key: str, result: QueryResult) -> None:
        """Save result to cache with cleanup if needed."""
        # Clean old entries if cache is full
        if len(self.cache) >= self.config.max_cache_size:
            # Remove oldest entries (10% of cache)
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )[:self.config.max_cache_size // 10]
            
            for key in oldest_keys:
                del self.cache[key]
        
        self.cache[cache_key] = (result, datetime.now(timezone.utc))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return {
            "cache_enabled": self.config.enable_cache,
            "cache_entries": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            "sentence_transformer_available": self.sentence_transformer is not None,
            "embedding_manager_available": self.embedding_manager is not None,
            "model": self.config.embedding_model
        }
    
    def clear_cache(self) -> None:
        """Clear the query processing cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Query preprocessor cache cleared")