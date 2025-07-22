"""
Query Expansion and Optimization for RAG System.

This module provides intelligent query expansion and optimization features to improve
search quality through synonym expansion, related term discovery, phrase detection,
term weighting, and personalized expansion based on query history.
"""

import time
import re
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import hashlib

import numpy as np
try:
    import spacy
    from spacy.tokens import Doc, Token, Span
except ImportError:
    spacy = None

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.corpus import stopwords
except ImportError:
    nltk = None
    wordnet = None
    stopwords = None

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.logging import LoggerMixin
from src.core.exceptions import ConfigurationError, ProcessingError
from src.rag.query_preprocessor import QueryResult, QueryMetadata


class ExpansionStrategy(Enum):
    """Query expansion strategy types."""
    CONSERVATIVE = "conservative"    # Minimal expansion, high precision
    BALANCED = "balanced"           # Moderate expansion with balance
    AGGRESSIVE = "aggressive"       # Maximum expansion for recall
    ADAPTIVE = "adaptive"           # Strategy based on query analysis
    CUSTOM = "custom"              # User-defined strategy


class WeightingMethod(Enum):
    """Term weighting methods."""
    UNIFORM = "uniform"             # All terms weighted equally
    TF_IDF = "tf_idf"              # TF-IDF based weighting
    SEMANTIC = "semantic"           # Semantic similarity based
    HYBRID = "hybrid"              # Combination of methods
    CUSTOM = "custom"              # Custom weighting function


class LanguageModel(Enum):
    """Supported language models for multi-language processing."""
    ENGLISH = "en_core_web_sm"
    ENGLISH_LARGE = "en_core_web_lg"
    SPANISH = "es_core_news_sm"
    FRENCH = "fr_core_news_sm"
    GERMAN = "de_core_news_sm"
    CHINESE = "zh_core_web_sm"
    MULTILINGUAL = "xx_ent_wiki_sm"


@dataclass
class ExpansionConfig:
    """Configuration for query expansion and optimization."""
    
    # Expansion strategy
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.BALANCED
    max_expanded_terms: int = 10
    min_similarity_threshold: float = 0.6
    max_query_length: int = 500
    
    # Term weighting
    weighting_method: WeightingMethod = WeightingMethod.HYBRID
    base_term_weight: float = 1.0
    synonym_weight: float = 0.8
    semantic_weight: float = 0.7
    phrase_weight: float = 1.2
    
    # Language processing
    language_model: LanguageModel = LanguageModel.ENGLISH
    enable_multi_language: bool = True
    auto_detect_language: bool = True
    
    # Expansion types
    enable_synonym_expansion: bool = True
    enable_semantic_expansion: bool = True
    enable_phrase_detection: bool = True
    enable_entity_expansion: bool = True
    enable_context_expansion: bool = True
    
    # Optimization settings
    enable_query_reordering: bool = True
    enable_stop_word_handling: bool = True
    enable_term_deduplication: bool = True
    preserve_original_terms: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 10000
    batch_size: int = 50
    
    # History and personalization
    enable_personalization: bool = True
    history_window_days: int = 30
    min_history_queries: int = 5
    
    # Advanced options
    phrase_min_length: int = 2
    phrase_max_length: int = 5
    semantic_model_name: str = "all-MiniLM-L6-v2"
    wordnet_max_synonyms: int = 5


@dataclass
class TermWeight:
    """Term weight with metadata."""
    term: str
    weight: float
    term_type: str = "original"  # original, synonym, semantic, phrase, entity
    source: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "weight": self.weight,
            "term_type": self.term_type,
            "source": self.source,
            "confidence": self.confidence
        }


@dataclass
class ExpansionResult:
    """Result of query expansion and optimization."""
    original_query: str
    expanded_query: str
    expanded_terms: List[str]
    term_weights: List[TermWeight]
    phrase_boundaries: List[Tuple[int, int]]
    detected_phrases: List[str]
    expansion_metadata: Dict[str, Any] = field(default_factory=dict)
    expansion_time: float = 0.0
    
    def get_weighted_terms_dict(self) -> Dict[str, float]:
        """Get dictionary of terms to weights."""
        return {tw.term: tw.weight for tw in self.term_weights}
    
    def get_terms_by_type(self, term_type: str) -> List[TermWeight]:
        """Get terms filtered by type."""
        return [tw for tw in self.term_weights if tw.term_type == term_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "expanded_terms": self.expanded_terms,
            "term_weights": [tw.to_dict() for tw in self.term_weights],
            "phrase_boundaries": self.phrase_boundaries,
            "detected_phrases": self.detected_phrases,
            "expansion_metadata": self.expansion_metadata,
            "expansion_time": self.expansion_time
        }


@dataclass
class QueryHistoryEntry:
    """Query history entry for personalization."""
    query: str
    expanded_query: str
    user_id: int
    timestamp: datetime
    success_score: float = 0.0  # Based on user interaction/feedback
    context: Dict[str, Any] = field(default_factory=dict)


class QueryExpander(LoggerMixin):
    """
    Intelligent query expansion and optimization system for RAG.
    
    Features:
    - Synonym expansion using WordNet and custom dictionaries
    - Semantic expansion using word embeddings
    - Phrase detection and boundary preservation
    - Term weighting and importance scoring
    - Multi-language support with automatic detection
    - Query history analysis for personalization
    - Context-aware expansion based on document metadata
    - Performance optimization with caching
    """
    
    def __init__(
        self,
        config: Optional[ExpansionConfig] = None,
        custom_synonyms: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize Query Expander.
        
        Args:
            config: Expansion configuration
            custom_synonyms: Custom synonym dictionary
        """
        self.config = config or ExpansionConfig()
        self.custom_synonyms = custom_synonyms or {}
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Initialize caches
        self.synonym_cache: Dict[str, List[str]] = {}
        self.semantic_cache: Dict[str, List[Tuple[str, float]]] = {}
        self.phrase_cache: Dict[str, List[str]] = {}
        
        # Query history for personalization
        self.query_history: List[QueryHistoryEntry] = []
        self.user_patterns: Dict[int, Dict[str, Any]] = {}
        
        # Performance metrics
        self.expansion_stats = {
            "total_expansions": 0,
            "total_expansion_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "synonym_expansions": 0,
            "semantic_expansions": 0,
            "phrase_detections": 0
        }
        
        self.logger.info("QueryExpander initialized successfully")
    
    def _initialize_nlp_components(self):
        """Initialize NLP components and models."""
        try:
            # Initialize spaCy
            if spacy is None:
                raise ConfigurationError("spaCy is required for query expansion")
            
            try:
                self.nlp = spacy.load(self.config.language_model.value)
            except OSError:
                self.logger.warning(f"Could not load {self.config.language_model.value}, falling back to en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize NLTK components
            if nltk is not None and self.config.enable_synonym_expansion:
                try:
                    # Download required NLTK data if not already present
                    nltk.data.find('corpora/wordnet')
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
            
            # Initialize semantic model
            if self.config.enable_semantic_expansion:
                try:
                    self.semantic_model = SentenceTransformer(self.config.semantic_model_name)
                except Exception as e:
                    self.logger.warning(f"Could not load semantic model: {e}")
                    self.config.enable_semantic_expansion = False
                    self.semantic_model = None
            else:
                self.semantic_model = None
            
            # Initialize TF-IDF vectorizer for term weighting
            if self.config.weighting_method in [WeightingMethod.TF_IDF, WeightingMethod.HYBRID]:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            else:
                self.tfidf_vectorizer = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {e}")
            raise ConfigurationError(f"NLP component initialization failed: {e}")
    
    def expand_query(
        self,
        query_result: QueryResult,
        user_context: Optional[Dict[str, Any]] = None,
        document_context: Optional[List[str]] = None
    ) -> ExpansionResult:
        """
        Expand and optimize a query for improved search performance.
        
        Args:
            query_result: Preprocessed query result
            user_context: User context for personalization
            document_context: Document context for contextual expansion
            
        Returns:
            Expansion result with enhanced query
        """
        start_time = time.time()
        
        try:
            # Extract user ID for personalization
            user_id = user_context.get("user_id") if user_context else None
            
            # Analyze original query
            doc = self.nlp(query_result.processed_query)
            
            # Step 1: Detect important phrases and entities
            phrase_boundaries, detected_phrases = self._detect_phrases(doc)
            
            # Step 2: Extract base terms for expansion
            base_terms = self._extract_expansion_terms(doc, phrase_boundaries)
            
            # Step 3: Apply different expansion strategies
            expanded_terms = []
            term_weights = []
            
            # Original terms with base weights
            for term in base_terms:
                term_weights.append(TermWeight(
                    term=term,
                    weight=self.config.base_term_weight,
                    term_type="original",
                    source="query",
                    confidence=1.0
                ))
            
            # Synonym expansion
            if self.config.enable_synonym_expansion:
                synonym_terms = self._expand_synonyms(base_terms)
                for term, weight in synonym_terms:
                    if term not in [t.term for t in term_weights]:
                        term_weights.append(TermWeight(
                            term=term,
                            weight=weight * self.config.synonym_weight,
                            term_type="synonym",
                            source="wordnet",
                            confidence=0.8
                        ))
                        expanded_terms.append(term)
            
            # Semantic expansion
            if self.config.enable_semantic_expansion and self.semantic_model:
                semantic_terms = self._expand_semantic_similar(
                    base_terms, query_result.processed_query
                )
                for term, similarity in semantic_terms:
                    if term not in [t.term for t in term_weights]:
                        term_weights.append(TermWeight(
                            term=term,
                            weight=similarity * self.config.semantic_weight,
                            term_type="semantic",
                            source="embeddings",
                            confidence=similarity
                        ))
                        expanded_terms.append(term)
            
            # Entity expansion
            if self.config.enable_entity_expansion:
                entity_terms = self._expand_entities(doc, query_result.metadata)
                for term, weight in entity_terms:
                    if term not in [t.term for t in term_weights]:
                        term_weights.append(TermWeight(
                            term=term,
                            weight=weight,
                            term_type="entity",
                            source="ner",
                            confidence=0.9
                        ))
                        expanded_terms.append(term)
            
            # Context expansion
            if self.config.enable_context_expansion and document_context:
                context_terms = self._expand_from_context(
                    base_terms, document_context
                )
                for term, weight in context_terms:
                    if term not in [t.term for t in term_weights]:
                        term_weights.append(TermWeight(
                            term=term,
                            weight=weight,
                            term_type="context",
                            source="documents",
                            confidence=0.7
                        ))
                        expanded_terms.append(term)
            
            # Personalized expansion
            if self.config.enable_personalization and user_id:
                personal_terms = self._expand_personalized(
                    query_result.processed_query, user_id
                )
                for term, weight in personal_terms:
                    if term not in [t.term for t in term_weights]:
                        term_weights.append(TermWeight(
                            term=term,
                            weight=weight,
                            term_type="personal",
                            source="history",
                            confidence=0.6
                        ))
                        expanded_terms.append(term)
            
            # Step 4: Apply term weighting optimization
            if self.config.weighting_method != WeightingMethod.UNIFORM:
                term_weights = self._optimize_term_weights(
                    term_weights, query_result.processed_query
                )
            
            # Step 5: Build expanded query
            expanded_query = self._build_expanded_query(
                query_result.processed_query, 
                term_weights, 
                detected_phrases
            )
            
            # Step 6: Apply query optimization
            if self.config.enable_query_reordering:
                expanded_query = self._optimize_query_structure(
                    expanded_query, term_weights
                )
            
            expansion_time = time.time() - start_time
            
            # Create result
            result = ExpansionResult(
                original_query=query_result.original_query,
                expanded_query=expanded_query,
                expanded_terms=expanded_terms[:self.config.max_expanded_terms],
                term_weights=term_weights,
                phrase_boundaries=phrase_boundaries,
                detected_phrases=detected_phrases,
                expansion_metadata={
                    "strategy": self.config.expansion_strategy.value,
                    "weighting_method": self.config.weighting_method.value,
                    "total_terms": len(term_weights),
                    "expansion_ratio": len(expanded_terms) / max(len(base_terms), 1),
                    "language": self._detect_language(query_result.processed_query),
                    "personalized": user_id is not None,
                    "context_used": document_context is not None
                },
                expansion_time=expansion_time
            )
            
            # Update statistics
            self._update_expansion_stats(expansion_time, len(expanded_terms))
            
            # Store in history for personalization
            if user_id and self.config.enable_personalization:
                self._add_to_history(query_result.original_query, expanded_query, user_id)
            
            self.logger.info(
                f"Query expansion completed: {len(expanded_terms)} terms added "
                f"in {expansion_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            expansion_time = time.time() - start_time
            self.logger.error(f"Query expansion failed: {e}")
            raise ProcessingError(f"Query expansion failed: {e}")
    
    def _detect_phrases(self, doc: Doc) -> Tuple[List[Tuple[int, int]], List[str]]:
        """Detect important phrases that should be preserved."""
        if not self.config.enable_phrase_detection:
            return [], []
        
        phrase_boundaries = []
        detected_phrases = []
        
        # Named entities
        for ent in doc.ents:
            phrase_boundaries.append((ent.start, ent.end))
            detected_phrases.append(ent.text)
        
        # Noun chunks
        for chunk in doc.noun_chunks:
            if (self.config.phrase_min_length <= len(chunk) <= self.config.phrase_max_length and
                chunk.text not in detected_phrases):
                phrase_boundaries.append((chunk.start, chunk.end))
                detected_phrases.append(chunk.text)
        
        # Custom phrase patterns (compound words, technical terms)
        for i, token in enumerate(doc[:-1]):
            if (token.pos_ in ["NOUN", "ADJ"] and 
                doc[i+1].pos_ in ["NOUN"] and
                not token.is_stop and not doc[i+1].is_stop):
                phrase_text = f"{token.text} {doc[i+1].text}"
                if phrase_text not in detected_phrases:
                    phrase_boundaries.append((i, i+2))
                    detected_phrases.append(phrase_text)
        
        self.expansion_stats["phrase_detections"] += len(detected_phrases)
        return phrase_boundaries, detected_phrases
    
    def _extract_expansion_terms(
        self, 
        doc: Doc, 
        phrase_boundaries: List[Tuple[int, int]]
    ) -> List[str]:
        """Extract terms suitable for expansion."""
        terms = []
        phrase_indices = set()
        
        # Mark tokens that are part of phrases
        for start, end in phrase_boundaries:
            phrase_indices.update(range(start, end))
        
        for i, token in enumerate(doc):
            # Skip if part of a phrase (phrases will be handled separately)
            if i in phrase_indices:
                continue
            
            # Include meaningful tokens
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                len(token.text) > 2 and
                token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]):
                
                # Use lemma for better expansion
                term = token.lemma_.lower()
                if term not in terms:
                    terms.append(term)
        
        return terms
    
    def _expand_synonyms(self, terms: List[str]) -> List[Tuple[str, float]]:
        """Expand terms using synonym lookup."""
        if not self.config.enable_synonym_expansion or wordnet is None:
            return []
        
        synonym_terms = []
        
        for term in terms:
            # Check cache first
            cache_key = f"syn:{term}"
            if self.config.enable_caching and cache_key in self.synonym_cache:
                synonyms = self.synonym_cache[cache_key]
                self.expansion_stats["cache_hits"] += 1
            else:
                synonyms = []
                
                # Custom synonyms
                if term in self.custom_synonyms:
                    synonyms.extend(self.custom_synonyms[term])
                
                # WordNet synonyms
                try:
                    for syn in wordnet.synsets(term):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace('_', ' ').lower()
                            if synonym != term and synonym not in synonyms:
                                synonyms.append(synonym)
                                if len(synonyms) >= self.config.wordnet_max_synonyms:
                                    break
                        if len(synonyms) >= self.config.wordnet_max_synonyms:
                            break
                except Exception:
                    pass
                
                # Cache results
                if self.config.enable_caching:
                    self.synonym_cache[cache_key] = synonyms
                    self._cleanup_cache(self.synonym_cache)
                
                self.expansion_stats["cache_misses"] += 1
            
            # Add synonyms with confidence weights
            for synonym in synonyms:
                synonym_terms.append((synonym, 0.8))
        
        self.expansion_stats["synonym_expansions"] += len(synonym_terms)
        return synonym_terms
    
    def _expand_semantic_similar(
        self, 
        terms: List[str], 
        context: str
    ) -> List[Tuple[str, float]]:
        """Expand terms using semantic similarity."""
        if not self.config.enable_semantic_expansion or not self.semantic_model:
            return []
        
        semantic_terms = []
        
        try:
            # Get embeddings for original terms
            term_embeddings = self.semantic_model.encode(terms)
            
            # Create a vocabulary of potential expansion terms
            # This could be enhanced with a larger vocabulary database
            candidate_terms = self._get_candidate_terms(terms, context)
            
            if candidate_terms:
                candidate_embeddings = self.semantic_model.encode(candidate_terms)
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(term_embeddings, candidate_embeddings)
                
                # Find similar terms above threshold
                for i, term in enumerate(terms):
                    for j, candidate in enumerate(candidate_terms):
                        similarity = similarities[i][j]
                        if (similarity >= self.config.min_similarity_threshold and 
                            candidate not in terms):
                            semantic_terms.append((candidate, similarity))
        
        except Exception as e:
            self.logger.warning(f"Semantic expansion failed: {e}")
        
        # Sort by similarity and limit results
        semantic_terms.sort(key=lambda x: x[1], reverse=True)
        semantic_terms = semantic_terms[:self.config.max_expanded_terms // 2]
        
        self.expansion_stats["semantic_expansions"] += len(semantic_terms)
        return semantic_terms
    
    def _get_candidate_terms(self, original_terms: List[str], context: str) -> List[str]:
        """Get candidate terms for semantic expansion."""
        candidates = []
        
        # Extract terms from context
        if context:
            doc = self.nlp(context)
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2 and
                    token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] and
                    token.lemma_.lower() not in original_terms):
                    candidates.append(token.lemma_.lower())
        
        # Could be enhanced with:
        # - Domain-specific vocabularies
        # - Previous successful expansions
        # - Common related terms database
        
        return list(set(candidates))[:100]  # Limit for performance
    
    def _expand_entities(
        self, 
        doc: Doc, 
        metadata: QueryMetadata
    ) -> List[Tuple[str, float]]:
        """Expand based on named entities and metadata."""
        entity_terms = []
        
        # Extract entity types and add related terms
        entity_types = {ent.label_ for ent in doc.ents}
        
        # Add related terms based on entity types
        entity_expansions = {
            "PERSON": ["individual", "people", "person", "human"],
            "ORG": ["company", "organization", "business", "corporation"],
            "GPE": ["location", "place", "country", "city"],
            "MONEY": ["cost", "price", "amount", "value"],
            "DATE": ["time", "period", "when", "schedule"],
            "PRODUCT": ["item", "goods", "service", "offering"]
        }
        
        for entity_type in entity_types:
            if entity_type in entity_expansions:
                for term in entity_expansions[entity_type]:
                    entity_terms.append((term, 0.7))
        
        # Add terms from metadata entities and keywords
        if hasattr(metadata, 'entities'):
            for entity in metadata.entities[:5]:  # Limit to top entities
                entity_terms.append((entity.lower(), 0.8))
        
        if hasattr(metadata, 'keywords'):
            for keyword in metadata.keywords[:5]:  # Limit to top keywords
                entity_terms.append((keyword.lower(), 0.7))
        
        return entity_terms
    
    def _expand_from_context(
        self, 
        terms: List[str], 
        document_context: List[str]
    ) -> List[Tuple[str, float]]:
        """Expand terms based on document context."""
        if not document_context:
            return []
        
        context_terms = []
        
        # Analyze document context to find related terms
        context_text = " ".join(document_context[:10])  # Limit context size
        doc = self.nlp(context_text)
        
        # Extract significant terms from context
        term_freq = Counter()
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2 and
                token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]):
                term_freq[token.lemma_.lower()] += 1
        
        # Find terms that co-occur with original terms
        for term, freq in term_freq.most_common(10):
            if term not in terms:
                # Weight based on frequency and position
                weight = min(0.8, freq / len(doc) * 10)
                context_terms.append((term, weight))
        
        return context_terms
    
    def _expand_personalized(
        self, 
        query: str, 
        user_id: int
    ) -> List[Tuple[str, float]]:
        """Expand based on user's query history and preferences."""
        if user_id not in self.user_patterns:
            self._analyze_user_patterns(user_id)
        
        personal_terms = []
        patterns = self.user_patterns.get(user_id, {})
        
        # Add terms from successful past queries
        preferred_terms = patterns.get("preferred_terms", [])
        for term, weight in preferred_terms[:5]:
            personal_terms.append((term, weight * 0.6))
        
        # Add terms from similar successful queries
        similar_queries = patterns.get("similar_queries", [])
        for similar_query, score in similar_queries[:3]:
            if score > 0.7:
                # Extract key terms from similar query
                doc = self.nlp(similar_query)
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        len(token.text) > 2 and
                        token.pos_ in ["NOUN", "VERB", "ADJ"]):
                        personal_terms.append((token.lemma_.lower(), score * 0.5))
        
        return personal_terms[:5]  # Limit personalized terms
    
    def _optimize_term_weights(
        self, 
        term_weights: List[TermWeight], 
        query: str
    ) -> List[TermWeight]:
        """Optimize term weights based on weighting method."""
        if self.config.weighting_method == WeightingMethod.UNIFORM:
            return term_weights
        
        # Apply TF-IDF weighting if available
        if (self.config.weighting_method in [WeightingMethod.TF_IDF, WeightingMethod.HYBRID] and
            self.tfidf_vectorizer is not None):
            
            try:
                # Fit on query and calculate weights
                self.tfidf_vectorizer.fit([query])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_scores = self.tfidf_vectorizer.transform([query]).toarray()[0]
                
                tfidf_dict = dict(zip(feature_names, tfidf_scores))
                
                for term_weight in term_weights:
                    term = term_weight.term.lower()
                    if term in tfidf_dict:
                        if self.config.weighting_method == WeightingMethod.TF_IDF:
                            term_weight.weight = tfidf_dict[term]
                        else:  # HYBRID
                            term_weight.weight = (term_weight.weight + tfidf_dict[term]) / 2
            
            except Exception as e:
                self.logger.warning(f"TF-IDF weighting failed: {e}")
        
        # Apply phrase weighting boost
        for term_weight in term_weights:
            if term_weight.term_type == "phrase":
                term_weight.weight *= self.config.phrase_weight
        
        # Normalize weights to reasonable range
        max_weight = max((tw.weight for tw in term_weights), default=1.0)
        if max_weight > 2.0:
            for term_weight in term_weights:
                term_weight.weight = term_weight.weight / max_weight * 2.0
        
        return term_weights
    
    def _build_expanded_query(
        self, 
        original_query: str, 
        term_weights: List[TermWeight], 
        phrases: List[str]
    ) -> str:
        """Build the expanded query string."""
        # Start with original query if preserving original terms
        if self.config.preserve_original_terms:
            expanded_parts = [original_query]
        else:
            expanded_parts = []
        
        # Add high-weight expanded terms
        high_weight_terms = [
            tw.term for tw in term_weights 
            if tw.weight >= 0.7 and tw.term_type != "original"
        ]
        
        # Add phrases
        expanded_parts.extend(phrases)
        
        # Add significant expansion terms
        expanded_parts.extend(high_weight_terms[:self.config.max_expanded_terms])
        
        # Join and clean up
        expanded_query = " ".join(expanded_parts)
        expanded_query = re.sub(r'\s+', ' ', expanded_query).strip()
        
        # Limit length
        if len(expanded_query) > self.config.max_query_length:
            expanded_query = expanded_query[:self.config.max_query_length].rsplit(' ', 1)[0]
        
        return expanded_query
    
    def _optimize_query_structure(
        self, 
        query: str, 
        term_weights: List[TermWeight]
    ) -> str:
        """Optimize the structure and ordering of the expanded query."""
        if not self.config.enable_query_reordering:
            return query
        
        # Create weight mapping
        weight_map = {tw.term: tw.weight for tw in term_weights}
        
        # Tokenize and sort by importance
        terms = query.split()
        
        # Sort terms by weight (descending) while preserving some original order
        weighted_terms = []
        for i, term in enumerate(terms):
            weight = weight_map.get(term, 0.5)
            position_bonus = max(0, (len(terms) - i) / len(terms) * 0.2)
            weighted_terms.append((term, weight + position_bonus))
        
        # Sort but keep most important terms first
        weighted_terms.sort(key=lambda x: x[1], reverse=True)
        optimized_terms = [term for term, _ in weighted_terms]
        
        return " ".join(optimized_terms)
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not self.config.auto_detect_language:
            return self.config.language_model.value.split('_')[0]
        
        try:
            # Simple language detection based on spaCy
            doc = self.nlp(text[:100])  # Use first 100 chars
            return doc.lang_
        except:
            return "en"  # Default to English
    
    def _analyze_user_patterns(self, user_id: int):
        """Analyze user query patterns for personalization."""
        if not self.config.enable_personalization:
            return
        
        # Get user's recent query history
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.history_window_days)
        user_queries = [
            entry for entry in self.query_history
            if entry.user_id == user_id and entry.timestamp >= cutoff_date
        ]
        
        if len(user_queries) < self.config.min_history_queries:
            return
        
        # Analyze patterns
        patterns = {
            "preferred_terms": [],
            "similar_queries": [],
            "successful_expansions": [],
            "query_topics": []
        }
        
        # Find frequently used terms
        term_freq = Counter()
        successful_queries = [q for q in user_queries if q.success_score > 0.7]
        
        for query in successful_queries:
            doc = self.nlp(query.expanded_query)
            for token in doc:
                if (not token.is_stop and not token.is_punct and 
                    len(token.text) > 2):
                    term_freq[token.lemma_.lower()] += 1
        
        # Convert to preferred terms with weights
        total_queries = len(successful_queries)
        for term, freq in term_freq.most_common(20):
            weight = freq / total_queries
            if weight > 0.2:  # At least 20% of queries
                patterns["preferred_terms"].append((term, weight))
        
        self.user_patterns[user_id] = patterns
    
    def _add_to_history(self, original_query: str, expanded_query: str, user_id: int):
        """Add query to history for future personalization."""
        entry = QueryHistoryEntry(
            query=original_query,
            expanded_query=expanded_query,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
        self.query_history.append(entry)
        
        # Cleanup old entries
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.history_window_days * 2)
        self.query_history = [
            e for e in self.query_history if e.timestamp >= cutoff_date
        ]
    
    def _cleanup_cache(self, cache: Dict[str, Any]):
        """Clean up cache if it exceeds size limit."""
        if len(cache) > self.config.max_cache_size:
            # Remove oldest 10% of entries (simple FIFO)
            remove_count = len(cache) // 10
            keys_to_remove = list(cache.keys())[:remove_count]
            for key in keys_to_remove:
                del cache[key]
    
    def _update_expansion_stats(self, expansion_time: float, num_terms: int):
        """Update expansion statistics."""
        self.expansion_stats["total_expansions"] += 1
        self.expansion_stats["total_expansion_time"] += expansion_time
    
    def update_expansion_feedback(
        self, 
        query: str, 
        expansion_result: ExpansionResult, 
        success_score: float
    ):
        """Update expansion effectiveness based on feedback."""
        # Find matching history entry and update success score
        for entry in self.query_history:
            if (entry.query == query and 
                entry.expanded_query == expansion_result.expanded_query):
                entry.success_score = success_score
                break
        
        # Update user patterns if needed
        # This could trigger re-analysis of user patterns
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get expansion statistics."""
        total_expansions = self.expansion_stats["total_expansions"]
        
        return {
            "total_expansions": total_expansions,
            "average_expansion_time": (
                self.expansion_stats["total_expansion_time"] / total_expansions
                if total_expansions > 0 else 0.0
            ),
            "cache_hit_rate": (
                self.expansion_stats["cache_hits"] / 
                (self.expansion_stats["cache_hits"] + self.expansion_stats["cache_misses"])
                if (self.expansion_stats["cache_hits"] + self.expansion_stats["cache_misses"]) > 0
                else 0.0
            ),
            "synonym_expansions": self.expansion_stats["synonym_expansions"],
            "semantic_expansions": self.expansion_stats["semantic_expansions"],
            "phrase_detections": self.expansion_stats["phrase_detections"],
            "cache_sizes": {
                "synonym_cache": len(self.synonym_cache),
                "semantic_cache": len(self.semantic_cache),
                "phrase_cache": len(self.phrase_cache)
            },
            "config": {
                "expansion_strategy": self.config.expansion_strategy.value,
                "weighting_method": self.config.weighting_method.value,
                "max_expanded_terms": self.config.max_expanded_terms,
                "personalization_enabled": self.config.enable_personalization
            }
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self.synonym_cache.clear()
        self.semantic_cache.clear()
        self.phrase_cache.clear()
        self.logger.info("Query expansion caches cleared")


def create_query_expander(
    config: Optional[ExpansionConfig] = None,
    custom_synonyms: Optional[Dict[str, List[str]]] = None
) -> QueryExpander:
    """
    Create and configure a QueryExpander instance.
    
    Args:
        config: Expansion configuration
        custom_synonyms: Custom synonym dictionary
        
    Returns:
        Configured QueryExpander
    """
    return QueryExpander(
        config=config or ExpansionConfig(),
        custom_synonyms=custom_synonyms
    )


def create_expansion_config(
    strategy: ExpansionStrategy = ExpansionStrategy.BALANCED,
    max_terms: int = 10,
    enable_personalization: bool = True,
    **kwargs
) -> ExpansionConfig:
    """
    Create expansion configuration with simplified interface.
    
    Args:
        strategy: Expansion strategy
        max_terms: Maximum expanded terms
        enable_personalization: Enable personalization
        **kwargs: Additional configuration parameters
        
    Returns:
        Expansion configuration
    """
    return ExpansionConfig(
        expansion_strategy=strategy,
        max_expanded_terms=max_terms,
        enable_personalization=enable_personalization,
        **kwargs
    )