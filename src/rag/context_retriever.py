"""
Context Retrieval and Ranking System for RAG.

This module provides intelligent document context retrieval and ranking algorithms
based on similarity scores, document metadata, user context, and advanced ranking strategies.
"""

import time
import re
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np

from src.core.logging import LoggerMixin
from src.core.exceptions import RetrievalError, ConfigurationError
from src.rag.vector_search_engine import VectorSearchEngine, VectorSearchResult, SearchHit
from src.rag.query_preprocessor import QueryResult


class RankingStrategy(Enum):
    """Ranking strategies for context retrieval."""
    SIMILARITY_ONLY = "similarity_only"           # Pure similarity ranking
    BM25_HYBRID = "bm25_hybrid"                   # BM25 + similarity hybrid
    TEMPORAL_AWARE = "temporal_aware"             # Time-aware ranking
    METADATA_WEIGHTED = "metadata_weighted"       # Metadata-based weighting
    DIVERSIFIED = "diversified"                   # Diversified ranking
    LEARNING_TO_RANK = "learning_to_rank"         # ML-based ranking
    CUSTOM = "custom"                             # Custom ranking function


class RetrievalMode(Enum):
    """Retrieval modes for different use cases."""
    PRECISE = "precise"                           # High precision retrieval
    COMPREHENSIVE = "comprehensive"               # Broad context retrieval
    FOCUSED = "focused"                          # Narrow, focused retrieval
    EXPLORATORY = "exploratory"                 # Exploratory retrieval


@dataclass
class ContextConfig:
    """Configuration for context retrieval and ranking."""
    
    # Retrieval parameters
    max_contexts: int = 10
    min_contexts: int = 1
    min_similarity_threshold: float = 0.1
    max_context_length: int = 2000
    context_overlap_threshold: float = 0.8
    
    # Ranking parameters
    ranking_strategy: RankingStrategy = RankingStrategy.BM25_HYBRID
    retrieval_mode: RetrievalMode = RetrievalMode.COMPREHENSIVE
    
    # Weighting factors
    similarity_weight: float = 0.6
    recency_weight: float = 0.2
    popularity_weight: float = 0.1
    relevance_weight: float = 0.1
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Diversity parameters
    enable_diversification: bool = True
    diversity_threshold: float = 0.7
    max_similar_contexts: int = 3
    
    # Context aggregation
    enable_context_fusion: bool = True
    context_fusion_method: str = "weighted"  # "weighted", "hierarchical", "graph"
    
    # Advanced options
    enable_reranking: bool = True
    enable_deduplication: bool = True
    enable_context_expansion: bool = False
    expansion_window: int = 2


@dataclass
class DocumentContext:
    """Document context with metadata and ranking information."""
    id: Union[str, int]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    relevance_score: float = 0.0
    ranking_score: float = 0.0
    source_info: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Ranking factors
    recency_score: float = 0.0
    popularity_score: float = 0.0
    diversity_score: float = 0.0
    
    def __post_init__(self):
        """Initialize derived fields."""
        if not hasattr(self, 'word_count'):
            self.word_count = len(self.content.split())
        
        if not hasattr(self, 'char_count'):
            self.char_count = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "ranking_score": self.ranking_score,
            "source_info": self.source_info,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat(),
            "word_count": getattr(self, 'word_count', 0),
            "char_count": getattr(self, 'char_count', 0),
            "scores": {
                "recency": self.recency_score,
                "popularity": self.popularity_score,
                "diversity": self.diversity_score
            }
        }


@dataclass
class RetrievalResult:
    """Result of context retrieval and ranking."""
    contexts: List[DocumentContext]
    total_retrieved: int
    total_ranked: int
    retrieval_time: float
    ranking_time: float
    query_info: Dict[str, Any] = field(default_factory=dict)
    ranking_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_content_texts(self) -> List[str]:
        """Get list of context content texts."""
        return [ctx.content for ctx in self.contexts]
    
    def get_top_contexts(self, k: int) -> List[DocumentContext]:
        """Get top k contexts by ranking score."""
        return sorted(
            self.contexts, 
            key=lambda ctx: ctx.ranking_score, 
            reverse=True
        )[:k]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "contexts": [ctx.to_dict() for ctx in self.contexts],
            "total_retrieved": self.total_retrieved,
            "total_ranked": self.total_ranked,
            "retrieval_time": self.retrieval_time,
            "ranking_time": self.ranking_time,
            "query_info": self.query_info,
            "ranking_metadata": self.ranking_metadata
        }


class ContextRetriever(LoggerMixin):
    """
    Intelligent context retrieval and ranking system for RAG.
    
    Features:
    - Multiple ranking strategies (similarity, BM25, temporal, metadata)
    - Context deduplication and diversification
    - Metadata-aware ranking and filtering
    - User context and personalization support
    - Advanced text processing and relevance scoring
    - Context fusion and aggregation
    - Performance optimization and caching
    """
    
    def __init__(
        self,
        vector_search_engine: VectorSearchEngine,
        config: Optional[ContextConfig] = None
    ):
        """
        Initialize Context Retriever.
        
        Args:
            vector_search_engine: Vector search engine instance
            config: Retrieval configuration
        """
        self.vector_search_engine = vector_search_engine
        self.config = config or ContextConfig()
        
        # Document frequency cache for BM25
        self.document_frequencies: Dict[str, int] = {}
        self.total_documents: int = 0
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        
        # Performance metrics
        self.retrieval_stats = {
            "total_retrievals": 0,
            "total_retrieval_time": 0.0,
            "total_ranking_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info("ContextRetriever initialized successfully")
    
    def retrieve_contexts(
        self,
        collection_name: str,
        query_result: QueryResult,
        user_context: Optional[Dict[str, Any]] = None,
        custom_filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve and rank relevant document contexts.
        
        Args:
            collection_name: Name of the collection to search
            query_result: Preprocessed query result with embeddings
            user_context: User context for personalization
            custom_filters: Custom filtering criteria
            
        Returns:
            Ranked context retrieval results
        """
        start_time = time.time()
        
        try:
            # Step 1: Vector similarity search
            search_result = self._perform_vector_search(
                collection_name, query_result, custom_filters
            )
            retrieval_time = time.time() - start_time
            
            # Step 2: Convert search hits to document contexts
            contexts = self._convert_search_hits(search_result, query_result)
            
            # Step 3: Apply ranking strategy
            ranking_start = time.time()
            ranked_contexts = self._rank_contexts(
                contexts, query_result, user_context
            )
            ranking_time = time.time() - ranking_start
            
            # Step 4: Apply post-processing
            final_contexts = self._post_process_contexts(
                ranked_contexts, query_result
            )
            
            # Create result
            result = RetrievalResult(
                contexts=final_contexts,
                total_retrieved=len(contexts),
                total_ranked=len(final_contexts),
                retrieval_time=retrieval_time,
                ranking_time=ranking_time,
                query_info={
                    "original_query": query_result.original_query,
                    "processed_query": query_result.processed_query,
                    "query_type": query_result.metadata.query_type.value,
                    "intent": query_result.metadata.intent.value
                },
                ranking_metadata={
                    "strategy": self.config.ranking_strategy.value,
                    "mode": self.config.retrieval_mode.value,
                    "diversification_enabled": self.config.enable_diversification,
                    "reranking_enabled": self.config.enable_reranking
                }
            )
            
            # Update metrics
            self._update_retrieval_stats(retrieval_time, ranking_time)
            
            self.logger.info(
                f"Context retrieval completed: {len(final_contexts)} contexts "
                f"in {retrieval_time + ranking_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            raise RetrievalError(f"Context retrieval failed: {e}")
    
    def _perform_vector_search(
        self,
        collection_name: str,
        query_result: QueryResult,
        custom_filters: Optional[Dict[str, Any]]
    ) -> VectorSearchResult:
        """Perform vector similarity search."""
        from src.rag.vector_search_engine import VectorSearchRequest, SearchMode
        
        # Determine search parameters based on retrieval mode
        mode_mapping = {
            RetrievalMode.PRECISE: SearchMode.PRECISION,
            RetrievalMode.COMPREHENSIVE: SearchMode.BALANCED,
            RetrievalMode.FOCUSED: SearchMode.SPEED,
            RetrievalMode.EXPLORATORY: SearchMode.ADAPTIVE
        }
        
        search_mode = mode_mapping.get(self.config.retrieval_mode, SearchMode.BALANCED)
        
        # Build filter expression
        filter_expr = self._build_filter_expression(custom_filters)
        
        # Create search request
        search_request = VectorSearchRequest(
            query_vectors=[query_result.embedding.tolist()],
            limit=self.config.max_contexts * 2,  # Retrieve more for better ranking
            similarity_threshold=self.config.min_similarity_threshold,
            filter_expression=filter_expr,
            output_fields=["content", "metadata", "source_info"],
            search_mode=search_mode
        )
        
        # Perform search
        return self.vector_search_engine.search(collection_name, search_request)
    
    def _convert_search_hits(
        self,
        search_result: VectorSearchResult,
        query_result: QueryResult
    ) -> List[DocumentContext]:
        """Convert search hits to document contexts."""
        contexts = []
        
        for hit_list in search_result.hits:
            for hit in hit_list:
                # Extract content and metadata
                content = hit.entity.get("content", "")
                metadata = hit.entity.get("metadata", {})
                source_info = hit.entity.get("source_info", {})
                
                # Create document context
                context = DocumentContext(
                    id=hit.id,
                    content=content,
                    metadata=metadata,
                    similarity_score=hit.score,
                    source_info=source_info
                )
                
                contexts.append(context)
        
        return contexts
    
    def _rank_contexts(
        self,
        contexts: List[DocumentContext],
        query_result: QueryResult,
        user_context: Optional[Dict[str, Any]]
    ) -> List[DocumentContext]:
        """Apply ranking strategy to contexts."""
        if self.config.ranking_strategy == RankingStrategy.SIMILARITY_ONLY:
            return self._rank_by_similarity(contexts)
        elif self.config.ranking_strategy == RankingStrategy.BM25_HYBRID:
            return self._rank_bm25_hybrid(contexts, query_result)
        elif self.config.ranking_strategy == RankingStrategy.TEMPORAL_AWARE:
            return self._rank_temporal_aware(contexts, user_context)
        elif self.config.ranking_strategy == RankingStrategy.METADATA_WEIGHTED:
            return self._rank_metadata_weighted(contexts, query_result)
        elif self.config.ranking_strategy == RankingStrategy.DIVERSIFIED:
            return self._rank_diversified(contexts, query_result)
        else:
            # Default to similarity-only ranking
            return self._rank_by_similarity(contexts)
    
    def _rank_by_similarity(self, contexts: List[DocumentContext]) -> List[DocumentContext]:
        """Rank contexts by similarity score only."""
        for context in contexts:
            context.ranking_score = context.similarity_score
        
        return sorted(contexts, key=lambda ctx: ctx.ranking_score, reverse=True)
    
    def _rank_bm25_hybrid(
        self,
        contexts: List[DocumentContext],
        query_result: QueryResult
    ) -> List[DocumentContext]:
        """Rank contexts using BM25 + similarity hybrid approach."""
        # Extract query terms
        query_terms = self._extract_query_terms(query_result.processed_query)
        
        # Calculate BM25 scores
        for context in contexts:
            bm25_score = self._calculate_bm25_score(
                context.content, query_terms
            )
            
            # Combine BM25 and similarity scores
            context.relevance_score = bm25_score
            context.ranking_score = (
                self.config.similarity_weight * context.similarity_score +
                (1 - self.config.similarity_weight) * bm25_score
            )
        
        return sorted(contexts, key=lambda ctx: ctx.ranking_score, reverse=True)
    
    def _rank_temporal_aware(
        self,
        contexts: List[DocumentContext],
        user_context: Optional[Dict[str, Any]]
    ) -> List[DocumentContext]:
        """Rank contexts with temporal awareness."""
        current_time = datetime.utcnow()
        
        for context in contexts:
            # Calculate recency score
            context.recency_score = self._calculate_recency_score(
                context, current_time
            )
            
            # Calculate popularity score
            context.popularity_score = self._calculate_popularity_score(context)
            
            # Combine scores
            context.ranking_score = (
                self.config.similarity_weight * context.similarity_score +
                self.config.recency_weight * context.recency_score +
                self.config.popularity_weight * context.popularity_score
            )
        
        return sorted(contexts, key=lambda ctx: ctx.ranking_score, reverse=True)
    
    def _rank_metadata_weighted(
        self,
        contexts: List[DocumentContext],
        query_result: QueryResult
    ) -> List[DocumentContext]:
        """Rank contexts using metadata-based weighting."""
        for context in contexts:
            # Calculate metadata relevance
            metadata_score = self._calculate_metadata_relevance(
                context, query_result
            )
            
            # Combine with similarity
            context.ranking_score = (
                self.config.similarity_weight * context.similarity_score +
                self.config.relevance_weight * metadata_score
            )
        
        return sorted(contexts, key=lambda ctx: ctx.ranking_score, reverse=True)
    
    def _rank_diversified(
        self,
        contexts: List[DocumentContext],
        query_result: QueryResult
    ) -> List[DocumentContext]:
        """Rank contexts with diversification."""
        if not contexts:
            return contexts
        
        # Sort by initial similarity
        contexts.sort(key=lambda ctx: ctx.similarity_score, reverse=True)
        
        # Apply Maximal Marginal Relevance (MMR) for diversification
        selected = [contexts[0]]  # Start with top result
        remaining = contexts[1:]
        
        while remaining and len(selected) < self.config.max_contexts:
            best_idx = 0
            best_score = -float('inf')
            
            for i, candidate in enumerate(remaining):
                # Calculate MMR score
                relevance = candidate.similarity_score
                max_similarity = max(
                    self._calculate_content_similarity(candidate, selected_ctx)
                    for selected_ctx in selected
                )
                
                mmr_score = (
                    self.config.similarity_weight * relevance -
                    (1 - self.config.similarity_weight) * max_similarity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best candidate
            best_candidate = remaining.pop(best_idx)
            best_candidate.ranking_score = best_score
            best_candidate.diversity_score = 1.0 - max(
                self._calculate_content_similarity(best_candidate, selected_ctx)
                for selected_ctx in selected
            )
            selected.append(best_candidate)
        
        return selected
    
    def _post_process_contexts(
        self,
        contexts: List[DocumentContext],
        query_result: QueryResult
    ) -> List[DocumentContext]:
        """Apply post-processing to ranked contexts."""
        processed_contexts = contexts
        
        # Apply deduplication
        if self.config.enable_deduplication:
            processed_contexts = self._deduplicate_contexts(processed_contexts)
        
        # Apply reranking
        if self.config.enable_reranking:
            processed_contexts = self._rerank_contexts(processed_contexts, query_result)
        
        # Limit to max contexts
        processed_contexts = processed_contexts[:self.config.max_contexts]
        
        # Ensure minimum contexts
        if len(processed_contexts) < self.config.min_contexts:
            self.logger.warning(
                f"Retrieved {len(processed_contexts)} contexts, "
                f"below minimum of {self.config.min_contexts}"
            )
        
        return processed_contexts
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract terms from query for BM25 calculation."""
        # Simple tokenization - could be enhanced with proper NLP
        terms = re.findall(r'\b\w+\b', query.lower())
        return [term for term in terms if len(term) > 2]
    
    def _calculate_bm25_score(self, document: str, query_terms: List[str]) -> float:
        """Calculate BM25 score for document given query terms."""
        if not query_terms:
            return 0.0
        
        # Tokenize document
        doc_terms = re.findall(r'\b\w+\b', document.lower())
        doc_length = len(doc_terms)
        
        if doc_length == 0:
            return 0.0
        
        # Calculate term frequencies
        term_freqs = Counter(doc_terms)
        
        # Estimate average document length (could be cached)
        avg_doc_length = 100  # Default estimate
        
        score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            if tf > 0:
                # BM25 formula
                idf = math.log((self.total_documents + 1) / (self.document_frequencies.get(term, 1) + 1))
                score += idf * (tf * (self.config.bm25_k1 + 1)) / (
                    tf + self.config.bm25_k1 * (
                        1 - self.config.bm25_b + 
                        self.config.bm25_b * doc_length / avg_doc_length
                    )
                )
        
        return score / len(query_terms)
    
    def _calculate_recency_score(
        self, 
        context: DocumentContext, 
        current_time: datetime
    ) -> float:
        """Calculate recency score based on document timestamp."""
        # Try to get timestamp from metadata
        doc_time = None
        if 'timestamp' in context.metadata:
            try:
                if isinstance(context.metadata['timestamp'], str):
                    doc_time = datetime.fromisoformat(context.metadata['timestamp'])
                elif isinstance(context.metadata['timestamp'], datetime):
                    doc_time = context.metadata['timestamp']
            except:
                pass
        
        if doc_time is None:
            # Default to retrieval time if no document time available
            doc_time = context.retrieval_timestamp
        
        # Calculate time difference in days
        time_diff = (current_time - doc_time).total_seconds() / (24 * 3600)
        
        # Exponential decay for recency (higher score for more recent)
        return math.exp(-time_diff / 30)  # 30-day half-life
    
    def _calculate_popularity_score(self, context: DocumentContext) -> float:
        """Calculate popularity score based on metadata."""
        # Try to get popularity indicators from metadata
        view_count = context.metadata.get('view_count', 0)
        like_count = context.metadata.get('like_count', 0)
        comment_count = context.metadata.get('comment_count', 0)
        
        # Simple popularity formula (could be enhanced)
        popularity = math.log(1 + view_count + like_count * 2 + comment_count * 3)
        
        # Normalize to 0-1 range (simple approach)
        return min(popularity / 10.0, 1.0)
    
    def _calculate_metadata_relevance(
        self,
        context: DocumentContext,
        query_result: QueryResult
    ) -> float:
        """Calculate relevance score based on metadata matching."""
        relevance = 0.0
        
        # Check for keyword matches in metadata
        query_keywords = set(query_result.metadata.keywords)
        doc_keywords = set(context.metadata.get('keywords', []))
        
        if query_keywords and doc_keywords:
            keyword_overlap = len(query_keywords & doc_keywords) / len(query_keywords)
            relevance += keyword_overlap * 0.5
        
        # Check for entity matches
        query_entities = set(query_result.metadata.entities)
        doc_entities = set(context.metadata.get('entities', []))
        
        if query_entities and doc_entities:
            entity_overlap = len(query_entities & doc_entities) / len(query_entities)
            relevance += entity_overlap * 0.3
        
        # Check document type relevance
        if 'document_type' in context.metadata:
            doc_type = context.metadata['document_type']
            # Simple type relevance (could be enhanced with ML)
            if query_result.metadata.intent.value in ['question', 'explanation']:
                if doc_type in ['article', 'documentation', 'faq']:
                    relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _calculate_content_similarity(
        self,
        context1: DocumentContext,
        context2: DocumentContext
    ) -> float:
        """Calculate content similarity between two contexts."""
        # Simple Jaccard similarity on words (could be enhanced with embeddings)
        words1 = set(re.findall(r'\b\w+\b', context1.content.lower()))
        words2 = set(re.findall(r'\b\w+\b', context2.content.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_contexts(self, contexts: List[DocumentContext]) -> List[DocumentContext]:
        """Remove duplicate or highly similar contexts."""
        if not self.config.enable_deduplication:
            return contexts
        
        deduplicated = []
        
        for context in contexts:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self._calculate_content_similarity(context, existing)
                if similarity > self.config.context_overlap_threshold:
                    is_duplicate = True
                    # Keep the one with higher ranking score
                    if context.ranking_score > existing.ranking_score:
                        deduplicated.remove(existing)
                        deduplicated.append(context)
                    break
            
            if not is_duplicate:
                deduplicated.append(context)
        
        return deduplicated
    
    def _rerank_contexts(
        self,
        contexts: List[DocumentContext],
        query_result: QueryResult
    ) -> List[DocumentContext]:
        """Apply final reranking to contexts."""
        # Simple reranking based on query-context matching
        for context in contexts:
            # Calculate query-specific relevance
            query_relevance = self._calculate_query_context_relevance(
                context, query_result
            )
            
            # Adjust ranking score
            context.ranking_score = (
                0.8 * context.ranking_score +
                0.2 * query_relevance
            )
        
        return sorted(contexts, key=lambda ctx: ctx.ranking_score, reverse=True)
    
    def _calculate_query_context_relevance(
        self,
        context: DocumentContext,
        query_result: QueryResult
    ) -> float:
        """Calculate query-specific relevance for a context."""
        relevance = 0.0
        
        # Check for exact phrase matches
        query_lower = query_result.processed_query.lower()
        content_lower = context.content.lower()
        
        # Phrase matching
        if query_lower in content_lower:
            relevance += 0.5
        
        # Word overlap
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        if query_words:
            word_overlap = len(query_words & content_words) / len(query_words)
            relevance += word_overlap * 0.3
        
        # Context length penalty for very long contexts
        if len(context.content) > self.config.max_context_length:
            relevance *= 0.8
        
        return min(relevance, 1.0)
    
    def _build_filter_expression(
        self,
        custom_filters: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Build Milvus filter expression from custom filters."""
        if not custom_filters:
            return None
        
        conditions = []
        
        for field, value in custom_filters.items():
            if isinstance(value, str):
                conditions.append(f'{field} == "{value}"')
            elif isinstance(value, (int, float)):
                conditions.append(f'{field} == {value}')
            elif isinstance(value, list):
                # Handle list of values (IN operator)
                if all(isinstance(v, str) for v in value):
                    value_str = '", "'.join(value)
                    conditions.append(f'{field} in ["{value_str}"]')
                else:
                    value_str = ', '.join(str(v) for v in value)
                    conditions.append(f'{field} in [{value_str}]')
        
        return ' and '.join(conditions) if conditions else None
    
    def _update_retrieval_stats(self, retrieval_time: float, ranking_time: float) -> None:
        """Update retrieval statistics."""
        self.retrieval_stats["total_retrievals"] += 1
        self.retrieval_stats["total_retrieval_time"] += retrieval_time
        self.retrieval_stats["total_ranking_time"] += ranking_time
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        total_retrievals = self.retrieval_stats["total_retrievals"]
        
        return {
            "total_retrievals": total_retrievals,
            "average_retrieval_time": (
                self.retrieval_stats["total_retrieval_time"] / total_retrievals
                if total_retrievals > 0 else 0.0
            ),
            "average_ranking_time": (
                self.retrieval_stats["total_ranking_time"] / total_retrievals
                if total_retrievals > 0 else 0.0
            ),
            "cache_hit_rate": (
                self.retrieval_stats["cache_hits"] / 
                (self.retrieval_stats["cache_hits"] + self.retrieval_stats["cache_misses"])
                if (self.retrieval_stats["cache_hits"] + self.retrieval_stats["cache_misses"]) > 0 
                else 0.0
            ),
            "config": {
                "ranking_strategy": self.config.ranking_strategy.value,
                "retrieval_mode": self.config.retrieval_mode.value,
                "max_contexts": self.config.max_contexts,
                "diversification_enabled": self.config.enable_diversification
            }
        }


def create_context_retriever(
    vector_search_engine: VectorSearchEngine,
    config: Optional[ContextConfig] = None
) -> ContextRetriever:
    """
    Create and configure a ContextRetriever instance.
    
    Args:
        vector_search_engine: Vector search engine instance
        config: Retrieval configuration
        
    Returns:
        Configured ContextRetriever
    """
    return ContextRetriever(
        vector_search_engine=vector_search_engine,
        config=config or ContextConfig()
    )