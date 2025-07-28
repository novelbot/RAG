"""
Relevance Scoring Algorithms for RAG System.

This module provides sophisticated relevance scoring algorithms for ranking search results,
including BM25, TF-IDF, semantic similarity, hybrid scoring, and learning-to-rank capabilities
based on user feedback.
"""

import time
import math
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import json
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RankingSVM
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    StandardScaler = None
    RandomForestRegressor = None
    RankingSVM = None

from sentence_transformers import SentenceTransformer

from src.core.logging import LoggerMixin
from src.core.exceptions import ScoringError, ConfigurationError
from src.rag.query_preprocessor import QueryResult
from src.rag.query_expander import ExpansionResult
from src.rag.context_retriever import DocumentContext, RetrievalResult


class ScoringAlgorithm(Enum):
    """Scoring algorithm types."""
    VECTOR_SIMILARITY = "vector_similarity"    # Cosine similarity of embeddings
    BM25 = "bm25"                             # BM25 ranking function
    TF_IDF = "tf_idf"                         # TF-IDF with cosine similarity
    SEMANTIC = "semantic"                     # Semantic similarity with transformers
    HYBRID = "hybrid"                         # Weighted combination of algorithms
    LEARNING_TO_RANK = "learning_to_rank"     # ML-based ranking from feedback
    CUSTOM = "custom"                         # Custom scoring function


class ScoreNormalization(Enum):
    """Score normalization methods."""
    NONE = "none"                             # No normalization
    MIN_MAX = "min_max"                       # Min-max normalization [0,1]
    Z_SCORE = "z_score"                       # Z-score normalization
    SOFTMAX = "softmax"                       # Softmax normalization
    RANK = "rank"                             # Rank-based normalization


class LearningMode(Enum):
    """Learning-to-rank modes."""
    POINTWISE = "pointwise"                   # Individual document scoring
    PAIRWISE = "pairwise"                     # Pairwise comparison
    LISTWISE = "listwise"                     # List-based ranking


@dataclass
class ScoringConfig:
    """Configuration for relevance scoring algorithms."""
    
    # Algorithm selection
    primary_algorithms: List[ScoringAlgorithm] = field(default_factory=lambda: [
        ScoringAlgorithm.VECTOR_SIMILARITY,
        ScoringAlgorithm.BM25,
        ScoringAlgorithm.SEMANTIC
    ])
    use_hybrid: bool = True
    use_learning_to_rank: bool = True
    
    # Algorithm weights for hybrid scoring
    vector_weight: float = 0.3
    bm25_weight: float = 0.3
    tfidf_weight: float = 0.2
    semantic_weight: float = 0.2
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    
    # TF-IDF parameters
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_norm: str = "l2"
    
    # Semantic similarity parameters
    semantic_model_name: str = "all-MiniLM-L6-v2"
    semantic_batch_size: int = 32
    
    # Score normalization
    normalization_method: ScoreNormalization = ScoreNormalization.MIN_MAX
    normalize_individual_scores: bool = True
    normalize_final_scores: bool = True
    
    # Learning-to-rank parameters
    learning_mode: LearningMode = LearningMode.POINTWISE
    min_feedback_samples: int = 50
    retrain_frequency_hours: int = 24
    feature_selection: bool = True
    
    # Performance optimization
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 5000
    precompute_document_stats: bool = True
    
    # Advanced options
    query_dependent_weights: bool = True
    context_aware_scoring: bool = True
    temporal_decay_factor: float = 0.1
    diversity_bonus: float = 0.1


@dataclass
class ScoreComponents:
    """Individual scoring algorithm results."""
    vector_similarity: float = 0.0
    bm25_score: float = 0.0
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    learning_score: float = 0.0
    custom_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_similarity": self.vector_similarity,
            "bm25_score": self.bm25_score,
            "tfidf_score": self.tfidf_score,
            "semantic_score": self.semantic_score,
            "hybrid_score": self.hybrid_score,
            "learning_score": self.learning_score,
            "custom_scores": self.custom_scores
        }
    
    def get_scores_array(self) -> np.ndarray:
        """Get scores as numpy array for ML processing."""
        return np.array([
            self.vector_similarity,
            self.bm25_score,
            self.tfidf_score,
            self.semantic_score
        ])


@dataclass
class ScoringResult:
    """Result of relevance scoring with detailed breakdown."""
    scored_contexts: List[Tuple[DocumentContext, ScoreComponents]]
    final_scores: List[float]
    ranking: List[int]  # Indices sorted by relevance
    scoring_metadata: Dict[str, Any] = field(default_factory=dict)
    scoring_time: float = 0.0
    
    def get_ranked_contexts(self) -> List[DocumentContext]:
        """Get contexts sorted by relevance score."""
        return [self.scored_contexts[i][0] for i in self.ranking]
    
    def get_top_k(self, k: int) -> List[Tuple[DocumentContext, float]]:
        """Get top k results with scores."""
        top_indices = self.ranking[:k]
        return [
            (self.scored_contexts[i][0], self.final_scores[i])
            for i in top_indices
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scored_contexts": [
                {
                    "context": ctx.to_dict(),
                    "scores": scores.to_dict()
                }
                for ctx, scores in self.scored_contexts
            ],
            "final_scores": self.final_scores,
            "ranking": self.ranking,
            "scoring_metadata": self.scoring_metadata,
            "scoring_time": self.scoring_time
        }


@dataclass
class FeedbackEntry:
    """User feedback entry for learning-to-rank."""
    query: str
    document_id: str
    relevance_score: float  # 0.0 to 1.0
    feedback_type: str  # "click", "dwell", "rating", "save"
    user_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)


class BM25Calculator:
    """Efficient BM25 calculation with document statistics caching."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        Initialize BM25 calculator.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: IDF floor value
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Document statistics cache
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.term_doc_counts: Dict[str, int] = {}
        
        # IDF cache
        self.idf_cache: Dict[str, float] = {}
    
    def update_corpus_stats(self, documents: List[str], doc_ids: List[str] = None):
        """Update corpus statistics for BM25 calculation."""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Reset statistics
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.term_doc_counts.clear()
        self.idf_cache.clear()
        
        # Calculate document statistics
        total_length = 0
        vocab = set()
        
        for doc_id, doc in zip(doc_ids, documents):
            # Tokenize document
            tokens = self._tokenize(doc)
            doc_length = len(tokens)
            
            # Store document length
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
            
            # Count term frequencies
            term_freqs = Counter(tokens)
            self.doc_freqs[doc_id] = dict(term_freqs)
            
            # Update document counts for each term
            for term in set(tokens):
                self.term_doc_counts[term] = self.term_doc_counts.get(term, 0) + 1
                vocab.add(term)
        
        # Calculate average document length
        self.total_docs = len(documents)
        self.avg_doc_length = total_length / max(self.total_docs, 1)
        
        # Precompute IDF values
        for term in vocab:
            self.idf_cache[term] = self._calculate_idf(term)
    
    def score_document(self, query: str, doc_id: str) -> float:
        """Calculate BM25 score for a document given a query."""
        if doc_id not in self.doc_freqs:
            return 0.0
        
        query_terms = self._tokenize(query)
        doc_freqs = self.doc_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        score = 0.0
        for term in query_terms:
            if term in doc_freqs:
                tf = doc_freqs[term]
                idf = self.idf_cache.get(term, 0.0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score += idf * (numerator / denominator)
        
        return score
    
    def score_documents(self, query: str, doc_ids: List[str]) -> List[float]:
        """Score multiple documents for a query."""
        return [self.score_document(query, doc_id) for doc_id in doc_ids]
    
    def _calculate_idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        df = self.term_doc_counts.get(term, 0)
        if df == 0:
            return 0.0
        
        # BM25 IDF with epsilon floor
        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))
        return max(self.epsilon, idf)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - can be enhanced with better tokenizers."""
        import re
        # Convert to lowercase and extract word tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 1]


class LearningToRankManager:
    """Learning-to-rank system for improving relevance scoring."""
    
    def __init__(self, config: ScoringConfig):
        """Initialize learning-to-rank manager."""
        self.config = config
        self.feedback_data: List[FeedbackEntry] = []
        self.model = None
        self.feature_scaler = None
        self.last_training_time: Optional[datetime] = None
        self.feature_importance: Optional[np.ndarray] = None
        
        # Initialize ML components if available
        if RandomForestRegressor is not None:
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        if StandardScaler is not None:
            self.feature_scaler = StandardScaler()
    
    def add_feedback(self, feedback: FeedbackEntry):
        """Add user feedback for learning."""
        self.feedback_data.append(feedback)
        
        # Check if retraining is needed
        if self._should_retrain():
            self._retrain_model()
    
    def predict_relevance(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores using trained model."""
        if self.model is None or not hasattr(self.model, 'predict'):
            return np.zeros(features.shape[0])
        
        try:
            # Scale features if scaler is available
            if self.feature_scaler is not None:
                features_scaled = self.feature_scaler.transform(features)
            else:
                features_scaled = features
            
            return self.model.predict(features_scaled)
        except Exception:
            return np.zeros(features.shape[0])
    
    def extract_features(
        self,
        query_result: QueryResult,
        expansion_result: ExpansionResult,
        context: DocumentContext,
        scores: ScoreComponents
    ) -> np.ndarray:
        """Extract features for learning-to-rank."""
        features = []
        
        # Basic scoring features
        features.extend([
            scores.vector_similarity,
            scores.bm25_score,
            scores.tfidf_score,
            scores.semantic_score
        ])
        
        # Query features
        features.extend([
            len(query_result.processed_query.split()),
            len(query_result.metadata.keywords),
            len(query_result.metadata.entities),
            query_result.metadata.complexity_score
        ])
        
        # Expansion features
        if expansion_result:
            features.extend([
                len(expansion_result.expanded_terms),
                len(expansion_result.detected_phrases),
                expansion_result.expansion_time
            ])
        else:
            features.extend([0, 0, 0])
        
        # Document features
        features.extend([
            len(context.content.split()),
            len(context.content),
            context.similarity_score,
            context.relevance_score,
            len(context.metadata.get('keywords', [])),
            len(context.metadata.get('entities', []))
        ])
        
        # Temporal features
        doc_age = (datetime.now(timezone.utc) - context.retrieval_timestamp).total_seconds()
        features.append(doc_age)
        
        return np.array(features, dtype=np.float32)
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if len(self.feedback_data) < self.config.min_feedback_samples:
            return False
        
        if self.last_training_time is None:
            return True
        
        hours_since_training = (
            datetime.now(timezone.utc) - self.last_training_time
        ).total_seconds() / 3600
        
        return hours_since_training >= self.config.retrain_frequency_hours
    
    def _retrain_model(self):
        """Retrain the ranking model with accumulated feedback."""
        if len(self.feedback_data) < self.config.min_feedback_samples:
            return
        
        try:
            # Prepare training data
            # Note: This is a simplified implementation
            # In practice, you'd need to reconstruct features from feedback
            
            # For now, use a mock training process
            # Real implementation would need to store features with feedback
            self.last_training_time = datetime.now(timezone.utc)
            
        except Exception as e:
            # Log error but don't fail scoring
            pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        if (self.model is None or 
            not hasattr(self.model, 'feature_importances_') or
            self.feature_importance is None):
            return None
        
        feature_names = [
            "vector_similarity", "bm25_score", "tfidf_score", "semantic_score",
            "query_length", "query_keywords", "query_entities", "query_complexity",
            "expanded_terms", "detected_phrases", "expansion_time",
            "doc_word_count", "doc_char_count", "doc_similarity", "doc_relevance",
            "doc_keywords", "doc_entities", "doc_age"
        ]
        
        return dict(zip(feature_names, self.feature_importance))


class RelevanceScorer(LoggerMixin):
    """
    Sophisticated relevance scoring system for RAG search results.
    
    Features:
    - Multiple scoring algorithms (BM25, TF-IDF, semantic similarity)
    - Hybrid scoring with configurable weights
    - Learning-to-rank based on user feedback
    - Score normalization and calibration
    - Performance optimization with caching
    - Contextual and query-dependent scoring
    """
    
    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        custom_scorer: Optional[callable] = None
    ):
        """
        Initialize Relevance Scorer.
        
        Args:
            config: Scoring configuration
            custom_scorer: Custom scoring function
        """
        self.config = config or ScoringConfig()
        self.custom_scorer = custom_scorer
        
        # Initialize scoring components
        self._initialize_components()
        
        # Performance metrics
        self.scoring_stats = {
            "total_scorings": 0,
            "total_scoring_time": 0.0,
            "algorithm_usage": defaultdict(int),
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info("RelevanceScorer initialized successfully")
    
    def _initialize_components(self):
        """Initialize scoring algorithm components."""
        # BM25 calculator
        self.bm25_calculator = BM25Calculator(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            epsilon=self.config.bm25_epsilon
        )
        
        # TF-IDF vectorizer
        if TfidfVectorizer is not None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                norm=self.config.tfidf_norm,
                stop_words='english'
            )
        else:
            self.tfidf_vectorizer = None
        
        # Semantic similarity model
        try:
            self.semantic_model = SentenceTransformer(self.config.semantic_model_name)
        except Exception as e:
            self.logger.warning(f"Could not load semantic model: {e}")
            self.semantic_model = None
        
        # Learning-to-rank manager
        self.learning_manager = LearningToRankManager(self.config)
        
        # Score caches
        self.score_cache: Dict[str, Tuple[float, datetime]] = {}
        self.tfidf_cache: Dict[str, Any] = {}
    
    def score_contexts(
        self,
        query_result: QueryResult,
        retrieval_result: RetrievalResult,
        expansion_result: Optional[ExpansionResult] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """
        Score document contexts for relevance to query.
        
        Args:
            query_result: Preprocessed query information
            retrieval_result: Retrieved document contexts
            expansion_result: Query expansion results
            user_context: User context for personalization
            
        Returns:
            Scoring result with ranked contexts
        """
        start_time = time.time()
        
        try:
            contexts = retrieval_result.contexts
            if not contexts:
                return ScoringResult(
                    scored_contexts=[],
                    final_scores=[],
                    ranking=[],
                    scoring_time=time.time() - start_time
                )
            
            # Prepare documents for batch processing
            documents = [ctx.content for ctx in contexts]
            doc_ids = [str(ctx.id) for ctx in contexts]
            
            # Update BM25 corpus statistics if needed
            if self.config.precompute_document_stats:
                self.bm25_calculator.update_corpus_stats(documents, doc_ids)
            
            # Score each context with multiple algorithms
            scored_contexts = []
            all_scores = []
            
            for i, context in enumerate(contexts):
                scores = self._score_single_context(
                    query_result, context, expansion_result, doc_ids[i]
                )
                scored_contexts.append((context, scores))
                all_scores.append(scores)
            
            # Calculate hybrid scores if enabled
            if self.config.use_hybrid:
                hybrid_scores = self._calculate_hybrid_scores(all_scores, query_result)
                for i, hybrid_score in enumerate(hybrid_scores):
                    scored_contexts[i][1].hybrid_score = hybrid_score
            
            # Apply learning-to-rank if enabled and model is available
            if self.config.use_learning_to_rank:
                learning_scores = self._apply_learning_to_rank(
                    query_result, expansion_result, scored_contexts
                )
                for i, learning_score in enumerate(learning_scores):
                    scored_contexts[i][1].learning_score = learning_score
            
            # Calculate final scores and ranking
            final_scores = self._calculate_final_scores(scored_contexts)
            ranking = self._calculate_ranking(final_scores)
            
            # Apply score normalization if enabled
            if self.config.normalize_final_scores:
                final_scores = self._normalize_scores(
                    final_scores, self.config.normalization_method
                )
            
            scoring_time = time.time() - start_time
            
            # Create result
            result = ScoringResult(
                scored_contexts=scored_contexts,
                final_scores=final_scores,
                ranking=ranking,
                scoring_metadata={
                    "algorithms_used": [alg.value for alg in self.config.primary_algorithms],
                    "hybrid_enabled": self.config.use_hybrid,
                    "learning_enabled": self.config.use_learning_to_rank,
                    "normalization": self.config.normalization_method.value,
                    "total_contexts": len(contexts),
                    "query_complexity": query_result.metadata.complexity_score
                },
                scoring_time=scoring_time
            )
            
            # Update performance metrics
            self._update_scoring_stats(scoring_time, len(contexts))
            
            self.logger.info(
                f"Relevance scoring completed: {len(contexts)} contexts scored "
                f"in {scoring_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            scoring_time = time.time() - start_time
            self.logger.error(f"Relevance scoring failed: {e}")
            raise ScoringError(f"Relevance scoring failed: {e}")
    
    def _score_single_context(
        self,
        query_result: QueryResult,
        context: DocumentContext,
        expansion_result: Optional[ExpansionResult],
        doc_id: str
    ) -> ScoreComponents:
        """Score a single context with all enabled algorithms."""
        scores = ScoreComponents()
        
        # Vector similarity (from existing embeddings)
        if ScoringAlgorithm.VECTOR_SIMILARITY in self.config.primary_algorithms:
            scores.vector_similarity = context.similarity_score
            self.scoring_stats["algorithm_usage"]["vector_similarity"] += 1
        
        # BM25 scoring
        if ScoringAlgorithm.BM25 in self.config.primary_algorithms:
            query_text = expansion_result.expanded_query if expansion_result else query_result.processed_query
            scores.bm25_score = self.bm25_calculator.score_document(query_text, doc_id)
            self.scoring_stats["algorithm_usage"]["bm25"] += 1
        
        # TF-IDF scoring
        if (ScoringAlgorithm.TF_IDF in self.config.primary_algorithms and 
            self.tfidf_vectorizer is not None):
            scores.tfidf_score = self._calculate_tfidf_score(
                query_result, context, expansion_result
            )
            self.scoring_stats["algorithm_usage"]["tfidf"] += 1
        
        # Semantic similarity
        if (ScoringAlgorithm.SEMANTIC in self.config.primary_algorithms and 
            self.semantic_model is not None):
            scores.semantic_score = self._calculate_semantic_score(
                query_result, context, expansion_result
            )
            self.scoring_stats["algorithm_usage"]["semantic"] += 1
        
        # Custom scoring
        if (ScoringAlgorithm.CUSTOM in self.config.primary_algorithms and 
            self.custom_scorer is not None):
            try:
                custom_score = self.custom_scorer(query_result, context, expansion_result)
                scores.custom_scores["custom"] = custom_score
                self.scoring_stats["algorithm_usage"]["custom"] += 1
            except Exception as e:
                self.logger.warning(f"Custom scoring failed: {e}")
        
        # Normalize individual scores if enabled
        if self.config.normalize_individual_scores:
            scores = self._normalize_score_components(scores)
        
        return scores
    
    def _calculate_tfidf_score(
        self,
        query_result: QueryResult,
        context: DocumentContext,
        expansion_result: Optional[ExpansionResult]
    ) -> float:
        """Calculate TF-IDF based similarity score."""
        try:
            # Prepare query text
            query_text = expansion_result.expanded_query if expansion_result else query_result.processed_query
            
            # Create corpus with query and document
            corpus = [query_text, context.content]
            
            # Check cache
            cache_key = f"tfidf:{hash(query_text)}:{hash(context.content)}"
            if cache_key in self.tfidf_cache:
                return self.tfidf_cache[cache_key]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            if cosine_similarity is not None:
                similarity = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:2]
                )[0][0]
            else:
                # Fallback: simple dot product similarity
                query_vec = tfidf_matrix[0].toarray()[0]
                doc_vec = tfidf_matrix[1].toarray()[0]
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
                )
            
            # Cache result
            if self.config.enable_caching:
                self.tfidf_cache[cache_key] = similarity
                self._cleanup_cache(self.tfidf_cache)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"TF-IDF scoring failed: {e}")
            return 0.0
    
    def _calculate_semantic_score(
        self,
        query_result: QueryResult,
        context: DocumentContext,
        expansion_result: Optional[ExpansionResult]
    ) -> float:
        """Calculate semantic similarity score."""
        try:
            # Prepare query text
            query_text = expansion_result.expanded_query if expansion_result else query_result.processed_query
            
            # Check cache
            cache_key = f"semantic:{hash(query_text)}:{hash(context.content)}"
            if cache_key in self.score_cache:
                cached_score, timestamp = self.score_cache[cache_key]
                if (datetime.now(timezone.utc) - timestamp).total_seconds() < self.config.cache_ttl:
                    self.scoring_stats["cache_hits"] += 1
                    return cached_score
            
            # Calculate embeddings
            query_embedding = self.semantic_model.encode([query_text])
            doc_embedding = self.semantic_model.encode([context.content])
            
            # Calculate cosine similarity
            if cosine_similarity is not None:
                similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            else:
                similarity = np.dot(query_embedding[0], doc_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding[0]) + 1e-8
                )
            
            # Cache result
            if self.config.enable_caching:
                self.score_cache[cache_key] = (similarity, datetime.now(timezone.utc))
                self._cleanup_cache(self.score_cache)
            
            self.scoring_stats["cache_misses"] += 1
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Semantic scoring failed: {e}")
            return 0.0
    
    def _calculate_hybrid_scores(
        self,
        all_scores: List[ScoreComponents],
        query_result: QueryResult
    ) -> List[float]:
        """Calculate hybrid scores combining multiple algorithms."""
        hybrid_scores = []
        
        for scores in all_scores:
            # Base weighted combination
            hybrid_score = (
                self.config.vector_weight * scores.vector_similarity +
                self.config.bm25_weight * scores.bm25_score +
                self.config.tfidf_weight * scores.tfidf_score +
                self.config.semantic_weight * scores.semantic_score
            )
            
            # Query-dependent weight adjustment
            if self.config.query_dependent_weights:
                hybrid_score = self._adjust_weights_for_query(
                    hybrid_score, scores, query_result
                )
            
            hybrid_scores.append(hybrid_score)
        
        return hybrid_scores
    
    def _adjust_weights_for_query(
        self,
        base_score: float,
        scores: ScoreComponents,
        query_result: QueryResult
    ) -> float:
        """Adjust scoring weights based on query characteristics."""
        # Simple query type detection and weight adjustment
        query_length = len(query_result.processed_query.split())
        entity_count = len(query_result.metadata.entities)
        
        adjusted_score = base_score
        
        # For short queries, emphasize semantic similarity
        if query_length < 3:
            adjusted_score = (
                0.2 * scores.vector_similarity +
                0.2 * scores.bm25_score +
                0.1 * scores.tfidf_score +
                0.5 * scores.semantic_score
            )
        
        # For entity-rich queries, emphasize vector similarity
        elif entity_count > 2:
            adjusted_score = (
                0.5 * scores.vector_similarity +
                0.2 * scores.bm25_score +
                0.2 * scores.tfidf_score +
                0.1 * scores.semantic_score
            )
        
        # For long queries, emphasize TF-IDF and BM25
        elif query_length > 10:
            adjusted_score = (
                0.1 * scores.vector_similarity +
                0.4 * scores.bm25_score +
                0.4 * scores.tfidf_score +
                0.1 * scores.semantic_score
            )
        
        return adjusted_score
    
    def _apply_learning_to_rank(
        self,
        query_result: QueryResult,
        expansion_result: Optional[ExpansionResult],
        scored_contexts: List[Tuple[DocumentContext, ScoreComponents]]
    ) -> List[float]:
        """Apply learning-to-rank scoring."""
        learning_scores = []
        
        try:
            # Extract features for all contexts
            features_list = []
            for context, scores in scored_contexts:
                features = self.learning_manager.extract_features(
                    query_result, expansion_result, context, scores
                )
                features_list.append(features)
            
            if features_list:
                features_array = np.array(features_list)
                learning_scores = self.learning_manager.predict_relevance(features_array)
            
        except Exception as e:
            self.logger.warning(f"Learning-to-rank scoring failed: {e}")
            learning_scores = [0.0] * len(scored_contexts)
        
        return learning_scores.tolist() if hasattr(learning_scores, 'tolist') else learning_scores
    
    def _calculate_final_scores(
        self,
        scored_contexts: List[Tuple[DocumentContext, ScoreComponents]]
    ) -> List[float]:
        """Calculate final relevance scores."""
        final_scores = []
        
        for context, scores in scored_contexts:
            if self.config.use_learning_to_rank and scores.learning_score > 0:
                # Use learning-to-rank score as primary
                final_score = 0.7 * scores.learning_score + 0.3 * scores.hybrid_score
            elif self.config.use_hybrid:
                # Use hybrid score
                final_score = scores.hybrid_score
            else:
                # Use best individual score
                individual_scores = [
                    scores.vector_similarity,
                    scores.bm25_score,
                    scores.tfidf_score,
                    scores.semantic_score
                ]
                final_score = max(individual_scores)
            
            final_scores.append(final_score)
        
        return final_scores
    
    def _calculate_ranking(self, scores: List[float]) -> List[int]:
        """Calculate ranking indices based on scores."""
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    def _normalize_score_components(self, scores: ScoreComponents) -> ScoreComponents:
        """Normalize individual score components."""
        score_values = [
            scores.vector_similarity,
            scores.bm25_score,
            scores.tfidf_score,
            scores.semantic_score
        ]
        
        normalized_values = self._normalize_scores(
            score_values, self.config.normalization_method
        )
        
        return ScoreComponents(
            vector_similarity=normalized_values[0],
            bm25_score=normalized_values[1],
            tfidf_score=normalized_values[2],
            semantic_score=normalized_values[3],
            hybrid_score=scores.hybrid_score,
            learning_score=scores.learning_score,
            custom_scores=scores.custom_scores
        )
    
    def _normalize_scores(
        self,
        scores: List[float],
        method: ScoreNormalization
    ) -> List[float]:
        """Normalize scores using specified method."""
        if not scores or method == ScoreNormalization.NONE:
            return scores
        
        scores_array = np.array(scores)
        
        if method == ScoreNormalization.MIN_MAX:
            min_val = scores_array.min()
            max_val = scores_array.max()
            if max_val > min_val:
                normalized = (scores_array - min_val) / (max_val - min_val)
            else:
                normalized = scores_array
        
        elif method == ScoreNormalization.Z_SCORE:
            mean_val = scores_array.mean()
            std_val = scores_array.std()
            if std_val > 0:
                normalized = (scores_array - mean_val) / std_val
            else:
                normalized = scores_array
        
        elif method == ScoreNormalization.SOFTMAX:
            exp_scores = np.exp(scores_array - scores_array.max())
            normalized = exp_scores / exp_scores.sum()
        
        elif method == ScoreNormalization.RANK:
            # Convert to rank-based scores
            ranks = np.argsort(np.argsort(scores_array)[::-1])
            normalized = 1.0 - (ranks / len(ranks))
        
        else:
            normalized = scores_array
        
        return normalized.tolist()
    
    def _cleanup_cache(self, cache: Dict[str, Any]):
        """Clean up cache if it exceeds size limit."""
        if len(cache) > self.config.max_cache_size:
            # Remove oldest 10% of entries
            remove_count = len(cache) // 10
            keys_to_remove = list(cache.keys())[:remove_count]
            for key in keys_to_remove:
                del cache[key]
    
    def _update_scoring_stats(self, scoring_time: float, num_contexts: int):
        """Update scoring statistics."""
        self.scoring_stats["total_scorings"] += 1
        self.scoring_stats["total_scoring_time"] += scoring_time
    
    def add_feedback(self, feedback: FeedbackEntry):
        """Add user feedback for improving relevance scoring."""
        self.learning_manager.add_feedback(feedback)
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get scoring statistics and performance metrics."""
        total_scorings = self.scoring_stats["total_scorings"]
        
        return {
            "total_scorings": total_scorings,
            "average_scoring_time": (
                self.scoring_stats["total_scoring_time"] / total_scorings
                if total_scorings > 0 else 0.0
            ),
            "algorithm_usage": dict(self.scoring_stats["algorithm_usage"]),
            "cache_hit_rate": (
                self.scoring_stats["cache_hits"] / 
                (self.scoring_stats["cache_hits"] + self.scoring_stats["cache_misses"])
                if (self.scoring_stats["cache_hits"] + self.scoring_stats["cache_misses"]) > 0
                else 0.0
            ),
            "learning_stats": {
                "feedback_count": len(self.learning_manager.feedback_data),
                "model_trained": self.learning_manager.last_training_time is not None,
                "feature_importance": self.learning_manager.get_feature_importance()
            },
            "config": {
                "primary_algorithms": [alg.value for alg in self.config.primary_algorithms],
                "hybrid_enabled": self.config.use_hybrid,
                "learning_enabled": self.config.use_learning_to_rank,
                "normalization": self.config.normalization_method.value
            }
        }
    
    def clear_caches(self):
        """Clear all scoring caches."""
        self.score_cache.clear()
        self.tfidf_cache.clear()
        self.bm25_calculator.idf_cache.clear()
        self.logger.info("Relevance scoring caches cleared")


def create_relevance_scorer(
    config: Optional[ScoringConfig] = None,
    custom_scorer: Optional[callable] = None
) -> RelevanceScorer:
    """
    Create and configure a RelevanceScorer instance.
    
    Args:
        config: Scoring configuration
        custom_scorer: Custom scoring function
        
    Returns:
        Configured RelevanceScorer
    """
    return RelevanceScorer(
        config=config or ScoringConfig(),
        custom_scorer=custom_scorer
    )


def create_scoring_config(
    algorithms: Optional[List[ScoringAlgorithm]] = None,
    use_hybrid: bool = True,
    use_learning: bool = True,
    **kwargs
) -> ScoringConfig:
    """
    Create scoring configuration with simplified interface.
    
    Args:
        algorithms: List of scoring algorithms to use
        use_hybrid: Enable hybrid scoring
        use_learning: Enable learning-to-rank
        **kwargs: Additional configuration parameters
        
    Returns:
        Scoring configuration
    """
    return ScoringConfig(
        primary_algorithms=algorithms or [
            ScoringAlgorithm.VECTOR_SIMILARITY,
            ScoringAlgorithm.BM25,
            ScoringAlgorithm.SEMANTIC
        ],
        use_hybrid=use_hybrid,
        use_learning_to_rank=use_learning,
        **kwargs
    )


def create_feedback_entry(
    query: str,
    document_id: str,
    relevance_score: float,
    feedback_type: str = "click",
    user_id: Optional[int] = None
) -> FeedbackEntry:
    """
    Create a feedback entry for learning-to-rank.
    
    Args:
        query: Original query
        document_id: Document identifier
        relevance_score: Relevance score (0.0 to 1.0)
        feedback_type: Type of feedback
        user_id: User identifier
        
    Returns:
        Feedback entry
    """
    return FeedbackEntry(
        query=query,
        document_id=document_id,
        relevance_score=relevance_score,
        feedback_type=feedback_type,
        user_id=user_id
    )