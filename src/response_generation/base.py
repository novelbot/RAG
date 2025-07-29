"""
Base classes and data structures for Response Generation System.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.llm.base import LLMRequest, LLMResponse, LLMMessage
from src.rag.context_retriever import RetrievalResult


class ResponseMode(Enum):
    """Response generation modes."""
    SINGLE = "single"                    # Single LLM for fast response
    ENSEMBLE = "ensemble"                # Multiple LLMs for quality
    ADAPTIVE = "adaptive"                # Switch based on context/query
    HYBRID = "hybrid"                    # Combine single and ensemble


class ResponseQuality(Enum):
    """Response quality levels."""
    FAST = "fast"                        # Prioritize speed
    BALANCED = "balanced"                # Balance speed and quality
    HIGH = "high"                        # Prioritize quality
    PREMIUM = "premium"                  # Maximum quality


class EvaluationMetric(Enum):
    """Response evaluation metrics."""
    RELEVANCE = "relevance"              # How relevant to the query
    ACCURACY = "accuracy"                # Factual accuracy
    COMPLETENESS = "completeness"        # How complete the answer is
    CLARITY = "clarity"                  # How clear and understandable
    COHERENCE = "coherence"              # Internal consistency
    CONFIDENCE = "confidence"            # Model confidence score
    LENGTH = "length"                    # Response length appropriateness


@dataclass
class ResponseGeneratorConfig:
    """Configuration for response generation system."""
    
    # Mode settings
    response_mode: ResponseMode = ResponseMode.SINGLE
    quality_level: ResponseQuality = ResponseQuality.BALANCED
    
    # Single LLM settings
    preferred_provider: Optional[str] = None
    fallback_providers: List[str] = field(default_factory=list)
    single_timeout: float = 30.0
    
    # Ensemble settings
    ensemble_size: int = 3
    ensemble_timeout: float = 60.0
    enable_parallel_generation: bool = True
    consensus_threshold: float = 0.7
    
    # Prompt engineering
    max_context_length: int = 8000
    context_compression_ratio: float = 0.8
    enable_context_ranking: bool = True
    system_prompt_template: str = "You are a helpful AI assistant. Provide accurate and helpful responses based on the given context."
    
    # Response evaluation
    enable_quality_filtering: bool = True
    min_quality_score: float = 0.6
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.RELEVANCE,
        EvaluationMetric.ACCURACY,
        EvaluationMetric.CLARITY
    ])
    
    # Response processing
    enable_post_processing: bool = True
    enable_response_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_graceful_degradation: bool = True
    
    # Performance
    enable_streaming: bool = False
    chunk_size: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "response_mode": self.response_mode.value,
            "quality_level": self.quality_level.value,
            "preferred_provider": self.preferred_provider,
            "fallback_providers": self.fallback_providers,
            "single_timeout": self.single_timeout,
            "ensemble_size": self.ensemble_size,
            "ensemble_timeout": self.ensemble_timeout,
            "enable_parallel_generation": self.enable_parallel_generation,
            "consensus_threshold": self.consensus_threshold,
            "max_context_length": self.max_context_length,
            "context_compression_ratio": self.context_compression_ratio,
            "enable_context_ranking": self.enable_context_ranking,
            "system_prompt_template": self.system_prompt_template,
            "enable_quality_filtering": self.enable_quality_filtering,
            "min_quality_score": self.min_quality_score,
            "evaluation_metrics": [metric.value for metric in self.evaluation_metrics],
            "enable_post_processing": self.enable_post_processing,
            "enable_response_caching": self.enable_response_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "enable_graceful_degradation": self.enable_graceful_degradation,
            "enable_streaming": self.enable_streaming,
            "chunk_size": self.chunk_size
        }


@dataclass
class ResponseRequest:
    """Request for response generation."""
    query: str
    context: Optional[str] = None
    retrieval_result: Optional[RetrievalResult] = None
    user_context: Optional[Dict[str, Any]] = None
    conversation_history: List[LLMMessage] = field(default_factory=list)
    
    # Generation parameters
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    
    # Custom instructions
    system_prompt: Optional[str] = None
    custom_instructions: Optional[str] = None
    response_format: Optional[str] = None
    
    # Metadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "query": self.query,
            "context": self.context,
            "user_context": self.user_context,
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "system_prompt": self.system_prompt,
            "custom_instructions": self.custom_instructions,
            "response_format": self.response_format,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ResponseEvaluationScore:
    """Evaluation score for a response."""
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    confidence: float = 0.0
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert score to dictionary."""
        return {
            "metric": self.metric.value,
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation
        }


@dataclass
class ResponseResult:
    """Result of response generation."""
    response: str
    llm_response: LLMResponse
    generation_time: float
    
    # Quality and evaluation
    quality_scores: List[ResponseEvaluationScore] = field(default_factory=list)
    overall_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    # Generation metadata
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    
    # Processing metadata
    processing_steps: List[str] = field(default_factory=list)
    context_length: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Error information
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Timestamps
    request_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_quality_score(self, metric: EvaluationMetric) -> Optional[float]:
        """Get score for specific evaluation metric."""
        for score in self.quality_scores:
            if score.metric == metric:
                return score.score
        return None
    
    def add_quality_score(self, metric: EvaluationMetric, score: float, confidence: float = 0.0, explanation: Optional[str] = None):
        """Add a quality score."""
        eval_score = ResponseEvaluationScore(
            metric=metric,
            score=score,
            confidence=confidence,
            explanation=explanation
        )
        self.quality_scores.append(eval_score)
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall quality score from individual metrics."""
        if not self.quality_scores:
            return 0.0
        
        total_score = sum(score.score for score in self.quality_scores)
        self.overall_quality_score = total_score / len(self.quality_scores)
        return self.overall_quality_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "response": self.response,
            "llm_response": self.llm_response.to_dict(),
            "generation_time": self.generation_time,
            "quality_scores": [score.to_dict() for score in self.quality_scores],
            "overall_quality_score": self.overall_quality_score,
            "confidence_score": self.confidence_score,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
            "processing_steps": self.processing_steps,
            "context_length": self.context_length,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "warnings": self.warnings,
            "errors": self.errors,
            "request_timestamp": self.request_timestamp.isoformat(),
            "response_timestamp": self.response_timestamp.isoformat()
        }


@dataclass
class EnsembleResult:
    """Result of ensemble response generation."""
    best_response: ResponseResult
    all_responses: List[ResponseResult]
    consensus_score: float
    selection_method: str
    ensemble_time: float
    
    # Ensemble metadata
    providers_used: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    total_tokens_used: int = 0
    total_cost_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ensemble result to dictionary."""
        return {
            "best_response": self.best_response.to_dict(),
            "all_responses": [response.to_dict() for response in self.all_responses],
            "consensus_score": self.consensus_score,
            "selection_method": self.selection_method,
            "ensemble_time": self.ensemble_time,
            "providers_used": self.providers_used,
            "models_used": self.models_used,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_estimate": self.total_cost_estimate
        }


class BaseResponseGenerator(ABC):
    """Abstract base class for response generators."""
    
    def __init__(self, config: ResponseGeneratorConfig):
        """Initialize generator with configuration."""
        self.config = config
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0
        }
    
    @abstractmethod
    async def generate_response_async(self, request: ResponseRequest) -> ResponseResult:
        """Generate response asynchronously."""
        pass
    
    @abstractmethod
    def generate_response(self, request: ResponseRequest) -> ResponseResult:
        """Generate response synchronously."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return self._stats.copy()
    
    def _update_stats_success(self, generation_time: float):
        """Update statistics for successful generation."""
        self._stats["total_requests"] += 1
        self._stats["successful_requests"] += 1
        self._stats["total_generation_time"] += generation_time
        self._stats["average_generation_time"] = (
            self._stats["total_generation_time"] / self._stats["successful_requests"]
        )
    
    def _update_stats_failure(self):
        """Update statistics for failed generation."""
        self._stats["total_requests"] += 1
        self._stats["failed_requests"] += 1