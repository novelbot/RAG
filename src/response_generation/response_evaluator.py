"""
Response Quality Evaluation and Selection System for Ensemble Mode.
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.logging import LoggerMixin
from src.response_generation.base import (
    ResponseResult, ResponseRequest, EvaluationMetric, 
    ResponseEvaluationScore, ResponseGeneratorConfig
)
from src.response_generation.exceptions import ResponseEvaluationError


class EvaluationMethod(Enum):
    """Response evaluation methods."""
    HEURISTIC = "heuristic"           # Rule-based heuristic evaluation
    STATISTICAL = "statistical"       # Statistical analysis based
    SEMANTIC = "semantic"             # Semantic similarity based
    COMPOSITE = "composite"           # Combination of multiple methods
    WEIGHTED = "weighted"             # Weighted scoring system


@dataclass
class EvaluationWeights:
    """Weights for different evaluation metrics."""
    relevance: float = 0.3
    accuracy: float = 0.25
    completeness: float = 0.2
    clarity: float = 0.15
    coherence: float = 0.1
    
    def normalize(self) -> 'EvaluationWeights':
        """Normalize weights to sum to 1.0."""
        total = self.relevance + self.accuracy + self.completeness + self.clarity + self.coherence
        if total == 0:
            return self
        
        return EvaluationWeights(
            relevance=self.relevance / total,
            accuracy=self.accuracy / total,
            completeness=self.completeness / total,
            clarity=self.clarity / total,
            coherence=self.coherence / total
        )


@dataclass
class QualityThresholds:
    """Quality thresholds for response filtering."""
    min_relevance: float = 0.6
    min_accuracy: float = 0.5
    min_completeness: float = 0.4
    min_clarity: float = 0.5
    min_coherence: float = 0.5
    min_overall: float = 0.6


class ResponseEvaluator(LoggerMixin):
    """
    Advanced Response Quality Evaluation and Selection System.
    
    Features:
    - Multiple evaluation metrics (relevance, accuracy, completeness, etc.)
    - Configurable scoring algorithms and weights
    - Ensemble response comparison and ranking
    - Quality threshold filtering
    - Confidence scoring and uncertainty estimation
    - Best response selection strategies
    """
    
    def __init__(self, config: ResponseGeneratorConfig):
        """
        Initialize Response Evaluator.
        
        Args:
            config: Response generator configuration
        """
        self.config = config
        
        # Initialize evaluation weights
        self.weights = EvaluationWeights().normalize()
        
        # Initialize quality thresholds
        self.thresholds = QualityThresholds(
            min_overall=config.min_quality_score
        )
        
        # Evaluation method
        self.evaluation_method = EvaluationMethod.COMPOSITE
        
        self.logger.info("ResponseEvaluator initialized successfully")
    
    def evaluate_response(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> ResponseResult:
        """
        Evaluate a single response and add quality scores.
        
        Args:
            response: Response to evaluate
            request: Original request for context
            
        Returns:
            Response with updated quality scores
        """
        try:
            # Clear existing quality scores
            response.quality_scores = []
            
            # Evaluate each metric
            for metric in self.config.evaluation_metrics:
                score, confidence, explanation = self._evaluate_metric(
                    response, request, metric
                )
                
                response.add_quality_score(
                    metric, score, confidence, explanation
                )
            
            # Calculate overall quality score
            response.calculate_overall_quality()
            
            # Update confidence score
            response.confidence_score = self._calculate_confidence_score(response)
            
            self.logger.debug(f"Evaluated response with overall quality: {response.overall_quality_score:.3f}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response evaluation failed: {e}")
            raise ResponseEvaluationError(f"Response evaluation failed: {e}")
    
    def evaluate_ensemble_responses(
        self,
        responses: List[ResponseResult],
        request: ResponseRequest
    ) -> List[ResponseResult]:
        """
        Evaluate multiple ensemble responses.
        
        Args:
            responses: List of responses to evaluate
            request: Original request for context
            
        Returns:
            List of responses with updated quality scores
        """
        evaluated_responses = []
        
        for response in responses:
            try:
                evaluated_response = self.evaluate_response(response, request)
                evaluated_responses.append(evaluated_response)
            except Exception as e:
                self.logger.warning(f"Failed to evaluate response: {e}")
                # Keep original response if evaluation fails
                evaluated_responses.append(response)
        
        # Apply comparative evaluation
        if len(evaluated_responses) > 1:
            evaluated_responses = self._apply_comparative_evaluation(
                evaluated_responses, request
            )
        
        return evaluated_responses
    
    def select_best_response(
        self,
        responses: List[ResponseResult],
        request: ResponseRequest
    ) -> Tuple[ResponseResult, float, str]:
        """
        Select the best response from evaluated ensemble responses.
        
        Args:
            responses: List of evaluated responses
            request: Original request for context
            
        Returns:
            Tuple of (best_response, consensus_score, selection_method)
        """
        if not responses:
            raise ResponseEvaluationError("No responses provided for selection")
        
        if len(responses) == 1:
            return responses[0], 1.0, "single_response"
        
        # Filter responses by quality threshold
        qualified_responses = self._filter_by_quality_threshold(responses)
        
        if not qualified_responses:
            self.logger.warning("No responses meet quality threshold, using best available")
            qualified_responses = responses
        
        # Select best response using configured method
        best_response = self._select_by_weighted_score(qualified_responses)
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(responses)
        
        selection_method = "weighted_composite"
        
        self.logger.info(
            f"Selected best response with quality {best_response.overall_quality_score:.3f}, "
            f"consensus {consensus_score:.3f}"
        )
        
        return best_response, consensus_score, selection_method
    
    def _evaluate_metric(
        self,
        response: ResponseResult,
        request: ResponseRequest,
        metric: EvaluationMetric
    ) -> Tuple[float, float, str]:
        """
        Evaluate a specific metric for a response.
        
        Returns:
            Tuple of (score, confidence, explanation)
        """
        if metric == EvaluationMetric.RELEVANCE:
            return self._evaluate_relevance(response, request)
        elif metric == EvaluationMetric.ACCURACY:
            return self._evaluate_accuracy(response, request)
        elif metric == EvaluationMetric.COMPLETENESS:
            return self._evaluate_completeness(response, request)
        elif metric == EvaluationMetric.CLARITY:
            return self._evaluate_clarity(response, request)
        elif metric == EvaluationMetric.COHERENCE:
            return self._evaluate_coherence(response, request)
        elif metric == EvaluationMetric.CONFIDENCE:
            return self._evaluate_model_confidence(response, request)
        elif metric == EvaluationMetric.LENGTH:
            return self._evaluate_length_appropriateness(response, request)
        else:
            return 0.5, 0.3, f"Unknown metric: {metric}"
    
    def _evaluate_relevance(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate response relevance to the query."""
        
        query_terms = set(request.query.lower().split())
        response_terms = set(response.response.lower().split())
        
        if not query_terms:
            return 0.5, 0.3, "Empty query"
        
        # Calculate term overlap
        overlap = len(query_terms & response_terms)
        overlap_ratio = overlap / len(query_terms)
        
        # Check for direct question answering patterns
        question_patterns = [
            r'\?',  # Contains question mark
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b'
        ]
        
        is_question = any(re.search(pattern, request.query, re.IGNORECASE) for pattern in question_patterns)
        
        # Boost score if response directly addresses question
        if is_question:
            answer_indicators = [
                r'\bthe answer is\b', r'\bis\b', r'\bare\b', r'\bbecause\b',
                r'\bby\b', r'\bthrough\b', r'\bwhen\b', r'\bwhere\b'
            ]
            has_answer_pattern = any(
                re.search(pattern, response.response, re.IGNORECASE) 
                for pattern in answer_indicators
            )
            
            if has_answer_pattern:
                overlap_ratio = min(1.0, overlap_ratio + 0.2)
        
        # Consider context relevance if available
        if request.context:
            context_terms = set(request.context.lower().split())
            context_overlap = len(response_terms & context_terms)
            if context_terms:
                context_relevance = context_overlap / len(context_terms)
                overlap_ratio = (overlap_ratio + context_relevance * 0.3) / 1.3
        
        score = min(1.0, overlap_ratio)
        confidence = 0.7
        explanation = f"Term overlap: {overlap}/{len(query_terms)} ({overlap_ratio:.2f})"
        
        return score, confidence, explanation
    
    def _evaluate_accuracy(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate response factual accuracy (heuristic-based)."""
        
        # Check for uncertainty indicators (good for accuracy)
        uncertainty_patterns = [
            r'\bmight\b', r'\bcould\b', r'\bmay\b', r'\bpossibly\b',
            r'\bperhaps\b', r'\blikely\b', r'\bprobably\b'
        ]
        
        uncertainty_count = sum(
            len(re.findall(pattern, response.response, re.IGNORECASE))
            for pattern in uncertainty_patterns
        )
        
        # Check for confidence indicators
        confidence_patterns = [
            r'\bdefinitely\b', r'\bcertainly\b', r'\balways\b', r'\bnever\b',
            r'\bguarantee\b', r'\bwithout doubt\b'
        ]
        
        confidence_count = sum(
            len(re.findall(pattern, response.response, re.IGNORECASE))
            for pattern in confidence_patterns
        )
        
        # Check for citations or source references
        citation_patterns = [
            r'\baccording to\b', r'\bsource\b', r'\breference\b',
            r'\bstudy\b', r'\bresearch\b', r'\bdata\b'
        ]
        
        citation_count = sum(
            len(re.findall(pattern, response.response, re.IGNORECASE))
            for pattern in citation_patterns
        )
        
        # Calculate accuracy score
        base_score = 0.7  # Default neutral score
        
        # Moderate uncertainty is good (shows awareness of limitations)
        if 1 <= uncertainty_count <= 3:
            base_score += 0.1
        elif uncertainty_count > 5:
            base_score -= 0.1
        
        # Too much overconfidence is bad
        if confidence_count > 3:
            base_score -= 0.15
        
        # Citations boost accuracy
        if citation_count > 0:
            base_score += min(0.2, citation_count * 0.05)
        
        score = max(0.0, min(1.0, base_score))
        confidence = 0.5  # Moderate confidence for heuristic method
        explanation = f"Uncertainty: {uncertainty_count}, Confidence: {confidence_count}, Citations: {citation_count}"
        
        return score, confidence, explanation
    
    def _evaluate_completeness(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate response completeness."""
        
        # Check response length relative to query complexity
        query_length = len(request.query)
        response_length = len(response.response)
        
        # Basic length appropriateness
        if query_length < 50:  # Simple query
            target_length = 100
        elif query_length < 200:  # Medium query
            target_length = 300
        else:  # Complex query
            target_length = 500
        
        length_score = min(1.0, response_length / target_length)
        
        # Check for comprehensive elements
        comprehensive_indicators = [
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\bfinally\b',
            r'\bin addition\b', r'\bfurthermore\b', r'\bmoreover\b',
            r'\balso\b', r'\badditionally\b', r'\bfor example\b'
        ]
        
        structure_count = sum(
            len(re.findall(pattern, response.response, re.IGNORECASE))
            for pattern in comprehensive_indicators
        )
        
        structure_score = min(0.3, structure_count * 0.1)
        
        # Check if response addresses multiple aspects
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        query_aspects = sum(1 for word in question_words if word in request.query.lower())
        
        if query_aspects > 1:
            # Multi-aspect query should have multi-aspect response
            aspect_score = min(0.2, structure_count * 0.05)
        else:
            aspect_score = 0.1  # Bonus for any structure
        
        score = min(1.0, length_score + structure_score + aspect_score)
        confidence = 0.6
        explanation = f"Length ratio: {length_score:.2f}, Structure indicators: {structure_count}"
        
        return score, confidence, explanation
    
    def _evaluate_clarity(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate response clarity and readability."""
        
        text = response.response
        
        # Calculate basic readability metrics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.1, 0.8, "Empty response"
        
        words = text.split()
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Sentence length score (ideal: 15-25 words)
        if 15 <= avg_sentence_length <= 25:
            sentence_score = 1.0
        elif 10 <= avg_sentence_length <= 30:
            sentence_score = 0.8
        else:
            sentence_score = 0.6
        
        # Check for clarity indicators
        clarity_indicators = [
            r'\bfor example\b', r'\bspecifically\b', r'\bin other words\b',
            r'\bthat is\b', r'\bnamely\b', r'\bto clarify\b'
        ]
        
        clarity_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in clarity_indicators
        )
        
        clarity_bonus = min(0.2, clarity_count * 0.1)
        
        # Check for jargon/complexity (negative for clarity)
        complex_patterns = [
            r'\b\w{12,}\b',  # Very long words
            r'\betc\.\b', r'\bi\.e\.\b', r'\be\.g\.\b'  # Abbreviations
        ]
        
        complexity_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in complex_patterns
        )
        
        complexity_penalty = min(0.2, complexity_count * 0.02)
        
        score = max(0.0, min(1.0, sentence_score + clarity_bonus - complexity_penalty))
        confidence = 0.7
        explanation = f"Avg sentence length: {avg_sentence_length:.1f}, Clarity indicators: {clarity_count}"
        
        return score, confidence, explanation
    
    def _evaluate_coherence(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate response internal coherence."""
        
        text = response.response
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.8, 0.5, "Single sentence response"
        
        # Check for transition words (good for coherence)
        transition_patterns = [
            r'\bhowever\b', r'\btherefore\b', r'\bmoreover\b', r'\bfurthermore\b',
            r'\bconsequently\b', r'\bin contrast\b', r'\bsimilarly\b', r'\bbut\b',
            r'\band\b', r'\bso\b', r'\bthus\b', r'\bhence\b'
        ]
        
        transition_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in transition_patterns
        )
        
        transition_score = min(0.4, transition_count * 0.1)
        
        # Check for contradiction indicators (bad for coherence)
        contradiction_patterns = [
            r'\bbut\s+(?:also|conversely)\b', r'\bhowever.*however\b',
            r'\balthough.*but\b', r'\bdespite.*however\b'
        ]
        
        contradiction_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in contradiction_patterns
        )
        
        contradiction_penalty = contradiction_count * 0.2
        
        # Check for logical flow indicators
        flow_patterns = [
            r'\bfirst\b', r'\bsecond\b', r'\bfinally\b', r'\bin conclusion\b',
            r'\bto summarize\b', r'\boverall\b'
        ]
        
        flow_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in flow_patterns
        )
        
        flow_score = min(0.3, flow_count * 0.15)
        
        base_score = 0.7  # Default coherence assumption
        score = max(0.0, min(1.0, base_score + transition_score + flow_score - contradiction_penalty))
        confidence = 0.6
        explanation = f"Transitions: {transition_count}, Flow indicators: {flow_count}, Contradictions: {contradiction_count}"
        
        return score, confidence, explanation
    
    def _evaluate_model_confidence(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate model confidence from metadata."""
        
        # Extract confidence from LLM response metadata
        llm_confidence = response.llm_response.metadata.get('confidence', 0.7)
        
        # Consider response characteristics that indicate confidence
        hedging_patterns = [
            r'\bi think\b', r'\bi believe\b', r'\bit seems\b', r'\bapparently\b',
            r'\bpossibly\b', r'\bmaybe\b', r'\bperhaps\b'
        ]
        
        hedging_count = sum(
            len(re.findall(pattern, response.response, re.IGNORECASE))
            for pattern in hedging_patterns
        )
        
        # More hedging typically indicates lower confidence
        hedging_penalty = min(0.3, hedging_count * 0.1)
        
        score = max(0.0, min(1.0, llm_confidence - hedging_penalty))
        confidence = 0.8
        explanation = f"LLM confidence: {llm_confidence}, Hedging expressions: {hedging_count}"
        
        return score, confidence, explanation
    
    def _evaluate_length_appropriateness(
        self,
        response: ResponseResult,
        request: ResponseRequest
    ) -> Tuple[float, float, str]:
        """Evaluate if response length is appropriate."""
        
        response_length = len(response.response)
        
        # Determine appropriate length based on query
        query_length = len(request.query)
        
        if query_length < 50:  # Short query
            ideal_min, ideal_max = 50, 300
        elif query_length < 200:  # Medium query
            ideal_min, ideal_max = 100, 600
        else:  # Long/complex query
            ideal_min, ideal_max = 200, 1000
        
        if ideal_min <= response_length <= ideal_max:
            score = 1.0
        elif response_length < ideal_min:
            score = response_length / ideal_min
        else:  # Too long
            score = max(0.3, ideal_max / response_length)
        
        score = max(0.0, min(1.0, score))
        confidence = 0.8
        explanation = f"Length: {response_length} chars (ideal: {ideal_min}-{ideal_max})"
        
        return score, confidence, explanation
    
    def _calculate_confidence_score(self, response: ResponseResult) -> float:
        """Calculate overall confidence score for response."""
        
        if not response.quality_scores:
            return 0.5
        
        # Weight confidence by the confidence of individual scores
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for score in response.quality_scores:
            weight = score.confidence
            weighted_confidence += score.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_confidence / total_weight
    
    def _apply_comparative_evaluation(
        self,
        responses: List[ResponseResult],
        request: ResponseRequest
    ) -> List[ResponseResult]:
        """Apply comparative evaluation across ensemble responses."""
        
        # Calculate relative scores for each metric
        for metric in self.config.evaluation_metrics:
            metric_scores = []
            for response in responses:
                score = response.get_quality_score(metric)
                if score is not None:
                    metric_scores.append(score)
            
            if not metric_scores:
                continue
            
            # Apply normalization to spread scores
            min_score = min(metric_scores)
            max_score = max(metric_scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for i, response in enumerate(responses):
                    current_score = response.get_quality_score(metric)
                    if current_score is not None:
                        # Enhance score differences
                        normalized_score = (current_score - min_score) / score_range
                        enhanced_score = min(1.0, normalized_score * 1.2)
                        
                        # Update the score
                        for score_obj in response.quality_scores:
                            if score_obj.metric == metric:
                                score_obj.score = enhanced_score
                                break
        
        # Recalculate overall scores
        for response in responses:
            response.calculate_overall_quality()
        
        return responses
    
    def _filter_by_quality_threshold(self, responses: List[ResponseResult]) -> List[ResponseResult]:
        """Filter responses by quality thresholds."""
        
        qualified_responses = []
        
        for response in responses:
            meets_threshold = True
            
            # Check overall threshold
            if response.overall_quality_score < self.thresholds.min_overall:
                meets_threshold = False
            
            # Check individual metric thresholds
            for score in response.quality_scores:
                metric_threshold = getattr(
                    self.thresholds, 
                    f"min_{score.metric.value}", 
                    self.thresholds.min_overall
                )
                
                if score.score < metric_threshold:
                    meets_threshold = False
                    break
            
            if meets_threshold:
                qualified_responses.append(response)
        
        return qualified_responses
    
    def _select_by_weighted_score(self, responses: List[ResponseResult]) -> ResponseResult:
        """Select response with highest weighted score."""
        
        best_response = None
        best_weighted_score = -1.0
        
        for response in responses:
            weighted_score = 0.0
            total_weight = 0.0
            
            for score in response.quality_scores:
                weight = getattr(self.weights, score.metric.value, 0.1)
                weighted_score += score.score * weight
                total_weight += weight
            
            if total_weight > 0:
                normalized_score = weighted_score / total_weight
            else:
                normalized_score = response.overall_quality_score
            
            if normalized_score > best_weighted_score:
                best_weighted_score = normalized_score
                best_response = response
        
        return best_response or responses[0]
    
    def _calculate_consensus_score(self, responses: List[ResponseResult]) -> float:
        """Calculate consensus score among responses."""
        
        if len(responses) < 2:
            return 1.0
        
        # Calculate average similarity between all pairs
        similarities = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_response_similarity(
                    responses[i].response,
                    responses[j].response
                )
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        return sum(similarities) / len(similarities)
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses."""
        
        # Simple word-based similarity
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def set_evaluation_weights(self, weights: EvaluationWeights) -> None:
        """Set custom evaluation weights."""
        self.weights = weights.normalize()
        self.logger.info("Updated evaluation weights")
    
    def set_quality_thresholds(self, thresholds: QualityThresholds) -> None:
        """Set custom quality thresholds."""
        self.thresholds = thresholds
        self.logger.info("Updated quality thresholds")
    
    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get evaluator configuration and statistics."""
        return {
            "evaluation_method": self.evaluation_method.value,
            "weights": {
                "relevance": self.weights.relevance,
                "accuracy": self.weights.accuracy,
                "completeness": self.weights.completeness,
                "clarity": self.weights.clarity,
                "coherence": self.weights.coherence
            },
            "thresholds": {
                "min_relevance": self.thresholds.min_relevance,
                "min_accuracy": self.thresholds.min_accuracy,
                "min_completeness": self.thresholds.min_completeness,
                "min_clarity": self.thresholds.min_clarity,
                "min_coherence": self.thresholds.min_coherence,
                "min_overall": self.thresholds.min_overall
            },
            "configured_metrics": [metric.value for metric in self.config.evaluation_metrics]
        }