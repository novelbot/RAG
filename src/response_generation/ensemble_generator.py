"""
Multi-LLM Ensemble Response Generator - High-quality response generation using multiple LLMs.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm.manager import LLMManager
from src.llm.base import LLMRequest, LLMResponse
from src.core.logging import LoggerMixin
from src.response_generation.base import (
    BaseResponseGenerator, ResponseRequest, ResponseResult, EnsembleResult,
    ResponseGeneratorConfig, EvaluationMetric
)
from src.response_generation.single_generator import SingleLLMGenerator
from src.response_generation.exceptions import (
    ResponseGenerationError, EnsembleError, TimeoutError, ProviderUnavailableError
)
from src.response_generation.error_handler import ErrorHandler


class EnsembleLLMGenerator(BaseResponseGenerator, LoggerMixin):
    """
    Multi-LLM Ensemble Response Generator for high-quality responses.
    
    Features:
    - Multiple LLM coordination and parallel requests
    - Response collection and aggregation
    - Load balancing and provider availability checks
    - Consensus scoring and best response selection
    - Performance monitoring and optimization
    """
    
    def __init__(self, llm_manager: LLMManager, config: ResponseGeneratorConfig):
        """
        Initialize Ensemble LLM Generator.
        
        Args:
            llm_manager: LLM manager instance for provider access
            config: Response generator configuration
        """
        super().__init__(config)
        self.llm_manager = llm_manager
        self.single_generator = SingleLLMGenerator(llm_manager, config)
        
        # Validate configuration
        if config.ensemble_size < 2:
            raise ValueError("Ensemble size must be at least 2")
        if config.ensemble_timeout <= 0:
            raise ValueError("Ensemble timeout must be positive")
        if not (0.0 <= config.consensus_threshold <= 1.0):
            raise ValueError("Consensus threshold must be between 0.0 and 1.0")
        
        # Initialize error handler for comprehensive error management
        self.error_handler = ErrorHandler(config)
        
        self.logger.info(f"EnsembleLLMGenerator initialized with ensemble size: {config.ensemble_size} and error handler")
    
    async def generate_response_async(self, request: ResponseRequest) -> EnsembleResult:
        """
        Generate ensemble response asynchronously using multiple LLMs with comprehensive error handling.
        
        Args:
            request: Response generation request
            
        Returns:
            Ensemble response result with best response selected
        """
        start_time = time.time()
        retry_count = 0
        max_retries = 2  # Ensemble retries are more expensive
        
        while retry_count <= max_retries:
            try:
                # Select providers for ensemble
                providers = self._select_ensemble_providers()
                
                if len(providers) < 2:
                    self.logger.warning("Insufficient providers for ensemble, falling back to single LLM")
                    single_result = await self.single_generator.generate_response_async(request)
                    return self._create_single_ensemble_result(single_result, time.time() - start_time)
                
                # Get adaptive timeout for ensemble operation
                ensemble_timeout = self.error_handler.get_timeout(
                    operation_type="ensemble",
                    provider=None,  # Multiple providers
                    complexity=self.single_generator._estimate_request_complexity(request)
                )
                
                # Generate responses with enhanced error handling
                if self.config.enable_parallel_generation:
                    responses = await self._generate_parallel_async_with_error_handling(
                        request, providers, ensemble_timeout
                    )
                else:
                    responses = await self._generate_sequential_async_with_error_handling(
                        request, providers, ensemble_timeout
                    )
                
                # Filter successful responses
                successful_responses = [r for r in responses if r is not None and not r.errors]
                
                if not successful_responses:
                    raise EnsembleError("All ensemble providers failed to generate responses")
                
                # Select best response
                best_response, consensus_score, selection_method = self._select_best_response(
                    successful_responses, request
                )
                
                # Create ensemble result
                ensemble_result = EnsembleResult(
                    best_response=best_response,
                    all_responses=successful_responses,
                    consensus_score=consensus_score,
                    selection_method=selection_method,
                    ensemble_time=time.time() - start_time,
                    providers_used=[r.provider_used for r in successful_responses if r.provider_used],
                    models_used=[r.model_used for r in successful_responses if r.model_used],
                    total_tokens_used=sum(r.tokens_used or 0 for r in successful_responses),
                    total_cost_estimate=sum(r.cost_estimate or 0.0 for r in successful_responses)
                )
                
                # Update statistics
                self._update_stats_success(ensemble_result.ensemble_time)
                
                self.logger.info(
                    f"Generated ensemble response in {ensemble_result.ensemble_time:.3f}s "
                    f"with {len(successful_responses)} providers, consensus: {consensus_score:.3f} "
                    f"(attempt {retry_count + 1}, timeout: {ensemble_timeout:.1f}s)"
                )
                
                return ensemble_result
                
            except Exception as e:
                # Handle error using comprehensive error handler
                error_context = {
                    "provider": "ensemble",
                    "model": "multiple",
                    "operation_type": "ensemble",
                    "retry_count": retry_count,
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "ensemble_size": self.config.ensemble_size
                }
                
                # Get error handling decision
                decision = self.error_handler.handle_error(e, error_context, retry_count)
                
                self.logger.warning(
                    f"Ensemble error on attempt {retry_count + 1}: {e} "
                    f"(decision: {decision['action']}, delay: {decision['delay']}s)"
                )
                
                if decision["action"] == "retry" and retry_count < max_retries:
                    retry_count += 1
                    if decision["delay"] > 0:
                        await asyncio.sleep(decision["delay"])
                    continue
                    
                elif decision["action"] == "fallback" and self.config.enable_graceful_degradation:
                    # Fallback to single LLM
                    try:
                        single_result = await self.single_generator.generate_response_async(request)
                        return self._create_single_ensemble_result(single_result, time.time() - start_time)
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback to single LLM also failed: {fallback_error}")
                        raise EnsembleError(f"Both ensemble and fallback failed: {e}")
                        
                else:
                    # Update statistics and re-raise
                    self._update_stats_failure()
                    self.logger.error(f"Ensemble generation failed after {retry_count + 1} attempts: {e}")
                    raise EnsembleError(f"Ensemble response generation failed: {e}")
        
        # Should not reach here, but just in case
        self._update_stats_failure()
        raise EnsembleError("Maximum ensemble retry attempts exceeded")
    
    def generate_response(self, request: ResponseRequest) -> EnsembleResult:
        """
        Generate ensemble response synchronously using multiple LLMs.
        
        Args:
            request: Response generation request
            
        Returns:
            Ensemble response result with best response selected
        """
        start_time = time.time()
        
        try:
            # Select providers for ensemble
            providers = self._select_ensemble_providers()
            
            if len(providers) < 2:
                self.logger.warning("Insufficient providers for ensemble, falling back to single LLM")
                single_result = self.single_generator.generate_response(request)
                return self._create_single_ensemble_result(single_result, time.time() - start_time)
            
            # Generate responses
            if self.config.enable_parallel_generation:
                responses = self._generate_parallel_sync(request, providers)
            else:
                responses = self._generate_sequential_sync(request, providers)
            
            # Filter successful responses
            successful_responses = [r for r in responses if r is not None and not r.errors]
            
            if not successful_responses:
                raise EnsembleError("All ensemble providers failed to generate responses")
            
            # Select best response
            best_response, consensus_score, selection_method = self._select_best_response(
                successful_responses, request
            )
            
            # Create ensemble result
            ensemble_result = EnsembleResult(
                best_response=best_response,
                all_responses=successful_responses,
                consensus_score=consensus_score,
                selection_method=selection_method,
                ensemble_time=time.time() - start_time,
                providers_used=[r.provider_used for r in successful_responses if r.provider_used],
                models_used=[r.model_used for r in successful_responses if r.model_used],
                total_tokens_used=sum(r.tokens_used or 0 for r in successful_responses),
                total_cost_estimate=sum(r.cost_estimate or 0.0 for r in successful_responses)
            )
            
            # Update statistics
            self._update_stats_success(ensemble_result.ensemble_time)
            
            self.logger.info(
                f"Generated ensemble response in {ensemble_result.ensemble_time:.3f}s "
                f"with {len(successful_responses)} providers, consensus: {consensus_score:.3f}"
            )
            
            return ensemble_result
            
        except Exception as e:
            self._update_stats_failure()
            self.logger.error(f"Ensemble response generation failed: {e}")
            
            if self.config.enable_graceful_degradation:
                # Fallback to single LLM
                try:
                    single_result = self.single_generator.generate_response(request)
                    return self._create_single_ensemble_result(single_result, time.time() - start_time)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback to single LLM also failed: {fallback_error}")
            
            raise EnsembleError(f"Ensemble response generation failed: {e}")
    
    def _select_ensemble_providers(self) -> List[str]:
        """Select providers for ensemble generation."""
        # Get available models from all providers
        available_models = self.llm_manager.get_available_models()
        
        # Filter healthy providers
        provider_stats = self.llm_manager.get_provider_stats()
        healthy_providers = [
            provider for provider, stats in provider_stats.items()
            if stats.get("is_healthy", True) and available_models.get(provider)
        ]
        
        if not healthy_providers:
            raise ProviderUnavailableError("No healthy providers available for ensemble")
        
        # Select up to ensemble_size providers
        selected_providers = healthy_providers[:self.config.ensemble_size]
        
        self.logger.debug(f"Selected {len(selected_providers)} providers for ensemble: {selected_providers}")
        
        return selected_providers
    
    async def _generate_parallel_async(
        self, 
        request: ResponseRequest, 
        providers: List[str]
    ) -> List[Optional[ResponseResult]]:
        """Generate responses from multiple providers in parallel asynchronously."""
        
        # Create tasks for each provider
        tasks = []
        for provider in providers:
            # Create provider-specific request
            provider_request = self._create_provider_request(request, provider)
            task = asyncio.create_task(
                self._generate_single_with_timeout_async(provider_request)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.ensemble_timeout
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Collect partial results
            responses = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    try:
                        result = task.result()
                        responses.append(result)
                    except Exception:
                        responses.append(None)
                else:
                    responses.append(None)
        
        # Convert exceptions to None
        processed_responses = []
        for response in responses:
            if isinstance(response, Exception):
                self.logger.warning(f"Provider failed with exception: {response}")
                processed_responses.append(None)
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    def _generate_parallel_sync(
        self, 
        request: ResponseRequest, 
        providers: List[str]
    ) -> List[Optional[ResponseResult]]:
        """Generate responses from multiple providers in parallel synchronously."""
        
        responses = [None] * len(providers)
        
        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            # Submit tasks for each provider
            future_to_index = {}
            for i, provider in enumerate(providers):
                provider_request = self._create_provider_request(request, provider)
                future = executor.submit(self._generate_single_with_timeout_sync, provider_request)
                future_to_index[future] = i
            
            # Collect results with timeout
            try:
                for future in as_completed(future_to_index.keys(), timeout=self.config.ensemble_timeout):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        responses[index] = result
                    except Exception as e:
                        self.logger.warning(f"Provider {providers[index]} failed: {e}")
                        responses[index] = None
            except TimeoutError:
                self.logger.warning("Some providers timed out in ensemble generation")
        
        return responses
    
    async def _generate_sequential_async(
        self, 
        request: ResponseRequest, 
        providers: List[str]
    ) -> List[Optional[ResponseResult]]:
        """Generate responses from multiple providers sequentially asynchronously."""
        
        responses = []
        remaining_time = self.config.ensemble_timeout
        
        for provider in providers:
            if remaining_time <= 0:
                break
            
            start_time = time.time()
            provider_request = self._create_provider_request(request, provider)
            
            try:
                response = await asyncio.wait_for(
                    self.single_generator.generate_response_async(provider_request),
                    timeout=min(remaining_time, self.config.single_timeout)
                )
                responses.append(response)
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed: {e}")
                responses.append(None)
            
            elapsed = time.time() - start_time
            remaining_time -= elapsed
        
        return responses
    
    def _generate_sequential_sync(
        self, 
        request: ResponseRequest, 
        providers: List[str]
    ) -> List[Optional[ResponseResult]]:
        """Generate responses from multiple providers sequentially synchronously."""
        
        responses = []
        remaining_time = self.config.ensemble_timeout
        
        for provider in providers:
            if remaining_time <= 0:
                break
            
            start_time = time.time()
            provider_request = self._create_provider_request(request, provider)
            
            try:
                response = self.single_generator.generate_response(provider_request)
                responses.append(response)
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed: {e}")
                responses.append(None)
            
            elapsed = time.time() - start_time
            remaining_time -= elapsed
        
        return responses
    
    async def _generate_single_with_timeout_async(self, request: ResponseRequest) -> Optional[ResponseResult]:
        """Generate single response with timeout handling asynchronously."""
        try:
            return await asyncio.wait_for(
                self.single_generator.generate_response_async(request),
                timeout=self.config.single_timeout
            )
        except Exception as e:
            self.logger.warning(f"Single generation failed: {e}")
            return None
    
    def _generate_single_with_timeout_sync(self, request: ResponseRequest) -> Optional[ResponseResult]:
        """Generate single response with timeout handling synchronously."""
        try:
            return self.single_generator.generate_response(request)
        except Exception as e:
            self.logger.warning(f"Single generation failed: {e}")
            return None
    
    def _create_provider_request(self, request: ResponseRequest, provider: str) -> ResponseRequest:
        """Create provider-specific request."""
        # Get appropriate model for this provider
        available_models = self.llm_manager.get_available_models()
        provider_models = available_models.get(provider, [])
        
        if not provider_models:
            raise ProviderUnavailableError(f"No models available for provider: {provider}")
        
        # Create new request with provider-specific model
        provider_request = ResponseRequest(
            query=request.query,
            context=request.context,
            retrieval_result=request.retrieval_result,
            user_context=request.user_context,
            conversation_history=request.conversation_history,
            model=provider_models[0],  # Use first available model
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,  # Disable streaming for ensemble
            system_prompt=request.system_prompt,
            custom_instructions=request.custom_instructions,
            response_format=request.response_format,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata={**request.metadata, "ensemble_provider": provider},
            timestamp=request.timestamp
        )
        
        return provider_request
    
    def _select_best_response(
        self, 
        responses: List[ResponseResult], 
        request: ResponseRequest
    ) -> Tuple[ResponseResult, float, str]:
        """
        Select the best response from ensemble results.
        
        Returns:
            Tuple of (best_response, consensus_score, selection_method)
        """
        if len(responses) == 1:
            return responses[0], 1.0, "single_response"
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(responses)
        
        # Select based on quality scores
        if self.config.enable_quality_filtering:
            best_response = self._select_by_quality(responses)
            selection_method = "quality_based"
        else:
            # Fallback to length-based selection
            best_response = max(responses, key=lambda r: len(r.response))
            selection_method = "length_based"
        
        return best_response, consensus_score, selection_method
    
    def _calculate_consensus_score(self, responses: List[ResponseResult]) -> float:
        """Calculate consensus score among responses."""
        if len(responses) < 2:
            return 1.0
        
        # Simple consensus based on response similarity
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
    
    def _select_by_quality(self, responses: List[ResponseResult]) -> ResponseResult:
        """Select response with highest overall quality score."""
        # Calculate overall quality for each response if not already done
        for response in responses:
            if response.overall_quality_score == 0.0:
                response.calculate_overall_quality()
        
        # Select response with highest quality score
        best_response = max(responses, key=lambda r: r.overall_quality_score)
        
        return best_response
    
    def _create_single_ensemble_result(
        self, 
        single_result: ResponseResult, 
        ensemble_time: float
    ) -> EnsembleResult:
        """Create ensemble result from single response (fallback)."""
        return EnsembleResult(
            best_response=single_result,
            all_responses=[single_result],
            consensus_score=1.0,
            selection_method="single_fallback",
            ensemble_time=ensemble_time,
            providers_used=[single_result.provider_used] if single_result.provider_used else [],
            models_used=[single_result.model_used] if single_result.model_used else [],
            total_tokens_used=single_result.tokens_used or 0,
            total_cost_estimate=single_result.cost_estimate
        )
    
    async def _generate_parallel_async_with_error_handling(
        self, 
        request: ResponseRequest, 
        providers: List[str],
        ensemble_timeout: float
    ) -> List[Optional[ResponseResult]]:
        """Generate responses from multiple providers in parallel with enhanced error handling."""
        
        # Create tasks for each provider with individual circuit breaker protection
        tasks = []
        for provider in providers:
            # Create provider-specific request
            provider_request = self._create_provider_request(request, provider)
            
            # Create task with circuit breaker and error handling
            task = asyncio.create_task(
                self._generate_single_with_circuit_breaker_async(provider_request, provider)
            )
            tasks.append((task, provider))
        
        responses = []
        
        # Wait for all tasks to complete with timeout
        try:
            # Use asyncio.gather with return_exceptions to handle individual failures
            task_results = await asyncio.wait_for(
                asyncio.gather(*[task for task, _ in tasks], return_exceptions=True),
                timeout=ensemble_timeout
            )
            
            # Process results
            for i, result in enumerate(task_results):
                provider = tasks[i][1]
                if isinstance(result, Exception):
                    self.logger.warning(f"Provider {provider} failed: {result}")
                    responses.append(None)
                else:
                    responses.append(result)
                    
        except asyncio.TimeoutError:
            self.logger.warning(f"Ensemble generation timed out after {ensemble_timeout}s")
            
            # Cancel remaining tasks and collect partial results
            for task, provider in tasks:
                if not task.done():
                    task.cancel()
                    
            # Collect what we have
            for task, provider in tasks:
                if task.done() and not task.cancelled():
                    try:
                        result = task.result()
                        responses.append(result)
                    except Exception as e:
                        self.logger.warning(f"Provider {provider} failed during timeout: {e}")
                        responses.append(None)
                else:
                    responses.append(None)
        
        return responses
    
    async def _generate_sequential_async_with_error_handling(
        self, 
        request: ResponseRequest, 
        providers: List[str],
        ensemble_timeout: float
    ) -> List[Optional[ResponseResult]]:
        """Generate responses from multiple providers sequentially with enhanced error handling."""
        
        responses = []
        remaining_time = ensemble_timeout
        
        for provider in providers:
            if remaining_time <= 0:
                self.logger.warning(f"Skipping provider {provider} due to timeout")
                responses.append(None)
                continue
            
            start_time = time.time()
            provider_request = self._create_provider_request(request, provider)
            
            try:
                # Use circuit breaker and timeout for individual provider
                provider_timeout = min(remaining_time, self.config.single_timeout)
                
                response = await self._generate_single_with_circuit_breaker_async(
                    provider_request, provider, provider_timeout
                )
                responses.append(response)
                
            except Exception as e:
                # Handle individual provider error
                error_context = {
                    "provider": provider,
                    "model": provider_request.model,
                    "operation_type": "ensemble_sequential",
                    "request_id": request.request_id
                }
                
                decision = self.error_handler.handle_error(e, error_context, 0)
                self.logger.warning(f"Provider {provider} failed in sequential mode: {e}")
                responses.append(None)
            
            elapsed = time.time() - start_time
            remaining_time -= elapsed
        
        return responses
    
    async def _generate_single_with_circuit_breaker_async(
        self, 
        request: ResponseRequest, 
        provider: str,
        timeout: Optional[float] = None
    ) -> Optional[ResponseResult]:
        """Generate single response with circuit breaker protection."""
        try:
            # Use error handler's circuit breaker
            if timeout is None:
                timeout = self.config.single_timeout
                
            return await self.error_handler.with_circuit_breaker_async(
                self.single_generator.generate_response_async,
                provider,
                request
            )
            
        except Exception as e:
            self.logger.warning(f"Circuit breaker protected generation failed for {provider}: {e}")
            return None

    def get_generator_info(self) -> Dict[str, Any]:
        """Get generator information and configuration."""
        return {
            "generator_type": "ensemble_llm",
            "config": self.config.to_dict(),
            "stats": self.get_stats(),
            "llm_manager_info": self.llm_manager.get_manager_info(),
            "single_generator_info": self.single_generator.get_generator_info(),
            "error_handler_info": self.error_handler.get_handler_info()
        }