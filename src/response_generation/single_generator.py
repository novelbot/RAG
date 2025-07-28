"""
Single LLM Response Generator - Fast response generation using a single LLM provider.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional

from src.llm.manager import LLMManager
from src.llm.base import LLMRequest, LLMMessage, LLMRole
from src.core.logging import LoggerMixin
from src.response_generation.base import (
    BaseResponseGenerator, ResponseRequest, ResponseResult, 
    ResponseGeneratorConfig, EvaluationMetric
)
from src.response_generation.exceptions import (
    ResponseGenerationError, ProviderUnavailableError, 
    TimeoutError, ContextTooLongError
)
from src.response_generation.error_handler import ErrorHandler


class SingleLLMGenerator(BaseResponseGenerator, LoggerMixin):
    """
    Single LLM Response Generator for fast response generation.
    
    Features:
    - Provider selection and fallback
    - Prompt formatting with context injection
    - Basic error handling and timeout management
    - Response processing and quality scoring
    - Performance metrics and monitoring
    """
    
    def __init__(self, llm_manager: LLMManager, config: ResponseGeneratorConfig):
        """
        Initialize Single LLM Generator.
        
        Args:
            llm_manager: LLM manager instance for provider access
            config: Response generator configuration
        """
        super().__init__(config)
        self.llm_manager = llm_manager
        
        # Validate configuration
        if config.single_timeout <= 0:
            raise ValueError("Single timeout must be positive")
        
        # Initialize error handler for comprehensive error management
        self.error_handler = ErrorHandler(config)
        
        self.logger.info("SingleLLMGenerator initialized successfully with error handler")
    
    async def generate_response_async(self, request: ResponseRequest) -> ResponseResult:
        """
        Generate response asynchronously using single LLM with comprehensive error handling.
        
        Args:
            request: Response generation request
            
        Returns:
            Generated response result
        """
        start_time = time.time()
        retry_count = 0
        max_retries = 3  # Will be determined by error handler
        
        while retry_count <= max_retries:
            try:
                # Validate request
                self._validate_request(request)
                
                # Build LLM request
                llm_request = await self._build_llm_request_async(request)
                
                # Determine provider and check circuit breaker
                provider = self._extract_provider_from_model(llm_request.model)
                context = {
                    "provider": provider,
                    "model": llm_request.model,
                    "operation_type": "single",
                    "retry_count": retry_count
                }
                
                # Get adaptive timeout based on provider performance and complexity
                timeout = self.error_handler.get_timeout(
                    operation_type="single",
                    provider=provider,
                    complexity=self._estimate_request_complexity(request)
                )
                
                # Generate response with circuit breaker and timeout protection
                llm_response = await self.error_handler.with_circuit_breaker_async(
                    self._generate_with_timeout_async,
                    provider,
                    llm_request,
                    timeout
                )
                
                # Create response result
                result = self._create_response_result(
                    request, llm_response, time.time() - start_time
                )
                
                # Update statistics
                self._update_stats_success(result.generation_time)
                
                self.logger.info(
                    f"Generated response in {result.generation_time:.3f}s using {result.provider_used} "
                    f"(attempt {retry_count + 1}, timeout: {timeout:.1f}s)"
                )
                
                return result
                
            except Exception as e:
                # Handle error using comprehensive error handler
                error_context = {
                    "provider": provider if 'provider' in locals() else None,
                    "model": llm_request.model if 'llm_request' in locals() else None,
                    "operation_type": "single",
                    "retry_count": retry_count,
                    "request_id": request.request_id,
                    "user_id": request.user_id
                }
                
                # Get error handling decision
                decision = self.error_handler.handle_error(e, error_context, retry_count)
                
                self.logger.warning(
                    f"Error on attempt {retry_count + 1}: {e} "
                    f"(decision: {decision['action']}, delay: {decision['delay']}s)"
                )
                
                if decision["action"] == "retry" and retry_count < max_retries:
                    retry_count += 1
                    if decision["delay"] > 0:
                        await asyncio.sleep(decision["delay"])
                    continue
                    
                elif decision["action"] == "fallback" and self.config.enable_graceful_degradation:
                    return self._create_fallback_response(request, str(e), time.time() - start_time)
                    
                else:
                    # Update statistics and re-raise
                    self._update_stats_failure()
                    self.logger.error(f"Response generation failed after {retry_count + 1} attempts: {e}")
                    raise ResponseGenerationError(f"Response generation failed: {e}")
        
        # Should not reach here, but just in case
        self._update_stats_failure()
        raise ResponseGenerationError("Maximum retry attempts exceeded")
    
    def generate_response(self, request: ResponseRequest) -> ResponseResult:
        """
        Generate response synchronously using single LLM with comprehensive error handling.
        
        Args:
            request: Response generation request
            
        Returns:
            Generated response result
        """
        start_time = time.time()
        retry_count = 0
        max_retries = 3  # Will be determined by error handler
        
        while retry_count <= max_retries:
            try:
                # Validate request
                self._validate_request(request)
                
                # Build LLM request
                llm_request = self._build_llm_request(request)
                
                # Determine provider and check circuit breaker
                provider = self._extract_provider_from_model(llm_request.model)
                context = {
                    "provider": provider,
                    "model": llm_request.model,
                    "operation_type": "single",
                    "retry_count": retry_count
                }
                
                # Get adaptive timeout based on provider performance and complexity
                timeout = self.error_handler.get_timeout(
                    operation_type="single",
                    provider=provider,
                    complexity=self._estimate_request_complexity(request)
                )
                
                # Generate response with circuit breaker and timeout protection
                llm_response = self.error_handler.with_circuit_breaker(
                    self._generate_with_timeout_sync,
                    provider,
                    llm_request,
                    timeout
                )
                
                # Create response result
                result = self._create_response_result(
                    request, llm_response, time.time() - start_time
                )
                
                # Update statistics
                self._update_stats_success(result.generation_time)
                
                self.logger.info(
                    f"Generated response in {result.generation_time:.3f}s using {result.provider_used} "
                    f"(attempt {retry_count + 1}, timeout: {timeout:.1f}s)"
                )
                
                return result
                
            except Exception as e:
                # Handle error using comprehensive error handler
                error_context = {
                    "provider": provider if 'provider' in locals() else None,
                    "model": llm_request.model if 'llm_request' in locals() else None,
                    "operation_type": "single",
                    "retry_count": retry_count,
                    "request_id": request.request_id,
                    "user_id": request.user_id
                }
                
                # Get error handling decision
                decision = self.error_handler.handle_error(e, error_context, retry_count)
                
                self.logger.warning(
                    f"Error on attempt {retry_count + 1}: {e} "
                    f"(decision: {decision['action']}, delay: {decision['delay']}s)"
                )
                
                if decision["action"] == "retry" and retry_count < max_retries:
                    retry_count += 1
                    if decision["delay"] > 0:
                        time.sleep(decision["delay"])
                    continue
                    
                elif decision["action"] == "fallback" and self.config.enable_graceful_degradation:
                    return self._create_fallback_response(request, str(e), time.time() - start_time)
                    
                else:
                    # Update statistics and re-raise
                    self._update_stats_failure()
                    self.logger.error(f"Response generation failed after {retry_count + 1} attempts: {e}")
                    raise ResponseGenerationError(f"Response generation failed: {e}")
        
        # Should not reach here, but just in case
        self._update_stats_failure()
        raise ResponseGenerationError("Maximum retry attempts exceeded")
    
    def _validate_request(self, request: ResponseRequest) -> None:
        """Validate the request parameters."""
        if not request.query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check context length if provided
        if request.context:
            estimated_tokens = len(request.context) // 4  # Rough estimation
            if estimated_tokens > self.config.max_context_length:
                raise ContextTooLongError(
                    f"Context too long: {estimated_tokens} tokens exceeds limit of {self.config.max_context_length}",
                    estimated_tokens,
                    self.config.max_context_length
                )
    
    async def _build_llm_request_async(self, request: ResponseRequest) -> LLMRequest:
        """Build LLM request from response request asynchronously."""
        return self._build_llm_request(request)
    
    def _build_llm_request(self, request: ResponseRequest) -> LLMRequest:
        """Build LLM request from response request."""
        # Build messages
        messages = []
        
        # Add system prompt
        system_prompt = self._build_system_prompt(request)
        if system_prompt:
            messages.append(LLMMessage(
                role=LLMRole.SYSTEM,
                content=system_prompt
            ))
        
        # Add conversation history
        messages.extend(request.conversation_history)
        
        # Add user query with context
        user_content = self._build_user_message(request)
        messages.append(LLMMessage(
            role=LLMRole.USER,
            content=user_content
        ))
        
        # Determine model
        model = request.model or self._select_model()
        
        # Create LLM request
        llm_request = LLMRequest(
            messages=messages,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return llm_request
    
    def _build_system_prompt(self, request: ResponseRequest) -> str:
        """Build system prompt from request and configuration."""
        if request.system_prompt:
            return request.system_prompt
        
        system_prompt = self.config.system_prompt_template
        
        # Add custom instructions if provided
        if request.custom_instructions:
            system_prompt += f"\n\nAdditional Instructions: {request.custom_instructions}"
        
        # Add response format instructions if specified
        if request.response_format:
            system_prompt += f"\n\nResponse Format: {request.response_format}"
        
        return system_prompt
    
    def _build_user_message(self, request: ResponseRequest) -> str:
        """Build user message with context injection."""
        message_parts = []
        
        # Add context if provided
        if request.context:
            message_parts.append(f"Context: {request.context}")
        
        # Add retrieval results if provided
        if request.retrieval_result and request.retrieval_result.contexts:
            context_texts = request.retrieval_result.get_content_texts()
            if context_texts:
                contexts_str = "\n\n".join(context_texts[:5])  # Limit to top 5 contexts
                message_parts.append(f"Retrieved Information:\n{contexts_str}")
        
        # Add user context if provided
        if request.user_context:
            user_info = ", ".join(f"{k}: {v}" for k, v in request.user_context.items())
            message_parts.append(f"User Context: {user_info}")
        
        # Add the actual query
        message_parts.append(f"Query: {request.query}")
        
        return "\n\n".join(message_parts)
    
    def _select_model(self) -> str:
        """Select appropriate model based on configuration and availability."""
        # Try preferred provider first
        if self.config.preferred_provider:
            available_models = self.llm_manager.get_available_models()
            provider_models = available_models.get(self.config.preferred_provider, [])
            if provider_models:
                return provider_models[0]  # Use first available model
        
        # Try fallback providers
        for provider in self.config.fallback_providers:
            available_models = self.llm_manager.get_available_models()
            provider_models = available_models.get(provider, [])
            if provider_models:
                return provider_models[0]
        
        # Use any available model
        all_models = self.llm_manager.get_available_models()
        for provider, models in all_models.items():
            if models:
                return models[0]
        
        raise ProviderUnavailableError("No available models found")
    
    def _create_response_result(
        self, 
        request: ResponseRequest, 
        llm_response, 
        generation_time: float
    ) -> ResponseResult:
        """Create response result from LLM response."""
        
        # Extract response content
        response_content = llm_response.content
        
        # Create result
        result = ResponseResult(
            response=response_content,
            llm_response=llm_response,
            generation_time=generation_time,
            provider_used=llm_response.metadata.get("provider"),
            model_used=llm_response.model,
            tokens_used=llm_response.usage.total_tokens if llm_response.usage else None,
            context_length=len(request.context or ""),
            prompt_tokens=llm_response.usage.prompt_tokens if llm_response.usage else 0,
            completion_tokens=llm_response.usage.completion_tokens if llm_response.usage else 0,
            request_timestamp=request.timestamp,
            response_timestamp=datetime.now(timezone.utc)
        )
        
        # Add processing steps
        result.processing_steps.extend([
            "Request validation",
            "LLM request building", 
            "Single LLM generation",
            "Response processing"
        ])
        
        # Calculate basic quality scores
        self._calculate_basic_quality_scores(result, request)
        
        return result
    
    def _calculate_basic_quality_scores(self, result: ResponseResult, request: ResponseRequest) -> None:
        """Calculate basic quality scores for the response."""
        
        # Length-based quality (simple heuristic)
        response_length = len(result.response)
        if response_length > 0:
            # Length score based on reasonable response length (50-2000 chars)
            length_score = min(1.0, max(0.1, (response_length - 10) / 1990))
            result.add_quality_score(
                EvaluationMetric.LENGTH,
                length_score,
                confidence=0.8,
                explanation=f"Response length: {response_length} characters"
            )
        
        # Confidence score from LLM metadata
        if hasattr(result.llm_response, 'metadata') and 'confidence' in result.llm_response.metadata:
            confidence = result.llm_response.metadata['confidence']
            result.confidence_score = confidence
        else:
            # Default confidence based on successful generation
            result.confidence_score = 0.7
        
        # Basic relevance score (placeholder - would need more sophisticated evaluation)
        relevance_score = 0.8  # Default assumption for successful generation
        result.add_quality_score(
            EvaluationMetric.RELEVANCE,
            relevance_score,
            confidence=0.6,
            explanation="Basic relevance estimation"
        )
        
        # Calculate overall quality
        result.calculate_overall_quality()
    
    def _create_fallback_response(
        self,
        request: ResponseRequest,
        error_message: str,
        generation_time: float
    ) -> ResponseResult:
        """Create fallback response when generation fails."""
        
        fallback_content = (
            "I apologize, but I'm unable to generate a response at the moment. "
            "Please try again later or rephrase your question."
        )
        
        from src.llm.base import LLMResponse, LLMProvider, LLMUsage
        from datetime import datetime, timezone
        
        # Create minimal LLM response
        fallback_llm_response = LLMResponse(
            content=fallback_content,
            model="fallback",
            provider=LLMProvider.OPENAI,  # Default
            finish_reason="error",
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            response_time=generation_time,
            metadata={"fallback": True, "error": error_message}
        )
        
        result = ResponseResult(
            response=fallback_content,
            llm_response=fallback_llm_response,
            generation_time=generation_time,
            provider_used="fallback",
            model_used="fallback",
            tokens_used=0,
            context_length=len(request.context or ""),
            request_timestamp=request.timestamp,
            response_timestamp=datetime.now(timezone.utc)
        )
        
        # Add error information
        result.errors.append(error_message)
        result.warnings.append("Fallback response generated due to error")
        
        # Add processing steps
        result.processing_steps.extend([
            "Request validation",
            "Generation failed",
            "Fallback response created"
        ])
        
        # Set low quality scores for fallback
        result.add_quality_score(EvaluationMetric.RELEVANCE, 0.3, 0.9, "Fallback response")
        result.calculate_overall_quality()
        
        return result
    
    async def _generate_with_timeout_async(self, llm_request: LLMRequest, timeout: float):
        """Generate LLM response with timeout protection (async)."""
        return await asyncio.wait_for(
            self.llm_manager.generate_async(llm_request),
            timeout=timeout
        )
    
    def _generate_with_timeout_sync(self, llm_request: LLMRequest, timeout: float):
        """Generate LLM response with timeout protection (sync)."""
        # For sync operations, we rely on the error handler's timeout mechanism
        # The actual timeout implementation may vary based on the LLM manager implementation
        return self.llm_manager.generate(llm_request)
    
    def _extract_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string."""
        # Try to get provider from LLM manager first
        available_models = self.llm_manager.get_available_models()
        for provider, models in available_models.items():
            if model in models:
                return provider
        
        # Fallback: extract from model name patterns
        model_lower = model.lower()
        if "gpt" in model_lower or "chatgpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "bard" in model_lower:
            return "google"
        elif "llama" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral"
        else:
            return "unknown"
    
    def _estimate_request_complexity(self, request: ResponseRequest) -> int:
        """
        Estimate request complexity on a scale of 1-10.
        
        Args:
            request: Response generation request
            
        Returns:
            Complexity score (1=simple, 10=very complex)
        """
        complexity = 1
        
        # Base complexity from query length
        query_length = len(request.query)
        if query_length > 1000:
            complexity += 3
        elif query_length > 500:
            complexity += 2
        elif query_length > 100:
            complexity += 1
        
        # Context complexity
        if request.context:
            context_length = len(request.context)
            if context_length > 5000:
                complexity += 3
            elif context_length > 2000:
                complexity += 2
            elif context_length > 500:
                complexity += 1
        
        # Conversation history complexity
        if request.conversation_history:
            history_length = len(request.conversation_history)
            if history_length > 10:
                complexity += 2
            elif history_length > 5:
                complexity += 1
        
        # Retrieval results complexity
        if request.retrieval_result and request.retrieval_result.contexts:
            if len(request.retrieval_result.contexts) > 5:
                complexity += 2
            elif len(request.retrieval_result.contexts) > 2:
                complexity += 1
        
        # Custom instructions add complexity
        if request.custom_instructions:
            complexity += 1
        
        # Response format requirements add complexity
        if request.response_format:
            complexity += 1
        
        # Streaming is generally simpler
        if request.stream:
            complexity -= 1
        
        # Ensure within bounds
        return max(1, min(10, complexity))

    def get_generator_info(self) -> Dict[str, Any]:
        """Get generator information and configuration."""
        return {
            "generator_type": "single_llm",
            "config": self.config.to_dict(),
            "stats": self.get_stats(),
            "llm_manager_info": self.llm_manager.get_manager_info(),
            "error_handler_info": self.error_handler.get_handler_info()
        }