"""
LLM Manager with Load Balancing and Failover Support.
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import json
from collections import defaultdict

from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk, 
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.llm.providers import OpenAIProvider, GeminiProvider, ClaudeProvider, OllamaProvider
from src.core.logging import LoggerMixin
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for LLM providers."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    FASTEST_RESPONSE = "fastest_response"
    HEALTH_BASED = "health_based"


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    is_healthy: bool = True
    
    def update_success(self, response_time: float):
        """Update stats for successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.successful_requests
        self.last_request_time = datetime.now(timezone.utc)
        self.consecutive_errors = 0
        self.is_healthy = True
    
    def update_failure(self, error: str):
        """Update stats for failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.consecutive_errors += 1
        self.last_request_time = datetime.now(timezone.utc)
        
        # Mark as unhealthy if too many consecutive errors
        if self.consecutive_errors >= 5:
            self.is_healthy = False
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0


@dataclass
class ProviderConfig:
    """Configuration for a provider in the manager."""
    provider: LLMProvider
    config: LLMConfig
    priority: int = 1  # Higher priority = preferred
    max_requests_per_minute: int = 60
    enabled: bool = True
    weight: float = 1.0  # For weighted load balancing


class LLMManager(LoggerMixin):
    """
    LLM Manager with Load Balancing, Failover, and Unified Response Handling.
    
    Features:
    - Multiple provider support (OpenAI, Gemini, Claude, Ollama)
    - Load balancing strategies (round-robin, random, least-used, etc.)
    - Automatic failover and retry logic
    - Health monitoring and circuit breaker pattern
    - Rate limiting and request throttling
    - Unified response handling and normalization
    """
    
    def __init__(self, provider_configs: List[ProviderConfig]):
        """
        Initialize LLM Manager.
        
        Args:
            provider_configs: List of provider configurations
        """
        self.provider_configs = provider_configs
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.provider_stats: Dict[LLMProvider, ProviderStats] = {}
        self.load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
        self.round_robin_index = 0
        self.max_retries = 3
        self.retry_delay = 1.0
        self.request_counts: Dict[LLMProvider, List[datetime]] = defaultdict(list)
        
        # Initialize providers
        self._initialize_providers()
        
        # Health check interval
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now(timezone.utc)
    
    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        provider_classes = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.GEMINI: GeminiProvider,
            LLMProvider.CLAUDE: ClaudeProvider,
            LLMProvider.OLLAMA: OllamaProvider,
        }
        
        for provider_config in self.provider_configs:
            if not provider_config.enabled:
                continue
                
            provider_class = provider_classes.get(provider_config.provider)
            if not provider_class:
                self.logger.warning(f"Unknown provider: {provider_config.provider}")
                continue
            
            try:
                provider = provider_class(provider_config.config)
                self.providers[provider_config.provider] = provider
                self.provider_stats[provider_config.provider] = ProviderStats()
                self.logger.info(f"Initialized provider: {provider_config.provider}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {provider_config.provider}: {e}")
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Set the load balancing strategy."""
        self.load_balancing_strategy = strategy
        self.logger.info(f"Set load balancing strategy to: {strategy.value}")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        stats = {}
        for provider, stat in self.provider_stats.items():
            stats[provider.value] = {
                "total_requests": stat.total_requests,
                "successful_requests": stat.successful_requests,
                "failed_requests": stat.failed_requests,
                "success_rate": stat.get_success_rate(),
                "average_response_time": stat.average_response_time,
                "is_healthy": stat.is_healthy,
                "consecutive_errors": stat.consecutive_errors,
                "last_request_time": stat.last_request_time.isoformat() if stat.last_request_time else None,
                "last_error": stat.last_error
            }
        return stats
    
    def _select_provider(self, request: LLMRequest) -> Optional[LLMProvider]:
        """Select a provider based on the load balancing strategy."""
        available_providers = self._get_available_providers()
        
        if not available_providers:
            return None
        
        # Filter by model if specified
        if request.model:
            compatible_providers = []
            for provider_type in available_providers:
                provider = self.providers[provider_type]
                try:
                    # Check if provider supports the model
                    available_models = provider.get_available_models()
                    if request.model in available_models:
                        compatible_providers.append(provider_type)
                except Exception:
                    # If we can't get models, assume it's compatible
                    compatible_providers.append(provider_type)
            
            if compatible_providers:
                available_providers = compatible_providers
        
        if not available_providers:
            return None
        
        # Apply load balancing strategy
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            return self._random_select(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_USED:
            return self._least_used_select(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_select(available_providers)
        else:
            return available_providers[0]
    
    def _get_available_providers(self) -> List[LLMProvider]:
        """Get list of available and healthy providers."""
        available = []
        current_time = datetime.now(timezone.utc)
        
        for provider_config in self.provider_configs:
            if not provider_config.enabled:
                continue
                
            provider_type = provider_config.provider
            if provider_type not in self.providers:
                continue
            
            # Check rate limits
            if self._is_rate_limited(provider_type, provider_config):
                continue
            
            # Check health
            stats = self.provider_stats.get(provider_type)
            if stats and not stats.is_healthy:
                # Skip if recently failed
                if stats.last_request_time and current_time - stats.last_request_time < timedelta(minutes=5):
                    continue
            
            available.append(provider_type)
        
        return available
    
    def _is_rate_limited(self, provider_type: LLMProvider, config: ProviderConfig) -> bool:
        """Check if provider is rate limited."""
        current_time = datetime.now(timezone.utc)
        requests = self.request_counts[provider_type]
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        self.request_counts[provider_type] = [
            req_time for req_time in requests if req_time > cutoff_time
        ]
        
        # Check if we've exceeded the limit
        return len(self.request_counts[provider_type]) >= config.max_requests_per_minute
    
    def _round_robin_select(self, providers: List[LLMProvider]) -> Optional[LLMProvider]:
        """Round-robin provider selection."""
        if not providers:
            return None
        
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    def _random_select(self, providers: List[LLMProvider]) -> Optional[LLMProvider]:
        """Random provider selection."""
        if not providers:
            return None
        return random.choice(providers)
    
    def _least_used_select(self, providers: List[LLMProvider]) -> Optional[LLMProvider]:
        """Select provider with least usage."""
        if not providers:
            return None
            
        min_requests = float('inf')
        selected_provider = None
        
        for provider in providers:
            stats = self.provider_stats.get(provider)
            if stats and stats.total_requests < min_requests:
                min_requests = stats.total_requests
                selected_provider = provider
        
        return selected_provider or providers[0]
    
    def _fastest_response_select(self, providers: List[LLMProvider]) -> Optional[LLMProvider]:
        """Select provider with fastest average response time."""
        if not providers:
            return None
            
        min_response_time = float('inf')
        selected_provider = None
        
        for provider in providers:
            stats = self.provider_stats.get(provider)
            if stats and stats.average_response_time > 0 and stats.average_response_time < min_response_time:
                min_response_time = stats.average_response_time
                selected_provider = provider
        
        return selected_provider or providers[0]
    
    def _health_based_select(self, providers: List[LLMProvider]) -> Optional[LLMProvider]:
        """Select provider based on health score."""
        if not providers:
            return None
            
        best_score = -1
        selected_provider = None
        
        for provider in providers:
            stats = self.provider_stats.get(provider)
            if not stats:
                continue
            
            # Calculate health score (0-100)
            health_score = stats.get_success_rate()
            
            # Penalize for consecutive errors
            health_score -= stats.consecutive_errors * 10
            
            # Bonus for fast response times
            if stats.average_response_time > 0:
                health_score += max(0, 50 - stats.average_response_time)
            
            if health_score > best_score:
                best_score = health_score
                selected_provider = provider
        
        return selected_provider or providers[0]
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously with load balancing and failover.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        for attempt in range(self.max_retries):
            provider_type = self._select_provider(request)
            
            if not provider_type:
                raise LLMError("No available providers")
            
            provider = self.providers[provider_type]
            
            try:
                start_time = time.time()
                
                # Track request count for rate limiting
                self.request_counts[provider_type].append(datetime.now(timezone.utc))
                
                # Make the request
                response = await provider.generate_async(request)
                
                # Update stats
                response_time = time.time() - start_time
                self.provider_stats[provider_type].update_success(response_time)
                
                # Add provider info to response metadata
                response.metadata.update({
                    "provider": provider_type.value,
                    "attempt": attempt + 1,
                    "load_balancing_strategy": self.load_balancing_strategy.value
                })
                
                return response
                
            except Exception as e:
                # Update stats
                self.provider_stats[provider_type].update_failure(str(e))
                
                self.logger.warning(f"Provider {provider_type.value} failed (attempt {attempt + 1}): {e}")
                
                # If it's the last attempt, raise the error
                if attempt == self.max_retries - 1:
                    raise
                
                # Wait before retrying
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        raise LLMError("All providers failed after maximum retries")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously with load balancing and failover.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        for attempt in range(self.max_retries):
            provider_type = self._select_provider(request)
            
            if not provider_type:
                raise LLMError("No available providers")
            
            provider = self.providers[provider_type]
            
            try:
                start_time = time.time()
                
                # Track request count for rate limiting
                self.request_counts[provider_type].append(datetime.now(timezone.utc))
                
                # Make the request
                response = provider.generate(request)
                
                # Update stats
                response_time = time.time() - start_time
                self.provider_stats[provider_type].update_success(response_time)
                
                # Add provider info to response metadata
                response.metadata.update({
                    "provider": provider_type.value,
                    "attempt": attempt + 1,
                    "load_balancing_strategy": self.load_balancing_strategy.value
                })
                
                return response
                
            except Exception as e:
                # Update stats
                self.provider_stats[provider_type].update_failure(str(e))
                
                self.logger.warning(f"Provider {provider_type.value} failed (attempt {attempt + 1}): {e}")
                
                # If it's the last attempt, raise the error
                if attempt == self.max_retries - 1:
                    raise
                
                # Wait before retrying
                time.sleep(self.retry_delay * (2 ** attempt))
        
        raise LLMError("All providers failed after maximum retries")
    
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Generate streaming response asynchronously with load balancing.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        provider_type = self._select_provider(request)
        
        if not provider_type:
            raise LLMError("No available providers")
        
        provider = self.providers[provider_type]
        
        try:
            start_time = time.time()
            
            # Track request count for rate limiting
            self.request_counts[provider_type].append(datetime.now(timezone.utc))
            
            # Make the streaming request
            async for chunk in provider.generate_stream_async(request):
                # Add provider info to chunk metadata
                chunk.metadata.update({
                    "provider": provider_type.value,
                    "load_balancing_strategy": self.load_balancing_strategy.value
                })
                
                yield chunk
                
                # Update stats on final chunk
                if chunk.finish_reason:
                    response_time = time.time() - start_time
                    self.provider_stats[provider_type].update_success(response_time)
                    
        except Exception as e:
            # Update stats
            self.provider_stats[provider_type].update_failure(str(e))
            raise
    
    def generate_stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """
        Generate streaming response synchronously with load balancing.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        provider_type = self._select_provider(request)
        
        if not provider_type:
            raise LLMError("No available providers")
        
        provider = self.providers[provider_type]
        
        try:
            start_time = time.time()
            
            # Track request count for rate limiting
            self.request_counts[provider_type].append(datetime.now(timezone.utc))
            
            # Make the streaming request
            for chunk in provider.generate_stream(request):
                # Add provider info to chunk metadata
                chunk.metadata.update({
                    "provider": provider_type.value,
                    "load_balancing_strategy": self.load_balancing_strategy.value
                })
                
                yield chunk
                
                # Update stats on final chunk
                if chunk.finish_reason:
                    response_time = time.time() - start_time
                    self.provider_stats[provider_type].update_success(response_time)
                    
        except Exception as e:
            # Update stats
            self.provider_stats[provider_type].update_failure(str(e))
            raise
    
    async def count_tokens_async(self, messages: List[LLMMessage], model: str) -> int:
        """
        Count tokens in messages asynchronously.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Token count
        """
        # Try to find a provider that supports the model
        for provider_type, provider in self.providers.items():
            try:
                return await provider.count_tokens_async(messages, model)
            except Exception as e:
                self.logger.warning(f"Token counting failed for provider {provider_type.value}: {e}")
                continue
        
        # Fallback to character-based estimation
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4  # Rough approximation: 4 chars per token
    
    def count_tokens(self, messages: List[LLMMessage], model: str) -> int:
        """
        Count tokens in messages synchronously.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Token count
        """
        # Try to find a provider that supports the model
        for provider_type, provider in self.providers.items():
            try:
                return provider.count_tokens(messages, model)
            except Exception as e:
                self.logger.warning(f"Token counting failed for provider {provider_type.value}: {e}")
                continue
        
        # Fallback to character-based estimation
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4  # Rough approximation: 4 chars per token
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get available models from all providers.
        
        Returns:
            Dictionary mapping provider names to model lists
        """
        models = {}
        for provider_type, provider in self.providers.items():
            try:
                models[provider_type.value] = provider.get_available_models()
            except Exception as e:
                self.logger.warning(f"Failed to get models for provider {provider_type.value}: {e}")
                models[provider_type.value] = []
        
        return models
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform health check on all providers.
        
        Returns:
            Health check results
        """
        results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                health_result = await provider.health_check_async()
                results[provider_type.value] = health_result
                
                # Update health status based on result
                if health_result.get("status") == "healthy":
                    self.provider_stats[provider_type].is_healthy = True
                    self.provider_stats[provider_type].consecutive_errors = 0
                else:
                    self.provider_stats[provider_type].is_healthy = False
                    
            except Exception as e:
                results[provider_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                self.provider_stats[provider_type].is_healthy = False
        
        self.last_health_check = datetime.now(timezone.utc)
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all providers synchronously.
        
        Returns:
            Health check results
        """
        results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                health_result = provider.health_check()
                results[provider_type.value] = health_result
                
                # Update health status based on result
                if health_result.get("status") == "healthy":
                    self.provider_stats[provider_type].is_healthy = True
                    self.provider_stats[provider_type].consecutive_errors = 0
                else:
                    self.provider_stats[provider_type].is_healthy = False
                    
            except Exception as e:
                results[provider_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                self.provider_stats[provider_type].is_healthy = False
        
        self.last_health_check = datetime.now(timezone.utc)
        return results
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information and status."""
        return {
            "total_providers": len(self.providers),
            "healthy_providers": sum(1 for stats in self.provider_stats.values() if stats.is_healthy),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "last_health_check": self.last_health_check.isoformat(),
            "provider_stats": self.get_provider_stats(),
            "available_models": self.get_available_models()
        }