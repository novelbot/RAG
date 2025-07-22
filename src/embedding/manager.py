"""
Embedding Manager with Load Balancing and Caching Support.
"""

import time
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import json

from src.embedding.base import (
    BaseEmbeddingProvider, EmbeddingRequest, EmbeddingResponse, 
    EmbeddingConfig, EmbeddingProvider, EmbeddingDimension, EmbeddingUsage
)
from src.embedding.providers import OpenAIEmbeddingProvider, GoogleEmbeddingProvider, OllamaEmbeddingProvider
from src.core.logging import LoggerMixin
from src.core.exceptions import EmbeddingError, RateLimitError, ConfigurationError


class EmbeddingLoadBalancingStrategy(Enum):
    """Load balancing strategies for embedding providers."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    FASTEST_RESPONSE = "fastest_response"
    DIMENSION_BASED = "dimension_based"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class EmbeddingProviderStats:
    """Statistics for an embedding provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    is_healthy: bool = True
    
    def update_success(self, response_time: float, tokens: int, cost: float = 0.0):
        """Update stats for successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.successful_requests
        self.total_tokens += tokens
        self.total_cost += cost
        self.last_request_time = datetime.utcnow()
        self.consecutive_errors = 0
        self.is_healthy = True
    
    def update_failure(self, error: str):
        """Update stats for failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.consecutive_errors += 1
        self.last_request_time = datetime.utcnow()
        
        # Mark as unhealthy if too many consecutive errors
        if self.consecutive_errors >= 5:
            self.is_healthy = False
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0


@dataclass
class EmbeddingProviderConfig:
    """Configuration for an embedding provider in the manager."""
    provider: EmbeddingProvider
    config: EmbeddingConfig
    priority: int = 1  # Higher priority = preferred
    max_requests_per_minute: int = 60
    enabled: bool = True
    weight: float = 1.0  # For weighted load balancing
    cost_per_1m_tokens: float = 0.0  # Cost optimization


@dataclass
class EmbeddingCache:
    """Cache entry for embeddings."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    timestamp: datetime
    ttl: int = 3600  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl)


class EmbeddingManager(LoggerMixin):
    """
    Embedding Manager with Load Balancing, Caching, and Optimization.
    
    Features:
    - Multiple provider support (OpenAI, Google, Ollama)
    - Load balancing strategies
    - Intelligent caching with TTL
    - Dimension-based routing
    - Cost optimization
    - Health monitoring and failover
    - Batch processing optimization
    """
    
    def __init__(self, provider_configs: List[EmbeddingProviderConfig], enable_cache: bool = True):
        """
        Initialize Embedding Manager.
        
        Args:
            provider_configs: List of provider configurations
            enable_cache: Whether to enable embedding caching
        """
        self.provider_configs = provider_configs
        self.providers: Dict[EmbeddingProvider, BaseEmbeddingProvider] = {}
        self.provider_stats: Dict[EmbeddingProvider, EmbeddingProviderStats] = {}
        self.load_balancing_strategy = EmbeddingLoadBalancingStrategy.ROUND_ROBIN
        self.round_robin_index = 0
        self.max_retries = 3
        self.retry_delay = 1.0
        self.request_counts: Dict[EmbeddingProvider, List[datetime]] = defaultdict(list)
        
        # Caching
        self.enable_cache = enable_cache
        self.cache: Dict[str, EmbeddingCache] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.default_cache_ttl = 3600  # 1 hour
        
        # Initialize providers
        self._initialize_providers()
        
        # Health check interval
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.utcnow()
    
    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        provider_classes = {
            EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider,
            EmbeddingProvider.GOOGLE: GoogleEmbeddingProvider,
            EmbeddingProvider.OLLAMA: OllamaEmbeddingProvider,
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
                self.provider_stats[provider_config.provider] = EmbeddingProviderStats()
                self.logger.info(f"Initialized embedding provider: {provider_config.provider}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {provider_config.provider}: {e}")
    
    def set_load_balancing_strategy(self, strategy: EmbeddingLoadBalancingStrategy) -> None:
        """Set the load balancing strategy."""
        self.load_balancing_strategy = strategy
        self.logger.info(f"Set load balancing strategy to: {strategy.value}")
    
    async def generate_embeddings_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings asynchronously with load balancing and caching."""
        # Check cache first
        if self.enable_cache:
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
            self.cache_misses += 1
        
        # Attempt generation with retries
        for attempt in range(self.max_retries):
            provider_type = self._select_provider(request)
            
            if not provider_type:
                raise EmbeddingError("No available providers")
            
            provider = self.providers[provider_type]
            
            try:
                start_time = time.time()
                
                # Track request count for rate limiting
                self.request_counts[provider_type].append(datetime.utcnow())
                
                # Make the request
                response = await provider.generate_embeddings_async(request)
                
                # Update stats
                response_time = time.time() - start_time
                cost = self._calculate_cost(response.usage.total_tokens, provider_type, request.model or provider.config.model)
                self.provider_stats[provider_type].update_success(response_time, response.usage.total_tokens, cost)
                
                # Add provider info to response metadata
                response.metadata.update({
                    "provider": provider_type.value,
                    "attempt": attempt + 1,
                    "load_balancing_strategy": self.load_balancing_strategy.value,
                    "cached": False,
                    "cost": cost
                })
                
                # Cache the result
                if self.enable_cache:
                    self._save_to_cache(cache_key, response)
                
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
        
        raise EmbeddingError("All providers failed after maximum retries")
    
    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings synchronously with load balancing and caching."""
        # Check cache first
        if self.enable_cache:
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
            self.cache_misses += 1
        
        # Attempt generation with retries
        for attempt in range(self.max_retries):
            provider_type = self._select_provider(request)
            
            if not provider_type:
                raise EmbeddingError("No available providers")
            
            provider = self.providers[provider_type]
            
            try:
                start_time = time.time()
                
                # Track request count for rate limiting
                self.request_counts[provider_type].append(datetime.utcnow())
                
                # Make the request
                response = provider.generate_embeddings(request)
                
                # Update stats
                response_time = time.time() - start_time
                cost = self._calculate_cost(response.usage.total_tokens, provider_type, request.model or provider.config.model)
                self.provider_stats[provider_type].update_success(response_time, response.usage.total_tokens, cost)
                
                # Add provider info to response metadata
                response.metadata.update({
                    "provider": provider_type.value,
                    "attempt": attempt + 1,
                    "load_balancing_strategy": self.load_balancing_strategy.value,
                    "cached": False,
                    "cost": cost
                })
                
                # Cache the result
                if self.enable_cache:
                    self._save_to_cache(cache_key, response)
                
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
        
        raise EmbeddingError("All providers failed after maximum retries")
    
    def _select_provider(self, request: EmbeddingRequest) -> Optional[EmbeddingProvider]:
        """Select a provider based on the load balancing strategy."""
        available_providers = self._get_available_providers(request)
        
        if not available_providers:
            return None
        
        # Apply load balancing strategy
        if self.load_balancing_strategy == EmbeddingLoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_providers)
        elif self.load_balancing_strategy == EmbeddingLoadBalancingStrategy.RANDOM:
            return self._random_select(available_providers)
        elif self.load_balancing_strategy == EmbeddingLoadBalancingStrategy.LEAST_USED:
            return self._least_used_select(available_providers)
        elif self.load_balancing_strategy == EmbeddingLoadBalancingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(available_providers)
        elif self.load_balancing_strategy == EmbeddingLoadBalancingStrategy.DIMENSION_BASED:
            return self._dimension_based_select(available_providers, request)
        elif self.load_balancing_strategy == EmbeddingLoadBalancingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(available_providers, request)
        else:
            return available_providers[0]
    
    def _get_available_providers(self, request: EmbeddingRequest) -> List[EmbeddingProvider]:
        """Get list of available and healthy providers."""
        available = []
        current_time = datetime.utcnow()
        
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
            
            # Check model compatibility
            if request.model:
                provider = self.providers[provider_type]
                supported_models = provider.get_supported_models()
                if request.model not in supported_models:
                    continue
            
            available.append(provider_type)
        
        return available
    
    def _is_rate_limited(self, provider_type: EmbeddingProvider, config: EmbeddingProviderConfig) -> bool:
        """Check if provider is rate limited."""
        current_time = datetime.utcnow()
        requests = self.request_counts[provider_type]
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        self.request_counts[provider_type] = [
            req_time for req_time in requests if req_time > cutoff_time
        ]
        
        # Check if we've exceeded the limit
        return len(self.request_counts[provider_type]) >= config.max_requests_per_minute
    
    def _round_robin_select(self, providers: List[EmbeddingProvider]) -> EmbeddingProvider:
        """Round-robin provider selection."""
        if not providers:
            return None
        
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    def _random_select(self, providers: List[EmbeddingProvider]) -> EmbeddingProvider:
        """Random provider selection."""
        import random
        return random.choice(providers)
    
    def _least_used_select(self, providers: List[EmbeddingProvider]) -> EmbeddingProvider:
        """Select provider with least usage."""
        min_requests = float('inf')
        selected_provider = None
        
        for provider in providers:
            stats = self.provider_stats.get(provider)
            if stats and stats.total_requests < min_requests:
                min_requests = stats.total_requests
                selected_provider = provider
        
        return selected_provider or providers[0]
    
    def _fastest_response_select(self, providers: List[EmbeddingProvider]) -> EmbeddingProvider:
        """Select provider with fastest average response time."""
        min_response_time = float('inf')
        selected_provider = None
        
        for provider in providers:
            stats = self.provider_stats.get(provider)
            if stats and stats.average_response_time > 0 and stats.average_response_time < min_response_time:
                min_response_time = stats.average_response_time
                selected_provider = provider
        
        return selected_provider or providers[0]
    
    def _dimension_based_select(self, providers: List[EmbeddingProvider], request: EmbeddingRequest) -> EmbeddingProvider:
        """Select provider based on dimension requirements."""
        if not request.dimensions:
            return providers[0]
        
        # Find providers that support the requested dimensions
        compatible_providers = []
        for provider_type in providers:
            provider = self.providers[provider_type]
            try:
                model = request.model or provider.config.model
                dim_info = provider.get_embedding_dimension(model)
                if dim_info.dimensions == request.dimensions:
                    compatible_providers.append(provider_type)
            except Exception:
                continue
        
        if compatible_providers:
            return self._fastest_response_select(compatible_providers)
        else:
            return providers[0]
    
    def _cost_optimized_select(self, providers: List[EmbeddingProvider], request: EmbeddingRequest) -> EmbeddingProvider:
        """Select provider based on cost optimization."""
        min_cost = float('inf')
        selected_provider = None
        
        for provider_type in providers:
            provider = self.providers[provider_type]
            try:
                model = request.model or provider.config.model
                cost = provider.estimate_cost(100, model)  # Estimate for 100 tokens
                if cost < min_cost:
                    min_cost = cost
                    selected_provider = provider_type
            except Exception:
                continue
        
        return selected_provider or providers[0]
    
    def _generate_cache_key(self, request: EmbeddingRequest) -> str:
        """Generate cache key for request."""
        # Create deterministic key from request parameters
        key_data = {
            "input": request.input,
            "model": request.model,
            "dimensions": request.dimensions,
            "encoding_format": request.encoding_format,
            "normalize": request.normalize
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[EmbeddingResponse]:
        """Get embedding from cache."""
        if cache_key not in self.cache:
            return None
        
        cache_entry = self.cache[cache_key]
        
        if cache_entry.is_expired():
            del self.cache[cache_key]
            return None
        
        # Create response from cache
        response = EmbeddingResponse(
            embeddings=cache_entry.embeddings,
            model=cache_entry.model,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),  # Cached, no usage
            dimensions=cache_entry.dimensions,
            metadata={"cached": True, "cache_timestamp": cache_entry.timestamp.isoformat()}
        )
        
        return response
    
    def _save_to_cache(self, cache_key: str, response: EmbeddingResponse) -> None:
        """Save embedding to cache."""
        cache_entry = EmbeddingCache(
            embeddings=response.embeddings,
            model=response.model,
            dimensions=response.dimensions,
            timestamp=datetime.utcnow(),
            ttl=self.default_cache_ttl
        )
        
        self.cache[cache_key] = cache_entry
    
    def _calculate_cost(self, tokens: int, provider_type: EmbeddingProvider, model: str) -> float:
        """Calculate cost for embedding generation."""
        try:
            provider = self.providers[provider_type]
            return provider.estimate_cost(tokens, model)
        except Exception:
            return 0.0
    
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
                "total_tokens": stat.total_tokens,
                "total_cost": stat.total_cost,
                "is_healthy": stat.is_healthy,
                "consecutive_errors": stat.consecutive_errors,
                "last_request_time": stat.last_request_time.isoformat() if stat.last_request_time else None,
                "last_error": stat.last_error
            }
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "enabled": self.enable_cache,
            "total_entries": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "memory_usage": sum(len(str(entry.embeddings)) for entry in self.cache.values())
        }
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Embedding cache cleared")
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get supported models from all providers."""
        models = {}
        for provider_type, provider in self.providers.items():
            try:
                models[provider_type.value] = provider.get_supported_models()
            except Exception as e:
                self.logger.warning(f"Failed to get models for provider {provider_type.value}: {e}")
                models[provider_type.value] = []
        
        return models
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
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
        
        self.last_health_check = datetime.utcnow()
        return results
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information and status."""
        return {
            "total_providers": len(self.providers),
            "healthy_providers": sum(1 for stats in self.provider_stats.values() if stats.is_healthy),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "cache_enabled": self.enable_cache,
            "last_health_check": self.last_health_check.isoformat(),
            "provider_stats": self.get_provider_stats(),
            "cache_stats": self.get_cache_stats(),
            "supported_models": self.get_supported_models()
        }