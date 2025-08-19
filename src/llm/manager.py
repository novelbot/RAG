"""
LLM Manager with LangChain Integration.

This module provides a unified manager for multiple LLM providers using LangChain,
with load balancing, failover support, and backward compatibility.
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Iterator
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration

# Local imports
from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk,
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.llm.providers.ollama import OllamaProvider  # Keep for special features
from src.core.logging import LoggerMixin
from src.core.exceptions import LLMError, RateLimitError


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
        
        if self.consecutive_errors >= 5:
            self.is_healthy = False


@dataclass
class ProviderConfig:
    """Configuration for a provider in the manager."""
    provider: LLMProvider
    config: LLMConfig
    priority: int = 1
    max_requests_per_minute: int = 60
    enabled: bool = True
    weight: float = 1.0


class LangChainProviderAdapter(BaseLLMProvider):
    """Adapter to make LangChain models compatible with our BaseLLMProvider interface."""
    
    def __init__(self, langchain_model: BaseChatModel, config: LLMConfig):
        """Initialize adapter with LangChain model."""
        super().__init__(config)
        self.langchain_model = langchain_model
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize sync and async clients - handled by LangChain model."""
        pass  # LangChain models handle their own client initialization
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """Generate response using LangChain model."""
        start_time = time.time()
        
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                langchain_messages.insert(0, SystemMessage(content=request.system_prompt))
            
            # Generate response
            response = await self.langchain_model.ainvoke(langchain_messages)
            
            # Convert response
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content,
                model=self.config.model,
                provider=self.config.provider,
                usage=LLMUsage(
                    prompt_tokens=len(str(langchain_messages).split()),
                    completion_tokens=len(response.content.split()),
                    total_tokens=len(str(langchain_messages).split()) + len(response.content.split())
                ),
                response_time=response_time,
                finish_reason="stop"
            )
            
        except Exception as e:
            self.logger.error(f"LangChain generation failed: {e}")
            raise LLMError(f"Generation failed: {e}")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response synchronously."""
        return asyncio.run(self.generate_async(request))
    
    async def stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream response using LangChain model."""
        try:
            langchain_messages = self._convert_messages(request.messages)
            
            if request.system_prompt:
                langchain_messages.insert(0, SystemMessage(content=request.system_prompt))
            
            async for chunk in self.langchain_model.astream(langchain_messages):
                yield LLMStreamChunk(
                    content=chunk.content,
                    finish_reason=None
                )
                
        except Exception as e:
            self.logger.error(f"LangChain streaming failed: {e}")
            raise LLMError(f"Streaming failed: {e}")
    
    def _convert_messages(self, messages: List[LLMMessage]) -> List:
        """Convert our message format to LangChain format."""
        langchain_messages = []
        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == LLMRole.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == LLMRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
        return langchain_messages
    
    def generate_stream(self, request: LLMRequest):
        """Generate streaming response synchronously."""
        # For synchronous streaming, we'll need to use threading to bridge async
        import asyncio
        import threading
        from queue import Queue
        
        q = Queue()
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _stream():
                try:
                    async for chunk in self.stream_async(request):
                        q.put(chunk)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(None)
            
            loop.run_until_complete(_stream())
            loop.close()
        
        thread = threading.Thread(target=run_async)
        thread.start()
        
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        
        thread.join()
    
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming response asynchronously - alias for stream_async."""
        async for chunk in self.stream_async(request):
            yield chunk
    
    async def count_tokens_async(self, messages: List[LLMMessage], model: str) -> int:
        """Count tokens in messages asynchronously."""
        # Simple token estimation for LangChain models
        total_text = " ".join([msg.content for msg in messages])
        return int(len(total_text.split()) * 1.3)  # Rough estimate
    
    def count_tokens(self, messages: List[LLMMessage], model: str) -> int:
        """Count tokens in messages synchronously."""
        # Simple token estimation for LangChain models
        total_text = " ".join([msg.content for msg in messages])
        return int(len(total_text.split()) * 1.3)  # Rough estimate
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        # Return the configured model
        return [self.config.model]
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate provider configuration."""
        return {
            "valid": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "has_api_key": bool(self.config.api_key)
        }


class LangChainLLMManager(LoggerMixin):
    """
    LLM Manager using LangChain providers with load balancing and failover.
    
    Features:
    - LangChain integration for all major providers
    - Backward compatibility with existing code
    - Load balancing strategies
    - Automatic failover and retry logic
    - Health monitoring
    """
    
    def __init__(self, provider_configs: List[ProviderConfig]):
        """Initialize manager with provider configurations."""
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.provider_stats: Dict[LLMProvider, ProviderStats] = {}
        self.provider_configs = {pc.provider: pc for pc in provider_configs}
        
        # Load balancing
        self.strategy = LoadBalancingStrategy.ROUND_ROBIN
        self.current_index = 0
        
        # Initialize providers
        for config in provider_configs:
            if config.enabled:
                self._initialize_provider(config)
    
    def _initialize_provider(self, provider_config: ProviderConfig):
        """Initialize a single provider using LangChain."""
        provider = provider_config.provider
        config = provider_config.config
        
        try:
            if provider == LLMProvider.OPENAI:
                langchain_model = ChatOpenAI(
                    model=config.model or "gpt-3.5-turbo",
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    streaming=getattr(config, 'stream', True)  # Default to True if not present
                )
                self.providers[provider] = LangChainProviderAdapter(langchain_model, config)
                
            elif provider == LLMProvider.GOOGLE:
                langchain_model = ChatGoogleGenerativeAI(
                    model=config.model or "gemini-pro",
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                    google_api_key=config.api_key,
                    streaming=getattr(config, 'stream', True)  # Default to True if not present
                )
                self.providers[provider] = LangChainProviderAdapter(langchain_model, config)
                
            elif provider == LLMProvider.CLAUDE:
                langchain_model = ChatAnthropic(
                    model=config.model or "claude-3-sonnet-20240229",
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=config.api_key,
                    streaming=getattr(config, 'stream', True)  # Default to True if not present
                )
                self.providers[provider] = LangChainProviderAdapter(langchain_model, config)
                
            elif provider == LLMProvider.OLLAMA:
                # Use original Ollama provider for special features
                self.providers[provider] = OllamaProvider(config)
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Initialize stats
            self.provider_stats[provider] = ProviderStats()
            self.logger.info(f"Initialized LangChain provider: {provider.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize provider {provider.value}: {e}")
            raise LLMError(f"Provider initialization failed: {e}")
    
    def get_provider(self, preferred_provider: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get a provider based on strategy and health."""
        if preferred_provider and preferred_provider in self.providers:
            stats = self.provider_stats[preferred_provider]
            if stats.is_healthy:
                return self.providers[preferred_provider]
        
        # Get healthy providers
        healthy_providers = [
            p for p, stats in self.provider_stats.items()
            if stats.is_healthy
        ]
        
        if not healthy_providers:
            # Try to reset unhealthy providers
            for stats in self.provider_stats.values():
                if stats.consecutive_errors < 10:
                    stats.is_healthy = True
            healthy_providers = list(self.providers.keys())
        
        if not healthy_providers:
            raise LLMError("No healthy providers available")
        
        # Select based on strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            provider = healthy_providers[self.current_index % len(healthy_providers)]
            self.current_index += 1
            
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            provider = random.choice(healthy_providers)
            
        elif self.strategy == LoadBalancingStrategy.LEAST_USED:
            provider = min(healthy_providers, 
                          key=lambda p: self.provider_stats[p].total_requests)
            
        elif self.strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            provider = min(healthy_providers,
                          key=lambda p: self.provider_stats[p].average_response_time or float('inf'))
            
        else:  # HEALTH_BASED
            provider = max(healthy_providers,
                          key=lambda p: self.provider_stats[p].get_success_rate())
        
        return self.providers[provider]
    
    async def generate_async(
        self,
        request: LLMRequest,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """Generate response with automatic failover."""
        errors = []
        
        # Try preferred provider first
        if preferred_provider:
            try:
                provider = self.providers.get(preferred_provider)
                if provider:
                    start_time = time.time()
                    response = await provider.generate_async(request)
                    self.provider_stats[preferred_provider].update_success(
                        time.time() - start_time
                    )
                    return response
            except Exception as e:
                errors.append((preferred_provider, str(e)))
                self.provider_stats[preferred_provider].update_failure(str(e))
        
        # Try other providers
        for provider_enum in self.providers:
            if provider_enum == preferred_provider:
                continue
                
            try:
                provider = self.providers[provider_enum]
                start_time = time.time()
                response = await provider.generate_async(request)
                self.provider_stats[provider_enum].update_success(
                    time.time() - start_time
                )
                return response
                
            except Exception as e:
                errors.append((provider_enum, str(e)))
                self.provider_stats[provider_enum].update_failure(str(e))
        
        # All providers failed
        error_msg = "All providers failed:\n"
        for provider, error in errors:
            error_msg += f"  {provider.value}: {error}\n"
        raise LLMError(error_msg)
    
    def generate(
        self,
        request: LLMRequest,
        preferred_provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """Synchronous wrapper for generate_async."""
        return asyncio.run(self.generate_async(request, preferred_provider))
    
    async def stream_async(
        self,
        request: LLMRequest,
        preferred_provider: Optional[LLMProvider] = None
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream response with automatic failover."""
        errors = []
        
        # Try preferred provider first
        if preferred_provider:
            try:
                provider = self.providers.get(preferred_provider)
                if provider:
                    start_time = time.time()
                    async for chunk in provider.stream_async(request):
                        yield chunk
                    self.provider_stats[preferred_provider].update_success(
                        time.time() - start_time
                    )
                    return
            except Exception as e:
                errors.append((preferred_provider, str(e)))
                self.provider_stats[preferred_provider].update_failure(str(e))
        
        # Try other providers
        for provider_enum in self.providers:
            if provider_enum == preferred_provider:
                continue
                
            try:
                provider = self.providers[provider_enum]
                start_time = time.time()
                async for chunk in provider.stream_async(request):
                    yield chunk
                self.provider_stats[provider_enum].update_success(
                    time.time() - start_time
                )
                return
                
            except Exception as e:
                errors.append((provider_enum, str(e)))
                self.provider_stats[provider_enum].update_failure(str(e))
        
        # All providers failed
        error_msg = "All providers failed:\n"
        for provider, error in errors:
            error_msg += f"  {provider.value}: {error}\n"
        raise LLMError(error_msg)
    
    async def generate_stream_async(
        self,
        request: LLMRequest,
        preferred_provider: Optional[LLMProvider] = None
    ) -> AsyncIterator[LLMStreamChunk]:
        """Alias for stream_async for backward compatibility."""
        async for chunk in self.stream_async(request, preferred_provider):
            yield chunk
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        stats = {}
        for provider, provider_stats in self.provider_stats.items():
            stats[provider.value] = {
                "total_requests": provider_stats.total_requests,
                "successful_requests": provider_stats.successful_requests,
                "failed_requests": provider_stats.failed_requests,
                "average_response_time": provider_stats.average_response_time,
                "success_rate": provider_stats.get_success_rate(),
                "is_healthy": provider_stats.is_healthy,
                "last_error": provider_stats.last_error
            }
        return stats
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Set the load balancing strategy."""
        self.strategy = strategy
        self.logger.info(f"Load balancing strategy set to: {strategy.value}")


# Factory function for backward compatibility
def create_llm_manager(provider_configs: List[ProviderConfig]) -> LangChainLLMManager:
    """Create LLM manager with LangChain providers."""
    return LangChainLLMManager(provider_configs)