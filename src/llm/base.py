"""
Abstract LLM Provider Interface - Base classes for all LLM implementations.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from src.core.logging import LoggerMixin
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError


class LLMRole(Enum):
    """Message roles in LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"


@dataclass
class LLMMessage:
    """Standardized message format for LLM conversations."""
    role: LLMRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessage":
        """Create message from dictionary."""
        return cls(
            role=LLMRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
        )


@dataclass
class LLMRequest:
    """Standardized request format for LLM providers."""
    messages: List[LLMMessage]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary format."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "user_id": self.user_id,
            "metadata": self.metadata
        }


@dataclass
class LLMUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert usage to dictionary format."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class LLMResponse:
    """Standardized response format for LLM providers."""
    content: str
    model: str
    provider: LLMProvider
    finish_reason: Optional[str] = None
    usage: Optional[LLMUsage] = None
    response_time: float = 0.0
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider.value,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "response_time": self.response_time,
            "request_id": self.request_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LLMStreamChunk:
    """Standardized streaming chunk format."""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[LLMUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "metadata": self.metadata
        }


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 100
    rate_limit_tokens: int = 10000
    rate_limit_window: int = 60
    enable_streaming: bool = True
    enable_function_calling: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    proxy: Optional[str] = None
    verify_ssl: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            "provider": self.provider.value,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_tokens": self.rate_limit_tokens,
            "rate_limit_window": self.rate_limit_window,
            "enable_streaming": self.enable_streaming,
            "enable_function_calling": self.enable_function_calling,
            "custom_headers": self.custom_headers,
            "proxy": self.proxy,
            "verify_ssl": self.verify_ssl,
            "metadata": self.metadata
        }


class BaseLLMProvider(ABC, LoggerMixin):
    """
    Abstract base class for all LLM providers.
    
    Based on Context7 documentation for various LLM APIs:
    - Provides unified interface for OpenAI, Gemini, Claude, and Ollama
    - Implements async request handling and streaming
    - Handles rate limiting, retries, and error management
    - Supports function calling and tool usage
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.provider = config.provider
        self._client = None
        self._async_client = None
        self._rate_limiter = None
        self._health_status = True
        self._last_health_check = datetime.utcnow()
        
        # Initialize clients
        self._initialize_clients()
    
    @abstractmethod
    def _initialize_clients(self) -> None:
        """Initialize sync and async clients."""
        pass
    
    @abstractmethod
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Generate streaming response asynchronously.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        pass
    
    @abstractmethod
    def generate_stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """
        Generate streaming response synchronously.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        pass
    
    @abstractmethod
    async def count_tokens_async(self, messages: List[LLMMessage], model: str) -> int:
        """
        Count tokens in messages asynchronously.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Token count
        """
        pass
    
    @abstractmethod
    def count_tokens(self, messages: List[LLMMessage], model: str) -> int:
        """
        Count tokens in messages synchronously.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Token count
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate provider configuration.
        
        Returns:
            Validation result
        """
        pass
    
    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform health check asynchronously.
        
        Returns:
            Health check result
        """
        try:
            start_time = time.time()
            
            # Simple health check with minimal request
            test_request = LLMRequest(
                messages=[LLMMessage(
                    role=LLMRole.USER,
                    content="Hello"
                )],
                model=self.config.model,
                max_tokens=1,
                temperature=0.0
            )
            
            # Test basic functionality
            response = await self.generate_async(test_request)
            
            response_time = time.time() - start_time
            
            self._health_status = True
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "provider": self.provider.value,
                "model": self.config.model,
                "response_time": response_time,
                "last_check": self._last_health_check.isoformat(),
                "test_response": response.content[:50] if response.content else ""
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed for {self.provider.value}: {e}")
            self._health_status = False
            self._last_health_check = datetime.utcnow()
            
            return {
                "status": "unhealthy",
                "provider": self.provider.value,
                "model": self.config.model,
                "error": str(e),
                "last_check": self._last_health_check.isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check synchronously.
        
        Returns:
            Health check result
        """
        try:
            return asyncio.run(self.health_check_async())
        except Exception as e:
            self.logger.error(f"Health check failed for {self.provider.value}: {e}")
            return {
                "status": "unhealthy",
                "provider": self.provider.value,
                "error": str(e)
            }
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self._health_status
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": self.provider.value,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "streaming_enabled": self.config.enable_streaming,
            "function_calling_enabled": self.config.enable_function_calling,
            "rate_limits": {
                "requests": self.config.rate_limit_requests,
                "tokens": self.config.rate_limit_tokens,
                "window": self.config.rate_limit_window
            },
            "health_status": self._health_status,
            "last_health_check": self._last_health_check.isoformat()
        }
    
    def _handle_rate_limit(self, retry_count: int) -> None:
        """Handle rate limiting with exponential backoff."""
        if retry_count > self.config.max_retries:
            raise RateLimitError(f"Rate limit exceeded for {self.provider.value}")
        
        delay = self.config.retry_delay * (2 ** retry_count)
        self.logger.warning(f"Rate limited, retrying in {delay}s (attempt {retry_count + 1})")
        time.sleep(delay)
    
    def _validate_request(self, request: LLMRequest) -> None:
        """Validate request parameters."""
        if not request.messages:
            raise LLMError("No messages provided")
        
        if not request.model:
            raise LLMError("No model specified")
        
        # Check token limits
        if self.config.max_tokens and request.max_tokens:
            if request.max_tokens > self.config.max_tokens:
                raise TokenLimitError(f"Requested tokens {request.max_tokens} exceed limit {self.config.max_tokens}")
    
    def _convert_messages_to_provider_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert standardized messages to provider-specific format."""
        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in messages
        ]
    
    def _parse_provider_response(self, response: Any) -> LLMResponse:
        """Parse provider response into standardized format."""
        # Default implementation - should be overridden by providers
        return LLMResponse(
            content=str(response),
            model=self.config.model,
            provider=self.provider
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self) -> None:
        """Close clients and cleanup resources."""
        if hasattr(self, '_client') and self._client:
            if hasattr(self._client, 'close'):
                self._client.close()
        
        if hasattr(self, '_async_client') and self._async_client:
            if hasattr(self._async_client, 'close'):
                try:
                    asyncio.run(self._async_client.close())
                except:
                    pass
    
    def __del__(self):
        """Cleanup on object deletion."""
        try:
            self.close()
        except:
            pass