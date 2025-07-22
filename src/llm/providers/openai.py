"""
OpenAI LLM Provider implementation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator
import json

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk, 
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Based on Context7 documentation for OpenAI Python SDK:
    - Supports async and sync operations
    - Implements streaming with proper chunk handling
    - Handles rate limiting and retries
    - Provides token counting and model management
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        
        # Default models if not specified
        if not config.model:
            config.model = "gpt-4"
        
        self.provider = LLMProvider.OPENAI
    
    def _initialize_clients(self) -> None:
        """Initialize OpenAI sync and async clients."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }
        
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        if self.config.custom_headers:
            client_kwargs["default_headers"] = self.config.custom_headers
        
        # Initialize synchronous client
        self._client = OpenAI(**client_kwargs)
        
        # Initialize asynchronous client
        self._async_client = AsyncOpenAI(**client_kwargs)
        
        self.logger.info(f"Initialized OpenAI clients for model: {self.config.model}")
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously using OpenAI API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            messages = self._convert_messages_to_openai_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "temperature": request.temperature,
                "stream": False,
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make API call
            response = await self._async_client.chat.completions.create(**params)
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_openai_response(response, response_time)
            
        except Exception as e:
            self.logger.error(f"OpenAI async generation failed: {e}")
            
            # Handle specific OpenAI errors
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"OpenAI token limit exceeded: {e}")
            else:
                raise LLMError(f"OpenAI API error: {e}")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously using OpenAI API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            messages = self._convert_messages_to_openai_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "temperature": request.temperature,
                "stream": False,
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make API call
            response = self._client.chat.completions.create(**params)
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_openai_response(response, response_time)
            
        except Exception as e:
            self.logger.error(f"OpenAI sync generation failed: {e}")
            
            # Handle specific OpenAI errors
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"OpenAI token limit exceeded: {e}")
            else:
                raise LLMError(f"OpenAI API error: {e}")
    
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Generate streaming response asynchronously using OpenAI API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        try:
            # Convert messages to OpenAI format
            messages = self._convert_messages_to_openai_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "temperature": request.temperature,
                "stream": True,
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make streaming API call
            async with self._async_client.chat.completions.stream(**params) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        yield LLMStreamChunk(
                            content=chunk.choices[0].delta.content,
                            finish_reason=chunk.choices[0].finish_reason,
                            metadata={"chunk_id": chunk.id}
                        )
                    
                    # Handle final chunk with usage
                    if chunk.choices and chunk.choices[0].finish_reason:
                        usage = None
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage = LLMUsage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                                total_tokens=chunk.usage.total_tokens
                            )
                        
                        yield LLMStreamChunk(
                            content="",
                            finish_reason=chunk.choices[0].finish_reason,
                            usage=usage,
                            metadata={"final_chunk": True}
                        )
                        
        except Exception as e:
            self.logger.error(f"OpenAI async streaming failed: {e}")
            
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"OpenAI token limit exceeded: {e}")
            else:
                raise LLMError(f"OpenAI streaming API error: {e}")
    
    def generate_stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """
        Generate streaming response synchronously using OpenAI API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        try:
            # Convert messages to OpenAI format
            messages = self._convert_messages_to_openai_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "temperature": request.temperature,
                "stream": True,
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make streaming API call
            with self._client.chat.completions.stream(**params) as stream:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        yield LLMStreamChunk(
                            content=chunk.choices[0].delta.content,
                            finish_reason=chunk.choices[0].finish_reason,
                            metadata={"chunk_id": chunk.id}
                        )
                    
                    # Handle final chunk with usage
                    if chunk.choices and chunk.choices[0].finish_reason:
                        usage = None
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage = LLMUsage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                                total_tokens=chunk.usage.total_tokens
                            )
                        
                        yield LLMStreamChunk(
                            content="",
                            finish_reason=chunk.choices[0].finish_reason,
                            usage=usage,
                            metadata={"final_chunk": True}
                        )
                        
        except Exception as e:
            self.logger.error(f"OpenAI sync streaming failed: {e}")
            
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"OpenAI token limit exceeded: {e}")
            else:
                raise LLMError(f"OpenAI streaming API error: {e}")
    
    async def count_tokens_async(self, messages: List[LLMMessage], model: str) -> int:
        """
        Count tokens in messages asynchronously.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Token count
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            # For OpenAI, we can use tiktoken for approximate counting
            # or make a minimal API call with max_tokens=1
            test_params = {
                "model": model or self.config.model,
                "messages": openai_messages,
                "max_tokens": 1,
                "temperature": 0.0
            }
            
            response = await self._async_client.chat.completions.create(**test_params)
            
            if response.usage:
                return response.usage.prompt_tokens
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"OpenAI token counting failed: {e}")
            # Return approximate count based on character length
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
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            # For OpenAI, we can use tiktoken for approximate counting
            # or make a minimal API call with max_tokens=1
            test_params = {
                "model": model or self.config.model,
                "messages": openai_messages,
                "max_tokens": 1,
                "temperature": 0.0
            }
            
            response = self._client.chat.completions.create(**test_params)
            
            if response.usage:
                return response.usage.prompt_tokens
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"OpenAI token counting failed: {e}")
            # Return approximate count based on character length
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4  # Rough approximation: 4 chars per token
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models.
        
        Returns:
            List of model names
        """
        try:
            models = self._client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Failed to get OpenAI models: {e}")
            # Return common OpenAI models as fallback
            return [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ]
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate OpenAI provider configuration.
        
        Returns:
            Validation result
        """
        validation_result = {
            "status": "valid",
            "issues": [],
            "warnings": []
        }
        
        # Check API key
        if not self.config.api_key:
            validation_result["status"] = "invalid"
            validation_result["issues"].append("API key is required")
        
        # Check model
        if not self.config.model:
            validation_result["warnings"].append("No model specified, using default: gpt-4")
        
        # Check temperature
        if not 0.0 <= self.config.temperature <= 2.0:
            validation_result["warnings"].append("Temperature should be between 0.0 and 2.0")
        
        # Check max_tokens
        if self.config.max_tokens and self.config.max_tokens <= 0:
            validation_result["issues"].append("max_tokens must be positive")
        
        # Test connection
        try:
            if self._client:
                models = self._client.models.list()
                validation_result["available_models"] = len(models.data)
        except Exception as e:
            validation_result["warnings"].append(f"Unable to connect to OpenAI API: {e}")
        
        return validation_result
    
    def _convert_messages_to_openai_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLM messages to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            openai_role = msg.role.value
            
            # Convert role if needed
            if openai_role == "function":
                openai_role = "assistant"
            elif openai_role == "tool":
                openai_role = "tool"
            
            openai_messages.append({
                "role": openai_role,
                "content": msg.content
            })
        
        return openai_messages
    
    def _parse_openai_response(self, response: ChatCompletion, response_time: float) -> LLMResponse:
        """Parse OpenAI response into standardized format."""
        # Extract content from first choice
        content = ""
        finish_reason = None
        
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                content = choice.message.content
            finish_reason = choice.finish_reason
        
        # Extract usage information
        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider,
            finish_reason=finish_reason,
            usage=usage,
            response_time=response_time,
            request_id=response.id,
            metadata={
                "system_fingerprint": getattr(response, 'system_fingerprint', None),
                "created": response.created
            }
        )