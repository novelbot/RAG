"""
Anthropic Claude LLM Provider implementation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator
import json

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, Usage, ContentBlock, TextBlock
from anthropic.types.beta import BetaMessageParam
from anthropic.lib.streaming import MessageStream

from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk, 
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError


class ClaudeProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider implementation.
    
    Based on Context7 documentation for Anthropic Python SDK:
    - Supports async and sync operations with AsyncAnthropic/Anthropic
    - Implements streaming with client.messages.stream()
    - Handles rate limiting and retries
    - Provides token counting with count_tokens
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Claude provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        
        # Default models if not specified
        if not config.model:
            config.model = "claude-3-5-sonnet-latest"
        
        self.provider = LLMProvider.CLAUDE
    
    def _initialize_clients(self) -> None:
        """Initialize Anthropic sync and async clients."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }
        
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        # Initialize synchronous client
        self._client = Anthropic(**client_kwargs)
        
        # Initialize asynchronous client
        self._async_client = AsyncAnthropic(**client_kwargs)
        
        self.logger.info(f"Initialized Claude clients for model: {self.config.model}")
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously using Claude API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to Claude format
            messages = self._convert_messages_to_claude_format(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "max_tokens": request.max_tokens or 1024,
                "messages": messages,
                "temperature": request.temperature,
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                params["system"] = request.system_prompt
            
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make API call
            response = await self._async_client.messages.create(**params)
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_claude_response(response, response_time)
            
        except Exception as e:
            self.logger.error(f"Claude async generation failed: {e}")
            
            # Handle specific Claude errors
            if "rate_limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Claude rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Claude token limit exceeded: {e}")
            else:
                raise LLMError(f"Claude API error: {e}")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously using Claude API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to Claude format
            messages = self._convert_messages_to_claude_format(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "max_tokens": request.max_tokens or 1024,
                "messages": messages,
                "temperature": request.temperature,
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                params["system"] = request.system_prompt
            
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make API call
            response = self._client.messages.create(**params)
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_claude_response(response, response_time)
            
        except Exception as e:
            self.logger.error(f"Claude sync generation failed: {e}")
            
            # Handle specific Claude errors
            if "rate_limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Claude rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Claude token limit exceeded: {e}")
            else:
                raise LLMError(f"Claude API error: {e}")
    
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Generate streaming response asynchronously using Claude API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        # Ensure this is always treated as an async generator
        if False:
            yield  # This never executes but ensures the function is an async generator
        
        try:
            # Convert messages to Claude format
            messages = self._convert_messages_to_claude_format(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "max_tokens": request.max_tokens or 1024,
                "messages": messages,
                "temperature": request.temperature,
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                params["system"] = request.system_prompt
            
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make streaming API call using context manager
            async with self._async_client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield LLMStreamChunk(
                        content=text,
                        finish_reason=None,
                        metadata={"chunk_type": "text"}
                    )
                
                # Get final message for usage info
                final_message = await stream.get_final_message()
                
                if final_message.usage:
                    usage = LLMUsage(
                        prompt_tokens=final_message.usage.input_tokens,
                        completion_tokens=final_message.usage.output_tokens,
                        total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens
                    )
                    
                    yield LLMStreamChunk(
                        content="",
                        finish_reason=final_message.stop_reason,
                        usage=usage,
                        metadata={"final_chunk": True}
                    )
                    
        except Exception as e:
            self.logger.error(f"Claude async streaming failed: {e}")
            
            if "rate_limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Claude rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Claude token limit exceeded: {e}")
            else:
                raise LLMError(f"Claude streaming API error: {e}")
    
    def generate_stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """
        Generate streaming response synchronously using Claude API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        try:
            # Convert messages to Claude format
            messages = self._convert_messages_to_claude_format(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "max_tokens": request.max_tokens or 1024,
                "messages": messages,
                "temperature": request.temperature,
            }
            
            # Add system prompt if provided
            if request.system_prompt:
                params["system"] = request.system_prompt
            
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            if request.tools and self.config.enable_function_calling:
                params["tools"] = request.tools
            
            # Make streaming API call using context manager
            with self._client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield LLMStreamChunk(
                        content=text,
                        finish_reason=None,
                        metadata={"chunk_type": "text"}
                    )
                
                # Get final message for usage info
                final_message = stream.get_final_message()
                
                if final_message.usage:
                    usage = LLMUsage(
                        prompt_tokens=final_message.usage.input_tokens,
                        completion_tokens=final_message.usage.output_tokens,
                        total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens
                    )
                    
                    yield LLMStreamChunk(
                        content="",
                        finish_reason=final_message.stop_reason,
                        usage=usage,
                        metadata={"final_chunk": True}
                    )
                    
        except Exception as e:
            self.logger.error(f"Claude sync streaming failed: {e}")
            
            if "rate_limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Claude rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Claude token limit exceeded: {e}")
            else:
                raise LLMError(f"Claude streaming API error: {e}")
    
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
            # Convert messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Use Claude's count_tokens method
            # Cast to BetaMessageParam for type compatibility
            response = await self._async_client.beta.messages.count_tokens(
                model=model or self.config.model,
                messages=claude_messages  # type: ignore[arg-type]
            )
            
            return response.input_tokens
                
        except Exception as e:
            self.logger.error(f"Claude token counting failed: {e}")
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
            # Convert messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Use Claude's count_tokens method
            # Cast to BetaMessageParam for type compatibility
            response = self._client.beta.messages.count_tokens(
                model=model or self.config.model,
                messages=claude_messages  # type: ignore[arg-type]
            )
            
            return response.input_tokens
                
        except Exception as e:
            self.logger.error(f"Claude token counting failed: {e}")
            # Return approximate count based on character length
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4  # Rough approximation: 4 chars per token
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Claude models.
        
        Returns:
            List of model names
        """
        try:
            models = []
            for model in self._client.models.list():
                models.append(model.id)
            return models
        except Exception as e:
            self.logger.error(f"Failed to get Claude models: {e}")
            # Return common Claude models as fallback
            return [
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
                "claude-3-opus-latest",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate Claude provider configuration.
        
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
            validation_result["warnings"].append("No model specified, using default: claude-3-5-sonnet-latest")
        
        # Check temperature
        if not 0.0 <= self.config.temperature <= 1.0:
            validation_result["warnings"].append("Temperature should be between 0.0 and 1.0")
        
        # Check max_tokens
        if self.config.max_tokens and self.config.max_tokens <= 0:
            validation_result["issues"].append("max_tokens must be positive")
        
        # Test connection
        try:
            if self._client:
                models = list(self._client.models.list())
                validation_result["available_models"] = len(models)
        except Exception as e:
            validation_result["warnings"].append(f"Unable to connect to Claude API: {e}")
        
        return validation_result
    
    def _convert_messages_to_claude_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLM messages to Claude format."""
        claude_messages = []
        
        for msg in messages:
            # Map roles to Claude format
            claude_role = msg.role.value
            
            # Claude uses 'user' and 'assistant' roles
            if claude_role == "system":
                # System messages are handled separately in Claude
                continue
            elif claude_role == "function":
                claude_role = "assistant"
            elif claude_role == "tool":
                claude_role = "user"
            
            claude_messages.append({
                "role": claude_role,
                "content": msg.content
            })
        
        return claude_messages
    
    def _parse_claude_response(self, response: Message, response_time: float) -> LLMResponse:
        """Parse Claude response into standardized format."""
        # Extract content from response
        content = ""
        if response.content:
            # Claude returns content as a list of content blocks
            for block in response.content:
                # Only TextBlock has a text attribute
                if isinstance(block, TextBlock) and block.text:
                    content += block.text
                # For other block types, we could add additional handling here if needed
                # For now, we only extract text content
        
        # Extract usage information
        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider,
            finish_reason=response.stop_reason,
            usage=usage,
            response_time=response_time,
            request_id=response.id,
            metadata={
                "type": response.type,
                "role": response.role
            }
        )