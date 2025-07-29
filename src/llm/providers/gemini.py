"""
Google Gemini LLM Provider implementation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator
import json

from google import genai
from google.genai import types

from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk, 
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider implementation.
    
    Based on Context7 documentation for Google GenAI Python SDK:
    - Supports async and sync operations with genai.Client
    - Implements streaming with generate_content_stream
    - Handles rate limiting and retries
    - Provides token counting and model management
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Gemini provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        
        # Default models if not specified
        if not config.model:
            config.model = "gemini-2.0-flash-001"
        
        self.provider = LLMProvider.GEMINI
    
    def _initialize_clients(self) -> None:
        """Initialize Google GenAI sync and async clients."""
        client_kwargs = {
            "api_key": self.config.api_key,
        }
        
        # Configure HTTP options
        http_options = None
        if self.config.timeout or self.config.custom_headers:
            http_options = types.HttpOptions()
            if self.config.custom_headers:
                http_options.headers = self.config.custom_headers
        
        # Initialize synchronous client
        self._client = genai.Client(
            api_key=self.config.api_key,
            http_options=http_options
        )
        
        # Async client is accessed via client.aio
        self._async_client = self._client.aio
        
        self.logger.info(f"Initialized Gemini clients for model: {self.config.model}")
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously using Gemini API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(request.messages)
            
            # Add system prompt if provided
            system_instruction = request.system_prompt if request.system_prompt else None
            
            # Prepare generation configuration
            generation_config = types.GenerateContentConfig(
                temperature=request.temperature,
                system_instruction=system_instruction,
            )
            
            if request.max_tokens:
                generation_config.max_output_tokens = request.max_tokens
            
            if request.stop_sequences:
                generation_config.stop_sequences = request.stop_sequences
            
            # Make async API call
            response = await self._async_client.models.generate_content(
                model=request.model or self.config.model,
                contents=contents,
                config=generation_config
            )
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_gemini_response(response, response_time)
            
        except Exception as e:
            self.logger.error(f"Gemini async generation failed: {e}")
            
            # Handle specific Gemini errors
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Gemini token limit exceeded: {e}")
            else:
                raise LLMError(f"Gemini API error: {e}")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously using Gemini API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(request.messages)
            
            # Add system prompt if provided
            system_instruction = request.system_prompt if request.system_prompt else None
            
            # Prepare generation configuration
            generation_config = types.GenerateContentConfig(
                temperature=request.temperature,
                system_instruction=system_instruction,
            )
            
            if request.max_tokens:
                generation_config.max_output_tokens = request.max_tokens
            
            if request.stop_sequences:
                generation_config.stop_sequences = request.stop_sequences
            
            # Make sync API call
            response = self._client.models.generate_content(
                model=request.model or self.config.model,
                contents=contents,
                config=generation_config
            )
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_gemini_response(response, response_time)
            
        except Exception as e:
            self.logger.error(f"Gemini sync generation failed: {e}")
            
            # Handle specific Gemini errors
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Gemini token limit exceeded: {e}")
            else:
                raise LLMError(f"Gemini API error: {e}")
    
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Generate streaming response asynchronously using Gemini API.
        
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
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(request.messages)
            
            # Add system prompt if provided
            system_instruction = request.system_prompt if request.system_prompt else None
            
            # Prepare generation configuration
            generation_config = types.GenerateContentConfig(
                temperature=request.temperature,
                system_instruction=system_instruction,
            )
            
            if request.max_tokens:
                generation_config.max_output_tokens = request.max_tokens
            
            if request.stop_sequences:
                generation_config.stop_sequences = request.stop_sequences
            
            # Make streaming API call
            async for chunk in await self._async_client.models.generate_content_stream(
                model=request.model or self.config.model,
                contents=contents,
                config=generation_config
            ):
                if chunk.text:
                    yield LLMStreamChunk(
                        content=chunk.text,
                        finish_reason=None,
                        metadata={"chunk_type": "text"}
                    )
                
                # Handle final chunk with usage info
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = LLMUsage(
                        prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        completion_tokens=chunk.usage_metadata.candidates_token_count or 0,
                        total_tokens=chunk.usage_metadata.total_token_count or 0
                    )
                    
                    yield LLMStreamChunk(
                        content="",
                        finish_reason="stop",
                        usage=usage,
                        metadata={"final_chunk": True}
                    )
                    
        except Exception as e:
            self.logger.error(f"Gemini async streaming failed: {e}")
            
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Gemini token limit exceeded: {e}")
            else:
                raise LLMError(f"Gemini streaming API error: {e}")
    
    def generate_stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """
        Generate streaming response synchronously using Gemini API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        try:
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(request.messages)
            
            # Add system prompt if provided
            system_instruction = request.system_prompt if request.system_prompt else None
            
            # Prepare generation configuration
            generation_config = types.GenerateContentConfig(
                temperature=request.temperature,
                system_instruction=system_instruction,
            )
            
            if request.max_tokens:
                generation_config.max_output_tokens = request.max_tokens
            
            if request.stop_sequences:
                generation_config.stop_sequences = request.stop_sequences
            
            # Make streaming API call
            for chunk in self._client.models.generate_content_stream(
                model=request.model or self.config.model,
                contents=contents,
                config=generation_config
            ):
                if chunk.text:
                    yield LLMStreamChunk(
                        content=chunk.text,
                        finish_reason=None,
                        metadata={"chunk_type": "text"}
                    )
                
                # Handle final chunk with usage info
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = LLMUsage(
                        prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        completion_tokens=chunk.usage_metadata.candidates_token_count or 0,
                        total_tokens=chunk.usage_metadata.total_token_count or 0
                    )
                    
                    yield LLMStreamChunk(
                        content="",
                        finish_reason="stop",
                        usage=usage,
                        metadata={"final_chunk": True}
                    )
                    
        except Exception as e:
            self.logger.error(f"Gemini sync streaming failed: {e}")
            
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(f"Gemini token limit exceeded: {e}")
            else:
                raise LLMError(f"Gemini streaming API error: {e}")
    
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
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(messages)
            
            # Use Gemini's count_tokens method
            response = await self._async_client.models.count_tokens(
                model=model or self.config.model,
                contents=contents
            )
            
            return response.total_tokens or 0
                
        except Exception as e:
            self.logger.error(f"Gemini token counting failed: {e}")
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
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(messages)
            
            # Use Gemini's count_tokens method
            response = self._client.models.count_tokens(
                model=model or self.config.model,
                contents=contents
            )
            
            return response.total_tokens or 0
                
        except Exception as e:
            self.logger.error(f"Gemini token counting failed: {e}")
            # Return approximate count based on character length
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4  # Rough approximation: 4 chars per token
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Gemini models.
        
        Returns:
            List of model names
        """
        try:
            models = []
            for model in self._client.models.list():
                models.append(model.name)
            return models
        except Exception as e:
            self.logger.error(f"Failed to get Gemini models: {e}")
            # Return common Gemini models as fallback
            return [
                "gemini-2.0-flash-001",
                "gemini-1.5-pro-002",
                "gemini-1.5-flash-002",
                "gemini-1.0-pro"
            ]
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate Gemini provider configuration.
        
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
            validation_result["warnings"].append("No model specified, using default: gemini-2.0-flash-001")
        
        # Check temperature
        if not 0.0 <= self.config.temperature <= 2.0:
            validation_result["warnings"].append("Temperature should be between 0.0 and 2.0")
        
        # Check max_tokens
        if self.config.max_tokens and self.config.max_tokens <= 0:
            validation_result["issues"].append("max_tokens must be positive")
        
        # Test connection
        try:
            if self._client:
                models = list(self._client.models.list())
                validation_result["available_models"] = len(models)
        except Exception as e:
            validation_result["warnings"].append(f"Unable to connect to Gemini API: {e}")
        
        return validation_result
    
    def _convert_messages_to_gemini_format(self, messages: List[LLMMessage]) -> List[Any]:
        """Convert LLM messages to Gemini format."""
        gemini_contents = []
        
        for msg in messages:
            # Convert to types.Content format for proper typing
            if msg.role == LLMRole.USER:
                content = types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=msg.content)]
                )
            elif msg.role == LLMRole.ASSISTANT:
                content = types.Content(
                    role='model',
                    parts=[types.Part.from_text(text=msg.content)]
                )
            elif msg.role == LLMRole.SYSTEM:
                # System messages will be handled separately as system_instruction
                continue
            else:
                # Default to user role
                content = types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=msg.content)]
                )
            
            gemini_contents.append(content)
        
        return gemini_contents
    
    def _parse_gemini_response(self, response: Any, response_time: float) -> LLMResponse:
        """Parse Gemini response into standardized format."""
        # Extract content from response
        content = ""
        finish_reason = None
        
        if hasattr(response, 'text') and response.text:
            content = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Extract from first candidate
            if response.candidates[0].content and response.candidates[0].content.parts:
                content = response.candidates[0].content.parts[0].text
            if hasattr(response.candidates[0], 'finish_reason'):
                finish_reason = response.candidates[0].finish_reason
        
        # Extract usage information
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = LLMUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0
            )
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.provider,
            finish_reason=finish_reason,
            usage=usage,
            response_time=response_time,
            request_id=getattr(response, 'response_id', None),
            metadata={
                "model_version": getattr(response, 'model_version', None),
                "create_time": getattr(response, 'create_time', None)
            }
        )