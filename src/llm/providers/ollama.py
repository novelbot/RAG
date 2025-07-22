"""
Ollama LLM Provider implementation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator
import json

import ollama
from ollama import Client, AsyncClient, ChatResponse, ResponseError

from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk, 
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider implementation.
    
    Based on Context7 documentation for Ollama Python SDK:
    - Supports async and sync operations with AsyncClient/Client
    - Implements streaming with stream=True parameter
    - Handles rate limiting and retries
    - Provides model management and local inference
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        
        # Default models if not specified
        if not config.model:
            config.model = "llama3.2"
        
        self.provider = LLMProvider.OLLAMA
    
    def _initialize_clients(self) -> None:
        """Initialize Ollama sync and async clients."""
        client_kwargs = {
            "host": self.config.base_url or "http://localhost:11434",
        }
        
        if self.config.custom_headers:
            client_kwargs["headers"] = self.config.custom_headers
        
        # Initialize synchronous client
        self._client = Client(**client_kwargs)
        
        # Initialize asynchronous client
        self._async_client = AsyncClient(**client_kwargs)
        
        self.logger.info(f"Initialized Ollama clients for model: {self.config.model}")
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously using Ollama API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to Ollama format
            messages = self._convert_messages_to_ollama_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "stream": False,
            }
            
            # Add optional parameters
            if request.temperature is not None:
                params["temperature"] = request.temperature
            
            # Note: Ollama doesn't have a direct max_tokens parameter
            # It uses other parameters like num_predict
            if request.max_tokens:
                params["options"] = {"num_predict": request.max_tokens}
            
            if request.stop_sequences:
                if "options" not in params:
                    params["options"] = {}
                params["options"]["stop"] = request.stop_sequences
            
            # Make API call
            response = await self._async_client.chat(**params)
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_ollama_response(response, response_time)
            
        except ResponseError as e:
            self.logger.error(f"Ollama async generation failed: {e}")
            
            # Handle specific Ollama errors
            if e.status_code == 404:
                raise LLMError(f"Model not found: {request.model or self.config.model}")
            elif e.status_code == 429:
                raise RateLimitError(f"Ollama rate limit exceeded: {e}")
            else:
                raise LLMError(f"Ollama API error: {e}")
        except Exception as e:
            self.logger.error(f"Ollama async generation failed: {e}")
            raise LLMError(f"Ollama API error: {e}")
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response synchronously using Ollama API.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        self._validate_request(request)
        
        start_time = time.time()
        
        try:
            # Convert messages to Ollama format
            messages = self._convert_messages_to_ollama_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "stream": False,
            }
            
            # Add optional parameters
            if request.temperature is not None:
                params["temperature"] = request.temperature
            
            # Note: Ollama doesn't have a direct max_tokens parameter
            # It uses other parameters like num_predict
            if request.max_tokens:
                params["options"] = {"num_predict": request.max_tokens}
            
            if request.stop_sequences:
                if "options" not in params:
                    params["options"] = {}
                params["options"]["stop"] = request.stop_sequences
            
            # Make API call
            response = self._client.chat(**params)
            
            # Parse response
            response_time = time.time() - start_time
            
            return self._parse_ollama_response(response, response_time)
            
        except ResponseError as e:
            self.logger.error(f"Ollama sync generation failed: {e}")
            
            # Handle specific Ollama errors
            if e.status_code == 404:
                raise LLMError(f"Model not found: {request.model or self.config.model}")
            elif e.status_code == 429:
                raise RateLimitError(f"Ollama rate limit exceeded: {e}")
            else:
                raise LLMError(f"Ollama API error: {e}")
        except Exception as e:
            self.logger.error(f"Ollama sync generation failed: {e}")
            raise LLMError(f"Ollama API error: {e}")
    
    async def generate_stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Generate streaming response asynchronously using Ollama API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        try:
            # Convert messages to Ollama format
            messages = self._convert_messages_to_ollama_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "stream": True,
            }
            
            # Add optional parameters
            if request.temperature is not None:
                params["temperature"] = request.temperature
            
            if request.max_tokens:
                params["options"] = {"num_predict": request.max_tokens}
            
            if request.stop_sequences:
                if "options" not in params:
                    params["options"] = {}
                params["options"]["stop"] = request.stop_sequences
            
            # Make streaming API call
            async for chunk in await self._async_client.chat(**params):
                if chunk.get("message", {}).get("content"):
                    yield LLMStreamChunk(
                        content=chunk["message"]["content"],
                        finish_reason=None,
                        metadata={"chunk_type": "text"}
                    )
                
                # Handle final chunk
                if chunk.get("done", False):
                    # Ollama provides some usage stats in the final chunk
                    usage = None
                    if "prompt_eval_count" in chunk and "eval_count" in chunk:
                        usage = LLMUsage(
                            prompt_tokens=chunk.get("prompt_eval_count", 0),
                            completion_tokens=chunk.get("eval_count", 0),
                            total_tokens=chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                        )
                    
                    yield LLMStreamChunk(
                        content="",
                        finish_reason="stop",
                        usage=usage,
                        metadata={"final_chunk": True}
                    )
                    
        except ResponseError as e:
            self.logger.error(f"Ollama async streaming failed: {e}")
            
            if e.status_code == 404:
                raise LLMError(f"Model not found: {request.model or self.config.model}")
            elif e.status_code == 429:
                raise RateLimitError(f"Ollama rate limit exceeded: {e}")
            else:
                raise LLMError(f"Ollama streaming API error: {e}")
        except Exception as e:
            self.logger.error(f"Ollama async streaming failed: {e}")
            raise LLMError(f"Ollama streaming API error: {e}")
    
    def generate_stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """
        Generate streaming response synchronously using Ollama API.
        
        Args:
            request: LLM request
            
        Yields:
            LLM stream chunks
        """
        self._validate_request(request)
        
        if not self.config.enable_streaming:
            raise LLMError("Streaming is not enabled for this provider")
        
        try:
            # Convert messages to Ollama format
            messages = self._convert_messages_to_ollama_format(request.messages)
            
            # Add system prompt if provided
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model,
                "messages": messages,
                "stream": True,
            }
            
            # Add optional parameters
            if request.temperature is not None:
                params["temperature"] = request.temperature
            
            if request.max_tokens:
                params["options"] = {"num_predict": request.max_tokens}
            
            if request.stop_sequences:
                if "options" not in params:
                    params["options"] = {}
                params["options"]["stop"] = request.stop_sequences
            
            # Make streaming API call
            for chunk in self._client.chat(**params):
                if chunk.get("message", {}).get("content"):
                    yield LLMStreamChunk(
                        content=chunk["message"]["content"],
                        finish_reason=None,
                        metadata={"chunk_type": "text"}
                    )
                
                # Handle final chunk
                if chunk.get("done", False):
                    # Ollama provides some usage stats in the final chunk
                    usage = None
                    if "prompt_eval_count" in chunk and "eval_count" in chunk:
                        usage = LLMUsage(
                            prompt_tokens=chunk.get("prompt_eval_count", 0),
                            completion_tokens=chunk.get("eval_count", 0),
                            total_tokens=chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                        )
                    
                    yield LLMStreamChunk(
                        content="",
                        finish_reason="stop",
                        usage=usage,
                        metadata={"final_chunk": True}
                    )
                    
        except ResponseError as e:
            self.logger.error(f"Ollama sync streaming failed: {e}")
            
            if e.status_code == 404:
                raise LLMError(f"Model not found: {request.model or self.config.model}")
            elif e.status_code == 429:
                raise RateLimitError(f"Ollama rate limit exceeded: {e}")
            else:
                raise LLMError(f"Ollama streaming API error: {e}")
        except Exception as e:
            self.logger.error(f"Ollama sync streaming failed: {e}")
            raise LLMError(f"Ollama streaming API error: {e}")
    
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
            # Ollama doesn't have a direct token counting API
            # We'll use a rough approximation based on character count
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4  # Rough approximation: 4 chars per token
        except Exception as e:
            self.logger.error(f"Ollama token counting failed: {e}")
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
            # Ollama doesn't have a direct token counting API
            # We'll use a rough approximation based on character count
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4  # Rough approximation: 4 chars per token
        except Exception as e:
            self.logger.error(f"Ollama token counting failed: {e}")
            # Return approximate count based on character length
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4  # Rough approximation: 4 chars per token
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            models = self._client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            self.logger.error(f"Failed to get Ollama models: {e}")
            # Return common Ollama models as fallback
            return [
                "llama3.2",
                "llama3.1",
                "phi3",
                "mistral",
                "codellama",
                "gemma2"
            ]
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate Ollama provider configuration.
        
        Returns:
            Validation result
        """
        validation_result = {
            "status": "valid",
            "issues": [],
            "warnings": []
        }
        
        # Check model
        if not self.config.model:
            validation_result["warnings"].append("No model specified, using default: llama3.2")
        
        # Check temperature
        if not 0.0 <= self.config.temperature <= 2.0:
            validation_result["warnings"].append("Temperature should be between 0.0 and 2.0")
        
        # Check max_tokens
        if self.config.max_tokens and self.config.max_tokens <= 0:
            validation_result["issues"].append("max_tokens must be positive")
        
        # Test connection
        try:
            if self._client:
                models = self._client.list()
                validation_result["available_models"] = len(models.get('models', []))
        except Exception as e:
            validation_result["warnings"].append(f"Unable to connect to Ollama API: {e}")
        
        return validation_result
    
    def _convert_messages_to_ollama_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLM messages to Ollama format."""
        ollama_messages = []
        
        for msg in messages:
            # Map roles to Ollama format
            ollama_role = msg.role.value
            
            # Convert role if needed
            if ollama_role == "function":
                ollama_role = "assistant"
            elif ollama_role == "tool":
                ollama_role = "user"
            
            ollama_messages.append({
                "role": ollama_role,
                "content": msg.content
            })
        
        return ollama_messages
    
    def _parse_ollama_response(self, response: ChatResponse, response_time: float) -> LLMResponse:
        """Parse Ollama response into standardized format."""
        # Extract content from response
        content = ""
        if hasattr(response, 'message') and response.message:
            content = response.message.content
        elif isinstance(response, dict) and 'message' in response:
            content = response['message'].get('content', '')
        
        # Extract usage information if available
        usage = None
        if hasattr(response, 'prompt_eval_count') and hasattr(response, 'eval_count'):
            usage = LLMUsage(
                prompt_tokens=response.prompt_eval_count,
                completion_tokens=response.eval_count,
                total_tokens=response.prompt_eval_count + response.eval_count
            )
        elif isinstance(response, dict):
            if 'prompt_eval_count' in response and 'eval_count' in response:
                usage = LLMUsage(
                    prompt_tokens=response.get('prompt_eval_count', 0),
                    completion_tokens=response.get('eval_count', 0),
                    total_tokens=response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                )
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.provider,
            finish_reason="stop",
            usage=usage,
            response_time=response_time,
            request_id=None,  # Ollama doesn't provide request IDs
            metadata={
                "model": getattr(response, 'model', None) or (response.get('model') if isinstance(response, dict) else None),
                "done": getattr(response, 'done', None) or (response.get('done') if isinstance(response, dict) else None)
            }
        )