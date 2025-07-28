"""
OpenAI Embedding Provider implementation.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, cast
from openai import OpenAI, AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from src.embedding.base import (
    BaseEmbeddingProvider, EmbeddingConfig, EmbeddingRequest, 
    EmbeddingResponse, EmbeddingUsage, EmbeddingDimension, EmbeddingProvider
)
from src.core.exceptions import EmbeddingError, ConfigurationError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider implementation.
    
    Supports OpenAI's text-embedding models including:
    - text-embedding-3-large
    - text-embedding-3-small
    - text-embedding-ada-002
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "text-embedding-3-large": {
            "max_dimensions": 3072,
            "default_dimensions": 3072,
            "supports_shortening": True,
            "max_tokens": 8192
        },
        "text-embedding-3-small": {
            "max_dimensions": 1536,
            "default_dimensions": 1536,
            "supports_shortening": True,
            "max_tokens": 8192
        },
        "text-embedding-ada-002": {
            "max_dimensions": 1536,
            "default_dimensions": 1536,
            "supports_shortening": False,
            "max_tokens": 8192
        }
    }
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI embedding provider."""
        if config.provider != EmbeddingProvider.OPENAI:
            raise ConfigurationError(f"Invalid provider: {config.provider}")
        
        super().__init__(config)
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI clients."""
        try:
            client_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            }
            
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            
            if self.config.extra_headers:
                client_kwargs["default_headers"] = self.config.extra_headers
            
            self._client = OpenAI(**client_kwargs)
            self._async_client = AsyncOpenAI(**client_kwargs)
            
            self.logger.info(f"Initialized OpenAI embedding provider with model: {self.config.model}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}")
    
    async def generate_embeddings_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings asynchronously."""
        start_time = time.time()
        
        try:
            # Prepare request parameters
            params = self._prepare_request_params(request)
            
            # Handle batching for large requests
            if len(request.input) > self.config.batch_size:
                return await self._process_batched_request_async(request, params)
            
            # Make API call
            response = await self._async_client.embeddings.create(**params)
            
            # Process response
            result = self._process_response(response, request.model or self.config.model)
            result.response_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            raise self._handle_api_error(e)
    
    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings synchronously."""
        start_time = time.time()
        
        try:
            # Prepare request parameters
            params = self._prepare_request_params(request)
            
            # Handle batching for large requests
            if len(request.input) > self.config.batch_size:
                return self._process_batched_request(request, params)
            
            # Make API call
            response = self._client.embeddings.create(**params)
            
            # Process response
            result = self._process_response(response, request.model or self.config.model)
            result.response_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            raise self._handle_api_error(e)
    
    def _prepare_request_params(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        model = request.model or self.config.model
        
        params = {
            "input": request.input,
            "model": model,
            "encoding_format": request.encoding_format or self.config.encoding_format
        }
        
        # Add dimensions if supported and specified
        if request.dimensions and self._supports_dimensions(model):
            params["dimensions"] = request.dimensions
        elif self.config.dimensions and self._supports_dimensions(model):
            params["dimensions"] = self.config.dimensions
        
        # Add user if specified
        if request.user:
            params["user"] = request.user
        
        # Add extra parameters
        if self.config.extra_query:
            params.update(self.config.extra_query)
        
        if self.config.extra_body:
            params.update(self.config.extra_body)
        
        return params
    
    def _supports_dimensions(self, model: str) -> bool:
        """Check if model supports dimension specification."""
        model_config = self.MODEL_CONFIGS.get(model, {})
        return model_config.get("supports_shortening", False)
    
    async def _process_batched_request_async(self, request: EmbeddingRequest, params: Dict[str, Any]) -> EmbeddingResponse:
        """Process large requests in batches asynchronously."""
        batch_size = request.batch_size or self.config.batch_size
        # request.input is guaranteed to be List[str] after __post_init__
        batches = self._batch_texts(cast(List[str], request.input), batch_size)
        
        all_embeddings = []
        total_usage = EmbeddingUsage()
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            batch_params = params.copy()
            batch_params["input"] = batch
            tasks.append(self._async_client.embeddings.create(**batch_params))
        
        responses = await asyncio.gather(*tasks)
        
        # Combine results
        for response in responses:
            batch_result = self._process_response(response, request.model or self.config.model)
            all_embeddings.extend(batch_result.embeddings)
            total_usage += batch_result.usage
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=request.model or self.config.model,
            usage=total_usage,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0
        )
    
    def _process_batched_request(self, request: EmbeddingRequest, params: Dict[str, Any]) -> EmbeddingResponse:
        """Process large requests in batches synchronously."""
        batch_size = request.batch_size or self.config.batch_size
        # request.input is guaranteed to be List[str] after __post_init__
        batches = self._batch_texts(cast(List[str], request.input), batch_size)
        
        all_embeddings = []
        total_usage = EmbeddingUsage()
        
        for batch in batches:
            batch_params = params.copy()
            batch_params["input"] = batch
            
            response = self._client.embeddings.create(**batch_params)
            batch_result = self._process_response(response, request.model or self.config.model)
            
            all_embeddings.extend(batch_result.embeddings)
            total_usage += batch_result.usage
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=request.model or self.config.model,
            usage=total_usage,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0
        )
    
    def _process_response(self, response: CreateEmbeddingResponse, model: str) -> EmbeddingResponse:
        """Process OpenAI API response."""
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Calculate usage
        usage = EmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        # Get dimensions
        dimensions = len(embeddings[0]) if embeddings else 0
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage=usage,
            dimensions=dimensions,
            metadata={
                "provider": "openai",
                "encoding_format": "float"
            }
        )
    
    def get_embedding_dimension(self, model: str) -> EmbeddingDimension:
        """Get embedding dimension information."""
        if model not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model}")
        
        config = self.MODEL_CONFIGS[model]
        
        # Determine dimensions
        dimensions = self.config.dimensions or config["default_dimensions"]
        
        # Generate supported dimensions for models that support shortening
        supported_dims = None
        if config["supports_shortening"]:
            # Generate common dimension options
            max_dim = config["max_dimensions"]
            supported_dims = [256, 512, 768, 1024, 1536, 2048, 3072]
            supported_dims = [d for d in supported_dims if d <= max_dim]
        
        return EmbeddingDimension(
            dimensions=dimensions,
            model_name=model,
            max_dimensions=config["max_dimensions"],
            supported_dimensions=supported_dims
        )
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return list(self.MODEL_CONFIGS.keys())
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Check API key
            if not self.config.api_key:
                self.logger.error("OpenAI API key is required")
                return False
            
            # Check model
            if self.config.model not in self.MODEL_CONFIGS:
                self.logger.error(f"Unsupported model: {self.config.model}")
                return False
            
            # Check dimensions
            if self.config.dimensions:
                model_config = self.MODEL_CONFIGS[self.config.model]
                if self.config.dimensions > model_config["max_dimensions"]:
                    self.logger.error(f"Dimensions {self.config.dimensions} exceeds maximum {model_config['max_dimensions']}")
                    return False
            
            # Test client initialization
            test_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=5.0
            )
            
            self.logger.info("OpenAI configuration is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _handle_api_error(self, error: Exception) -> EmbeddingError:
        """Handle OpenAI API errors."""
        error_msg = str(error)
        
        if "rate_limit" in error_msg.lower():
            return EmbeddingError(f"OpenAI rate limit exceeded: {error_msg}")
        elif "invalid_request_error" in error_msg.lower():
            return EmbeddingError(f"OpenAI invalid request: {error_msg}")
        elif "authentication_error" in error_msg.lower():
            return EmbeddingError(f"OpenAI authentication error: {error_msg}")
        elif "permission_error" in error_msg.lower():
            return EmbeddingError(f"OpenAI permission error: {error_msg}")
        else:
            return EmbeddingError(f"OpenAI API error: {error_msg}")
    
    def get_pricing_info(self, model: str) -> Dict[str, Any]:
        """Get pricing information for model."""
        # Pricing per 1M tokens (as of current rates)
        pricing = {
            "text-embedding-3-large": {"price_per_1m_tokens": 0.13},
            "text-embedding-3-small": {"price_per_1m_tokens": 0.02},
            "text-embedding-ada-002": {"price_per_1m_tokens": 0.10}
        }
        
        return pricing.get(model, {"price_per_1m_tokens": 0.0})
    
    def estimate_cost(self, num_tokens: int, model: str) -> float:
        """Estimate cost for embedding generation."""
        pricing = self.get_pricing_info(model)
        price_per_1m = pricing.get("price_per_1m_tokens", 0.0)
        return (num_tokens / 1_000_000) * price_per_1m