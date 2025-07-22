"""
Google Embedding Provider implementation.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from google.genai import Client, types

from src.embedding.base import (
    BaseEmbeddingProvider, EmbeddingConfig, EmbeddingRequest, 
    EmbeddingResponse, EmbeddingUsage, EmbeddingDimension, EmbeddingProvider
)
from src.core.exceptions import EmbeddingError, RateLimitError, ConfigurationError


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Google embedding provider implementation.
    
    Supports Google's text embedding models including:
    - text-embedding-004
    - text-multilingual-embedding-002
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "text-embedding-004": {
            "max_dimensions": 768,
            "default_dimensions": 768,
            "supports_shortening": True,
            "max_tokens": 2048,
            "supported_tasks": ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"]
        },
        "text-multilingual-embedding-002": {
            "max_dimensions": 768,
            "default_dimensions": 768,
            "supports_shortening": False,
            "max_tokens": 2048,
            "supported_tasks": ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"]
        },
        "embedding-001": {
            "max_dimensions": 768,
            "default_dimensions": 768,
            "supports_shortening": False,
            "max_tokens": 2048,
            "supported_tasks": ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"]
        }
    }
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Google embedding provider."""
        if config.provider != EmbeddingProvider.GOOGLE:
            raise ConfigurationError(f"Invalid provider: {config.provider}")
        
        super().__init__(config)
    
    def _initialize_client(self) -> None:
        """Initialize Google GenAI client."""
        try:
            client_kwargs = {
                "api_key": self.config.api_key,
                "http_options": {
                    "timeout": self.config.timeout,
                    "retry_attempts": self.config.max_retries
                }
            }
            
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            
            self._client = Client(**client_kwargs)
            self._async_client = Client(**client_kwargs)  # Google client handles both sync/async
            
            self.logger.info(f"Initialized Google embedding provider with model: {self.config.model}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google client: {e}")
    
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
            response = await self._async_client.models.embed_content(**params)
            
            # Process response
            result = self._process_response(response, request.model or self.config.model)
            result.response_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Google embedding generation failed: {e}")
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
            response = self._client.models.embed_content(**params)
            
            # Process response
            result = self._process_response(response, request.model or self.config.model)
            result.response_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Google embedding generation failed: {e}")
            raise self._handle_api_error(e)
    
    def _prepare_request_params(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """Prepare parameters for Google API call."""
        model = request.model or self.config.model
        
        params = {
            "model": model,
            "contents": request.input
        }
        
        # Configure embedding parameters
        config_params = {}
        
        # Add dimensions if supported and specified
        if request.dimensions and self._supports_dimensions(model):
            config_params["output_dimensionality"] = request.dimensions
        elif self.config.dimensions and self._supports_dimensions(model):
            config_params["output_dimensionality"] = self.config.dimensions
        
        # Add task type if specified in metadata
        task_type = request.metadata.get("task_type")
        if task_type and self._supports_task_type(model, task_type):
            config_params["task_type"] = task_type
        
        # Add title if specified in metadata
        title = request.metadata.get("title")
        if title:
            config_params["title"] = title
        
        # Add auto-truncate if specified
        if request.truncate:
            config_params["auto_truncate"] = True
        
        # Add configuration if any parameters are set
        if config_params:
            params["config"] = types.EmbedContentConfig(**config_params)
        
        return params
    
    def _supports_dimensions(self, model: str) -> bool:
        """Check if model supports dimension specification."""
        model_config = self.MODEL_CONFIGS.get(model, {})
        return model_config.get("supports_shortening", False)
    
    def _supports_task_type(self, model: str, task_type: str) -> bool:
        """Check if model supports task type specification."""
        model_config = self.MODEL_CONFIGS.get(model, {})
        supported_tasks = model_config.get("supported_tasks", [])
        return task_type in supported_tasks
    
    async def _process_batched_request_async(self, request: EmbeddingRequest, params: Dict[str, Any]) -> EmbeddingResponse:
        """Process large requests in batches asynchronously."""
        batch_size = request.batch_size or self.config.batch_size
        batches = self._batch_texts(request.input, batch_size)
        
        all_embeddings = []
        total_usage = EmbeddingUsage()
        
        # Process batches sequentially (Google API may have concurrency limits)
        for batch in batches:
            batch_params = params.copy()
            batch_params["contents"] = batch
            
            response = await self._async_client.models.embed_content(**batch_params)
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
        batches = self._batch_texts(request.input, batch_size)
        
        all_embeddings = []
        total_usage = EmbeddingUsage()
        
        for batch in batches:
            batch_params = params.copy()
            batch_params["contents"] = batch
            
            response = self._client.models.embed_content(**batch_params)
            batch_result = self._process_response(response, request.model or self.config.model)
            
            all_embeddings.extend(batch_result.embeddings)
            total_usage += batch_result.usage
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=request.model or self.config.model,
            usage=total_usage,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0
        )
    
    def _process_response(self, response: Any, model: str) -> EmbeddingResponse:
        """Process Google API response."""
        # Extract embeddings from response
        embeddings = []
        total_tokens = 0
        
        for embedding_obj in response.embeddings:
            embeddings.append(embedding_obj.values)
            
            # Calculate token count from statistics if available
            if hasattr(embedding_obj, 'statistics') and embedding_obj.statistics:
                total_tokens += embedding_obj.statistics.token_count
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Calculate usage
        usage = EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
        
        # Get dimensions
        dimensions = len(embeddings[0]) if embeddings else 0
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage=usage,
            dimensions=dimensions,
            metadata={
                "provider": "google",
                "encoding_format": "float",
                "total_embeddings": len(embeddings)
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
            supported_dims = [128, 256, 512, 768]
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
    
    def get_supported_tasks(self, model: str) -> List[str]:
        """Get list of supported task types for a model."""
        if model not in self.MODEL_CONFIGS:
            return []
        
        return self.MODEL_CONFIGS[model].get("supported_tasks", [])
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Check API key
            if not self.config.api_key:
                self.logger.error("Google API key is required")
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
            test_client = Client(
                api_key=self.config.api_key,
                http_options={"timeout": 5.0}
            )
            
            self.logger.info("Google configuration is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _handle_api_error(self, error: Exception) -> EmbeddingError:
        """Handle Google API errors."""
        error_msg = str(error)
        
        if "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return RateLimitError(f"Google rate limit exceeded: {error_msg}")
        elif "invalid" in error_msg.lower():
            return EmbeddingError(f"Google invalid request: {error_msg}")
        elif "permission" in error_msg.lower() or "auth" in error_msg.lower():
            return EmbeddingError(f"Google authentication error: {error_msg}")
        else:
            return EmbeddingError(f"Google API error: {error_msg}")
    
    def get_pricing_info(self, model: str) -> Dict[str, Any]:
        """Get pricing information for model."""
        # Pricing per 1M tokens (estimated rates)
        pricing = {
            "text-embedding-004": {"price_per_1m_tokens": 0.025},
            "text-multilingual-embedding-002": {"price_per_1m_tokens": 0.025},
            "embedding-001": {"price_per_1m_tokens": 0.025}
        }
        
        return pricing.get(model, {"price_per_1m_tokens": 0.0})
    
    def estimate_cost(self, num_tokens: int, model: str) -> float:
        """Estimate cost for embedding generation."""
        pricing = self.get_pricing_info(model)
        price_per_1m = pricing.get("price_per_1m_tokens", 0.0)
        return (num_tokens / 1_000_000) * price_per_1m