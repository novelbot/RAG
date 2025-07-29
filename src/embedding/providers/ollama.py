"""
Ollama Embedding Provider implementation.
"""

import time
from typing import List, Dict, Any, cast
from ollama import Client, AsyncClient

from src.embedding.base import (
    BaseEmbeddingProvider, EmbeddingRequest, 
    EmbeddingResponse, EmbeddingUsage, EmbeddingDimension
)
from src.embedding.types import EmbeddingConfig
from src.core.exceptions import EmbeddingError, ConfigurationError


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama embedding provider implementation.
    
    Supports locally hosted Ollama models with embedding capabilities including:
    - nomic-embed-text
    - mxbai-embed-large  
    - llama3.2
    - all-minilm
    """
    
    # Model configurations (common embedding models)
    MODEL_CONFIGS = {
        "nomic-embed-text": {
            "dimensions": 768,
            "max_tokens": 8192,
            "description": "High-quality text embedding model"
        },
        "mxbai-embed-large": {
            "dimensions": 1024,
            "max_tokens": 512,
            "description": "Large multilingual embedding model"
        },
        "llama3.2": {
            "dimensions": 4096,
            "max_tokens": 2048,
            "description": "Llama 3.2 model with embedding support"
        },
        "all-minilm": {
            "dimensions": 384,
            "max_tokens": 256,
            "description": "Lightweight multilingual embedding model"
        },
        "snowflake-arctic-embed": {
            "dimensions": 1024,
            "max_tokens": 2048,
            "description": "Snowflake Arctic embedding model"
        },
        "jeffh/intfloat-multilingual-e5-large-instruct": {
            "dimensions": 1024,
            "max_tokens": 512,
            "description": "Multilingual E5 large instruct model",
            "requires_instruction_prompt": True
        }
    }
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Ollama embedding provider."""
        # Handle both string and enum provider values (Context7 MCP pattern)
        provider_value = config.provider
        if hasattr(provider_value, 'value'):
            provider_value = provider_value.value
        
        if provider_value != "ollama":
            raise ConfigurationError(f"Invalid provider: {config.provider}")
        
        super().__init__(config)
    
    def _initialize_client(self) -> None:
        """Initialize Ollama clients."""
        try:
            base_url = self.config.base_url or "http://localhost:11434"
            
            client_kwargs = {
                "host": base_url,
                "timeout": self.config.timeout
            }
            
            if self.config.extra_headers:
                client_kwargs["headers"] = self.config.extra_headers
            
            self._client = Client(**client_kwargs)
            self._async_client = AsyncClient(**client_kwargs)
            
            self.logger.info(f"Initialized Ollama embedding provider with model: {self.config.model}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Ollama client: {e}")
    
    async def generate_embeddings_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings asynchronously."""
        start_time = time.time()
        
        try:
            # Handle batching for large requests
            if len(request.input) > self.config.batch_size:
                return await self._process_batched_request_async(request)
            
            # Make API call
            model = request.model or self.config.model
            
            # Apply instruction prompt if model requires it
            is_query = request.metadata.get("is_query", False) if request.metadata else False
            # Ensure input is a list (should already be converted in __post_init__)
            input_list = request.input if isinstance(request.input, list) else [request.input]
            processed_input = self._apply_instruction_prompt(input_list, model, is_query)
            
            # Ollama embed accepts either single string or list of strings
            response = await self._async_client.embed(
                model=model,
                input=processed_input
            )
            
            # Process response
            result = self._process_response(response, model, len(request.input))
            result.response_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ollama embedding generation failed: {e}")
            raise self._handle_api_error(e)
    
    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings synchronously."""
        start_time = time.time()
        
        try:
            # Handle batching for large requests
            if len(request.input) > self.config.batch_size:
                return self._process_batched_request(request)
            
            # Make API call
            model = request.model or self.config.model
            
            # Apply instruction prompt if model requires it
            is_query = request.metadata.get("is_query", False) if request.metadata else False
            # Ensure input is a list (should already be converted in __post_init__)
            input_list = request.input if isinstance(request.input, list) else [request.input]
            processed_input = self._apply_instruction_prompt(input_list, model, is_query)
            
            # Ollama embed accepts either single string or list of strings
            response = self._client.embed(
                model=model,
                input=processed_input
            )
            
            # Process response
            result = self._process_response(response, model, len(request.input))
            result.response_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ollama embedding generation failed: {e}")
            raise self._handle_api_error(e)
    
    async def _process_batched_request_async(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Process large requests in batches asynchronously."""
        batch_size = request.batch_size or self.config.batch_size
        # request.input is guaranteed to be List[str] after __post_init__
        batches = self._batch_texts(cast(List[str], request.input), batch_size)
        
        all_embeddings = []
        total_usage = EmbeddingUsage()
        
        model = request.model or self.config.model
        is_query = request.metadata.get("is_query", False) if request.metadata else False
        
        for batch in batches:
            processed_batch = self._apply_instruction_prompt(batch, model, is_query)
            response = await self._async_client.embed(
                model=model,
                input=processed_batch
            )
            
            batch_result = self._process_response(response, model, len(batch))
            all_embeddings.extend(batch_result.embeddings)
            total_usage += batch_result.usage
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model,
            usage=total_usage,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0
        )
    
    def _process_batched_request(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Process large requests in batches synchronously."""
        batch_size = request.batch_size or self.config.batch_size
        # request.input is guaranteed to be List[str] after __post_init__
        batches = self._batch_texts(cast(List[str], request.input), batch_size)
        
        all_embeddings = []
        total_usage = EmbeddingUsage()
        
        model = request.model or self.config.model
        is_query = request.metadata.get("is_query", False) if request.metadata else False
        
        for batch in batches:
            processed_batch = self._apply_instruction_prompt(batch, model, is_query)
            response = self._client.embed(
                model=model,
                input=processed_batch
            )
            
            batch_result = self._process_response(response, model, len(batch))
            all_embeddings.extend(batch_result.embeddings)
            total_usage += batch_result.usage
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model,
            usage=total_usage,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0
        )
    
    def _process_response(self, response: Any, model: str, num_inputs: int) -> EmbeddingResponse:
        """Process Ollama API response."""
        # Extract embeddings from response
        # Handle both single embedding and multiple embeddings cases
        embeddings = response.get("embeddings", response.get("embedding", []))
        if isinstance(embeddings, list) and len(embeddings) > 0 and not isinstance(embeddings[0], list):
            # If we got a single embedding (list of floats), wrap it in a list
            embeddings = [embeddings]
        
        if not embeddings:
            raise EmbeddingError("No embeddings returned from Ollama")
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Estimate token usage (Ollama doesn't provide exact token counts)
        # Since embeddings are vectors, we estimate based on number of inputs
        estimated_tokens = num_inputs * 50
        
        usage = EmbeddingUsage(
            prompt_tokens=int(estimated_tokens),
            total_tokens=int(estimated_tokens)
        )
        
        # Get dimensions
        dimensions = len(embeddings[0]) if embeddings else 0
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage=usage,
            dimensions=dimensions,
            metadata={
                "provider": "ollama",
                "encoding_format": "float",
                "host": self.config.base_url
            }
        )
    
    def get_embedding_dimension(self, model: str) -> EmbeddingDimension:
        """Get embedding dimension information."""
        # Try to get from known configurations
        if model in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model]
            return EmbeddingDimension(
                dimensions=config["dimensions"],
                model_name=model,
                max_dimensions=config["dimensions"]
            )
        
        # For unknown models, try to determine dimensions by making a test call
        try:
            test_response = self._client.embed(model=model, input=["test"])
            if test_response.get("embeddings"):
                dimensions = len(test_response["embeddings"][0])
                return EmbeddingDimension(
                    dimensions=dimensions,
                    model_name=model,
                    max_dimensions=dimensions
                )
        except Exception as e:
            self.logger.warning(f"Could not determine dimensions for model {model}: {e}")
        
        # Default fallback
        return EmbeddingDimension(
            dimensions=768,
            model_name=model,
            max_dimensions=768
        )
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        try:
            # Get list of available models from Ollama
            models_response = self._client.list()
            available_models = []
            
            for model in models_response.get("models", []):
                # Try 'name' field first, then 'model' field
                model_name = model.get("name", "") or model.get("model", "")
                if model_name:
                    # Add both full name and base name (without tag)
                    available_models.append(model_name)
                    base_name = model_name.split(":")[0]
                    if base_name != model_name and base_name not in available_models:
                        available_models.append(base_name)
            
            return available_models
            
        except Exception as e:
            self.logger.warning(f"Could not get model list from Ollama: {e}")
            # Return known embedding models as fallback
            return list(self.MODEL_CONFIGS.keys())
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about available models."""
        try:
            models_response = self._client.list()
            model_info = {}
            
            for model in models_response.get("models", []):
                # Try 'name' field first, then 'model' field
                model_name = model.get("name", "") or model.get("model", "")
                if model_name:
                    model_info[model_name] = {
                        "name": model_name,
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at", ""),
                        "digest": model.get("digest", "")
                    }
            
            return model_info
            
        except Exception as e:
            self.logger.warning(f"Could not get detailed model info from Ollama: {e}")
            return {}
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Test connection to Ollama
            test_client = Client(host=self.config.base_url or "http://localhost:11434")
            
            # Try to list models to verify connection
            models_response = test_client.list()
            
            # Check if the specified model is available
            available_models = []
            for m in models_response.get("models", []):
                model_name = m.get("name", "") or m.get("model", "")
                if model_name:
                    available_models.append(model_name)
                    base_name = model_name.split(":")[0]
                    if base_name != model_name:
                        available_models.append(base_name)
            
            if self.config.model not in available_models:
                self.logger.warning(f"Model {self.config.model} not found in available models: {available_models}")
                # Don't fail validation - model might be pullable
            
            self.logger.info("Ollama configuration is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _apply_instruction_prompt(self, texts: List[str], model: str, is_query: bool = False) -> List[str]:
        """Apply instruction prompt formatting for models that require it."""
        # Check if model requires instruction prompt
        model_base = model.split(':')[0]  # Remove tag (e.g., :f32)
        model_config = self.MODEL_CONFIGS.get(model_base, {})
        
        if not model_config.get("requires_instruction_prompt", False):
            return texts
        
        # Apply E5 instruction format
        if "intfloat-multilingual-e5-large-instruct" in model_base:
            if is_query:
                # For queries: add instruction prompt
                task_description = "Given a web search query, retrieve relevant passages that answer the query"
                formatted_texts = [f"Instruct: {task_description}\nQuery: {text}" for text in texts]
            else:
                # For documents: no special formatting needed
                formatted_texts = texts
            
            self.logger.debug(f"Applied E5 instruction formatting for {'query' if is_query else 'document'}")
            return formatted_texts
        
        return texts

    def _handle_api_error(self, error: Exception) -> EmbeddingError:
        """Handle Ollama API errors."""
        error_msg = str(error)
        
        if "connection" in error_msg.lower():
            return EmbeddingError(f"Ollama connection error: {error_msg}")
        elif "not found" in error_msg.lower():
            return EmbeddingError(f"Ollama model not found: {error_msg}")
        elif "timeout" in error_msg.lower():
            return EmbeddingError(f"Ollama request timeout: {error_msg}")
        else:
            return EmbeddingError(f"Ollama API error: {error_msg}")
    
    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            self.logger.info(f"Pulling model {model} from Ollama registry...")
            self._client.pull(model)
            self.logger.info(f"Successfully pulled model {model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pull model {model}: {e}")
            return False
    
    async def pull_model_async(self, model: str) -> bool:
        """Pull a model from Ollama registry asynchronously."""
        try:
            self.logger.info(f"Pulling model {model} from Ollama registry...")
            await self._async_client.pull(model)
            self.logger.info(f"Successfully pulled model {model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pull model {model}: {e}")
            return False
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            response = self._client.show(model)
            return dict(response)
            
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model}: {e}")
            return {}
    
    def get_pricing_info(self, _model: str) -> Dict[str, Any]:
        """Get pricing information for model (free for local models)."""
        return {
            "price_per_1m_tokens": 0.0,
            "note": "Local model - no API costs"
        }
    
    def estimate_cost(self, _num_tokens: int, _model: str) -> float:
        """Estimate cost for embedding generation (always 0 for local models)."""
        return 0.0