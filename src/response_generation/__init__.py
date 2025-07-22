"""
Response Generation System for Multi-LLM RAG Applications.

This module provides sophisticated response generation capabilities including:
- Single LLM response generation for fast responses
- Multi-LLM ensemble mode for accuracy and quality
- Advanced prompt engineering with context injection
- Response quality evaluation and selection
- Post-processing and formatting
- Comprehensive error handling and timeouts
"""

from .base import (
    ResponseMode,
    ResponseQuality,
    ResponseGeneratorConfig,
    ResponseRequest,
    ResponseResult,
    EnsembleResult,
    BaseResponseGenerator
)

from .exceptions import (
    ResponseGenerationError,
    PromptEngineeringError,
    ResponseEvaluationError,
    ResponseProcessingError,
    EnsembleError,
    TimeoutError as ResponseTimeoutError
)

from .single_generator import SingleLLMGenerator
from .ensemble_generator import EnsembleLLMGenerator
from .prompt_engineer import PromptEngineer
from .response_evaluator import ResponseEvaluator
from .response_processor import ResponseProcessor
from .error_handler import ErrorHandler
from .manager import ResponseGenerationManager

__all__ = [
    # Base classes and data types
    "ResponseMode",
    "ResponseQuality", 
    "ResponseGeneratorConfig",
    "ResponseRequest",
    "ResponseResult",
    "EnsembleResult",
    "BaseResponseGenerator",
    
    # Exceptions
    "ResponseGenerationError",
    "PromptEngineeringError", 
    "ResponseEvaluationError",
    "ResponseProcessingError",
    "EnsembleError",
    "ResponseTimeoutError",
    
    # Core generators
    "SingleLLMGenerator",
    "EnsembleLLMGenerator",
    
    # Processing components
    "PromptEngineer",
    "ResponseEvaluator",
    "ResponseProcessor",
    "ErrorHandler",
    
    # Manager
    "ResponseGenerationManager"
]

# Version info
__version__ = "1.0.0"
__author__ = "RAG Development Team"