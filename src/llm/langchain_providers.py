"""
Enhanced LLM providers using latest LangChain integrations.

This module provides modern LangChain-based implementations for various LLM providers,
leveraging the latest features and improvements in langchain packages.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator
from dataclasses import dataclass
import os

# LangChain imports using updated packages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Provider-specific imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_anthropic import ChatAnthropic

from src.llm.base import (
    BaseLLMProvider, LLMRequest, LLMResponse, LLMStreamChunk,
    LLMConfig, LLMMessage, LLMRole, LLMProvider, LLMUsage
)
from src.core.exceptions import LLMError, RateLimitError, TokenLimitError
from src.core.logging import LoggerMixin


@dataclass
class LangChainLLMConfig:
    """Configuration for LangChain-based LLM providers."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    timeout: int = 30
    retry_attempts: int = 3
    
    # Provider-specific configurations
    google_project_id: Optional[str] = None
    anthropic_version: Optional[str] = None


class LangChainLLMManager(LoggerMixin):
    """
    Modern LLM manager using latest LangChain providers.
    
    Features:
    - Latest langchain-google-genai for Gemini models
    - Updated langchain-openai integration
    - Anthropic Claude integration
    - Async/streaming support
    - Chain composition capabilities
    - Enhanced error handling and retries
    """
    
    def __init__(self, config: LangChainLLMConfig):
        """Initialize LangChain LLM manager."""
        self.config = config
        self.llm_client = None
        self.chain = None
        self._initialize_provider()
    
    def _initialize_provider(self) -> None:
        """Initialize the appropriate LangChain provider."""
        provider = self.config.provider.lower()
        
        try:
            if provider == "google":
                self._initialize_google_provider()
            elif provider == "openai":
                self._initialize_openai_provider()
            elif provider == "anthropic" or provider == "claude":
                self._initialize_anthropic_provider()
            else:
                raise LLMError(f"Unsupported provider: {provider}")
                
            self._setup_chain()
            self.logger.info(f"Initialized LangChain provider: {provider} with model: {self.config.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize provider {provider}: {e}")
            raise LLMError(f"Provider initialization failed: {e}")
    
    def _initialize_google_provider(self) -> None:
        """Initialize Google Gemini provider using langchain-google-genai."""
        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("Google API key not provided")
        
        # Use latest ChatGoogleGenerativeAI features
        self.llm_client = ChatGoogleGenerativeAI(
            model=self.config.model or "gemini-pro",
            google_api_key=api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            streaming=self.config.streaming,
            timeout=self.config.timeout,
            # New features from updated package
            convert_system_message_to_human=True,
            safety_settings={
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        )
    
    def _initialize_openai_provider(self) -> None:
        """Initialize OpenAI provider using langchain-openai."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OpenAI API key not provided")
        
        self.llm_client = ChatOpenAI(
            model=self.config.model or "gpt-3.5-turbo",
            api_key=api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.streaming,
            timeout=self.config.timeout,
            max_retries=self.config.retry_attempts
        )
    
    def _initialize_anthropic_provider(self) -> None:
        """Initialize Anthropic Claude provider using langchain-anthropic."""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("Anthropic API key not provided")
        
        self.llm_client = ChatAnthropic(
            model=self.config.model or "claude-3-sonnet-20240229",
            api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=self.config.retry_attempts,
            anthropic_api_key=api_key
        )
    
    def _setup_chain(self) -> None:
        """Setup LangChain expression language chain."""
        # Create a flexible prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. {system_context}"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{user_input}")
        ])
        
        # Create chain with output parser
        self.chain = (
            {
                "user_input": RunnablePassthrough(),
                "system_context": lambda x: x.get("system_context", ""),
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | self.llm_client
            | StrOutputParser()
        )
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response asynchronously using LangChain.
        
        Args:
            request: LLM request with messages
            
        Returns:
            LLM response with content and metadata
        """
        start_time = time.time()
        
        try:
            # Convert custom message format to LangChain format
            messages = self._convert_messages_to_langchain(request.messages)
            
            # Prepare input for chain
            chain_input = {
                "user_input": messages[-1].content if messages else "",
                "system_context": self._extract_system_context(messages),
                "chat_history": self._extract_chat_history(messages)
            }
            
            # Execute chain asynchronously
            if self.config.streaming:
                return await self._generate_streaming_async(chain_input)
            else:
                response_content = await self.chain.ainvoke(chain_input)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Create standardized response
                return LLMResponse(
                    content=response_content,
                    model=self.config.model,
                    provider=self.config.provider,
                    response_time=response_time,
                    usage=LLMUsage(
                        total_tokens=len(response_content.split()) * 1.3  # Rough estimate
                    ),
                    finish_reason="stop"
                )
                
        except Exception as e:
            self.logger.error(f"LangChain generation failed: {e}")
            raise LLMError(f"Generation failed: {e}")
    
    async def _generate_streaming_async(self, chain_input: Dict[str, Any]) -> LLMResponse:
        """Generate streaming response."""
        content_chunks = []
        start_time = time.time()
        
        try:
            async for chunk in self.chain.astream(chain_input):
                if isinstance(chunk, str):
                    content_chunks.append(chunk)
                    yield LLMStreamChunk(
                        content=chunk,
                        delta=chunk,
                        finish_reason=None
                    )
            
            # Final response
            full_content = "".join(content_chunks)
            response_time = time.time() - start_time
            
            final_response = LLMResponse(
                content=full_content,
                model=self.config.model,
                provider=self.config.provider,
                response_time=response_time,
                usage=LLMUsage(
                    total_tokens=len(full_content.split()) * 1.3
                ),
                finish_reason="stop"
            )
            yield final_response
            
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise LLMError(f"Streaming failed: {e}")
    
    def _convert_messages_to_langchain(self, messages: List[LLMMessage]) -> List:
        """Convert custom message format to LangChain messages."""
        langchain_messages = []
        
        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == LLMRole.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == LLMRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
        
        return langchain_messages
    
    def _extract_system_context(self, messages: List) -> str:
        """Extract system context from messages."""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                return msg.content
        return ""
    
    def _extract_chat_history(self, messages: List) -> List:
        """Extract chat history (excluding system and last user message)."""
        history = []
        for i, msg in enumerate(messages[:-1]):  # Exclude last message
            if not isinstance(msg, SystemMessage):
                history.append(msg)
        return history
    
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for async generation."""
        return asyncio.run(self.generate_async(request))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "supports_streaming": self.config.streaming,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "langchain_version": "latest",
            "features": [
                "async_generation",
                "chain_composition", 
                "prompt_templates",
                "output_parsing",
                "streaming_support"
            ]
        }


class RAGLangChainProvider(LangChainLLMManager):
    """
    Specialized LangChain provider for RAG applications.
    
    Features:
    - Context injection for retrieved documents
    - Citation generation
    - Answer quality assessment
    - Multi-turn conversation support
    """
    
    def __init__(self, config: LangChainLLMConfig):
        super().__init__(config)
        self._setup_rag_chain()
    
    def _setup_rag_chain(self) -> None:
        """Setup RAG-specific chain with context injection."""
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on provided context.
            
            Instructions:
            - Use the provided context to answer the user's question
            - If the context doesn't contain relevant information, say so clearly
            - Cite relevant parts of the context when possible
            - Be concise but comprehensive
            
            Context: {context}
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}")
        ])
        
        self.rag_chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda x: x.get("context", "No context provided"),
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | rag_prompt
            | self.llm_client
            | StrOutputParser()
        )
    
    async def generate_rag_response(
        self,
        question: str,
        context: str,
        chat_history: Optional[List] = None
    ) -> LLMResponse:
        """
        Generate RAG response with context injection.
        
        Args:
            question: User question
            context: Retrieved document context
            chat_history: Previous conversation turns
            
        Returns:
            LLM response with context-aware answer
        """
        start_time = time.time()
        
        try:
            chain_input = {
                "question": question,
                "context": context,
                "chat_history": chat_history or []
            }
            
            response_content = await self.rag_chain.ainvoke(chain_input)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_content,
                model=self.config.model,
                provider=f"{self.config.provider}_rag",
                response_time=response_time,
                usage=LLMUsage(
                    total_tokens=len(response_content.split()) * 1.3
                ),
                finish_reason="stop",
                metadata={
                    "context_length": len(context),
                    "has_chat_history": len(chat_history or []) > 0,
                    "rag_enhanced": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"RAG generation failed: {e}")
            raise LLMError(f"RAG generation failed: {e}")


def create_langchain_provider(config: LangChainLLMConfig) -> LangChainLLMManager:
    """Factory function to create appropriate LangChain provider."""
    return LangChainLLMManager(config)


def create_rag_provider(config: LangChainLLMConfig) -> RAGLangChainProvider:
    """Factory function to create RAG-enhanced LangChain provider.""" 
    return RAGLangChainProvider(config)