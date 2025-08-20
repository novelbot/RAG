"""
LangChain-based RAG (Retrieval-Augmented Generation) Implementation.

This module provides a comprehensive RAG system using LangChain's chains,
retrievers, and prompt templates for flexible and powerful information retrieval
and generation.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# LangChain core imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel

# LangChain chain imports
from langchain.chains import (
    RetrievalQA, ConversationalRetrievalChain,
    LLMChain, StuffDocumentsChain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

# LangChain retrievers
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    SelfQueryRetriever
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
    LLMChainFilter
)

# LangChain memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

from src.core.logging import LoggerMixin
from src.core.exceptions import RAGError


class RAGStrategy(Enum):
    """RAG strategy types."""
    SIMPLE = "simple"  # Basic retrieval and generation
    CONVERSATIONAL = "conversational"  # With conversation history
    MULTI_QUERY = "multi_query"  # Generate multiple queries
    CONTEXTUAL_COMPRESSION = "contextual_compression"  # Compress retrieved docs
    ENSEMBLE = "ensemble"  # Combine multiple retrievers
    PARENT_DOCUMENT = "parent_document"  # Retrieve parent docs
    SELF_QUERY = "self_query"  # Self-querying with metadata
    HYDE = "hyde"  # Hypothetical Document Embeddings


@dataclass
class LangChainRAGConfig:
    """Configuration for LangChain RAG system."""
    
    # Strategy
    strategy: RAGStrategy = RAGStrategy.SIMPLE
    
    # Retrieval settings
    retrieval_k: int = 5
    retrieval_score_threshold: float = 0.7
    rerank_top_k: int = 3
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    
    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = False
    
    # Chain settings
    chain_type: str = "stuff"  # stuff, map_reduce, refine, map_rerank
    return_source_documents: bool = True
    return_intermediate_steps: bool = False
    
    # Memory settings
    use_memory: bool = True
    memory_key: str = "chat_history"
    max_memory_tokens: int = 2000
    
    # Compression settings
    compression_threshold: float = 0.8
    use_llm_compression: bool = False
    
    # Multi-query settings
    num_queries: int = 3
    
    # Prompt customization
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None


class LangChainRAG(LoggerMixin):
    """
    Comprehensive RAG system using LangChain.
    
    Features:
    - Multiple RAG strategies
    - Conversation history management
    - Document compression and reranking
    - Multi-query generation
    - Ensemble retrieval
    - Custom prompt templates
    - Streaming support
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        config: LangChainRAGConfig
    ):
        """
        Initialize LangChain RAG system.
        
        Args:
            llm: LangChain chat model
            retriever: LangChain retriever
            config: RAG configuration
        """
        self.llm = llm
        self.base_retriever = retriever
        self.config = config
        
        # Initialize memory if needed
        self.memory = self._init_memory() if config.use_memory else None
        
        # Setup retriever based on strategy
        self.retriever = self._setup_retriever()
        
        # Setup RAG chain
        self.rag_chain = self._setup_rag_chain()
        
        self.logger.info(f"Initialized LangChain RAG with strategy: {config.strategy.value}")
    
    def _init_memory(self) -> Union[ConversationBufferMemory, ConversationSummaryMemory]:
        """Initialize conversation memory."""
        if self.config.max_memory_tokens > 0:
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key=self.config.memory_key,
                return_messages=True,
                max_token_limit=self.config.max_memory_tokens
            )
        else:
            return ConversationBufferMemory(
                memory_key=self.config.memory_key,
                return_messages=True,
                output_key="answer"
            )
    
    def _setup_retriever(self) -> BaseRetriever:
        """Setup retriever based on strategy."""
        strategy = self.config.strategy
        
        if strategy == RAGStrategy.SIMPLE:
            return self._setup_simple_retriever()
        elif strategy == RAGStrategy.MULTI_QUERY:
            return self._setup_multi_query_retriever()
        elif strategy == RAGStrategy.CONTEXTUAL_COMPRESSION:
            return self._setup_compression_retriever()
        elif strategy == RAGStrategy.ENSEMBLE:
            return self._setup_ensemble_retriever()
        elif strategy == RAGStrategy.PARENT_DOCUMENT:
            return self._setup_parent_document_retriever()
        elif strategy == RAGStrategy.SELF_QUERY:
            return self._setup_self_query_retriever()
        elif strategy == RAGStrategy.HYDE:
            return self._setup_hyde_retriever()
        else:
            return self.base_retriever
    
    def _setup_simple_retriever(self) -> BaseRetriever:
        """Setup simple retriever with MMR if enabled."""
        retriever = self.base_retriever
        
        # Configure search parameters
        if hasattr(retriever, 'search_kwargs'):
            search_kwargs = {
                'k': self.config.retrieval_k,
                'score_threshold': self.config.retrieval_score_threshold
            }
            
            if self.config.use_mmr:
                search_kwargs['fetch_k'] = self.config.retrieval_k * 3
                search_kwargs['lambda_mult'] = self.config.mmr_lambda
                
            retriever.search_kwargs = search_kwargs
        
        return retriever
    
    def _setup_multi_query_retriever(self) -> MultiQueryRetriever:
        """Setup multi-query retriever."""
        return MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm,
            include_original=True
        )
    
    def _setup_compression_retriever(self) -> ContextualCompressionRetriever:
        """Setup retriever with contextual compression."""
        # Create compressor
        if self.config.use_llm_compression:
            compressor = LLMChainExtractor.from_llm(self.llm)
        else:
            # Use embeddings filter
            if hasattr(self.base_retriever, 'embeddings'):
                compressor = EmbeddingsFilter(
                    embeddings=self.base_retriever.embeddings,
                    similarity_threshold=self.config.compression_threshold
                )
            else:
                # Fallback to LLM filter
                compressor = LLMChainFilter.from_llm(self.llm)
        
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
    
    def _setup_ensemble_retriever(self) -> EnsembleRetriever:
        """Setup ensemble retriever combining multiple retrievers."""
        # For now, just use the base retriever
        # In production, you'd combine multiple retrievers
        return EnsembleRetriever(
            retrievers=[self.base_retriever],
            weights=[1.0]
        )
    
    def _setup_parent_document_retriever(self) -> BaseRetriever:
        """Setup parent document retriever."""
        # This requires additional setup with document store
        # For now, return base retriever
        return self.base_retriever
    
    def _setup_self_query_retriever(self) -> BaseRetriever:
        """Setup self-query retriever with metadata filtering."""
        # This requires metadata field descriptions
        # For now, return base retriever
        return self.base_retriever
    
    def _setup_hyde_retriever(self) -> BaseRetriever:
        """Setup HyDE (Hypothetical Document Embeddings) retriever."""
        # Create a chain to generate hypothetical documents
        hyde_prompt = ChatPromptTemplate.from_template(
            "Please write a passage to answer the question: {question}"
        )
        
        # For now, return base retriever
        # Full implementation would generate hypothetical docs
        return self.base_retriever
    
    def _setup_rag_chain(self) -> Any:
        """Setup the main RAG chain."""
        # Create prompt template
        prompt = self._create_prompt_template()
        
        if self.config.strategy == RAGStrategy.CONVERSATIONAL:
            return self._setup_conversational_chain(prompt)
        else:
            return self._setup_simple_chain(prompt)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for RAG."""
        system_prompt = self.config.system_prompt or """당신은 웹소설 독자를 위한 전문 AI 어시스턴트 'NovelBot'입니다. 독자가 읽고 있는 웹소설에 대한 질문에 정확하고 도움이 되는 답변을 제공하는 것이 당신의 역할입니다. 제공된 맥락을 기반으로 질문에 답변하세요.
        
        ## 핵심 지침

        ### 1. 맥락 이해와 답변
        - 제공된 context를 깊이 있게 분석하여 답변하세요
        - 모호한 질문("그때 그 사건", "예전에 나온 인물")도 문맥을 파악해 정확히 답변하세요
        - 답변할 때는 구체적인 화차 정보를 함께 제공하세요
        - 예: "이 내용은 23화에서 처음 언급되었습니다"

        ### 2. 정보 제공 방식
        - 간결하면서도 충분한 정보를 제공하세요
        - 인물, 사건, 아이템 등을 설명할 때는 다음을 포함하세요:
        * 첫 등장/언급 시점 (화차)
        * 핵심 특징이나 역할
        * 다른 요소와의 관계
        * 중요한 관련 사건

        ### 3. 답변 구조
        - 질문에 대한 직접적인 답변을 먼저 제시
        - 필요시 추가 맥락 정보 제공
        - 관련 화차 정보 명시
        - 추가로 확인이 필요한 경우 안내
        - 반드시 '소설을 확인해 보았을 때,'로 시작

        ### 4. 톤과 매너
        - 친근하고 도움이 되는 어조 유지
        - 웹소설 독자의 관점에서 공감하며 답변
        - 작품에 대한 흥미를 유지시키는 방향으로 설명

        ## 답변 템플릿 예시

        **캐릭터 관련 질문:**
        "[캐릭터명]은 [첫 등장 화차]화에서 처음 등장했습니다. [핵심 특징/역할]. [주요 관계나 사건]. 더 자세한 내용은 [관련 화차]화를 참고하시면 됩니다."

        **사건/스토리 관련 질문:**
        "말씀하신 사건은 [화차]화에서 발생했습니다. [사건 요약]. 이 사건으로 인해 [영향이나 결과]. 관련된 내용은 [화차]화에서도 확인하실 수 있습니다."

        **아이템/설정 관련 질문:**
        "[아이템/설정명]은 [화차]화에서 처음 언급되었습니다. [설명]. [작품 내 중요도나 역할]. 추가 정보는 [화차]화에 있습니다."

        ## 중요 제약사항

        1. Context에 없는 정보는 추측하지 마세요
        2. "잘 모르겠습니다"보다는 "제공된 정보에서는 확인할 수 없습니다"라고 답변
        3. 여러 해석이 가능한 경우 가능한 옵션들을 제시

        ## 메타데이터 활용
        - Relevance: 정보의 중요도

        항상 독자의 읽기 경험을 보호하면서도 충실한 정보를 제공하는 것이 최우선 목표임을 기억하세요.
        """
        
        user_prompt = self.config.user_prompt_template or """
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        if self.config.use_memory:
            messages = [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name=self.config.memory_key, optional=True),
                ("human", user_prompt)
            ]
        else:
            messages = [
                ("system", system_prompt),
                ("human", user_prompt)
            ]
        
        return ChatPromptTemplate.from_messages(messages)
    
    def _setup_simple_chain(self, prompt: ChatPromptTemplate) -> Any:
        """Setup simple RAG chain."""
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(
            self.retriever,
            document_chain
        )
        
        return retrieval_chain
    
    def _setup_conversational_chain(self, prompt: ChatPromptTemplate) -> Any:
        """Setup conversational RAG chain."""
        # Create history-aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt
        )
        
        # Create question-answer chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt or "Answer the question based on the context."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("assistant", "Context: {context}\n\nAnswer:")
        ])
        
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Create final chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )
        
        return rag_chain
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            chat_history: Optional conversation history
            metadata_filter: Optional metadata filter for retrieval
            
        Returns:
            Response dictionary with answer and sources
        """
        try:
            start_time = time.time()
            
            # Prepare input
            chain_input = {"input": question, "question": question}
            
            # Add chat history if available
            if chat_history and self.config.use_memory:
                messages = []
                for human, ai in chat_history:
                    messages.append(HumanMessage(content=human))
                    messages.append(AIMessage(content=ai))
                chain_input["chat_history"] = messages
            
            # Apply metadata filter if provided
            if metadata_filter and hasattr(self.retriever, 'search_kwargs'):
                self.retriever.search_kwargs['filter'] = metadata_filter
            
            # Execute chain
            if self.config.streaming:
                response = self._stream_response(chain_input)
            else:
                response = self.rag_chain.invoke(chain_input)
            
            # Process response
            result = self._process_response(response, time.time() - start_time)
            
            # Update memory if enabled
            if self.memory:
                self.memory.save_context(
                    {"input": question},
                    {"output": result["answer"]}
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            raise RAGError(f"Query failed: {e}")
    
    async def aquery(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system asynchronously.
        
        Args:
            question: User question
            chat_history: Optional conversation history
            metadata_filter: Optional metadata filter
            
        Returns:
            Response dictionary with answer and sources
        """
        try:
            start_time = time.time()
            
            # Prepare input
            chain_input = {"input": question, "question": question}
            
            # Add chat history
            if chat_history and self.config.use_memory:
                messages = []
                for human, ai in chat_history:
                    messages.append(HumanMessage(content=human))
                    messages.append(AIMessage(content=ai))
                chain_input["chat_history"] = messages
            
            # Apply metadata filter
            if metadata_filter and hasattr(self.retriever, 'search_kwargs'):
                self.retriever.search_kwargs['filter'] = metadata_filter
            
            # Execute chain
            if self.config.streaming:
                response = await self._astream_response(chain_input)
            else:
                response = await self.rag_chain.ainvoke(chain_input)
            
            # Process response
            result = self._process_response(response, time.time() - start_time)
            
            # Update memory
            if self.memory:
                self.memory.save_context(
                    {"input": question},
                    {"output": result["answer"]}
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Async RAG query failed: {e}")
            raise RAGError(f"Query failed: {e}")
    
    def _stream_response(self, chain_input: Dict[str, Any]) -> Dict[str, Any]:
        """Stream response from chain."""
        collected_response = []
        documents = []
        
        for chunk in self.rag_chain.stream(chain_input):
            if isinstance(chunk, dict):
                if "answer" in chunk:
                    collected_response.append(chunk["answer"])
                if "context" in chunk and not documents:
                    documents = chunk["context"]
        
        return {
            "answer": "".join(collected_response),
            "context": documents
        }
    
    async def _astream_response(self, chain_input: Dict[str, Any]) -> Dict[str, Any]:
        """Stream response asynchronously."""
        collected_response = []
        documents = []
        
        async for chunk in self.rag_chain.astream(chain_input):
            if isinstance(chunk, dict):
                if "answer" in chunk:
                    collected_response.append(chunk["answer"])
                if "context" in chunk and not documents:
                    documents = chunk["context"]
        
        return {
            "answer": "".join(collected_response),
            "context": documents
        }
    
    def _process_response(
        self,
        response: Dict[str, Any],
        response_time: float
    ) -> Dict[str, Any]:
        """Process and format RAG response."""
        # Extract answer
        answer = response.get("answer", response.get("result", ""))
        
        # Extract source documents
        sources = []
        if "context" in response:
            documents = response["context"]
        elif "source_documents" in response:
            documents = response["source_documents"]
        else:
            documents = []
        
        for doc in documents:
            if isinstance(doc, Document):
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        # Build result
        result = {
            "answer": answer,
            "sources": sources,
            "response_time": response_time,
            "strategy": self.config.strategy.value
        }
        
        # Add intermediate steps if configured
        if self.config.return_intermediate_steps:
            result["intermediate_steps"] = response.get("intermediate_steps", [])
        
        return result
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            self.logger.info("Cleared conversation memory")
    
    def get_memory_messages(self) -> List[Dict[str, str]]:
        """Get conversation history from memory."""
        if not self.memory:
            return []
        
        messages = []
        if hasattr(self.memory, 'chat_memory'):
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    messages.append({"role": "human", "content": message.content})
                elif isinstance(message, AIMessage):
                    messages.append({"role": "assistant", "content": message.content})
        
        return messages


def create_langchain_rag(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    config: LangChainRAGConfig
) -> LangChainRAG:
    """Factory function to create LangChain RAG system."""
    return LangChainRAG(llm, retriever, config)