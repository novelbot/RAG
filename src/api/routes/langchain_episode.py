"""
LangChain-based Episode RAG API Routes.

This module provides API endpoints for episode-specific RAG operations
using LangChain integrations for improved flexibility and performance.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, List, Any, Optional
import asyncio
import time
import logging

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports
from ...auth.dependencies import get_current_user, SimpleUser
from ..schemas import (
    EpisodeChatRequest, EpisodeChatResponse, EpisodeChatConversation,
    EpisodeChatError, EpisodeSource, ChatMessage
)
from ...core.config import get_config
from ...core.exceptions import RAGError
from ...embedding.langchain_embeddings import (
    LangChainEmbeddingProvider, LangChainEmbeddingConfig
)
from ...vector_stores.langchain_milvus import (
    LangChainMilvusVectorStore, LangChainMilvusConfig
)
from ...rag.langchain_rag import (
    LangChainRAG, LangChainRAGConfig, RAGStrategy
)
from ...llm.langchain_providers import (
    LangChainLLMManager, LangChainLLMConfig
)

# Create router
router = APIRouter(prefix="/api/v2/episodes", tags=["langchain_episodes"])
logger = logging.getLogger(__name__)


class LangChainEpisodeRAGService:
    """Service for LangChain-based episode RAG operations."""
    
    def __init__(self):
        """Initialize the LangChain RAG service."""
        self.config = get_config()
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.rag_system = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize LangChain components."""
        if self._initialized:
            return
        
        try:
            # Initialize LLM
            self.llm = self._init_llm()
            
            # Initialize embeddings
            self.embeddings = self._init_embeddings()
            
            # Initialize vector store
            self.vector_store = await self._init_vector_store()
            
            # Initialize RAG system
            self.rag_system = self._init_rag_system()
            
            self._initialized = True
            logger.info("LangChain Episode RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain RAG service: {e}")
            raise RAGError(f"Service initialization failed: {e}")
    
    def _init_llm(self):
        """Initialize LangChain LLM."""
        llm_config = self.config.llm
        provider = llm_config.provider.lower()
        
        if provider == "openai":
            return ChatOpenAI(
                model=llm_config.model or "gpt-3.5-turbo",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                streaming=llm_config.stream
            )
        elif provider in ["google", "gemini"]:
            return ChatGoogleGenerativeAI(
                model=llm_config.model or "gemini-pro",
                temperature=llm_config.temperature,
                max_output_tokens=llm_config.max_tokens,
                streaming=llm_config.stream
            )
        elif provider in ["anthropic", "claude"]:
            return ChatAnthropic(
                model=llm_config.model or "claude-3-sonnet-20240229",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                streaming=llm_config.stream
            )
        else:
            # Use LangChainLLMManager for flexibility
            config = LangChainLLMConfig(
                provider=provider,
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                streaming=llm_config.stream
            )
            manager = LangChainLLMManager(config)
            return manager.llm_client
    
    def _init_embeddings(self):
        """Initialize LangChain embeddings."""
        embed_config = self.config.embedding
        provider = embed_config.provider.lower()
        
        if provider == "openai":
            return OpenAIEmbeddings(
                model=embed_config.model or "text-embedding-3-small"
            )
        elif provider in ["google", "gemini"]:
            return GoogleGenerativeAIEmbeddings(
                model=embed_config.model or "models/embedding-001"
            )
        elif provider == "ollama":
            return OllamaEmbeddings(
                model=embed_config.model or "nomic-embed-text",
                base_url=embed_config.base_url or "http://localhost:11434"
            )
        else:
            # Use LangChainEmbeddingProvider for flexibility
            config = LangChainEmbeddingConfig(
                provider=provider,
                model=embed_config.model,
                dimensions=embed_config.dimensions,
                batch_size=embed_config.batch_size
            )
            provider_instance = LangChainEmbeddingProvider(config)
            return provider_instance.embeddings_client
    
    async def _init_vector_store(self):
        """Initialize LangChain Milvus vector store."""
        milvus_config = LangChainMilvusConfig(
            host=self.config.milvus.host,
            port=self.config.milvus.port,
            user=self.config.milvus.user,
            password=self.config.milvus.password,
            database=self.config.milvus.database,
            collection_name="langchain_episodes",
            dimension=self.config.embedding.dimensions or 768,
            metric_type="L2"
        )
        
        return LangChainMilvusVectorStore(
            embedding_function=self.embeddings,
            config=milvus_config
        )
    
    def _init_rag_system(self):
        """Initialize LangChain RAG system."""
        # Get retriever from vector store
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": self.config.rag.retrieval_k,
                "score_threshold": self.config.rag.similarity_threshold
            }
        )
        
        # Determine RAG strategy based on config
        if self.config.rag.use_reranking:
            strategy = RAGStrategy.CONTEXTUAL_COMPRESSION
        elif self.config.rag.use_multi_query:
            strategy = RAGStrategy.MULTI_QUERY
        else:
            strategy = RAGStrategy.CONVERSATIONAL
        
        # Create RAG config
        rag_config = LangChainRAGConfig(
            strategy=strategy,
            retrieval_k=self.config.rag.retrieval_k,
            retrieval_score_threshold=self.config.rag.similarity_threshold,
            rerank_top_k=self.config.rag.rerank_top_k if hasattr(self.config.rag, 'rerank_top_k') else 3,
            use_mmr=True,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            streaming=self.config.llm.stream,
            return_source_documents=True,
            use_memory=True,
            system_prompt=self.config.rag.system_prompt if hasattr(self.config.rag, 'system_prompt') else None
        )
        
        return LangChainRAG(
            llm=self.llm,
            retriever=retriever,
            config=rag_config
        )
    
    async def process_episode_query(
        self,
        request: EpisodeChatRequest,
        user: SimpleUser
    ) -> EpisodeChatResponse:
        """
        Process episode query using LangChain RAG.
        
        Args:
            request: Chat request
            user: Current user
            
        Returns:
            Chat response with sources
        """
        # Ensure initialization
        await self.initialize()
        
        try:
            start_time = time.time()
            
            # Build metadata filter if episode IDs provided
            metadata_filter = None
            if request.episode_ids:
                metadata_filter = {
                    "episode_id": {"$in": request.episode_ids}
                }
            
            # Convert chat history to format expected by LangChain
            chat_history = []
            if request.conversation and request.conversation.messages:
                for msg in request.conversation.messages:
                    if msg.role == "user":
                        # Find corresponding assistant message
                        for i, next_msg in enumerate(request.conversation.messages):
                            if next_msg.role == "assistant" and i > request.conversation.messages.index(msg):
                                chat_history.append((msg.content, next_msg.content))
                                break
            
            # Query RAG system
            result = await self.rag_system.aquery(
                question=request.question,
                chat_history=chat_history,
                metadata_filter=metadata_filter
            )
            
            # Process sources
            sources = []
            for source in result.get("sources", []):
                metadata = source.get("metadata", {})
                sources.append(EpisodeSource(
                    episode_id=metadata.get("episode_id", 0),
                    episode_number=metadata.get("episode_number", 0),
                    episode_title=metadata.get("episode_title", "Unknown"),
                    content=source.get("content", ""),
                    relevance_score=metadata.get("score", 0.0),
                    character_mentions=metadata.get("characters", [])
                ))
            
            # Update conversation
            updated_conversation = request.conversation or EpisodeChatConversation(
                id=str(time.time()),
                messages=[],
                metadata={}
            )
            
            # Add user message
            updated_conversation.messages.append(ChatMessage(
                role="user",
                content=request.question,
                timestamp=int(time.time())
            ))
            
            # Add assistant response
            updated_conversation.messages.append(ChatMessage(
                role="assistant",
                content=result["answer"],
                timestamp=int(time.time())
            ))
            
            # Calculate metrics
            response_time = time.time() - start_time
            total_tokens = len(request.question.split()) + len(result["answer"].split())
            
            return EpisodeChatResponse(
                answer=result["answer"],
                sources=sources,
                conversation=updated_conversation,
                search_metadata={
                    "total_results": len(sources),
                    "search_time": response_time,
                    "rag_strategy": result.get("strategy", "unknown"),
                    "model_used": self.config.llm.model,
                    "embedding_model": self.config.embedding.model
                },
                usage={
                    "prompt_tokens": len(request.question.split()),
                    "completion_tokens": len(result["answer"].split()),
                    "total_tokens": total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Episode query processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    async def ingest_episode_content(
        self,
        episode_id: int,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Ingest episode content into vector store.
        
        Args:
            episode_id: Episode ID
            content: Episode content
            metadata: Episode metadata
            
        Returns:
            Success status
        """
        await self.initialize()
        
        try:
            # Split content into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.rag.chunk_size if hasattr(self.config.rag, 'chunk_size') else 1000,
                chunk_overlap=self.config.rag.chunk_overlap if hasattr(self.config.rag, 'chunk_overlap') else 200
            )
            
            chunks = text_splitter.split_text(content)
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                meta = metadata.copy()
                meta.update({
                    "episode_id": episode_id,
                    "chunk_index": i,
                    "chunk_total": len(chunks)
                })
                chunk_metadata.append(meta)
            
            # Add to vector store
            ids = await self.vector_store.aadd_texts(
                texts=chunks,
                metadatas=chunk_metadata
            )
            
            logger.info(f"Ingested episode {episode_id} with {len(ids)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest episode {episode_id}: {e}")
            return False


# Initialize service
rag_service = LangChainEpisodeRAGService()


@router.post("/chat", response_model=EpisodeChatResponse)
async def chat_with_episodes(
    request: EpisodeChatRequest,
    background_tasks: BackgroundTasks,
    current_user: SimpleUser = Depends(get_current_user)
) -> EpisodeChatResponse:
    """
    Chat with episode content using LangChain RAG.
    
    This endpoint provides conversational AI capabilities over episode content
    using state-of-the-art LangChain integrations.
    """
    try:
        response = await rag_service.process_episode_query(request, current_user)
        
        # Log query in background
        background_tasks.add_task(
            log_query,
            user_id=current_user.id,
            question=request.question,
            response=response.answer,
            episode_ids=request.episode_ids
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/ingest/{episode_id}")
async def ingest_episode(
    episode_id: int,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: SimpleUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Ingest episode content into the LangChain vector store.
    
    This endpoint processes and stores episode content for RAG retrieval.
    """
    try:
        metadata = metadata or {}
        success = await rag_service.ingest_episode_content(
            episode_id=episode_id,
            content=content,
            metadata=metadata
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Episode {episode_id} ingested successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ingestion failed"
            )
            
    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check health status of LangChain RAG service."""
    try:
        await rag_service.initialize()
        
        return {
            "status": "healthy",
            "service": "langchain_episode_rag",
            "components": {
                "llm": rag_service.llm is not None,
                "embeddings": rag_service.embeddings is not None,
                "vector_store": rag_service.vector_store is not None,
                "rag_system": rag_service.rag_system is not None
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "langchain_episode_rag",
            "error": str(e)
        }


async def log_query(
    user_id: str,
    question: str,
    response: str,
    episode_ids: Optional[List[int]] = None
) -> None:
    """Log query for analytics (background task)."""
    try:
        logger.info(f"Query logged - User: {user_id}, Episodes: {episode_ids}")
        # In production, save to database
    except Exception as e:
        logger.error(f"Failed to log query: {e}")