"""
RAG-based chat API routes that combine vector search with conversation context.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer

from ...auth.dependencies import get_current_user, MockUser
from ..schemas import (
    ChatRequest, ChatResponse, ChatMessage, ChatConversation,
    ChatSearchMetadata, ChatConversationMetadata, ChatError,
    AnswerSource, ErrorResponse
)
from ...llm.base import LLMProvider
from ...core.config import get_config
from ...milvus.client import MilvusClient
from ...milvus.collection import MilvusCollection
from ...milvus.search import SearchManager, SearchQuery, SearchStrategy, create_search_query
from ...embedding.manager import EmbeddingManager
from ...embedding.base import EmbeddingRequest
from ...embedding.types import EmbeddingProvider, EmbeddingConfig
from ...embedding.providers import OllamaEmbeddingProvider
from ...services.query_logger import query_logger, QueryContext, QueryMetrics
from ...models.query_log import QueryType
from ...services.conversation_manager import conversation_manager

router = APIRouter(prefix="/chat", tags=["chat"])
security = HTTPBearer()

# Global instances (initialized lazily)
_milvus_client: Optional[MilvusClient] = None
_embedding_manager: Optional[EmbeddingManager] = None
_search_manager: Optional[SearchManager] = None
_collection: Optional[MilvusCollection] = None
_llm_provider: Optional[LLMProvider] = None


def get_chat_components():
    """Initialize and return chat components."""
    global _milvus_client, _embedding_manager, _search_manager, _collection, _llm_provider
    
    if not all([_milvus_client, _embedding_manager, _search_manager, _collection, _llm_provider]):
        config = get_config()
        
        # Initialize Milvus client
        _milvus_client = MilvusClient(config.milvus)
        _milvus_client.connect()
        
        # Initialize embedding manager with Ollama provider
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model="nomic-embed-text",
            api_base="http://localhost:11434"
        )
        
        ollama_provider = OllamaEmbeddingProvider(embedding_config)
        _embedding_manager = EmbeddingManager([ollama_provider])
        
        # Initialize collection
        _collection = MilvusCollection(_milvus_client, config.milvus.collection_name)
        
        # Initialize search manager
        _search_manager = SearchManager(_milvus_client, _embedding_manager)
        
        # Initialize LLM provider
        _llm_provider = LLMProvider(config.llm)
    
    return _milvus_client, _embedding_manager, _search_manager, _collection, _llm_provider


async def retrieve_conversation_context(
    conversation_id: Optional[str],
    user_id: str,
    max_context_turns: int
) -> tuple[Optional[ChatConversation], List[ChatMessage]]:
    """Retrieve conversation context for the chat request."""
    if not conversation_id:
        return None, []
    
    try:
        # Get conversation from conversation manager
        conversation_data = await conversation_manager.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not conversation_data:
            return None, []
        
        # Convert to ChatConversation format
        messages = []
        recent_messages = conversation_data.get('messages', [])[-max_context_turns*2:]  # Get recent messages
        
        for msg in recent_messages:
            chat_msg = ChatMessage(
                role=msg.get('role', 'user'),
                content=msg.get('content', ''),
                timestamp=msg.get('timestamp', datetime.now(timezone.utc)),
                metadata=msg.get('metadata', {})
            )
            messages.append(chat_msg)
        
        conversation = ChatConversation(
            id=conversation_id,
            messages=messages,
            created_at=conversation_data.get('created_at', datetime.now(timezone.utc)),
            updated_at=conversation_data.get('updated_at', datetime.now(timezone.utc)),
            user_id=user_id,
            total_messages=len(conversation_data.get('messages', []))
        )
        
        return conversation, messages
        
    except Exception as e:
        # Log error but don't fail the request
        print(f"Error retrieving conversation context: {e}")
        return None, []


async def perform_vector_search(
    query: str,
    max_results: int,
    search_filters: Optional[Dict[str, Any]],
    conversation_context: Optional[List[ChatMessage]]
) -> tuple[List[AnswerSource], ChatSearchMetadata]:
    """Perform vector search for relevant documents."""
    start_time = time.time()
    
    try:
        # Get components
        _, embedding_manager, search_manager, collection, _ = get_chat_components()
        
        # Enhance query with conversation context if available
        enhanced_query = query
        if conversation_context:
            # Extract recent context for query enhancement
            recent_context = " ".join([
                msg.content for msg in conversation_context[-2:]  # Last 2 messages
                if msg.role == "user"
            ])
            if recent_context:
                enhanced_query = f"{recent_context} {query}"
        
        # Create search query
        search_query = create_search_query(
            query=enhanced_query,
            strategy=SearchStrategy.SEMANTIC_SIMILARITY,
            top_k=max_results,
            filters=search_filters or {}
        )
        
        # Perform search
        search_results = await search_manager.search(
            collection_name=collection.name,
            search_query=search_query
        )
        
        # Convert to AnswerSource format
        sources = []
        for result in search_results.get('results', []):
            source = AnswerSource(
                document_id=result.get('id', ''),
                title=result.get('metadata', {}).get('title'),
                excerpt=result.get('content', ''),
                relevance_score=result.get('score', 0.0)
            )
            sources.append(source)
        
        # Create search metadata
        search_time_ms = (time.time() - start_time) * 1000
        metadata = ChatSearchMetadata(
            query_used=enhanced_query,
            documents_found=len(sources),
            search_time_ms=search_time_ms,
            filters_applied=search_filters
        )
        
        return sources, metadata
        
    except Exception as e:
        # Create error metadata
        search_time_ms = (time.time() - start_time) * 1000
        metadata = ChatSearchMetadata(
            query_used=query,
            documents_found=0,
            search_time_ms=search_time_ms,
            filters_applied=search_filters
        )
        
        print(f"Vector search failed: {e}")
        return [], metadata


async def generate_chat_response(
    user_message: str,
    sources: List[AnswerSource],
    conversation_context: Optional[List[ChatMessage]],
    response_format: str
) -> tuple[str, Optional[float]]:
    """Generate AI response using LLM with RAG context."""
    try:
        _, _, _, _, llm_provider = get_chat_components()
        
        # Build context for LLM prompt
        context_parts = []
        
        # Add document sources
        if sources:
            context_parts.append("=== RELEVANT DOCUMENTS ===")
            for i, source in enumerate(sources, 1):
                context_parts.append(f"Document {i} (Relevance: {source.relevance_score:.2f}):")
                if source.title:
                    context_parts.append(f"Title: {source.title}")
                context_parts.append(f"Content: {source.excerpt}")
                context_parts.append("")
        
        # Add conversation history
        if conversation_context:
            context_parts.append("=== CONVERSATION HISTORY ===")
            for msg in conversation_context[-6:]:  # Last 6 messages for context
                role_label = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")
            context_parts.append("")
        
        # Build the full prompt
        context_text = "\n".join(context_parts) if context_parts else ""
        
        if response_format == "concise":
            instruction = "Provide a concise, direct answer based on the context above."
        else:
            instruction = "Provide a detailed, helpful answer based on the context above. Include relevant information from the documents and consider the conversation history."
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question using the provided context.

{context_text}

User Question: {user_message}

{instruction}

If you cannot find relevant information in the provided context, say so clearly and provide a general response based on your knowledge."""
        
        # Generate response
        response = await llm_provider.generate_response(prompt)
        
        # Calculate confidence score based on source relevance and context
        confidence = None
        if sources:
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
            confidence = min(avg_relevance + 0.1, 1.0)  # Boost confidence slightly for having sources
        
        return response, confidence
        
    except Exception as e:
        print(f"LLM response generation failed: {e}")
        return f"I apologize, but I'm having trouble generating a response right now. Please try again later.", None


async def save_conversation_turn(
    conversation_id: str,
    user_message: str,
    assistant_response: str,
    user_id: str,
    is_new_conversation: bool
) -> None:
    """Save the conversation turn to the conversation manager."""
    try:
        # Save user message
        await conversation_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=user_message,
            create_conversation=is_new_conversation
        )
        
        # Save assistant response
        await conversation_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=assistant_response
        )
        
    except Exception as e:
        print(f"Error saving conversation turn: {e}")
        # Don't fail the request if conversation saving fails


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: MockUser = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ChatResponse:
    """
    RAG-based chat endpoint that combines vector search with conversation context.
    
    This endpoint:
    1. Retrieves conversation context if conversation_id is provided
    2. Performs vector search for relevant documents
    3. Combines search results with conversation context
    4. Generates AI response using LLM
    5. Saves the conversation turn
    6. Returns structured response with sources and metadata
    """
    start_time = time.time()
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        is_new_conversation = request.conversation_id is None
        
        # Step 1: Retrieve conversation context
        conversation_context = []
        if request.use_conversation_context and request.conversation_id:
            conversation, conversation_context = await retrieve_conversation_context(
                conversation_id=request.conversation_id,
                user_id=current_user.id,
                max_context_turns=request.max_context_turns
            )
        
        # Step 2: Perform vector search
        sources, search_metadata = await perform_vector_search(
            query=request.message,
            max_results=request.max_results,
            search_filters=request.search_filters,
            conversation_context=conversation_context if request.use_conversation_context else None
        )
        
        # Step 3: Generate AI response
        ai_response, confidence_score = await generate_chat_response(
            user_message=request.message,
            sources=sources if request.include_sources else [],
            conversation_context=conversation_context if request.use_conversation_context else None,
            response_format=request.response_format
        )
        
        # Step 4: Save conversation turn (in background)
        background_tasks.add_task(
            save_conversation_turn,
            conversation_id=conversation_id,
            user_message=request.message,
            assistant_response=ai_response,
            user_id=current_user.id,
            is_new_conversation=is_new_conversation
        )
        
        # Step 5: Log query (in background)
        query_context = QueryContext(
            user_id=current_user.id,
            session_id=conversation_id,
            endpoint="/chat",
            user_agent="",
            ip_address=""
        )
        
        query_metrics = QueryMetrics(
            query_length=len(request.message),
            result_count=len(sources),
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used="combined-rag-chat"
        )
        
        background_tasks.add_task(
            query_logger.log_query,
            query=request.message,
            query_type=QueryType.ASK,
            context=query_context,
            metrics=query_metrics,
            response_preview=ai_response[:200]
        )
        
        # Step 6: Build response
        conversation_metadata = ChatConversationMetadata(
            conversation_id=conversation_id,
            total_messages=len(conversation_context) + 2,  # +2 for current turn
            context_messages_used=len(conversation_context),
            is_new_conversation=is_new_conversation
        )
        
        response = ChatResponse(
            message=ai_response,
            conversation_id=conversation_id,
            sources=sources if request.include_sources else [],
            conversation_metadata=conversation_metadata,
            search_metadata=search_metadata,
            confidence_score=confidence_score,
            has_context=len(conversation_context) > 0,
            response_time_ms=(time.time() - start_time) * 1000,
            user_id=current_user.id
        )
        
        return response
        
    except Exception as e:
        # Return error response
        error_response = ChatError(
            error_type="invalid_request",
            message=f"Failed to process chat request: {str(e)}",
            conversation_id=request.conversation_id,
            retry_suggestions=["Check your message content", "Try again with a simpler question"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


@router.get("/{conversation_id}", response_model=ChatConversation)
async def get_conversation(
    conversation_id: str,
    current_user: MockUser = Depends(get_current_user)
) -> ChatConversation:
    """
    Retrieve a conversation by ID.
    """
    try:
        conversation_data = await conversation_manager.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not conversation_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Convert to ChatConversation format
        messages = []
        for msg in conversation_data.get('messages', []):
            chat_msg = ChatMessage(
                role=msg.get('role', 'user'),
                content=msg.get('content', ''),
                timestamp=msg.get('timestamp', datetime.now(timezone.utc)),
                metadata=msg.get('metadata', {})
            )
            messages.append(chat_msg)
        
        conversation = ChatConversation(
            id=conversation_id,
            messages=messages,
            created_at=conversation_data.get('created_at', datetime.now(timezone.utc)),
            updated_at=conversation_data.get('updated_at', datetime.now(timezone.utc)),
            user_id=current_user.id,
            total_messages=len(messages)
        )
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a conversation by ID.
    """
    try:
        success = await conversation_manager.delete_conversation(
            conversation_id=conversation_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )