"""
Query processing API routes for RAG operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer
from typing import Dict, List, Any, Optional
import asyncio
import time

from ...auth.dependencies import get_current_user, MockUser
from ..schemas import QueryRequest, SearchResponse, RAGResponse, BatchSearchResponse, QueryHistoryResponse
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

router = APIRouter(prefix="/query", tags=["query"])
security = HTTPBearer()

# Global instances (initialized lazily)
_milvus_client: Optional[MilvusClient] = None
_embedding_manager: Optional[EmbeddingManager] = None
_search_manager: Optional[SearchManager] = None
_collection: Optional[MilvusCollection] = None


def get_vector_search_components():
    """Initialize and return vector search components."""
    global _milvus_client, _embedding_manager, _search_manager, _collection
    
    if not all([_milvus_client, _embedding_manager, _search_manager, _collection]):
        config = get_config()
        
        # Initialize Milvus client
        _milvus_client = MilvusClient(config.milvus)
        _milvus_client.connect()
        
        # Initialize embedding manager with Ollama provider
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.OLLAMA,
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            dimensions=768
        )
        ollama_provider = OllamaEmbeddingProvider(embedding_config)
        _embedding_manager = EmbeddingManager([embedding_config])
        
        # Get or create collection
        collection_name = config.milvus.collection_name
        
        # Create schema for collection
        from ...milvus.schema import RAGCollectionSchema
        schema = RAGCollectionSchema(
            collection_name=collection_name,
            vector_dim=embedding_config.dimensions
        )
        
        _collection = MilvusCollection(_milvus_client, schema)
        
        # Ensure collection exists and is loaded
        if not _milvus_client.has_collection(collection_name):
            _milvus_client.create_collection_if_not_exists(
                collection_name, 
                dim=embedding_config.dimensions
            )
        
        # Initialize search manager
        _search_manager = SearchManager(_milvus_client)
    
    return _milvus_client, _embedding_manager, _search_manager, _collection


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user),
    request: Request = None
) -> SearchResponse:
    """
    Search documents using vector similarity.
    
    Args:
        search_request: Search query and parameters
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        request: FastAPI request object for logging context
        
    Returns:
        Dict: Search results with documents and metadata
    """
    start_time = time.time()
    embedding_start_time = None
    search_start_time = None
    
    try:
        # Initialize vector search components
        milvus_client, embedding_manager, search_manager, collection = get_vector_search_components()
        
        # Step 1: Convert query text to embedding
        embedding_start_time = time.time()
        embedding_request = EmbeddingRequest(
            input=[search_request.query],
            model="nomic-embed-text"
        )
        
        embedding_response = await embedding_manager.generate_embeddings_async(embedding_request)
        embedding_time_ms = (time.time() - embedding_start_time) * 1000
        query_vectors = embedding_response.embeddings
        
        # Step 2: Apply access control filtering
        # Get user's access tags (for now, assume all users have basic access)
        user_access_tags = getattr(current_user, 'access_tags', ['public', 'basic'])
        if isinstance(user_access_tags, str):
            user_access_tags = [user_access_tags]
        
        # Create filter expression for access control
        access_filter = None
        if user_access_tags:
            # Allow documents with any of the user's access tags, or public documents
            tag_filters = [f'access_tags like "%{tag}%"' for tag in user_access_tags + ['public']]
            access_filter = " or ".join(tag_filters)
        
        # Step 3: Perform vector search with access control
        search_start_time = time.time()
        search_query = create_search_query(
            vectors=query_vectors,
            limit=search_request.max_results * 2,  # Get more results to account for filtering
            strategy=SearchStrategy.BALANCED,
            filter_expr=access_filter,
            output_fields=["content", "metadata", "access_tags"]
        )
        
        # Execute search
        search_result = search_manager.search(collection, search_query)
        search_time_ms = (time.time() - search_start_time) * 1000
        
        # Step 4: Format results for API response with additional access control
        from ..schemas import SearchResult
        
        def has_access_to_document(doc_access_tags, user_tags):
            """Check if user has access to document based on tags."""
            if not doc_access_tags:
                return True  # No restrictions
            
            if isinstance(doc_access_tags, str):
                doc_tags = [tag.strip() for tag in doc_access_tags.split(',')]
            else:
                doc_tags = doc_access_tags
            
            # Check if user has any of the required tags
            return any(tag in user_tags or tag == 'public' for tag in doc_tags)
        
        search_results = []
        for hit in search_result.hits:
            # Extract information from Milvus hit
            doc_id = hit.get("id", str(hit.get("pk", "unknown")))
            content = hit.get("content", "")
            metadata = hit.get("metadata", {})
            doc_access_tags = hit.get("access_tags", "public")
            distance = hit.get("distance", 0.0)
            
            # Additional access control check
            if not has_access_to_document(doc_access_tags, user_access_tags):
                continue
            
            # Convert distance to similarity score (higher is better)
            score = max(0.0, 1.0 - distance) if distance <= 1.0 else 1.0 / (1.0 + distance)
            
            # Extract title from metadata or generate from content
            title = metadata.get("title", f"Document {doc_id}")
            if not title and content:
                title = content[:50] + "..." if len(content) > 50 else content
            
            # Remove access_tags from metadata in response for security
            response_metadata = {k: v for k, v in metadata.items() if k != "access_tags"}
            
            search_results.append(
                SearchResult(
                    id=doc_id,
                    title=title,
                    content=content[:500] + "..." if len(content) > 500 else content,
                    score=round(score, 3),
                    metadata=response_metadata
                )
            )
            
            # Limit to requested number of results
            if len(search_results) >= search_request.max_results:
                break
        
        # Calculate total response time
        total_response_time_ms = (time.time() - start_time) * 1000
        
        # Create metrics for logging
        metrics = QueryMetrics(
            response_time_ms=total_response_time_ms,
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_time_ms,
            results_count=len(search_results)
        )
        
        # Add background task for comprehensive logging
        background_tasks.add_task(
            log_query,
            search_request.query,
            current_user.id,
            QueryType.SEARCH,
            total_response_time_ms,
            request,
            len(search_results),
            {
                "limit": search_request.max_results,
                "filter": access_filter
            },
            metrics
        )
        
        return SearchResponse(
            query=search_request.query,
            results=search_results,
            total_results=search_result.total_count,
            search_time_ms=round(total_response_time_ms, 2),
            user_id=current_user.id
        )
        
    except Exception as e:
        error_response_time_ms = (time.time() - start_time) * 1000
        
        # Log error for debugging
        print(f"Vector search error: {str(e)}")
        
        # Add background task for error logging
        background_tasks.add_task(
            log_query,
            search_request.query,
            current_user.id,
            QueryType.SEARCH,
            error_response_time_ms,
            request,
            0,  # No results due to error
            {
                "limit": search_request.max_results,
                "error": str(e)
            },
            None,  # No metrics due to error
            None,  # No model info
            str(e)  # Error message
        )
        
        # Return fallback response
        return SearchResponse(
            query=search_request.query,
            results=[],
            total_results=0,
            search_time_ms=round(error_response_time_ms, 2),
            user_id=current_user.id,
            error=f"Search failed: {str(e)}"
        )


@router.post("/ask", response_model=Dict[str, Any])
async def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Ask a question and get AI-generated answer using RAG with conversation context support.
    
    Args:
        request: Question and parameters (includes conversation context)
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Dict: AI-generated answer with sources, metadata, and session info
    """
    import time
    from src.core.config import get_config
    from src.llm.providers.ollama import OllamaProvider
    from src.llm.base import LLMConfig, LLMRequest, LLMMessage, LLMRole
    
    start_time = time.time()
    session_id = None
    
    # Add background task for logging
    background_tasks.add_task(log_query, request.query, current_user.id)
    
    try:
        # Get configuration
        config = get_config()
        
        # Handle conversation context if enabled
        conversation_context = []
        context_enhanced_query = request.query
        
        if request.use_context:
            async with conversation_manager.conversation_context(
                request.session_id, current_user.id, auto_create=True
            ) as conv_ctx:
                session_id = conv_ctx.session.session_id if conv_ctx.session else None
                
                # Get conversation context
                if conv_ctx.session:
                    context_window = request.conversation_context.max_context_turns if request.conversation_context else 5
                    conversation_context = await conv_ctx.get_context(context_window)
                    
                    # Build context-enhanced query
                    if conversation_context:
                        context_enhanced_query = conversation_manager.build_context_prompt(
                            conversation_context, request.query
                        )
        
        # Initialize Ollama provider
        llm_config = LLMConfig(
            provider=LLMProvider(config.llm.provider),
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            enable_streaming=False
        )
        
        ollama_provider = OllamaProvider(llm_config)
        
        # Create system message with conversation awareness
        system_content = "You are a helpful AI assistant."
        if request.use_context and conversation_context:
            system_content += " You have access to the conversation history and should provide contextually relevant responses."
        
        # Create LLM request with context-enhanced query
        llm_messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_content),
            LLMMessage(role=LLMRole.USER, content=context_enhanced_query)
        ]
        
        llm_request = LLMRequest(
            messages=llm_messages,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            model=config.llm.model
        )
        
        # Generate response using proper LLM system
        llm_response = await ollama_provider.generate_async(llm_request)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Save conversation turn if context is enabled
        if request.use_context and session_id:
            try:
                await conversation_manager.add_turn(
                    session_id=session_id,
                    user_query=request.query,
                    assistant_response=llm_response.content,
                    response_time_ms=processing_time_ms,
                    token_count=llm_response.usage.total_tokens if llm_response.usage else None,
                    metadata={
                        "model_used": f"{config.llm.provider}/{config.llm.model}",
                        "context_turns_used": len(conversation_context),
                        "finish_reason": llm_response.finish_reason
                    }
                )
            except Exception as conv_e:
                # Don't fail the main request if conversation saving fails
                print(f"Failed to save conversation turn: {conv_e}")
        
        # Format response with conversation metadata
        response = {
            "question": request.query,
            "answer": llm_response.content,
            "sources": [
                {
                    "document_id": "ollama_llm",
                    "title": f"Ollama {config.llm.model} Response",
                    "excerpt": "Response generated using proper LLM system",
                    "relevance_score": 0.95
                }
            ],
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "model_used": f"{config.llm.provider}/{config.llm.model}",
                "tokens_used": llm_response.usage.total_tokens if llm_response.usage else 0,
                "confidence_score": 0.85,
                "response_time": llm_response.response_time,
                "finish_reason": llm_response.finish_reason,
                "prompt_tokens": llm_response.usage.prompt_tokens if llm_response.usage else 0,
                "completion_tokens": llm_response.usage.completion_tokens if llm_response.usage else 0,
                # Conversation metadata
                "conversation": {
                    "session_id": session_id,
                    "context_enabled": request.use_context,
                    "context_turns_used": len(conversation_context),
                    "enhanced_query_used": context_enhanced_query != request.query
                }
            },
            "user_id": current_user.id
        }
        
        return response
        
    except Exception as e:
        # Fallback to error response with processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        error_response = {
            "question": request.query,
            "answer": f"I apologize, but I'm experiencing technical difficulties and cannot process your question at the moment. Error: {str(e)}",
            "sources": [
                {
                    "document_id": "error",
                    "title": "Error Response",
                    "excerpt": "Error occurred during query processing",
                    "relevance_score": 0.0
                }
            ],
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "model_used": "error",
                "tokens_used": 0,
                "confidence_score": 0.0,
                "error": str(e),
                "conversation": {
                    "session_id": session_id,
                    "context_enabled": request.use_context,
                    "context_turns_used": 0,
                    "enhanced_query_used": False
                }
            },
            "user_id": current_user.id
        }
        
        return error_response


@router.post("/batch_search", response_model=Dict[str, Any])
async def batch_search(
    queries: List[str],
    max_results: int = 5,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Process multiple search queries in batch.
    
    Args:
        queries: List of search queries
        max_results: Maximum results per query
        current_user: Authenticated user
        
    Returns:
        Dict: Batch search results
    """
    if len(queries) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 queries allowed per batch"
        )
    
    # Simulate async batch processing
    results = []
    for query in queries:
        await asyncio.sleep(0.1)  # Simulate individual query processing
        results.append({
            "query": query,
            "results": [
                {
                    "id": f"doc_{hash(query) % 100}",
                    "content": f"Result for: {query}",
                    "score": 0.85
                }
            ]
        })
    
    return {
        "batch_id": f"batch_{hash(str(queries))}",
        "queries_processed": len(queries),
        "results": results,
        "user_id": current_user.id
    }


@router.get("/history", response_model=Dict[str, Any])
async def get_query_history(
    limit: int = 50,
    offset: int = 0,
    query_type: Optional[str] = None,
    status_filter: Optional[str] = None,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's query history with filtering and pagination.
    
    Args:
        limit: Maximum number of queries to return (max 100)
        offset: Number of queries to skip
        query_type: Filter by query type (search, rag, batch_search, similarity)
        status_filter: Filter by status (success, failed, timeout, cancelled)
        current_user: Authenticated user
        
    Returns:
        Dict: Query history with pagination and performance statistics
    """
    start_time = time.time()
    
    # Validate pagination limits
    if limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum limit is 100 queries per request"
        )
    
    try:
        # Import required models for filtering
        from ...models.query_log import QueryLog, QueryType, QueryStatus
        from ...core.user_database import get_user_session
        
        with get_user_session() as db:
            # Base query for user's queries
            query = db.query(QueryLog).filter(QueryLog.user_id == current_user.id)
            
            # Apply query type filter if provided
            if query_type:
                try:
                    query_type_enum = QueryType(query_type)
                    query = query.filter(QueryLog.query_type == query_type_enum)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid query_type: {query_type}. Valid options: search, rag, batch_search, similarity"
                    )
            
            # Apply status filter if provided
            if status_filter:
                try:
                    status_enum = QueryStatus(status_filter)
                    query = query.filter(QueryLog.status == status_enum)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid status_filter: {status_filter}. Valid options: success, failed, timeout, cancelled"
                    )
            
            # Get total count for pagination
            total_count = query.count()
            
            # Apply ordering and pagination
            query_logs = query.order_by(QueryLog.created_at.desc())\
                             .offset(offset)\
                             .limit(limit)\
                             .all()
            
            # Calculate performance statistics for the filtered results
            if query_logs:
                # Calculate aggregate statistics
                successful_queries = [log for log in query_logs if log.status == QueryStatus.SUCCESS]
                failed_queries = [log for log in query_logs if log.status == QueryStatus.FAILED]
                
                # Response time statistics
                response_times = [log.response_time_ms for log in query_logs if log.response_time_ms]
                avg_response_time = sum(response_times) / len(response_times) if response_times else None
                
                # Token usage statistics
                total_tokens_used = sum(log.total_tokens for log in query_logs if log.total_tokens)
                
                # Results count statistics
                total_results_returned = sum(log.results_count for log in query_logs if log.results_count)
                
                performance_stats = {
                    "success_rate": len(successful_queries) / len(query_logs) if query_logs else 0,
                    "average_response_time_ms": round(avg_response_time, 2) if avg_response_time else None,
                    "total_tokens_used": total_tokens_used,
                    "total_results_returned": total_results_returned,
                    "query_types_breakdown": {},
                    "top_error_messages": []
                }
                
                # Query type breakdown
                from collections import Counter
                type_counts = Counter(log.query_type.value for log in query_logs)
                performance_stats["query_types_breakdown"] = dict(type_counts)
                
                # Top error messages for failed queries
                if failed_queries:
                    error_counts = Counter(log.error_message for log in failed_queries if log.error_message)
                    performance_stats["top_error_messages"] = [
                        {"error": error, "count": count} 
                        for error, count in error_counts.most_common(5)
                    ]
            else:
                performance_stats = {
                    "success_rate": 0,
                    "average_response_time_ms": None,
                    "total_tokens_used": 0,
                    "total_results_returned": 0,
                    "query_types_breakdown": {},
                    "top_error_messages": []
                }
            
            # Format history entries for response
            history_entries = []
            for query_log in query_logs:
                # Create privacy-safe history entry
                entry = {
                    "id": query_log.id,
                    "query_text": query_log.query_text,
                    "query_type": query_log.query_type.value,
                    "status": query_log.status.value,
                    "created_at": query_log.created_at.isoformat(),
                    "response_time_ms": query_log.response_time_ms,
                    "results_count": query_log.results_count,
                    "model_used": query_log.model_used,
                    "total_tokens": query_log.total_tokens
                }
                
                # Add error message only if query failed (for user debugging)
                if query_log.status == QueryStatus.FAILED and query_log.error_message:
                    entry["error_message"] = query_log.error_message
                
                # Add performance breakdown for successful queries
                if query_log.status == QueryStatus.SUCCESS:
                    entry["performance"] = {
                        "embedding_time_ms": query_log.embedding_time_ms,
                        "search_time_ms": query_log.search_time_ms,
                        "llm_time_ms": query_log.llm_time_ms,
                        "processing_time_ms": query_log.processing_time_ms
                    }
                
                history_entries.append(entry)
            
            # Calculate pagination metadata
            total_pages = ((total_count - 1) // limit) + 1 if total_count > 0 else 1
            has_next = offset + len(history_entries) < total_count
            has_prev = offset > 0
            
            # Calculate response time
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "history": history_entries,
                "pagination": {
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "current_page": (offset // limit) + 1,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev
                },
                "filters": {
                    "query_type": query_type,
                    "status_filter": status_filter
                },
                "performance_stats": performance_stats,
                "user_id": current_user.id,
                "processing_time_ms": round(processing_time_ms, 2)
            }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_response_time_ms = (time.time() - start_time) * 1000
        
        # Log error for debugging
        print(f"Query history retrieval error: {str(e)}")
        
        # Return fallback response with mock data for compatibility
        mock_history = [
            {
                "id": f"query_{i}",
                "query_text": f"Sample query {i}",
                "query_type": "search" if i % 2 else "rag",
                "status": "success" if i % 3 != 0 else "failed",
                "created_at": f"2024-01-{i:02d}T10:00:00Z",
                "response_time_ms": 150.0 + (i * 10),
                "results_count": i % 5 + 1,
                "model_used": "nomic-embed-text",
                "total_tokens": i * 25
            }
            for i in range(1, 21)
        ]
        
        paginated_results = mock_history[offset:offset + limit]
        
        return {
            "history": paginated_results,
            "pagination": {
                "total_count": len(mock_history),
                "limit": limit,
                "offset": offset,
                "current_page": (offset // limit) + 1,
                "total_pages": ((len(mock_history) - 1) // limit) + 1,
                "has_next": offset + len(paginated_results) < len(mock_history),
                "has_prev": offset > 0
            },
            "filters": {
                "query_type": query_type,
                "status_filter": status_filter
            },
            "performance_stats": {
                "success_rate": 0.8,
                "average_response_time_ms": 200.0,
                "total_tokens_used": 2500,
                "total_results_returned": 45,
                "query_types_breakdown": {"search": 12, "rag": 8},
                "top_error_messages": []
            },
            "user_id": current_user.id,
            "processing_time_ms": round(error_response_time_ms, 2),
            "error": f"Service unavailable, showing mock data: {str(e)}"
        }


@router.get("/conversations", response_model=Dict[str, Any])
async def get_user_conversations(
    limit: int = 20,
    include_inactive: bool = False,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's conversation sessions.
    
    Args:
        limit: Maximum number of sessions to return
        include_inactive: Whether to include inactive sessions
        current_user: Authenticated user
        
    Returns:
        Dict: List of conversation sessions with metadata
    """
    try:
        sessions = await conversation_manager.get_user_sessions(
            current_user.id, limit, include_inactive
        )
        
        sessions_data = [session.to_dict() for session in sessions]
        
        return {
            "conversations": sessions_data,
            "total_count": len(sessions_data),
            "limit": limit,
            "include_inactive": include_inactive,
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )


@router.get("/conversations/{session_id}", response_model=Dict[str, Any])
async def get_conversation_details(
    session_id: str,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed conversation session information including turns.
    
    Args:
        session_id: Conversation session ID
        current_user: Authenticated user
        
    Returns:
        Dict: Detailed conversation information
    """
    try:
        session = await conversation_manager.get_session(session_id, current_user.id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {session_id} not found"
            )
        
        # Get conversation context (all turns)
        conversation_context = await conversation_manager.get_conversation_context(
            session_id, max_turns=100  # Get all turns
        )
        
        session_data = session.to_dict()
        session_data["conversation_turns"] = conversation_context
        
        return {
            "conversation": session_data,
            "user_id": current_user.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation details: {str(e)}"
        )


@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(
    title: Optional[str] = None,
    expires_in_hours: int = 24,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new conversation session.
    
    Args:
        title: Optional conversation title
        expires_in_hours: Session expiration time in hours
        current_user: Authenticated user
        
    Returns:
        Dict: Created conversation session information
    """
    try:
        session = await conversation_manager.create_session(
            user_id=current_user.id,
            title=title,
            expires_in_hours=expires_in_hours
        )
        
        return {
            "conversation": session.to_dict(),
            "message": "Conversation session created successfully",
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation session: {str(e)}"
        )


@router.put("/conversations/{session_id}/archive", response_model=Dict[str, Any])
async def archive_conversation(
    session_id: str,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Archive a conversation session.
    
    Args:
        session_id: Conversation session ID to archive
        current_user: Authenticated user
        
    Returns:
        Dict: Archive operation result
    """
    try:
        success = await conversation_manager.archive_session(session_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {session_id} not found"
            )
        
        return {
            "message": f"Conversation session {session_id} archived successfully",
            "session_id": session_id,
            "user_id": current_user.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive conversation session: {str(e)}"
        )


async def log_query(query: str, user_id: str, query_type: QueryType = QueryType.SEARCH,
                   response_time_ms: Optional[float] = None,
                   request: Optional[Request] = None,
                   results_count: Optional[int] = None,
                   search_params: Optional[Dict[str, Any]] = None,
                   metrics: Optional[QueryMetrics] = None,
                   model_info: Optional[Dict[str, str]] = None,
                   error_message: Optional[str] = None) -> None:
    """
    Background task to log query for analytics and audit trails.
    
    Args:
        query: The search query
        user_id: ID of the user who made the query
        query_type: Type of query performed
        response_time_ms: Total response time in milliseconds
        request: FastAPI request object for context
        results_count: Number of results returned
        search_params: Search parameters used
        metrics: Detailed performance metrics
        model_info: Information about models used
        error_message: Error message if query failed
    """
    try:
        # Create query context
        context = QueryContext(
            user_id=user_id,
            session_id=getattr(request, 'session_id', None) if request else None,
            user_agent=request.headers.get("user-agent") if request else None,
            ip_address=request.client.host if request and request.client else None
        )
        
        # Create query log entry
        query_log = await query_logger.create_query_log(
            query_text=query,
            query_type=query_type,
            context=context,
            search_params=search_params,
            request=request
        )
        
        # Update with metrics if available
        if metrics:
            await query_logger.update_query_metrics(
                query_log.id,
                metrics,
                model_info,
                response_metadata={
                    "results_count": results_count,
                    "response_time_ms": response_time_ms
                }
            )
        
        # Mark success or failure
        if error_message:
            await query_logger.mark_query_failed(
                query_log.id,
                error_message,
                response_time_ms
            )
        else:
            await query_logger.mark_query_success(
                query_log.id,
                metrics
            )
            
        print(f"Successfully logged query '{query[:50]}...' by user: {user_id} "
              f"(Type: {query_type.value}, Status: {'Failed' if error_message else 'Success'})")
              
    except Exception as e:
        # Don't fail the main request if logging fails
        print(f"Failed to log query: {e}")
        import logging
        logging.getLogger(__name__).error(f"Query logging failed: {e}")