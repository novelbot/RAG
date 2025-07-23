"""
Query processing API routes for RAG operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import Dict, List, Any, Optional
import asyncio

from ...auth.dependencies import get_current_user, MockUser
from ..schemas import QueryRequest, SearchResponse, RAGResponse, BatchSearchResponse, QueryHistoryResponse

router = APIRouter(prefix="/query", tags=["query"])
security = HTTPBearer()


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user)
) -> SearchResponse:
    """
    Search documents using vector similarity.
    
    Args:
        request: Search query and parameters
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Dict: Search results with documents and metadata
    """
    # Simulate async vector search
    await asyncio.sleep(0.3)
    
    # Add background task for logging
    background_tasks.add_task(log_query, request.query, current_user.id)
    
    # TODO: Implement actual vector search
    mock_results = [
        {
            "id": "doc_1",
            "title": "Sample Document 1",
            "content": "This is a sample document content...",
            "score": 0.95,
            "metadata": {"source": "file1.pdf", "page": 1}
        },
        {
            "id": "doc_2", 
            "title": "Sample Document 2",
            "content": "Another sample document...",
            "score": 0.87,
            "metadata": {"source": "file2.pdf", "page": 3}
        }
    ]
    
    from ..schemas import SearchResult
    
    search_results = [
        SearchResult(
            id=result["id"],
            title=result["title"],
            content=result["content"],
            score=result["score"],
            metadata=result["metadata"]
        )
        for result in mock_results[:request.max_results]
    ]
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        total_results=len(mock_results),
        search_time_ms=300.0,
        user_id=current_user.id
    )


@router.post("/ask", response_model=Dict[str, Any])
async def ask_question(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Ask a question and get AI-generated answer using RAG.
    
    Args:
        request: Question and parameters
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Dict: AI-generated answer with sources and metadata
    """
    import time
    from src.core.config import get_config
    from src.llm.providers.ollama import OllamaProvider
    from src.llm.base import LLMConfig, LLMRequest, LLMMessage, LLMRole
    
    start_time = time.time()
    
    # Add background task for logging
    background_tasks.add_task(log_query, request.query, current_user.id)
    
    try:
        # Get configuration
        config = get_config()
        
        # Initialize Ollama provider
        llm_config = LLMConfig(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            enable_streaming=False
        )
        
        ollama_provider = OllamaProvider(llm_config)
        
        # Create LLM request
        llm_messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="You are a helpful AI assistant."),
            LLMMessage(role=LLMRole.USER, content=request.query)
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
        
        # Format response
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
                "completion_tokens": llm_response.usage.completion_tokens if llm_response.usage else 0
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
                "error": str(e)
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
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's query history.
    
    Args:
        limit: Maximum number of queries to return
        offset: Number of queries to skip
        current_user: Authenticated user
        
    Returns:
        Dict: Query history with pagination
    """
    # Simulate async history retrieval
    await asyncio.sleep(0.2)
    
    # TODO: Implement actual query history retrieval
    mock_history = [
        {
            "id": f"query_{i}",
            "query": f"Sample query {i}",
            "timestamp": f"2024-01-{i:02d}T10:00:00Z",
            "type": "search" if i % 2 else "ask"
        }
        for i in range(1, 21)
    ]
    
    paginated_results = mock_history[offset:offset + limit]
    
    return {
        "history": paginated_results,
        "total": len(mock_history),
        "limit": limit,
        "offset": offset,
        "user_id": current_user.id
    }


async def log_query(query: str, user_id: str) -> None:
    """
    Background task to log query for analytics.
    
    Args:
        query: The search query
        user_id: ID of the user who made the query
    """
    # Simulate async logging
    await asyncio.sleep(0.1)
    print(f"Logged query: '{query}' by user: {user_id}")
    # TODO: Implement actual query logging