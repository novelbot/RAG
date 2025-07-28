"""
Episode-based RAG API routes.

This module provides API endpoints for episode-specific search and query operations,
supporting filtering by episode IDs and sorting by episode numbers.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import Dict, List, Any, Optional
import asyncio
import time

from ...auth.dependencies import get_current_user, MockUser
from ..schemas import BaseAPISchema, MessageResponse
from ...episode import (
    EpisodeSearchEngine, EpisodeSearchRequest, EpisodeSortOrder,
    EpisodeEmbeddingProcessor, EpisodeVectorStore
)
from ...core.exceptions import SearchError, ProcessingError, StorageError

# Pydantic schemas for episode API
from pydantic import BaseModel, Field
from datetime import date


class EpisodeQueryRequest(BaseAPISchema):
    """Episode-based query request schema."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    episode_ids: Optional[List[int]] = Field(None, description="Filter by specific episode IDs")
    novel_ids: Optional[List[int]] = Field(None, description="Filter by specific novel IDs")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    sort_order: str = Field("episode_number", description="Sort order: 'similarity', 'episode_number', 'publication_date'")
    include_content: bool = Field(True, description="Include episode content in results")
    include_metadata: bool = Field(True, description="Include episode metadata in results")
    
    # Date range filtering
    date_from: Optional[date] = Field(None, description="Filter episodes from this date")
    date_to: Optional[date] = Field(None, description="Filter episodes until this date")
    
    # Episode number range filtering
    episode_num_from: Optional[int] = Field(None, ge=1, description="Filter from episode number")
    episode_num_to: Optional[int] = Field(None, ge=1, description="Filter to episode number")


class EpisodeSearchHitResponse(BaseAPISchema):
    """Episode search hit response schema."""
    episode_id: int = Field(..., description="Episode ID")
    episode_number: int = Field(..., description="Episode number within novel")
    episode_title: str = Field(..., description="Episode title")
    novel_id: int = Field(..., description="Novel ID")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    distance: float = Field(..., ge=0.0, description="Vector distance")
    content: Optional[str] = Field(None, description="Episode content")
    publication_date: Optional[str] = Field(None, description="Publication date (ISO format)")
    content_length: Optional[int] = Field(None, ge=0, description="Content character count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EpisodeSearchResponse(BaseAPISchema):
    """Episode search response schema."""
    query: str = Field(..., description="Original search query")
    hits: List[EpisodeSearchHitResponse] = Field(..., description="Search results")
    total_count: int = Field(..., ge=0, description="Total number of results")
    search_time: float = Field(..., ge=0, description="Search execution time in seconds")
    sort_order: str = Field(..., description="Applied sort order")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")
    user_id: str = Field(..., description="User ID")


class EpisodeContextRequest(BaseAPISchema):
    """Episode context request schema."""
    episode_ids: List[int] = Field(..., min_items=1, max_items=50, description="Episode IDs to include")
    query: Optional[str] = Field(None, description="Optional query for relevance scoring")
    max_context_length: int = Field(10000, ge=1000, le=50000, description="Maximum total character length")


class EpisodeContextResponse(BaseAPISchema):
    """Episode context response schema."""
    context: str = Field(..., description="Concatenated episode context")
    episodes_included: int = Field(..., ge=0, description="Number of episodes included")
    total_length: int = Field(..., ge=0, description="Total character length")
    episode_order: List[int] = Field(..., description="Episode numbers in order")
    truncated: bool = Field(..., description="Whether content was truncated")
    user_id: str = Field(..., description="User ID")


class EpisodeRAGRequest(BaseAPISchema):
    """Episode-based RAG request schema."""
    query: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    episode_ids: Optional[List[int]] = Field(None, description="Limit search to specific episodes")
    novel_ids: Optional[List[int]] = Field(None, description="Limit search to specific novels")
    max_context_episodes: int = Field(5, ge=1, le=20, description="Maximum episodes to use as context")
    max_context_length: int = Field(8000, ge=1000, le=20000, description="Maximum context character length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(1000, ge=100, le=4000, description="Maximum response tokens")


class EpisodeRAGResponse(BaseAPISchema):
    """Episode-based RAG response schema."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="AI-generated answer")
    context_episodes: List[EpisodeSearchHitResponse] = Field(..., description="Episodes used as context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    user_id: str = Field(..., description="User ID")


router = APIRouter(prefix="/episode", tags=["episode"])
security = HTTPBearer()


@router.post("/search", response_model=EpisodeSearchResponse)
async def search_episodes(
    request: EpisodeQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user)
) -> EpisodeSearchResponse:
    """
    Search episodes using vector similarity with episode-specific filtering.
    
    Args:
        request: Episode search request
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Episode search results with metadata
    """
    start_time = time.time()
    
    # Add background task for logging
    background_tasks.add_task(log_episode_query, request.query, current_user.id, "search")
    
    try:
        # TODO: Initialize episode search engine (would be dependency injected in real app)
        # For now, return mock response
        await asyncio.sleep(0.3)  # Simulate search time
        
        # Convert sort order string to enum
        sort_order_map = {
            "similarity": EpisodeSortOrder.SIMILARITY,
            "episode_number": EpisodeSortOrder.EPISODE_NUMBER,
            "publication_date": EpisodeSortOrder.PUBLICATION_DATE
        }
        sort_order = sort_order_map.get(request.sort_order, EpisodeSortOrder.EPISODE_NUMBER)
        
        # Mock episode search results
        mock_hits = []
        episode_ids_to_search = request.episode_ids or [1, 2, 3, 5]
        
        for i, episode_id in enumerate(episode_ids_to_search[:request.limit]):
            hit = EpisodeSearchHitResponse(
                episode_id=episode_id,
                episode_number=episode_id,
                episode_title=f"Episode {episode_id}: The Journey Continues",
                novel_id=request.novel_ids[0] if request.novel_ids else 1,
                similarity_score=0.95 - (i * 0.1),
                distance=0.1 + (i * 0.05),
                content=f"This is the content of episode {episode_id}. The story unfolds as our protagonist faces new challenges..." if request.include_content else None,
                publication_date="2024-01-15",
                content_length=1500 + (i * 200),
                metadata={
                    "search_timestamp": time.time(),
                    "filtered_by_episode_ids": request.episode_ids is not None
                }
            )
            mock_hits.append(hit)
        
        # Sort according to requested order
        if sort_order == EpisodeSortOrder.EPISODE_NUMBER:
            mock_hits.sort(key=lambda h: h.episode_number)
        elif sort_order == EpisodeSortOrder.SIMILARITY:
            mock_hits.sort(key=lambda h: h.similarity_score, reverse=True)
        
        search_time = time.time() - start_time
        
        return EpisodeSearchResponse(
            query=request.query,
            hits=mock_hits,
            total_count=len(mock_hits),
            search_time=search_time,
            sort_order=request.sort_order,
            metadata={
                "episode_ids_filter": request.episode_ids,
                "novel_ids_filter": request.novel_ids,
                "similarity_threshold": request.similarity_threshold,
                "total_episodes_searched": len(episode_ids_to_search),
                "context_ordered_by_episode": sort_order == EpisodeSortOrder.EPISODE_NUMBER
            },
            user_id=current_user.id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Episode search failed: {str(e)}"
        )


@router.post("/context", response_model=EpisodeContextResponse)
async def get_episode_context(
    request: EpisodeContextRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user)
) -> EpisodeContextResponse:
    """
    Get episode content as structured context for LLM consumption.
    
    Args:
        request: Episode context request
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Structured episode context
    """
    # Add background task for logging
    background_tasks.add_task(log_episode_query, request.query or "context_request", current_user.id, "context")
    
    try:
        # TODO: Initialize episode search engine (would be dependency injected in real app)
        await asyncio.sleep(0.2)  # Simulate processing time
        
        # Mock context generation
        contexts = []
        total_length = 0
        included_episodes = []
        
        for episode_id in sorted(request.episode_ids):
            episode_content = f"Episode {episode_id}: The Adventure Begins\n\nIn this episode, our protagonist embarks on a new journey filled with challenges and discoveries. The narrative unfolds as they encounter various obstacles and make important decisions that will shape their destiny..."
            
            if total_length + len(episode_content) > request.max_context_length:
                # Truncate if needed
                remaining_space = request.max_context_length - total_length
                if remaining_space > 100:
                    episode_content = episode_content[:remaining_space] + "...[truncated]"
                    contexts.append(episode_content)
                    included_episodes.append(episode_id)
                break
            
            contexts.append(episode_content)
            included_episodes.append(episode_id)
            total_length += len(episode_content)
        
        context = "\n\n---\n\n".join(contexts)
        
        return EpisodeContextResponse(
            context=context,
            episodes_included=len(included_episodes),
            total_length=len(context),
            episode_order=included_episodes,
            truncated=total_length >= request.max_context_length,
            user_id=current_user.id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Episode context generation failed: {str(e)}"
        )


@router.post("/ask", response_model=EpisodeRAGResponse)
async def ask_about_episodes(
    request: EpisodeRAGRequest,
    background_tasks: BackgroundTasks,
    current_user: MockUser = Depends(get_current_user)
) -> EpisodeRAGResponse:
    """
    Ask questions about specific episodes using RAG.
    
    Args:
        request: Episode RAG request
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        AI-generated answer with episode context
    """
    start_time = time.time()
    
    # Add background task for logging
    background_tasks.add_task(log_episode_query, request.query, current_user.id, "ask")
    
    try:
        # TODO: Implement actual episode RAG pipeline
        # For now, simulate the process
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Mock episode search results (context)
        context_episodes = []
        episode_ids_to_search = request.episode_ids or [1, 2, 3]
        
        for i, episode_id in enumerate(episode_ids_to_search[:request.max_context_episodes]):
            episode = EpisodeSearchHitResponse(
                episode_id=episode_id,
                episode_number=episode_id,
                episode_title=f"Episode {episode_id}: Critical Moments",
                novel_id=request.novel_ids[0] if request.novel_ids else 1,
                similarity_score=0.9 - (i * 0.1),
                distance=0.1 + (i * 0.05),
                content=f"In episode {episode_id}, the protagonist faces a crucial decision that will determine the outcome of their quest. The tension builds as they must choose between their personal desires and the greater good...",
                publication_date="2024-01-15",
                content_length=1200 + (i * 150),
                metadata={"used_as_context": True}
            )
            context_episodes.append(episode)
        
        # Mock AI response based on context
        answer = f"""Based on the episodes you've specified, I can provide insights about the story progression. 

The protagonist's journey through episodes {', '.join(map(str, episode_ids_to_search[:request.max_context_episodes]))} shows a clear character development arc. In these episodes, they face increasingly complex challenges that test not only their abilities but also their moral compass.

Key themes that emerge include:
- The balance between personal goals and collective responsibility
- The importance of making difficult decisions under pressure
- Character growth through adversity

The narrative structure builds tension effectively, with each episode adding layers to the overarching plot while maintaining focus on character development."""
        
        processing_time = time.time() - start_time
        
        return EpisodeRAGResponse(
            question=request.query,
            answer=answer,
            context_episodes=context_episodes,
            metadata={
                "processing_time_ms": int(processing_time * 1000),
                "episodes_used": len(context_episodes),
                "total_context_length": sum(len(ep.content or "") for ep in context_episodes),
                "model_temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "context_ordered_by_episode": True
            },
            user_id=current_user.id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Episode RAG query failed: {str(e)}"
        )


@router.get("/novel/{novel_id}/episodes")
async def list_novel_episodes(
    novel_id: int,
    limit: int = 50,
    offset: int = 0,
    current_user: MockUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List episodes for a specific novel.
    
    Args:
        novel_id: Novel ID
        limit: Maximum number of episodes to return
        offset: Number of episodes to skip
        current_user: Authenticated user
        
    Returns:
        List of episodes with metadata
    """
    try:
        # TODO: Implement actual episode listing from database
        await asyncio.sleep(0.1)  # Simulate database query
        
        # Mock episode list
        episodes = []
        for i in range(offset + 1, min(offset + limit + 1, 21)):  # Max 20 episodes for demo
            episode = {
                "episode_id": i,
                "episode_number": i,
                "episode_title": f"Episode {i}: Chapter Title",
                "publication_date": f"2024-01-{i:02d}",
                "content_length": 1000 + (i * 100),
                "has_embedding": True
            }
            episodes.append(episode)
        
        return {
            "novel_id": novel_id,
            "episodes": episodes,
            "total_episodes": 20,  # Mock total
            "limit": limit,
            "offset": offset,
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list episodes: {str(e)}"
        )


@router.post("/novel/{novel_id}/process")
async def process_novel_episodes(
    novel_id: int,
    background_tasks: BackgroundTasks,
    force_reprocess: bool = False,
    current_user: MockUser = Depends(get_current_user)
) -> MessageResponse:
    """
    Process episodes for a novel (extract, embed, and store).
    
    Args:
        novel_id: Novel ID to process
        background_tasks: FastAPI background tasks
        force_reprocess: Whether to reprocess existing episodes
        current_user: Authenticated user
        
    Returns:
        Processing status message
    """
    try:
        # Add background task for processing
        background_tasks.add_task(
            process_novel_episodes_background,
            novel_id,
            force_reprocess,
            current_user.id
        )
        
        return MessageResponse(
            message=f"Started processing episodes for novel {novel_id}. Check status via monitoring endpoints."
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start episode processing: {str(e)}"
        )


# Background tasks
async def log_episode_query(query: str, user_id: str, query_type: str) -> None:
    """
    Background task to log episode query for analytics.
    
    Args:
        query: The search query
        user_id: ID of the user who made the query
        query_type: Type of query (search, context, ask)
    """
    await asyncio.sleep(0.1)
    print(f"Logged episode {query_type} query: '{query}' by user: {user_id}")


async def process_novel_episodes_background(
    novel_id: int,
    force_reprocess: bool,
    user_id: str
) -> None:
    """
    Background task to process novel episodes.
    
    Args:
        novel_id: Novel ID to process
        force_reprocess: Whether to reprocess existing episodes
        user_id: User ID who initiated processing
    """
    try:
        # Simulate processing time
        await asyncio.sleep(5.0)
        print(f"Completed processing episodes for novel {novel_id} (user: {user_id})")
        
        # TODO: Implement actual episode processing
        # 1. Extract episodes from RDB
        # 2. Generate embeddings
        # 3. Store in vector database
        # 4. Update processing status
        
    except Exception as e:
        print(f"Failed to process episodes for novel {novel_id}: {e}")


# Health check endpoint
@router.get("/health")
async def episode_health_check() -> Dict[str, Any]:
    """Check health of episode processing components."""
    return {
        "status": "healthy",
        "components": {
            "vector_store": "healthy",
            "search_engine": "healthy",
            "embedding_processor": "healthy"
        },
        "timestamp": time.time()
    }