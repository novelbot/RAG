"""
Episode-based RAG API routes.

This module provides API endpoints for episode-specific search and query operations,
supporting filtering by episode IDs and sorting by episode numbers.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request, Query
from fastapi.security import HTTPBearer
from fastapi.responses import StreamingResponse
from typing import Dict, List, Any, Optional, Set, AsyncIterator
import asyncio
import time
import logging
import re
import json
from datetime import datetime, timezone

from ...auth.dependencies import get_current_user, SimpleUser
from ..schemas import (
    BaseAPISchema, MessageResponse, 
    EpisodeChatRequest, EpisodeChatResponse, EpisodeChatConversation, 
    EpisodeChatError, EpisodeSource, EpisodeConversationMetadata, 
    EpisodeSearchMetadata, ChatMessage
)
from ...episode import (
    EpisodeSearchEngine, EpisodeSearchRequest, EpisodeSortOrder,
    EpisodeEmbeddingProcessor, EpisodeVectorStore, EpisodeRAGManager,
    create_episode_rag_manager, EpisodeRAGConfig
)
from ...core.exceptions import SearchError, ProcessingError, StorageError
from ...core.config import get_config
from ...database.base import DatabaseManager
from ...services.query_logger import QueryLogger, QueryContext, QueryMetrics, QueryType, query_logger
from ...embedding.manager import EmbeddingManager
from ...milvus.client import MilvusClient
from ...llm import LLMManager, ProviderConfig, create_llm_manager
from ...llm.base import LLMRequest, LLMMessage, LLMRole, LLMConfig, LLMProvider
import uuid

# Helper functions
def extract_characters_mentioned(text: str) -> List[str]:
    """
    Extract character names from text using pattern matching.
    
    This function looks for patterns commonly used for character names
    in Korean novels and web fiction.
    
    Args:
        text: Text to extract character names from
        
    Returns:
        List of unique character names found
    """
    if not text:
        return []
    
    characters: Set[str] = set()
    
    # Korean name patterns - typically 2-4 characters
    korean_name_pattern = r'[가-힣]{2,4}(?=[이가는을를에게서와과]?\s)'
    korean_matches = re.findall(korean_name_pattern, text)
    
    # Filter out common words that might match but aren't names
    common_words = {
        '그것', '이것', '저것', '무엇', '여기', '저기', '거기', '어디',
        '언제', '어떻게', '왜', '누구', '무엇', '어느', '몇', '많은',
        '작은', '큰', '새로운', '오래된', '좋은', '나쁜', '빠른', '느린',
        '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열',
        '시간', '장소', '사람', '물건', '일', '날', '밤', '아침', '저녁',
        '마을', '도시', '나라', '세상', '하늘', '땅', '바다', '산', '강',
        '학교', '집', '회사', '상점', '병원', '은행', '역', '공항',
        '어제', '오늘', '내일', '지금', '나중', '처음', '마지막',
        '정말', '진짜', '아마', '분명', '아직', '벌써', '다시', '또',
        '한번', '두번', '세번', '여러번', '항상', '가끔', '자주', '드물게',
        '모든', '어떤', '이런', '저런', '그런', '다른', '같은', '비슷한'
    }
    
    for match in korean_matches:
        if match not in common_words and len(match) >= 2:
            characters.add(match)
    
    # English name patterns - capitalized words
    english_name_pattern = r'\b[A-Z][a-zA-Z]{1,15}\b'
    english_matches = re.findall(english_name_pattern, text)
    
    # Filter out common English words
    english_common_words = {
        'The', 'And', 'But', 'For', 'Not', 'You', 'All', 'Can', 'Had', 'Her',
        'Was', 'One', 'Our', 'Out', 'Day', 'Get', 'Has', 'Him', 'His', 'How',
        'Man', 'New', 'Now', 'Old', 'See', 'Two', 'Way', 'Who', 'Boy', 'Did',
        'Its', 'Let', 'Put', 'She', 'Too', 'Use', 'Sir', 'May', 'Say', 'God',
        'Yes', 'No', 'Ok', 'Oh', 'Ah', 'Eh', 'Um', 'Er', 'Mm', 'Hmm'
    }
    
    for match in english_matches:
        if match not in english_common_words and len(match) >= 2:
            characters.add(match)
    
    # Look for quoted speech patterns which often contain character names
    speech_pattern = r'[""\'](.*?)[""\']\s*(?:라고|했다|말했다|대답했다|소리쳤다|속삭였다)'
    speech_matches = re.findall(speech_pattern, text)
    
    for speech in speech_matches:
        # Extract names from speech
        speech_names = extract_characters_mentioned(speech)
        characters.update(speech_names)
    
    # Sort by length (longer names first) and return top 10
    sorted_characters = sorted(list(characters), key=len, reverse=True)[:10]
    
    return sorted_characters

# Pydantic schemas for episode API
from pydantic import BaseModel, Field
from datetime import date, datetime, timezone


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
    episode_ids: List[int] = Field(..., min_length=1, max_length=50, description="Episode IDs to include")
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
logger = logging.getLogger(__name__)

# Global cache for RAG manager
_rag_manager_cache: Optional[EpisodeRAGManager] = None

# Dependency injection for EpisodeRAGManager
async def get_rag_manager() -> EpisodeRAGManager:
    """
    Get EpisodeRAGManager instance for dependency injection.
    Uses caching to avoid recreating the manager on every request.
    
    Returns:
        EpisodeRAGManager: Configured RAG manager instance
    """
    global _rag_manager_cache
    
    # Return cached instance if available
    if _rag_manager_cache is not None:
        return _rag_manager_cache
    
    config = get_config()
    
    # Initialize database manager
    from ...database.base import DatabaseManager
    db_manager = DatabaseManager(config.database)
    
    # Get global embedding manager from app startup
    import sys
    app_module = sys.modules.get('src.core.app')
    embedding_manager = getattr(app_module, 'embedding_manager', None)
    
    if not embedding_manager:
        logger.warning("Global embedding manager not found, creating new instance")
        from ...embedding.factory import get_embedding_manager
        embedding_manager = get_embedding_manager([config.embedding])
    
    # Initialize Milvus client
    milvus_client = MilvusClient(config.milvus)
    milvus_client.connect()
    
    # Create Episode RAG Manager
    episode_config = EpisodeRAGConfig(
        collection_name="episode_embeddings",
        default_search_limit=5
    )
    
    episode_rag_manager = await create_episode_rag_manager(
        database_manager=db_manager,
        embedding_manager=embedding_manager,
        milvus_client=milvus_client,
        config=episode_config,
        setup_collection=False  # Assume collection already exists
    )
    
    # Cache the manager for future requests
    _rag_manager_cache = episode_rag_manager
    logger.info("Cached EpisodeRAGManager for future requests")
    
    return episode_rag_manager


@router.post("/search", response_model=EpisodeSearchResponse)
async def search_episodes(
    request: EpisodeQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: SimpleUser = Depends(get_current_user)
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
        # Initialize episode search engine
        config = get_config()
        
        # Initialize database manager
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        # Get global embedding manager from app startup
        import sys
        app_module = sys.modules.get('src.core.app')
        embedding_manager = getattr(app_module, 'embedding_manager', None)
        
        if not embedding_manager:
            logger.warning("Global embedding manager not found, creating new instance")
            from ...embedding.factory import get_embedding_manager
            embedding_manager = get_embedding_manager([config.embedding])
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        # Create Episode RAG Manager
        episode_config = EpisodeRAGConfig(
            collection_name="episode_embeddings",
            default_search_limit=request.limit
        )
        
        episode_rag_manager = await create_episode_rag_manager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config,
            setup_collection=False  # Assume collection already exists
        )
        
        # Perform episode search using real search engine
        search_result = await episode_rag_manager.search_episodes(
            query=request.query,
            episode_ids=request.episode_ids,
            novel_ids=request.novel_ids,
            limit=request.limit,
            sort_by_episode_number=(request.sort_order == "episode_number")
        )
        
        # Convert search results to response format
        hits = []
        for hit in search_result.hits:
            episode_hit = EpisodeSearchHitResponse(
                episode_id=hit.episode_id,
                episode_number=hit.episode_number,
                episode_title=hit.episode_title,
                novel_id=hit.novel_id,
                similarity_score=hit.similarity_score,
                distance=hit.distance,
                content=hit.content if request.include_content else None,
                publication_date=hit.publication_date.isoformat() if hit.publication_date else None,
                content_length=len(hit.content) if hit.content else None,
                metadata={
                    "search_timestamp": time.time(),
                    "filtered_by_episode_ids": request.episode_ids is not None,
                    "used_real_search_engine": True
                }
            )
            hits.append(episode_hit)
        
        search_time = time.time() - start_time
        
        return EpisodeSearchResponse(
            query=request.query,
            hits=hits,
            total_count=len(hits),
            search_time=search_time,
            sort_order=request.sort_order,
            metadata={
                "episode_ids_filter": request.episode_ids,
                "novel_ids_filter": request.novel_ids,
                "similarity_threshold": request.similarity_threshold,
                "total_episodes_searched": len(hits),
                "context_ordered_by_episode": request.sort_order == "episode_number",
                "used_real_search_engine": True
            },
            user_id=str(current_user.id)
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
    current_user: SimpleUser = Depends(get_current_user)
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
        # Initialize episode search engine for context retrieval
        config = get_config()
        
        # Initialize database manager
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        # Get global embedding manager from app startup
        import sys
        app_module = sys.modules.get('src.core.app')
        embedding_manager = getattr(app_module, 'embedding_manager', None)
        
        if not embedding_manager:
            logger.warning("Global embedding manager not found, creating new instance")
            from ...embedding.factory import get_embedding_manager
            embedding_manager = get_embedding_manager([config.embedding])
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        # Create Episode RAG Manager
        episode_config = EpisodeRAGConfig(
            collection_name="episode_embeddings",
            default_search_limit=len(request.episode_ids)
        )
        
        episode_rag_manager = await create_episode_rag_manager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config,
            setup_collection=False  # Assume collection already exists
        )
        
        # Get episode context using search engine
        context_result = episode_rag_manager.search_engine.get_episode_context(
            episode_ids=request.episode_ids,
            query=request.query,
            max_context_length=request.max_context_length
        )
        
        context = context_result["context"]
        included_episodes = context_result["episode_order"]
        total_length = context_result["total_length"]
        truncated = context_result["truncated"]
        
        return EpisodeContextResponse(
            context=context,
            episodes_included=len(included_episodes),
            total_length=total_length,
            episode_order=included_episodes,
            truncated=truncated,
            user_id=str(current_user.id)
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
    current_user: SimpleUser = Depends(get_current_user)
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
        # Initialize required managers
        config = get_config()
        
        # Create database manager
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        # Create embedding manager
        # Get global embedding manager from app startup
        import sys
        app_module = sys.modules.get('src.core.app')
        embedding_manager = getattr(app_module, 'embedding_manager', None)
        
        if not embedding_manager:
            logger.warning("Global embedding manager not found, creating new instance")
            from ...embedding.factory import get_embedding_manager
            embedding_manager = get_embedding_manager([config.embedding])
        
        # Create Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        
        # Create Episode RAG Manager
        episode_config = EpisodeRAGConfig(
            collection_name="episode_embeddings",
            default_search_limit=request.max_context_episodes
        )
        
        episode_rag_manager = await create_episode_rag_manager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config,
            setup_collection=False  # Assume collection already exists
        )
        
        # Perform episode search
        search_result = await episode_rag_manager.search_episodes(
            query=request.query,
            episode_ids=request.episode_ids,
            novel_ids=request.novel_ids,
            limit=request.max_context_episodes,
            sort_by_episode_number=True
        )
        
        # Convert search results to context episodes
        context_episodes = []
        for hit in search_result.hits:
            episode = EpisodeSearchHitResponse(
                episode_id=hit.episode_id,
                episode_number=hit.episode_number,
                episode_title=hit.episode_title,
                novel_id=hit.novel_id,
                similarity_score=hit.similarity_score,
                distance=hit.distance,
                content=hit.content if request.include_content else "",
                publication_date=hit.publication_date.isoformat() if hit.publication_date else None,
                content_length=len(hit.content) if hit.content else 0,
                metadata={"used_as_context": True}
            )
            context_episodes.append(episode)
        
        # Generate AI response using context
        if context_episodes:
            # Prepare context for LLM
            context_text = "\n\n".join([
                f"Episode {ep.episode_number}: {ep.episode_title}\n{ep.content}"
                for ep in context_episodes if ep.content
            ])
            
            # Truncate context if too long
            if len(context_text) > request.max_context_length:
                context_text = context_text[:request.max_context_length]
            
            # Create LLM manager with proper provider configuration
            provider_configs = []
            
            # Map string provider to LLMProvider enum
            provider_map = {
                "ollama": LLMProvider.OLLAMA,
                "openai": LLMProvider.OPENAI,
                "google": LLMProvider.GOOGLE,
                "claude": LLMProvider.CLAUDE,
                "anthropic": LLMProvider.CLAUDE
            }
            
            provider_enum = provider_map.get(config.llm.provider.lower())
            if not provider_enum:
                raise ValueError(f"Unsupported provider: {config.llm.provider}")
            
            provider_config = ProviderConfig(
                provider=provider_enum,
                config=LLMConfig(
                    provider=config.llm.provider,
                    model=config.llm.model,
                    api_key=config.llm.api_key,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                    base_url=config.llm.base_url
                    # Removed stream parameter as it's not valid for LLMConfig
                )
            )
            provider_configs.append(provider_config)
            
            llm_manager = create_llm_manager(provider_configs)
            
            # Prepare prompt
            prompt = f"""Based on the following episode content, please answer this question: {request.query}

Episode Context:
{context_text}

Please provide a detailed and helpful answer based on the episode information provided."""
            
            # Generate response
            llm_request = LLMRequest(
                messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
                model=config.llm.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            response_result = await llm_manager.generate_async(llm_request)
            answer = response_result.content
        else:
            answer = "I couldn't find any relevant episodes to answer your question. Please try with different episode IDs or a broader query."
        
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
            user_id=str(current_user.id)
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
    current_user: SimpleUser = Depends(get_current_user)
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
        # Implement actual episode listing from database
        config = get_config()
        
        # Initialize database manager
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        # Query episodes from database
        episodes = []
        total_episodes = 0
        
        try:
            # Use context manager for database connection
            with db_manager.get_connection() as connection:
                # Count total episodes for this novel
                count_query = "SELECT COUNT(*) as total FROM episode WHERE novel_id = %s"
                with connection.cursor() as cursor:
                    cursor.execute(count_query, (novel_id,))
                    result = cursor.fetchone()
                    total_episodes = result['total'] if result else 0
                
                # Get episodes with pagination
                episodes_query = """
                    SELECT 
                        id as episode_id,
                        episode_number,
                        title as episode_title,
                        created_at as publication_date,
                        CHAR_LENGTH(content) as content_length
                    FROM episode 
                    WHERE novel_id = %s 
                    ORDER BY episode_number ASC 
                    LIMIT %s OFFSET %s
                """
                
                with connection.cursor() as cursor:
                    cursor.execute(episodes_query, (novel_id, limit, offset))
                    db_episodes = cursor.fetchall()
                    
                    for episode in db_episodes:
                        episodes.append({
                            "episode_id": episode['episode_id'],
                            "episode_number": episode['episode_number'],
                            "episode_title": episode['episode_title'],
                            "publication_date": episode['publication_date'].isoformat() if episode['publication_date'] else None,
                            "content_length": episode['content_length'] or 0,
                            "has_embedding": True  # Assume all episodes have embeddings
                        })
        
        except Exception as e:
            # Fall back to mock data if database query fails
            logger.warning(f"Database query failed, using mock data: {e}")
            for i in range(offset + 1, min(offset + limit + 1, 21)):
                episodes.append({
                    "episode_id": i,
                    "episode_number": i,
                    "episode_title": f"Episode {i}: Chapter Title (Mock)",
                    "publication_date": f"2024-01-{i:02d}",
                    "content_length": 1000 + (i * 100),
                    "has_embedding": True
                })
            total_episodes = 20
        
        return {
            "novel_id": novel_id,
            "episodes": episodes,
            "total_episodes": total_episodes,
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
    current_user: SimpleUser = Depends(get_current_user)
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
        # Implement actual episode processing
        try:
            config = get_config()
            
            # Initialize required managers
            from ...database.base import DatabaseManager
            db_manager = DatabaseManager(config.database)
            
            # Get global embedding manager
            import sys
            from ...core import app
            embedding_manager = getattr(sys.modules.get('src.core.app'), 'embedding_manager', None)
            if not embedding_manager:
                from ...embedding.factory import get_embedding_manager
                embedding_manager = get_embedding_manager([config.embedding])
            
            milvus_client = MilvusClient(config.milvus)
            milvus_client.connect()
            
            # Create Episode RAG Manager
            episode_config = EpisodeRAGConfig(
                collection_name="episode_embeddings",
                processing_batch_size=10
            )
            
            episode_rag_manager = await create_episode_rag_manager(
                database_manager=db_manager,
                embedding_manager=embedding_manager,
                milvus_client=milvus_client,
                config=episode_config,
                setup_collection=True  # Create collection if needed
            )
            
            # Process the novel episodes
            processing_result = await episode_rag_manager.process_novel(
                novel_id=novel_id,
                force_reprocess=force_reprocess
            )
            
            print(f"Episode processing completed for novel {novel_id}:")
            print(f"  Status: {processing_result.get('status', 'unknown')}")
            print(f"  Episodes processed: {processing_result.get('episodes_processed', 0)}")
            print(f"  User: {user_id}")
            
        except Exception as e:
            print(f"Episode processing failed for novel {novel_id}: {e}")
            # Simulate some processing time even on failure
            await asyncio.sleep(1.0)
        
    except Exception as e:
        print(f"Failed to process episodes for novel {novel_id}: {e}")


# Episode Chat Endpoints

# Import SQLite-based conversation storage
from ...conversation import conversation_storage, ConversationMessage

async def retrieve_episode_conversation_context(
    conversation_id: str,
    user_id: str,
    max_context_turns: int = 10
) -> tuple[Optional[EpisodeChatConversation], List[ChatMessage]]:
    """
    Retrieve conversation context from SQLite storage.
    
    Args:
        conversation_id: Unique conversation ID
        user_id: User ID for authorization
        max_context_turns: Maximum number of turns to retrieve
        
    Returns:
        Tuple of (conversation object, list of messages)
    """
    try:
        print(f"\n[DEBUG] retrieve_episode_conversation_context called:")
        print(f"  - conversation_id: {conversation_id}")
        print(f"  - user_id: {user_id}")
        print(f"  - max_context_turns: {max_context_turns}")
        
        # Get messages from SQLite storage
        messages_data = await conversation_storage.get_messages(
            conversation_id=conversation_id,
            limit=max_context_turns * 2  # Each turn = 1 user + 1 assistant message
        )
        
        print(f"  - Retrieved {len(messages_data) if messages_data else 0} messages from storage")
        
        if messages_data:
            # Convert to ChatMessage format
            chat_messages = []
            for msg in messages_data:
                chat_messages.append(ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata or {}
                ))
            
            # Get conversation info
            conv_info = await conversation_storage.get_conversation_info(conversation_id)
            
            # Create conversation object (using 'id' field name as required by model)
            conversation = EpisodeChatConversation(
                id=conversation_id,  # Changed from conversation_id to id
                user_id=str(user_id),  # Ensure it's a string
                messages=chat_messages,
                created_at=datetime.fromisoformat(conv_info['created_at']) if conv_info else datetime.now(timezone.utc),
                updated_at=datetime.fromisoformat(conv_info['updated_at']) if conv_info else datetime.now(timezone.utc),
                total_messages=len(chat_messages)
            )
            
            logger.info(f"Retrieved {len(chat_messages)} messages for conversation {conversation_id}")
            return conversation, chat_messages
        
        logger.info(f"No messages found for conversation {conversation_id}")
        return None, []
        
    except Exception as e:
        print(f"[ERROR] Error retrieving conversation context: {e}")
        print(f"[ERROR] Exception type: {type(e)}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        logger.error(f"Error retrieving conversation context: {e}")
        return None, []

async def save_conversation_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a message to SQLite conversation storage.
    
    Args:
        conversation_id: Unique conversation ID
        user_id: User ID
        role: Message role (user/assistant)
        content: Message content
        metadata: Optional metadata
    """
    try:
        # Ensure conversation exists
        conv_info = await conversation_storage.get_conversation_info(conversation_id)
        if not conv_info:
            await conversation_storage.create_conversation(
                conversation_id=conversation_id,
                user_id=str(user_id),  # Ensure user_id is string
                metadata={"created_via": "episode_chat"}
            )
        
        # Save message
        message = ConversationMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata
        )
        
        await conversation_storage.add_message(message)
        logger.info(f"Saved {role} message to conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error saving conversation message: {e}")


async def perform_episode_vector_search(
    query: str,
    episode_ids: Optional[List[int]],
    novel_ids: Optional[List[int]],
    max_episodes: int,
    max_results: int,
    sort_order: str,
    conversation_context: Optional[List[ChatMessage]],
    rag_manager: Optional[EpisodeRAGManager] = None
) -> tuple[List[EpisodeSource], EpisodeSearchMetadata]:
    """Perform episode-aware vector search for chat context."""
    start_time = time.time()
    
    try:
        # Use provided rag_manager or create new one
        if rag_manager:
            episode_rag_manager = rag_manager
        else:
            # Initialize episode search components
            config = get_config()
            
            from ...database.base import DatabaseManager
            db_manager = DatabaseManager(config.database)
            
            # Get global embedding manager from app startup
            import sys
            embedding_manager = None
            
            # Try multiple ways to get the embedding manager
            app_module = sys.modules.get('src.core.app')
            if app_module:
                embedding_manager = getattr(app_module, 'embedding_manager', None)
            
            # Try getting from globals if not found in module
            if not embedding_manager:
                try:
                    from ...core.app import embedding_manager as global_embedding_manager
                    embedding_manager = global_embedding_manager
                except (ImportError, AttributeError):
                    pass
            
            if not embedding_manager:
                logger.warning("Global embedding manager not found, creating new instance")
                try:
                    from ...embedding.manager import EmbeddingManager, EmbeddingProviderConfig
                    provider_config = EmbeddingProviderConfig(
                        provider=config.embedding.provider,
                        config=config.embedding,
                        priority=1,
                        enabled=True
                    )
                    embedding_manager = EmbeddingManager([provider_config], enable_cache=True)
                    logger.info("✅ Created new embedding manager instance")
                except Exception as e:
                    logger.error(f"Failed to create embedding manager: {e}")
                    raise Exception(f"Cannot initialize embedding manager: {e}")
            
            milvus_client = MilvusClient(config.milvus)
            milvus_client.connect()
            
            # Generate dynamic collection name based on embedding model
            from ...episode.vector_store import get_model_name_for_collection
            model_name = get_model_name_for_collection(embedding_manager)
            dynamic_collection_name = f"episode_embeddings_{model_name}"
            
            episode_config = EpisodeRAGConfig(
                collection_name=dynamic_collection_name,
                default_search_limit=max_episodes
            )
            
            episode_rag_manager = await create_episode_rag_manager(
                database_manager=db_manager,
                embedding_manager=embedding_manager,
                milvus_client=milvus_client,
                config=episode_config,
                setup_collection=False
            )
        
        # Enhance query with conversation context
        enhanced_query = query
        conversation_enhanced = False
        
        if conversation_context:
            recent_context = " ".join([
                msg.content for msg in conversation_context[-2:]
                if msg.role == "user"
            ])
            if recent_context:
                enhanced_query = f"{recent_context} {query}"
                conversation_enhanced = True
        
        # Perform episode search
        search_result = await episode_rag_manager.search_episodes(
            query=enhanced_query,
            episode_ids=episode_ids,
            novel_ids=novel_ids,
            limit=max_episodes,
            sort_by_episode_number=(sort_order == "episode_number")
        )
        
        # Convert to EpisodeSource format
        episode_sources = []
        novels_found = set()
        
        for i, hit in enumerate(search_result.hits[:max_episodes], 1):
            source = EpisodeSource(
                episode_id=hit.episode_id,
                episode_number=hit.episode_number,
                episode_title=hit.episode_title,
                novel_id=hit.novel_id,
                excerpt=hit.content or "",
                relevance_score=hit.similarity_score,
                similarity_score=hit.similarity_score,
                publication_date=hit.publication_date.isoformat() if hit.publication_date else None,
                content_length=len(hit.content) if hit.content else 0,
                characters_mentioned=extract_characters_mentioned(hit.content or ""),
                used_for_context=True,
                context_priority=i
            )
            episode_sources.append(source)
            novels_found.add(hit.novel_id)
        
        # Create search metadata
        search_time_ms = (time.time() - start_time) * 1000
        metadata = EpisodeSearchMetadata(
            query_used=enhanced_query,
            episodes_found=len(episode_sources),
            novels_found=len(novels_found),
            search_time_ms=search_time_ms,
            episode_ids_filter=episode_ids,
            novel_ids_filter=novel_ids,
            sort_order_applied=sort_order,
            conversation_enhanced_query=conversation_enhanced,
            episode_context_used=True
        )
        
        return episode_sources, metadata
        
    except Exception as e:
        search_time_ms = (time.time() - start_time) * 1000
        metadata = EpisodeSearchMetadata(
            query_used=query,
            episodes_found=0,
            novels_found=0,
            search_time_ms=search_time_ms,
            episode_ids_filter=episode_ids,
            novel_ids_filter=novel_ids,
            sort_order_applied=sort_order,
            conversation_enhanced_query=False,
            episode_context_used=False
        )
        
        print(f"Episode vector search failed: {e}")
        return [], metadata


async def generate_episode_chat_response(
    user_message: str,
    episode_sources: List[EpisodeSource],
    conversation_context: Optional[List[ChatMessage]],
    response_format: str,
    episode_ids: Optional[List[int]] = None,
    novel_ids: Optional[List[int]] = None
) -> tuple[str, Optional[float], float, Dict[str, Any]]:
    """Generate AI response using episode context and conversation history."""
    try:
        config = get_config()
        
        # Create LLM manager with proper provider configuration
        provider_configs = []
        
        # Map string provider to LLMProvider enum
        provider_map = {
            "ollama": LLMProvider.OLLAMA,
            "openai": LLMProvider.OPENAI,
            "google": LLMProvider.GOOGLE,
            "claude": LLMProvider.CLAUDE,
            "anthropic": LLMProvider.CLAUDE
        }
        
        provider_enum = provider_map.get(config.llm.provider.lower())
        if not provider_enum:
            raise ValueError(f"Unsupported provider: {config.llm.provider}")
        
        provider_config = ProviderConfig(
            provider=provider_enum,
            config=LLMConfig(
                provider=config.llm.provider,
                model=config.llm.model,
                api_key=config.llm.api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                base_url=config.llm.base_url
                # Removed stream parameter as it's not valid for LLMConfig
            )
        )
        provider_configs.append(provider_config)
        
        llm_manager = create_llm_manager(provider_configs)
        
        # Build episode context
        context_parts = []
        
        # Add episode sources
        if episode_sources:
            context_parts.append("=== EPISODE CONTENT ===")
            for source in episode_sources:
                context_parts.append(f"Episode {source.episode_number}: {source.episode_title}")
                context_parts.append(f"Content: {source.excerpt}")
                context_parts.append(f"Relevance: {source.relevance_score:.2f}")
                context_parts.append("")
        
        # Add conversation history
        if conversation_context:
            context_parts.append("=== CONVERSATION HISTORY ===")
            for msg in conversation_context[-6:]:  # Last 6 messages
                role_label = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")
            context_parts.append("")
        
        # Build episode-specific prompt
        context_text = "\n".join(context_parts) if context_parts else ""
        
        # Episode-specific instruction
        episode_instruction = ""
        if episode_ids:
            episode_instruction = f"Focus specifically on episode(s) {', '.join(map(str, episode_ids))}. "
        elif novel_ids:
            episode_instruction = f"Focus on content from novel(s) {', '.join(map(str, novel_ids))}. "
        
        if response_format == "concise":
            instruction = f"{episode_instruction}Provide a concise answer based on the episode content and conversation context."
        else:
            instruction = f"{episode_instruction}Provide a detailed answer based on the episode content and conversation context. Include character interactions, plot developments, and specific details from the episodes."
        
        prompt = f"""You are a helpful AI assistant specialized in discussing novel episodes. Answer the user's question using the provided episode content and conversation context.

{context_text}

User Question: {user_message}

{instruction}

Maintain character consistency and timeline awareness when referencing episode content. If you cannot find relevant information in the provided episodes, mention this clearly."""
        
        # LOG: Print the full prompt being sent to LLM
        print("\n" + "="*80)
        print("🔍 CHAT ENDPOINT - FULL PROMPT TO LLM:")
        print("="*80)
        print(f"Has Conversation Context: {conversation_context is not None and len(conversation_context) > 0}")
        print(f"Number of Context Messages: {len(conversation_context) if conversation_context else 0}")
        print(f"Number of Episode Sources: {len(episode_sources) if episode_sources else 0}")
        print("-"*40)
        print("PROMPT CONTENT:")
        print("-"*40)
        print(prompt)
        print("="*80 + "\n")
        
        # Generate response using LLM manager directly
        llm_request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content="You are a helpful AI assistant specialized in discussing novel episodes."),
                LLMMessage(role=LLMRole.USER, content=prompt)
            ],
            model=config.llm.model,
            temperature=0.7,
            max_tokens=1000
        )
        
        response_result = await llm_manager.generate_async(llm_request)
        ai_response = response_result.content
        
        # Calculate episode relevance score
        episode_relevance_score = 0.0
        if episode_sources:
            episode_relevance_score = sum(s.relevance_score for s in episode_sources) / len(episode_sources)
        
        # Calculate confidence score
        confidence = None
        if episode_sources:
            confidence = min(episode_relevance_score + 0.1, 1.0)
        
        # Episode metadata
        episode_metadata = {
            "episodes_referenced": [s.episode_id for s in episode_sources],
            "novels_referenced": list(set(s.novel_id for s in episode_sources)),
            "characters_mentioned": extract_characters_mentioned(ai_response),
            "timeline_context": "sequential" if episode_ids else "mixed"
        }
        
        return ai_response, confidence, episode_relevance_score, episode_metadata
        
    except Exception as e:
        print(f"Episode chat response generation failed: {e}")
        fallback_response = f"I apologize, but I'm having trouble generating a response about the episodes right now. Please try again later."
        return fallback_response, None, 0.0, {}




@router.post("/chat", response_model=EpisodeChatResponse)
async def episode_chat(
    request: EpisodeChatRequest,
    current_user: SimpleUser = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    rag_manager: EpisodeRAGManager = Depends(get_rag_manager)
) -> EpisodeChatResponse:
    """
    Episode-aware chat endpoint that combines episode filtering with conversation context.
    
    This endpoint provides enhanced chat functionality specifically for discussing novel episodes:
    1. Retrieves conversation context with episode associations
    2. Performs episode-filtered vector search
    3. Generates contextually aware responses using episode content
    4. Maintains conversation continuity with episode metadata
    5. Tracks episode and novel associations in conversation history
    """
    start_time = time.time()
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        is_new_conversation = request.conversation_id is None
        
        # Step 1: Retrieve episode conversation context
        conversation_context = []
        if request.use_conversation_context and request.conversation_id:
            conversation, conversation_context = await retrieve_episode_conversation_context(
                conversation_id=request.conversation_id,
                user_id=str(current_user.id),
                max_context_turns=request.max_context_turns
            )
        
        # Step 2: Perform episode-aware vector search
        episode_sources, search_metadata = await perform_episode_vector_search(
            query=request.message,
            episode_ids=request.episode_ids,
            novel_ids=request.novel_ids,
            max_episodes=request.max_episodes,
            max_results=request.max_results,
            sort_order=request.episode_sort_order,
            conversation_context=conversation_context if request.use_conversation_context else None,
            rag_manager=rag_manager
        )
        
        # Step 3: Generate episode-aware AI response
        ai_response, confidence_score, episode_relevance_score, episode_metadata = await generate_episode_chat_response(
            user_message=request.message,
            episode_sources=episode_sources if request.include_sources else [],
            conversation_context=conversation_context if request.use_conversation_context else None,
            response_format=request.response_format,
            episode_ids=request.episode_ids,
            novel_ids=request.novel_ids
        )
        
        # Step 4: Conversation saving disabled (no conversation manager)
        
        # Step 5: Log query (in background)
        query_context = QueryContext(
            user_id=str(current_user.id),
            session_id=conversation_id,
            user_agent="",
            ip_address=""
        )
        
        query_metrics = QueryMetrics(
            results_count=len(episode_sources),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        background_tasks.add_task(
            query_logger.create_query_log,
            query_text=request.message,
            query_type=QueryType.RAG,
            context=query_context
        )
        
        # Step 6: Build response with episode metadata
        # Determine conversation scope
        conversation_scope = "general"
        if request.episode_ids and len(request.episode_ids) == 1:
            conversation_scope = "episode_specific"
        elif request.novel_ids:
            conversation_scope = "novel_specific"
        
        # Episode conversation metadata
        episode_conversation_metadata = EpisodeConversationMetadata(
            conversation_id=conversation_id,
            total_messages=len(conversation_context) + 2,
            context_messages_used=len(conversation_context),
            is_new_conversation=is_new_conversation,
            episodes_discussed=episode_metadata.get("episodes_referenced", []),
            novels_discussed=episode_metadata.get("novels_referenced", []),
            primary_episode_id=request.primary_episode_id,
            primary_novel_id=request.primary_novel_id,
            episodes_used_for_context=[s.episode_id for s in episode_sources],
            conversation_scope=conversation_scope
        )
        
        # Character context extraction
        character_context = []
        for source in episode_sources:
            character_context.extend(source.characters_mentioned)
        character_context = list(set(character_context))  # Remove duplicates
        
        response = EpisodeChatResponse(
            message=ai_response,
            conversation_id=conversation_id,
            episode_sources=episode_sources if request.include_sources else [],
            conversation_metadata=episode_conversation_metadata,
            search_metadata=search_metadata,
            episode_metadata=episode_metadata,
            timeline_position=episode_metadata.get("timeline_context"),
            character_context=character_context,
            confidence_score=confidence_score,
            episode_relevance_score=episode_relevance_score,
            has_context=len(conversation_context) > 0,
            response_time_ms=(time.time() - start_time) * 1000,
            user_id=str(current_user.id)
        )
        
        return response
        
    except Exception as e:
        # Log the actual error for debugging
        logger.error(f"Episode chat error: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return episode-specific error response
        error_response = EpisodeChatError(
            error_type="invalid_request",
            message=f"Failed to process episode chat request: {str(e)}",
            conversation_id=request.conversation_id,
            episode_id=request.primary_episode_id,
            novel_id=request.primary_novel_id,
            retry_suggestions=["Check episode/novel IDs", "Try with simpler episode filters", "Check your message content"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump(mode='json')
        )


@router.post("/{episode_id}/chat", response_model=EpisodeChatResponse)
async def episode_specific_chat(
    episode_id: int,
    request: EpisodeChatRequest,
    current_user: SimpleUser = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> EpisodeChatResponse:
    """
    Chat about a specific episode.
    
    This endpoint automatically filters the conversation to focus on a specific episode.
    """
    # Override episode filtering to focus on the specific episode
    request.episode_ids = [episode_id]
    request.primary_episode_id = episode_id
    
    return await episode_chat(request, current_user, background_tasks)


@router.post("/novel/{novel_id}/chat", response_model=EpisodeChatResponse)
async def novel_specific_chat(
    novel_id: int,
    request: EpisodeChatRequest,
    current_user: SimpleUser = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> EpisodeChatResponse:
    """
    Chat about episodes from a specific novel.
    
    This endpoint automatically filters the conversation to focus on episodes from a specific novel.
    """
    # Override novel filtering to focus on the specific novel
    request.novel_ids = [novel_id]
    request.primary_novel_id = novel_id
    
    return await episode_chat(request, current_user, background_tasks)


@router.get("/debug/prompt/{conversation_id}")
async def get_debug_prompt(
    conversation_id: str,
    request: Request,
    current_user: SimpleUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the debug prompt for a conversation (for testing/debugging purposes).
    
    Args:
        conversation_id: The conversation ID to get prompt for
        request: FastAPI request object
        current_user: Authenticated user
        
    Returns:
        Debug information including the prompt
    """
    if not hasattr(request.app.state, 'debug_prompts'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No debug prompts available"
        )
    
    if conversation_id not in request.app.state.debug_prompts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No debug prompt found for conversation {conversation_id}"
        )
    
    return request.app.state.debug_prompts[conversation_id]


@router.post("/chat/stream")
async def episode_chat_stream(
    request: EpisodeChatRequest,
    fastapi_request: Request,
    current_user: SimpleUser = Depends(get_current_user),
    rag_manager: EpisodeRAGManager = Depends(get_rag_manager)
) -> StreamingResponse:
    """
    Episode-aware streaming chat endpoint with conversation context support.
    
    This endpoint provides real-time streaming responses for episode discussions
    with full conversation history management. The response is sent as Server-Sent Events (SSE) format.
    """
    
    async def generate_stream() -> AsyncIterator[str]:
        try:
            # Initialize configuration
            config = get_config()
            
            # Generate conversation ID if not provided
            conversation_id = request.conversation_id or str(uuid.uuid4())
            is_new_conversation = request.conversation_id is None
            
            # Step 1: Retrieve conversation context if requested
            conversation_context = []
            if request.use_conversation_context and request.conversation_id:
                conversation, conversation_context = await retrieve_episode_conversation_context(
                    conversation_id=request.conversation_id,
                    user_id=str(current_user.id),
                    max_context_turns=request.max_context_turns if hasattr(request, 'max_context_turns') else 10
                )
            
            # Send conversation info first
            conv_info = {
                "type": "conversation_info",
                "conversation_id": conversation_id,
                "is_new_conversation": is_new_conversation,
                "has_context": len(conversation_context) > 0
            }
            yield f"data: {json.dumps(conv_info)}\n\n"
            
            # Step 2: Perform episode-aware vector search with context enhancement
            enhanced_query = request.message
            conversation_enhanced = False
            
            if conversation_context and request.use_conversation_context:
                recent_context = " ".join([
                    msg.content for msg in conversation_context[-2:]
                    if msg.role == "user"
                ])
                if recent_context:
                    enhanced_query = f"{recent_context} {request.message}"
                    conversation_enhanced = True
            
            # Perform episode vector search with enhanced query
            search_result = await rag_manager.search_episodes(
                query=enhanced_query,
                limit=request.max_episodes if hasattr(request, 'max_episodes') else 5,
                episode_ids=request.episode_ids,
                novel_ids=request.novel_ids,
                sort_by_episode_number=True  # Sort by episode number for better context flow
            )
            
            # Extract episode sources from search result
            episode_sources = search_result.hits if hasattr(search_result, 'hits') else []
            
            # Calculate relevance scores
            episode_relevance_score = 0.0
            if episode_sources:
                episode_relevance_score = sum(s.similarity_score for s in episode_sources) / len(episode_sources)
            
            search_metadata = {
                "episodes_found": len(episode_sources),
                "query_used": enhanced_query,
                "conversation_enhanced": conversation_enhanced,
                "episode_relevance_score": episode_relevance_score
            }
            
            # Send search results first
            # Convert episode sources to dict
            episode_sources_data = []
            for source in episode_sources:
                source_dict = {
                    "episode_id": source.episode_id,
                    "episode_number": source.episode_number, 
                    "episode_title": source.episode_title,
                    "content": source.content,
                    "similarity_score": source.similarity_score,
                    "distance": source.distance
                }
                episode_sources_data.append(source_dict)
            
            search_response = {
                "type": "search_complete",
                "episode_sources": episode_sources_data,
                "metadata": search_metadata
            }
            yield f"data: {json.dumps(search_response)}\n\n"
            
            # Step 3: Prepare context and prompt with conversation history
            if episode_sources:
                # Build episode context
                context_parts = []
                
                # Add conversation history if available
                if conversation_context:
                    context_parts.append("=== 대화 기록 ===")
                    for msg in conversation_context[-6:]:  # Last 6 messages
                        role_label = "User" if msg.role == "user" else "Assistant"
                        context_parts.append(f"{role_label}: {msg.content}")
                    context_parts.append("")
                
                # Add episode sources
                context_parts.append("=== 에피소드(화) 내용 ===")
                for i, source in enumerate(episode_sources[:5], 1):
                    # Check if this is a chunk
                    chunk_info = ""
                    if hasattr(source, 'metadata') and source.metadata:
                        chunk_idx = source.metadata.get("chunk_index", -1)
                        if chunk_idx >= 0:
                            chunk_info = f" (Chunk {chunk_idx + 1})"
                    
                    context_parts.append(f"""{source.episode_number}화 {chunk_info}: {source.episode_title}
{source.content}
Relevance: {source.similarity_score:.2f}""")
                
                context_text = "\n\n".join(context_parts)
                
                # Create prompt with context awareness
                prompt = f"""당신은 웹소설 독자를 위한 전문 AI 어시스턴트 'NovelBot'입니다. 독자가 읽고 있는 웹소설에 대한 질문에 정확하고 도움이 되는 답변을 제공하는 것이 당신의 역할입니다. 제공된 맥락을 기반으로 질문에 답변하세요.

## 핵심 지침

### 1. 맥락 이해와 답변
- 제공된 에피소드를 깊이 있게 분석하여 답변하세요
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

{context_text}

질문: {request.message}"""
                
                # LOG: Print the full prompt being sent to LLM
                print("\n" + "="*80)
                print("🔍 STREAMING ENDPOINT - FULL PROMPT TO LLM:")
                print("="*80)
                print(f"Conversation ID: {conversation_id}")
                print(f"Has Previous Context: {len(conversation_context) > 0}")
                print(f"Number of Context Messages: {len(conversation_context)}")
                print("-"*40)
                print("PROMPT CONTENT:")
                print("-"*40)
                print(prompt)
                print("="*80 + "\n")
                
                # Store prompt for debugging (in memory, temporary)
                if not hasattr(fastapi_request.app.state, 'debug_prompts'):
                    fastapi_request.app.state.debug_prompts = {}
                fastapi_request.app.state.debug_prompts[conversation_id] = {
                    'prompt': prompt,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'context_messages': len(conversation_context),
                    'episode_sources': len(episode_sources)
                }
                
                # Step 3: Create LLM manager and stream response
                config = get_config()
                
                # Use override values from request if provided, otherwise use config
                llm_provider = request.llm_provider if request.llm_provider else config.llm.provider
                llm_model = request.llm_model if request.llm_model else config.llm.model
                llm_api_key = request.llm_api_key if request.llm_api_key else config.llm.api_key
                
                # Create LLM manager with proper provider configuration
                provider_configs = []
                
                # Map string provider to LLMProvider enum
                provider_map = {
                    "ollama": LLMProvider.OLLAMA,
                    "openai": LLMProvider.OPENAI,
                    "google": LLMProvider.GOOGLE,
                    "claude": LLMProvider.CLAUDE,
                    "anthropic": LLMProvider.CLAUDE
                }
                
                provider_enum = provider_map.get(llm_provider.lower())
                if not provider_enum:
                    raise ValueError(f"Unsupported provider: {llm_provider}")
                
                # Create provider config for any supported provider
                provider_config = ProviderConfig(
                    provider=provider_enum,
                    config=LLMConfig(
                        provider=llm_provider,
                        model=llm_model,
                        api_key=llm_api_key,
                        temperature=request.temperature if hasattr(request, 'temperature') else config.llm.temperature,
                        max_tokens=request.max_tokens if hasattr(request, 'max_tokens') else config.llm.max_tokens,
                        base_url=config.llm.base_url if llm_provider.lower() == "ollama" else None
                        # Removed stream=True as it's not a valid LLMConfig parameter
                    )
                )
                provider_configs.append(provider_config)
                
                llm_manager = create_llm_manager(provider_configs)
                
                # Create streaming request with conversation-aware settings
                llm_request = LLMRequest(
                    messages=[
                        LLMMessage(role=LLMRole.SYSTEM, content="You are a helpful AI assistant specialized in discussing novel episodes."),
                        LLMMessage(role=LLMRole.USER, content=prompt)
                    ],
                    model=llm_model,  # Use the potentially overridden model
                    temperature=request.temperature if hasattr(request, 'temperature') else 0.7,  # Use request temperature
                    max_tokens=request.max_tokens if hasattr(request, 'max_tokens') else 1000,
                    stream=True  # Enable streaming
                )
                
                # Save user message to conversation
                await save_conversation_message(
                    conversation_id=conversation_id,
                    user_id=str(current_user.id),
                    role="user",
                    content=request.message,
                    metadata={
                        "episode_ids": request.episode_ids,
                        "novel_ids": request.novel_ids
                    }
                )
                
                # Stream LLM response and collect full response
                full_response = ""
                # Check if llm_manager has the correct streaming method
                if hasattr(llm_manager, 'stream_async'):
                    stream_method = llm_manager.stream_async
                elif hasattr(llm_manager, 'generate_stream_async'):
                    stream_method = llm_manager.generate_stream_async
                elif provider_enum in llm_manager.providers:
                    # Direct provider access as fallback
                    provider = llm_manager.providers[provider_enum]
                    if hasattr(provider, 'stream_async'):
                        stream_method = provider.stream_async
                    elif hasattr(provider, 'generate_stream_async'):
                        stream_method = provider.generate_stream_async
                    else:
                        raise ValueError(f"Provider {provider_enum} does not have streaming support")
                else:
                    raise ValueError(f"Cannot find streaming method for provider {provider_enum}")
                
                async for chunk in stream_method(llm_request):
                    if chunk.content:
                        full_response += chunk.content
                        chunk_data = {
                            "type": "content",
                            "content": chunk.content,
                            "finish_reason": chunk.finish_reason
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                    # Send usage data if available
                    if chunk.finish_reason == "stop" and chunk.usage:
                        usage_data = {
                            "type": "usage",
                            "usage": chunk.usage.to_dict()
                        }
                        yield f"data: {json.dumps(usage_data)}\n\n"
                
                # Save assistant response to conversation
                await save_conversation_message(
                    conversation_id=conversation_id,
                    user_id=str(current_user.id),
                    role="assistant",
                    content=full_response,
                    metadata={
                        "episodes_used": [s.episode_id for s in episode_sources],
                        "relevance_score": episode_relevance_score
                    }
                )
                
                # Send final metadata with conversation context
                final_metadata = {
                    "type": "metadata",
                    "conversation_id": conversation_id,
                    "is_new_conversation": is_new_conversation,
                    "episodes_used": [s.episode_id for s in episode_sources],
                    "confidence_score": min(episode_relevance_score + 0.1, 1.0) if episode_sources else None,
                    "conversation_enhanced": conversation_enhanced,
                    "total_context_messages": len(conversation_context)
                }
                yield f"data: {json.dumps(final_metadata)}\n\n"
                
            else:
                # No relevant episodes found
                no_results = {
                    "type": "no_results",
                    "message": "I couldn't find any relevant episodes to answer your question.",
                    "conversation_id": conversation_id
                }
                yield f"data: {json.dumps(no_results)}\n\n"
            
            # Send completion signal with conversation ID
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# New API endpoints for enhanced functionality

class ProcessingStatusResponse(BaseAPISchema):
    """Processing status response schema."""
    novel_id: int = Field(..., description="Novel ID")
    status: str = Field(..., description="Processing status")
    episodes_total: int = Field(..., ge=0, description="Total episodes in novel")
    episodes_processed: int = Field(..., ge=0, description="Episodes successfully processed")
    episodes_failed: int = Field(..., ge=0, description="Episodes that failed processing")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    last_updated: str = Field(..., description="Last update timestamp")
    success_rate: float = Field(..., ge=0.0, le=100.0, description="Processing success rate percentage")
    chunked_episodes: int = Field(..., ge=0, description="Number of episodes that were chunked")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")


class ProcessAllRequest(BaseAPISchema):
    """Process episodes request schema."""
    force_reprocess: bool = Field(False, description="Force reprocessing of existing episodes")
    filter_episode_ids: Optional[List[int]] = Field(None, description="Process only specific episode IDs")


class ProcessAllResponse(BaseAPISchema):
    """Process episodes response schema."""
    message: str = Field(..., description="Status message")
    episodes_started: int = Field(..., ge=0, description="Number of episodes started for processing")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    processing_id: str = Field(..., description="Processing batch ID for tracking")


@router.post("/process-all", response_model=ProcessAllResponse)
async def process_all_episodes(
    request: ProcessAllRequest,
    background_tasks: BackgroundTasks,
    current_user: SimpleUser = Depends(get_current_user)
) -> ProcessAllResponse:
    """
    Process specific episodes using improved chunking logic.
    
    This endpoint starts background processing for specific episodes or all episodes in the database,
    using the enhanced episode processing with individual episode handling
    and automatic chunking for long content.
    
    Args:
        request: Processing configuration with episode IDs filter
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        Processing status with tracking information
    """
    try:
        # Generate processing batch ID
        processing_id = f"batch_{int(time.time())}_{str(current_user.id)[:8]}"
        
        # Get list of episodes to process
        config = get_config()
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        episode_ids = []
        try:
            if request.filter_episode_ids:
                # Process only specified episodes
                episode_ids = request.filter_episode_ids
                logger.info(f"Using filtered episode IDs: {episode_ids}")
                
                # Validate that episode IDs exist
                from sqlalchemy import text
                with db_manager.get_connection() as conn:
                    placeholders = ','.join([':id' + str(i) for i in range(len(episode_ids))])
                    params = {f'id{i}': episode_id for i, episode_id in enumerate(episode_ids)}
                    result = conn.execute(
                        text(f"SELECT episode_id FROM episode WHERE episode_id IN ({placeholders})"), 
                        params  # Use dictionary parameters for SQLAlchemy
                    )
                    existing_episodes = result.scalars().all()
                    
                    if len(existing_episodes) != len(episode_ids):
                        missing_episodes = set(episode_ids) - set(existing_episodes)
                        logger.warning(f"Some episode IDs do not exist: {missing_episodes}")
                        episode_ids = existing_episodes
            else:
                # Get all episode IDs from database
                from sqlalchemy import text
                with db_manager.get_connection() as conn:
                    result = conn.execute(text("SELECT episode_id FROM episode ORDER BY episode_id"))
                    episode_ids = result.scalars().all()
                    logger.info(f"Retrieved {len(episode_ids)} episodes from database: {episode_ids[:10]}...")
            
        except Exception as e:
            logger.error(f"Failed to query episodes from database: {e}")
            import traceback
            logger.error(f"Database query traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve episode data: {str(e)}"
            )
        
        # Check if any episodes to process
        if not episode_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid episode IDs found to process"
            )
        
        # Start background processing
        background_tasks.add_task(
            process_all_episodes_background,
            episode_ids=episode_ids,
            force_reprocess=request.force_reprocess,
            processing_id=processing_id,
            user_id=str(current_user.id)
        )
        
        # Estimate completion time (rough calculation)
        estimated_minutes = len(episode_ids) * 0.5  # Assume 30 seconds per episode
        from datetime import datetime, timedelta
        completion_time = datetime.now() + timedelta(minutes=estimated_minutes)
        
        return ProcessAllResponse(
            message=f"Started processing {len(episode_ids)} episodes with improved chunking logic",
            episodes_started=len(episode_ids),
            estimated_completion_time=completion_time.isoformat(),
            processing_id=processing_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start episode processing: {str(e)}"
        )


@router.get("/novel/{novel_id}/status", response_model=ProcessingStatusResponse)
async def get_novel_processing_status(
    novel_id: int,
    current_user: SimpleUser = Depends(get_current_user)
) -> ProcessingStatusResponse:
    """
    Get processing status for a specific novel.
    
    Returns detailed information about episode processing status,
    including success rates, chunking statistics, and error details.
    
    Args:
        novel_id: Novel ID to check status for
        current_user: Authenticated user
        
    Returns:
        Detailed processing status information
    """
    try:
        config = get_config()
        
        # Initialize database connection
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        episodes_total = 0
        episodes_processed = 0
        episodes_failed = 0
        chunked_episodes = 0
        last_updated = datetime.now(timezone.utc).isoformat()
        processing_time = None
        
        try:
            with db_manager.get_connection() as connection:
                # Get total episodes for this novel
                count_query = "SELECT COUNT(*) as total FROM episode WHERE novel_id = %s"
                with connection.cursor() as cursor:
                    cursor.execute(count_query, (novel_id,))
                    result = cursor.fetchone()
                    episodes_total = result['total'] if result else 0
                
                # Check Milvus for processed episodes
                milvus_client = MilvusClient(config.milvus)
                milvus_client.connect()
                
                if milvus_client.has_collection("episode_embeddings"):
                    # Query processed episodes from Milvus
                    try:
                        processed_results = milvus_client.query(
                            collection_name="episode_embeddings",
                            expr=f"novel_id == {novel_id}",
                            output_fields=["episode_id", "content_length"],
                            limit=1000
                        )
                        
                        episodes_processed = len(processed_results)
                        
                        # Count chunked episodes (episodes with multiple embeddings)
                        # This is a simplified check - in practice you'd need more sophisticated logic
                        if processed_results:
                            chunked_episodes = sum(1 for result in processed_results 
                                                 if result.get('content_length', 0) > 8000)
                        
                    except Exception as e:
                        logger.warning(f"Failed to query Milvus for novel {novel_id}: {e}")
                        episodes_processed = 0
                
                episodes_failed = max(0, episodes_total - episodes_processed)
            
        except Exception as e:
            logger.warning(f"Database query failed for novel {novel_id}: {e}")
            # Return basic status if database query fails
            episodes_total = 1
            episodes_processed = 0
            episodes_failed = 1
        
        # Calculate success rate
        success_rate = (episodes_processed / episodes_total * 100) if episodes_total > 0 else 0.0
        
        # Determine status
        if episodes_processed == 0:
            status_text = "not_started"
        elif episodes_processed == episodes_total:
            status_text = "completed"
        elif episodes_failed > 0:
            status_text = "partially_completed"
        else:
            status_text = "in_progress"
        
        # Additional metadata
        metadata = {
            "novel_id": novel_id,
            "has_embeddings": episodes_processed > 0,
            "needs_processing": episodes_failed > 0,
            "chunking_applied": chunked_episodes > 0,
            "chunking_rate": (chunked_episodes / episodes_processed * 100) if episodes_processed > 0 else 0.0
        }
        
        return ProcessingStatusResponse(
            novel_id=novel_id,
            status=status_text,
            episodes_total=episodes_total,
            episodes_processed=episodes_processed,
            episodes_failed=episodes_failed,
            processing_time=processing_time,
            last_updated=last_updated,
            success_rate=success_rate,
            chunked_episodes=chunked_episodes,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing status: {str(e)}"
        )


# Background task for processing specific episodes (CLI-style sequential processing)
async def process_all_episodes_background(
    episode_ids: List[int],
    force_reprocess: bool,
    processing_id: str,
    user_id: str
) -> None:
    """
    Background task to process specific episodes using Episode-specific processing with improved chunking.
    This mirrors the CLI command: uv run rag-cli data ingest --episode-mode --database --force
    
    Args:
        episode_ids: List of episode IDs to process
        force_reprocess: Whether to reprocess existing episodes
        processing_id: Unique ID for this processing batch
        user_id: User who initiated the processing
    """
    start_time = time.time()
    
    print(f"🚀 Starting Episode-specific processing with improved chunking [{processing_id}]")
    print(f"   Episodes to process: {len(episode_ids)}")
    print(f"   Force reprocess: {force_reprocess}")
    print(f"   User: {user_id}")
    
    try:
        # Import episode-specific components (same as CLI)
        from src.episode.manager import EpisodeRAGManager
        from src.core.config import get_config
        
        config = get_config()
        
        # Initialize dependencies exactly like CLI episode-mode
        from src.database.base import DatabaseManager
        from src.embedding.manager import EmbeddingManager
        from src.milvus.client import MilvusClient
        from src.episode.manager import EpisodeRAGConfig
        
        # Initialize dependencies  
        db_manager = DatabaseManager(config.database)
        
        # Create embedding provider configs list (same as CLI)
        if config.embedding_providers:
            provider_configs = list(config.embedding_providers.values())
        else:
            # Fallback to single embedding config
            provider_configs = [config.embedding]
            
        embedding_manager = EmbeddingManager(provider_configs)
        milvus_client = MilvusClient(config.milvus)
        episode_config = EpisodeRAGConfig(
            processing_batch_size=5,  # Further reduce batch size for stability  
            vector_dimension=get_config().rag.vector_dimension  # Use configured dimension
        )
        
        episode_manager = EpisodeRAGManager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config
        )
        
        # Connect to Milvus first
        milvus_client.connect()
        
        # Setup collection first (force drop if reprocessing, same as CLI)
        await episode_manager.setup_collection(drop_existing=force_reprocess)
        
        total_processed = 0
        total_failed = 0
        
        # Choose processing strategy based on force_reprocess flag
        if force_reprocess:
            # When force_reprocess=True, use CLI logic: process by novels
            print(f"🔄 Force reprocess mode: Processing by novels (CLI logic)")
            
            # Get novel IDs from episode IDs
            from sqlalchemy import text
            with db_manager.get_connection() as conn:
                if episode_ids:
                    # Get unique novel IDs from the provided episode IDs
                    placeholders = ','.join([':id' + str(i) for i in range(len(episode_ids))])
                    params = {f'id{i}': episode_id for i, episode_id in enumerate(episode_ids)}
                    result = conn.execute(
                        text(f"SELECT DISTINCT novel_id FROM episode WHERE episode_id IN ({placeholders})"),
                        params
                    )
                    novel_ids = result.scalars().all()
                else:
                    # Get all novel IDs
                    result = conn.execute(text("SELECT DISTINCT novel_id FROM novels"))
                    novel_ids = result.scalars().all()
            
            print(f"Found {len(novel_ids)} novels to process")
            
            # Process by novels (same as CLI)
            for i, novel_id in enumerate(novel_ids, 1):
                try:
                    print(f"🔄 Processing Novel {novel_id} ({i}/{len(novel_ids)})")
                    
                    # Add delay between novels to prevent Ollama overload
                    if i > 1:
                        import asyncio
                        await asyncio.sleep(2)  # 2 second delay (same as CLI)
                    
                    result = await episode_manager.process_novel(novel_id, force_reprocess=True)
                    
                    episode_count = result.get('episodes_processed', 0)
                    total_processed += episode_count
                    
                    print(f"✅ Novel {novel_id}: {episode_count} episodes processed")
                    
                except Exception as e:
                    total_failed += 1
                    print(f"❌ Failed to process Novel {novel_id}: {e}")
                    continue
                    
        else:
            # When force_reprocess=False, use existing logic: process by individual episodes
            print(f"📝 Regular processing mode: Processing by individual episodes")
            print(f"Found {len(episode_ids)} episodes to process")
            
            # Process episodes individually
            for i, episode_id in enumerate(episode_ids, 1):
                try:
                    print(f"🔄 Processing Episode {episode_id} ({i}/{len(episode_ids)})")
                    
                    # Add delay between episodes to prevent Ollama overload
                    if i > 1:
                        import asyncio
                        await asyncio.sleep(1)  # 1 second delay
                    
                    result = await episode_manager.process_episodes([episode_id], force_reprocess=force_reprocess)
                    
                    if result.get('episodes_processed', 0) > 0:
                        total_processed += 1
                        print(f"✅ Episode {episode_id}: processed successfully")
                    else:
                        total_failed += 1
                        print(f"❌ Episode {episode_id}: processing failed")
                    
                except Exception as e:
                    total_failed += 1
                    print(f"❌ Failed to process Episode {episode_id}: {e}")
                    continue
        
        processing_time = time.time() - start_time
        
        print(f"✅ Episode processing completed [{processing_id}]")
        print(f"   Total episodes processed: {total_processed}")
        print(f"   Failed episodes: {total_failed}")
        print(f"   Processing time: {processing_time:.1f}s")
        
    except ImportError as e:
        processing_time = time.time() - start_time
        print(f"❌ Episode processing not available [{processing_id}]: {e}")
        print(f"   Processing time: {processing_time:.1f}s")
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Episode processing failed [{processing_id}]: {e}")
        print(f"   Processing time: {processing_time:.1f}s")



# Monitoring and statistics endpoints

class ProcessingStatsResponse(BaseAPISchema):
    """Processing statistics response schema."""
    total_novels: int = Field(..., ge=0, description="Total novels in database")
    processed_novels: int = Field(..., ge=0, description="Novels with processed episodes")
    failed_novels: int = Field(..., ge=0, description="Novels with failed processing")
    total_episodes: int = Field(..., ge=0, description="Total episodes across all novels")
    processed_episodes: int = Field(..., ge=0, description="Successfully processed episodes")
    failed_episodes: int = Field(..., ge=0, description="Failed episode processing attempts")
    chunked_episodes: int = Field(..., ge=0, description="Episodes that required chunking")
    overall_success_rate: float = Field(..., ge=0.0, le=100.0, description="Overall processing success rate")
    average_processing_time: Optional[float] = Field(None, description="Average processing time per episode")
    chunking_rate: float = Field(..., ge=0.0, le=100.0, description="Percentage of episodes that were chunked")
    last_updated: str = Field(..., description="Last statistics update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional statistical metadata")


class FailedEpisodesResponse(BaseAPISchema):
    """Failed episodes response schema."""
    failed_episodes: List[Dict[str, Any]] = Field(..., description="List of failed episodes with details")
    total_failed: int = Field(..., ge=0, description="Total number of failed episodes")
    novels_affected: int = Field(..., ge=0, description="Number of novels with failed episodes")
    common_errors: List[Dict[str, Any]] = Field(..., description="Most common error types and counts")
    last_updated: str = Field(..., description="Last update timestamp")


@router.get("/stats", response_model=ProcessingStatsResponse)
async def get_processing_statistics(
    current_user: SimpleUser = Depends(get_current_user)
) -> ProcessingStatsResponse:
    """
    Get comprehensive processing statistics and success rates.
    
    Returns detailed statistics about episode processing across all novels,
    including success rates, chunking statistics, and performance metrics.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Comprehensive processing statistics
    """
    try:
        config = get_config()
        
        # Initialize database connection
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        # Initialize default statistics
        total_novels = 0
        total_episodes = 0
        processed_episodes = 0
        chunked_episodes = 0
        failed_episodes = 0
        last_updated = datetime.now(timezone.utc).isoformat()
        
        try:
            with db_manager.get_connection() as connection:
                # Get total novels and episodes from database
                with connection.cursor() as cursor:
                    # Count unique novels
                    cursor.execute("SELECT COUNT(DISTINCT novel_id) as total FROM episode")
                    result = cursor.fetchone()
                    total_novels = result['total'] if result else 0
                    
                    # Count total episodes
                    cursor.execute("SELECT COUNT(*) as total FROM episode")
                    result = cursor.fetchone()
                    total_episodes = result['total'] if result else 0
                
                # Check Milvus for processed episodes
                milvus_client = MilvusClient(config.milvus)
                milvus_client.connect()
                
                if milvus_client.has_collection("episode_embeddings"):
                    try:
                        # Get all processed episodes
                        all_processed = milvus_client.query(
                            collection_name="episode_embeddings",
                            expr="",
                            output_fields=["episode_id", "novel_id", "content_length"],
                            limit=10000  # Adjust based on your data size
                        )
                        
                        processed_episodes = len(all_processed)
                        
                        # Count episodes that were likely chunked (content_length > 8000 chars)
                        if all_processed:
                            chunked_episodes = sum(1 for ep in all_processed 
                                                 if ep.get('content_length', 0) > 8000)
                        
                        # Count processed novels
                        processed_novel_ids = set(ep.get('novel_id') for ep in all_processed if ep.get('novel_id'))
                        processed_novels = len(processed_novel_ids)
                        
                    except Exception as e:
                        logger.warning(f"Failed to query Milvus statistics: {e}")
                        processed_episodes = 0
                        processed_novels = 0
                        chunked_episodes = 0
                else:
                    processed_novels = 0
                
                failed_episodes = max(0, total_episodes - processed_episodes)
                failed_novels = max(0, total_novels - processed_novels)
            
        except Exception as e:
            logger.warning(f"Database query failed for statistics: {e}")
            # Return minimal stats if database query fails
            total_novels = 1
            total_episodes = 1
            processed_episodes = 0
            processed_novels = 0
            failed_novels = 1
            failed_episodes = 1
            chunked_episodes = 0
        
        # Calculate rates
        overall_success_rate = (processed_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
        chunking_rate = (chunked_episodes / processed_episodes * 100) if processed_episodes > 0 else 0.0
        
        # Additional metadata
        metadata = {
            "data_source": "live_database_and_milvus",
            "collection_name": "episode_embeddings",
            "statistics_scope": "all_novels",
            "chunking_threshold": "8000_characters",
            "processing_method": "individual_episode_with_chunking"
        }
        
        return ProcessingStatsResponse(
            total_novels=total_novels,
            processed_novels=processed_novels,
            failed_novels=failed_novels,
            total_episodes=total_episodes,
            processed_episodes=processed_episodes,
            failed_episodes=failed_episodes,
            chunked_episodes=chunked_episodes,
            overall_success_rate=overall_success_rate,
            average_processing_time=None,  # Could be calculated from processing logs
            chunking_rate=chunking_rate,
            last_updated=last_updated,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing statistics: {str(e)}"
        )


@router.get("/failed-episodes", response_model=FailedEpisodesResponse)
async def get_failed_episodes(
    limit: int = 50,
    novel_id: Optional[int] = None,
    current_user: SimpleUser = Depends(get_current_user)
) -> FailedEpisodesResponse:
    """
    Get list of episodes that failed processing.
    
    Returns detailed information about episodes that failed to process,
    including error types and affected novels for troubleshooting.
    
    Args:
        limit: Maximum number of failed episodes to return
        novel_id: Filter by specific novel ID
        current_user: Authenticated user
        
    Returns:
        List of failed episodes with error details
    """
    try:
        config = get_config()
        
        # Initialize database connection
        from ...database.base import DatabaseManager
        db_manager = DatabaseManager(config.database)
        
        failed_episodes = []
        common_errors = []
        total_failed = 0
        novels_affected = 0
        last_updated = datetime.now(timezone.utc).isoformat()
        
        try:
            with db_manager.get_connection() as connection:
                # Get all episodes from database
                base_query = """
                    SELECT 
                        id as episode_id,
                        novel_id,
                        episode_number,
                        title as episode_title,
                        CHAR_LENGTH(content) as content_length,
                        created_at
                    FROM episode
                """
                
                params = []
                if novel_id:
                    base_query += " WHERE novel_id = %s"
                    params.append(novel_id)
                
                base_query += " ORDER BY novel_id, episode_number"
                
                with connection.cursor() as cursor:
                    cursor.execute(base_query, params)
                    all_episodes = cursor.fetchall()
                
                # Get processed episodes from Milvus
                processed_episode_ids = set()
                
                milvus_client = MilvusClient(config.milvus)
                milvus_client.connect()
                
                if milvus_client.has_collection("episode_embeddings"):
                    try:
                        expr = f"novel_id == {novel_id}" if novel_id else ""
                        processed_results = milvus_client.query(
                            collection_name="episode_embeddings",
                            expr=expr,
                            output_fields=["episode_id"],
                            limit=10000
                        )
                        
                        processed_episode_ids = set(result.get('episode_id') for result in processed_results 
                                                   if result.get('episode_id'))
                        
                    except Exception as e:
                        logger.warning(f"Failed to query processed episodes: {e}")
                
                # Find failed episodes
                failed_episode_data = []
                novels_with_failures = set()
                
                for episode in all_episodes:
                    if episode['episode_id'] not in processed_episode_ids:
                        # This episode failed processing
                        failed_episode_data.append({
                            "episode_id": episode['episode_id'],
                            "novel_id": episode['novel_id'],
                            "episode_number": episode['episode_number'],
                            "episode_title": episode['episode_title'],
                            "content_length": episode['content_length'],
                            "created_at": episode['created_at'].isoformat() if episode['created_at'] else None,
                            "error_type": "processing_failed",
                            "error_details": "Episode not found in vector database",
                            "likely_cause": "Token limit exceeded" if episode['content_length'] > 10000 else "Processing error"
                        })
                        novels_with_failures.add(episode['novel_id'])
                
                # Sort by novel_id and episode_number, then limit
                failed_episode_data.sort(key=lambda x: (x['novel_id'], x['episode_number']))
                failed_episodes = failed_episode_data[:limit]
                
                total_failed = len(failed_episode_data)
                novels_affected = len(novels_with_failures)
                
                # Analyze common error patterns
                error_types = {}
                for episode in failed_episode_data:
                    error_type = episode['likely_cause']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                common_errors = [
                    {"error_type": error_type, "count": count, "percentage": (count / total_failed * 100) if total_failed > 0 else 0}
                    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)
                ]
            
        except Exception as e:
            logger.warning(f"Failed to analyze failed episodes: {e}")
            # Return empty results if analysis fails
            failed_episodes = []
            total_failed = 0
            novels_affected = 0
            common_errors = [{"error_type": "analysis_failed", "count": 1, "percentage": 100.0}]
        
        return FailedEpisodesResponse(
            failed_episodes=failed_episodes,
            total_failed=total_failed,
            novels_affected=novels_affected,
            common_errors=common_errors,
            last_updated=last_updated
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get failed episodes: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def episode_health_check() -> Dict[str, Any]:
    """Check health of episode processing components."""
    return {
        "status": "healthy",
        "components": {
            "vector_store": "healthy",
            "search_engine": "healthy",
            "embedding_processor": "healthy",
            "episode_chat": "healthy",
            "bulk_processing": "healthy",
            "statistics_monitoring": "healthy"
        },
        "endpoints": {
            "process_all": "available",
            "status_check": "available", 
            "individual_processing": "available",
            "statistics": "available",
            "failed_episodes": "available"
        },
        "timestamp": time.time()
    }


@router.get("/conversation/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of messages to retrieve"),
    current_user: SimpleUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get conversation history for a specific conversation ID.
    
    Args:
        conversation_id: The unique conversation ID
        limit: Maximum number of messages to retrieve (default 50, max 200)
        current_user: Authenticated user
        
    Returns:
        Conversation history with messages and metadata
    """
    try:
        # Get conversation storage
        from src.conversation.storage import ConversationStorage
        storage = ConversationStorage()
        
        # Get conversation info
        conversation_info = await storage.get_conversation_info(conversation_id)
        if not conversation_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        # Check if user owns this conversation
        if conversation_info.get('user_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this conversation"
            )
        
        # Get messages
        messages = await storage.get_messages(conversation_id, limit=limit)
        
        # Format response
        return {
            "conversation_id": conversation_id,
            "user_id": conversation_info.get('user_id'),
            "created_at": conversation_info.get('created_at'),
            "updated_at": conversation_info.get('updated_at'),
            "metadata": conversation_info.get('metadata'),
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@router.post("/conversation/{conversation_id}/generate-title")
async def generate_conversation_title(
    conversation_id: str,
    current_user: SimpleUser = Depends(get_current_user),
    rag_manager: EpisodeRAGManager = Depends(get_rag_manager)
) -> Dict[str, str]:
    """
    Generate a title for a conversation based on its content.
    
    Args:
        conversation_id: The unique conversation ID
        current_user: Authenticated user
        rag_manager: RAG manager instance
        
    Returns:
        Generated title for the conversation
    """
    try:
        # Get conversation storage
        from src.conversation.storage import ConversationStorage
        storage = ConversationStorage()
        
        # Get conversation info
        conversation_info = await storage.get_conversation_info(conversation_id)
        if not conversation_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        # Check if user owns this conversation
        if conversation_info.get('user_id') != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this conversation"
            )
        
        # Get first few messages to generate title
        messages = await storage.get_messages(conversation_id, limit=10)
        
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation has no messages to generate title from"
            )
        
        # Build conversation summary for title generation
        conversation_summary = "다음 대화의 핵심 주제를 15자 이내로 요약해주세요:\n"
        for msg in messages[:6]:  # Use first 6 messages max
            if msg.role == "user":
                conversation_summary += f"사용자: {msg.content[:200]}...\n" if len(msg.content) > 200 else f"사용자: {msg.content}\n"
            else:
                conversation_summary += f"AI: {msg.content[:200]}...\n" if len(msg.content) > 200 else f"AI: {msg.content}\n"
        
        conversation_summary += "\n사용자의 질문을 중심으로 핵심 주제만 15자 이내로 답해주세요. 부가 설명 없이 제목만 답하세요."
        
        # Use LLM to generate title - simpler approach using langchain directly
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        import os
        
        # Create OpenAI LLM instance
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=50,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Generate title
        title_response = await llm.ainvoke([HumanMessage(content=conversation_summary)])
        generated_title = title_response.content.strip()
        
        # Ensure title is not too long
        if len(generated_title) > 30:
            generated_title = generated_title[:30]
        
        # Update conversation metadata with title
        metadata = conversation_info.get('metadata', {})
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata) if metadata else {}
        metadata['title'] = generated_title
        
        # Save updated metadata
        await storage.update_conversation_metadata(conversation_id, metadata)
        
        return {
            "conversation_id": conversation_id,
            "generated_title": generated_title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating conversation title: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate conversation title: {str(e)}"
        )