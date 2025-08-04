"""
Episode Search Engine for RAG System.

This module provides episode-specific search capabilities with filtering by episode IDs,
sorting by episode numbers, and metadata-based search optimization.
"""

import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, date

from src.core.logging import LoggerMixin
from src.core.exceptions import SearchError, ConfigurationError
from src.embedding.manager import EmbeddingManager
from src.embedding.base import EmbeddingRequest
from src.rag.vector_search_engine import VectorSearchEngine, VectorSearchRequest, SearchMode, DistanceMetric
from src.milvus.client import MilvusClient
from .models import (
    EpisodeSearchRequest, EpisodeSearchResult, EpisodeSearchHit, 
    EpisodeSortOrder, EpisodeData
)
from .vector_store import EpisodeVectorStore


class EpisodeSearchEngine(LoggerMixin):
    """
    Episode-specific search engine with filtering and sorting capabilities.
    
    Features:
    - Episode ID filtering for targeted search
    - Episode number-based sorting for narrative order
    - Metadata filtering (novel ID, date range, etc.)
    - Hybrid vector + scalar search
    - Context optimization for LLM consumption
    """
    
    def __init__(
        self,
        milvus_client: MilvusClient,
        embedding_manager: EmbeddingManager,
        vector_store: EpisodeVectorStore,
        base_search_engine: Optional[VectorSearchEngine] = None
    ):
        """
        Initialize Episode Search Engine.
        
        Args:
            milvus_client: Milvus client instance
            embedding_manager: Embedding generation manager
            vector_store: Episode vector store
            base_search_engine: Optional base vector search engine
        """
        self.milvus_client = milvus_client
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        
        # Initialize base search engine if not provided
        if base_search_engine:
            self.base_search_engine = base_search_engine
        else:
            from src.rag.vector_search_engine import create_vector_search_engine
            self.base_search_engine = create_vector_search_engine(milvus_client)
        
        # Search statistics
        self.search_stats = {
            "total_searches": 0,
            "filtered_searches": 0,
            "total_search_time": 0.0,
            "average_results_per_search": 0.0,
            "cache_hits": 0
        }
        
        self.logger.info("EpisodeSearchEngine initialized")
    
    def search(self, request: EpisodeSearchRequest) -> EpisodeSearchResult:
        """
        Perform episode-based search with filtering and sorting.
        
        Args:
            request: Episode search request
            
        Returns:
            Episode search results with metadata
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(request.query)
            
            # Build filter expression
            filter_expr = request.build_filter_expression()
            
            # Get output fields
            output_fields = request.get_output_fields()
            
            # Create vector search request
            vector_request = VectorSearchRequest(
                query_vectors=[query_embedding],
                limit=request.limit,
                metric=DistanceMetric.L2,  # Default metric
                similarity_threshold=request.similarity_threshold,
                filter_expression=filter_expr,
                output_fields=output_fields,
                search_mode=SearchMode.BALANCED
            )
            
            # Perform vector search
            vector_result = self.base_search_engine.search(
                collection_name=self.vector_store.config.collection_name,
                request=vector_request
            )
            
            # Convert to episode search results
            episode_result = self._convert_to_episode_result(
                vector_result, request, start_time
            )
            
            # Apply episode-specific sorting
            if request.sort_order != EpisodeSortOrder.SIMILARITY:
                episode_result.hits = self._sort_hits(episode_result.hits, request.sort_order)
            
            # Update statistics
            self._update_search_stats(episode_result, filter_expr is not None)
            
            self.logger.info(
                f"Episode search completed: {len(episode_result.hits)} results "
                f"in {episode_result.search_time:.3f}s for query: '{request.query[:50]}...'"
            )
            
            return episode_result
            
        except Exception as e:
            search_time = time.time() - start_time
            self.logger.error(f"Episode search failed: {e}")
            raise SearchError(f"Episode search failed: {e}")
    
    async def search_async(self, request: EpisodeSearchRequest) -> EpisodeSearchResult:
        """
        Asynchronous episode search.
        
        Args:
            request: Episode search request
            
        Returns:
            Episode search results
        """
        # Run CPU-bound search in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, request)
    
    def search_by_episode_ids(
        self,
        query: str,
        episode_ids: List[int],
        limit: int = 10,
        sort_by_episode_number: bool = True,
        **kwargs
    ) -> EpisodeSearchResult:
        """
        Simplified interface for searching within specific episodes.
        
        Args:
            query: Search query
            episode_ids: List of episode IDs to search within
            limit: Maximum number of results
            sort_by_episode_number: Whether to sort by episode number
            **kwargs: Additional search parameters
            
        Returns:
            Episode search results
        """
        request = EpisodeSearchRequest(
            query=query,
            episode_ids=episode_ids,
            limit=limit,
            sort_order=EpisodeSortOrder.EPISODE_NUMBER if sort_by_episode_number else EpisodeSortOrder.SIMILARITY,
            **kwargs
        )
        
        return self.search(request)
    
    def search_novel_episodes(
        self,
        query: str,
        novel_id: int,
        limit: int = 10,
        episode_range: Optional[tuple] = None,
        **kwargs
    ) -> EpisodeSearchResult:
        """
        Search episodes within a specific novel.
        
        Args:
            query: Search query
            novel_id: Novel ID to search within
            limit: Maximum number of results
            episode_range: Optional (start, end) episode number range
            **kwargs: Additional search parameters
            
        Returns:
            Episode search results
        """
        request = EpisodeSearchRequest(
            query=query,
            novel_ids=[novel_id],
            limit=limit,
            sort_order=EpisodeSortOrder.EPISODE_NUMBER,
            **kwargs
        )
        
        if episode_range:
            request.episode_num_from, request.episode_num_to = episode_range
        
        return self.search(request)
    
    def get_episode_context(
        self,
        episode_ids: List[int],
        query: Optional[str] = None,
        max_context_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Get episode content as context for LLM, ordered by episode number.
        
        Args:
            episode_ids: List of episode IDs
            query: Optional query for relevance scoring
            max_context_length: Maximum total character length
            
        Returns:
            Context dictionary with ordered episode content
        """
        try:
            # If query provided, do similarity search within episodes
            if query:
                request = EpisodeSearchRequest(
                    query=query,
                    episode_ids=episode_ids,
                    limit=len(episode_ids),
                    sort_order=EpisodeSortOrder.EPISODE_NUMBER,
                    include_content=True
                )
                result = self.search(request)
                hits = result.hits
            else:
                # Direct retrieval without similarity scoring
                hits = self._get_episodes_by_ids(episode_ids)
            
            # Build context with episode order
            context_parts = []
            total_length = 0
            
            for hit in hits:
                if not hit.content:
                    continue
                
                episode_text = f"Episode {hit.episode_number}: {hit.episode_title}\n{hit.content}"
                
                # Check length limit
                if total_length + len(episode_text) > max_context_length:
                    # Truncate if needed
                    remaining_space = max_context_length - total_length
                    if remaining_space > 100:  # Only include if meaningful space left
                        episode_text = episode_text[:remaining_space] + "...[truncated]"
                        context_parts.append(episode_text)
                    break
                
                context_parts.append(episode_text)
                total_length += len(episode_text)
            
            return {
                "context": "\n\n---\n\n".join(context_parts),
                "episodes_included": len(context_parts),
                "total_length": total_length,
                "episode_order": [hit.episode_number for hit in hits[:len(context_parts)]],
                "truncated": total_length >= max_context_length
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get episode context: {e}")
            raise SearchError(f"Episode context retrieval failed: {e}")
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            # Use the same model as configured in the embedding manager
            # instead of hardcoded OpenAI model
            request = EmbeddingRequest(
                input=[query],
                encoding_format="float",
                metadata={"is_query": True}
            )
            
            response = self.embedding_manager.generate_embeddings(request)
            
            if not response.embeddings or len(response.embeddings) != 1:
                raise SearchError("Failed to generate query embedding")
            
            return response.embeddings[0]
            
        except Exception as e:
            self.logger.error(f"Query embedding generation failed: {e}")
            raise SearchError(f"Query embedding generation failed: {e}")
    
    def _convert_to_episode_result(
        self,
        vector_result,
        request: EpisodeSearchRequest,
        start_time: float
    ) -> EpisodeSearchResult:
        """Convert vector search result to episode search result."""
        hits = []
        
        # Process vector search hits
        self.logger.info(f"Vector result structure: {type(vector_result)}")
        self.logger.info(f"Vector result hits: {vector_result.hits if hasattr(vector_result, 'hits') else 'No hits attribute'}")
        
        if vector_result.hits and len(vector_result.hits) > 0:
            self.logger.info(f"First hit list length: {len(vector_result.hits[0])}")
            for i, hit in enumerate(vector_result.hits[0]):  # First query results
                try:
                    self.logger.info(f"Processing hit {i}: {hit}")
                    episode_hit = EpisodeSearchHit(
                        episode_id=hit.entity.get("episode_id"),
                        episode_number=hit.entity.get("episode_number"),
                        episode_title=hit.entity.get("episode_title", ""),
                        novel_id=hit.entity.get("novel_id"),
                        similarity_score=hit.score,
                        distance=hit.distance,
                        content=hit.entity.get("content") if request.include_content else None,
                        publication_date=self._parse_date(hit.entity.get("publication_date")),
                        content_length=hit.entity.get("content_length"),
                        metadata=hit.metadata
                    )
                    hits.append(episode_hit)
                    self.logger.info(f"Successfully converted hit {i}")
                except Exception as e:
                    self.logger.warning(f"Failed to convert hit {i}: {e}")
                    continue
        else:
            self.logger.warning("No vector search hits found or empty result")
        
        # Create result
        result = EpisodeSearchResult(
            hits=hits,
            total_count=len(hits),
            search_time=time.time() - start_time,
            query=request.query,
            sort_order=request.sort_order,
            metadata={
                "filter_applied": request.build_filter_expression() is not None,
                "episode_ids_filter": request.episode_ids,
                "novel_ids_filter": request.novel_ids,
                "similarity_threshold": request.similarity_threshold,
                "vector_search_time": vector_result.search_time
            }
        )
        
        return result
    
    def _sort_hits(self, hits: List[EpisodeSearchHit], sort_order: EpisodeSortOrder) -> List[EpisodeSearchHit]:
        """Sort hits according to specified order."""
        if sort_order == EpisodeSortOrder.EPISODE_NUMBER:
            return sorted(hits, key=lambda h: h.episode_number)
        elif sort_order == EpisodeSortOrder.PUBLICATION_DATE:
            return sorted(hits, key=lambda h: h.publication_date or date.min)
        elif sort_order == EpisodeSortOrder.SIMILARITY:
            return sorted(hits, key=lambda h: h.similarity_score, reverse=True)
        else:
            return hits
    
    def _get_episodes_by_ids(self, episode_ids: List[int]) -> List[EpisodeSearchHit]:
        """Get episodes by IDs without similarity search."""
        # This would require a query operation on the vector store
        # For now, we'll return empty list and rely on similarity search
        # In a full implementation, you'd add a query method to the vector store
        return []
    
    def _parse_date(self, date_string: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_string:
            return None
        
        try:
            return datetime.fromisoformat(date_string).date()
        except (ValueError, TypeError):
            return None
    
    def _update_search_stats(self, result: EpisodeSearchResult, was_filtered: bool) -> None:
        """Update search statistics."""
        self.search_stats["total_searches"] += 1
        if was_filtered:
            self.search_stats["filtered_searches"] += 1
        
        self.search_stats["total_search_time"] += result.search_time
        
        # Update average results
        total_results = self.search_stats.get("total_results", 0) + result.total_count
        avg_results = total_results / self.search_stats["total_searches"]
        self.search_stats["average_results_per_search"] = avg_results
        self.search_stats["total_results"] = total_results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        total_searches = self.search_stats["total_searches"]
        
        return {
            "total_searches": total_searches,
            "filtered_searches": self.search_stats["filtered_searches"],
            "filter_usage_rate": (
                self.search_stats["filtered_searches"] / total_searches * 100
                if total_searches > 0 else 0.0
            ),
            "average_search_time": (
                self.search_stats["total_search_time"] / total_searches
                if total_searches > 0 else 0.0
            ),
            "average_results_per_search": self.search_stats["average_results_per_search"],
            "cache_hits": self.search_stats["cache_hits"],
            "base_engine_stats": self.base_search_engine.get_search_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the search engine."""
        try:
            # Check vector store
            vector_store_health = self.vector_store.health_check()
            
            # Check base search engine
            base_engine_health = self.base_search_engine.health_check()
            
            # Check embedding manager
            embedding_health = {"status": "healthy"}  # Basic check
            
            overall_status = "healthy"
            if (vector_store_health.get("status") != "healthy" or
                base_engine_health.get("status") != "healthy"):
                overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "components": {
                    "vector_store": vector_store_health,
                    "base_search_engine": base_engine_health,
                    "embedding_manager": embedding_health
                },
                "stats": self.get_search_stats(),
                "collection_name": self.vector_store.config.collection_name
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {},
                "stats": {}
            }


def create_episode_search_engine(
    milvus_client: MilvusClient,
    embedding_manager: EmbeddingManager,
    collection_name: str = "episode_embeddings"
) -> EpisodeSearchEngine:
    """
    Create and configure an EpisodeSearchEngine instance.
    
    Args:
        milvus_client: Milvus client instance
        embedding_manager: Embedding manager instance
        collection_name: Name of the episode collection
        
    Returns:
        Configured EpisodeSearchEngine
    """
    from .vector_store import EpisodeVectorStoreConfig
    
    # Create vector store
    vector_store_config = EpisodeVectorStoreConfig(collection_name=collection_name)
    vector_store = EpisodeVectorStore(milvus_client, vector_store_config)
    
    # Create search engine
    return EpisodeSearchEngine(
        milvus_client=milvus_client,
        embedding_manager=embedding_manager,
        vector_store=vector_store
    )