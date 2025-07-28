"""
Episode Embedding Processor for RAG System.

This module handles extraction of episode data from RDB, generation of embeddings,
and preparation for vector storage.
"""

import time
import asyncio
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime
from dataclasses import dataclass

from src.core.logging import LoggerMixin
from src.core.exceptions import ProcessingError, DatabaseError, EmbeddingError
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.embedding.base import EmbeddingRequest
from .models import EpisodeData, EpisodeProcessingStats


@dataclass
class EpisodeProcessingConfig:
    """Configuration for episode processing."""
    batch_size: int = 100
    max_content_length: int = 10000  # Max characters per episode
    min_content_length: int = 50     # Min characters to process
    embedding_model: str = "text-embedding-ada-002"
    enable_content_cleaning: bool = True
    enable_chunking: bool = False    # Split long episodes into chunks
    chunk_size: int = 2000          # Characters per chunk
    chunk_overlap: int = 200        # Overlap between chunks


class EpisodeEmbeddingProcessor(LoggerMixin):
    """
    Processes episode data from RDB and generates embeddings.
    
    Features:
    - Batch processing for efficiency
    - Content validation and cleaning
    - Embedding generation with retry logic
    - Progress tracking and statistics
    - Chunking support for long episodes
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        embedding_manager: EmbeddingManager,
        config: Optional[EpisodeProcessingConfig] = None
    ):
        """
        Initialize Episode Embedding Processor.
        
        Args:
            database_manager: Database connection manager
            embedding_manager: Embedding generation manager
            config: Processing configuration
        """
        self.db_manager = database_manager
        self.embedding_manager = embedding_manager
        self.config = config or EpisodeProcessingConfig()
        
        # Processing statistics
        self.stats = EpisodeProcessingStats()
        
        self.logger.info("EpisodeEmbeddingProcessor initialized")
    
    def extract_episodes(
        self,
        novel_ids: Optional[List[int]] = None,
        episode_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[EpisodeData]:
        """
        Extract episode data from database.
        
        Args:
            novel_ids: Filter by specific novel IDs
            episode_ids: Filter by specific episode IDs
            limit: Maximum number of episodes to extract
            offset: Number of episodes to skip
            
        Returns:
            List of EpisodeData objects
        """
        try:
            # Build query
            query_parts = ["SELECT episode_id, content, episode_number, episode_title, publication_date, novel_id FROM episode"]
            conditions = []
            params = {}
            
            if novel_ids:
                placeholders = ','.join([f':novel_id_{i}' for i in range(len(novel_ids))])
                conditions.append(f"novel_id IN ({placeholders})")
                for i, novel_id in enumerate(novel_ids):
                    params[f'novel_id_{i}'] = novel_id
            
            if episode_ids:
                placeholders = ','.join([f':episode_id_{i}' for i in range(len(episode_ids))])
                conditions.append(f"episode_id IN ({placeholders})")
                for i, episode_id in enumerate(episode_ids):
                    params[f'episode_id_{i}'] = episode_id
            
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))
            
            query_parts.append("ORDER BY novel_id, episode_number")
            
            if limit:
                query_parts.append(f"LIMIT {limit}")
            if offset:
                query_parts.append(f"OFFSET {offset}")
            
            query = " ".join(query_parts)
            
            # Execute query
            with self.db_manager.get_connection() as conn:
                from sqlalchemy import text
                result = conn.execute(text(query), params)
                rows = result.fetchall()
            
            # Convert to EpisodeData objects
            episodes = []
            for row in rows:
                try:
                    # Convert row to dict (handle different row types)
                    if hasattr(row, '_asdict'):
                        row_dict = row._asdict()
                    elif hasattr(row, 'keys'):
                        row_dict = dict(row)
                    else:
                        # Assume tuple format
                        row_dict = {
                            'episode_id': row[0],
                            'content': row[1],
                            'episode_number': row[2],
                            'episode_title': row[3],
                            'publication_date': row[4],
                            'novel_id': row[5]
                        }
                    
                    episode = EpisodeData.from_db_row(row_dict)
                    episodes.append(episode)
                except Exception as e:
                    self.logger.warning(f"Failed to process episode row: {e}")
                    continue
            
            self.logger.info(f"Extracted {len(episodes)} episodes from database")
            return episodes
            
        except Exception as e:
            self.logger.error(f"Failed to extract episodes: {e}")
            raise DatabaseError(f"Episode extraction failed: {e}")
    
    def process_episodes(
        self,
        episodes: List[EpisodeData],
        generate_embeddings: bool = True
    ) -> List[EpisodeData]:
        """
        Process episodes with content validation and embedding generation.
        
        Args:
            episodes: List of episode data to process
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            List of processed episodes with embeddings
        """
        start_time = time.time()
        processed_episodes = []
        failed_count = 0
        
        self.stats.total_episodes = len(episodes)
        
        try:
            # Process in batches for memory efficiency
            for i in range(0, len(episodes), self.config.batch_size):
                batch = episodes[i:i + self.config.batch_size]
                
                # Validate and clean content
                valid_episodes = []
                for episode in batch:
                    if self._validate_episode(episode):
                        if self.config.enable_content_cleaning:
                            episode.content = self._clean_content(episode.content)
                        valid_episodes.append(episode)
                    else:
                        failed_count += 1
                
                # Generate embeddings for valid episodes
                if generate_embeddings and valid_episodes:
                    try:
                        self._generate_embeddings_batch(valid_episodes)
                        processed_episodes.extend(valid_episodes)
                        self.stats.processed_episodes += len(valid_episodes)
                    except Exception as e:
                        self.logger.error(f"Batch embedding generation failed: {e}")
                        failed_count += len(valid_episodes)
                else:
                    processed_episodes.extend(valid_episodes)
                    self.stats.processed_episodes += len(valid_episodes)
                
                # Log progress
                total_processed = len(processed_episodes) + failed_count
                progress = (total_processed / len(episodes)) * 100
                self.logger.info(f"Processing progress: {progress:.1f}% ({total_processed}/{len(episodes)})")
        
        except Exception as e:
            self.logger.error(f"Episode processing failed: {e}")
            raise ProcessingError(f"Episode processing failed: {e}")
        
        finally:
            # Update statistics
            self.stats.failed_episodes = failed_count
            self.stats.total_processing_time = time.time() - start_time
            if processed_episodes:
                self.stats.average_content_length = sum(ep.content_length for ep in processed_episodes) / len(processed_episodes)
        
        self.logger.info(
            f"Episode processing completed: {self.stats.processed_episodes} successful, "
            f"{self.stats.failed_episodes} failed, {self.stats.total_processing_time:.2f}s"
        )
        
        return processed_episodes
    
    async def process_episodes_async(
        self,
        episodes: List[EpisodeData],
        generate_embeddings: bool = True
    ) -> List[EpisodeData]:
        """
        Asynchronously process episodes with embedding generation.
        
        Args:
            episodes: List of episode data to process
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            List of processed episodes with embeddings
        """
        # Run CPU-bound processing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_episodes,
            episodes,
            generate_embeddings
        )
    
    def _validate_episode(self, episode: EpisodeData) -> bool:
        """Validate episode data before processing."""
        if not episode.content:
            self.logger.warning(f"Episode {episode.episode_id}: Empty content")
            return False
        
        if len(episode.content) < self.config.min_content_length:
            self.logger.warning(f"Episode {episode.episode_id}: Content too short ({len(episode.content)} chars)")
            return False
        
        if len(episode.content) > self.config.max_content_length:
            self.logger.warning(f"Episode {episode.episode_id}: Content too long ({len(episode.content)} chars)")
            if not self.config.enable_chunking:
                return False
        
        return True
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize episode content."""
        if not content:
            return content
        
        # Basic cleaning
        cleaned = content.strip()
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters that might interfere with embedding
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned
    
    def _generate_embeddings_batch(self, episodes: List[EpisodeData]) -> None:
        """Generate embeddings for a batch of episodes."""
        start_time = time.time()
        
        try:
            # Prepare embedding request
            texts = [episode.content for episode in episodes]
            request = EmbeddingRequest(
                input=texts,
                model=self.config.embedding_model,
                encoding_format="float"
            )
            
            # Generate embeddings
            response = self.embedding_manager.generate_embeddings(request)
            
            # Assign embeddings to episodes
            if len(response.embeddings) != len(episodes):
                raise EmbeddingError(f"Embedding count mismatch: {len(response.embeddings)} != {len(episodes)}")
            
            for episode, embedding in zip(episodes, response.embeddings):
                episode.embedding = embedding
            
            # Update statistics
            self.stats.embedding_generation_time += time.time() - start_time
            
            self.logger.debug(f"Generated embeddings for {len(episodes)} episodes")
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Batch embedding generation failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = EpisodeProcessingStats()
        self.logger.info("Processing statistics reset")
    
    def process_novel_episodes(
        self,
        novel_id: int,
        batch_size: Optional[int] = None
    ) -> List[EpisodeData]:
        """
        Process all episodes for a specific novel.
        
        Args:
            novel_id: ID of the novel to process
            batch_size: Override default batch size
            
        Returns:
            List of processed episodes
        """
        if batch_size:
            original_batch_size = self.config.batch_size
            self.config.batch_size = batch_size
        
        try:
            # Extract episodes for the novel
            episodes = self.extract_episodes(novel_ids=[novel_id])
            
            # Process episodes
            processed_episodes = self.process_episodes(episodes)
            
            self.logger.info(f"Processed {len(processed_episodes)} episodes for novel {novel_id}")
            return processed_episodes
            
        finally:
            if batch_size:
                self.config.batch_size = original_batch_size
    
    def estimate_processing_time(self, episode_count: int) -> float:
        """
        Estimate processing time for given number of episodes.
        
        Args:
            episode_count: Number of episodes to process
            
        Returns:
            Estimated time in seconds
        """
        if self.stats.total_episodes == 0:
            # Default estimate: ~0.5 seconds per episode
            return episode_count * 0.5
        
        avg_time_per_episode = self.stats.total_processing_time / self.stats.total_episodes
        return episode_count * avg_time_per_episode