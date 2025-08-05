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
    embedding_model: Optional[str] = None  # Use embedding manager's default model
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
        """Clean and normalize episode content while preserving structure."""
        if not content:
            return content
        
        # Basic cleaning - only strip leading/trailing whitespace
        cleaned = content.strip()
        
        # Preserve line breaks and paragraph structure
        # Only normalize excessive consecutive spaces within lines
        import re
        
        # Replace multiple consecutive spaces (but not newlines) with single space
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Remove only control characters that could break processing
        # Keep newlines (\n) and preserve text structure
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
        
        # Normalize excessive consecutive newlines to maximum 2
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned
    
    def _generate_embeddings_batch(self, episodes: List[EpisodeData]) -> None:
        """Generate embeddings for episodes with individual processing and chunking."""
        start_time = time.time()
        total_episodes = len(episodes)
        processed_count = 0
        
        try:
            self.logger.info(f"🚀 에피소드 임베딩 개별 처리 시작: {total_episodes}개")
            
            for i, episode in enumerate(episodes, 1):
                try:
                    self._generate_single_episode_embedding(episode)
                    processed_count += 1
                    
                    if i % 5 == 0 or i == total_episodes:
                        self.logger.info(f"📊 진행상황: {i}/{total_episodes} ({(i/total_episodes)*100:.1f}%)")
                
                except Exception as e:
                    self.logger.error(f"❌ Episode {episode.episode_id} 임베딩 실패: {e}")
                    # 개별 에피소드 실패는 전체 처리를 중단하지 않음
                    continue
            
            # Update statistics
            self.stats.embedding_generation_time += time.time() - start_time
            
            self.logger.info(f"✅ 임베딩 처리 완료: {processed_count}/{total_episodes} 성공")
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Batch embedding generation failed: {e}")
    
    def _generate_single_episode_embedding(self, episode: EpisodeData) -> None:
        """Generate embedding for a single episode with chunking if needed."""
        max_retries = 5  # 3 -> 5로 증가
        base_delay = 5.0  # 2.0 -> 5.0초로 증가
        
        for attempt in range(max_retries):
            try:
                # Check provider health before processing
                primary_provider = list(self.embedding_manager.providers.values())[0] if self.embedding_manager.providers else None
                if primary_provider and hasattr(primary_provider, 'health_check'):
                    health = primary_provider.health_check()
                    if health.get('status') != 'healthy':
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) 
                            self.logger.warning(f"⚠️ Provider unhealthy, waiting {delay}s before retry {attempt + 1}")
                            time.sleep(delay)
                            continue
                        else:
                            raise EmbeddingError(f"Provider unhealthy after {max_retries} attempts")
                
                # Additional delay before processing to reduce server load
                import time
                time.sleep(1)  # 모든 에피소드 처리 전 1초 대기
                
                content = episode.content
                content_length = len(content)
                
                # 토큰 수 대략 추정 (한국어: 문자당 약 1.5토큰, 영어: 4문자당 1토큰)
                estimated_tokens = int(content_length * 1.5)  # 한국어 기준 보수적 추정
                
                # 모델의 max_tokens에 따라 동적으로 청킹 임계값 결정
                max_tokens = self._get_model_max_tokens()
                chunking_threshold = int(max_tokens * 0.85)  # 85% 안전 마진
                
                if estimated_tokens <= chunking_threshold:
                    # 단일 임베딩
                    self.logger.debug(f"📄 Episode {episode.episode_id}: {estimated_tokens}토큰 (임계값: {chunking_threshold}) → 단일 임베딩")
                    self._generate_single_embedding(episode, content)
                else:
                    # 청킹 필요
                    self.logger.debug(f"📚 Episode {episode.episode_id}: {estimated_tokens}토큰 (임계값: {chunking_threshold}) → 청킹 처리")
                    self._generate_chunked_embedding(episode, content)
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if "no available providers" in error_msg or "provider unavailable" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"⚠️ Episode {episode.episode_id}: Provider unavailable, retry {attempt + 1}/{max_retries} in {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"❌ Episode {episode.episode_id}: Provider permanently unavailable after {max_retries} attempts")
                        raise
                else:
                    # For other errors, don't retry
                    self.logger.error(f"❌ Episode {episode.episode_id}: Non-retryable error: {e}")
                    raise
    
    def _generate_single_embedding(self, episode: EpisodeData, content: str) -> None:
        """Generate single embedding for episode content."""
        try:
            request = EmbeddingRequest(
                input=[content],
                encoding_format="float"
            )
            
            response = self.embedding_manager.generate_embeddings(request)
            
            if len(response.embeddings) != 1:
                raise EmbeddingError(f"Expected 1 embedding, got {len(response.embeddings)}")
            
            episode.embedding = response.embeddings[0]
            self.logger.debug(f"✅ Episode {episode.episode_id}: 단일 임베딩 완료")
            
        except Exception as e:
            self.logger.error(f"❌ Episode {episode.episode_id} 단일 임베딩 실패: {e}")
            raise
    
    def _generate_chunked_embedding(self, episode: EpisodeData, content: str) -> None:
        """Generate individual embeddings for episode chunks and store as separate chunks."""
        try:
            from .models import EpisodeChunk
            
            # 모델의 max_tokens에 따라 동적으로 청크 크기 계산
            chunk_size, overlap = self._get_optimal_chunk_settings()
            
            chunks_text = self._split_content_into_chunks(content, chunk_size, overlap)
            self.logger.debug(f"📚 Episode {episode.episode_id}: {len(chunks_text)}개 청크로 분할 (크기: {chunk_size}자, 겹침: {overlap}자)")
            
            # 각 청크를 개별 EpisodeChunk 객체로 생성
            episode_chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunk = EpisodeChunk(
                    episode_id=episode.episode_id,
                    chunk_index=i,
                    content=chunk_text,
                    episode_number=episode.episode_number,
                    episode_title=episode.episode_title,
                    publication_date=episode.publication_date,
                    novel_id=episode.novel_id,
                    total_chunks=len(chunks_text)
                )
                
                # 각 청크에 대해 개별 임베딩 생성
                chunk_retries = 3
                chunk_success = False
                
                for chunk_attempt in range(chunk_retries):
                    try:
                        # 청크 간 더 긴 대기 시간
                        if i > 0:  # 첫 번째 청크가 아닌 경우
                            time.sleep(3)
                        
                        request = EmbeddingRequest(
                            input=[chunk_text],
                            encoding_format="float"
                        )
                        
                        response = self.embedding_manager.generate_embeddings(request)
                        chunk.embedding = response.embeddings[0]
                        chunk_success = True
                        self.logger.debug(f"✅ Episode {episode.episode_id} 청크 {i+1}/{len(chunks_text)} 임베딩 완료")
                        break
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Episode {episode.episode_id} 청크 {i+1} 시도 {chunk_attempt+1} 실패: {e}")
                        if chunk_attempt < chunk_retries - 1:
                            chunk_delay = 3 * (2 ** chunk_attempt)
                            self.logger.info(f"청크 재시도 대기: {chunk_delay}초")
                            time.sleep(chunk_delay)
                        else:
                            self.logger.error(f"❌ Episode {episode.episode_id} 청크 {i+1} 최종 실패")
                            raise
                
                if not chunk_success:
                    raise EmbeddingError(f"Episode {episode.episode_id} 청크 {i+1} 처리 실패")
                
                episode_chunks.append(chunk)
            
            # 청크들을 episode에 저장 (개별 저장용)
            episode.chunks = episode_chunks
            episode.embedding = None  # 원본 에피소드는 임베딩 없음 (청크만 임베딩 있음)
            
            self.logger.debug(f"✅ Episode {episode.episode_id}: {len(episode_chunks)}개 청크 개별 임베딩 완료")
                
        except Exception as e:
            self.logger.error(f"❌ Episode {episode.episode_id} 청킹 임베딩 실패: {e}")
            raise
    
    def _split_content_into_chunks(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Split content into overlapping chunks with proper boundary detection."""
        if len(content) <= chunk_size:
            return [content]  # No need to chunk
        
        chunks = []
        start = 0
        
        while start < len(content):
            # Calculate end position
            end = start + chunk_size
            
            # If this would be the last chunk, take all remaining content
            if end >= len(content):
                final_chunk = content[start:].strip()
                if final_chunk:  # Only add if not empty
                    chunks.append(final_chunk)
                break
            
            # Find the best cutting point within the chunk
            chunk_candidate = content[start:end]
            
            # Priority 1: Try to cut at sentence boundary (. ! ? followed by space/newline)
            sentence_endings = []
            import re
            for match in re.finditer(r'[.!?][\s\n]+', chunk_candidate):
                pos = match.end()
                if pos > chunk_size * 0.6:  # At least 60% of chunk size
                    sentence_endings.append(pos)
            
            if sentence_endings:
                # Use the last sentence ending within the chunk
                cut_pos = sentence_endings[-1]
                actual_end = start + cut_pos
                chunk = content[start:actual_end].strip()
            else:
                # Priority 2: Try to cut at paragraph boundary (newline)
                newline_pos = chunk_candidate.rfind('\n')
                if newline_pos > chunk_size * 0.6:
                    actual_end = start + newline_pos
                    chunk = content[start:actual_end].strip()
                else:
                    # Priority 3: Cut at word boundary (space)
                    space_pos = chunk_candidate.rfind(' ')
                    if space_pos > chunk_size * 0.7:  # 70% threshold for word boundary
                        actual_end = start + space_pos
                        chunk = content[start:actual_end].strip()
                    else:
                        # Last resort: cut at character boundary
                        actual_end = end
                        chunk = content[start:actual_end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Calculate next start position with proper overlap
            if actual_end < len(content):
                # Move back by overlap amount, but ensure we make progress
                next_start = max(actual_end - overlap, start + chunk_size // 2)
                start = next_start
            else:
                break
        
        # Remove empty chunks and ensure no duplicates
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        # Remove exact duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _get_model_max_tokens(self) -> int:
        """Get max_tokens for the current embedding model."""
        try:
            # Get current model from embedding manager
            if not self.embedding_manager or not self.embedding_manager.providers:
                self.logger.warning("No embedding providers available, using default max_tokens")
                return 2048  # Safe default
            
            # Get primary provider
            primary_provider = list(self.embedding_manager.providers.values())[0]
            
            # For Ollama provider, check MODEL_CONFIGS
            if hasattr(primary_provider, 'MODEL_CONFIGS'):
                model_name = getattr(primary_provider, 'model', None)
                if model_name and model_name in primary_provider.MODEL_CONFIGS:
                    max_tokens = primary_provider.MODEL_CONFIGS[model_name].get('max_tokens', 2048)
                    self.logger.debug(f"📏 Model {model_name} max_tokens: {max_tokens}")
                    return max_tokens
            
            # Try to get model from provider config
            if hasattr(primary_provider, 'config'):
                config = primary_provider.config
                model_name = getattr(config, 'model', None)
                
                # Check if provider has MODEL_CONFIGS and model is in it
                if hasattr(primary_provider, 'MODEL_CONFIGS') and model_name:
                    if model_name in primary_provider.MODEL_CONFIGS:
                        max_tokens = primary_provider.MODEL_CONFIGS[model_name].get('max_tokens', 2048)
                        self.logger.debug(f"📏 Model {model_name} max_tokens: {max_tokens}")
                        return max_tokens
                
                # Check config for max_tokens
                if hasattr(config, 'max_tokens') and config.max_tokens:
                    return config.max_tokens
            
            # Check embedding manager's provider_configs
            if hasattr(self.embedding_manager, 'provider_configs'):
                # Handle both single config and list of configs
                configs = self.embedding_manager.provider_configs
                if not isinstance(configs, list):
                    configs = [configs]
                
                for config in configs:
                    if hasattr(config, 'model'):
                        model_name = config.model
                        # Try to find max_tokens from Ollama MODEL_CONFIGS
                        from src.embedding.providers.ollama import OllamaEmbeddingProvider
                        
                        # Try exact match first
                        if model_name in OllamaEmbeddingProvider.MODEL_CONFIGS:
                            max_tokens = OllamaEmbeddingProvider.MODEL_CONFIGS[model_name].get('max_tokens', 2048)
                            self.logger.debug(f"📏 Model {model_name} max_tokens: {max_tokens} (from Ollama configs)")
                            return max_tokens
                        
                        # Try without version tag (e.g., remove :f32, :q8_0, etc.)
                        base_model_name = model_name.split(':')[0] if ':' in model_name else model_name
                        if base_model_name in OllamaEmbeddingProvider.MODEL_CONFIGS:
                            max_tokens = OllamaEmbeddingProvider.MODEL_CONFIGS[base_model_name].get('max_tokens', 2048)
                            self.logger.debug(f"📏 Model {base_model_name} (from {model_name}) max_tokens: {max_tokens} (from Ollama configs)")
                            return max_tokens
            
            # Safe default
            self.logger.warning("Could not determine model max_tokens, using default 2048")
            return 2048
            
        except Exception as e:
            self.logger.error(f"Error getting model max_tokens: {e}")
            return 2048  # Safe fallback
    
    def _get_optimal_chunk_settings(self) -> tuple[int, int]:
        """Calculate optimal chunk size and overlap based on model's max_tokens."""
        try:
            # Get model's max_tokens
            max_tokens = self._get_model_max_tokens()
            
            # Calculate safe token count (85% of max to be conservative)
            safe_tokens = int(max_tokens * 0.85)
            
            # Convert to character count (Korean text: ~1.5 tokens per character)
            safe_chars = int(safe_tokens / 1.5)
            
            # Use smaller of default (1500) or calculated safe size
            chunk_size = min(1500, safe_chars)
            
            # Calculate proportional overlap (maintain ~13.3% ratio from 200/1500)
            overlap_ratio = 200 / 1500  # 0.133
            overlap = max(20, min(200, int(chunk_size * overlap_ratio)))
            
            self.logger.debug(f"📏 Dynamic chunking: max_tokens={max_tokens}, chunk_size={chunk_size}, overlap={overlap}")
            
            return chunk_size, overlap
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal chunk settings: {e}")
            # Return safe defaults
            return 300, 50
    
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