"""
Multi-Model Episode RAG Manager - 모델별 별도 컬렉션 관리.

여러 임베딩 모델을 사용하여 각각 다른 컬렉션에 저장하고 검색할 수 있도록 지원.
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from src.core.logging import LoggerMixin
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient

from .manager import EpisodeRAGManager, EpisodeRAGConfig


@dataclass
class MultiModelConfig:
    """다중 모델 설정."""
    models: Dict[str, Dict[str, Any]]  # model_name -> {dimension, collection_suffix}
    default_model: str
    processing_batch_size: int = 5


class MultiModelEpisodeRAGManager(LoggerMixin):
    """
    여러 임베딩 모델을 동시에 지원하는 Episode RAG Manager.
    
    Features:
    - 모델별 별도 컬렉션 관리
    - 동적 모델 전환
    - 모델별 검색 지원
    - 모델간 결과 비교
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        embedding_manager: EmbeddingManager,
        milvus_client: MilvusClient,
        config: MultiModelConfig
    ):
        self.db_manager = database_manager
        self.embedding_manager = embedding_manager
        self.milvus_client = milvus_client
        self.config = config
        
        # 모델별 RAG Manager 인스턴스
        self.model_managers: Dict[str, EpisodeRAGManager] = {}
        
        self._initialize_model_managers()
        self.logger.info(f"MultiModelEpisodeRAGManager initialized with {len(self.model_managers)} models")
    
    def _initialize_model_managers(self) -> None:
        """각 모델별로 EpisodeRAGManager 인스턴스 생성."""
        for model_name, model_info in self.config.models.items():
            # 모델별 컬렉션 이름 생성
            collection_name = f"episode_embeddings_{model_info['collection_suffix']}"
            
            # 모델별 설정 생성
            rag_config = EpisodeRAGConfig(
                collection_name=collection_name,
                vector_dimension=model_info['dimension'],
                embedding_model=model_name,
                processing_batch_size=self.config.processing_batch_size
            )
            
            # 모델별 RAG Manager 생성
            manager = EpisodeRAGManager(
                database_manager=self.db_manager,
                embedding_manager=self.embedding_manager,
                milvus_client=self.milvus_client,
                config=rag_config
            )
            
            self.model_managers[model_name] = manager
            self.logger.info(f"Initialized manager for model '{model_name}' with collection '{collection_name}'")
    
    async def setup_collections_for_all_models(self, drop_existing: bool = False) -> None:
        """모든 모델의 컬렉션을 설정."""
        for model_name, manager in self.model_managers.items():
            try:
                await manager.setup_collection(drop_existing=drop_existing)
                self.logger.info(f"✓ Collection setup completed for model '{model_name}'")
            except Exception as e:
                self.logger.error(f"✗ Collection setup failed for model '{model_name}': {e}")
                raise
    
    async def process_novel_with_model(
        self,
        novel_id: int,
        model_name: str,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """특정 모델로 소설 처리."""
        if model_name not in self.model_managers:
            raise ValueError(f"Model '{model_name}' not configured")
        
        manager = self.model_managers[model_name]
        return await manager.process_novel(novel_id, force_reprocess)
    
    async def process_novel_with_all_models(
        self,
        novel_id: int,
        force_reprocess: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """모든 모델로 소설 처리."""
        results = {}
        
        for model_name, manager in self.model_managers.items():
            try:
                self.logger.info(f"Processing novel {novel_id} with model '{model_name}'")
                result = await manager.process_novel(novel_id, force_reprocess)
                results[model_name] = result
                self.logger.info(f"✓ Novel {novel_id} processed with model '{model_name}'")
            except Exception as e:
                self.logger.error(f"✗ Novel {novel_id} failed with model '{model_name}': {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def search_with_model(
        self,
        query: str,
        model_name: str,
        limit: int = 10,
        **kwargs
    ) -> Any:
        """특정 모델로 검색."""
        if model_name not in self.model_managers:
            raise ValueError(f"Model '{model_name}' not configured")
        
        manager = self.model_managers[model_name]
        return manager.search_engine.search_by_query(query, limit=limit, **kwargs)
    
    def search_with_all_models(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """모든 모델로 검색하여 결과 비교."""
        results = {}
        
        for model_name, manager in self.model_managers.items():
            try:
                result = manager.search_engine.search_by_query(query, limit=limit, **kwargs)
                results[model_name] = result
            except Exception as e:
                self.logger.error(f"Search failed with model '{model_name}': {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """모델별 정보 반환."""
        info = {}
        
        for model_name, manager in self.model_managers.items():
            info[model_name] = {
                "collection_name": manager.config.collection_name,
                "vector_dimension": manager.config.vector_dimension,
                "embedding_model": manager.config.embedding_model,
                "is_collection_ready": hasattr(manager.vector_store, 'collection') and manager.vector_store.collection is not None
            }
        
        return info
    
    def switch_default_model(self, model_name: str) -> None:
        """기본 모델 변경."""
        if model_name not in self.model_managers:
            raise ValueError(f"Model '{model_name}' not configured")
        
        self.config.default_model = model_name
        self.logger.info(f"Default model switched to '{model_name}'")
    
    def get_default_manager(self) -> EpisodeRAGManager:
        """기본 모델의 매니저 반환."""
        return self.model_managers[self.config.default_model]


# 사용 예시 설정
def create_multi_model_config() -> MultiModelConfig:
    """다중 모델 설정 예시."""
    return MultiModelConfig(
        models={
            "ollama-nomic": {
                "dimension": 1024,
                "collection_suffix": "ollama_nomic"
            },
            "openai-small": {
                "dimension": 1536, 
                "collection_suffix": "openai_small"
            },
            "openai-large": {
                "dimension": 3072,
                "collection_suffix": "openai_large"
            }
        },
        default_model="ollama-nomic",
        processing_batch_size=5
    )