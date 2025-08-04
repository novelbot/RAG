"""
Multi-Model Episode RAG 사용 예시.
"""

import asyncio
from src.core.config import get_config
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.milvus.client import MilvusClient
from src.episode.multi_model_manager import MultiModelEpisodeRAGManager, create_multi_model_config


async def main():
    """다중 모델 RAG 사용 예시."""
    
    # 설정 로드
    config = get_config()
    
    # 기본 컴포넌트 초기화
    db_manager = DatabaseManager(config.database)
    embedding_manager = EmbeddingManager(config.embedding_providers)
    milvus_client = MilvusClient(config.milvus)
    milvus_client.connect()
    
    # 다중 모델 설정
    multi_config = create_multi_model_config()
    
    # 다중 모델 매니저 생성
    multi_manager = MultiModelEpisodeRAGManager(
        database_manager=db_manager,
        embedding_manager=embedding_manager,
        milvus_client=milvus_client,
        config=multi_config
    )
    
    # 1. 모든 모델의 컬렉션 설정
    print("🔧 Setting up collections for all models...")
    await multi_manager.setup_collections_for_all_models(drop_existing=True)
    
    # 2. 모델 정보 확인
    print("\n📊 Model Information:")
    model_info = multi_manager.get_model_info()
    for model_name, info in model_info.items():
        print(f"  {model_name}:")
        print(f"    Collection: {info['collection_name']}")
        print(f"    Dimension: {info['vector_dimension']}")
        print(f"    Ready: {info['is_collection_ready']}")
    
    # 3. 특정 모델로 소설 처리
    novel_id = 1
    print(f"\n🔄 Processing novel {novel_id} with Ollama model...")
    result = await multi_manager.process_novel_with_model(
        novel_id=novel_id,
        model_name="ollama-nomic",
        force_reprocess=True
    )
    print(f"Result: {result.get('episodes_processed', 0)} episodes processed")
    
    # 4. 모든 모델로 소설 처리 (선택사항)
    print(f"\n🔄 Processing novel {novel_id} with ALL models...")
    all_results = await multi_manager.process_novel_with_all_models(
        novel_id=novel_id,
        force_reprocess=True
    )
    
    for model_name, result in all_results.items():
        if "error" in result:
            print(f"  {model_name}: ERROR - {result['error']}")
        else:
            print(f"  {model_name}: {result.get('episodes_processed', 0)} episodes")
    
    # 5. 특정 모델로 검색
    query = "주인공이 마법을 사용하는 장면"
    print(f"\n🔍 Searching with Ollama model: '{query}'")
    search_result = multi_manager.search_with_model(
        query=query,
        model_name="ollama-nomic",
        limit=5
    )
    print(f"Found {len(search_result.hits)} results")
    
    # 6. 모든 모델로 검색 비교
    print(f"\n🔍 Searching with ALL models: '{query}'")
    all_search_results = multi_manager.search_with_all_models(
        query=query,
        limit=3
    )
    
    for model_name, result in all_search_results.items():
        if "error" in result:
            print(f"  {model_name}: ERROR - {result['error']}")
        else:
            print(f"  {model_name}: {len(result.hits)} results, top score: {result.hits[0].score:.4f}")
    
    # 7. 기본 모델 변경
    print(f"\n⚙️ Switching default model to OpenAI...")
    multi_manager.switch_default_model("openai-small")
    default_manager = multi_manager.get_default_manager()
    print(f"New default collection: {default_manager.config.collection_name}")


if __name__ == "__main__":
    asyncio.run(main())