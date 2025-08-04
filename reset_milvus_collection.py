#!/usr/bin/env python3
"""
Milvus 컬렉션 초기화 스크립트
기존 에피소드 컬렉션을 삭제하고 새로 생성합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.milvus.client import MilvusClient
from src.milvus.schema import RAGCollectionSchema
from src.milvus.collection import CollectionManager
from src.core.config import get_config

def reset_collection():
    """기존 컬렉션을 삭제하고 새로 생성"""
    try:
        # 설정 로드
        config = get_config()
        print(f"📋 컬렉션명: {config.milvus.collection_name}")
        
        # Milvus 클라이언트 초기화
        print("🔌 Milvus 클라이언트 연결 중...")
        client = MilvusClient(config.milvus)
        client.connect()
        print("✅ Milvus 연결 완료")
        
        # 스키마 생성
        schema = RAGCollectionSchema(
            collection_name=config.milvus.collection_name,
            vector_dim=768,  # E5 모델 차원
            description="Episode content embeddings for RAG"
        )
        
        # 기존 컬렉션 삭제 (존재할 경우)
        print("🗑️ 기존 컬렉션 확인 및 삭제 중...")
        try:
            from pymilvus import utility
            if utility.has_collection(config.milvus.collection_name):
                utility.drop_collection(config.milvus.collection_name)
                print("✅ 기존 컬렉션 삭제 완료")
            else:
                print("ℹ️ 기존 컬렉션이 없습니다")
        except Exception as e:
            print(f"⚠️ 컬렉션 삭제 중 오류 (무시): {e}")
        
        # CollectionManager를 통해 새 컬렉션 생성
        print("🔨 새 컬렉션 생성 중...")
        collection_manager = CollectionManager(client)
        collection = collection_manager.create_collection(schema)
        print("✅ 새 컬렉션 생성 완료")
        
        # 컬렉션 정보 확인
        try:
            from pymilvus import utility
            if utility.has_collection(config.milvus.collection_name):
                collection_stats = utility.get_collection_stats(config.milvus.collection_name)
                print(f"📊 컬렉션 생성 완료: {config.milvus.collection_name}")
                print(f"   - 상태: 활성화")
                print(f"   - 벡터 차원: 768")
            else:
                print("⚠️ 컬렉션 생성 확인 실패")
        except Exception as e:
            print(f"📊 컬렉션 생성 완료: {config.milvus.collection_name} (정보 확인 중 오류: {e})")
        
        # 연결 해제
        client.disconnect()
        print("🔌 Milvus 연결 해제 완료")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Milvus 컬렉션 초기화 시작")
    success = reset_collection()
    if success:
        print("🎉 컬렉션 초기화 완료!")
    else:
        print("💥 컬렉션 초기화 실패!")
        sys.exit(1)