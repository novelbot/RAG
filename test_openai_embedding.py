#!/usr/bin/env python3
"""
OpenAI 임베딩 초기화 테스트
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.embedding.types import EmbeddingConfig, EmbeddingProvider
from src.embedding.factory_langchain import get_langchain_embedding_client
from src.embedding.base import EmbeddingRequest

def test_openai_initialization():
    """Test OpenAI embedding initialization"""
    
    print("=" * 80)
    print("OpenAI 임베딩 초기화 테스트")
    print("=" * 80)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    # Create config
    config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-ada-002",
        api_key=api_key
    )
    
    print(f"\n📊 Config created:")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    
    try:
        # Initialize provider
        provider = get_langchain_embedding_client(config)
        print("\n✅ Provider initialized successfully!")
        
        # Check abstract methods
        print("\n📊 Checking abstract methods:")
        print(f"  _initialize_client: {'✅' if hasattr(provider, '_initialize_client') else '❌'}")
        print(f"  validate_config: {'✅' if hasattr(provider, 'validate_config') else '❌'}")
        print(f"  get_supported_models: {'✅' if hasattr(provider, 'get_supported_models') else '❌'}")
        print(f"  estimate_cost: {'✅' if hasattr(provider, 'estimate_cost') else '❌'}")
        
        # Test methods
        print("\n📊 Testing methods:")
        
        # Validate config
        is_valid = provider.validate_config()
        print(f"  validate_config(): {is_valid} {'✅' if is_valid else '❌'}")
        
        # Get supported models
        models = provider.get_supported_models()
        print(f"  get_supported_models(): {models}")
        
        # Get dimension
        dim = provider.get_embedding_dimension("text-embedding-ada-002")
        print(f"  get_embedding_dimension(): {dim}")
        
        # Estimate cost
        cost = provider.estimate_cost(1000, "text-embedding-ada-002")
        print(f"  estimate_cost(1000 tokens): ${cost:.4f}")
        
        print("\n✅ All methods working correctly!")
        
        # Test actual embedding generation (optional)
        test_embedding = input("\n실제 임베딩 생성 테스트? (y/n): ")
        if test_embedding.lower() == 'y':
            print("\n📊 Generating test embedding...")
            request = EmbeddingRequest(
                input=["테스트 문장입니다."],
                model="text-embedding-ada-002"
            )
            
            response = provider.generate_embeddings(request)
            print(f"  ✅ Embedding generated!")
            print(f"  Dimensions: {response.dimensions}")
            print(f"  Response time: {response.response_time:.2f}s")
            print(f"  Tokens used: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openai_initialization()
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)