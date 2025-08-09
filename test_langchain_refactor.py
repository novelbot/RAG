"""
Test script to verify LangChain refactoring works correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_embedding_factory():
    """Test embedding factory with LangChain."""
    print("\n=== Testing Embedding Factory ===")
    
    from src.embedding.types import EmbeddingConfig, EmbeddingProvider
    from src.embedding.factory import get_embedding_client
    from src.embedding.base import EmbeddingRequest
    
    # Test OpenAI embeddings
    if os.getenv("OPENAI_API_KEY"):
        print("\nTesting OpenAI embeddings...")
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        client = get_embedding_client(config)
        request = EmbeddingRequest(input=["Hello, world!"])
        
        try:
            response = await client.generate_embeddings_async(request)
            print(f"✅ OpenAI embeddings work! Dimension: {response.dimensions}")
        except Exception as e:
            print(f"❌ OpenAI embeddings failed: {e}")
    
    # Test Ollama embeddings (if available)
    print("\nTesting Ollama embeddings...")
    config = EmbeddingConfig(
        provider=EmbeddingProvider.OLLAMA,
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    client = get_embedding_client(config)
    request = EmbeddingRequest(input=["Hello, Ollama!"])
    
    try:
        response = await client.generate_embeddings_async(request)
        print(f"✅ Ollama embeddings work! Dimension: {response.dimensions}")
    except Exception as e:
        print(f"⚠️ Ollama embeddings failed (expected if Ollama not running): {e}")


async def test_llm_manager():
    """Test LLM manager with LangChain."""
    print("\n=== Testing LLM Manager ===")
    
    from src.llm import (
        LLMManager, LLMProvider, LLMConfig, LLMRequest, 
        LLMMessage, LLMRole, ProviderConfig
    )
    
    configs = []
    
    # Add OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        configs.append(ProviderConfig(
            provider=LLMProvider.OPENAI,
            config=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3
            )
        ))
    
    # Add Ollama
    configs.append(ProviderConfig(
        provider=LLMProvider.OLLAMA,
        config=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
            base_url="http://localhost:11434"
        )
    ))
    
    if not configs:
        print("❌ No LLM providers configured. Set OPENAI_API_KEY or run Ollama.")
        return
    
    # Create manager
    manager = LLMManager(configs)
    
    # Test generation
    # Use the first available model
    test_model = configs[0].config.model if configs else "llama3.2"
    request = LLMRequest(
        model=test_model,
        messages=[
            LLMMessage(role=LLMRole.USER, content="Say hello in one word")
        ],
        max_tokens=10
    )
    
    try:
        response = await manager.generate_async(request)
        print(f"✅ LLM Manager works! Response: {response.content}")
        
        # Print stats
        stats = manager.get_stats()
        print("\nProvider stats:")
        for provider, provider_stats in stats.items():
            print(f"  {provider}: {provider_stats}")
            
    except Exception as e:
        print(f"❌ LLM Manager failed: {e}")


def test_backward_compatibility():
    """Test that old imports still work (with deprecation)."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test that old provider imports raise helpful errors
    try:
        from src.llm import OpenAIProvider
        provider = OpenAIProvider()
        print("❌ OpenAIProvider should raise NotImplementedError")
    except NotImplementedError as e:
        print(f"✅ OpenAIProvider correctly deprecated: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
    
    # Test that Ollama provider still works
    try:
        from src.llm import OllamaProvider
        print("✅ OllamaProvider still available")
    except Exception as e:
        print(f"❌ OllamaProvider import failed: {e}")
    
    # Test embedding providers
    try:
        from src.embedding import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider()
        print("❌ OpenAIEmbeddingProvider should raise NotImplementedError")
    except NotImplementedError as e:
        print(f"✅ OpenAIEmbeddingProvider correctly deprecated: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")


async def test_langchain_rag():
    """Test LangChain RAG integration."""
    print("\n=== Testing LangChain RAG ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Skipping RAG test - OPENAI_API_KEY not set")
        return
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        from src.rag.langchain_rag import LangChainRAG, LangChainRAGConfig, RAGStrategy
        
        # Setup components
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = InMemoryVectorStore(embeddings)
        
        # Add some documents
        texts = [
            "LangChain is a framework for building LLM applications.",
            "RAG stands for Retrieval-Augmented Generation.",
            "Milvus is a vector database for similarity search."
        ]
        vector_store.add_texts(texts)
        
        # Create RAG system
        retriever = vector_store.as_retriever()
        config = LangChainRAGConfig(
            strategy=RAGStrategy.SIMPLE,
            retrieval_k=2,
            temperature=0.3
        )
        
        rag = LangChainRAG(llm, retriever, config)
        
        # Test query
        result = await rag.aquery("What is LangChain?")
        print(f"✅ LangChain RAG works!")
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Sources: {len(result['sources'])} documents retrieved")
        
    except Exception as e:
        print(f"❌ LangChain RAG test failed: {e}")


async def main():
    """Run all tests."""
    print("🧪 Testing LangChain Refactoring")
    print("=" * 50)
    
    # Run tests
    await test_embedding_factory()
    await test_llm_manager()
    test_backward_compatibility()
    await test_langchain_rag()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())