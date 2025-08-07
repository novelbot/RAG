#!/usr/bin/env python3
"""
Simple test to verify LangChain refactoring basic structure.
"""

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    # Test LLM imports
    try:
        from src.llm import (
            LLMManager, LLMProvider, LLMConfig, 
            LLMRequest, LLMMessage, LLMRole,
            OllamaProvider
        )
        print("✅ LLM imports successful")
    except ImportError as e:
        print(f"❌ LLM import failed: {e}")
        return False
    
    # Test embedding imports
    try:
        from src.embedding import (
            EmbeddingManager, EmbeddingProvider, EmbeddingConfig,
            OllamaEmbeddingProvider, get_embedding_client
        )
        print("✅ Embedding imports successful")
    except ImportError as e:
        print(f"❌ Embedding import failed: {e}")
        return False
    
    # Test deprecated providers
    try:
        from src.llm import OpenAIProvider
        # Should be importable but raise error on instantiation
        try:
            provider = OpenAIProvider()
            print("❌ OpenAIProvider should raise NotImplementedError")
            return False
        except NotImplementedError:
            print("✅ OpenAIProvider correctly deprecated")
    except ImportError as e:
        print(f"❌ Could not import OpenAIProvider: {e}")
        return False
    
    return True


def test_config_creation():
    """Test that configurations can be created."""
    print("\nTesting configuration creation...")
    
    from src.llm import LLMConfig, LLMProvider
    from src.embedding import EmbeddingConfig, EmbeddingProvider
    
    # Test LLM config
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama3.2",
        base_url="http://localhost:11434"
    )
    print(f"✅ Created LLM config: {llm_config.provider.value}")
    
    # Test embedding config
    embed_config = EmbeddingConfig(
        provider=EmbeddingProvider.OLLAMA,
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    print(f"✅ Created embedding config: {embed_config.provider.value}")
    
    return True


def test_langchain_modules():
    """Test that LangChain modules are available."""
    print("\nTesting LangChain modules...")
    
    try:
        import langchain_openai
        print("✅ langchain-openai installed")
    except ImportError:
        print("❌ langchain-openai not installed")
    
    try:
        import langchain_google_genai
        print("✅ langchain-google-genai installed")
    except ImportError:
        print("❌ langchain-google-genai not installed")
    
    try:
        import langchain_anthropic
        print("✅ langchain-anthropic installed")
    except ImportError:
        print("❌ langchain-anthropic not installed")
    
    try:
        import langchain_community
        print("✅ langchain-community installed")
    except ImportError:
        print("❌ langchain-community not installed")
    
    return True


def main():
    """Run all tests."""
    print("🧪 LangChain Refactoring - Simple Tests")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_config_creation():
        all_passed = False
    
    if not test_langchain_modules():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All simple tests passed!")
    else:
        print("❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)