#!/usr/bin/env python3
"""
Simple test script for implemented TODO features
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from cli.commands.model import (
        _get_model_capabilities, 
        _get_embedding_models,
        _validate_model_config,
        BenchmarkResult
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_model_capabilities():
    """Test model capabilities function"""
    # Test LLM capabilities
    gpt4_caps = _get_model_capabilities('gpt-4', 'llm')
    assert 'Advanced reasoning' in gpt4_caps
    
    claude_caps = _get_model_capabilities('claude-3-5-sonnet-20241022', 'llm')
    assert 'Balanced performance' in claude_caps
    
    # Test embedding capabilities
    ada_caps = _get_model_capabilities('text-embedding-ada-002', 'embedding')
    assert 'embeddings' in ada_caps
    
    minilm_caps = _get_model_capabilities('all-MiniLM-L6-v2', 'embedding')
    assert 'Efficient' in minilm_caps


def test_embedding_models():
    """Test embedding models listing"""
    # Test all models
    all_models = _get_embedding_models(None)
    assert len(all_models) > 0
    
    # Test filtered models
    st_models = _get_embedding_models('sentence-transformers')
    assert len(st_models) > 0
    assert all(m['provider'] == 'sentence-transformers' for m in st_models)
    
    hf_models = _get_embedding_models('huggingface')
    assert len(hf_models) > 0
    assert all(m['provider'] == 'huggingface' for m in hf_models)


def test_config_validation():
    """Test model configuration validation"""
    from types import SimpleNamespace
    
    # Test valid config
    config = SimpleNamespace()
    config.llm = SimpleNamespace()
    config.llm.provider = 'openai' 
    config.llm.model = 'gpt-4'
    config.llm.openai_api_key = 'test-key'
    
    errors = _validate_model_config(config)
    assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"
    
    # Test invalid provider
    config.llm.provider = 'invalid_provider'
    errors = _validate_model_config(config)
    assert len(errors) > 0
    assert any('Unsupported LLM provider' in error for error in errors)
    
    # Test missing API key
    config.llm.provider = 'openai'
    delattr(config.llm, 'openai_api_key')
    errors = _validate_model_config(config)
    assert len(errors) > 0
    assert any('API key required' in error for error in errors)


def test_benchmark_result():
    """Test BenchmarkResult class"""
    result = BenchmarkResult(
        model_name='test-model',
        provider='test-provider', 
        model_type='llm',
        response_times=[1.0, 1.2, 0.8, 1.5],
        success_count=4,
        error_count=1
    )
    
    # Test calculated properties
    assert result.avg_response_time == 1.125
    assert result.min_response_time == 0.8
    assert result.max_response_time == 1.5
    assert result.success_rate == 80.0  # 4 out of 5 total
    assert result.throughput_per_min > 0


if __name__ == "__main__":
    # Run tests
    print("Running implemented feature tests...")
    
    try:
        test_model_capabilities()
        print("✓ Model capabilities test passed")
        
        test_embedding_models()
        print("✓ Embedding models test passed")
        
        test_config_validation()
        print("✓ Configuration validation test passed")
        
        test_benchmark_result()
        print("✓ Benchmark result test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)