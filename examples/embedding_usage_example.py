"""
Example usage of the Embedding Model Integration Layer.

This example demonstrates how to use the Embedding Manager with multiple providers,
dimension management, caching, and optimization features.
"""

import asyncio
import os
from typing import List

from src.embedding import (
    EmbeddingManager, EmbeddingProvider, EmbeddingConfig, EmbeddingRequest,
    EmbeddingProviderConfig, EmbeddingLoadBalancingStrategy
)
from src.embedding.utils import (
    EmbeddingOptimizer, EmbeddingOptimizationConfig, 
    DimensionReductionMethod, SimilarityMetric
)


async def main():
    """Main example function demonstrating Embedding Manager usage."""
    
    # Configure multiple providers
    provider_configs = [
        EmbeddingProviderConfig(
            provider=EmbeddingProvider.OPENAI,
            config=EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-large",
                dimensions=1024,
                timeout=30.0,
                max_retries=3,
                batch_size=100,
                normalize_embeddings=True
            ),
            priority=1,
            max_requests_per_minute=100,
            weight=1.0,
            cost_per_1m_tokens=0.13
        ),
        EmbeddingProviderConfig(
            provider=EmbeddingProvider.GOOGLE,
            config=EmbeddingConfig(
                provider=EmbeddingProvider.GOOGLE,
                api_key=os.getenv("GOOGLE_API_KEY"),
                model="text-embedding-004",
                dimensions=768,
                timeout=30.0,
                max_retries=3,
                batch_size=100,
                normalize_embeddings=True
            ),
            priority=2,
            max_requests_per_minute=60,
            weight=1.0,
            cost_per_1m_tokens=0.025
        ),
        EmbeddingProviderConfig(
            provider=EmbeddingProvider.OLLAMA,
            config=EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                base_url="http://localhost:11434",
                model="nomic-embed-text",
                timeout=30.0,
                max_retries=3,
                batch_size=50,
                normalize_embeddings=True
            ),
            priority=3,
            max_requests_per_minute=120,  # Local model, higher rate limit
            weight=1.0,
            cost_per_1m_tokens=0.0  # Free local model
        )
    ]
    
    # Initialize Embedding Manager with caching enabled
    embedding_manager = EmbeddingManager(provider_configs, enable_cache=True)
    
    # Set load balancing strategy
    embedding_manager.set_load_balancing_strategy(EmbeddingLoadBalancingStrategy.COST_OPTIMIZED)
    
    print("=== Embedding Manager Status ===")
    print(f"Manager Info: {embedding_manager.get_manager_info()}")
    print()
    
    # Example 1: Single text embedding
    print("=== Example 1: Single Text Embedding ===")
    
    single_text = "What is machine learning and how does it work?"
    
    request = EmbeddingRequest(
        input=[single_text],
        model="text-embedding-3-large",
        dimensions=512,
        normalize=True
    )
    
    try:
        response = await embedding_manager.generate_embeddings_async(request)
        print(f"Text: {single_text}")
        print(f"Embedding dimensions: {response.dimensions}")
        print(f"Provider: {response.metadata.get('provider', 'unknown')}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Response Time: {response.response_time:.2f}s")
        print(f"Cached: {response.metadata.get('cached', False)}")
        print(f"Cost: ${response.metadata.get('cost', 0.0):.6f}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 2: Batch embedding processing
    print("=== Example 2: Batch Embedding Processing ===")
    
    batch_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "Deep learning uses multiple layers of neural networks.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual data."
    ]
    
    batch_request = EmbeddingRequest(
        input=batch_texts,
        model="text-embedding-004",
        dimensions=768,
        normalize=True,
        batch_size=3
    )
    
    try:
        batch_response = await embedding_manager.generate_embeddings_async(batch_request)
        print(f"Processed {len(batch_texts)} texts")
        print(f"Total embeddings: {len(batch_response.embeddings)}")
        print(f"Dimensions: {batch_response.dimensions}")
        print(f"Provider: {batch_response.metadata.get('provider', 'unknown')}")
        print(f"Total tokens: {batch_response.usage.total_tokens}")
        print(f"Response Time: {batch_response.response_time:.2f}s")
        print(f"Cost: ${batch_response.metadata.get('cost', 0.0):.6f}")
        print()
    except Exception as e:
        print(f"Batch processing error: {e}")
        print()
    
    # Example 3: Dimension optimization
    print("=== Example 3: Dimension Optimization ===")
    
    # Generate high-dimensional embeddings
    high_dim_request = EmbeddingRequest(
        input=["This is a test sentence for dimension reduction."],
        model="text-embedding-3-large",
        dimensions=3072  # Full dimensions
    )
    
    try:
        high_dim_response = await embedding_manager.generate_embeddings_async(high_dim_request)
        print(f"Original dimensions: {high_dim_response.dimensions}")
        
        # Initialize optimizer
        optimizer_config = EmbeddingOptimizationConfig(
            target_dimensions=256,
            reduction_method=DimensionReductionMethod.PCA,
            normalize_before_reduction=True,
            normalize_after_reduction=True
        )
        
        optimizer = EmbeddingOptimizer(optimizer_config)
        
        # Reduce dimensions
        optimized_response = optimizer.reduce_dimensions(high_dim_response)
        print(f"Optimized dimensions: {optimized_response.dimensions}")
        print(f"Compression ratio: {optimized_response.metadata.get('compression_ratio', 0.0):.2f}")
        print(f"Reduction method: {optimized_response.metadata.get('reduction_method', 'unknown')}")
        print()
    except Exception as e:
        print(f"Optimization error: {e}")
        print()
    
    # Example 4: Similarity computation
    print("=== Example 4: Similarity Computation ===")
    
    # Generate embeddings for two sets of texts
    set1 = ["Machine learning", "Artificial intelligence", "Deep learning"]
    set2 = ["Neural networks", "Computer vision", "Natural language processing"]
    
    request1 = EmbeddingRequest(input=set1, model="nomic-embed-text", normalize=True)
    request2 = EmbeddingRequest(input=set2, model="nomic-embed-text", normalize=True)
    
    try:
        response1 = await embedding_manager.generate_embeddings_async(request1)
        response2 = await embedding_manager.generate_embeddings_async(request2)
        
        # Compute similarity
        optimizer = EmbeddingOptimizer(EmbeddingOptimizationConfig())
        similarity_matrix = optimizer.compute_similarity(
            response1, response2, SimilarityMetric.COSINE
        )
        
        print("Similarity matrix (cosine similarity):")
        print("Set1 vs Set2:")
        for i, text1 in enumerate(set1):
            for j, text2 in enumerate(set2):
                similarity = similarity_matrix[i, j]
                print(f"  '{text1}' vs '{text2}': {similarity:.3f}")
        print()
    except Exception as e:
        print(f"Similarity computation error: {e}")
        print()
    
    # Example 5: Provider statistics and health check
    print("=== Example 5: Provider Statistics and Health Check ===")
    
    # Get provider statistics
    stats = embedding_manager.get_provider_stats()
    for provider, stat in stats.items():
        print(f"{provider}:")
        print(f"  Total requests: {stat['total_requests']}")
        print(f"  Success rate: {stat['success_rate']:.1f}%")
        print(f"  Avg response time: {stat['average_response_time']:.2f}s")
        print(f"  Total tokens: {stat['total_tokens']}")
        print(f"  Total cost: ${stat['total_cost']:.6f}")
        print(f"  Healthy: {stat['is_healthy']}")
        print()
    
    # Health check
    try:
        health_results = await embedding_manager.health_check_async()
        print("Health Check Results:")
        for provider, result in health_results.items():
            print(f"{provider}: {result.get('status', 'unknown')}")
            if result.get('response_time'):
                print(f"  Response time: {result['response_time']:.2f}s")
            if result.get('error'):
                print(f"  Error: {result['error']}")
        print()
    except Exception as e:
        print(f"Health check error: {e}")
        print()
    
    # Example 6: Cache statistics
    print("=== Example 6: Cache Statistics ===")
    
    cache_stats = embedding_manager.get_cache_stats()
    print(f"Cache enabled: {cache_stats['enabled']}")
    print(f"Total entries: {cache_stats['total_entries']}")
    print(f"Cache hits: {cache_stats['cache_hits']}")
    print(f"Cache misses: {cache_stats['cache_misses']}")
    print(f"Hit rate: {cache_stats['hit_rate']:.1f}%")
    print(f"Memory usage: {cache_stats['memory_usage']} bytes")
    print()
    
    # Example 7: Testing different load balancing strategies
    print("=== Example 7: Load Balancing Strategies ===")
    
    strategies = [
        EmbeddingLoadBalancingStrategy.ROUND_ROBIN,
        EmbeddingLoadBalancingStrategy.RANDOM,
        EmbeddingLoadBalancingStrategy.LEAST_USED,
        EmbeddingLoadBalancingStrategy.FASTEST_RESPONSE,
        EmbeddingLoadBalancingStrategy.COST_OPTIMIZED
    ]
    
    test_request = EmbeddingRequest(
        input=["Test message for load balancing"],
        dimensions=512,
        normalize=True
    )
    
    for strategy in strategies:
        embedding_manager.set_load_balancing_strategy(strategy)
        
        try:
            response = await embedding_manager.generate_embeddings_async(test_request)
            print(f"{strategy.value}: Used {response.metadata.get('provider', 'unknown')}")
        except Exception as e:
            print(f"{strategy.value}: Error - {e}")
    
    print()
    
    # Example 8: Dimension analysis
    print("=== Example 8: Dimension Analysis ===")
    
    # Generate sample embeddings for analysis
    analysis_texts = [
        "Machine learning algorithms",
        "Deep neural networks",
        "Computer vision systems",
        "Natural language processing",
        "Artificial intelligence applications"
    ]
    
    analysis_request = EmbeddingRequest(
        input=analysis_texts,
        model="text-embedding-3-large",
        dimensions=1024
    )
    
    try:
        analysis_response = await embedding_manager.generate_embeddings_async(analysis_request)
        
        # Analyze dimensions
        optimizer = EmbeddingOptimizer(EmbeddingOptimizationConfig())
        dimension_analysis = optimizer.get_dimension_analysis(analysis_response.embeddings)
        
        print("Dimension Analysis:")
        print(f"  Number of embeddings: {dimension_analysis['num_embeddings']}")
        print(f"  Dimensions: {dimension_analysis['dimensions']}")
        print(f"  Total variance: {dimension_analysis['variance']['total']:.6f}")
        print(f"  Top 10 important dimensions: {dimension_analysis['variance']['top_10_dimensions']}")
        print(f"  Recommended dims (95% variance): {dimension_analysis['recommendations']['preserve_variance_dims']}")
        print(f"  Low variance dimensions: {dimension_analysis['recommendations']['low_variance_dims']}")
        print()
    except Exception as e:
        print(f"Dimension analysis error: {e}")
        print()
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())