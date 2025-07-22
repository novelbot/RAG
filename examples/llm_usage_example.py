"""
Example usage of the Multi-LLM Integration Layer.

This example demonstrates how to use the LLM Manager with multiple providers,
load balancing, and unified response handling.
"""

import asyncio
import os
from typing import List

from src.llm import (
    LLMManager, LLMProvider, LLMConfig, LLMRequest, LLMMessage, LLMRole,
    ProviderConfig, LoadBalancingStrategy
)


async def main():
    """Main example function demonstrating LLM Manager usage."""
    
    # Configure multiple providers
    provider_configs = [
        ProviderConfig(
            provider=LLMProvider.OPENAI,
            config=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4",
                temperature=0.7,
                max_tokens=1000,
                timeout=30.0,
                max_retries=3,
                enable_streaming=True,
                enable_function_calling=True
            ),
            priority=1,
            max_requests_per_minute=60,
            weight=1.0
        ),
        ProviderConfig(
            provider=LLMProvider.GEMINI,
            config=LLMConfig(
                provider=LLMProvider.GEMINI,
                api_key=os.getenv("GOOGLE_API_KEY"),
                model="gemini-2.0-flash-001",
                temperature=0.7,
                max_tokens=1000,
                timeout=30.0,
                max_retries=3,
                enable_streaming=True,
                enable_function_calling=True
            ),
            priority=2,
            max_requests_per_minute=60,
            weight=1.0
        ),
        ProviderConfig(
            provider=LLMProvider.CLAUDE,
            config=LLMConfig(
                provider=LLMProvider.CLAUDE,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-3-5-sonnet-latest",
                temperature=0.7,
                max_tokens=1000,
                timeout=30.0,
                max_retries=3,
                enable_streaming=True,
                enable_function_calling=True
            ),
            priority=3,
            max_requests_per_minute=60,
            weight=1.0
        ),
        ProviderConfig(
            provider=LLMProvider.OLLAMA,
            config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://localhost:11434",
                model="llama3.2",
                temperature=0.7,
                max_tokens=1000,
                timeout=30.0,
                max_retries=3,
                enable_streaming=True,
                enable_function_calling=False
            ),
            priority=4,
            max_requests_per_minute=120,  # Local model, higher rate limit
            weight=1.0
        )
    ]
    
    # Initialize LLM Manager
    llm_manager = LLMManager(provider_configs)
    
    # Set load balancing strategy
    llm_manager.set_load_balancing_strategy(LoadBalancingStrategy.HEALTH_BASED)
    
    print("=== LLM Manager Status ===")
    print(f"Manager Info: {llm_manager.get_manager_info()}")
    print()
    
    # Example 1: Simple chat completion
    print("=== Example 1: Simple Chat Completion ===")
    
    messages = [
        LLMMessage(role=LLMRole.USER, content="What is the capital of France?")
    ]
    
    request = LLMRequest(
        messages=messages,
        model="gpt-4",  # This will be used to select compatible providers
        temperature=0.7,
        max_tokens=100
    )
    
    try:
        response = await llm_manager.generate_async(request)
        print(f"Response: {response.content}")
        print(f"Provider: {response.metadata.get('provider', 'unknown')}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Response Time: {response.response_time:.2f}s")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 2: Streaming response
    print("=== Example 2: Streaming Response ===")
    
    messages = [
        LLMMessage(role=LLMRole.USER, content="Write a short story about a robot learning to paint.")
    ]
    
    request = LLMRequest(
        messages=messages,
        model="claude-3-5-sonnet-latest",
        temperature=0.8,
        max_tokens=500,
        stream=True
    )
    
    try:
        print("Streaming response: ", end="", flush=True)
        async for chunk in llm_manager.generate_stream_async(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            
            if chunk.finish_reason:
                print(f"\n\nStream finished. Reason: {chunk.finish_reason}")
                if chunk.usage:
                    print(f"Usage: {chunk.usage}")
                print(f"Provider: {chunk.metadata.get('provider', 'unknown')}")
                print()
        
    except Exception as e:
        print(f"Streaming error: {e}")
        print()
    
    # Example 3: Multi-turn conversation
    print("=== Example 3: Multi-turn Conversation ===")
    
    conversation = [
        LLMMessage(role=LLMRole.USER, content="Hello, I'm learning about machine learning."),
        LLMMessage(role=LLMRole.ASSISTANT, content="Hello! I'd be happy to help you learn about machine learning. What specific topic would you like to explore?"),
        LLMMessage(role=LLMRole.USER, content="Can you explain what neural networks are?")
    ]
    
    request = LLMRequest(
        messages=conversation,
        model="gemini-2.0-flash-001",
        temperature=0.6,
        max_tokens=200
    )
    
    try:
        response = await llm_manager.generate_async(request)
        print(f"Response: {response.content}")
        print(f"Provider: {response.metadata.get('provider', 'unknown')}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 4: Token counting
    print("=== Example 4: Token Counting ===")
    
    messages = [
        LLMMessage(role=LLMRole.USER, content="This is a test message for token counting.")
    ]
    
    try:
        token_count = await llm_manager.count_tokens_async(messages, "gpt-4")
        print(f"Token count: {token_count}")
        print()
    except Exception as e:
        print(f"Token counting error: {e}")
        print()
    
    # Example 5: Provider statistics
    print("=== Example 5: Provider Statistics ===")
    
    stats = llm_manager.get_provider_stats()
    for provider, stat in stats.items():
        print(f"{provider}:")
        print(f"  Total requests: {stat['total_requests']}")
        print(f"  Success rate: {stat['success_rate']:.1f}%")
        print(f"  Avg response time: {stat['average_response_time']:.2f}s")
        print(f"  Healthy: {stat['is_healthy']}")
        print()
    
    # Example 6: Available models
    print("=== Example 6: Available Models ===")
    
    try:
        models = llm_manager.get_available_models()
        for provider, model_list in models.items():
            print(f"{provider}: {len(model_list)} models available")
            if model_list:
                print(f"  Examples: {model_list[:3]}")
        print()
    except Exception as e:
        print(f"Error getting models: {e}")
        print()
    
    # Example 7: Health check
    print("=== Example 7: Health Check ===")
    
    try:
        health_results = await llm_manager.health_check_async()
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
    
    # Example 8: Testing different load balancing strategies
    print("=== Example 8: Load Balancing Strategies ===")
    
    strategies = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.RANDOM,
        LoadBalancingStrategy.LEAST_USED,
        LoadBalancingStrategy.FASTEST_RESPONSE,
        LoadBalancingStrategy.HEALTH_BASED
    ]
    
    for strategy in strategies:
        llm_manager.set_load_balancing_strategy(strategy)
        
        request = LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="Test message")],
            model="gpt-4",
            temperature=0.5,
            max_tokens=50
        )
        
        try:
            response = await llm_manager.generate_async(request)
            print(f"{strategy.value}: Used {response.metadata.get('provider', 'unknown')}")
        except Exception as e:
            print(f"{strategy.value}: Error - {e}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())