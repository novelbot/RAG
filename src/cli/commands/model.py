"""
Model testing and configuration commands for the CLI.
"""

import click
from rich.console import Console
from rich.table import Table

from ..utils import (
    console, confirm_action, create_progress_bar, display_table
)

from src.core.config import get_config
from src.llm.manager import LLMManager
from src.embedding.manager import EmbeddingManager
from src.llm.base import LLMRequest, LLMMessage, LLMRole
from src.embedding.base import EmbeddingRequest
from src.embedding.types import EmbeddingProvider, EmbeddingConfig
from src.embedding.manager import EmbeddingProviderConfig
from src.core.exceptions import LLMError, EmbeddingError
import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

console = Console()


@dataclass
class BenchmarkResult:
    """Benchmark result for a single model"""
    model_name: str
    provider: str
    model_type: str  # 'llm' or 'embedding'
    response_times: List[float]
    success_count: int
    error_count: int
    total_tokens: Optional[int] = None
    cost_estimate: Optional[float] = None
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def min_response_time(self) -> float:
        return min(self.response_times) if self.response_times else 0.0
    
    @property
    def max_response_time(self) -> float:
        return max(self.response_times) if self.response_times else 0.0
    
    @property
    def std_response_time(self) -> float:
        return statistics.stdev(self.response_times) if len(self.response_times) > 1 else 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0
    
    @property
    def throughput_per_min(self) -> float:
        if not self.response_times:
            return 0.0
        return 60.0 / self.avg_response_time if self.avg_response_time > 0 else 0.0


def prepare_benchmark_data() -> Dict[str, List[str]]:
    """Prepare test data for benchmarking"""
    return {
        'short_texts': [
            "Hello world",
            "What is AI?",
            "Python programming",
            "Machine learning basics",
            "Data science"
        ],
        'medium_texts': [
            "Explain the concept of artificial intelligence and its applications in modern technology.",
            "What are the key differences between supervised and unsupervised machine learning?",
            "Describe the process of natural language processing and its importance in AI systems.",
            "How do neural networks work and what makes them effective for pattern recognition?",
            "What are the ethical considerations in AI development and deployment?"
        ],
        'long_texts': [
            """Artificial intelligence has revolutionized numerous industries and continues to transform how we approach complex problems. From healthcare diagnostics to autonomous vehicles, AI systems are becoming increasingly sophisticated and capable. The development of large language models has particularly advanced natural language understanding and generation capabilities, enabling more intuitive human-computer interactions. However, these advancements also raise important questions about ethics, bias, and the responsible deployment of AI technologies in society.""",
            """The field of machine learning encompasses various approaches to pattern recognition and predictive modeling. Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Unsupervised learning methods discover hidden patterns in data without explicit labels. Reinforcement learning agents learn optimal behaviors through trial and error interactions with their environment. Each approach has unique strengths and is suited to different types of problems and data characteristics."""
        ],
        'llm_prompts': [
            "Summarize the benefits of renewable energy",
            "Explain quantum computing in simple terms",
            "Write a brief analysis of climate change impacts",
            "Describe the history of artificial intelligence",
            "Compare different programming paradigms"
        ]
    }


async def benchmark_embedding_model(config: EmbeddingProviderConfig, test_texts: List[str], 
                                  iterations: int) -> BenchmarkResult:
    """Benchmark a single embedding model"""
    response_times = []
    success_count = 0
    error_count = 0
    
    try:
        # Initialize embedding manager
        embedding_manager = EmbeddingManager([config])
        
        for i in range(iterations):
            # Select test text (cycle through available texts)
            text = test_texts[i % len(test_texts)]
            
            try:
                # Create embedding request
                request = EmbeddingRequest(
                    texts=[text],
                    model=config.model
                )
                
                # Measure response time
                start_time = time.time()
                response = await embedding_manager.generate_embeddings_async(request)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response.embeddings and len(response.embeddings) > 0:
                    response_times.append(response_time)
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                console.print(f"[dim red]Error in iteration {i+1}: {str(e)[:50]}...[/dim red]")
                
    except Exception as e:
        console.print(f"[red]Failed to initialize embedding model {config.model}: {e}[/red]")
        error_count = iterations
    
    return BenchmarkResult(
        model_name=config.model,
        provider=config.provider.value,
        model_type='embedding',
        response_times=response_times,
        success_count=success_count,
        error_count=error_count
    )


async def benchmark_llm_model(provider_name: str, model_name: str, test_prompts: List[str],
                            iterations: int) -> BenchmarkResult:
    """Benchmark a single LLM model"""
    response_times = []
    success_count = 0
    error_count = 0
    total_tokens = 0
    
    try:
        # Get config and initialize LLM manager
        config = get_config()
        llm_manager = LLMManager(config.llm)
        
        for i in range(iterations):
            # Select test prompt (cycle through available prompts)
            prompt = test_prompts[i % len(test_prompts)]
            
            try:
                # Create LLM request
                request = LLMRequest(
                    messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
                    model=model_name,
                    max_tokens=150  # Limit tokens for faster benchmarking
                )
                
                # Measure response time
                start_time = time.time()
                response = await llm_manager.generate_response_async(request)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response.content:
                    response_times.append(response_time)
                    success_count += 1
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens += getattr(response.usage, 'total_tokens', 0)
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                console.print(f"[dim red]Error in iteration {i+1}: {str(e)[:50]}...[/dim red]")
                
    except Exception as e:
        console.print(f"[red]Failed to initialize LLM model {model_name}: {e}[/red]")
        error_count = iterations
    
    return BenchmarkResult(
        model_name=model_name,
        provider=provider_name,
        model_type='llm',
        response_times=response_times,
        success_count=success_count,
        error_count=error_count,
        total_tokens=total_tokens if total_tokens > 0 else None
    )


async def run_concurrent_benchmark(benchmark_func, *args, concurrent: int) -> List[BenchmarkResult]:
    """Run benchmark function concurrently"""
    tasks = []
    for _ in range(concurrent):
        task = asyncio.create_task(benchmark_func(*args))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    valid_results = []
    for result in results:
        if isinstance(result, BenchmarkResult):
            valid_results.append(result)
        elif isinstance(result, Exception):
            console.print(f"[red]Concurrent benchmark error: {result}[/red]")
    
    return valid_results


def display_benchmark_results(results: Dict[str, BenchmarkResult]):
    """Display benchmark results in a formatted table"""
    if not results:
        console.print("[yellow]No benchmark results to display[/yellow]")
        return
    
    # Separate LLM and embedding results
    llm_results = {k: v for k, v in results.items() if v.model_type == 'llm'}
    embedding_results = {k: v for k, v in results.items() if v.model_type == 'embedding'}
    
    # Display LLM results
    if llm_results:
        llm_table_data = []
        for model_name, result in llm_results.items():
            llm_table_data.append({
                'Model': f"{result.provider}/{model_name}",
                'Avg Time': f"{result.avg_response_time:.2f}s",
                'Min Time': f"{result.min_response_time:.2f}s",
                'Max Time': f"{result.max_response_time:.2f}s",
                'Std Dev': f"{result.std_response_time:.2f}s",
                'Success Rate': f"{result.success_rate:.1f}%",
                'Throughput': f"{result.throughput_per_min:.1f}/min",
                'Total Tokens': str(result.total_tokens) if result.total_tokens else 'N/A'
            })
        
        display_table(llm_table_data, title="LLM Benchmark Results")
    
    # Display embedding results
    if embedding_results:
        embedding_table_data = []
        for model_name, result in embedding_results.items():
            embedding_table_data.append({
                'Model': f"{result.provider}/{model_name}",
                'Avg Time': f"{result.avg_response_time:.2f}s",
                'Min Time': f"{result.min_response_time:.2f}s",
                'Max Time': f"{result.max_response_time:.2f}s",
                'Std Dev': f"{result.std_response_time:.2f}s",
                'Success Rate': f"{result.success_rate:.1f}%",
                'Throughput': f"{result.throughput_per_min:.1f}/min"
            })
        
        display_table(embedding_table_data, title="Embedding Benchmark Results")


@click.group()
def model_group():
    """Model testing and configuration commands.
    
    Commands for testing LLM providers, embedding models,
    and managing model configurations.
    """
    pass


@model_group.command(name='test-llm')
@click.option('--provider', type=click.Choice(['openai', 'gemini', 'claude', 'ollama']),
              help='Specific LLM provider to test (tests all if not specified).')
@click.option('--model', help='Specific model to test.')
@click.option('--prompt', default='Hello, how are you?',
              help='Test prompt to send to the model.')
def test_llm(provider, model, prompt):
    """Test LLM provider connectivity and functionality.
    
    Tests the configured LLM providers to ensure they are
    working correctly and can generate responses.
    
    Examples:
        rag-cli model test-llm
        rag-cli model test-llm --provider openai
        rag-cli model test-llm --provider openai --model gpt-4
        rag-cli model test-llm --prompt "Explain quantum computing"
    """
    console.print(f"[yellow]Testing LLM providers with prompt: '{prompt[:50]}...'[/yellow]")
    
    try:
        # Get configuration
        config = get_config()
        
        # Initialize LLM manager
        from src.llm.manager import ProviderConfig
        from src.llm.base import LLMProvider
        
        # Convert config.llm to ProviderConfig format
        provider_configs = []
        
        # Map string provider names to enum values
        provider_mapping = {
            'openai': LLMProvider.OPENAI,
            'gemini': LLMProvider.GEMINI,
            'claude': LLMProvider.CLAUDE,
            'ollama': LLMProvider.OLLAMA
        }
        
        # Create provider config for the configured provider
        if config.llm.provider in provider_mapping:
            provider_config = ProviderConfig(
                provider=provider_mapping[config.llm.provider],
                config=config.llm,
                enabled=True
            )
            provider_configs.append(provider_config)
        
        llm_manager = LLMManager(provider_configs)
        
        providers_to_test = [provider] if provider else ['openai', 'gemini', 'claude', 'ollama']
        results = []
        
        for prov in providers_to_test:
            console.print(f"\n[cyan]Testing {prov}...[/cyan]")
            
            try:
                # Check if provider is configured
                provider_enum = provider_mapping.get(prov)
                if not provider_enum or provider_enum not in llm_manager.providers:
                    console.print(f"[yellow]⚠ {prov} not configured[/yellow]")
                    results.append({
                        'provider': prov,
                        'model': 'Not configured',
                        'status': 'Not configured',
                        'response_time': 'N/A',
                        'tokens': 'N/A',
                        'response_preview': 'N/A'
                    })
                    continue
                
                with create_progress_bar() as progress:
                    task = progress.add_task(f"Testing {prov}...", total=None)
                    
                    # Create test request
                    start_time = time.time()
                    
                    try:
                        # Prepare test message
                        test_request = LLMRequest(
                            messages=[
                                LLMMessage(role=LLMRole.USER, content=prompt)
                            ],
                            model=model or config.llm.model,
                            max_tokens=100,
                            temperature=0.7
                        )
                        
                        # Test the provider directly
                        provider_instance = llm_manager.providers[provider_enum]
                        if hasattr(provider_instance, 'generate_async'):
                            response = asyncio.run(provider_instance.generate_async(test_request))
                        else:
                            response = provider_instance.generate(test_request)
                        
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # Extract response details
                        response_content = response.content if hasattr(response, 'content') else str(response)
                        response_preview = response_content[:100] + "..." if len(response_content) > 100 else response_content
                        
                        # Get token usage if available
                        tokens_used = "N/A"
                        if hasattr(response, 'usage') and response.usage:
                            tokens_used = str(response.usage.total_tokens)
                        
                        results.append({
                            'provider': prov,
                            'model': model or config.llm.model,
                            'status': 'Success',
                            'response_time': f"{response_time:.2f}s",
                            'tokens': tokens_used,
                            'response_preview': response_preview
                        })
                        
                        console.print(f"[green]✓ {prov} test passed[/green]")
                        console.print(f"[dim]Response: {response_preview}[/dim]")
                        
                    except Exception as e:
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        results.append({
                            'provider': prov,
                            'model': model or config.llm.model,
                            'status': f'Failed: {str(e)[:50]}',
                            'response_time': f"{response_time:.2f}s",
                            'tokens': 'N/A',
                            'response_preview': 'Error'
                        })
                        
                        console.print(f"[red]✗ {prov} test failed: {e}[/red]")
                    
                    progress.remove_task(task)
                    
            except Exception as e:
                console.print(f"[red]✗ Error testing {prov}: {e}[/red]")
                results.append({
                    'provider': prov,
                    'model': 'Error',
                    'status': f'Error: {str(e)[:50]}',
                    'response_time': 'N/A',
                    'tokens': 'N/A',
                    'response_preview': 'Error'
                })
        
        # Display results table
        console.print("\n")
        display_columns = ['provider', 'model', 'status', 'response_time', 'tokens']
        display_table(results, title="LLM Test Results", columns=display_columns)
        
        # Show detailed responses if any succeeded
        successful_tests = [r for r in results if r['status'] == 'Success']
        if successful_tests:
            console.print("\n[cyan]Response Examples:[/cyan]")
            for result in successful_tests:
                console.print(f"[dim]{result['provider']}: {result['response_preview']}[/dim]")
        
        console.print(f"\n[dim]LLM testing completed: {len(successful_tests)}/{len(results)} providers successful[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Error during LLM testing: {e}[/red]")


@model_group.command(name='test-embedding')
@click.option('--provider', type=click.Choice(['openai', 'sentence-transformers', 'huggingface']),
              help='Specific embedding provider to test.')
@click.option('--model', help='Specific embedding model to test.')
@click.option('--text', default='This is a test sentence for embedding generation.',
              help='Test text to generate embeddings for.')
def test_embedding(provider, model, text):
    """Test embedding model connectivity and functionality.
    
    Tests the configured embedding providers to ensure they
    can generate embeddings correctly.
    
    Examples:
        rag-cli model test-embedding
        rag-cli model test-embedding --provider openai
        rag-cli model test-embedding --text "Custom test text"
    """
    # Testing embedding models
    asyncio.run(_test_embedding_async(provider, model, text))


async def _test_embedding_async(provider, model, text):
    """Async implementation of embedding testing."""
    config = get_config()
    
    providers_to_test = [provider] if provider else ['openai', 'sentence-transformers', 'huggingface']
    
    console.print(f"[yellow]Testing embedding providers with text: '{text[:50]}...'[/yellow]")
    
    results = []
    
    for prov in providers_to_test:
        console.print(f"\n[cyan]Testing {prov}...[/cyan]")
        
        with create_progress_bar() as progress:
            task = progress.add_task(f"Testing {prov}...", total=None)
            
            # Implement actual embedding testing
            try:
                # Create embedding manager for this provider
                
                # Configure provider based on type
                if prov == 'openai':
                    if not config.embedding.openai_api_key:
                        results.append({
                            'provider': prov,
                            'model': 'Not configured',
                            'status': 'No API key',
                            'dimensions': 'N/A',
                            'generation_time': 'N/A'
                        })
                        console.print(f"[yellow]⚠ {prov} not configured (missing API key)[/yellow]")
                        continue
                        
                    provider_config = EmbeddingProviderConfig(
                        provider=EmbeddingProvider.OPENAI,
                        model=model or config.embedding.openai_model,
                        api_key=config.embedding.openai_api_key,
                        enabled=True
                    )
                elif prov == 'sentence-transformers':
                    provider_config = EmbeddingProviderConfig(
                        provider=EmbeddingProvider.HUGGINGFACE,  # Using HUGGINGFACE for sentence-transformers
                        model=model or config.embedding.sentence_transformers_model,
                        enabled=True
                    )
                elif prov == 'huggingface':
                    provider_config = EmbeddingProviderConfig(
                        provider=EmbeddingProvider.HUGGINGFACE,
                        model=model or config.embedding.huggingface_model,
                        api_key=getattr(config.embedding, 'huggingface_api_key', None),
                        enabled=True
                    )
                else:
                    results.append({
                        'provider': prov,
                        'model': 'Unknown provider',
                        'status': 'Not supported',
                        'dimensions': 'N/A',
                        'generation_time': 'N/A'
                    })
                    console.print(f"[red]✗ {prov} not supported[/red]")
                    continue
                
                # Initialize embedding manager with single provider
                embedding_manager = EmbeddingManager([provider_config])
                
                # Create embedding request
                request = EmbeddingRequest(
                    texts=[text],
                    model=provider_config.model
                )
                
                # Test embedding generation
                start_time = time.time()
                response = await embedding_manager.generate_embeddings_async(request)
                generation_time = time.time() - start_time
                
                # Check if embeddings were generated successfully  
                if response.embeddings and len(response.embeddings) > 0:
                    results.append({
                        'provider': prov,
                        'model': provider_config.model,
                        'status': 'Success',
                        'dimensions': str(len(response.embeddings[0])),
                        'generation_time': f"{generation_time:.2f}s"
                    })
                    console.print(f"[green]✓ {prov} test passed[/green]")
                else:
                    results.append({
                        'provider': prov,
                        'model': provider_config.model,
                        'status': 'Failed - No embeddings',
                        'dimensions': 'N/A',
                        'generation_time': f"{generation_time:.2f}s"
                    })
                    console.print(f"[red]✗ {prov} test failed (no embeddings generated)[/red]")
                    
            except Exception as e:
                results.append({
                    'provider': prov,
                    'model': provider_config.model if 'provider_config' in locals() else 'Unknown',
                    'status': f'Error: {str(e)[:50]}...',
                    'dimensions': 'N/A',
                    'generation_time': 'N/A'
                })
                console.print(f"[red]✗ {prov} test failed: {str(e)[:50]}...[/red]")
            
            progress.remove_task(task)
    
    # Display results table
    console.print("\n")
    display_table(results, title="Embedding Test Results")
    
    # Embedding testing completed


@model_group.command(name='benchmark')
@click.option('--provider', help='Specific provider to benchmark.')
@click.option('--iterations', default=10, type=int,
              help='Number of test iterations to run.')
@click.option('--concurrent', default=1, type=int,
              help='Number of concurrent requests.')
def benchmark_models(provider, iterations, concurrent):
    """Benchmark model performance.
    
    Runs performance benchmarks on configured models
    including response time, throughput, and quality metrics.
    
    Examples:
        rag-cli model benchmark
        rag-cli model benchmark --provider openai --iterations 20
        rag-cli model benchmark --concurrent 5
    """
    asyncio.run(_benchmark_models_async(provider, iterations, concurrent))


async def _benchmark_models_async(provider: Optional[str], iterations: int, concurrent: int):
    """Async implementation of model benchmarking"""
    console.print(f"[yellow]Running performance benchmark...[/yellow]")
    console.print(f"[dim]Iterations: {iterations}, Concurrent requests: {concurrent}[/dim]")
    
    try:
        # Prepare test data
        test_data = prepare_benchmark_data()
        
        # Get configuration
        config = get_config()
        
        # Collect models to benchmark
        models_to_test = []
        benchmark_results = {}
        
        # Test embedding models
        if hasattr(config, 'embedding') and config.embedding:
            try:
                embedding_providers = ['openai', 'google', 'ollama']  # Add more as needed
                
                for prov in embedding_providers:
                    if provider and prov != provider:
                        continue
                        
                    try:
                        provider_config = EmbeddingProviderConfig(
                            provider=EmbeddingProvider(prov),
                            model="text-embedding-ada-002" if prov == "openai" else "embedding-001",
                            api_key=getattr(config.embedding, f"{prov}_api_key", None) or "",
                            base_url=getattr(config.embedding, f"{prov}_base_url", None)
                        )
                        
                        if provider_config.api_key or prov == "ollama":
                            models_to_test.append(('embedding', prov, provider_config))
                            
                    except Exception as e:
                        console.print(f"[dim yellow]Skipping {prov} embedding: {e}[/dim yellow]")
                        
            except Exception as e:
                console.print(f"[yellow]Could not load embedding config: {e}[/yellow]")
        
        # Test LLM models
        if hasattr(config, 'llm') and config.llm:
            try:
                llm_providers = ['openai', 'anthropic', 'google', 'ollama']
                
                for prov in llm_providers:
                    if provider and prov != provider:
                        continue
                        
                    try:
                        # Get provider-specific model names
                        model_name = {
                            'openai': 'gpt-3.5-turbo',
                            'anthropic': 'claude-3-haiku-20240307',
                            'google': 'gemini-pro',
                            'ollama': 'llama2'
                        }.get(prov, 'default-model')
                        
                        # Check if provider is configured
                        api_key = getattr(config.llm, f"{prov}_api_key", None)
                        if api_key or prov == "ollama":
                            models_to_test.append(('llm', prov, model_name))
                            
                    except Exception as e:
                        console.print(f"[dim yellow]Skipping {prov} LLM: {e}[/dim yellow]")
                        
            except Exception as e:
                console.print(f"[yellow]Could not load LLM config: {e}[/yellow]")
        
        if not models_to_test:
            console.print("[red]No models available for benchmarking. Please check your configuration.[/red]")
            return
        
        console.print(f"[green]Found {len(models_to_test)} models to benchmark[/green]")
        
        # Run benchmarks with progress tracking
        with create_progress_bar() as progress:
            main_task = progress.add_task("Overall Progress", total=len(models_to_test))
            
            for model_type, prov, model_config in models_to_test:
                model_task = progress.add_task(f"Benchmarking {prov}", total=iterations)
                
                try:
                    if model_type == 'embedding':
                        # Test embedding models
                        all_texts = test_data['short_texts'] + test_data['medium_texts']
                        
                        if concurrent > 1:
                            # Run concurrent benchmarks
                            per_thread_iterations = max(1, iterations // concurrent)
                            concurrent_results = await run_concurrent_benchmark(
                                benchmark_embedding_model, 
                                model_config, 
                                all_texts, 
                                per_thread_iterations, 
                                concurrent=concurrent
                            )
                            
                            # Combine results
                            if concurrent_results:
                                combined_times = []
                                total_success = 0
                                total_errors = 0
                                
                                for result in concurrent_results:
                                    combined_times.extend(result.response_times)
                                    total_success += result.success_count
                                    total_errors += result.error_count
                                
                                benchmark_results[f"{prov}_embedding"] = BenchmarkResult(
                                    model_name=model_config.model,
                                    provider=prov,
                                    model_type='embedding',
                                    response_times=combined_times,
                                    success_count=total_success,
                                    error_count=total_errors
                                )
                        else:
                            # Single-threaded benchmark
                            result = await benchmark_embedding_model(model_config, all_texts, iterations)
                            benchmark_results[f"{prov}_embedding"] = result
                            
                        # Update progress as we go
                        for i in range(iterations):
                            progress.update(model_task, advance=1)
                            await asyncio.sleep(0.01)  # Allow UI updates
                            
                    elif model_type == 'llm':
                        # Test LLM models
                        test_prompts = test_data['llm_prompts']
                        
                        if concurrent > 1:
                            # Run concurrent benchmarks
                            per_thread_iterations = max(1, iterations // concurrent)
                            concurrent_results = await run_concurrent_benchmark(
                                benchmark_llm_model,
                                prov,
                                model_config,
                                test_prompts,
                                per_thread_iterations,
                                concurrent=concurrent
                            )
                            
                            # Combine results
                            if concurrent_results:
                                combined_times = []
                                total_success = 0
                                total_errors = 0
                                total_tokens = 0
                                
                                for result in concurrent_results:
                                    combined_times.extend(result.response_times)
                                    total_success += result.success_count
                                    total_errors += result.error_count
                                    if result.total_tokens:
                                        total_tokens += result.total_tokens
                                
                                benchmark_results[f"{prov}_llm"] = BenchmarkResult(
                                    model_name=model_config,
                                    provider=prov,
                                    model_type='llm',
                                    response_times=combined_times,
                                    success_count=total_success,
                                    error_count=total_errors,
                                    total_tokens=total_tokens if total_tokens > 0 else None
                                )
                        else:
                            # Single-threaded benchmark
                            result = await benchmark_llm_model(prov, model_config, test_prompts, iterations)
                            benchmark_results[f"{prov}_llm"] = result
                            
                        # Update progress
                        for i in range(iterations):
                            progress.update(model_task, advance=1)
                            await asyncio.sleep(0.01)
                            
                    progress.update(main_task, advance=1)
                    progress.remove_task(model_task)
                    
                except KeyboardInterrupt:
                    console.print("[yellow]Benchmark interrupted by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error benchmarking {prov}: {e}[/red]")
                    progress.update(main_task, advance=1)
                    continue
        
        # Display results
        console.print(f"\n[green]Benchmark completed![/green]")
        console.print(f"[dim]Total models tested: {len(benchmark_results)}[/dim]")
        
        if benchmark_results:
            display_benchmark_results(benchmark_results)
            
            # Save results to file
            try:
                results_file = Path("benchmark_results.json")
                with open(results_file, 'w') as f:
                    # Convert results to JSON-serializable format
                    json_data = {}
                    for key, result in benchmark_results.items():
                        json_data[key] = {
                            'model_name': result.model_name,
                            'provider': result.provider,
                            'model_type': result.model_type,
                            'avg_response_time': result.avg_response_time,
                            'min_response_time': result.min_response_time,
                            'max_response_time': result.max_response_time,
                            'std_response_time': result.std_response_time,
                            'success_rate': result.success_rate,
                            'throughput_per_min': result.throughput_per_min,
                            'total_tokens': result.total_tokens,
                            'success_count': result.success_count,
                            'error_count': result.error_count
                        }
                    
                    json.dump(json_data, f, indent=2)
                    
                console.print(f"[dim]Results saved to {results_file}[/dim]")
                    
            except Exception as e:
                console.print(f"[yellow]Could not save results: {e}[/yellow]")
        else:
            console.print("[yellow]No successful benchmark results to display[/yellow]")
            
    except KeyboardInterrupt:
        console.print("[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        import traceback
        console.print(f"[dim red]{traceback.format_exc()}[/dim red]")


@model_group.command(name='list-models')
@click.option('--provider', help='Filter by provider.')
@click.option('--type', type=click.Choice(['llm', 'embedding', 'all']),
              default='all', help='Type of models to list.')
def list_models(provider, type):
    """List available models.
    
    Shows all available models for the configured providers
    including their capabilities and status.
    
    Examples:
        rag-cli model list-models
        rag-cli model list-models --provider openai
        rag-cli model list-models --type llm
    """
    asyncio.run(_list_models_async(provider, type))


async def _list_models_async(provider: Optional[str], model_type: str):
    """Async implementation of model listing"""
    console.print("[yellow]Fetching available models...[/yellow]")
    
    try:
        config = get_config()
        all_models = []
        
        # Fetch OpenAI models
        if not provider or provider == 'openai':
            try:
                openai_models = await _fetch_openai_models(config, model_type)
                all_models.extend(openai_models)
            except Exception as e:
                console.print(f"[yellow]Could not fetch OpenAI models: {e}[/yellow]")
        
        # Fetch Anthropic models
        if not provider or provider == 'anthropic':
            try:
                anthropic_models = await _fetch_anthropic_models(config, model_type)
                all_models.extend(anthropic_models)
            except Exception as e:
                console.print(f"[yellow]Could not fetch Anthropic models: {e}[/yellow]")
        
        # Fetch Google models (limited API access)
        if not provider or provider == 'google':
            try:
                google_models = _get_google_models(model_type)
                all_models.extend(google_models)
            except Exception as e:
                console.print(f"[yellow]Could not fetch Google models: {e}[/yellow]")
        
        # Fetch Ollama models (if available)
        if not provider or provider == 'ollama':
            try:
                ollama_models = await _fetch_ollama_models(model_type)
                all_models.extend(ollama_models)
            except Exception as e:
                console.print(f"[yellow]Could not fetch Ollama models: {e}[/yellow]")
        
        # Add known embedding models
        if model_type in ['all', 'embedding']:
            embedding_models = _get_embedding_models(provider)
            all_models.extend(embedding_models)
        
        # Filter and display results
        if all_models:
            display_table(all_models, title="Available Models")
            
            # Save results to file
            try:
                models_file = Path("available_models.json")
                with open(models_file, 'w') as f:
                    json.dump(all_models, f, indent=2)
                console.print(f"[dim]Models list saved to {models_file}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not save models list: {e}[/yellow]")
        else:
            console.print("[yellow]No models found matching criteria[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error fetching models: {e}[/red]")


async def _fetch_openai_models(config, model_type: str) -> List[Dict[str, Any]]:
    """Fetch available models from OpenAI"""
    models = []
    
    try:
        # Check if OpenAI is configured
        if not hasattr(config, 'llm') or not hasattr(config.llm, 'openai_api_key'):
            return models
            
        api_key = getattr(config.llm, 'openai_api_key', None)
        if not api_key:
            return models
        
        # Import OpenAI client
        try:
            from openai import OpenAI
        except ImportError:
            console.print("[yellow]OpenAI library not installed[/yellow]")
            return models
        
        client = OpenAI(api_key=api_key)
        
        # Fetch models list
        models_response = client.models.list()
        
        for model in models_response.data:
            model_id = model.id
            
            # Categorize models
            if model_type in ['all', 'llm']:
                if any(keyword in model_id.lower() for keyword in ['gpt', 'claude', 'text-davinci', 'text-curie']):
                    if 'embedding' not in model_id.lower():
                        models.append({
                            'provider': 'openai',
                            'type': 'llm',
                            'model': model_id,
                            'status': 'available',
                            'capabilities': _get_model_capabilities(model_id, 'llm'),
                            'created': getattr(model, 'created', None),
                            'owned_by': getattr(model, 'owned_by', 'openai')
                        })
            
            if model_type in ['all', 'embedding']:
                if 'embedding' in model_id.lower():
                    models.append({
                        'provider': 'openai',
                        'type': 'embedding',
                        'model': model_id,
                        'status': 'available',
                        'capabilities': _get_model_capabilities(model_id, 'embedding'),
                        'created': getattr(model, 'created', None),
                        'owned_by': getattr(model, 'owned_by', 'openai')
                    })
        
    except Exception as e:
        console.print(f"[dim red]OpenAI API error: {str(e)[:100]}...[/dim red]")
    
    return models


async def _fetch_anthropic_models(config, model_type: str) -> List[Dict[str, Any]]:
    """Fetch available models from Anthropic"""
    models = []
    
    try:
        # Check if Anthropic is configured
        if not hasattr(config, 'llm') or not hasattr(config.llm, 'anthropic_api_key'):
            return models
            
        api_key = getattr(config.llm, 'anthropic_api_key', None)
        if not api_key:
            return models
        
        # Import Anthropic client
        try:
            from anthropic import Anthropic
        except ImportError:
            console.print("[yellow]Anthropic library not installed[/yellow]")
            return models
        
        client = Anthropic(api_key=api_key)
        
        # Fetch models list
        try:
            models_response = client.models.list()
            
            for model in models_response.data:
                if model_type in ['all', 'llm']:
                    models.append({
                        'provider': 'anthropic',
                        'type': 'llm',
                        'model': model.id,
                        'status': 'available',
                        'capabilities': _get_model_capabilities(model.id, 'llm'),
                        'display_name': getattr(model, 'display_name', model.id),
                        'created_at': getattr(model, 'created_at', None)
                    })
        except Exception as e:
            # If models.list() fails, provide known Claude models
            if model_type in ['all', 'llm']:
                known_claude_models = [
                    'claude-3-5-sonnet-20241022',
                    'claude-3-5-haiku-20241022',
                    'claude-3-opus-20240229',
                    'claude-3-sonnet-20240229',
                    'claude-3-haiku-20240307'
                ]
                
                for model_id in known_claude_models:
                    models.append({
                        'provider': 'anthropic',
                        'type': 'llm',
                        'model': model_id,
                        'status': 'available*',
                        'capabilities': _get_model_capabilities(model_id, 'llm'),
                        'note': 'Known model (API list unavailable)'
                    })
        
    except Exception as e:
        console.print(f"[dim red]Anthropic API error: {str(e)[:100]}...[/dim red]")
    
    return models


def _get_google_models(model_type: str) -> List[Dict[str, Any]]:
    """Get known Google models (limited API access)"""
    models = []
    
    if model_type in ['all', 'llm']:
        google_llm_models = [
            'gemini-pro',
            'gemini-pro-vision',
            'gemini-1.5-pro',
            'gemini-1.5-flash'
        ]
        
        for model_id in google_llm_models:
            models.append({
                'provider': 'google',
                'type': 'llm',
                'model': model_id,
                'status': 'available*',
                'capabilities': _get_model_capabilities(model_id, 'llm'),
                'note': 'Known model (requires API setup)'
            })
    
    if model_type in ['all', 'embedding']:
        google_embedding_models = [
            'embedding-001',
            'text-embedding-004'
        ]
        
        for model_id in google_embedding_models:
            models.append({
                'provider': 'google',
                'type': 'embedding',
                'model': model_id,
                'status': 'available*',
                'capabilities': _get_model_capabilities(model_id, 'embedding'),
                'note': 'Known model (requires API setup)'
            })
    
    return models


async def _fetch_ollama_models(model_type: str) -> List[Dict[str, Any]]:
    """Fetch available models from Ollama"""
    models = []
    
    try:
        import httpx
        
        # Try to connect to Ollama API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    
                    for model in data.get('models', []):
                        model_name = model.get('name', '').split(':')[0]  # Remove tag
                        
                        if model_type in ['all', 'llm']:
                            models.append({
                                'provider': 'ollama',
                                'type': 'llm',
                                'model': model_name,
                                'status': 'local',
                                'capabilities': _get_model_capabilities(model_name, 'llm'),
                                'size': model.get('size', 0),
                                'modified_at': model.get('modified_at', None)
                            })
                        
                        # Some Ollama models can be used for embeddings
                        if model_type in ['all', 'embedding']:
                            if any(embed_keyword in model_name.lower() for embed_keyword in ['embed', 'nomic']):
                                models.append({
                                    'provider': 'ollama',
                                    'type': 'embedding',
                                    'model': model_name,
                                    'status': 'local',
                                    'capabilities': _get_model_capabilities(model_name, 'embedding'),
                                    'size': model.get('size', 0),
                                    'modified_at': model.get('modified_at', None)
                                })
                            
            except httpx.TimeoutException:
                console.print("[dim yellow]Ollama not running (timeout)[/dim yellow]")
            except httpx.ConnectError:
                console.print("[dim yellow]Ollama not running (connection refused)[/dim yellow]")
                
    except ImportError:
        console.print("[yellow]httpx library not available for Ollama check[/yellow]")
    
    return models


def _get_embedding_models(provider_filter: Optional[str]) -> List[Dict[str, Any]]:
    """Get known embedding models that don't require API calls"""
    models = []
    
    embedding_models_data = {
        'sentence-transformers': [
            ('all-MiniLM-L6-v2', 'Local, multilingual, efficient'),
            ('all-mpnet-base-v2', 'Local, high quality, semantic search'),
            ('all-MiniLM-L12-v2', 'Local, balanced performance'),
            ('paraphrase-MiniLM-L6-v2', 'Local, paraphrase detection'),
        ],
        'huggingface': [
            ('BAAI/bge-large-en-v1.5', 'Local, high performance English'),
            ('BAAI/bge-base-en-v1.5', 'Local, balanced English'),
            ('intfloat/e5-large-v2', 'Local, multilingual'),
            ('sentence-transformers/all-MiniLM-L6-v2', 'Local, efficient'),
        ]
    }
    
    for provider_name, model_list in embedding_models_data.items():
        if provider_filter and provider_filter != provider_name:
            continue
            
        for model_name, capabilities in model_list:
            models.append({
                'provider': provider_name,
                'type': 'embedding',
                'model': model_name,
                'status': 'local',
                'capabilities': capabilities,
                'note': 'Available for download'
            })
    
    return models


def _get_model_capabilities(model_id: str, model_type: str) -> str:
    """Get capabilities description for a model"""
    model_id_lower = model_id.lower()
    
    if model_type == 'llm':
        if 'gpt-4' in model_id_lower:
            return 'Advanced reasoning, coding, multimodal'
        elif 'gpt-3.5' in model_id_lower:
            return 'Text generation, conversation'
        elif 'claude-3.5' in model_id_lower:
            return 'Advanced reasoning, coding, analysis'
        elif 'claude-3' in model_id_lower:
            if 'opus' in model_id_lower:
                return 'Highest intelligence, complex tasks'
            elif 'sonnet' in model_id_lower:
                return 'Balanced performance, versatile'
            elif 'haiku' in model_id_lower:
                return 'Fast, efficient, lightweight'
        elif 'gemini' in model_id_lower:
            if 'pro' in model_id_lower:
                return 'Advanced reasoning, multimodal'
            elif 'flash' in model_id_lower:
                return 'Fast, efficient responses'
        elif any(keyword in model_id_lower for keyword in ['llama', 'mistral', 'phi']):
            return 'Open source, local deployment'
        else:
            return 'Text generation, conversation'
    
    elif model_type == 'embedding':
        if 'ada-002' in model_id_lower:
            return 'General purpose embeddings'
        elif 'embedding-001' in model_id_lower:
            return 'Google embeddings'
        elif 'minilm' in model_id_lower:
            return 'Efficient, lightweight'
        elif 'mpnet' in model_id_lower:
            return 'High quality semantic search'
        elif 'bge' in model_id_lower:
            return 'High performance, multilingual'
        elif 'e5' in model_id_lower:
            return 'Multilingual, versatile'
        else:
            return 'Text embeddings'
    
    return 'General purpose'


@model_group.command(name='set-model')
@click.option('--llm-provider', type=click.Choice(['openai', 'gemini', 'claude', 'ollama']),
              help='LLM provider to configure.')
@click.option('--llm-model', help='LLM model to use.')
@click.option('--embedding-provider', type=click.Choice(['openai', 'sentence-transformers', 'huggingface']),
              help='Embedding provider to configure.')
@click.option('--embedding-model', help='Embedding model to use.')
@click.option('--save/--no-save', default=True,
              help='Save configuration to file.')
def set_model(llm_provider, llm_model, embedding_provider, embedding_model, save):
    """Configure model settings.
    
    Updates the configuration for LLM and embedding models.
    Changes can be saved to the configuration file.
    
    Examples:
        rag-cli model set-model --llm-provider openai --llm-model gpt-4
        rag-cli model set-model --embedding-provider sentence-transformers
        rag-cli model set-model --llm-model gpt-3.5-turbo --no-save
    """
    asyncio.run(_set_model_async(llm_provider, llm_model, embedding_provider, embedding_model, save))


async def _set_model_async(llm_provider: Optional[str], llm_model: Optional[str], 
                          embedding_provider: Optional[str], embedding_model: Optional[str], 
                          save: bool):
    # Configuring model settings
    
    changes = {}
    if llm_provider:
        changes['LLM Provider'] = llm_provider
    if llm_model:
        changes['LLM Model'] = llm_model
    if embedding_provider:
        changes['Embedding Provider'] = embedding_provider
    if embedding_model:
        changes['Embedding Model'] = embedding_model
    
    if not changes:
        console.print("[yellow]No changes specified[/yellow]")
        return
    
    console.print("[yellow]Model configuration changes:[/yellow]")
    for key, value in changes.items():
        console.print(f"  • {key}: {value}")
    
    if save:
        console.print(f"\n[dim]Changes will be saved to configuration file[/dim]")
    else:
        console.print(f"\n[dim]Changes will only apply to current session[/dim]")
    
    if not confirm_action("Apply these model configuration changes?"):
        # Model configuration cancelled
        return
    
    try:
        # Get current configuration
        config = get_config()
        
        # Create backup of current configuration
        backup_config = None
        if save:
            backup_config = _create_config_backup(config)
        
        # Apply changes to configuration
        updated_config = _apply_model_config_changes(
            config, 
            llm_provider, 
            llm_model, 
            embedding_provider, 
            embedding_model
        )
        
        # Validate the new configuration
        validation_errors = _validate_model_config(updated_config)
        if validation_errors:
            console.print("[red]Configuration validation failed:[/red]")
            for error in validation_errors:
                console.print(f"  • {error}")
            return
        
        # Test the new configuration (optional)
        if confirm_action("Test new configuration before saving?", default=True):
            test_results = await _test_model_config(updated_config)
            if not test_results['success']:
                console.print("[red]Configuration test failed:[/red]")
                for error in test_results['errors']:
                    console.print(f"  • {error}")
                
                if not confirm_action("Save configuration anyway?", default=False):
                    console.print("[yellow]Configuration update cancelled[/yellow]")
                    return
        
        # Save configuration if requested
        if save:
            try:
                _save_model_config(updated_config)
                console.print("[green]✓ Configuration saved successfully[/green]")
                
                # Log the change
                _log_config_change(changes, updated_config)
                
            except Exception as e:
                console.print(f"[red]Failed to save configuration: {e}[/red]")
                
                # Restore backup if available
                if backup_config:
                    try:
                        _restore_config_backup(backup_config)
                        console.print("[yellow]Configuration restored from backup[/yellow]")
                    except Exception as backup_error:
                        console.print(f"[red]Failed to restore backup: {backup_error}[/red]")
                return
        else:
            # Apply configuration to current session only
            _apply_session_config(updated_config)
            console.print("[green]✓ Configuration applied to current session[/green]")
        
        # Display final configuration summary
        _display_config_summary(updated_config)
        
    except Exception as e:
        console.print(f"[red]Error updating model configuration: {e}[/red]")
        import traceback
        console.print(f"[dim red]{traceback.format_exc()}[/dim red]")


def _create_config_backup(config) -> Dict[str, Any]:
    """Create a backup of current configuration"""
    import copy
    from datetime import datetime
    
    backup = {
        'timestamp': datetime.now().isoformat(),
        'config': copy.deepcopy(config.__dict__ if hasattr(config, '__dict__') else config)
    }
    
    # Save backup to file
    backup_file = Path(f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(backup_file, 'w') as f:
            json.dump(backup, f, indent=2, default=str)
        console.print(f"[dim]Configuration backup saved to {backup_file}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not save backup: {e}[/yellow]")
    
    return backup


def _apply_model_config_changes(config, llm_provider: Optional[str], llm_model: Optional[str],
                               embedding_provider: Optional[str], embedding_model: Optional[str]):
    """Apply model configuration changes to config object"""
    import copy
    
    # Create a copy of the configuration
    if hasattr(config, '__dict__'):
        updated_config = copy.deepcopy(config)
    else:
        updated_config = copy.deepcopy(config)
    
    # Apply LLM changes
    if llm_provider or llm_model:
        if not hasattr(updated_config, 'llm'):
            # Create LLM config section if it doesn't exist
            from types import SimpleNamespace
            updated_config.llm = SimpleNamespace()
        
        if llm_provider:
            updated_config.llm.provider = llm_provider
            
        if llm_model:
            updated_config.llm.model = llm_model
    
    # Apply embedding changes
    if embedding_provider or embedding_model:
        if not hasattr(updated_config, 'embedding'):
            # Create embedding config section if it doesn't exist
            from types import SimpleNamespace
            updated_config.embedding = SimpleNamespace()
        
        if embedding_provider:
            updated_config.embedding.provider = embedding_provider
            
        if embedding_model:
            updated_config.embedding.model = embedding_model
    
    return updated_config


def _validate_model_config(config) -> List[str]:
    """Validate model configuration and return list of errors"""
    errors = []
    
    # Check LLM configuration
    if hasattr(config, 'llm'):
        if hasattr(config.llm, 'provider'):
            provider = config.llm.provider
            if provider not in ['openai', 'anthropic', 'google', 'ollama']:
                errors.append(f"Unsupported LLM provider: {provider}")
            
            # Check for required API keys
            if provider == 'openai' and not hasattr(config.llm, 'openai_api_key'):
                errors.append("OpenAI API key required for OpenAI provider")
            elif provider == 'anthropic' and not hasattr(config.llm, 'anthropic_api_key'):
                errors.append("Anthropic API key required for Anthropic provider")
            elif provider == 'google' and not hasattr(config.llm, 'google_api_key'):
                errors.append("Google API key required for Google provider")
        
        if hasattr(config.llm, 'model'):
            model = config.llm.model
            if not model or not isinstance(model, str):
                errors.append("LLM model name must be a non-empty string")
    
    # Check embedding configuration
    if hasattr(config, 'embedding'):
        if hasattr(config.embedding, 'provider'):
            provider = config.embedding.provider
            valid_providers = ['openai', 'sentence-transformers', 'huggingface', 'google']
            if provider not in valid_providers:
                errors.append(f"Unsupported embedding provider: {provider}")
        
        if hasattr(config.embedding, 'model'):
            model = config.embedding.model
            if not model or not isinstance(model, str):
                errors.append("Embedding model name must be a non-empty string")
    
    return errors


async def _test_model_config(config) -> Dict[str, Any]:
    """Test the model configuration by attempting to initialize models"""
    results = {
        'success': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Test LLM configuration
        if hasattr(config, 'llm') and hasattr(config.llm, 'provider'):
            try:
                from src.llm.manager import LLMManager
                llm_manager = LLMManager(config.llm)
                # Basic initialization test - don't make actual API calls
                results['warnings'].append("LLM configuration appears valid (not tested with API)")
            except Exception as e:
                results['errors'].append(f"LLM configuration test failed: {e}")
                results['success'] = False
        
        # Test embedding configuration
        if hasattr(config, 'embedding') and hasattr(config.embedding, 'provider'):
            try:
                from src.embedding.manager import EmbeddingManager
                # Test embedding manager initialization
                if config.embedding.provider in ['openai', 'google']:
                    # API-based providers - just validate config structure
                    results['warnings'].append("Embedding configuration appears valid (not tested with API)")
                else:
                    # Local providers - could potentially test model loading
                    results['warnings'].append("Local embedding configuration appears valid")
            except Exception as e:
                results['errors'].append(f"Embedding configuration test failed: {e}")
                results['success'] = False
    
    except Exception as e:
        results['errors'].append(f"Configuration test failed: {e}")
        results['success'] = False
    
    return results


def _save_model_config(config):
    """Save model configuration to file"""
    config_file = Path("config.json")  # Adjust path as needed
    
    try:
        # Convert config object to dictionary
        if hasattr(config, '__dict__'):
            config_dict = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    config_dict[key] = value.__dict__
                else:
                    config_dict[key] = value
        else:
            config_dict = config
        
        # Save to file
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
            
    except Exception as e:
        raise Exception(f"Failed to save configuration to {config_file}: {e}")


def _restore_config_backup(backup_config):
    """Restore configuration from backup"""
    config_file = Path("config.json")
    
    try:
        original_config = backup_config['config']
        with open(config_file, 'w') as f:
            json.dump(original_config, f, indent=2, default=str)
    except Exception as e:
        raise Exception(f"Failed to restore configuration backup: {e}")


def _apply_session_config(config):
    """Apply configuration to current session without saving to file"""
    # This would typically update global configuration objects
    # For now, just indicate that session config was applied
    global _session_config
    _session_config = config


def _display_config_summary(config):
    """Display a summary of the current configuration"""
    summary_data = []
    
    # LLM configuration
    if hasattr(config, 'llm'):
        llm_provider = getattr(config.llm, 'provider', 'Not set')
        llm_model = getattr(config.llm, 'model', 'Not set')
        summary_data.append({
            'Component': 'LLM',
            'Provider': llm_provider,
            'Model': llm_model,
            'Status': '✓' if llm_provider != 'Not set' else '○'
        })
    
    # Embedding configuration
    if hasattr(config, 'embedding'):
        embedding_provider = getattr(config.embedding, 'provider', 'Not set')
        embedding_model = getattr(config.embedding, 'model', 'Not set')
        summary_data.append({
            'Component': 'Embedding', 
            'Provider': embedding_provider,
            'Model': embedding_model,
            'Status': '✓' if embedding_provider != 'Not set' else '○'
        })
    
    if summary_data:
        display_table(summary_data, title="Current Model Configuration")
    else:
        console.print("[yellow]No model configuration found[/yellow]")


def _log_config_change(changes: Dict[str, str], config):
    """Log configuration changes"""
    from datetime import datetime
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'changes': changes,
        'user': 'cli-user',  # Could be enhanced with actual user info
        'config_summary': {
            'llm_provider': getattr(config.llm, 'provider', None) if hasattr(config, 'llm') else None,
            'llm_model': getattr(config.llm, 'model', None) if hasattr(config, 'llm') else None,
            'embedding_provider': getattr(config.embedding, 'provider', None) if hasattr(config, 'embedding') else None,
            'embedding_model': getattr(config.embedding, 'model', None) if hasattr(config, 'embedding') else None,
        }
    }
    
    # Write to log file
    log_file = Path("config_changes.log")
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        console.print(f"[dim]Configuration change logged to {log_file}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not write to log file: {e}[/yellow]")


# Global variable to store session-only configuration
_session_config = None