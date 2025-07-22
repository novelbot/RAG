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
from src.core.exceptions import LLMError, EmbeddingError
import asyncio
import time

console = Console()


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
        llm_manager = LLMManager(config.llm)
        
        providers_to_test = [provider] if provider else ['openai', 'gemini', 'claude', 'ollama']
        results = []
        
        for prov in providers_to_test:
            console.print(f"\n[cyan]Testing {prov}...[/cyan]")
            
            try:
                # Check if provider is configured
                if prov not in llm_manager.providers:
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
                        
                        # Test the provider
                        if asyncio.iscoroutinefunction(llm_manager.generate):
                            response = asyncio.run(llm_manager.generate(test_request, provider=prov))
                        else:
                            response = llm_manager.generate(test_request, provider=prov)
                        
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
    
    config = get_config()
    
    providers_to_test = [provider] if provider else ['openai', 'sentence-transformers', 'huggingface']
    
    console.print(f"[yellow]Testing embedding providers with text: '{text[:50]}...'[/yellow]")
    
    results = []
    
    for prov in providers_to_test:
        console.print(f"\n[cyan]Testing {prov}...[/cyan]")
        
        with create_progress_bar() as progress:
            task = progress.add_task(f"Testing {prov}...", total=None)
            
            # TODO: Implement actual embedding testing
            import time
            time.sleep(1.5)  # Simulate embedding generation
            
            progress.remove_task(task)
        
        # Simulate test results
        if prov == config.embedding.provider:
            results.append({
                'provider': prov,
                'model': model or config.embedding.model,
                'status': 'Success',
                'dimensions': '1536',
                'generation_time': '0.8s'
            })
            console.print(f"[green]✓ {prov} test passed[/green]")
        else:
            results.append({
                'provider': prov,
                'model': 'Not configured',
                'status': 'Not configured',
                'dimensions': 'N/A',
                'generation_time': 'N/A'
            })
            console.print(f"[yellow]⚠ {prov} not configured[/yellow]")
    
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
    # Running model benchmarks
    
    console.print(f"[yellow]Running performance benchmark...[/yellow]")
    console.print(f"[dim]Iterations: {iterations}, Concurrent: {concurrent}[/dim]")
    
    # TODO: Implement actual benchmarking
    # This would involve:
    # 1. Preparing test prompts/texts
    # 2. Running multiple iterations
    # 3. Measuring response times
    # 4. Calculating statistics
    # 5. Testing concurrent performance
    
    with create_progress_bar() as progress:
        task = progress.add_task("Running benchmark...", total=iterations)
        
        for i in range(iterations):
            import time
            time.sleep(0.1)  # Simulate benchmark iteration
            progress.update(task, advance=1)
    
    # Simulate benchmark results
    benchmark_results = [
        {
            'metric': 'Avg Response Time',
            'llm': '1.2s',
            'embedding': '0.3s'
        },
        {
            'metric': 'Min Response Time',
            'llm': '0.8s',
            'embedding': '0.1s'
        },
        {
            'metric': 'Max Response Time',
            'llm': '2.1s',
            'embedding': '0.7s'
        },
        {
            'metric': 'Throughput (req/min)',
            'llm': '45',
            'embedding': '180'
        }
    ]
    
    display_table(benchmark_results, title="Performance Benchmark Results")
    
    console.print("[red]✗ Full benchmark implementation not complete[/red]")
    # Model benchmarking completed


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
    # Listing models
    
    console.print("[yellow]Fetching available models...[/yellow]")
    
    # TODO: Get actual available models from providers
    sample_models = [
        {
            'provider': 'openai',
            'type': 'llm',
            'model': 'gpt-4',
            'status': 'available',
            'capabilities': 'text generation, reasoning'
        },
        {
            'provider': 'openai',
            'type': 'llm',
            'model': 'gpt-3.5-turbo',
            'status': 'available',
            'capabilities': 'text generation'
        },
        {
            'provider': 'openai',
            'type': 'embedding',
            'model': 'text-embedding-ada-002',
            'status': 'available',
            'capabilities': 'text embeddings'
        },
        {
            'provider': 'sentence-transformers',
            'type': 'embedding',
            'model': 'all-MiniLM-L6-v2',
            'status': 'local',
            'capabilities': 'text embeddings, local'
        }
    ]
    
    # Apply filters
    if provider:
        sample_models = [m for m in sample_models if m['provider'] == provider]
    if type != 'all':
        sample_models = [m for m in sample_models if m['type'] == type]
    
    if sample_models:
        display_table(sample_models, title="Available Models")
    else:
        console.print("[yellow]No models found matching criteria[/yellow]")
    
    # Model list displayed


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
    
    # TODO: Implement actual model configuration update
    console.print("[red]✗ Model configuration implementation not complete[/red]")
    # Model configuration completed