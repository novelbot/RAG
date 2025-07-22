"""
Main CLI application for managing the RAG server.
"""

import os
import sys
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import get_config, reload_config
from src.core.logging import setup_logging

console = Console()


class CLIContext:
    """Context object to hold CLI state and configuration."""
    
    def __init__(self, debug=False, verbose=False, config_file=None):
        self.debug = debug
        self.verbose = verbose
        self.config_file = config_file
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use default."""
        try:
            if self.config_file:
                # TODO: Load specific config file
                self.config = get_config()
            else:
                self.config = get_config()
            
            # Setup logging based on CLI options
            if self.debug:
                self.config.debug = True
            
            setup_logging(self.config.logging)
            
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            if self.debug:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    def log(self, message, level="info"):
        """Log a message based on verbosity settings."""
        if level == "debug" and not self.debug:
            return
        if level == "verbose" and not self.verbose:
            return
            
        if level == "error":
            console.print(f"[red]{message}[/red]")
        elif level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "success":
            console.print(f"[green]{message}[/green]")
        elif level == "debug":
            console.print(f"[dim]{message}[/dim]")
        else:
            console.print(message)


# Create a decorator to pass the CLI context
pass_cli_context = click.make_pass_decorator(CLIContext, ensure=True)


def handle_exceptions(f):
    """Decorator to handle common CLI exceptions."""
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except click.ClickException:
            # Let Click handle its own exceptions
            raise
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            # Print traceback if debug mode is enabled
            ctx = click.get_current_context(silent=True)
            if ctx and hasattr(ctx.obj, 'debug') and ctx.obj.debug:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    return wrapper


@click.group(context_settings={
    'help_option_names': ['-h', '--help'],
    'auto_envvar_prefix': 'RAG_CLI'
})
@click.option('--debug/--no-debug', default=False, envvar='RAG_CLI_DEBUG',
              help='Enable debug mode with verbose logging and error traces.')
@click.option('--verbose/--no-verbose', default=False, envvar='RAG_CLI_VERBOSE',
              help='Enable verbose output.')
@click.option('--config-file', type=click.Path(exists=True), envvar='RAG_CLI_CONFIG',
              help='Path to configuration file.')
@click.version_option(version="0.1.0", prog_name="RAG Server CLI")
@click.pass_context
def cli(ctx, debug, verbose, config_file):
    """RAG Server CLI - Management interface for the RAG server.
    
    This CLI provides comprehensive management capabilities for the RAG server,
    including database operations, user management, model configuration,
    data ingestion, and system monitoring.
    
    Environment Variables:
        RAG_CLI_DEBUG: Enable debug mode (true/false)
        RAG_CLI_VERBOSE: Enable verbose output (true/false)
        RAG_CLI_CONFIG: Path to configuration file
    
    Examples:
        rag-cli status                    # Show server status
        rag-cli --debug test             # Run tests with debug output
        rag-cli user create --help       # Show user creation help
    """
    # Initialize CLI context
    ctx.obj = CLIContext(debug=debug, verbose=verbose, config_file=config_file)
    
    if debug:
        console.print("[dim]Debug mode enabled[/dim]")
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")
    if config_file:
        console.print(f"[dim]Using config file: {config_file}[/dim]")


@click.command()
@pass_cli_context
@handle_exceptions
def status(ctx):
    """Show server status and configuration."""
    ctx.log("Checking server status...", "verbose")
    
    config = ctx.config
    
    # Create status table
    table = Table(title="RAG Server Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Add configuration info
    table.add_row("Environment", config.environment, f"Debug: {config.debug}")
    table.add_row("Database", config.database.driver, f"{config.database.host}:{config.database.port}")
    table.add_row("Milvus", "Configured", f"{config.milvus.host}:{config.milvus.port}")
    table.add_row("LLM Provider", config.llm.provider, config.llm.model)
    table.add_row("Embedding", config.embedding.provider, config.embedding.model)
    
    console.print(table)
    ctx.log("Status check completed", "verbose")


@click.command(name='show-config')
@click.option('--show-sensitive/--hide-sensitive', default=False,
              help='Show sensitive configuration values like API keys.')
@pass_cli_context
@handle_exceptions
def show_config(ctx, show_sensitive):
    """Show current configuration.
    
    Displays the current server configuration including database settings,
    model configurations, and API endpoints.
    """
    ctx.log("Loading configuration...", "verbose")
    
    config = ctx.config
    
    # Build configuration display
    config_text = f"[bold]Configuration[/bold]\n"
    config_text += f"Environment: {config.environment}\n"
    config_text += f"Debug: {config.debug}\n"
    config_text += f"Database: {config.database.driver}://{config.database.host}:{config.database.port}/{config.database.name}\n"
    config_text += f"Milvus: {config.milvus.host}:{config.milvus.port}\n"
    config_text += f"LLM: {config.llm.provider} ({config.llm.model})\n"
    config_text += f"Embedding: {config.embedding.provider} ({config.embedding.model})\n"
    config_text += f"API: {config.api.host}:{config.api.port}"
    
    if show_sensitive:
        config_text += f"\n\n[bold red]Sensitive Values:[/bold red]\n"
        config_text += f"Database Password: {'*' * 8 if hasattr(config.database, 'password') else 'Not set'}\n"
        config_text += f"LLM API Key: {'*' * 8 if hasattr(config.llm, 'api_key') else 'Not set'}"
        ctx.log("Showing sensitive configuration values", "debug")
    
    # Display configuration
    console.print(Panel.fit(config_text, title="RAG Server Configuration"))
    ctx.log("Configuration displayed", "verbose")


@click.command()
@click.option('--component', type=click.Choice(['all', 'config', 'database', 'milvus', 'llm', 'embedding']),
              default='all', help='Component to test.')
@pass_cli_context
@handle_exceptions
def test(ctx, component):
    """Test system components.
    
    Runs connectivity and functionality tests for various system components.
    Use --component to test specific components.
    """
    ctx.log(f"Testing system components: {component}", "verbose")
    
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    components_to_test = ['config', 'database', 'milvus', 'llm', 'embedding'] if component == 'all' else [component]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for comp in components_to_test:
            task = progress.add_task(f"Testing {comp}...", total=None)
            
            if comp == 'config':
                console.print("[green]✓ Configuration loaded successfully[/green]")
                ctx.log("Configuration test passed", "debug")
            elif comp == 'database':
                console.print("[red]✗ Database connection test not implemented[/red]")
                ctx.log("Database test skipped - not implemented", "debug")
            elif comp == 'milvus':
                console.print("[red]✗ Milvus connection test not implemented[/red]")
                ctx.log("Milvus test skipped - not implemented", "debug")
            elif comp == 'llm':
                console.print("[red]✗ LLM provider test not implemented[/red]")
                ctx.log("LLM test skipped - not implemented", "debug")
            elif comp == 'embedding':
                console.print("[red]✗ Embedding model test not implemented[/red]")
                ctx.log("Embedding test skipped - not implemented", "debug")
            
            progress.remove_task(task)
    
    ctx.log("Component testing completed", "verbose")


@click.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@pass_cli_context
@handle_exceptions
def serve(ctx, host, port, reload):
    """Start the RAG server.
    
    Starts the FastAPI server with the specified host and port.
    Use --reload for development to enable auto-reload on code changes.
    """
    ctx.log(f"Starting RAG server on {host}:{port}", "verbose")
    console.print(f"[green]Starting RAG server on {host}:{port}[/green]")
    
    # Override config
    config = ctx.config
    config.api.host = host
    config.api.port = port
    config.api.reload = reload
    
    if reload:
        ctx.log("Auto-reload enabled", "debug")
    
    # Start server
    try:
        from src.core.app import run_server
        run_server(host=host, port=port, reload=reload)
    except ImportError as e:
        ctx.log(f"Failed to import server module: {e}", "error")
        raise click.ClickException("Server module not available")


# Import and register command groups
from .commands.data import data_group
from .commands.user import user_group
from .commands.database import database_group
from .commands.model import model_group
from .commands.config import config_group

# Register command groups with main CLI
cli.add_command(data_group, name='data')
cli.add_command(user_group, name='user')
cli.add_command(database_group, name='database')
cli.add_command(model_group, name='model')
cli.add_command(config_group, name='config')

# Add individual commands manually
cli.add_command(status)
cli.add_command(show_config)
cli.add_command(test)
cli.add_command(serve)


if __name__ == "__main__":
    cli()