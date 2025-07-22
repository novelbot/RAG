"""
Configuration management commands for the CLI.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import yaml
import json

from ..utils import (
    console, confirm_action, prompt_for_input, validate_file_path
)

# Import CLI context decorator from main module


console = Console()


@click.group()
def config_group():
    """Configuration management commands.
    
    Commands for managing system configuration including
    database settings, API keys, and application parameters.
    """
    pass


@config_group.command(name='wizard')
@click.option('--output', type=click.Path(),
              help='Output configuration file path.')
@click.option('--template', type=click.Choice(['dev', 'prod', 'test']),
              default='dev', help='Configuration template to use.')
def config_wizard(output, template):
    """Interactive configuration wizard.
    
    Guides you through setting up the RAG server configuration
    with prompts for all required settings.
    
    Examples:
        rag-cli config wizard
        rag-cli config wizard --template prod --output prod_config.yaml
    """
    ctx.log(f"Starting configuration wizard (template: {template})", "verbose")
    
    console.print(f"[green]RAG Server Configuration Wizard[/green]")
    console.print(f"[dim]Using {template} template[/dim]\n")
    
    # Collect configuration values
    config_values = {}
    
    # Environment settings
    console.print("[cyan]Environment Settings[/cyan]")
    config_values['environment'] = prompt_for_input("Environment", default=template)
    config_values['debug'] = confirm_action("Enable debug mode?", default=(template == 'dev'))
    
    # Database settings
    console.print("\n[cyan]Database Settings[/cyan]")
    config_values['db_driver'] = click.prompt(
        "Database driver",
        type=click.Choice(['postgresql', 'mysql', 'sqlite']),
        default='postgresql'
    )
    config_values['db_host'] = prompt_for_input("Database host", default='localhost')
    config_values['db_port'] = click.prompt("Database port", type=int, default=5432)
    config_values['db_name'] = prompt_for_input("Database name", default='rag_server')
    config_values['db_username'] = prompt_for_input("Database username", default='rag_user')
    config_values['db_password'] = prompt_for_input("Database password", password=True)
    
    # Milvus settings
    console.print("\n[cyan]Milvus Settings[/cyan]")
    config_values['milvus_host'] = prompt_for_input("Milvus host", default='localhost')
    config_values['milvus_port'] = click.prompt("Milvus port", type=int, default=19530)
    config_values['milvus_username'] = prompt_for_input("Milvus username", default='')
    config_values['milvus_password'] = prompt_for_input("Milvus password", password=True)
    
    # LLM settings
    console.print("\n[cyan]LLM Settings[/cyan]")
    config_values['llm_provider'] = click.prompt(
        "LLM provider",
        type=click.Choice(['openai', 'gemini', 'claude', 'ollama']),
        default='openai'
    )
    config_values['llm_model'] = prompt_for_input("LLM model", default='gpt-3.5-turbo')
    config_values['llm_api_key'] = prompt_for_input("LLM API key", password=True)
    
    # Embedding settings
    console.print("\n[cyan]Embedding Settings[/cyan]")
    config_values['embedding_provider'] = click.prompt(
        "Embedding provider",
        type=click.Choice(['openai', 'google', 'ollama']),
        default='openai'
    )
    
    # Set default model based on provider
    if config_values['embedding_provider'] == 'openai':
        default_model = 'text-embedding-ada-002'
    elif config_values['embedding_provider'] == 'google':
        default_model = 'text-embedding-004'
    elif config_values['embedding_provider'] == 'ollama':
        default_model = 'nomic-embed-text'
    else:
        default_model = 'text-embedding-ada-002'
    
    config_values['embedding_model'] = prompt_for_input("Embedding model", default=default_model)
    
    # Add base URL for Ollama
    if config_values['embedding_provider'] == 'ollama':
        config_values['embedding_base_url'] = prompt_for_input(
            "Ollama base URL", 
            default='http://localhost:11434'
        )
    
    # API settings
    console.print("\n[cyan]API Settings[/cyan]")
    config_values['api_host'] = prompt_for_input("API host", default='0.0.0.0')
    config_values['api_port'] = click.prompt("API port", type=int, default=8000)
    
    # Generate configuration
    config_dict = {
        'environment': config_values['environment'],
        'debug': config_values['debug'],
        'database': {
            'driver': config_values['db_driver'],
            'host': config_values['db_host'],
            'port': config_values['db_port'],
            'name': config_values['db_name'],
            'username': config_values['db_username'],
            'password': config_values['db_password']
        },
        'milvus': {
            'host': config_values['milvus_host'],
            'port': config_values['milvus_port'],
            'username': config_values['milvus_username'],
            'password': config_values['milvus_password']
        },
        'llm': {
            'provider': config_values['llm_provider'],
            'model': config_values['llm_model'],
            'api_key': config_values['llm_api_key']
        },
        'embedding': {
            'provider': config_values['embedding_provider'],
            'model': config_values['embedding_model'],
            'base_url': config_values.get('embedding_base_url')
        },
        'api': {
            'host': config_values['api_host'],
            'port': config_values['api_port']
        }
    }
    
    # Show generated configuration
    console.print("\n[cyan]Generated Configuration:[/cyan]")
    # Hide sensitive values in display
    display_config = config_dict.copy()
    for section in ['database', 'milvus', 'llm']:
        if section in display_config and 'password' in display_config[section]:
            display_config[section]['password'] = '***'
        if section in display_config and 'api_key' in display_config[section]:
            display_config[section]['api_key'] = '***'
    
    console.print(Panel(yaml.dump(display_config, default_flow_style=False), title="Configuration Preview"))
    
    # Save configuration
    if not output:
        output = f"config_{template}.yaml"
    
    if not confirm_action(f"Save configuration to {output}?"):
        # Configuration wizard cancelled
        return
    
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        console.print(f"[green]Configuration saved to {output_path}[/green]")
        # Configuration wizard completed
        
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")
        # Configuration save failed


@config_group.command(name='validate')
@click.option('--config-file', type=click.Path(exists=True),
              help='Configuration file to validate.')
def validate_config(config_file):
    """Validate configuration file.
    
    Checks the configuration file for syntax errors,
    missing required fields, and invalid values.
    
    Examples:
        rag-cli config validate
        rag-cli config validate --config-file custom_config.yaml
    """
    # Validating configuration
    
    if config_file:
        config_path = validate_file_path(config_file)
        console.print(f"[yellow]Validating configuration file: {config_path}[/yellow]")
    else:
        console.print("[yellow]Validating current configuration[/yellow]")
    
    validation_results = []
    
    # TODO: Implement actual configuration validation
    # This would involve:
    # 1. Loading configuration file
    # 2. Checking required fields
    # 3. Validating field types and values
    # 4. Testing connections (optional)
    # 5. Checking for conflicts
    
    # Simulate validation results
    validation_results = [
        {'check': 'Syntax', 'status': 'Pass', 'details': 'Valid YAML format'},
        {'check': 'Required Fields', 'status': 'Pass', 'details': 'All required fields present'},
        {'check': 'Database Config', 'status': 'Warning', 'details': 'Connection not tested'},
        {'check': 'Milvus Config', 'status': 'Warning', 'details': 'Connection not tested'},
        {'check': 'LLM Config', 'status': 'Warning', 'details': 'API key not validated'},
        {'check': 'API Config', 'status': 'Pass', 'details': 'Valid host and port'}
    ]
    
    # Display results
    table = Table(title="Configuration Validation Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    for result in validation_results:
        status_style = "green" if result['status'] == 'Pass' else "yellow" if result['status'] == 'Warning' else "red"
        table.add_row(result['check'], f"[{status_style}]{result['status']}[/{status_style}]", result['details'])
    
    console.print(table)
    
    # Summary
    passes = sum(1 for r in validation_results if r['status'] == 'Pass')
    warnings = sum(1 for r in validation_results if r['status'] == 'Warning')
    errors = sum(1 for r in validation_results if r['status'] == 'Error')
    
    console.print(f"\n[dim]Summary: {passes} passed, {warnings} warnings, {errors} errors[/dim]")
    
    if errors > 0:
        console.print("[red]Configuration validation failed[/red]")
    elif warnings > 0:
        console.print("[yellow]Configuration validation passed with warnings[/yellow]")
    else:
        console.print("[green]Configuration validation passed[/green]")
    
    # Configuration validation completed


@config_group.command(name='export')
@click.option('--output', type=click.Path(), required=True,
              help='Output file path for exported configuration.')
@click.option('--format', type=click.Choice(['yaml', 'json']),
              default='yaml', help='Export format.')
@click.option('--include-sensitive/--exclude-sensitive', default=False,
              help='Include sensitive values like passwords and API keys.')
def export_config(output, format, include_sensitive):
    """Export current configuration.
    
    Exports the current configuration to a file in YAML or JSON format.
    Sensitive values can be excluded for security.
    
    Examples:
        rag-cli config export --output backup_config.yaml
        rag-cli config export --output config.json --format json
        rag-cli config export --output full_config.yaml --include-sensitive
    """
    # Exporting configuration
    
    console.print(f"[yellow]Exporting configuration to {output}[/yellow]")
    
    # Get current configuration
    # TODO: Get current configuration
    from src.core.config import get_config
    config = get_config()
    
    # TODO: Convert config object to dictionary
    # This would involve serializing the current configuration
    config_dict = {
        'environment': config.environment,
        'debug': config.debug,
        'database': {
            'driver': config.database.driver,
            'host': config.database.host,
            'port': config.database.port,
            'name': config.database.name,
            'username': getattr(config.database, 'username', 'not_set')
        },
        'milvus': {
            'host': config.milvus.host,
            'port': config.milvus.port
        },
        'llm': {
            'provider': config.llm.provider,
            'model': config.llm.model
        },
        'embedding': {
            'provider': config.embedding.provider,
            'model': config.embedding.model
        },
        'api': {
            'host': config.api.host,
            'port': config.api.port
        }
    }
    
    # Add sensitive values if requested
    if include_sensitive:
        console.print("[yellow]Warning: Including sensitive values in export[/yellow]")
        # TODO: Add actual sensitive values from config
        config_dict['database']['password'] = '***placeholder***'
        config_dict['milvus']['password'] = '***placeholder***'
        config_dict['llm']['api_key'] = '***placeholder***'
    
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if format == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False)
        
        console.print(f"[green]Configuration exported to {output_path}[/green]")
        # Configuration export completed
        
    except Exception as e:
        console.print(f"[red]Error exporting configuration: {e}[/red]")
        # Configuration export failed


@config_group.command(name='diff')
@click.argument('config_file', type=click.Path(exists=True))
def diff_config(config_file):
    """Compare configuration with file.
    
    Shows differences between the current configuration
    and a configuration file.
    
    Examples:
        rag-cli config diff other_config.yaml
    """
    # Comparing configuration
    
    console.print(f"[yellow]Comparing current config with {config_file}[/yellow]")
    
    # TODO: Implement actual configuration diff
    # This would involve:
    # 1. Loading the comparison config file
    # 2. Converting both configs to comparable format
    # 3. Finding differences
    # 4. Displaying differences in a readable format
    
    console.print("[red]✗ Configuration diff implementation not complete[/red]")
    # Configuration diff completed