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
import os
import secrets
from datetime import datetime

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
              help='Output .env file path (default: .env).')
@click.option('--template', type=click.Choice(['dev', 'prod', 'test']),
              default='dev', help='Configuration template to use.')
def config_wizard(output, template):
    """Interactive configuration wizard.
    
    Guides you through setting up the RAG server configuration
    and generates a .env file with all required settings.
    
    Examples:
        rag-cli config wizard
        rag-cli config wizard --template prod --output .env.prod
    """
    console.print(f"[green]RAG Server Configuration Wizard[/green]")
    console.print(f"[dim]Using {template} template - will generate .env file[/dim]\n")
    
    # Set default output file
    if not output:
        output = '.env'
    
    output_path = Path(output)
    
    # Check if .env file already exists and create backup
    if output_path.exists():
        backup_path = f"{output_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            output_path.rename(backup_path)
            console.print(f"[yellow]Existing {output_path} backed up to {backup_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]Warning: Could not backup existing file: {e}[/red]")
    
    # Collect configuration values
    config_values = {}
    
    # Environment settings
    console.print("[cyan]Environment Settings[/cyan]")
    config_values['app_env'] = click.prompt(
        "Application environment",
        type=click.Choice(['development', 'production', 'testing']),
        default='development' if template == 'dev' else template
    )
    config_values['debug'] = confirm_action("Enable debug mode?", default=(template == 'dev'))
    
    # Database settings
    console.print("\n[cyan]Database Settings[/cyan]")
    config_values['db_host'] = prompt_for_input("Database host", default='localhost')
    config_values['db_port'] = click.prompt("Database port", type=int, default=3306)
    config_values['db_name'] = prompt_for_input("Database name", default='novelbot')
    config_values['db_user'] = prompt_for_input("Database username", default='root')
    config_values['db_password'] = prompt_for_input("Database password", password=True)
    
    # Milvus settings
    console.print("\n[cyan]Vector Database (Milvus) Settings[/cyan]")
    config_values['milvus_host'] = prompt_for_input("Milvus host", default='localhost')
    config_values['milvus_port'] = click.prompt("Milvus port", type=int, default=19530)
    config_values['milvus_user'] = prompt_for_input("Milvus username (leave empty for local)", default='')
    config_values['milvus_password'] = prompt_for_input("Milvus password (leave empty for local)", password=True, default='')
    
    # LLM settings
    console.print("\n[cyan]LLM Provider Settings[/cyan]")
    config_values['llm_provider'] = click.prompt(
        "LLM provider",
        type=click.Choice(['ollama', 'openai', 'anthropic', 'google']),
        default='ollama'
    )
    
    # Set default model based on provider
    llm_model_defaults = {
        'ollama': 'gemma3:27b-it-q8_0',
        'openai': 'gpt-3.5-turbo',
        'anthropic': 'claude-3-5-sonnet-latest',
        'google': 'gemini-2.0-flash-001'
    }
    
    config_values['llm_model'] = prompt_for_input(
        "LLM model", 
        default=llm_model_defaults.get(config_values['llm_provider'], 'gpt-3.5-turbo')
    )
    
    # API Key for LLM (only if not ollama)
    if config_values['llm_provider'] != 'ollama':
        config_values['llm_api_key'] = prompt_for_input(f"{config_values['llm_provider'].upper()} API key", password=True)
    else:
        config_values['llm_api_key'] = ''
    
    # Additional API Keys
    console.print("\n[cyan]API Keys (for different providers)[/cyan]")
    console.print("[dim]You can leave these empty if not using the respective providers[/dim]")
    config_values['openai_api_key'] = prompt_for_input("OpenAI API key", password=True, default='your-openai-api-key-here')
    config_values['anthropic_api_key'] = prompt_for_input("Anthropic API key", password=True, default='your-anthropic-api-key-here')
    config_values['google_api_key'] = prompt_for_input("Google API key", password=True, default='your-google-api-key-here')
    
    # Embedding settings
    console.print("\n[cyan]Embedding Provider Settings[/cyan]")
    config_values['embedding_provider'] = click.prompt(
        "Embedding provider",
        type=click.Choice(['ollama', 'openai', 'google']),
        default='ollama'
    )
    
    # Set default embedding model based on provider
    embedding_model_defaults = {
        'ollama': 'jeffh/intfloat-multilingual-e5-large-instruct:f32',
        'openai': 'text-embedding-ada-002',
        'google': 'text-embedding-004'
    }
    
    config_values['embedding_model'] = prompt_for_input(
        "Embedding model",
        default=embedding_model_defaults.get(config_values['embedding_provider'], 'text-embedding-ada-002')
    )
    
    # Embedding API Key (only if not ollama)
    if config_values['embedding_provider'] != 'ollama':
        config_values['embedding_api_key'] = prompt_for_input(f"Embedding API key for {config_values['embedding_provider']}", password=True)
    else:
        config_values['embedding_api_key'] = ''
    
    # Security settings
    console.print("\n[cyan]Security Settings[/cyan]")
    if confirm_action("Generate a new secret key?", default=True):
        config_values['secret_key'] = secrets.token_urlsafe(32)
        console.print("[dim]Generated secure secret key[/dim]")
    else:
        config_values['secret_key'] = prompt_for_input("Secret key", password=True, default='your-secret-key-here')
    
    # API settings
    console.print("\n[cyan]API Server Settings[/cyan]")
    config_values['api_host'] = prompt_for_input("API host", default='0.0.0.0')
    config_values['api_port'] = click.prompt("API port", type=int, default=8000)
    
    # Optional Ollama settings
    if config_values['llm_provider'] == 'ollama' or config_values['embedding_provider'] == 'ollama':
        console.print("\n[cyan]Ollama Configuration (Optional)[/cyan]")
        config_values['ollama_base_url'] = prompt_for_input("Ollama base URL", default='http://localhost:11434')
    else:
        config_values['ollama_base_url'] = None
    
    # Generate .env file content
    env_content = _generate_env_content(config_values, template)
    
    # Show generated configuration preview (with sensitive values hidden)
    console.print("\n[cyan]Generated .env Configuration Preview:[/cyan]")
    preview_content = _hide_sensitive_values(env_content)
    console.print(Panel(preview_content, title=f".env Configuration Preview ({template} template)"))
    
    # Save configuration
    if not confirm_action(f"Save configuration to {output_path}?"):
        console.print("[yellow]Configuration wizard cancelled[/yellow]")
        return
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        console.print(f"[green]Configuration saved to {output_path}[/green]")
        console.print(f"[dim]You can now run your RAG server with the new configuration[/dim]")
        
        # Show next steps
        console.print("\n[cyan]Next Steps:[/cyan]")
        console.print("1. Review the generated .env file")
        console.print("2. Update any API keys if needed")
        console.print("3. Test database and Milvus connections")
        console.print("4. Start the RAG server")
        
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")


def _generate_env_content(config_values, template):
    """Generate .env file content from configuration values."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""# =============================================================================
# RAG Server Environment Configuration
# Generated by rag-cli config wizard on {timestamp}
# Template: {template}
# =============================================================================

# Application Settings
APP_ENV={config_values['app_env']}
DEBUG={'true' if config_values['debug'] else 'false'}

# =============================================================================
# Database Configuration
# =============================================================================
DB_HOST={config_values['db_host']}
DB_PORT={config_values['db_port']}
DB_NAME={config_values['db_name']}
DB_USER={config_values['db_user']}
DB_PASSWORD={config_values['db_password']}

# =============================================================================
# Vector Database Configuration (Milvus)
# =============================================================================
MILVUS_HOST={config_values['milvus_host']}
MILVUS_PORT={config_values['milvus_port']}
MILVUS_USER={config_values['milvus_user']}
MILVUS_PASSWORD={config_values['milvus_password']}

# =============================================================================
# LLM Provider Configuration
# =============================================================================
LLM_PROVIDER={config_values['llm_provider']}
LLM_MODEL={config_values['llm_model']}
LLM_API_KEY={config_values['llm_api_key']}

# =============================================================================
# API Keys for LLM Providers
# =============================================================================
OPENAI_API_KEY={config_values['openai_api_key']}
ANTHROPIC_API_KEY={config_values['anthropic_api_key']}
GOOGLE_API_KEY={config_values['google_api_key']}

# =============================================================================
# Embedding Provider Configuration
# =============================================================================
EMBEDDING_PROVIDER={config_values['embedding_provider']}
EMBEDDING_MODEL={config_values['embedding_model']}
EMBEDDING_API_KEY={config_values['embedding_api_key']}

# =============================================================================
# Authentication & Security
# =============================================================================
SECRET_KEY={config_values['secret_key']}

# =============================================================================
# API Server Configuration
# =============================================================================
API_HOST={config_values['api_host']}
API_PORT={config_values['api_port']}
"""
    
    # Add optional Ollama configuration
    if config_values.get('ollama_base_url'):
        content += f"""
# =============================================================================
# Optional: Ollama Configuration
# =============================================================================
OLLAMA_BASE_URL={config_values['ollama_base_url']}
"""
    
    return content


def _hide_sensitive_values(env_content):
    """Hide sensitive values in .env content for preview."""
    lines = env_content.split('\n')
    hidden_lines = []
    
    sensitive_keys = ['PASSWORD', 'API_KEY', 'SECRET_KEY']
    
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.split('=', 1)
            if any(sensitive in key.upper() for sensitive in sensitive_keys) and value.strip():
                if value.strip() and not value.startswith('your-') and value != 'true' and value != 'false':
                    hidden_lines.append(f"{key}=***")
                else:
                    hidden_lines.append(line)
            else:
                hidden_lines.append(line)
        else:
            hidden_lines.append(line)
    
    return '\n'.join(hidden_lines)


@config_group.command(name='validate')
@click.option('--config-file', type=click.Path(exists=True),
              help='Configuration file to validate (default: .env).')
@click.option('--check-connections/--no-check-connections', default=False,
              help='Test actual database and service connections.')
def validate_config(config_file, check_connections):
    """Validate configuration file.
    
    Checks the .env configuration file for syntax errors,
    missing required fields, and invalid values. Optionally
    tests actual connections to services.
    
    Examples:
        rag-cli config validate
        rag-cli config validate --config-file .env.prod
        rag-cli config validate --check-connections
    """
    # Determine config file to validate
    config_path = Path(config_file) if config_file else Path('.env')
    
    if not config_path.exists():
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        console.print("[dim]Run 'rag-cli config wizard' to create a configuration file first[/dim]")
        return
    
    console.print(f"[yellow]Validating configuration file: {config_path}[/yellow]")
    
    validation_results = []
    env_vars = {}
    
    # 1. Parse .env file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    try:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
                    except ValueError:
                        validation_results.append({
                            'check': f'Syntax (Line {line_num})',
                            'status': 'Error',
                            'details': f'Invalid format: {line[:50]}...'
                        })
        
        validation_results.append({
            'check': 'File Parsing',
            'status': 'Pass',
            'details': f'Successfully parsed {len(env_vars)} environment variables'
        })
    
    except Exception as e:
        validation_results.append({
            'check': 'File Parsing',
            'status': 'Error',
            'details': f'Could not read file: {e}'
        })
        env_vars = {}
    
    # 2. Check required fields
    required_fields = {
        'APP_ENV': 'Application environment',
        'DB_HOST': 'Database host',
        'DB_PORT': 'Database port',
        'DB_NAME': 'Database name',
        'DB_USER': 'Database user',
        'MILVUS_HOST': 'Milvus host',
        'MILVUS_PORT': 'Milvus port',
        'LLM_PROVIDER': 'LLM provider',
        'LLM_MODEL': 'LLM model',
        'EMBEDDING_PROVIDER': 'Embedding provider',
        'EMBEDDING_MODEL': 'Embedding model',
        'API_HOST': 'API host',
        'API_PORT': 'API port'
    }
    
    missing_fields = []
    for field, description in required_fields.items():
        if field not in env_vars or not env_vars[field]:
            missing_fields.append(f"{field} ({description})")
    
    if missing_fields:
        validation_results.append({
            'check': 'Required Fields',
            'status': 'Error',
            'details': f'Missing: {", ".join(missing_fields)}'
        })
    else:
        validation_results.append({
            'check': 'Required Fields',
            'status': 'Pass',
            'details': 'All required fields present'
        })
    
    # 3. Validate field values
    if env_vars:
        # Check APP_ENV
        if 'APP_ENV' in env_vars:
            valid_envs = ['development', 'production', 'testing']
            if env_vars['APP_ENV'] not in valid_envs:
                validation_results.append({
                    'check': 'APP_ENV Value',
                    'status': 'Warning',
                    'details': f'Unexpected value: {env_vars["APP_ENV"]}. Expected: {", ".join(valid_envs)}'
                })
            else:
                validation_results.append({
                    'check': 'APP_ENV Value',
                    'status': 'Pass',
                    'details': f'Valid environment: {env_vars["APP_ENV"]}'
                })
        
        # Check DEBUG
        if 'DEBUG' in env_vars:
            if env_vars['DEBUG'].lower() not in ['true', 'false']:
                validation_results.append({
                    'check': 'DEBUG Value',
                    'status': 'Warning',
                    'details': f'Expected "true" or "false", got: {env_vars["DEBUG"]}'
                })
            else:
                validation_results.append({
                    'check': 'DEBUG Value',
                    'status': 'Pass',
                    'details': f'Valid boolean: {env_vars["DEBUG"]}'
                })
        
        # Check numeric ports
        numeric_fields = ['DB_PORT', 'MILVUS_PORT', 'API_PORT']
        for field in numeric_fields:
            if field in env_vars:
                try:
                    port = int(env_vars[field])
                    if port < 1 or port > 65535:
                        validation_results.append({
                            'check': f'{field} Value',
                            'status': 'Warning',
                            'details': f'Port {port} outside valid range (1-65535)'
                        })
                    else:
                        validation_results.append({
                            'check': f'{field} Value',
                            'status': 'Pass',
                            'details': f'Valid port: {port}'
                        })
                except ValueError:
                    validation_results.append({
                        'check': f'{field} Value',
                        'status': 'Error',
                        'details': f'Invalid port number: {env_vars[field]}'
                    })
        
        # Check LLM provider
        if 'LLM_PROVIDER' in env_vars:
            valid_providers = ['ollama', 'openai', 'anthropic', 'google']
            if env_vars['LLM_PROVIDER'] not in valid_providers:
                validation_results.append({
                    'check': 'LLM Provider',
                    'status': 'Warning',
                    'details': f'Unexpected provider: {env_vars["LLM_PROVIDER"]}. Expected: {", ".join(valid_providers)}'
                })
            else:
                validation_results.append({
                    'check': 'LLM Provider',
                    'status': 'Pass',
                    'details': f'Valid provider: {env_vars["LLM_PROVIDER"]}'
                })
        
        # Check embedding provider
        if 'EMBEDDING_PROVIDER' in env_vars:
            valid_providers = ['ollama', 'openai', 'google']
            if env_vars['EMBEDDING_PROVIDER'] not in valid_providers:
                validation_results.append({
                    'check': 'Embedding Provider',
                    'status': 'Warning',
                    'details': f'Unexpected provider: {env_vars["EMBEDDING_PROVIDER"]}. Expected: {", ".join(valid_providers)}'
                })
            else:
                validation_results.append({
                    'check': 'Embedding Provider',
                    'status': 'Pass',
                    'details': f'Valid provider: {env_vars["EMBEDDING_PROVIDER"]}'
                })
        
        # Check API keys consistency
        if 'LLM_PROVIDER' in env_vars:
            provider = env_vars['LLM_PROVIDER']
            if provider == 'openai' and env_vars.get('LLM_API_KEY', '').startswith('your-'):
                validation_results.append({
                    'check': 'LLM API Key',
                    'status': 'Warning',
                    'details': 'OpenAI provider selected but API key appears to be placeholder'
                })
            elif provider == 'anthropic' and env_vars.get('LLM_API_KEY', '').startswith('your-'):
                validation_results.append({
                    'check': 'LLM API Key',
                    'status': 'Warning',
                    'details': 'Anthropic provider selected but API key appears to be placeholder'
                })
            elif provider == 'google' and env_vars.get('LLM_API_KEY', '').startswith('your-'):
                validation_results.append({
                    'check': 'LLM API Key',
                    'status': 'Warning',
                    'details': 'Google provider selected but API key appears to be placeholder'
                })
            elif provider == 'ollama':
                validation_results.append({
                    'check': 'LLM API Key',
                    'status': 'Pass',
                    'details': 'Ollama provider - no API key required'
                })
            else:
                validation_results.append({
                    'check': 'LLM API Key',
                    'status': 'Pass',
                    'details': 'API key appears to be configured'
                })
    
    # 4. Connection testing (if requested)
    if check_connections and env_vars:
        console.print("\n[yellow]Testing connections...[/yellow]")
        
        # Test database connection
        try:
            db_host = env_vars.get('DB_HOST', 'localhost')
            db_port = int(env_vars.get('DB_PORT', 3306))
            
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((db_host, db_port))
            sock.close()
            
            if result == 0:
                validation_results.append({
                    'check': 'Database Connection',
                    'status': 'Pass',
                    'details': f'Successfully connected to {db_host}:{db_port}'
                })
            else:
                validation_results.append({
                    'check': 'Database Connection',
                    'status': 'Error',
                    'details': f'Could not connect to {db_host}:{db_port}'
                })
        except Exception as e:
            validation_results.append({
                'check': 'Database Connection',
                'status': 'Error',
                'details': f'Connection test failed: {e}'
            })
        
        # Test Milvus connection
        try:
            milvus_host = env_vars.get('MILVUS_HOST', 'localhost')
            milvus_port = int(env_vars.get('MILVUS_PORT', 19530))
            
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((milvus_host, milvus_port))
            sock.close()
            
            if result == 0:
                validation_results.append({
                    'check': 'Milvus Connection',
                    'status': 'Pass',
                    'details': f'Successfully connected to {milvus_host}:{milvus_port}'
                })
            else:
                validation_results.append({
                    'check': 'Milvus Connection',
                    'status': 'Error',
                    'details': f'Could not connect to {milvus_host}:{milvus_port}'
                })
        except Exception as e:
            validation_results.append({
                'check': 'Milvus Connection',
                'status': 'Error',
                'details': f'Connection test failed: {e}'
            })
    
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
        console.print("[dim]Fix the errors above and run validation again[/dim]")
    elif warnings > 0:
        console.print("[yellow]Configuration validation passed with warnings[/yellow]")
        console.print("[dim]Consider reviewing the warnings above[/dim]")
    else:
        console.print("[green]Configuration validation passed[/green]")
        console.print("[dim]Your configuration looks good![/dim]")


@config_group.command(name='export')
@click.option('--output', type=click.Path(), required=True,
              help='Output file path for exported configuration.')
@click.option('--format', type=click.Choice(['env', 'yaml', 'json']),
              default='env', help='Export format (.env, YAML, or JSON).')
@click.option('--include-sensitive/--exclude-sensitive', default=False,
              help='Include sensitive values like passwords and API keys.')
@click.option('--source', type=click.Path(exists=True),
              help='Source .env file to export from (default: .env in current directory).')
def export_config(output, format, include_sensitive, source):
    """Export current configuration.
    
    Exports the current configuration from .env file to various formats.
    Sensitive values can be excluded for security.
    
    Examples:
        rag-cli config export --output backup_config.env
        rag-cli config export --output config.json --format json
        rag-cli config export --output config.yaml --format yaml --include-sensitive
    """
    console.print(f"[yellow]Exporting configuration to {output}[/yellow]")
    
    # Determine source .env file
    source_file = Path(source) if source else Path('.env')
    
    if not source_file.exists():
        console.print(f"[red]Source .env file not found: {source_file}[/red]")
        console.print("[dim]Run 'rag-cli config wizard' to create a configuration file first[/dim]")
        return
    
    try:
        # Read current .env file
        env_vars = {}
        with open(source_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    try:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
                    except ValueError:
                        console.print(f"[yellow]Warning: Could not parse line {line_num}: {line}[/yellow]")
        
        if not env_vars:
            console.print("[red]No environment variables found in source file[/red]")
            return
        
        # Process sensitive values
        if not include_sensitive:
            sensitive_keys = ['PASSWORD', 'API_KEY', 'SECRET_KEY']
            for key in env_vars:
                if any(sensitive in key.upper() for sensitive in sensitive_keys):
                    if env_vars[key] and not env_vars[key].startswith('your-'):
                        env_vars[key] = '***REDACTED***'
        else:
            console.print("[yellow]Warning: Including sensitive values in export[/yellow]")
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'env':
            # Export as .env format
            _export_as_env(env_vars, output_path)
        elif format == 'json':
            # Export as JSON
            _export_as_json(env_vars, output_path)
        elif format == 'yaml':
            # Export as YAML
            _export_as_yaml(env_vars, output_path)
        
        console.print(f"[green]Configuration exported to {output_path}[/green]")
        console.print(f"[dim]Format: {format.upper()}, Sensitive values: {'included' if include_sensitive else 'excluded'}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error exporting configuration: {e}[/red]")


def _export_as_env(env_vars, output_path):
    """Export environment variables as .env file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# =============================================================================\n")
        f.write(f"# RAG Server Environment Configuration (Exported)\n")
        f.write(f"# Exported on {timestamp}\n")
        f.write(f"# =============================================================================\n\n")
        
        # Group related variables
        sections = {
            'Application Settings': ['APP_ENV', 'DEBUG'],
            'Database Configuration': ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'],
            'Vector Database Configuration (Milvus)': ['MILVUS_HOST', 'MILVUS_PORT', 'MILVUS_USER', 'MILVUS_PASSWORD'],
            'LLM Provider Configuration': ['LLM_PROVIDER', 'LLM_MODEL', 'LLM_API_KEY'],
            'API Keys for LLM Providers': ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'],
            'Embedding Provider Configuration': ['EMBEDDING_PROVIDER', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY'],
            'Authentication & Security': ['SECRET_KEY'],
            'API Server Configuration': ['API_HOST', 'API_PORT'],
            'Optional Configuration': ['OLLAMA_BASE_URL']
        }
        
        for section_name, keys in sections.items():
            section_vars = {k: v for k, v in env_vars.items() if k in keys}
            if section_vars:
                f.write(f"# =============================================================================\n")
                f.write(f"# {section_name}\n")
                f.write(f"# =============================================================================\n")
                for key in keys:
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")
                f.write("\n")
        
        # Add any remaining variables
        written_keys = set()
        for keys in sections.values():
            written_keys.update(keys)
        
        remaining_vars = {k: v for k, v in env_vars.items() if k not in written_keys}
        if remaining_vars:
            f.write(f"# =============================================================================\n")
            f.write(f"# Additional Configuration\n")
            f.write(f"# =============================================================================\n")
            for key, value in remaining_vars.items():
                f.write(f"{key}={value}\n")


def _export_as_json(env_vars, output_path):
    """Export environment variables as JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(env_vars, f, indent=2, sort_keys=True)


def _export_as_yaml(env_vars, output_path):
    """Export environment variables as YAML file."""
    # Convert flat env vars to nested structure for better YAML representation
    nested_config = {}
    
    for key, value in env_vars.items():
        if key.startswith('DB_'):
            if 'database' not in nested_config:
                nested_config['database'] = {}
            nested_config['database'][key.lower().replace('db_', '')] = value
        elif key.startswith('MILVUS_'):
            if 'milvus' not in nested_config:
                nested_config['milvus'] = {}
            nested_config['milvus'][key.lower().replace('milvus_', '')] = value
        elif key.startswith('LLM_'):
            if 'llm' not in nested_config:
                nested_config['llm'] = {}
            nested_config['llm'][key.lower().replace('llm_', '')] = value
        elif key.startswith('EMBEDDING_'):
            if 'embedding' not in nested_config:
                nested_config['embedding'] = {}
            nested_config['embedding'][key.lower().replace('embedding_', '')] = value
        elif key.startswith('API_'):
            if 'api' not in nested_config:
                nested_config['api'] = {}
            nested_config['api'][key.lower().replace('api_', '')] = value
        else:
            nested_config[key.lower()] = value
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(nested_config, f, default_flow_style=False)


@config_group.command(name='diff')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--current', type=click.Path(exists=True), 
              help='Current config file to compare against (default: .env)')
def diff_config(config_file, current):
    """Compare configuration files.
    
    Shows differences between two .env configuration files.
    Useful for comparing different environment configurations.
    
    Examples:
        rag-cli config diff .env.prod
        rag-cli config diff .env.test --current .env.dev
    """
    # Determine current config file
    current_file = Path(current) if current else Path('.env')
    compare_file = Path(config_file)
    
    if not current_file.exists():
        console.print(f"[red]Current config file not found: {current_file}[/red]")
        console.print("[dim]Run 'rag-cli config wizard' to create a configuration file first[/dim]")
        return
    
    console.print(f"[yellow]Comparing {current_file} with {compare_file}[/yellow]")
    
    try:
        # Read both files
        current_vars = _parse_env_file(current_file)
        compare_vars = _parse_env_file(compare_file)
        
        # Find differences
        all_keys = set(current_vars.keys()) | set(compare_vars.keys())
        differences = []
        
        for key in sorted(all_keys):
            current_val = current_vars.get(key, '[NOT SET]')
            compare_val = compare_vars.get(key, '[NOT SET]')
            
            # Hide sensitive values for display
            if any(sensitive in key.upper() for sensitive in ['PASSWORD', 'API_KEY', 'SECRET_KEY']):
                if current_val != '[NOT SET]' and not current_val.startswith('your-'):
                    current_val = '***'
                if compare_val != '[NOT SET]' and not compare_val.startswith('your-'):
                    compare_val = '***'
            
            if current_val != compare_val:
                differences.append({
                    'key': key,
                    'current': current_val,
                    'compare': compare_val,
                    'status': _get_diff_status(current_val, compare_val)
                })
        
        if not differences:
            console.print("[green]✓ Configuration files are identical[/green]")
            return
        
        # Display differences table
        table = Table(title=f"Configuration Differences")
        table.add_column("Variable", style="cyan")
        table.add_column(f"{current_file.name}", style="blue")
        table.add_column(f"{compare_file.name}", style="magenta")
        table.add_column("Status", style="yellow")
        
        for diff in differences:
            status_style = _get_status_style(diff['status'])
            table.add_row(
                diff['key'],
                diff['current'],
                diff['compare'],
                f"[{status_style}]{diff['status']}[/{status_style}]"
            )
        
        console.print(table)
        
        # Summary
        added = sum(1 for d in differences if d['status'] == 'Added')
        removed = sum(1 for d in differences if d['status'] == 'Removed')
        changed = sum(1 for d in differences if d['status'] == 'Changed')
        
        console.print(f"\n[dim]Summary: {added} added, {removed} removed, {changed} changed[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error comparing configuration files: {e}[/red]")


def _parse_env_file(file_path):
    """Parse .env file and return dictionary of variables."""
    env_vars = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                try:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                except ValueError:
                    pass  # Skip malformed lines
    return env_vars


def _get_diff_status(current_val, compare_val):
    """Determine the status of a configuration difference."""
    if current_val == '[NOT SET]':
        return 'Added'
    elif compare_val == '[NOT SET]':
        return 'Removed'
    else:
        return 'Changed'


def _get_status_style(status):
    """Get the style for a diff status."""
    if status == 'Added':
        return 'green'
    elif status == 'Removed':
        return 'red'
    else:  # Changed
        return 'yellow'