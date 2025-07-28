"""
Data management commands for the CLI.
"""

import click
from rich.console import Console
from pathlib import Path
from typing import Optional

from ..utils import (
    console, confirm_action, create_progress_bar, 
    validate_directory_path, ProgressCallback
)

# This will be imported after the decorators are defined

console = Console()


@click.group()
def data_group():
    """Data management commands.
    
    Commands for ingesting, syncing, and managing data sources
    including file systems and databases.
    """
    pass


@data_group.command(name='ingest')
@click.option('--path', required=True, type=click.Path(exists=True),
              help='Path to data directory or file to ingest.')
@click.option('--recursive/--no-recursive', default=True,
              help='Recursively process subdirectories.')
@click.option('--file-types', default='txt,pdf,docx,md',
              help='Comma-separated list of file extensions to process.')
@click.option('--batch-size', default=100, type=int,
              help='Number of files to process in each batch.')
@click.option('--force/--no-force', default=False,
              help='Force re-ingestion of existing files.')
def ingest_data(path, recursive, file_types, batch_size, force):
    """Ingest data from directory or file.
    
    Processes files from the specified path and ingests them into
    the vector database. Supports multiple file formats including
    text, PDF, Word documents, and Markdown.
    
    Examples:
        rag-cli data ingest --path ./documents
        rag-cli data ingest --path ./file.pdf --file-types pdf
        rag-cli data ingest --path ./docs --batch-size 50 --force
    """
    console.print(f"[yellow]Starting data ingestion from {path}[/yellow]")
    
    # Validate path
    data_path = validate_directory_path(path, must_exist=True)
    console.print(f"[dim]Validated path: {data_path}[/dim]")
    
    # Parse file types
    allowed_extensions = [ext.strip().lower() for ext in file_types.split(',')]
    console.print(f"[dim]Processing file types: {allowed_extensions}[/dim]")
    
    console.print(f"[yellow]Ingesting data from {data_path}...[/yellow]")
    console.print(f"[dim]Recursive: {recursive}, Batch size: {batch_size}, Force: {force}[/dim]")
    
    if not force:
        if not confirm_action("This will process files and add them to the vector database. Continue?"):
            console.print("[yellow]Data ingestion cancelled by user[/yellow]")
            return
    
    # Implement actual data ingestion
    try:
        from src.core.app import create_app
        from src.extraction.factory import RDBExtractorFactory
        from src.extraction.base import ExtractionConfig, ExtractionMode, DataFormat
        from src.core.config import DatabaseType, DatabaseConfig, get_config
        from src.pipeline.pipeline import VectorPipeline, PipelineConfig, Document
        from src.milvus.client import MilvusClient
        from src.embedding.manager import EmbeddingManager
        from src.text_processing.text_cleaner import TextCleaner
        from src.text_processing.text_splitter import TextSplitter
        from src.text_processing.metadata_manager import MetadataManager
        import asyncio
        import json
        from datetime import datetime
        
        # Get application config
        config = get_config()
        
        # Check if we're processing files or database
        data_path_obj = Path(data_path)
        
        with create_progress_bar() as progress:
            if data_path_obj.is_file() or recursive:
                # File-based ingestion
                task = progress.add_task("Scanning files...", total=None)
                
                files_to_process = []
                if data_path_obj.is_file():
                    if data_path_obj.suffix.lower().lstrip('.') in allowed_extensions:
                        files_to_process.append(data_path_obj)
                else:
                    # Scan directory
                    for ext in allowed_extensions:
                        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
                        files_to_process.extend(data_path_obj.glob(pattern))
                
                progress.update(task, total=len(files_to_process), description=f"Found {len(files_to_process)} files")
                
                if not files_to_process:
                    console.print("[yellow]No files found matching the criteria[/yellow]")
                    return
                
                # Process files
                progress.update(task, description="Reading files...")
                documents = []
                
                for i, file_path in enumerate(files_to_process):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        doc = Document(
                            id=f"file_{file_path.stem}_{i}",
                            content=content,
                            metadata={
                                "source_type": "file",
                                "file_path": str(file_path),
                                "file_extension": file_path.suffix,
                                "file_size": file_path.stat().st_size,
                                "ingested_at": datetime.utcnow().isoformat()
                            },
                            source_path=str(file_path)
                        )
                        documents.append(doc)
                        
                        progress.update(task, advance=1, description=f"Processing {file_path.name}")
                        
                    except Exception as e:
                        console.print(f"[red]Error reading {file_path}: {e}[/red]")
                        if not force:
                            continue
                
                console.print(f"[green]✓ Loaded {len(documents)} documents from files[/green]")
                
            else:
                # Database-based ingestion using RDB Pipeline
                task = progress.add_task("Connecting to database...", total=None)
                
                # Check if RDB connections are configured
                if not config.rdb_connections:
                    console.print("[red]✗ No RDB connections configured[/red]")
                    console.print("[dim]Use 'rag-cli config database' to configure database connections[/dim]")
                    return
                
                # Use first configured RDB connection
                db_name = list(config.rdb_connections.keys())[0]
                db_config = config.rdb_connections[db_name]
                
                progress.update(task, description=f"Connecting to {db_config.host}")
                
                # Import RDB pipeline components
                from src.pipeline.rdb_pipeline import create_rdb_vector_pipeline, RDBPipelineConfig
                from src.pipeline.rdb_adapter import RDBAdapterConfig
                from src.extraction.base import ExtractionMode
                
                # Create adapter config
                adapter_config = RDBAdapterConfig(
                    content_format="structured",
                    include_table_name=True,
                    include_column_names=True,
                    exclude_null_values=True,
                    max_content_length=10000,
                    truncate_long_content=True
                )
                
                # Create RDB pipeline
                rdb_pipeline = create_rdb_vector_pipeline(
                    database_name=db_name,
                    database_config=db_config,
                    collection_name="documents",
                    adapter_config=adapter_config,
                    extraction_mode=ExtractionMode.FULL,
                    extraction_batch_size=batch_size,
                    continue_on_table_error=force,
                    continue_on_pipeline_error=force,
                    max_concurrent_tables=3,
                    enable_detailed_logging=True
                )
                
                progress.update(task, description="Processing database through RDB pipeline...")
                
                try:
                    # Process all tables through the integrated pipeline
                    async def process_rdb():
                        return await rdb_pipeline.process_all_tables()
                    
                    result = asyncio.run(process_rdb())
                    
                    # Display results
                    console.print(f"[green]✓ RDB Pipeline processing completed[/green]")
                    console.print(f"[dim]Database: {result.database_name}[/dim]")
                    console.print(f"[dim]Tables processed: {result.processed_tables}/{result.total_tables}[/dim]")
                    console.print(f"[dim]Documents processed: {result.successful_documents}/{result.total_documents}[/dim]")
                    console.print(f"[dim]Processing time: {result.processing_time:.2f}s[/dim]")
                    console.print(f"[dim]Table success rate: {result.table_success_rate:.1f}%[/dim]")
                    console.print(f"[dim]Document success rate: {result.document_success_rate:.1f}%[/dim]")
                    
                    if result.errors:
                        console.print(f"[yellow]Errors encountered:[/yellow]")
                        for error in result.errors[:5]:  # Show first 5 errors
                            console.print(f"[dim]  • {error.get('error', 'Unknown error')}[/dim]")
                        if len(result.errors) > 5:
                            console.print(f"[dim]  ... and {len(result.errors) - 5} more errors[/dim]")
                    
                    # Skip the manual pipeline processing since RDB pipeline handles everything
                    return
                    
                finally:
                    rdb_pipeline.close()
            
            if not documents:
                console.print("[yellow]No documents to process[/yellow]")
                return
            
            # Initialize pipeline components
            progress.update(task, description="Initializing embedding pipeline...")
            
            # Initialize Milvus client
            milvus_client = MilvusClient(config.milvus)
            
            # Initialize embedding manager
            embedding_manager = EmbeddingManager(config.embedding_providers)
            
            # Initialize text processing components
            text_cleaner = TextCleaner()
            text_splitter = TextSplitter()
            metadata_manager = MetadataManager()
            
            # Create pipeline config
            pipeline_config = PipelineConfig(
                batch_size=batch_size,
                max_concurrent_documents=50,
                enable_monitoring=True,
                continue_on_error=force
            )
            
            # Initialize pipeline
            pipeline = VectorPipeline(
                config=pipeline_config,
                milvus_client=milvus_client,
                embedding_manager=embedding_manager,
                text_cleaner=text_cleaner,
                text_splitter=text_splitter,
                metadata_manager=metadata_manager,
                collection_name="documents"
            )
            
            # Process documents through pipeline
            progress.update(task, description="Processing through embedding pipeline...")
            
            async def process_documents():
                await pipeline.initialize()
                result = await pipeline.process_documents(documents)
                await pipeline.shutdown()
                return result
            
            # Run pipeline
            result = asyncio.run(process_documents())
            
            # Display results
            console.print(f"[green]✓ Pipeline processing completed[/green]")
            console.print(f"[dim]Total documents: {result.total_documents}[/dim]")
            console.print(f"[dim]Successful: {result.successful_documents}[/dim]")
            console.print(f"[dim]Failed: {result.failed_documents}[/dim]")
            console.print(f"[dim]Processing time: {result.processing_time:.2f}s[/dim]")
            console.print(f"[dim]Success rate: {result.success_rate:.1f}%[/dim]")
            
            if result.errors:
                console.print(f"[yellow]Errors encountered:[/yellow]")
                for error in result.errors[:5]:  # Show first 5 errors
                    console.print(f"[dim]  • {error}[/dim]")
                if len(result.errors) > 5:
                    console.print(f"[dim]  ... and {len(result.errors) - 5} more errors[/dim]")
        
    except ImportError as e:
        console.print(f"[red]✗ Missing dependencies: {e}[/red]")
        console.print("[dim]Please ensure all required packages are installed[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Data ingestion failed: {e}[/red]")
        import traceback
        if force:
            console.print(f"[dim]Error details: {traceback.format_exc()}[/dim]")


@data_group.command(name='sync')
@click.option('--source', type=click.Choice(['filesystem', 'database', 'all']),
              default='all', help='Data source to sync.')
@click.option('--incremental/--full', default=True,
              help='Perform incremental sync (only changed data) or full sync.')
@click.option('--dry-run/--no-dry-run', default=False,
              help='Show what would be synced without making changes.')
def sync_data(source, incremental, dry_run):
    """Sync data sources.
    
    Synchronizes data from configured sources, detecting changes
    and updating the vector database accordingly.
    
    Examples:
        rag-cli data sync --source filesystem
        rag-cli data sync --source database --full
        rag-cli data sync --dry-run
    """
    console.print(f"[dim]Starting data sync for source: {source}[/dim]")
    
    sync_type = "incremental" if incremental else "full"
    console.print(f"[yellow]Syncing {source} data sources ({sync_type})...[/yellow]")
    
    if dry_run:
        console.print("[dim]Running in dry-run mode - no changes will be made[/dim]")
        console.print("[dim]Dry-run mode enabled[/dim]")
    
    # TODO: Implement actual data sync
    # This would involve:
    # 1. Checking configured data sources
    # 2. Detecting changes since last sync
    # 3. Processing changed/new data
    # 4. Updating vector database
    # 5. Cleaning up deleted data
    
    console.print("[red]✗ Data sync implementation not complete[/red]")
    console.print("[dim]Data sync completed (placeholder)[/dim]")


@data_group.command(name='status')
def data_status():
    """Show data ingestion status.
    
    Displays information about ingested data, including
    document counts, last sync times, and storage usage.
    """
    console.print("[dim]Checking data status[/dim]")
    
    from rich.table import Table
    
    # Create status table
    table = Table(title="Data Ingestion Status")
    table.add_column("Source", style="cyan")
    table.add_column("Documents", style="green")
    table.add_column("Last Sync", style="yellow")
    table.add_column("Status", style="magenta")
    
    # TODO: Get actual data from database
    table.add_row("File System", "0", "Never", "Not configured")
    table.add_row("Database", "0", "Never", "Not configured")
    table.add_row("Vector DB", "0", "Never", "Empty")
    
    console.print(table)
    
    console.print("\n[dim]Data status check completed[/dim]")
    console.print("[dim]Data status displayed[/dim]")


@data_group.command(name='validate')
@click.option('--database', 
              help='Specific database connection to validate (from configured RDB connections)')
@click.option('--detailed/--summary', default=False,
              help='Show detailed validation results or summary only')
def validate_data_config(database, detailed):
    """Validate RDB and pipeline configuration.
    
    Performs comprehensive validation of:
    - Database connections and configurations
    - Vector database (Milvus) connectivity
    - Embedding service availability
    - System resource requirements
    
    Examples:
        rag-cli data validate
        rag-cli data validate --database mydb --detailed
    """
    console.print("[yellow]Starting RDB pipeline configuration validation...[/yellow]")
    
    try:
        import asyncio
        from src.pipeline.rdb_config_validator import validate_rdb_system, test_database_connection
        from src.core.config import get_config
        from rich.table import Table
        from rich.panel import Panel
        
        async def run_validation():
            if database:
                # Validate specific database
                config = get_config()
                if not hasattr(config, 'rdb_connections') or database not in config.rdb_connections:
                    console.print(f"[red]✗ Database '{database}' not found in configured RDB connections[/red]")
                    return
                
                db_config = config.rdb_connections[database]
                console.print(f"[dim]Validating database connection: {database}[/dim]")
                report = await test_database_connection(db_config)
            else:
                # Validate entire system
                console.print("[dim]Validating complete RDB system...[/dim]")
                report = await validate_rdb_system()
            
            return report
        
        # Run validation
        report = asyncio.run(run_validation())
        
        if not report:
            console.print("[red]✗ Validation failed to complete[/red]")
            return
        
        # Display results
        status_style = "green" if report.overall_status else "red"
        status_text = "✓ PASSED" if report.overall_status else "✗ FAILED"
        
        console.print(f"\n[{status_style}]{status_text}[/{status_style}] Validation completed")
        
        # Summary table
        summary_table = Table(title="Validation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="yellow")
        
        for metric, count in report.summary.items():
            summary_table.add_row(metric.replace('_', ' ').title(), str(count))
        
        console.print(summary_table)
        
        # Show critical errors and errors
        if report.critical_errors:
            console.print("\n[red]Critical Errors:[/red]")
            for error in report.critical_errors:
                console.print(f"  [red]✗[/red] [{error.component}] {error.message}")
        
        if report.errors:
            console.print("\n[red]Errors:[/red]")
            for error in report.errors:
                console.print(f"  [red]✗[/red] [{error.component}] {error.message}")
        
        if report.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in report.warnings[:5]:  # Show first 5 warnings
                console.print(f"  [yellow]⚠[/yellow] [{warning.component}] {warning.message}")
            if len(report.warnings) > 5:
                console.print(f"  [dim]... and {len(report.warnings) - 5} more warnings[/dim]")
        
        # Detailed results if requested
        if detailed:
            console.print("\n[cyan]Detailed Results:[/cyan]")
            
            # Group results by component
            components = {}
            for result in report.results:
                if result.component not in components:
                    components[result.component] = []
                components[result.component].append(result)
            
            for component, results in components.items():
                detail_table = Table(title=f"Component: {component}")
                detail_table.add_column("Check", style="cyan")
                detail_table.add_column("Status", style="yellow")
                detail_table.add_column("Message", style="white")
                
                for result in results:
                    status_icon = "✓" if result.status else "✗"
                    status_color = "green" if result.status else ("red" if result.severity in ["critical", "error"] else "yellow")
                    
                    detail_table.add_row(
                        result.check_name,
                        f"[{status_color}]{status_icon}[/{status_color}]",
                        result.message
                    )
                
                console.print(detail_table)
                console.print()
        
        # Recommendations
        if not report.overall_status:
            console.print("\n[cyan]Recommendations:[/cyan]")
            if report.critical_errors:
                console.print("  • Fix critical errors first - these prevent the system from functioning")
            if report.errors:
                console.print("  • Address errors to ensure reliable operation")
            if report.warnings:
                console.print("  • Review warnings for potential performance or reliability issues")
        else:
            console.print(f"\n[green]✓ System validation passed! RDB pipeline is ready for use.[/green]")
    
    except ImportError as e:
        console.print(f"[red]✗ Missing dependencies for validation: {e}[/red]")
        console.print("[dim]Please ensure all required packages are installed[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        import traceback
        console.print(f"[dim]Error details: {traceback.format_exc()}[/dim]")


@data_group.command(name='cleanup')
@click.option('--orphaned/--no-orphaned', default=True,
              help='Clean up orphaned vectors (no source document).')
@click.option('--old-embeddings/--no-old-embeddings', default=False,
              help='Clean up old embedding versions.')
@click.option('--confirm/--no-confirm', default=True,
              help='Ask for confirmation before cleanup.')
def cleanup_data(orphaned, old_embeddings, confirm):
    """Clean up data and vectors.
    
    Removes orphaned vectors, old embeddings, and other
    unnecessary data to free up storage space.
    
    Examples:
        rag-cli data cleanup --orphaned
        rag-cli data cleanup --old-embeddings --no-confirm
    """
    console.print("[dim]Starting data cleanup[/dim]")
    
    console.print("[yellow]Preparing data cleanup...[/yellow]")
    
    cleanup_tasks = []
    if orphaned:
        cleanup_tasks.append("Remove orphaned vectors")
    if old_embeddings:
        cleanup_tasks.append("Remove old embedding versions")
    
    if not cleanup_tasks:
        console.print("[yellow]No cleanup tasks selected[/yellow]")
        return
    
    console.print(f"[dim]Cleanup tasks:[/dim]")
    for task in cleanup_tasks:
        console.print(f"  • {task}")
    
    if confirm:
        if not confirm_action("This operation cannot be undone. Continue with cleanup?"):
            console.print("[yellow]Data cleanup cancelled by user[/yellow]")
            return
    
    # TODO: Implement actual cleanup
    console.print("[red]✗ Data cleanup implementation not complete[/red]")
    console.print("[dim]Data cleanup completed (placeholder)[/dim]")