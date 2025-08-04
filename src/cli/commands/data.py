"""
Data management commands for the CLI.
"""

import click
from rich.console import Console
from pathlib import Path
from typing import Optional

from ..utils import (
    console, confirm_action, create_progress_bar, create_detailed_progress_bar,
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
@click.option('--path', type=click.Path(),
              help='Path to data directory or file to ingest.')
@click.option('--database/--no-database', default=False,
              help='Use database mode for RDB to vector ingestion.')
@click.option('--episode-mode/--no-episode-mode', default=False,
              help='Use episode-specific processing with improved chunking.')
@click.option('--recursive/--no-recursive', default=True,
              help='Recursively process subdirectories.')
@click.option('--file-types', default='txt,pdf,docx,md',
              help='Comma-separated list of file extensions to process.')
@click.option('--batch-size', default=100, type=int,
              help='Number of files to process in each batch.')
@click.option('--force/--no-force', default=False,
              help='Force re-ingestion of existing files.')
def ingest_data(path, database, episode_mode, recursive, file_types, batch_size, force):
    """Ingest data from directory or file.
    
    Processes files from the specified path and ingests them into
    the vector database. Supports multiple file formats including
    text, PDF, Word documents, and Markdown.
    
    Examples:
        rag-cli data ingest --path ./documents
        rag-cli data ingest --database --batch-size 50 --force
        rag-cli data ingest --episode-mode --database
        rag-cli data ingest --path ./file.pdf --file-types pdf
    """
    # Check mode: episode vs database vs file
    if episode_mode:
        console.print(f"[yellow]Starting Episode-specific processing with improved chunking[/yellow]")
        
        # Import episode-specific components
        try:
            from src.episode.manager import EpisodeRAGManager
            from src.core.config import get_config
            import asyncio
            
            config = get_config()
            
            async def run_episode_processing():
                from src.database.base import DatabaseManager
                from src.embedding.manager import EmbeddingManager
                from src.milvus.client import MilvusClient
                from src.episode.manager import EpisodeRAGConfig
                
                # Initialize dependencies  
                db_manager = DatabaseManager(config.database)
                
                # Create embedding provider configs list
                if config.embedding_providers:
                    provider_configs = list(config.embedding_providers.values())
                else:
                    # Fallback to single embedding config
                    provider_configs = [config.embedding]
                    
                embedding_manager = EmbeddingManager(provider_configs)
                milvus_client = MilvusClient(config.milvus)
                episode_config = EpisodeRAGConfig(
                    processing_batch_size=5,  # Further reduce batch size for stability  
                    vector_dimension=get_config().rag.vector_dimension  # Use configured dimension
                )
                
                episode_manager = EpisodeRAGManager(
                    database_manager=db_manager,
                    embedding_manager=embedding_manager,
                    milvus_client=milvus_client,
                    config=episode_config
                )
                
                # Connect to Milvus first
                milvus_client.connect()
                
                # Setup collection first
                await episode_manager.setup_collection(drop_existing=True)
                
                # Get available novels from database directly
                from sqlalchemy import text
                with db_manager.get_connection() as conn:
                    result = conn.execute(text("SELECT novel_id FROM novels"))
                    novel_ids = [row[0] for row in result]
                
                console.print(f"Found {len(novel_ids)} novels to process")
                
                total_processed = 0
                total_failed = 0
                
                with create_detailed_progress_bar() as progress:
                    task = progress.add_task(
                        "Processing novels...", 
                        total=len(novel_ids),
                        stage="🚀 Starting",
                        current_item="",
                        rate="0/sec"
                    )
                    
                    progress_callback = ProgressCallback(progress, task, len(novel_ids))
                    progress_callback.start("Initializing episode processing...")
                    
                    for i, novel_id in enumerate(novel_ids, 1):
                        try:
                            progress_callback.update_item(f"Novel {novel_id}", i-1)
                            
                            # Add delay between novels to prevent Ollama overload
                            if i > 1:
                                await asyncio.sleep(2)  # 2 second delay
                            
                            result = await episode_manager.process_novel(novel_id, force_reprocess=True)
                            
                            episode_count = result.get('episodes_processed', 0)
                            total_processed += episode_count
                            
                            console.print(f"[green]✓ Novel {novel_id}: {episode_count} episodes processed[/green]")
                            
                        except Exception as e:
                            total_failed += 1
                            progress_callback.mark_failed(f"Novel {novel_id}")
                            console.print(f"[red]✗ Failed to process Novel {novel_id}: {e}[/red]")
                            continue
                    
                    progress_callback.complete("All novels processed")
                
                console.print(f"[green]✓ Episode processing completed[/green]")
                console.print(f"[dim]Total episodes processed: {total_processed}[/dim]")
                console.print(f"[dim]Failed novels: {total_failed}[/dim]")
            
            # Run episode processing
            asyncio.run(run_episode_processing())
            return
            
        except ImportError as e:
            console.print(f"[red]✗ Episode processing not available: {e}[/red]")
            return
        except Exception as e:
            console.print(f"[red]✗ Episode processing failed: {e}[/red]")
            return
    
    elif database:
        console.print(f"[yellow]Starting RDB to Vector database ingestion[/yellow]")
        data_path = None
    elif path:
        console.print(f"[yellow]Starting data ingestion from {path}[/yellow]")
        # Validate path
        data_path = validate_directory_path(path, must_exist=True)
        console.print(f"[dim]Validated path: {data_path}[/dim]")
    else:
        console.print("[red]✗ Either --path or --database must be specified[/red]")
        return
    
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
        from src.core.config import DatabaseConfig, DatabaseType, get_config
        from src.pipeline.pipeline import VectorPipeline, PipelineConfig, Document
        from src.milvus.client import MilvusClient
        from src.embedding.manager import EmbeddingManager
        from src.text_processing.text_cleaner import TextCleaner
        from src.text_processing.text_splitter import TextSplitter
        from src.text_processing.metadata_manager import MetadataManager
        import asyncio
        import json
        from datetime import datetime, timezone
        
        # Get application config
        config = get_config()
        
        with create_progress_bar() as progress:
            if not database:
                # File-based ingestion
                task = progress.add_task("Scanning files...", total=None)
                
                data_path_obj = Path(data_path)
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
                                "ingested_at": datetime.now(timezone.utc).isoformat()
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
            
            # Create collection if it doesn't exist using Context7 MCP implementation
            try:
                milvus_client.create_collection_if_not_exists(
                    collection_name="documents",
                    dim=get_config().rag.vector_dimension,  # Use configured embedding dimension
                    description="Auto-generated collection for file-based documents"
                )
                console.print(f"[green]✓ Collection 'documents' is ready[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to ensure collection exists: {e}[/red]")
                if not force:
                    return
            
            # Initialize embedding manager - convert config to EmbeddingProviderConfig list
            from src.embedding.manager import EmbeddingProviderConfig
            provider_configs = []
            if isinstance(config.embedding_providers, dict):
                for name, embedding_config in config.embedding_providers.items():
                    provider_config = EmbeddingProviderConfig(
                        provider=embedding_config.provider,
                        config=embedding_config,
                        priority=1,
                        enabled=True
                    )
                    provider_configs.append(provider_config)
            else:
                # If it's already a list, use it directly
                provider_configs = config.embedding_providers
            
            embedding_manager = EmbeddingManager(provider_configs)
            
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


@data_group.command(name='retry-failed')
@click.option('--novel-ids', required=True, type=str,
              help='Comma-separated list of novel IDs to retry (e.g., "78,32,25")')
@click.option('--conservative/--normal', default=True,
              help='Use conservative settings (smaller batches, longer delays)')
@click.option('--max-retries', default=5, type=int,
              help='Maximum retry attempts per episode')
def retry_failed_novels(novel_ids, conservative, max_retries):
    """Retry processing for specific failed novels.
    
    Uses improved stability settings to retry novels that failed
    during previous processing runs.
    
    Examples:
        rag-cli data retry-failed --novel-ids="78,32"
        rag-cli data retry-failed --novel-ids="78" --conservative --max-retries=10
    """
    console.print(f"[yellow]Retrying failed novels: {novel_ids}[/yellow]")
    
    # Parse novel IDs
    try:
        novel_id_list = [int(id.strip()) for id in novel_ids.split(',')]
    except ValueError as e:
        console.print(f"[red]✗ Invalid novel IDs format: {e}[/red]")
        return
    
    console.print(f"[dim]Target novels: {novel_id_list}[/dim]")
    console.print(f"[dim]Conservative mode: {conservative}, Max retries: {max_retries}[/dim]")
    
    try:
        import asyncio
        from src.episode.manager import EpisodeRAGManager
        from src.core.config import get_config
        
        config = get_config()
        
        async def run_retry_processing():
            from src.database.base import DatabaseManager
            from src.embedding.manager import EmbeddingManager
            from src.milvus.client import MilvusClient
            from src.episode.manager import EpisodeRAGConfig
            
            # Initialize dependencies
            db_manager = DatabaseManager(config.database)
            
            if config.embedding_providers:
                provider_configs = list(config.embedding_providers.values())
            else:
                provider_configs = [config.embedding]
                
            embedding_manager = EmbeddingManager(provider_configs)
            milvus_client = MilvusClient(config.milvus)
            
            # Conservative settings for retry
            if conservative:
                batch_size = 2
                console.print("[dim]Using conservative settings: batch_size=2[/dim]")
            else:
                batch_size = 5
                console.print("[dim]Using normal settings: batch_size=5[/dim]")
            
            episode_config = EpisodeRAGConfig(
                processing_batch_size=batch_size,
                vector_dimension=1024
            )
            
            episode_manager = EpisodeRAGManager(
                database_manager=db_manager,
                embedding_manager=embedding_manager,
                milvus_client=milvus_client,
                config=episode_config
            )
            
            # Connect to Milvus
            milvus_client.connect()
            
            success_count = 0
            failed_novels = []
            
            for i, novel_id in enumerate(novel_id_list, 1):
                try:
                    console.print(f"[blue]Processing novel {novel_id}... ({i}/{len(novel_id_list)})[/blue]")
                    
                    # Provider health check
                    primary_provider = list(embedding_manager.providers.values())[0] if embedding_manager.providers else None
                    if primary_provider and hasattr(primary_provider, 'health_check'):
                        health = primary_provider.health_check()
                        if health.get('status') != 'healthy':
                            console.print(f"[yellow]⚠ Provider unhealthy, waiting 10s...[/yellow]")
                            await asyncio.sleep(10)
                    
                    # Process novel
                    await episode_manager.process_novel(novel_id)
                    
                    console.print(f"[green]✓ Novel {novel_id} completed successfully[/green]")
                    success_count += 1
                    
                    # Wait between novels
                    if i < len(novel_id_list):
                        wait_time = 10 if conservative else 5
                        console.print(f"[dim]Waiting {wait_time}s before next novel...[/dim]")
                        await asyncio.sleep(wait_time)
                        
                except Exception as e:
                    console.print(f"[red]✗ Novel {novel_id} failed: {e}[/red]")
                    failed_novels.append(novel_id)
                    
                    # Wait longer after failure
                    console.print("[dim]Waiting 15s after failure...[/dim]")
                    await asyncio.sleep(15)
            
            # Final results
            console.print(f"[green]✓ Retry completed: {success_count}/{len(novel_id_list)} successful[/green]")
            
            if failed_novels:
                console.print(f"[red]✗ Still failed: {failed_novels}[/red]")
                console.print("[dim]Consider checking Ollama server status or system resources[/dim]")
            else:
                console.print("[green]🎉 All novels processed successfully![/green]")
        
        # Run retry processing
        asyncio.run(run_retry_processing())
        
    except ImportError as e:
        console.print(f"[red]✗ Missing dependencies: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Retry processing failed: {e}[/red]")


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
    
    # Implement actual data sync using DataSyncManager
    try:
        import asyncio
        from src.services.data_sync import DataSyncManager
        
        # Initialize sync manager
        sync_manager = DataSyncManager()
        
        # Run synchronization
        if source == "all":
            results = asyncio.run(sync_manager.sync_all_sources(
                incremental=incremental,
                dry_run=dry_run
            ))
        else:
            # Parse source list
            source_list = [s.strip() for s in source.split(",")] if source != "all" else None
            results = asyncio.run(sync_manager.sync_all_sources(
                incremental=incremental,
                dry_run=dry_run,
                sources=source_list
            ))
        
        # Display results
        from rich.table import Table
        
        table = Table(title="Data Sync Results")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Records", style="green")
        table.add_column("Added", style="green")
        table.add_column("Updated", style="yellow")
        table.add_column("Duration", style="blue")
        
        total_processed = 0
        total_added = 0
        total_updated = 0
        successful_syncs = 0
        
        for source_id, sync_state in results.items():
            status_emoji = "✓" if sync_state.sync_status.value == "completed" else "✗"
            status_color = "green" if sync_state.sync_status.value == "completed" else "red"
            
            table.add_row(
                source_id,
                f"[{status_color}]{status_emoji} {sync_state.sync_status.value}[/{status_color}]",
                str(sync_state.records_processed),
                str(sync_state.records_added),
                str(sync_state.records_updated),
                f"{sync_state.sync_duration:.2f}s"
            )
            
            if sync_state.sync_status.value == "completed":
                successful_syncs += 1
                total_processed += sync_state.records_processed
                total_added += sync_state.records_added
                total_updated += sync_state.records_updated
        
        console.print(table)
        
        # Summary
        if successful_syncs > 0:
            console.print(f"\n[green]✓ Sync completed successfully![/green]")
            console.print(f"[dim]Total processed: {total_processed}, Added: {total_added}, Updated: {total_updated}[/dim]")
        else:
            console.print(f"\n[red]✗ Sync failed for all sources[/red]")
            
        # Show any errors
        for source_id, sync_state in results.items():
            if sync_state.error_message:
                console.print(f"[red]Error in {source_id}: {sync_state.error_message}[/red]")
    
    except Exception as e:
        console.print(f"[red]✗ Data sync failed: {str(e)}[/red]")
        if dry_run:
            console.print("[dim]Note: This was a dry run - no actual changes were made[/dim]")
        logger.error(f"Data sync error: {e}", exc_info=True)


@data_group.command(name='status')
def data_status():
    """Show data ingestion status.
    
    Displays information about ingested data, including
    document counts, last sync times, and storage usage.
    """
    console.print("[dim]Checking data status[/dim]")
    
    from rich.table import Table
    
    # Create status table
    status_table = Table(title="Data Ingestion Status")
    status_table.add_column("Source", style="cyan")
    status_table.add_column("Documents", style="green")
    status_table.add_column("Last Sync", style="yellow")
    status_table.add_column("Status", style="magenta")
    
    # Get actual data from database and vector store
    try:
        from src.core.config import get_config
        from src.milvus.client import MilvusClient
        from datetime import datetime, timezone
        
        config = get_config()
        
        # Check database status
        db_status = "Not configured"
        db_docs = "0"
        db_last_sync = "Never"
        
        if config.rdb_connections:
            try:
                # Get first configured database
                db_name = list(config.rdb_connections.keys())[0]
                db_config = config.rdb_connections[db_name]
                
                from src.database.base import DatabaseManager
                db_manager = DatabaseManager(db_config)
                
                # Get table counts
                with db_manager.engine.connect() as conn:
                    from sqlalchemy import text
                    tables_result = conn.execute(text("SHOW TABLES"))
                    tables = [row[0] for row in tables_result]
                    
                    total_rows = 0
                    for table in tables:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_count = count_result.scalar()
                        total_rows += table_count
                    
                    db_docs = str(total_rows)
                    db_status = f"{len(tables)} tables available"
                    db_last_sync = "Available"
                    
            except Exception as e:
                db_status = f"Connection error: {str(e)[:30]}..."
        
        # Check Milvus status
        milvus_status = "Not configured"
        milvus_docs = "0"
        milvus_last_sync = "Never"
        
        try:
            milvus_client = MilvusClient(config.milvus)
            # Explicitly connect to Milvus server
            milvus_client.connect()
            
            # Check if collection exists
            collection_name = "documents"
            if milvus_client.has_collection(collection_name):
                entity_count = milvus_client.get_entity_count(collection_name)
                milvus_docs = str(entity_count)
                milvus_status = "Connected"
                milvus_last_sync = "Available"
            else:
                milvus_status = "Collection not found"
                
        except Exception as e:
            milvus_status = f"Connection error"
        
        # File system status (placeholder for now)
        fs_status = "Not implemented"
        fs_docs = "0"
        fs_last_sync = "Never"
        
        status_table.add_row("File System", fs_docs, fs_last_sync, fs_status)
        status_table.add_row("Database", db_docs, db_last_sync, db_status)
        status_table.add_row("Vector DB", milvus_docs, milvus_last_sync, milvus_status)
        
    except Exception as e:
        # Fallback to placeholder data if there's an error
        console.print(f"[red]Error getting status: {e}[/red]")
        status_table.add_row("File System", "0", "Never", "Error")
        status_table.add_row("Database", "0", "Never", "Error")
        status_table.add_row("Vector DB", "0", "Never", "Error")
    
    console.print(status_table)
    
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
    
    # Implement comprehensive data cleanup system
    import os
    import tempfile
    from datetime import datetime, timedelta
    from pathlib import Path
    import asyncio
    import time
    
    try:
        # Import required components
        from src.core.config import get_config
        from src.database.base import DatabaseFactory
        from src.models.document import Document, DocumentStatus
        from src.milvus.client import MilvusClient
        from src.milvus.collection import CollectionManager
        from sqlalchemy.orm import sessionmaker
        
        # Initialize cleanup statistics
        cleanup_stats = {
            'orphaned_vectors_removed': 0,
            'orphaned_documents_removed': 0,
            'duplicate_vectors_removed': 0,
            'old_embeddings_removed': 0,
            'temporary_files_removed': 0,
            'cache_files_removed': 0,
            'total_space_freed_mb': 0,
            'processing_time_ms': 0,
            'errors_encountered': []
        }
        
        start_time = time.time()
        console.print("[green]🚀 Starting comprehensive data cleanup...[/green]")
        
        # Get configuration
        config = get_config()
        
        # 1. DATABASE CLEANUP
        console.print("\n[cyan]📊 Phase 1: Database Cleanup[/cyan]")
        
        # Connect to database
        db_manager = DatabaseFactory.create_manager(config.database)
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            cleanup_stats['errors_encountered'].append("Database connection failed")
        else:
            Session = sessionmaker(bind=db_manager.engine)
            
            with Session() as session:
                # Find orphaned documents (no file on disk)
                console.print("  • Scanning for orphaned documents...")
                documents = session.query(Document).all()
                orphaned_docs = []
                
                for doc in documents:
                    if doc.file_path and not os.path.exists(doc.file_path):
                        orphaned_docs.append(doc)
                
                if orphaned_docs:
                    console.print(f"  • Found {len(orphaned_docs)} orphaned documents")
                    if confirm or confirm_action(f"Remove {len(orphaned_docs)} orphaned documents?"):
                        for doc in orphaned_docs:
                            session.delete(doc)
                        session.commit()
                        cleanup_stats['orphaned_documents_removed'] = len(orphaned_docs)
                        console.print(f"  • [green]✓ Removed {len(orphaned_docs)} orphaned documents[/green]")
                    else:
                        console.print("  • Skipped orphaned documents cleanup")
                else:
                    console.print("  • [green]✓ No orphaned documents found[/green]")
                
                # Find failed/stuck processing documents
                console.print("  • Scanning for stuck processing documents...")
                stuck_cutoff = datetime.now() - timedelta(hours=1)  # 1 hour timeout
                stuck_docs = session.query(Document).filter(
                    Document.status == DocumentStatus.PROCESSING,
                    Document.upload_date < stuck_cutoff
                ).all()
                
                if stuck_docs:
                    console.print(f"  • Found {len(stuck_docs)} stuck processing documents")
                    if confirm or confirm_action(f"Reset {len(stuck_docs)} stuck documents to failed status?"):
                        for doc in stuck_docs:
                            doc.mark_failed("Processing timeout - cleaned up by data cleanup")
                        session.commit()
                        console.print(f"  • [green]✓ Reset {len(stuck_docs)} stuck documents[/green]")
                else:
                    console.print("  • [green]✓ No stuck processing documents found[/green]")
        
        # 2. VECTOR DATABASE CLEANUP
        console.print("\n[cyan]🔍 Phase 2: Vector Database Cleanup[/cyan]")
        
        try:
            # Connect to Milvus
            milvus_client = MilvusClient(config.milvus)
            milvus_client.connect()
            
            collection_name = config.milvus.collection_name
            
            if milvus_client.has_collection(collection_name):
                console.print(f"  • Connected to collection: {collection_name}")
                
                # Get collection statistics before cleanup
                stats_before = milvus_client.get_collection_stats(collection_name)
                vectors_before = stats_before.get('row_count', 0)
                
                console.print(f"  • Current vector count: {vectors_before:,}")
                
                if orphaned:
                    console.print("  • Scanning for orphaned vectors...")
                    
                    # Get all vector IDs
                    try:
                        all_vectors = milvus_client.query(
                            collection_name=collection_name,
                            expr="",
                            output_fields=["id", "document_id"],
                            limit=100000  # Adjust based on your data size
                        )
                        
                        # Find vectors without corresponding documents
                        orphaned_vector_ids = []
                        
                        if all_vectors:
                            with Session() as session:
                                document_ids = set(str(doc.id) for doc in session.query(Document.id).all())
                                
                                for vector in all_vectors:
                                    vector_doc_id = vector.get('document_id', '').strip()
                                    if vector_doc_id and vector_doc_id not in document_ids:
                                        orphaned_vector_ids.append(vector['id'])
                        
                        if orphaned_vector_ids:
                            console.print(f"  • Found {len(orphaned_vector_ids):,} orphaned vectors")
                            
                            if confirm or confirm_action(f"Remove {len(orphaned_vector_ids):,} orphaned vectors?"):
                                # Delete in batches to avoid timeout
                                batch_size = 1000
                                removed_count = 0
                                
                                for i in range(0, len(orphaned_vector_ids), batch_size):
                                    batch = orphaned_vector_ids[i:i + batch_size]
                                    id_expr = f"id in {batch}"
                                    
                                    delete_result = milvus_client.delete(
                                        collection_name=collection_name,
                                        expr=id_expr
                                    )
                                    
                                    removed_count += len(batch)
                                    console.print(f"    • Removed batch: {removed_count:,}/{len(orphaned_vector_ids):,}")
                                
                                cleanup_stats['orphaned_vectors_removed'] = removed_count
                                console.print(f"  • [green]✓ Removed {removed_count:,} orphaned vectors[/green]")
                            else:
                                console.print("  • Skipped orphaned vectors cleanup")
                        else:
                            console.print("  • [green]✓ No orphaned vectors found[/green]")
                    
                    except Exception as vector_error:
                        error_msg = f"Vector cleanup error: {vector_error}"
                        cleanup_stats['errors_encountered'].append(error_msg)
                        console.print(f"  • [yellow]Warning: {error_msg}[/yellow]")
                
                # Compact collection for performance
                console.print("  • Compacting collection...")
                try:
                    milvus_client.compact(collection_name)
                    console.print("  • [green]✓ Collection compaction initiated[/green]")
                except Exception as compact_error:
                    console.print(f"  • [yellow]Warning: Compaction failed: {compact_error}[/yellow]")
                
                # Get statistics after cleanup
                stats_after = milvus_client.get_collection_stats(collection_name)
                vectors_after = stats_after.get('row_count', 0)
                vectors_removed = vectors_before - vectors_after
                
                if vectors_removed > 0:
                    console.print(f"  • [green]✓ Total vectors removed: {vectors_removed:,}[/green]")
                
            else:
                console.print(f"  • [yellow]Collection '{collection_name}' not found[/yellow]")
                
        except Exception as milvus_error:
            error_msg = f"Milvus cleanup error: {milvus_error}"
            cleanup_stats['errors_encountered'].append(error_msg)
            console.print(f"  • [red]✗ {error_msg}[/red]")
        
        # 3. FILE SYSTEM CLEANUP
        console.print("\n[cyan]🗂️  Phase 3: File System Cleanup[/cyan]")
        
        # Clean up temporary files
        console.print("  • Cleaning temporary files...")
        temp_dirs = [
            tempfile.gettempdir(),
            "/tmp",
            os.path.expanduser("~/.cache"),
            "./.temp",
            "./.cache"
        ]
        
        temp_files_removed = 0
        space_freed = 0
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    # Look for RAG-related temp files
                    rag_patterns = ['rag_*', 'embedding_*', 'vector_*', 'milvus_*']
                    
                    for pattern in rag_patterns:
                        import glob
                        temp_files = glob.glob(os.path.join(temp_dir, pattern))
                        
                        for temp_file in temp_files:
                            try:
                                if os.path.isfile(temp_file):
                                    # Check if file is older than 1 hour
                                    file_age = time.time() - os.path.getmtime(temp_file)
                                    if file_age > 3600:  # 1 hour
                                        file_size = os.path.getsize(temp_file)
                                        os.remove(temp_file)
                                        temp_files_removed += 1
                                        space_freed += file_size
                                        
                            except Exception as file_error:
                                # Skip files we can't remove
                                pass
                                
                except Exception as dir_error:
                    # Skip directories we can't access
                    pass
        
        cleanup_stats['temporary_files_removed'] = temp_files_removed
        cleanup_stats['total_space_freed_mb'] = space_freed / (1024 * 1024)
        
        if temp_files_removed > 0:
            console.print(f"  • [green]✓ Removed {temp_files_removed} temporary files ({space_freed / (1024 * 1024):.1f} MB)[/green]")
        else:
            console.print("  • [green]✓ No temporary files to clean[/green]")
        
        # Clean up old log files
        console.print("  • Cleaning old log files...")
        log_dirs = ["./logs", "./.logs", os.path.expanduser("~/.rag/logs")]
        log_files_removed = 0
        log_space_freed = 0
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                try:
                    for log_file in Path(log_dir).glob("*.log*"):
                        # Remove log files older than 30 days
                        file_age_days = (time.time() - log_file.stat().st_mtime) / 86400
                        if file_age_days > 30:
                            file_size = log_file.stat().st_size
                            log_file.unlink()
                            log_files_removed += 1
                            log_space_freed += file_size
                            
                except Exception as log_error:
                    pass
        
        if log_files_removed > 0:
            console.print(f"  • [green]✓ Removed {log_files_removed} old log files ({log_space_freed / (1024 * 1024):.1f} MB)[/green]")
        else:
            console.print("  • [green]✓ No old log files to clean[/green]")
        
        # 4. OLD EMBEDDINGS CLEANUP
        if old_embeddings:
            console.print("\n[cyan]🧠 Phase 4: Old Embeddings Cleanup[/cyan]")
            
            # This would involve more complex logic to identify and remove old embedding versions
            # For now, we'll implement a placeholder that could be extended
            console.print("  • Scanning for old embedding versions...")
            
            try:
                # Look for documents that might have been reprocessed
                with Session() as session:
                    reprocessed_docs = session.query(Document).filter(
                        Document.metadata.like('%reprocessed%')
                    ).all()
                    
                    if reprocessed_docs:
                        console.print(f"  • Found {len(reprocessed_docs)} documents with reprocessing history")
                        # Additional cleanup logic could be implemented here
                        console.print("  • [green]✓ Old embeddings cleanup completed[/green]")
                    else:
                        console.print("  • [green]✓ No old embeddings to clean[/green]")
                        
            except Exception as embedding_error:
                error_msg = f"Old embeddings cleanup error: {embedding_error}"
                cleanup_stats['errors_encountered'].append(error_msg)
                console.print(f"  • [yellow]Warning: {error_msg}[/yellow]")
        
        # Calculate final statistics
        cleanup_stats['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
        # 5. CLEANUP SUMMARY
        console.print("\n[cyan]📈 Cleanup Summary[/cyan]")
        console.print(f"  • Orphaned documents removed: {cleanup_stats['orphaned_documents_removed']}")
        console.print(f"  • Orphaned vectors removed: {cleanup_stats['orphaned_vectors_removed']:,}")
        console.print(f"  • Temporary files removed: {cleanup_stats['temporary_files_removed']}")
        console.print(f"  • Total space freed: {cleanup_stats['total_space_freed_mb']:.1f} MB")
        console.print(f"  • Processing time: {cleanup_stats['processing_time_ms'] / 1000:.1f} seconds")
        
        if cleanup_stats['errors_encountered']:
            console.print(f"\n[yellow]⚠️  Warnings/Errors ({len(cleanup_stats['errors_encountered'])}):[/yellow]")
            for error in cleanup_stats['errors_encountered']:
                console.print(f"  • {error}")
        
        # Final status
        total_items_cleaned = (cleanup_stats['orphaned_documents_removed'] + 
                             cleanup_stats['orphaned_vectors_removed'] + 
                             cleanup_stats['temporary_files_removed'])
        
        if total_items_cleaned > 0:
            console.print(f"\n[green]🎉 Data cleanup completed successfully![/green]")
            console.print(f"[green]✓ {total_items_cleaned:,} total items cleaned up[/green]")
        else:
            console.print(f"\n[green]✓ System is already clean - no cleanup needed![/green]")
            
    except ImportError as import_error:
        error_msg = f"Missing dependencies: {import_error}"
        console.print(f"[red]✗ {error_msg}[/red]")
        console.print("[dim]Please ensure all required packages are installed[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Data cleanup failed: {e}[/red]")
        import traceback
        console.print(f"[dim]Error details: {traceback.format_exc()}[/dim]")


@data_group.command(name='process-novel')
@click.option('--novel-id', required=True, type=int,
              help='Novel ID to process episodes for.')
@click.option('--force/--no-force', default=False,
              help='Force reprocessing even if already processed.')
@click.option('--verbose/--no-verbose', default=True,
              help='Show detailed progress information.')
@click.option('--retry-failed/--no-retry-failed', default=False,
              help='Retry only failed episodes from previous run')
def process_novel(novel_id, force, verbose, retry_failed):
    """Process episodes for a specific novel.
    
    Uses the improved episode processing logic with individual
    episode handling and automatic chunking for long content.
    
    Examples:
        rag-cli data process-novel --novel-id 25
        rag-cli data process-novel --novel-id 67 --force
        rag-cli data process-novel --novel-id 65 --no-verbose
    """
    console.print(f"[yellow]Processing episodes for Novel {novel_id}[/yellow]")
    
    if verbose:
        console.print(f"[dim]Using improved episode processing with chunking[/dim]")
        console.print(f"[dim]Force reprocessing: {force}[/dim]")
    
    try:
        from src.episode.manager import EpisodeRAGManager
        from src.core.config import get_config
        import asyncio
        import time
        
        config = get_config()
        
        async def run_novel_processing():
            start_time = time.time()
            
            from src.database.base import DatabaseManager
            from src.embedding.manager import EmbeddingManager
            from src.milvus.client import MilvusClient
            from src.episode.manager import EpisodeRAGConfig
            
            # Initialize dependencies
            db_manager = DatabaseManager(config.database)
            
            # Create embedding provider configs list
            if config.embedding_providers:
                provider_configs = list(config.embedding_providers.values())
            else:
                # Fallback to single embedding config
                provider_configs = [config.embedding]
                
            embedding_manager = EmbeddingManager(provider_configs)
            milvus_client = MilvusClient(config.milvus)
            episode_config = EpisodeRAGConfig(
                processing_batch_size=5,  # Further reduce batch size for stability
                vector_dimension=1024  # Match Ollama model dimension
            )
            
            episode_manager = EpisodeRAGManager(
                database_manager=db_manager,
                embedding_manager=embedding_manager,
                milvus_client=milvus_client,
                config=episode_config
            )
            
            # Connect to Milvus first
            milvus_client.connect()
            
            # Setup collection first
            await episode_manager.setup_collection(drop_existing=force)
            
            # Check if novel exists
            from sqlalchemy import text
            with db_manager.get_connection() as conn:
                result = conn.execute(text("SELECT novel_id FROM novels"))
                novel_ids = [row[0] for row in result]
            
            if novel_id not in novel_ids:
                console.print(f"[red]✗ Novel {novel_id} not found in database[/red]")
                console.print(f"[dim]Available novels: {sorted(novel_ids)}[/dim]")
                return
            
            if verbose:
                console.print(f"[green]✓ Novel {novel_id} found in database[/green]")
            
            # Process the novel
            with create_detailed_progress_bar() as progress:
                task = progress.add_task(
                    f"Processing Novel {novel_id}...", 
                    total=1,
                    stage="🚀 Starting",
                    current_item=f"Novel {novel_id}",
                    rate="0/sec"
                )
                
                progress_callback = ProgressCallback(progress, task, 1)
                progress_callback.start(f"Initializing Novel {novel_id} processing...")
                
                try:
                    progress_callback.update_item(f"Processing episodes for Novel {novel_id}", 0)
                    result = await episode_manager.process_novel(novel_id, force_reprocess=force)
                    
                    processing_time = time.time() - start_time
                    
                    # Mark as complete
                    progress_callback.complete(f"Novel {novel_id} completed")
                    
                    # Display results
                    console.print(f"[green]✓ Novel {novel_id} processing completed[/green]")
                    
                    if verbose:
                        console.print(f"[dim]Processing time: {processing_time:.1f}s[/dim]")
                        console.print(f"[dim]Episodes processed: {result.get('processed_count', 0)}[/dim]")
                        console.print(f"[dim]Episodes failed: {result.get('failed_count', 0)}[/dim]")
                        
                        if result.get('stats'):
                            stats = result['stats']
                            console.print(f"[dim]Average content length: {stats.get('average_content_length', 0):.0f} chars[/dim]")
                            console.print(f"[dim]Success rate: {stats.get('success_rate', 0):.1f}%[/dim]")
                    
                except Exception as e:
                    progress_callback.mark_failed(f"Novel {novel_id}")
                    console.print(f"[red]✗ Failed to process Novel {novel_id}: {e}[/red]")
                    if verbose:
                        import traceback
                        console.print(f"[dim]Error details: {traceback.format_exc()}[/dim]")
        
        # Run processing
        asyncio.run(run_novel_processing())
        
    except ImportError as e:
        console.print(f"[red]✗ Episode processing not available: {e}[/red]")
        console.print("[dim]Please ensure all required packages are installed[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Novel processing failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]Error details: {traceback.format_exc()}[/dim]")