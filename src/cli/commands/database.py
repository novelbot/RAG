"""
Database management commands for the CLI.
"""

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy import text
from pathlib import Path
import subprocess
import json
from datetime import datetime

from ..utils import (
    console, confirm_action, create_progress_bar
)
from src.core.config import get_config
from src.database.base import DatabaseFactory
from src.auth.models import User, Role, Permission, UserRole, RolePermission
from src.core.exceptions import DatabaseError

console = Console()


@click.group()
def database_group():
    """Database management commands.
    
    Commands for managing database connections, schema, 
    migrations, and maintenance operations.
    """
    pass


@database_group.command(name='init')
@click.option('--force/--no-force', default=False,
              help='Force initialization even if database already exists.')
@click.option('--sqlite/--no-sqlite', default=False,
              help='Initialize SQLite databases using schema files.')
@click.argument('databases', nargs=-1, required=False,
                type=click.Choice(['auth', 'metrics', 'conversations', 'user_data']))
def init_database(force, sqlite, databases):
    """Initialize the database schema.
    
    Creates all necessary tables and indexes for the RAG server.
    This should be run once when setting up a new instance.
    
    For SQLite databases, you can initialize specific databases:
    - auth: Authentication database
    - metrics: Metrics tracking database  
    - conversations: Conversation history database
    - user_data: User data storage
    
    Examples:
        rag-cli database init
        rag-cli database init --force
        rag-cli database init --sqlite                    # Initialize all SQLite DBs
        rag-cli database init --sqlite auth metrics       # Initialize specific DBs
        rag-cli database init --sqlite --force           # Force reinitialize
    """
    # Handle SQLite initialization
    if sqlite:
        _init_sqlite_databases(databases, force)
        return
    console.print("[yellow]Initializing database schema...[/yellow]")
    
    try:
        # Get configuration
        config = get_config()
        console.print(f"[dim]Connecting to {config.database.driver} database at {config.database.host}:{config.database.port}[/dim]")
        
        # Create database manager
        db_manager = DatabaseFactory.create_manager(config.database)
        
        # Test connection first
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        # Check if tables already exist
        existing_tables = []
        with db_manager.get_connection() as conn:
            try:
                result = conn.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """))
                existing_tables = [row[0] for row in result]
            except Exception:
                # Handle case where information_schema doesn't exist (SQLite)
                pass
        
        if existing_tables and not force:
            console.print(f"[yellow]Database already contains {len(existing_tables)} tables:[/yellow]")
            for table in existing_tables[:5]:  # Show first 5 tables
                console.print(f"  • {table}")
            if len(existing_tables) > 5:
                console.print(f"  ... and {len(existing_tables) - 5} more")
            
            if not confirm_action("This will create/update database tables. Continue?"):
                console.print("[yellow]Database initialization cancelled by user[/yellow]")
                return
        
        with create_progress_bar() as progress:
            # Create tables
            task = progress.add_task("Creating tables...", total=None)
            
            try:
                # Import all models to ensure they're registered
                from src.auth.models import Base
                
                # Create all tables
                Base.metadata.create_all(db_manager.engine)
                console.print("[green]✓ Database tables created successfully[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Failed to create tables: {e}[/red]")
                return
            
            progress.update(task, description="Setting up default roles and permissions...")
            
            # Create default roles and permissions
            try:
                _create_default_roles_and_permissions(db_manager)
                console.print("[green]✓ Default roles and permissions created[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Warning: Could not create default roles: {e}[/yellow]")
            
            progress.update(task, description="Creating admin user...")
            
            # Create default admin user if it doesn't exist
            try:
                _create_default_admin_user(db_manager)
                console.print("[green]✓ Admin user setup completed[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Warning: Could not create admin user: {e}[/yellow]")
            
            progress.remove_task(task)
        
        console.print("[green]✅ Database initialization completed successfully[/green]")
        
    except DatabaseError as e:
        console.print(f"[red]✗ Database error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Unexpected error during database initialization: {e}[/red]")


@database_group.command(name='migrate')
@click.option('--target', help='Target migration version (latest if not specified).')
@click.option('--dry-run/--no-dry-run', default=False,
              help='Show migrations that would be applied without running them.')
def migrate_database(target, dry_run):
    """Run database migrations.
    
    Applies pending database migrations to update the schema
    to the latest version or a specific target version.
    
    Examples:
        rag-cli database migrate
        rag-cli database migrate --target 001_initial
        rag-cli database migrate --dry-run
    """
    console.print("[yellow]Checking for pending migrations...[/yellow]")
    
    if dry_run:
        console.print("[dim]Running in dry-run mode - no changes will be made[/dim]")
    
    try:
        # Get configuration
        config = get_config() 
        
        # Look for alembic.ini file
        alembic_ini_path = Path("alembic.ini")
        if not alembic_ini_path.exists():
            console.print("[red]✗ alembic.ini not found. Please initialize Alembic first.[/red]")
            console.print("[dim]Run 'alembic init alembic' to set up migrations[/dim]")
            return
        
        # Import Alembic components
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command
        from alembic.script import ScriptDirectory
        from alembic.migration import MigrationContext
        
        # Configure Alembic
        alembic_cfg = AlembicConfig(str(alembic_ini_path))
        
        # Create database manager to get connection
        db_manager = DatabaseFactory.create_manager(config.database)
        
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        # Get current and target revisions
        with db_manager.get_connection() as conn:
            context = MigrationContext.configure(conn)
            current_rev = context.get_current_revision()
            
            script_dir = ScriptDirectory.from_config(alembic_cfg)
            
            # Determine target revision
            if target:
                target_rev = target
            else:
                target_rev = script_dir.get_current_head()
            
            console.print(f"[dim]Current revision: {current_rev or 'None'}[/dim]")
            console.print(f"[dim]Target revision: {target_rev}[/dim]")
            
            # Check if migration is needed
            if current_rev == target_rev:
                console.print("[green]✓ Database is already up to date[/green]")
                return
            
            # Get migration path
            try:
                migration_steps = []
                for rev in script_dir.walk_revisions(target_rev, current_rev):
                    migration_steps.append(rev)
                
                if not migration_steps:
                    console.print("[green]✓ No pending migrations found[/green]")
                    return
                
                console.print(f"[yellow]Found {len(migration_steps)} pending migration(s):[/yellow]")
                
                # Show pending migrations
                for i, rev in enumerate(reversed(migration_steps), 1):
                    console.print(f"  {i}. {rev.revision[:8]} - {rev.doc or 'No description'}")
                
                if dry_run:
                    # Generate SQL for dry run
                    console.print("\n[yellow]Generated SQL (dry-run):[/yellow]")
                    
                    try:
                        # Capture SQL output
                        from io import StringIO
                        import sys
                        
                        # Redirect stdout to capture SQL
                        old_stdout = sys.stdout
                        sql_output = StringIO()
                        sys.stdout = sql_output
                        
                        try:
                            # Generate SQL without executing
                            alembic_command.upgrade(alembic_cfg, target_rev, sql=True)
                            sql_content = sql_output.getvalue()
                            
                            if sql_content.strip():
                                console.print(f"[dim]{sql_content}[/dim]")
                            else:
                                console.print("[dim]No SQL generated[/dim]")
                                
                        finally:
                            sys.stdout = old_stdout
                            
                    except Exception as e:
                        console.print(f"[red]✗ Failed to generate SQL: {e}[/red]")
                    
                    console.print("[green]✓ Dry-run completed[/green]")
                    
                else:
                    # Confirm before proceeding
                    if not confirm_action("Apply these migrations?"):
                        console.print("[yellow]Migration cancelled by user[/yellow]")
                        return
                    
                    # Apply migrations
                    with create_progress_bar() as progress:
                        task = progress.add_task("Applying migrations...", total=None)
                        
                        try:
                            # Run the migration
                            alembic_command.upgrade(alembic_cfg, target_rev)
                            console.print("[green]✓ Migrations applied successfully[/green]")
                            
                        except Exception as e:
                            console.print(f"[red]✗ Migration failed: {e}[/red]")
                            console.print("[yellow]⚠ Database may be in an inconsistent state[/yellow]")
                            return
                        finally:
                            progress.remove_task(task)
                    
                    # Verify final state
                    with db_manager.get_connection() as conn:
                        context = MigrationContext.configure(conn)
                        final_rev = context.get_current_revision()
                        console.print(f"[dim]Final revision: {final_rev}[/dim]")
                        
                        if final_rev == target_rev:
                            console.print("[green]✅ Migration completed successfully[/green]")
                        else:
                            console.print("[yellow]⚠ Migration may not have completed fully[/yellow]")
                            
            except Exception as e:
                console.print(f"[red]✗ Failed to analyze migrations: {e}[/red]")
                return
                
    except ImportError as e:
        console.print(f"[red]✗ Alembic not available: {e}[/red]")
        console.print("[dim]Install with: uv add alembic[/dim]")
    except Exception as e:
        console.print(f"[red]✗ Migration failed: {e}[/red]")


@database_group.command(name='backup')
@click.option('--output', type=click.Path(), help='Output file path for backup.')
@click.option('--compress/--no-compress', default=True,
              help='Compress the backup file.')
@click.option('--include-data/--schema-only', default=True,
              help='Include data in backup or schema only.')
def backup_database(output, compress, include_data):
    """Create a database backup.
    
    Creates a backup of the database schema and optionally data.
    The backup can be restored using the restore command.
    
    Examples:
        rag-cli database backup
        rag-cli database backup --output backup.sql
        rag-cli database backup --schema-only --no-compress
    """
    try:
        # Get configuration
        config = get_config()
        
        # Generate output filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = ".sql.gz" if compress else ".sql"
            output = f"rag_backup_{timestamp}{extension}"
        
        console.print(f"[yellow]Creating database backup: {output}[/yellow]")
        
        backup_type = "Full backup" if include_data else "Schema only"
        console.print(f"[dim]{backup_type}, Compressed: {compress}[/dim]")
        
        # Test database connection first
        db_manager = DatabaseFactory.create_manager(config.database)
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        with create_progress_bar() as progress:
            task = progress.add_task("Creating backup...", total=None)
            
            backup_success = False
            temp_file = None
            
            try:
                # Handle different database drivers
                driver = config.database.driver.lower()
                
                if driver.startswith('postgresql'):
                    backup_success = _backup_postgresql(config, output, compress, include_data, progress, task)
                elif driver.startswith('mysql') or driver.startswith('mariadb'):
                    backup_success = _backup_mysql(config, output, compress, include_data, progress, task)
                elif driver.startswith('sqlite'):
                    backup_success = _backup_sqlite(config, output, compress, include_data, progress, task)
                else:
                    console.print(f"[red]✗ Backup not supported for database driver: {driver}[/red]")
                    return
                
                if backup_success:
                    # Verify backup file was created
                    output_path = Path(output)
                    if output_path.exists():
                        file_size = output_path.stat().st_size
                        console.print(f"[green]✓ Backup created successfully[/green]")
                        console.print(f"[dim]File: {output} ({file_size:,} bytes)[/dim]")
                    else:
                        console.print("[red]✗ Backup file was not created[/red]")
                else:
                    console.print("[red]✗ Backup failed[/red]")
                    
            except Exception as e:
                console.print(f"[red]✗ Backup failed: {e}[/red]")
            finally:
                progress.remove_task(task)
                
    except Exception as e:
        console.print(f"[red]✗ Failed to create backup: {e}[/red]")


def _backup_postgresql(config, output, compress, include_data, progress, task):
    """Create PostgreSQL backup using pg_dump."""
    try:
        progress.update(task, description="Running pg_dump...")
        
        # Build pg_dump command
        cmd = ['pg_dump']
        
        # Connection parameters
        cmd.extend(['-h', config.database.host])
        cmd.extend(['-p', str(config.database.port)])
        cmd.extend(['-U', config.database.user])
        cmd.extend(['-d', config.database.name])
        
        # Backup options
        if not include_data:
            cmd.append('--schema-only')
        
        # Environment for password
        env = subprocess.os.environ.copy()
        if config.database.password:
            env['PGPASSWORD'] = config.database.password
        
        # Execute backup
        if compress and output.endswith('.gz'):
            # Create uncompressed backup first, then compress
            temp_output = output[:-3]  # Remove .gz extension
            with open(temp_output, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                      env=env, text=True, timeout=300)
            
            if result.returncode == 0:
                # Compress the file
                import gzip
                with open(temp_output, 'rb') as f_in:
                    with gzip.open(output, 'wb') as f_out:
                        f_out.writelines(f_in)
                Path(temp_output).unlink()  # Remove temp file
                return True
        else:
            # Direct output to file
            with open(output, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                      env=env, text=True, timeout=300)
            
            if result.returncode == 0:
                return True
        
        # Handle errors
        if result.stderr:
            console.print(f"[red]pg_dump error: {result.stderr}[/red]")
        return False
        
    except subprocess.TimeoutExpired:
        console.print("[red]✗ Backup timed out[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ pg_dump not found. Please install PostgreSQL client tools.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ PostgreSQL backup failed: {e}[/red]")
        return False


def _backup_mysql(config, output, compress, include_data, progress, task):
    """Create MySQL/MariaDB backup using mysqldump."""
    try:
        progress.update(task, description="Running mysqldump...")
        
        # Build mysqldump command
        cmd = ['mysqldump']
        
        # Connection parameters
        cmd.extend(['-h', config.database.host])
        cmd.extend(['-P', str(config.database.port)])
        cmd.extend(['-u', config.database.user])
        
        if config.database.password:
            cmd.extend(['-p' + config.database.password])
        
        # Backup options
        if not include_data:
            cmd.append('--no-data')
        
        cmd.append(config.database.name)
        
        # Execute backup
        if compress and output.endswith('.gz'):
            # Create uncompressed backup first, then compress
            temp_output = output[:-3]  # Remove .gz extension
            with open(temp_output, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                      text=True, timeout=300)
            
            if result.returncode == 0:
                # Compress the file
                import gzip
                with open(temp_output, 'rb') as f_in:
                    with gzip.open(output, 'wb') as f_out:
                        f_out.writelines(f_in)
                Path(temp_output).unlink()  # Remove temp file
                return True
        else:
            # Direct output to file
            with open(output, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                      text=True, timeout=300)
            
            if result.returncode == 0:
                return True
        
        # Handle errors
        if result.stderr:
            console.print(f"[red]mysqldump error: {result.stderr}[/red]")
        return False
        
    except subprocess.TimeoutExpired:
        console.print("[red]✗ Backup timed out[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ mysqldump not found. Please install MySQL client tools.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ MySQL backup failed: {e}[/red]")
        return False


def _backup_sqlite(config, output, compress, include_data, progress, task):
    """Create SQLite backup by copying the database file."""
    try:
        progress.update(task, description="Backing up SQLite database...")
        
        # For SQLite, the database "name" is the file path
        db_path = Path(config.database.name)
        
        if not db_path.exists():
            console.print(f"[red]✗ SQLite database file not found: {db_path}[/red]")
            return False
        
        if include_data:
            # Simple file copy for full backup
            import shutil
            
            if compress and output.endswith('.gz'):
                # Copy and compress
                import gzip
                with open(db_path, 'rb') as f_in:
                    with gzip.open(output, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(db_path, output)
            
            return True
        else:
            # Schema-only backup: use SQLite .schema command
            cmd = ['sqlite3', str(db_path), '.schema']
            
            if compress and output.endswith('.gz'):
                # Get schema and compress
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    import gzip
                    with gzip.open(output, 'wt') as f:
                        f.write(result.stdout)
                    return True
            else:
                # Direct output to file
                with open(output, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                          text=True, timeout=60)
                if result.returncode == 0:
                    return True
            
            # Handle errors
            if result.stderr:
                console.print(f"[red]sqlite3 error: {result.stderr}[/red]")
            return False
            
    except subprocess.TimeoutExpired:
        console.print("[red]✗ Backup timed out[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ sqlite3 not found. Please install SQLite.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ SQLite backup failed: {e}[/red]")
        return False


@database_group.command(name='restore')
@click.option('--input', required=True, type=click.Path(exists=True),
              help='Backup file to restore from.')
@click.option('--force/--no-force', default=False,
              help='Force restore even if database is not empty.')
def restore_database(input, force):
    """Restore database from backup.
    
    Restores the database from a backup file created with the backup command.
    This will overwrite existing data.
    
    Examples:
        rag-cli database restore --input backup.sql
        rag-cli database restore --input backup.sql.gz --force
    """
    try:
        # Get configuration
        config = get_config()
        
        console.print(f"[yellow]Restoring database from: {input}[/yellow]")
        console.print("[red]Warning: This will overwrite existing data![/red]")
        
        # Test database connection first
        db_manager = DatabaseFactory.create_manager(config.database)
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        # Check if database has data (unless force is used)
        if not force:
            try:
                with db_manager.get_connection() as conn:
                    if config.database.driver.startswith('sqlite'):
                        result = conn.execute(text("""
                            SELECT COUNT(*) FROM sqlite_master 
                            WHERE type='table' AND name NOT LIKE 'sqlite_%'
                        """))
                    else:
                        result = conn.execute(text("""
                            SELECT COUNT(*) FROM information_schema.tables 
                            WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
                        """))
                    table_count = result.scalar()
                    
                    if table_count > 0:
                        console.print(f"[yellow]Database contains {table_count} existing tables[/yellow]")
                        if not confirm_action("Continue with database restore? This will overwrite existing data."):
                            console.print("[yellow]Database restore cancelled by user[/yellow]")
                            return
            except Exception as e:
                console.print(f"[yellow]Could not check existing data: {e}[/yellow]")
                if not confirm_action("Continue with database restore?"):
                    console.print("[yellow]Database restore cancelled by user[/yellow]")
                    return
        
        # Validate backup file
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]✗ Backup file not found: {input}[/red]")
            return
        
        with create_progress_bar() as progress:
            task = progress.add_task("Restoring database...", total=None)
            
            restore_success = False
            
            try:
                # Handle different database drivers
                driver = config.database.driver.lower()
                
                if driver.startswith('postgresql'):
                    restore_success = _restore_postgresql(config, input, progress, task)
                elif driver.startswith('mysql') or driver.startswith('mariadb'):
                    restore_success = _restore_mysql(config, input, progress, task)
                elif driver.startswith('sqlite'):
                    restore_success = _restore_sqlite(config, input, progress, task)
                else:
                    console.print(f"[red]✗ Restore not supported for database driver: {driver}[/red]")
                    return
                
                if restore_success:
                    console.print("[green]✓ Database restored successfully[/green]")
                    console.print("[dim]Please verify the restored data[/dim]")
                else:
                    console.print("[red]✗ Database restore failed[/red]")
                    
            except Exception as e:
                console.print(f"[red]✗ Restore failed: {e}[/red]")
            finally:
                progress.remove_task(task)
                
    except Exception as e:
        console.print(f"[red]✗ Failed to restore database: {e}[/red]")


def _restore_postgresql(config, input_file, progress, task):
    """Restore PostgreSQL backup using psql."""
    try:
        progress.update(task, description="Running psql restore...")
        
        # Check if file is compressed
        input_path = Path(input_file)
        is_compressed = input_path.suffix == '.gz'
        
        # Build psql command
        cmd = ['psql']
        
        # Connection parameters
        cmd.extend(['-h', config.database.host])
        cmd.extend(['-p', str(config.database.port)])
        cmd.extend(['-U', config.database.user])
        cmd.extend(['-d', config.database.name])
        
        # Environment for password
        env = subprocess.os.environ.copy()
        if config.database.password:
            env['PGPASSWORD'] = config.database.password
        
        # Execute restore
        if is_compressed:
            # Decompress and pipe to psql
            import gzip
            with gzip.open(input_file, 'rt') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, 
                                      env=env, text=True, timeout=600)
        else:
            # Direct file input
            with open(input_file, 'r') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, 
                                      env=env, text=True, timeout=600)
        
        if result.returncode == 0:
            return True
        
        # Handle errors
        if result.stderr:
            console.print(f"[red]psql error: {result.stderr}[/red]")
        return False
        
    except subprocess.TimeoutExpired:
        console.print("[red]✗ Restore timed out[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ psql not found. Please install PostgreSQL client tools.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ PostgreSQL restore failed: {e}[/red]")
        return False


def _restore_mysql(config, input_file, progress, task):
    """Restore MySQL/MariaDB backup using mysql."""
    try:
        progress.update(task, description="Running mysql restore...")
        
        # Check if file is compressed
        input_path = Path(input_file)
        is_compressed = input_path.suffix == '.gz'
        
        # Build mysql command
        cmd = ['mysql']
        
        # Connection parameters
        cmd.extend(['-h', config.database.host])
        cmd.extend(['-P', str(config.database.port)])
        cmd.extend(['-u', config.database.user])
        
        if config.database.password:
            cmd.extend(['-p' + config.database.password])
        
        cmd.append(config.database.name)
        
        # Execute restore
        if is_compressed:
            # Decompress and pipe to mysql
            import gzip
            with gzip.open(input_file, 'rt') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, 
                                      text=True, timeout=600)
        else:
            # Direct file input  
            with open(input_file, 'r') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, 
                                      text=True, timeout=600)
        
        if result.returncode == 0:
            return True
        
        # Handle errors
        if result.stderr:
            console.print(f"[red]mysql error: {result.stderr}[/red]")
        return False
        
    except subprocess.TimeoutExpired:
        console.print("[red]✗ Restore timed out[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ mysql not found. Please install MySQL client tools.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ MySQL restore failed: {e}[/red]")
        return False


def _restore_sqlite(config, input_file, progress, task):
    """Restore SQLite backup."""
    try:
        progress.update(task, description="Restoring SQLite database...")
        
        # For SQLite, the database "name" is the file path
        db_path = Path(config.database.name)
        input_path = Path(input_file)
        is_compressed = input_path.suffix == '.gz'
        
        # Check if input file looks like a full database file or SQL schema
        if is_compressed:
            # Check if it's a compressed database file or SQL
            import gzip
            with gzip.open(input_file, 'rb') as f:
                # Read first few bytes to check format
                header = f.read(16)
                is_sqlite_db = header.startswith(b'SQLite format 3')
        else:
            # Check if it's a SQLite database file or SQL
            with open(input_file, 'rb') as f:
                header = f.read(16)
                is_sqlite_db = header.startswith(b'SQLite format 3')
        
        if is_sqlite_db:
            # Full database file restore - replace the existing file
            import shutil
            
            # Backup existing database
            if db_path.exists():
                backup_path = db_path.with_suffix(db_path.suffix + '.backup')
                shutil.copy2(db_path, backup_path)
                console.print(f"[dim]Existing database backed up to: {backup_path}[/dim]")
            
            if is_compressed:
                # Decompress to database location
                import gzip
                with gzip.open(input_file, 'rb') as f_in:
                    with open(db_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(input_file, db_path)
            
            return True
        else:
            # SQL file restore - execute SQL commands
            if is_compressed:
                # Decompress and execute SQL
                import gzip
                with gzip.open(input_file, 'rt') as f:
                    sql_content = f.read()
            else:
                # Read SQL file
                with open(input_file, 'r') as f:
                    sql_content = f.read()
            
            # Execute SQL using sqlite3 command line
            cmd = ['sqlite3', str(db_path)]
            
            result = subprocess.run(cmd, input=sql_content, stderr=subprocess.PIPE, 
                                  text=True, timeout=300)
            
            if result.returncode == 0:
                return True
            
            # Handle errors
            if result.stderr:
                console.print(f"[red]sqlite3 error: {result.stderr}[/red]")
            return False
            
    except subprocess.TimeoutExpired:
        console.print("[red]✗ Restore timed out[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ sqlite3 not found. Please install SQLite.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ SQLite restore failed: {e}[/red]")
        return False


@database_group.command(name='status')
def database_status():
    """Show database status and connection info.
    
    Displays information about the database connection,
    schema version, and basic statistics.
    """
    console.print("[yellow]Checking database status...[/yellow]")
    
    try:
        # Get configuration
        config = get_config()
        
        # Create database manager
        db_manager = DatabaseFactory.create_manager(config.database)
        
        # Create status table
        table = Table(title="Database Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic configuration
        table.add_row("Driver", config.database.driver)
        table.add_row("Host", f"{config.database.host}:{config.database.port}")
        table.add_row("Database", config.database.name)
        
        # Test connection
        connection_status = "✓ Connected" if db_manager.test_connection() else "✗ Failed"
        table.add_row("Connection", connection_status)
        
        if db_manager.test_connection():
            try:
                # Get schema version from Alembic
                from alembic.migration import MigrationContext
                from alembic.config import Config as AlembicConfig
                
                try:
                    with db_manager.get_connection() as conn:
                        context = MigrationContext.configure(conn)
                        current_rev = context.get_current_revision()
                        schema_version = current_rev if current_rev else "No migrations applied"
                except Exception:
                    schema_version = "Alembic not initialized"
                
                table.add_row("Schema Version", schema_version)
                
                # Count tables
                try:
                    with db_manager.get_connection() as conn:
                        if config.database.driver.startswith('sqlite'):
                            result = conn.execute(text("""
                                SELECT COUNT(*) FROM sqlite_master 
                                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                            """))
                        else:
                            result = conn.execute(text("""
                                SELECT COUNT(*) FROM information_schema.tables 
                                WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
                            """))
                        table_count = result.scalar()
                        table.add_row("Tables", str(table_count))
                except Exception as e:
                    table.add_row("Tables", f"Error: {e}")
                
                # Count users
                try:
                    with db_manager.get_connection() as conn:
                        result = conn.execute(text("SELECT COUNT(*) FROM users"))
                        user_count = result.scalar()
                        table.add_row("Users Count", str(user_count))
                except Exception:
                    table.add_row("Users Count", "Users table not found")
                
                # Pool status
                pool_status = db_manager.get_pool_status()
                if pool_status and 'error' not in pool_status:
                    pool_info = []
                    if 'size' in pool_status:
                        pool_info.append(f"Size: {pool_status['size']}")
                    if 'checked_out' in pool_status:
                        pool_info.append(f"Active: {pool_status['checked_out']}")
                    if 'checked_in' in pool_status:
                        pool_info.append(f"Available: {pool_status['checked_in']}")
                    
                    table.add_row("Connection Pool", ", ".join(pool_info) if pool_info else "Available")
                else:
                    table.add_row("Connection Pool", "Status unavailable")
                    
            except Exception as e:
                table.add_row("Schema Version", f"Error: {e}")
                table.add_row("Tables", "Unable to query")
                table.add_row("Users Count", "Unable to query")
        
        console.print(table)
        console.print("[green]✓ Database status check completed[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to get database status: {e}[/red]")


@database_group.command(name='test')
def test_database():
    """Test database connectivity.
    
    Tests the database connection and basic operations
    to ensure everything is working correctly.
    """
    console.print("[yellow]Testing database connectivity...[/yellow]")
    
    try:
        # Get configuration and create database manager
        config = get_config()
        db_manager = DatabaseFactory.create_manager(config.database)
        
        test_results = []
        
        with create_progress_bar() as progress:
            # Test 1: Basic Connection
            task = progress.add_task("Testing basic connection...", total=None)
            
            try:
                connection_success = db_manager.test_connection()
                if connection_success:
                    test_results.append(("Basic Connection", "✓ PASS", "green"))
                else:
                    test_results.append(("Basic Connection", "✗ FAIL", "red"))
            except Exception as e:
                test_results.append(("Basic Connection", f"✗ FAIL: {e}", "red"))
            
            # Test 2: Query Execution
            progress.update(task, description="Testing query execution...")
            
            try:
                with db_manager.get_connection() as conn:
                    result = conn.execute(text("SELECT 1 as test_value"))
                    row = result.fetchone()
                    if row and row[0] == 1:
                        test_results.append(("Query Execution", "✓ PASS", "green"))
                    else:
                        test_results.append(("Query Execution", "✗ FAIL: Unexpected result", "red"))
            except Exception as e:
                test_results.append(("Query Execution", f"✗ FAIL: {e}", "red"))
            
            # Test 3: Transaction Support
            progress.update(task, description="Testing transaction support...")
            
            try:
                with db_manager.get_transaction() as conn:
                    # Test transaction by creating a temporary table and rolling back
                    temp_table_name = f"test_temp_table_{int(datetime.now().timestamp())}"
                    
                    # Create a temporary table
                    conn.execute(text(f"""
                        CREATE TEMPORARY TABLE {temp_table_name} (
                            id INTEGER PRIMARY KEY,
                            test_data VARCHAR(50)
                        )
                    """))
                    
                    # Insert test data
                    conn.execute(text(f"""
                        INSERT INTO {temp_table_name} (test_data) VALUES ('test')
                    """))
                    
                    # Verify data exists
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {temp_table_name}"))
                    count = result.scalar()
                    
                    if count == 1:
                        test_results.append(("Transaction Support", "✓ PASS", "green"))
                    else:
                        test_results.append(("Transaction Support", "✗ FAIL: Transaction data error", "red"))
                        
            except Exception as e:
                test_results.append(("Transaction Support", f"✗ FAIL: {e}", "red"))
            
            # Test 4: Connection Pool
            progress.update(task, description="Testing connection pool...")
            
            try:
                pool_status = db_manager.get_pool_status()
                if pool_status and 'error' not in pool_status:
                    test_results.append(("Connection Pool", "✓ PASS", "green"))
                else:
                    test_results.append(("Connection Pool", "⚠ WARNING: Pool status unavailable", "yellow"))
            except Exception as e:
                test_results.append(("Connection Pool", f"✗ FAIL: {e}", "red"))
            
            # Test 5: Health Check
            progress.update(task, description="Testing health check...")
            
            try:
                health_result = db_manager.health_check()
                if health_result.get('status') == 'healthy':
                    test_results.append(("Health Check", "✓ PASS", "green"))
                else:
                    test_results.append(("Health Check", f"✗ FAIL: {health_result.get('message', 'Unknown error')}", "red"))
            except Exception as e:
                test_results.append(("Health Check", f"✗ FAIL: {e}", "red"))
            
            progress.remove_task(task)
        
        # Display results
        console.print("\n[bold]Test Results:[/bold]")
        
        table = Table()
        table.add_column("Test", style="cyan")
        table.add_column("Result", style="white")
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result, color in test_results:
            table.add_row(test_name, f"[{color}]{result}[/{color}]")
            if "PASS" in result:
                passed_tests += 1
        
        console.print(table)
        
        # Summary
        if passed_tests == total_tests:
            console.print(f"\n[green]✅ All {total_tests} tests passed![/green]")
        else:
            failed_tests = total_tests - passed_tests
            console.print(f"\n[yellow]⚠ {passed_tests}/{total_tests} tests passed, {failed_tests} failed[/yellow]")
        
        console.print("[dim]Database connectivity test completed[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to run database tests: {e}[/red]")


def _create_default_roles_and_permissions(db_manager):
    """Create default roles and permissions."""
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=db_manager.engine)
    
    with Session() as session:
        # Define default permissions
        default_permissions = [
            # User management
            {"name": "users.read", "resource": "users", "action": "read", "description": "View users"},
            {"name": "users.write", "resource": "users", "action": "write", "description": "Create/update users"},
            {"name": "users.delete", "resource": "users", "action": "delete", "description": "Delete users"},
            
            # Document management
            {"name": "documents.read", "resource": "documents", "action": "read", "description": "View documents"},
            {"name": "documents.write", "resource": "documents", "action": "write", "description": "Create/update documents"},
            {"name": "documents.delete", "resource": "documents", "action": "delete", "description": "Delete documents"},
            
            # System administration
            {"name": "system.admin", "resource": "system", "action": "admin", "description": "System administration"},
            {"name": "system.config", "resource": "system", "action": "config", "description": "System configuration"},
        ]
        
        # Create permissions
        created_permissions = {}
        for perm_data in default_permissions:
            existing_perm = session.query(Permission).filter_by(name=perm_data["name"]).first()
            if not existing_perm:
                permission = Permission(**perm_data)
                session.add(permission)
                created_permissions[perm_data["name"]] = permission
            else:
                created_permissions[perm_data["name"]] = existing_perm
        
        # Define default roles
        default_roles = [
            {
                "name": "admin",
                "description": "System administrator with full access",
                "is_system": True,
                "priority": 100,
                "permissions": ["users.read", "users.write", "users.delete", "documents.read", 
                              "documents.write", "documents.delete", "system.admin", "system.config"]
            },
            {
                "name": "editor",
                "description": "Content editor with document management access",
                "is_system": True,
                "priority": 50,
                "permissions": ["documents.read", "documents.write", "users.read"]
            },
            {
                "name": "user",
                "description": "Regular user with read access",
                "is_system": True,
                "is_default": True,
                "priority": 10,
                "permissions": ["documents.read"]
            },
            {
                "name": "readonly",
                "description": "Read-only access to documents",
                "is_system": True,
                "priority": 5,
                "permissions": ["documents.read"]
            }
        ]
        
        # Create roles and assign permissions
        for role_data in default_roles:
            permissions = role_data.pop("permissions", [])
            existing_role = session.query(Role).filter_by(name=role_data["name"]).first()
            
            if not existing_role:
                role = Role(**role_data)
                session.add(role)
                session.flush()  # Get the role ID
                
                # Assign permissions to role
                for perm_name in permissions:
                    if perm_name in created_permissions:
                        role_perm = RolePermission(
                            role_id=role.id,
                            permission_id=created_permissions[perm_name].id
                        )
                        session.add(role_perm)
        
        session.commit()


def _init_sqlite_databases(database_names, force):
    """Initialize SQLite databases using schema files."""
    import sqlite3
    import hashlib
    from pathlib import Path
    
    console.print("[yellow]Initializing SQLite databases...[/yellow]")
    
    # Define database configurations
    project_root = Path.cwd()
    schema_dir = project_root / "database" / "schemas"
    data_dir = project_root / "data"
    
    # Ensure directories exist
    data_dir.mkdir(exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    
    db_configs = {
        'auth': {
            'path': project_root / 'auth.db',
            'schema': schema_dir / 'auth.sql',
            'init_func': lambda p: _init_auth_db(p)
        },
        'metrics': {
            'path': project_root / 'metrics.db',
            'schema': schema_dir / 'metrics.sql',
            'init_func': None
        },
        'conversations': {
            'path': data_dir / 'conversations.db',
            'schema': schema_dir / 'conversations.sql',
            'init_func': None
        },
        'user_data': {
            'path': data_dir / 'user_data.db',
            'schema': schema_dir / 'user_data.sql',
            'init_func': None
        }
    }
    
    # Filter databases if specific ones requested
    if database_names:
        db_configs = {k: v for k, v in db_configs.items() if k in database_names}
    
    console.print(f"[dim]Databases to initialize: {list(db_configs.keys())}[/dim]")
    
    success_count = 0
    for db_name, config in db_configs.items():
        try:
            console.print(f"\n[cyan]Processing {db_name} database...[/cyan]")
            
            # Check if database exists
            if config['path'].exists() and not force:
                console.print(f"[yellow]{db_name}.db already exists[/yellow]")
                if not confirm_action(f"Reinitialize {db_name} database? This will delete existing data!"):
                    console.print(f"[dim]Skipping {db_name}[/dim]")
                    continue
            
            # Load schema
            if not config['schema'].exists():
                console.print(f"[yellow]Schema file not found: {config['schema']}[/yellow]")
                console.print(f"[dim]Creating empty {db_name} database[/dim]")
                schema_sql = ""
            else:
                with open(config['schema'], 'r') as f:
                    schema_sql = f.read()
            
            # Create/recreate database
            if config['path'].exists() and force:
                config['path'].unlink()
                console.print(f"[dim]Removed existing {db_name}.db[/dim]")
            
            # Connect and apply schema
            conn = sqlite3.connect(config['path'])
            cursor = conn.cursor()
            
            if schema_sql and not schema_sql.startswith("#"):
                cursor.executescript(schema_sql)
                conn.commit()
                console.print(f"[green]✓ Applied schema to {db_name}.db[/green]")
            else:
                console.print(f"[green]✓ Created empty {db_name}.db[/green]")
            
            # Run post-initialization if defined
            if config['init_func']:
                config['init_func'](config['path'])
            
            conn.close()
            success_count += 1
            
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize {db_name}: {e}[/red]")
    
    # Summary
    if success_count == len(db_configs):
        console.print(f"\n[green]✅ Successfully initialized {success_count}/{len(db_configs)} databases[/green]")
    else:
        console.print(f"\n[yellow]⚠ Initialized {success_count}/{len(db_configs)} databases[/yellow]")


def _init_auth_db(db_path):
    """Initialize auth database with default admin user."""
    import sqlite3
    import hashlib
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if admin already exists
        cursor.execute("SELECT username FROM users WHERE username = 'admin'")
        if cursor.fetchone():
            console.print("[dim]Admin user already exists[/dim]")
            conn.close()
            return
        
        # Create admin user with hashed password
        # Default password: admin123 (should be changed on first login)
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, email, role, is_active)
            VALUES ('admin', ?, 'admin@example.com', 'admin', 1)
        """, (password_hash,))
        
        conn.commit()
        conn.close()
        
        console.print("[green]✓ Created default admin user[/green]")
        console.print("[yellow]  Username: admin[/yellow]")
        console.print("[yellow]  Password: admin123[/yellow]")
        console.print("[red]  ⚠️  Please change the password after first login![/red]")
        
    except Exception as e:
        console.print(f"[yellow]⚠ Could not create admin user: {e}[/yellow]")


def _create_default_admin_user(db_manager):
    """Create default admin user if it doesn't exist."""
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=db_manager.engine)
    
    with Session() as session:
        # Check if admin user already exists
        existing_admin = session.query(User).filter_by(username="admin").first()
        if existing_admin:
            console.print("[dim]Admin user already exists[/dim]")
            return
        
        # Get admin role
        admin_role = session.query(Role).filter_by(name="admin").first()
        if not admin_role:
            console.print("[yellow]⚠ Admin role not found, skipping admin user creation[/yellow]")
            return
        
        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@rag-server.local",
            full_name="System Administrator",
            is_active=True,
            is_superuser=True,
            is_verified=True
        )
        admin_user.set_password("admin123")  # Default password - should be changed!
        
        session.add(admin_user)
        session.flush()  # Get user ID
        
        # Assign admin role
        user_role = UserRole(
            user_id=admin_user.id,
            role_id=admin_role.id
        )
        session.add(user_role)
        
        session.commit()
        
        console.print("[yellow]⚠ Default admin user created with username 'admin' and password 'admin123'[/yellow]")
        console.print("[yellow]⚠ Please change the default password immediately![/yellow]")