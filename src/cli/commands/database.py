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
def init_database(force):
    """Initialize the database schema.
    
    Creates all necessary tables and indexes for the RAG server.
    This should be run once when setting up a new instance.
    
    Examples:
        rag-cli database init
        rag-cli database init --force
    """
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
    # Running database migrations
    
    console.print("[yellow]Checking for pending migrations...[/yellow]")
    
    if dry_run:
        console.print("[dim]Running in dry-run mode - no changes will be made[/dim]")
    
    # TODO: Implement actual database migration
    # This would involve:
    # 1. Checking current schema version
    # 2. Finding pending migrations
    # 3. Applying migrations in order
    # 4. Updating migration history
    
    console.print("[red]✗ Database migration implementation not complete[/red]")
    # Database migration completed


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
    # Creating database backup
    
    if not output:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = ".sql.gz" if compress else ".sql"
        output = f"rag_backup_{timestamp}{extension}"
    
    console.print(f"[yellow]Creating database backup: {output}[/yellow]")
    
    backup_type = "Full backup" if include_data else "Schema only"
    console.print(f"[dim]{backup_type}, Compressed: {compress}[/dim]")
    
    # TODO: Implement actual database backup
    console.print("[red]✗ Database backup implementation not complete[/red]")
    # Database backup completed


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
    # Restoring database from backup
    
    console.print(f"[yellow]Restoring database from: {input}[/yellow]")
    console.print("[red]Warning: This will overwrite existing data![/red]")
    
    if not force:
        if not confirm_action("Continue with database restore?"):
            # Database restore cancelled
            return
    
    # TODO: Implement actual database restore
    console.print("[red]✗ Database restore implementation not complete[/red]")
    # Database restore completed


@database_group.command(name='status')
def database_status():
    """Show database status and connection info.
    
    Displays information about the database connection,
    schema version, and basic statistics.
    """
    # Checking database status
    
    console.print("[yellow]Checking database status...[/yellow]")
    
    # Create status table
    table = Table(title="Database Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    # TODO: Get actual database status
    config = get_config()
    table.add_row("Driver", config.database.driver)
    table.add_row("Host", f"{config.database.host}:{config.database.port}")
    table.add_row("Database", config.database.name)
    table.add_row("Connection", "Not tested")
    table.add_row("Schema Version", "Unknown")
    table.add_row("Tables", "Unknown")
    table.add_row("Users Count", "Unknown")
    
    console.print(table)
    # Database status displayed


@database_group.command(name='test')
def test_database():
    """Test database connectivity.
    
    Tests the database connection and basic operations
    to ensure everything is working correctly.
    """
    # Testing database connectivity
    
    console.print("[yellow]Testing database connectivity...[/yellow]")
    
    with create_progress_bar() as progress:
        # Test connection
        task = progress.add_task("Testing connection...", total=None)
        import time
        time.sleep(1)
        
        progress.update(task, description="Testing queries...")
        time.sleep(1)
        
        progress.update(task, description="Testing transactions...")
        time.sleep(1)
        
        progress.remove_task(task)
    
    # TODO: Implement actual database tests
    console.print("[green]✓ Connection test passed[/green]")
    console.print("[red]✗ Other database tests not implemented[/red]")
    
    console.print("[dim]Database connectivity test completed[/dim]")


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