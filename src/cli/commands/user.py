"""
User management commands for the CLI.
"""

import click
from rich.console import Console
from rich.table import Table
from typing import Optional

from ..utils import (
    console, confirm_action, prompt_for_input, display_table
)

from src.core.config import get_config
from src.database.base import DatabaseFactory
from src.auth.models import User, Role, Permission, UserRole, RolePermission
from src.core.exceptions import DatabaseError
from sqlalchemy.orm import sessionmaker

console = Console()


@click.group()
def user_group():
    """User management commands.
    
    Commands for creating, managing, and administering user accounts
    including roles and permissions.
    """
    pass


@user_group.command(name='create')
@click.option('--username', required=True, help='Username for the new user.')
@click.option('--email', required=True, help='Email address for the new user.')
@click.option('--password', help='Password for the new user (will prompt if not provided).')
@click.option('--role', default='user', type=click.Choice(['admin', 'editor', 'user', 'readonly']),
              help='Role to assign to the user.')
@click.option('--groups', help='Comma-separated list of groups to add user to.')
@click.option('--force/--no-force', default=False,
              help='Force creation even if user exists.')
def create_user(username, email, password, role, groups, force):
    """Create a new user account.
    
    Creates a new user with the specified username, email, and role.
    If password is not provided, you will be prompted to enter one.
    
    Examples:
        rag-cli user create --username john --email john@example.com
        rag-cli user create --username admin --email admin@company.com --role admin
        rag-cli user create --username editor --email editor@company.com --role editor --groups content,review
    """
    console.print(f"[yellow]Creating user '{username}'...[/yellow]")
    
    try:
        # Get configuration and database connection
        config = get_config()
        db_manager = DatabaseFactory.create_manager(config.database)
        
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        Session = sessionmaker(bind=db_manager.engine)
        
        with Session() as session:
            # Check if user already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                if existing_user.username == username:
                    console.print(f"[red]✗ User with username '{username}' already exists[/red]")
                else:
                    console.print(f"[red]✗ User with email '{email}' already exists[/red]")
                return
            
            # Prompt for password if not provided
            if not password:
                password = prompt_for_input("Enter password for user", password=True)
                confirm_password = prompt_for_input("Confirm password", password=True)
                
                if password != confirm_password:
                    console.print("[red]Passwords do not match[/red]")
                    return
            
            # Validate role exists
            user_role = session.query(Role).filter_by(name=role).first()
            if not user_role:
                console.print(f"[red]✗ Role '{role}' does not exist[/red]")
                available_roles = session.query(Role).all()
                if available_roles:
                    console.print("[dim]Available roles:[/dim]")
                    for r in available_roles:
                        console.print(f"  • {r.name} - {r.description}")
                return
            
            # Parse groups (for future group support)
            user_groups = [g.strip() for g in groups.split(',')] if groups else []
            
            # Show user details for confirmation
            user_details = f"""    Username: {username}
    Email: {email}
    Role: {role}
    Groups: {', '.join(user_groups) if user_groups else 'None'}"""
            
            console.print(f"[dim]{user_details}[/dim]")
            
            if not force:
                if not confirm_action("Create this user?"):
                    console.print("[yellow]User creation cancelled[/yellow]")
                    return
            
            # Create user
            new_user = User(
                username=username,
                email=email,
                is_active=True,
                is_verified=True
            )
            new_user.set_password(password)
            
            session.add(new_user)
            session.flush()  # Get user ID
            
            # Assign role
            user_role_assignment = UserRole(
                user_id=new_user.id,
                role_id=user_role.id
            )
            session.add(user_role_assignment)
            
            session.commit()
            
            console.print(f"[green]✅ User '{username}' created successfully[/green]")
            console.print(f"[dim]User ID: {new_user.id}, Role: {role}[/dim]")
            
    except DatabaseError as e:
        console.print(f"[red]✗ Database error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error creating user: {e}[/red]")


@user_group.command(name='list')
@click.option('--role', type=click.Choice(['admin', 'editor', 'user', 'readonly', 'all']),
              default='all', help='Filter users by role.')
@click.option('--active/--inactive', default=None,
              help='Filter by user status.')
@click.option('--format', type=click.Choice(['table', 'json']),
              default='table', help='Output format.')
def list_users(role, active, format):
    """List all users.
    
    Displays a list of all users with their roles, status, and
    last login information.
    
    Examples:
        rag-cli user list
        rag-cli user list --role admin
        rag-cli user list --active --format json
    """
    console.print("[yellow]Fetching user list...[/yellow]")
    
    try:
        # Get configuration and database connection
        config = get_config()
        db_manager = DatabaseFactory.create_manager(config.database)
        
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        Session = sessionmaker(bind=db_manager.engine)
        
        with Session() as session:
            # Build query
            query = session.query(User).join(UserRole).join(Role)
            
            # Apply filters
            if role != 'all':
                query = query.filter(Role.name == role)
            
            if active is not None:
                query = query.filter(User.is_active == active)
            
            # Execute query
            users = query.all()
            
            if not users:
                console.print("[yellow]No users found matching criteria[/yellow]")
                return
            
            # Format user data
            user_data = []
            for user in users:
                # Get user roles
                user_roles = [ur.role.name for ur in user.user_roles]
                
                user_info = {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name or '',
                    'roles': ', '.join(user_roles),
                    'status': 'active' if user.is_active else 'inactive',
                    'verified': 'yes' if user.is_verified else 'no',
                    'superuser': 'yes' if user.is_superuser else 'no',
                    'last_login': user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never',
                    'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(user, 'created_at') and user.created_at else 'Unknown'
                }
                user_data.append(user_info)
            
            # Output results
            if format == 'json':
                import json
                console.print(json.dumps(user_data, indent=2, default=str))
            else:
                # Display in table format
                display_columns = ['username', 'email', 'roles', 'status', 'verified', 'last_login']
                display_table(user_data, title="User List", columns=display_columns)
            
            console.print(f"\n[dim]Found {len(user_data)} users[/dim]")
            
    except DatabaseError as e:
        console.print(f"[red]✗ Database error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error listing users: {e}[/red]")


@user_group.command(name='update')
@click.argument('username')
@click.option('--email', help='New email address.')
@click.option('--role', type=click.Choice(['admin', 'editor', 'user', 'readonly']),
              help='New role for the user.')
@click.option('--password', help='New password (will prompt if not provided).')
@click.option('--status', type=click.Choice(['active', 'inactive']),
              help='User status.')
@click.option('--add-groups', help='Comma-separated list of groups to add.')
@click.option('--remove-groups', help='Comma-separated list of groups to remove.')
def update_user(username, email, role, password, status, add_groups, remove_groups):
    """Update an existing user.
    
    Updates user properties such as email, role, password, or group membership.
    Only specified fields will be updated.
    
    Examples:
        rag-cli user update john --email newemail@example.com
        rag-cli user update john --role editor
        rag-cli user update john --add-groups content,review
    """
    # Updating user
    
    console.print(f"[yellow]Updating user '{username}'...[/yellow]")
    
    # Collect changes
    changes = {}
    if email:
        changes['email'] = email
    if role:
        changes['role'] = role
    if status:
        changes['status'] = status
    
    # Handle password change
    if password is not None:
        if not password:  # Empty string means prompt
            password = prompt_for_input("Enter new password", password=True)
            confirm_password = prompt_for_input("Confirm new password", password=True)
            
            if password != confirm_password:
                console.print("[red]Passwords do not match[/red]")
                return
        changes['password'] = '***'  # Don't show actual password in logs
    
    # Handle group changes
    groups_to_add = [g.strip() for g in add_groups.split(',')] if add_groups else []
    groups_to_remove = [g.strip() for g in remove_groups.split(',')] if remove_groups else []
    
    if groups_to_add:
        changes['add_groups'] = groups_to_add
    if groups_to_remove:
        changes['remove_groups'] = groups_to_remove
    
    if not changes:
        console.print("[yellow]No changes specified[/yellow]")
        return
    
    # Show changes for confirmation
    console.print("[dim]Changes to apply:[/dim]")
    for key, value in changes.items():
        if key == 'password':
            console.print(f"  • {key}: [password will be updated]")
        else:
            console.print(f"  • {key}: {value}")
    
    if not confirm_action(f"Apply these changes to user '{username}'?"):
        # User update cancelled
        return
    
    # TODO: Implement actual user update
    console.print("[red]✗ User update implementation not complete[/red]")
    # User update completed


@user_group.command(name='delete')
@click.argument('username')
@click.option('--force/--no-force', default=False,
              help='Force deletion without confirmation.')
def delete_user(username, force):
    """Delete a user account.
    
    Permanently removes a user account and all associated data.
    This action cannot be undone.
    
    Examples:
        rag-cli user delete olduser
        rag-cli user delete testuser --force
    """
    # Deleting user
    
    console.print(f"[red]Deleting user '{username}'...[/red]")
    
    if not force:
        console.print("[yellow]Warning: This will permanently delete the user and all associated data.[/yellow]")
        if not confirm_action(f"Are you sure you want to delete user '{username}'?"):
            # User deletion cancelled
            return
    
    # TODO: Implement actual user deletion
    # This would involve:
    # 1. Checking if user exists
    # 2. Checking for dependencies (owned resources, etc.)
    # 3. Removing user from all groups
    # 4. Deleting user record
    # 5. Cleaning up associated data
    
    console.print("[red]✗ User deletion implementation not complete[/red]")
    # User deletion completed


@user_group.command(name='groups')
@click.argument('username')
def show_user_groups(username):
    """Show groups for a user.
    
    Displays all groups that a user belongs to along with
    the permissions granted by each group.
    
    Examples:
        rag-cli user groups john
    """
    # Showing groups for user
    
    console.print(f"[yellow]Groups for user '{username}':[/yellow]")
    
    # TODO: Get actual user groups from database
    sample_groups = [
        {'group': 'users', 'role': 'member', 'permissions': 'read'},
        {'group': 'content', 'role': 'editor', 'permissions': 'read, write'},
        {'group': 'reviewers', 'role': 'member', 'permissions': 'read, review'}
    ]
    
    if sample_groups:
        display_table(sample_groups, title=f"Groups for {username}")
    else:
        console.print("[yellow]User is not a member of any groups[/yellow]")
    
    # Groups displayed for user