"""
User management commands for the CLI.
"""

import click
from rich.console import Console
from rich.table import Table
from typing import Optional
from datetime import datetime

from ..utils import (
    console, confirm_action, prompt_for_input, display_table
)

from src.core.config import get_config
from src.database.base import DatabaseFactory
from src.auth.models import User, Role, Permission, UserRole, RolePermission
from src.core.exceptions import DatabaseError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

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
    
    # Implement actual user update
    try:
        # Get configuration and database connection
        config = get_config()
        db_manager = DatabaseFactory.create_manager(config.database)
        
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        Session = sessionmaker(bind=db_manager.engine)
        
        with Session() as session:
            # Find the user to update
            user = session.query(User).filter_by(username=username).first()
            if not user:
                console.print(f"[red]✗ User '{username}' not found[/red]")
                return
            
            # Check if user is trying to modify admin account without proper permissions
            if user.is_superuser and not confirm_action("This is an admin account. Are you sure you want to modify it?"):
                console.print("[yellow]Admin account modification cancelled[/yellow]")
                return
            
            updates_made = []
            
            # Update email
            if email:
                # Check if email is already in use
                existing_email_user = session.query(User).filter(
                    User.email == email,
                    User.id != user.id
                ).first()
                if existing_email_user:
                    console.print(f"[red]✗ Email '{email}' is already in use by another user[/red]")
                    return
                
                old_email = user.email
                user.email = email
                updates_made.append(f"Email: {old_email} → {email}")
            
            # Update role
            if role:
                # Find the new role
                new_role = session.query(Role).filter_by(name=role).first()
                if not new_role:
                    console.print(f"[red]✗ Role '{role}' does not exist[/red]")
                    available_roles = session.query(Role).all()
                    if available_roles:
                        console.print("[dim]Available roles:[/dim]")
                        for r in available_roles:
                            console.print(f"  • {r.name} - {r.description}")
                    return
                
                # Get current roles for comparison
                current_roles = [ur.role.name for ur in user.user_roles]
                
                # Remove all existing roles
                for user_role in user.user_roles:
                    session.delete(user_role)
                
                # Add new role
                new_user_role = UserRole(
                    user_id=user.id,
                    role_id=new_role.id
                )
                session.add(new_user_role)
                
                updates_made.append(f"Role: {', '.join(current_roles)} → {role}")
            
            # Update password
            if password is not None:
                user.set_password(password)
                updates_made.append("Password: [updated]")
            
            # Update status
            if status:
                old_status = 'active' if user.is_active else 'inactive'
                new_status = status == 'active'
                user.is_active = new_status
                
                if old_status != status:
                    updates_made.append(f"Status: {old_status} → {status}")
            
            # Handle group additions - implement actual group membership
            if groups_to_add:
                try:
                    from src.access_control.group_manager import GroupManager, group_users
                    
                    # Add user to each specified group
                    added_groups = []
                    failed_groups = []
                    
                    for group_name in groups_to_add:
                        try:
                            # Check if group exists
                            group = session.query(Group).filter_by(name=group_name).first()
                            if not group:
                                console.print(f"[yellow]Warning: Group '{group_name}' does not exist, skipping[/yellow]")
                                failed_groups.append(f"{group_name} (not found)")
                                continue
                            
                            # Check if user is already in the group
                            existing_membership = session.execute(
                                group_users.select().where(
                                    (group_users.c.group_id == group.id) &
                                    (group_users.c.user_id == user.id)
                                )
                            ).first()
                            
                            if existing_membership:
                                console.print(f"[dim]User already in group '{group_name}', updating status[/dim]")
                                # Update existing membership to active
                                session.execute(
                                    group_users.update().where(
                                        (group_users.c.group_id == group.id) &
                                        (group_users.c.user_id == user.id)
                                    ).values(
                                        status='active',
                                        updated_at=func.now()
                                    )
                                )
                                added_groups.append(f"{group_name} (reactivated)")
                            else:
                                # Add new membership
                                session.execute(
                                    group_users.insert().values(
                                        group_id=group.id,
                                        user_id=user.id,
                                        group_role='member',
                                        status='active'
                                    )
                                )
                                added_groups.append(group_name)
                            
                        except Exception as group_error:
                            console.print(f"[red]Error adding to group '{group_name}': {group_error}[/red]")
                            failed_groups.append(f"{group_name} (error)")
                    
                    if added_groups:
                        updates_made.append(f"Added to groups: {', '.join(added_groups)}")
                    if failed_groups:
                        updates_made.append(f"Failed to add to groups: {', '.join(failed_groups)}")
                        
                except ImportError:
                    # Group system not available, add placeholder message
                    updates_made.append(f"Groups to add: {', '.join(groups_to_add)} (group system not fully available)")
                    console.print("[yellow]Warning: Group management system not fully implemented[/yellow]")
            
            # Handle group removals - implement actual group membership removal
            if groups_to_remove:
                try:
                    from src.access_control.group_manager import GroupManager, group_users
                    
                    # Remove user from each specified group
                    removed_groups = []
                    failed_groups = []
                    
                    for group_name in groups_to_remove:
                        try:
                            # Check if group exists
                            group = session.query(Group).filter_by(name=group_name).first()
                            if not group:
                                console.print(f"[yellow]Warning: Group '{group_name}' does not exist, skipping[/yellow]")
                                failed_groups.append(f"{group_name} (not found)")
                                continue
                            
                            # Remove membership
                            result = session.execute(
                                group_users.delete().where(
                                    (group_users.c.group_id == group.id) &
                                    (group_users.c.user_id == user.id)
                                )
                            )
                            
                            if result.rowcount > 0:
                                removed_groups.append(group_name)
                            else:
                                failed_groups.append(f"{group_name} (not a member)")
                            
                        except Exception as group_error:
                            console.print(f"[red]Error removing from group '{group_name}': {group_error}[/red]")
                            failed_groups.append(f"{group_name} (error)")
                    
                    if removed_groups:
                        updates_made.append(f"Removed from groups: {', '.join(removed_groups)}")
                    if failed_groups:
                        updates_made.append(f"Failed to remove from groups: {', '.join(failed_groups)}")
                        
                except ImportError:
                    # Group system not available, add placeholder message
                    updates_made.append(f"Groups to remove: {', '.join(groups_to_remove)} (group system not fully available)")
                    console.print("[yellow]Warning: Group management system not fully implemented[/yellow]")
            
            # Commit all changes
            session.commit()
            
            # Show success message with summary of changes
            console.print(f"[green]✅ User '{username}' updated successfully[/green]")
            
            if updates_made:
                console.print("[dim]Changes applied:[/dim]")
                for update in updates_made:
                    console.print(f"  • {update}")
            
            # Display updated user information
            user_roles = [ur.role.name for ur in user.user_roles]
            console.print(f"\n[dim]Current user details:[/dim]")
            console.print(f"  Username: {user.username}")
            console.print(f"  Email: {user.email}")
            console.print(f"  Roles: {', '.join(user_roles) if user_roles else 'None'}")
            console.print(f"  Status: {'active' if user.is_active else 'inactive'}")
            console.print(f"  Verified: {'yes' if user.is_verified else 'no'}")
            console.print(f"  Last Login: {user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never'}")
            
    except DatabaseError as e:
        console.print(f"[red]✗ Database error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error updating user: {e}[/red]")


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
    
    # Implement actual user deletion with comprehensive cleanup
    try:
        # Get configuration and database connection
        config = get_config()
        db_manager = DatabaseFactory.create_manager(config.database)
        
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        Session = sessionmaker(bind=db_manager.engine)
        
        with Session() as session:
            # 1. Check if user exists
            user = session.query(User).filter_by(username=username).first()
            if not user:
                console.print(f"[red]✗ User '{username}' not found[/red]")
                return
            
            # 2. Admin account protection
            if user.is_superuser:
                console.print(f"[yellow]Warning: '{username}' is a superuser/admin account[/yellow]")
                
                # Count other admin users
                other_admins = session.query(User).filter(
                    User.is_superuser == True,
                    User.id != user.id,
                    User.is_active == True
                ).count()
                
                if other_admins == 0:
                    console.print("[red]✗ Cannot delete the last active admin account[/red]")
                    console.print("[dim]Create another admin account before deleting this one[/dim]")
                    return
                
                if not confirm_action("This is the last admin account. Are you absolutely sure you want to delete it?"):
                    console.print("[yellow]Admin account deletion cancelled[/yellow]")
                    return
            
            # 3. Check for dependencies and owned resources
            dependencies_found = []
            
            # Check for owned documents (if document model exists)
            try:
                from src.models.document import Document
                owned_documents = session.query(Document).filter_by(owner_id=user.id).count()
                if owned_documents > 0:
                    dependencies_found.append(f"{owned_documents} documents")
            except ImportError:
                pass  # Document model not available
            
            # Check for query logs
            try:
                from src.models.query_log import QueryLog
                user_queries = session.query(QueryLog).filter_by(user_id=str(user.id)).count()
                if user_queries > 0:
                    dependencies_found.append(f"{user_queries} query logs")
            except ImportError:
                pass  # QueryLog model not available
            
            # Check for role assignments to other users (if user assigned roles to others)
            role_assignments = session.query(UserRole).filter_by(assigned_by=user.id).count()
            if role_assignments > 0:
                dependencies_found.append(f"{role_assignments} role assignments")
            
            # Show dependency warning
            if dependencies_found:
                console.print(f"[yellow]Warning: User has associated data:[/yellow]")
                for dependency in dependencies_found:
                    console.print(f"  • {dependency}")
                console.print("[dim]This data will be cleaned up or orphaned after deletion[/dim]")
                
                if not confirm_action("Proceed with deletion and cleanup?"):
                    console.print("[yellow]User deletion cancelled[/yellow]")
                    return
            
            # Store user info for logging
            user_info = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'roles': [ur.role.name for ur in user.user_roles],
                'is_superuser': user.is_superuser,
                'created_at': user.created_at
            }
            
            # 4. Begin cleanup process
            console.print(f"[yellow]Cleaning up user data for '{username}'...[/yellow]")
            
            # Remove user from all groups (using group_users association table)
            try:
                from src.access_control.group_manager import group_users
                session.execute(
                    group_users.delete().where(group_users.c.user_id == user.id)
                )
                console.print("  • Removed from all groups")
            except Exception as e:
                console.print(f"  • [yellow]Warning: Could not remove from groups: {e}[/yellow]")
            
            # Update role assignments where this user was the assigner
            role_assignments = session.query(UserRole).filter_by(assigned_by=user.id).all()
            for assignment in role_assignments:
                assignment.assigned_by = None  # Orphan the assignment
            console.print(f"  • Orphaned {len(role_assignments)} role assignments")
            
            # Clean up owned documents (transfer to system or mark as orphaned)
            try:
                from src.models.document import Document
                owned_documents = session.query(Document).filter_by(owner_id=user.id).all()
                for document in owned_documents:
                    document.owner_id = None  # Orphan the document
                console.print(f"  • Orphaned {len(owned_documents)} documents")
            except ImportError:
                pass
            
            # Update query logs to maintain referential integrity
            try:
                from src.models.query_log import QueryLog
                user_queries = session.query(QueryLog).filter_by(user_id=str(user.id)).all()
                for query_log in user_queries:
                    query_log.user_id = f"deleted_user_{user.id}"  # Mark as deleted user
                console.print(f"  • Updated {len(user_queries)} query logs")
            except ImportError:
                pass
            
            # 5. Delete user roles (cascade will handle this, but being explicit)
            for user_role in user.user_roles:
                session.delete(user_role)
            console.print("  • Removed all role assignments")
            
            # 6. Finally, delete the user record
            session.delete(user)
            
            # Commit all changes
            session.commit()
            
            # Success message with audit information
            console.print(f"[green]✅ User '{username}' deleted successfully[/green]")
            console.print(f"[dim]Deleted user details:[/dim]")
            console.print(f"  User ID: {user_info['id']}")
            console.print(f"  Email: {user_info['email']}")
            console.print(f"  Roles: {', '.join(user_info['roles']) if user_info['roles'] else 'None'}")
            console.print(f"  Was Superuser: {'yes' if user_info['is_superuser'] else 'no'}")
            console.print(f"  Account Age: {datetime.now() - user_info['created_at']}")
            
            # Audit log entry (basic console logging)
            console.print(f"\n[dim]Audit: User '{username}' (ID: {user_info['id']}) deleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
            
    except DatabaseError as e:
        console.print(f"[red]✗ Database error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error deleting user: {e}[/red]")


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
    
    # Get actual user groups from database
    try:
        # Get configuration and database connection
        config = get_config()
        db_manager = DatabaseFactory.create_manager(config.database)
        
        if not db_manager.test_connection():
            console.print("[red]✗ Cannot connect to database[/red]")
            return
        
        Session = sessionmaker(bind=db_manager.engine)
        
        with Session() as session:
            # Find the user
            user = session.query(User).filter_by(username=username).first()
            if not user:
                console.print(f"[red]✗ User '{username}' not found[/red]")
                return
            
            # Get user's groups using the group_users association table
            user_groups = []
            
            try:
                # Import group models
                from src.access_control.group_manager import Group, group_users, GroupRole, MembershipStatus
                
                # Query groups through association table
                group_memberships = session.execute(
                    session.query(Group, group_users.c.group_role, group_users.c.status, 
                                group_users.c.joined_at, group_users.c.expires_at)\
                           .join(group_users, Group.id == group_users.c.group_id)\
                           .filter(group_users.c.user_id == user.id)\
                           .statement
                ).fetchall()
                
                for membership in group_memberships:
                    group, group_role, status, joined_at, expires_at = membership
                    
                    # Get group permissions through roles
                    group_permissions = []
                    try:
                        # Get roles assigned to the group
                        from src.access_control.group_manager import group_roles
                        group_role_assignments = session.execute(
                            session.query(Role, group_roles.c.assigned_at, group_roles.c.expires_at)\
                                   .join(group_roles, Role.id == group_roles.c.role_id)\
                                   .filter(group_roles.c.group_id == group.id)\
                                   .statement
                        ).fetchall()
                        
                        for role_assignment in group_role_assignments:
                            role, assigned_at, role_expires_at = role_assignment
                            # Skip expired role assignments
                            if role_expires_at and datetime.now() > role_expires_at:
                                continue
                            
                            # Get permissions for this role
                            role_permissions = role.get_all_permissions()
                            group_permissions.extend(role_permissions)
                        
                    except Exception as perm_error:
                        console.print(f"[yellow]Warning: Could not fetch permissions for group {group.name}: {perm_error}[/yellow]")
                    
                    # Determine membership status
                    is_expired = expires_at and datetime.now() > expires_at
                    effective_status = "expired" if is_expired else status
                    
                    user_groups.append({
                        'group': group.name,
                        'description': group.description or '',
                        'type': group.group_type,
                        'role': group_role,
                        'status': effective_status,
                        'permissions': ', '.join(sorted(set(group_permissions))) if group_permissions else 'None',
                        'joined_at': joined_at.strftime('%Y-%m-%d') if joined_at else 'Unknown',
                        'expires_at': expires_at.strftime('%Y-%m-%d') if expires_at else 'Never'
                    })
                
            except ImportError as e:
                # Groups system not available, show alternative information
                console.print(f"[yellow]Groups system not available: {e}[/yellow]")
                console.print(f"[dim]Showing role-based information instead...[/dim]")
                
                # Show user roles as a fallback
                user_roles_info = []
                for user_role in user.user_roles:
                    role = user_role.role
                    permissions = role.get_all_permissions()
                    
                    user_roles_info.append({
                        'group': f"Role: {role.name}",
                        'description': role.description or '',
                        'type': 'role',
                        'role': 'assigned',
                        'status': 'active' if not user_role.is_expired() else 'expired',
                        'permissions': ', '.join(sorted(permissions)) if permissions else 'None',
                        'joined_at': user_role.assigned_at.strftime('%Y-%m-%d') if user_role.assigned_at else 'Unknown',
                        'expires_at': user_role.expires_at.strftime('%Y-%m-%d') if user_role.expires_at else 'Never'
                    })
                
                user_groups = user_roles_info
            
            # Display results
            if user_groups:
                console.print(f"[green]Found {len(user_groups)} group memberships for '{username}'[/green]")
                
                # Display in detailed format
                for i, group_info in enumerate(user_groups):
                    console.print(f"\n[bold]{i+1}. {group_info['group']}[/bold]")
                    console.print(f"   Description: {group_info['description']}")
                    console.print(f"   Type: {group_info['type']}")
                    console.print(f"   Role: {group_info['role']}")
                    console.print(f"   Status: {group_info['status']}")
                    console.print(f"   Joined: {group_info['joined_at']}")
                    console.print(f"   Expires: {group_info['expires_at']}")
                    console.print(f"   Permissions: {group_info['permissions']}")
                
                # Also show in table format for overview
                console.print(f"\n[dim]Summary table:[/dim]")
                table_columns = ['group', 'type', 'role', 'status', 'permissions']
                display_table(user_groups, title=f"Groups for {username}", columns=table_columns)
                
            else:
                console.print(f"[yellow]User '{username}' is not a member of any groups[/yellow]")
                
                # Show user's direct roles as alternative
                if user.user_roles:
                    console.print(f"[dim]Direct role assignments:[/dim]")
                    for user_role in user.user_roles:
                        role = user_role.role
                        status = 'active' if not user_role.is_expired() else 'expired'
                        console.print(f"  • {role.name} ({status}) - {role.description or 'No description'}")
                else:
                    console.print(f"[yellow]User '{username}' has no roles assigned[/yellow]")
                    
    except DatabaseError as e:
        console.print(f"[red]✗ Database error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error fetching user groups: {e}[/red]")