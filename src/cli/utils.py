"""
Utility functions for CLI operations.
"""

import click
from typing import Any, Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from pathlib import Path

console = Console()


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        message: Confirmation message
        default: Default value if user just presses enter
        
    Returns:
        True if user confirms, False otherwise
    """
    return Confirm.ask(message, default=default, console=console)


def prompt_for_input(message: str, default: Optional[str] = None, password: bool = False) -> str:
    """
    Prompt user for input.
    
    Args:
        message: Prompt message
        default: Default value
        password: Whether to hide input
        
    Returns:
        User input
    """
    return Prompt.ask(message, default=default, password=password, console=console)


def create_progress_bar(description: str = "Processing") -> Progress:
    """
    Create a rich progress bar.
    
    Args:
        description: Description for the progress bar
        
    Returns:
        Progress bar instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )


def display_table(data: List[dict], title: Optional[str] = None, columns: Optional[List[str]] = None) -> None:
    """
    Display data in a table format.
    
    Args:
        data: List of dictionaries to display
        title: Optional table title
        columns: Optional list of column names to display
    """
    if not data:
        console.print("[yellow]No data to display[/yellow]")
        return
    
    # Auto-detect columns if not provided
    if columns is None:
        columns = list(data[0].keys()) if data else []
    
    table = Table(title=title)
    
    # Add columns
    for col in columns:
        table.add_column(col.replace('_', ' ').title(), style="cyan")
    
    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col, '')) for col in columns])
    
    console.print(table)


def format_error(error: Exception, show_traceback: bool = False) -> str:
    """
    Format error message for display.
    
    Args:
        error: Exception to format
        show_traceback: Whether to include traceback
        
    Returns:
        Formatted error message
    """
    message = f"[red]Error: {str(error)}[/red]"
    
    if show_traceback:
        import traceback
        tb = traceback.format_exc()
        message += f"\n[dim]{tb}[/dim]"
    
    return message


def validate_file_path(path: str, must_exist: bool = True) -> Path:
    """
    Validate and return a Path object.
    
    Args:
        path: File path string
        must_exist: Whether the file must exist
        
    Returns:
        Path object
        
    Raises:
        click.BadParameter: If path is invalid
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise click.BadParameter(f"Path does not exist: {path}")
    
    return path_obj


def validate_directory_path(path: str, must_exist: bool = True, create_if_missing: bool = False) -> Path:
    """
    Validate and return a directory Path object.
    
    Args:
        path: Directory path string
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create directory if missing
        
    Returns:
        Path object
        
    Raises:
        click.BadParameter: If path is invalid
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        if create_if_missing:
            path_obj.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Created directory: {path_obj}[/green]")
        elif must_exist:
            raise click.BadParameter(f"Directory does not exist: {path}")
    elif not path_obj.is_dir():
        raise click.BadParameter(f"Path is not a directory: {path}")
    
    return path_obj


def safe_filename(filename: str) -> str:
    """
    Make a filename safe for use on most filesystems.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    safe = safe.strip(' .')
    # Limit length
    if len(safe) > 200:
        safe = safe[:200]
    
    return safe or 'unnamed_file'


class ProgressCallback:
    """Callback class for tracking progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.progress = None
        self.task = None
    
    def start(self):
        """Start the progress bar."""
        self.progress = create_progress_bar(self.description)
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=self.total)
    
    def update(self, advance: int = 1):
        """Update progress."""
        if self.progress and self.task is not None:
            self.progress.update(self.task, advance=advance)
            self.current += advance
    
    def finish(self):
        """Finish and close the progress bar."""
        if self.progress:
            self.progress.stop()


def handle_keyboard_interrupt(func):
    """Decorator to handle keyboard interrupts gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            return None
    return wrapper