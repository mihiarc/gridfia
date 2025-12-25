"""
Console utilities using Rich for beautiful terminal output.

This module provides enhanced console output, progress tracking,
and formatted displays using the Rich library.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


# Global console instance
console = Console()


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a formatted header with title and optional subtitle."""
    console.rule(f"[bold blue]{title}")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", justify="center")
    console.print()


def print_success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"✅ [bold green]{message}[/bold green]")


def print_error(message: str) -> None:
    """Print an error message with red X."""
    console.print(f"❌ [bold red]{message}[/bold red]")


def print_warning(message: str) -> None:
    """Print a warning message with yellow warning sign."""
    console.print(f"⚠️  [bold yellow]{message}[/bold yellow]")


def print_info(message: str) -> None:
    """Print an info message with blue info icon."""
    console.print(f"ℹ️  [bold blue]{message}[/bold blue]")


def print_step(step_number: int, total_steps: int, description: str) -> None:
    """Print a numbered step with progress indication."""
    console.print(f"[bold cyan]Step {step_number}/{total_steps}:[/bold cyan] {description}")


def create_species_table(species_data: List[Dict[str, Any]]) -> Table:
    """
    Create a formatted table for species analysis results.
    
    Args:
        species_data: List of dictionaries containing species information
        
    Returns:
        Rich Table object
    """
    table = Table(title="Species Analysis Results", show_header=True, header_style="bold magenta")
    
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Species Code", style="cyan")
    table.add_column("Species Name", style="green")
    table.add_column("Coverage %", justify="right", style="yellow")
    table.add_column("Pixels", justify="right", style="blue")
    table.add_column("Mean Biomass", justify="right", style="red")
    table.add_column("Max Biomass", justify="right", style="red")
    
    for i, species in enumerate(species_data, 1):
        table.add_row(
            str(i),
            species.get('code', 'Unknown'),
            species.get('name', 'Unknown'),
            f"{species.get('coverage_pct', 0):.2f}",
            f"{species.get('pixels', 0):,}",
            f"{species.get('mean_biomass', 0):.1f}",
            f"{species.get('max_biomass', 0):.1f}"
        )
    
    return table


def create_processing_summary(stats: Dict[str, Any]) -> Panel:
    """
    Create a summary panel for processing results.
    
    Args:
        stats: Dictionary containing processing statistics
        
    Returns:
        Rich Panel object
    """
    content = []
    
    # Basic stats
    if 'total_files' in stats:
        content.append(f"📁 Total files processed: [bold]{stats['total_files']:,}[/bold]")
    
    if 'successful' in stats:
        content.append(f"✅ Successful: [bold green]{stats['successful']:,}[/bold green]")
    
    if 'failed' in stats:
        content.append(f"❌ Failed: [bold red]{stats['failed']:,}[/bold red]")
    
    if 'elapsed_time' in stats:
        content.append(f"⏱️  Total time: [bold]{stats['elapsed_time']:.1f}s[/bold]")
    
    if 'output_size_mb' in stats:
        content.append(f"💾 Output size: [bold]{stats['output_size_mb']:.1f} MB[/bold]")
    
    if 'compression_ratio' in stats:
        content.append(f"🗜️  Compression: [bold]{stats['compression_ratio']:.1f}x[/bold]")
    
    return Panel(
        "\n".join(content),
        title="Processing Summary",
        border_style="green",
        padding=(1, 2)
    )


def create_file_tree(root_path: Union[str, Path], max_depth: int = 3) -> Tree:
    """
    Create a file tree representation of a directory.
    
    Args:
        root_path: Root directory path
        max_depth: Maximum depth to traverse
        
    Returns:
        Rich Tree object
    """
    root_path = Path(root_path)
    tree = Tree(f"📁 [bold blue]{root_path.name}[/bold blue]")
    
    def add_files(current_path: Path, current_tree: Tree, depth: int) -> None:
        if depth >= max_depth:
            return
        
        try:
            items = sorted(current_path.iterdir())
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            # Add directories first
            for directory in dirs:
                if directory.name.startswith('.'):
                    continue  # Skip hidden directories
                
                dir_node = current_tree.add(f"📁 [cyan]{directory.name}[/cyan]")
                add_files(directory, dir_node, depth + 1)
            
            # Add files
            for file in files[:10]:  # Limit to first 10 files
                if file.name.startswith('.'):
                    continue  # Skip hidden files
                
                # Choose icon based on file extension
                if file.suffix in ['.py']:
                    icon = "🐍"
                elif file.suffix in ['.tif', '.tiff']:
                    icon = "🗺️"
                elif file.suffix in ['.zarr']:
                    icon = "📦"
                elif file.suffix in ['.json', '.yaml', '.yml']:
                    icon = "⚙️"
                else:
                    icon = "📄"
                
                size_mb = file.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    size_str = f" [dim]({size_mb:.1f} MB)[/dim]"
                else:
                    size_str = ""
                
                current_tree.add(f"{icon} [green]{file.name}[/green]{size_str}")
            
            if len(files) > 10:
                current_tree.add(f"[dim]... and {len(files) - 10} more files[/dim]")
                
        except PermissionError:
            current_tree.add("[red]Permission denied[/red]")
    
    add_files(root_path, tree, 0)
    return tree


def create_progress_tracker(description: str = "Processing") -> Progress:
    """
    Create a Rich progress tracker with multiple columns.
    
    Args:
        description: Description of the progress task
        
    Returns:
        Rich Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    )


def display_configuration(config_dict: Dict[str, Any]) -> None:
    """
    Display configuration settings in a formatted way.
    
    Args:
        config_dict: Configuration dictionary to display
    """
    table = Table(title="Configuration Settings", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="yellow", max_width=40)
    
    def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_config = flatten_dict(config_dict)
    
    for key, value in flat_config.items():
        # Format values nicely
        if isinstance(value, Path):
            value_str = str(value)
        elif isinstance(value, bool):
            value_str = "✅ Yes" if value else "❌ No"
        elif isinstance(value, (list, tuple)):
            value_str = f"[{len(value)} items]"
        else:
            value_str = str(value)
        
        table.add_row(key, value_str, "")
    
    console.print(table)


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation with a yes/no prompt.
    
    Args:
        message: Confirmation message
        default: Default value if user just presses Enter
        
    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    response = console.input(f"{message} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes', 'true', '1']


def print_package_info() -> None:
    """Print GridFIA package information in a nice format."""
    from gridfia import __version__, __author__, __email__

    info_panel = Panel(
        f"""[bold blue]GridFIA[/bold blue] v{__version__}

Spatial Raster Analysis for USDA Forest Service BIGMAP Data
Part of the FIA Python Ecosystem

[dim]Author:[/dim] {__author__}
[dim]Email:[/dim] {__email__}
[dim]License:[/dim] MIT

🌲 Analyze forest biomass at 30m resolution
📊 Calculate species diversity metrics
🗺️  Create publication-ready visualizations
📦 Efficient Zarr-based data storage

[dim]Ecosystem:[/dim] PyFIA | GridFIA | PyFVS | AskFIA""",
        title="Package Information",
        border_style="blue",
        padding=(1, 2)
    )

    console.print(info_panel) 