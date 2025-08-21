#!/usr/bin/env python
"""
Add a dominant species layer to the Montana forest zarr store.
This layer shows which species has the highest biomass at each location.
"""

import sys
from pathlib import Path
import numpy as np
import zarr
import zarr.storage
import zarr.codecs
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def add_dominant_species_layer():
    """
    Add dominant species layer to the zarr store.
    """
    # Load configuration
    config = load_montana_config()
    
    # Use the clipped zarr
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    
    console.print("[bold cyan]Adding Dominant Species Layer to Zarr Store[/bold cyan]")
    console.print(f"Zarr path: {zarr_path}")
    
    if not zarr_path.exists():
        console.print(f"[red]Error: Zarr store not found at {zarr_path}[/red]")
        return
    
    # Open zarr store in read-write mode
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r+')
    
    # Get existing arrays
    biomass = root['biomass']
    layer_codes = root['layer_codes']
    layer_names = root['layer_names']
    
    # Check current state
    expected_layers_without_dominant = config.timber_idx + 1  # Species layers + timber
    current_layers = root.attrs.get('num_layers', expected_layers_without_dominant)
    console.print(f"\n[cyan]Current state:[/cyan]")
    console.print(f"  Number of layers: {current_layers}")
    console.print(f"  Biomass shape: {biomass.shape}")
    
    # Check if dominant species layer already exists
    if current_layers > expected_layers_without_dominant or biomass.shape[0] > expected_layers_without_dominant:
        console.print("[yellow]Dominant species layer may already exist. Checking...[/yellow]")
        if 'DOMN' in [str(layer_codes[i]) for i in range(min(biomass.shape[0], layer_codes.shape[0]))]:
            console.print("[red]Dominant species layer already exists! Exiting.[/red]")
            return
    
    # Get species data using config indices
    console.print("\n[cyan]Loading species data...[/cyan]")
    num_species = config.species_end_idx - config.species_start_idx + 1
    species_data = np.zeros((num_species, biomass.shape[1], biomass.shape[2]), dtype=biomass.dtype)
    species_info = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading species layers...", total=num_species)
        
        for i in range(num_species):
            layer_idx = config.species_start_idx + i
            species_data[i] = biomass[layer_idx, :, :]
            species_info.append({
                'index': layer_idx,
                'code': str(layer_codes[layer_idx]),
                'name': str(layer_names[layer_idx])
            })
            progress.update(task, advance=1)
    
    # Calculate dominant species
    console.print("\n[cyan]Calculating dominant species...[/cyan]")
    
    # Find maximum biomass across species at each pixel
    dominant = np.argmax(species_data, axis=0).astype(np.uint8)
    
    # Set pixels with no biomass to 255 (no dominant species)
    total_biomass = np.sum(species_data, axis=0)
    dominant[total_biomass == 0] = 255
    
    # Calculate statistics
    console.print("\n[cyan]Dominant species statistics:[/cyan]")
    stats_table = Table(title="Dominant Species Coverage")
    stats_table.add_column("Species", style="green")
    stats_table.add_column("Code", style="yellow")
    stats_table.add_column("Pixels", justify="right")
    stats_table.add_column("Area %", justify="right")
    
    total_forest_pixels = np.sum(dominant < 255)
    
    # No forest
    no_forest_pixels = np.sum(dominant == 255)
    stats_table.add_row(
        "No forest",
        "255",
        f"{no_forest_pixels:,}",
        f"{no_forest_pixels / dominant.size * 100:.1f}%"
    )
    
    # Each species
    for i, info in enumerate(species_info):
        species_pixels = np.sum(dominant == i)
        if total_forest_pixels > 0:
            forest_pct = species_pixels / total_forest_pixels * 100
        else:
            forest_pct = 0
        stats_table.add_row(
            info['name'],
            info['code'],
            f"{species_pixels:,}",
            f"{species_pixels / dominant.size * 100:.1f}% ({forest_pct:.1f}% of forest)"
        )
    
    console.print(stats_table)
    
    # Resize arrays if needed to add new layer
    if biomass.shape[0] <= config.dominant_species_idx:
        console.print("\n[cyan]Resizing arrays to add new layer...[/cyan]")
        
        # Create new arrays with one additional layer
        new_shape = (config.dominant_species_idx + 1, biomass.shape[1], biomass.shape[2])
        
        # Create temporary arrays
        new_biomass_data = np.zeros(new_shape, dtype=biomass.dtype)
        new_codes = [''] * new_shape[0]
        new_names = [''] * new_shape[0]
        
        # Copy existing data
        for i in range(biomass.shape[0]):
            new_biomass_data[i] = biomass[i, :, :]
            try:
                new_codes[i] = str(layer_codes[i])
                new_names[i] = str(layer_names[i])
            except IndexError:
                new_codes[i] = ''
                new_names[i] = ''
        
        # Add dominant species layer
        new_biomass_data[config.dominant_species_idx] = dominant
        new_codes[config.dominant_species_idx] = 'DOMN'
        new_names[config.dominant_species_idx] = 'Dominant Species'
        
        # Delete old arrays and create new ones
        console.print("[yellow]Recreating arrays with new size...[/yellow]")
        
        # Get compression settings
        compression = root.attrs.get('compression', 'lz4')
        compression_level = root.attrs.get('compression_level', 5)
        chunk_size = biomass.chunks
        
        # Delete old arrays
        del root['biomass']
        del root['layer_codes']
        del root['layer_names']
        
        # Create new arrays
        if compression == 'lz4':
            codec = zarr.codecs.BloscCodec(cname='lz4', clevel=compression_level, shuffle='shuffle')
        else:
            codec = zarr.codecs.BloscCodec(cname=compression, clevel=compression_level, shuffle='shuffle')
        
        # Create new biomass array
        new_biomass = root.create_array(
            'biomass',
            shape=new_shape,
            chunks=chunk_size,
            dtype=biomass.dtype,
            compressors=[codec],
            fill_value=0
        )
        
        # Create new metadata arrays
        new_layer_codes = root.create_array(
            'layer_codes',
            shape=(new_shape[0],),
            dtype='<U10',
            fill_value=''
        )
        
        new_layer_names = root.create_array(
            'layer_names',
            shape=(new_shape[0],),
            dtype='<U100',
            fill_value=''
        )
        
        # Write data
        console.print("Writing data to new arrays...")
        for i in range(new_shape[0]):
            new_biomass[i, :, :] = new_biomass_data[i]
            new_layer_codes[i] = new_codes[i]
            new_layer_names[i] = new_names[i]
    
    else:
        # Just add to existing layer
        console.print(f"\n[cyan]Adding dominant species to layer {config.dominant_species_idx}...[/cyan]")
        biomass[config.dominant_species_idx, :, :] = dominant
        layer_codes[config.dominant_species_idx] = 'DOMN'
        layer_names[config.dominant_species_idx] = 'Dominant Species'
    
    # Update metadata to reflect actual number of layers
    if 'new_biomass' in locals():
        actual_num_layers = new_biomass.shape[0]
    else:
        actual_num_layers = biomass.shape[0]
    root.attrs['num_layers'] = actual_num_layers
    
    # Add dominant species mapping info based on actual species
    species_mapping = {'255': 'No forest'}
    for i, info in enumerate(species_info):
        species_mapping[str(i)] = info['name']
    root.attrs['dominant_species_mapping'] = species_mapping
    
    console.print("\n[bold green]✓ Dominant species layer added successfully![/bold green]")
    console.print(f"  Layer index: {config.dominant_species_idx}")
    console.print(f"  Layer code: DOMN")
    console.print(f"  Layer name: Dominant Species")
    
    # Display final summary
    console.print("\n[cyan]Final zarr store structure:[/cyan]")
    final_table = Table()
    final_table.add_column("Index", style="cyan")
    final_table.add_column("Code", style="yellow")
    final_table.add_column("Name", style="green")
    
    # Show all layers in the zarr store
    final_biomass = root['biomass']
    final_layer_codes = root['layer_codes']
    final_layer_names = root['layer_names']
    
    for i in range(final_biomass.shape[0]):
        try:
            code = str(final_layer_codes[i])
            name = str(final_layer_names[i])
            final_table.add_row(str(i), code, name)
        except IndexError:
            # Handle case where metadata arrays might be smaller
            final_table.add_row(str(i), "?", "Unknown")
    
    console.print(final_table)


if __name__ == "__main__":
    add_dominant_species_layer()