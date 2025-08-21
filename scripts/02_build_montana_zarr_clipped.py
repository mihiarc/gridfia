#!/usr/bin/env python
"""
Build Montana forest zarr store with correct state boundaries.
Combines the build and clip operations into a single efficient pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import zarr
import zarr.storage
import zarr.codecs
import rasterio
from rasterio.windows import Window
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def build_montana_zarr_clipped():
    """
    Build Montana forest zarr store with clipping to correct boundaries in one step.
    """
    # Load configuration
    config = load_montana_config()
    
    # Get parameters
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    download_dir = config.download_output_dir
    chunk_size = config.chunk_size
    compression = config.compression
    compression_level = config.compression_level
    
    # Montana correct bounds
    mt_bounds = config.state_plane_bbox  # (xmin, ymin, xmax, ymax)
    
    console.print("[bold cyan]Building Montana Forest Zarr Store (Clipped)[/bold cyan]")
    console.print(f"Output: {zarr_path}")
    console.print(f"Chunk size: {chunk_size}")
    console.print(f"Compression: {compression} (level {compression_level})")
    console.print(f"\n[cyan]Montana State Plane bounds:[/cyan]")
    console.print(f"  X: {mt_bounds[0]:,.0f} to {mt_bounds[2]:,.0f} ft")
    console.print(f"  Y: {mt_bounds[1]:,.0f} to {mt_bounds[3]:,.0f} ft")
    
    # Check if all required files exist
    console.print("\n[cyan]Checking input files...[/cyan]")
    
    species_files = []
    for species in config.species_list:
        file_path = download_dir / f"montana_{species['code']}_state_plane.tif"
        if file_path.exists():
            species_files.append((species['code'], species['name'], file_path))
            console.print(f"  ✓ {species['name']}: {file_path.name}")
        else:
            console.print(f"  [red]✗ Missing: {file_path}[/red]")
            return
    
    console.print(f"  [yellow]ℹ Timber layer will be calculated from species data[/yellow]")
    
    # Read first raster to determine clip window
    console.print("\n[cyan]Calculating clip window...[/cyan]")
    
    with rasterio.open(species_files[0][2]) as src:
        current_bounds = src.bounds
        current_transform = src.transform
        full_height = src.height
        full_width = src.width
        crs = src.crs
        dtype = src.dtypes[0]
        
        console.print(f"\n[cyan]Downloaded data bounds:[/cyan]")
        console.print(f"  X: {current_bounds.left:,.0f} to {current_bounds.right:,.0f} ft")
        console.print(f"  Y: {current_bounds.bottom:,.0f} to {current_bounds.top:,.0f} ft")
        
        # Calculate overhang
        western_overhang_miles = (mt_bounds[0] - current_bounds.left) / 5280
        console.print(f"  Western overhang: {western_overhang_miles:.1f} miles")
        
        # Calculate pixel indices for Montana bounds
        col_start = int((mt_bounds[0] - current_bounds.left) / src.res[0])
        col_end = int((mt_bounds[2] - current_bounds.left) / src.res[0])
        row_start = 0
        row_end = full_height
        
        # Calculate new dimensions
        new_width = col_end - col_start
        new_height = row_end - row_start
        
        # Create window for reading
        window = Window(col_start, row_start, new_width, new_height)
        
        # Calculate new transform
        new_transform = rasterio.windows.transform(window, current_transform)
        
        console.print(f"\n[cyan]Clip window:[/cyan]")
        console.print(f"  Original dimensions: {full_width} x {full_height}")
        console.print(f"  Clipped dimensions: {new_width} x {new_height}")
        console.print(f"  Width reduction: {(1 - new_width/full_width)*100:.1f}%")
    
    # Create zarr store
    console.print("\n[cyan]Creating zarr store...[/cyan]")
    
    # Ensure parent directory exists
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create store (Zarr v3)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='w')
    
    # Configure compression
    if compression == 'lz4':
        codec = zarr.codecs.BloscCodec(cname='lz4', clevel=compression_level, shuffle='shuffle')
    else:
        codec = zarr.codecs.BloscCodec(cname=compression, clevel=compression_level, shuffle='shuffle')
    
    # Create main data array (species layers + timber layer)
    num_species = len(species_files)
    num_layers = num_species + 1  # All species + timber layer
    data_array = root.create_array(
        'biomass',
        shape=(num_layers, new_height, new_width),
        chunks=chunk_size,
        dtype=dtype,
        compressors=[codec],
        fill_value=0
    )
    
    # Create metadata arrays
    layer_codes = root.create_array(
        'layer_codes',
        shape=(num_layers,),
        dtype='<U10',
        fill_value=''
    )
    
    layer_names = root.create_array(
        'layer_names',
        shape=(num_layers,),
        dtype='<U100',
        fill_value=''
    )
    
    # Store spatial metadata with CORRECT bounds
    root.attrs['crs'] = crs.to_string()
    root.attrs['transform'] = list(new_transform)
    root.attrs['bounds'] = [mt_bounds[0], mt_bounds[1], mt_bounds[2], mt_bounds[3]]
    root.attrs['width'] = new_width
    root.attrs['height'] = new_height
    root.attrs['num_layers'] = num_layers
    
    # Process layers
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing layers...", total=num_layers)
        
        # Process all species layers (clipped during read)
        for i, (code, name, file_path) in enumerate(species_files):
            console.print(f"Processing Layer {i}: {name}")
            with rasterio.open(file_path) as src:
                # Read only the Montana window
                data = src.read(1, window=window)
                data_array[i, :, :] = data
                console.print(f"  Loaded and clipped: {src.shape} → {data.shape}")
            
            layer_codes[i] = code
            layer_names[i] = name
            progress.update(task, advance=1)
        
        # Last layer: Timber (sum of all species)
        timber_layer_idx = num_species
        console.print(f"\nCalculating Layer {timber_layer_idx}: Timber ({num_species} species total)")
        timber_biomass = np.zeros((new_height, new_width), dtype=np.float32)
        for i in range(num_species):  # Sum all species layers
            timber_biomass += data_array[i, :, :].astype(np.float32)
        data_array[timber_layer_idx, :, :] = timber_biomass.astype(dtype)
        layer_codes[timber_layer_idx] = 'TMBR'
        layer_names[timber_layer_idx] = f'Timber ({num_species} Species Total)'
        progress.update(task, advance=1)
    
    # Create summary table
    console.print("\n[bold green]Zarr store created successfully![/bold green]")
    
    table = Table(title="Montana Forest Zarr Store Layers")
    table.add_column("Index", style="cyan")
    table.add_column("Code", style="yellow")
    table.add_column("Name", style="green")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    
    for i in range(num_layers):
        data = data_array[i, :, :]
        valid_data = data[data > 0]
        if len(valid_data) > 0:
            table.add_row(
                str(i),
                str(layer_codes[i]),
                str(layer_names[i]),
                f"{np.min(valid_data):.1f}",
                f"{np.max(valid_data):.1f}",
                f"{np.mean(valid_data):.1f}"
            )
        else:
            table.add_row(
                str(i),
                str(layer_codes[i]),
                str(layer_names[i]),
                "0",
                "0",
                "0"
            )
    
    console.print(table)
    
    # Save configuration info
    root.attrs['config'] = {
        'project': config.project_name,
        'crs_name': 'NAD83 / Montana State Plane (ft)',
        'resolution_ft': config.download_resolution_ft,
        'clipped': True,
        'original_download_bounds': list(current_bounds),
        'clip_window': {
            'col_start': col_start, 
            'col_end': col_end, 
            'width': new_width, 
            'height': new_height
        }
    }
    
    console.print(f"\n[cyan]Zarr store saved to:[/cyan] {zarr_path}")
    console.print(f"Total size: {sum(f.stat().st_size for f in zarr_path.rglob('*')) / 1024 / 1024:.1f} MB")
    
    # Statistics
    console.print(f"\n[cyan]Processing statistics:[/cyan]")
    console.print(f"  Downloaded data width: {full_width:,} pixels")
    console.print(f"  Clipped data width: {new_width:,} pixels")
    console.print(f"  Data reduction: {(1 - new_width*new_height/(full_width*full_height))*100:.1f}%")
    console.print(f"  Western overhang removed: {western_overhang_miles:.1f} miles")


if __name__ == "__main__":
    build_montana_zarr_clipped()