#!/usr/bin/env python
"""
Download forest species data for Montana directly in State Plane projection (EPSG:2256).
Uses ESRI REST API's native projection capabilities.
"""

import sys
from pathlib import Path
from typing import List, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.api import BigMapRestClient
from bigmap.console import print_info, print_success, print_error
from bigmap.utils.montana_config import load_montana_config

console = Console()


def download_montana_species_state_plane(overwrite: bool = False) -> List[Path]:
    """
    Download species data for Montana region directly in State Plane projection.
    
    Args:
        overwrite: Whether to overwrite existing files
        
    Returns:
        List of downloaded file paths
    """
    # Load configuration
    config = load_montana_config()
    
    # Get parameters from config
    output_dir = config.download_output_dir
    state_plane_bbox = config.state_plane_bbox  # (xmin, ymin, xmax, ymax) in feet
    resolution_ft = config.download_resolution_ft
    species_list = [(s['code'], s['name']) for s in config.species_list]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize BigMap client
    client = BigMapRestClient()
    
    # Show download plan
    table = Table(title="Montana Species Download Plan (Direct State Plane Export)")
    table.add_column("Species Code", style="cyan")
    table.add_column("Species Name", style="green")
    table.add_column("Output File")
    
    for code, name in species_list:
        output_file = output_dir / f"montana_{code}_state_plane.tif"
        table.add_row(code, name, output_file.name)
    
    console.print(table)
    
    console.print(f"\n[cyan]Export Parameters:")
    console.print(f"State Plane BBox (EPSG:2256): {state_plane_bbox[0]:.2f}, {state_plane_bbox[1]:.2f}, {state_plane_bbox[2]:.2f}, {state_plane_bbox[3]:.2f}")
    console.print(f"Resolution: {resolution_ft:.2f} ft")
    console.print(f"Output CRS: EPSG:2256 (NAD83 / Montana State Plane)")
    
    downloaded_files = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Download each species
        task = progress.add_task("Downloading species...", total=len(species_list))
        
        for code, name in species_list:
            output_file = output_dir / f"montana_{code}_state_plane.tif"
            
            if not output_file.exists() or overwrite:
                try:
                    # Export directly in State Plane projection
                    result = client.export_species_raster(
                        species_code=code,
                        bbox=state_plane_bbox,
                        output_path=output_file,
                        pixel_size=resolution_ft,
                        format='tiff',
                        bbox_srs='2256',  # Montana State Plane
                        output_srs='2256'  # Montana State Plane
                    )
                    
                    if result:
                        downloaded_files.append(output_file)
                        print_success(f"Downloaded: {name} ({code})")
                    else:
                        print_error(f"Failed to download {name}")
                except Exception as e:
                    print_error(f"Failed to download {name}: {e}")
            else:
                print_info(f"Skipping existing: {name} ({code})")
                downloaded_files.append(output_file)
            
            progress.update(task, advance=1)
    
    # Summary
    console.print(f"\n[green]Download complete![/green]")
    console.print(f"Downloaded {len(downloaded_files)} files to {output_dir}")
    console.print("\nAll files are natively in Montana State Plane projection (EPSG:2256)")
    console.print("No reprojection was needed!")
    
    # Verify first file to show metadata
    if downloaded_files:
        console.print(f"\n[cyan]Verifying first file metadata...")
        import rasterio
        with rasterio.open(downloaded_files[0]) as src:
            console.print(f"  File: {downloaded_files[0].name}")
            console.print(f"  CRS: {src.crs}")
            console.print(f"  Shape: {src.shape}")
            console.print(f"  Bounds: {src.bounds}")
    
    return downloaded_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Montana forest species data in State Plane projection")
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    # Show configuration summary
    config = load_montana_config()
    console.print("\n[bold cyan]Montana Forest Data Download (Direct State Plane)[/bold cyan]")
    console.print(f"Target CRS: {config.target_crs} (NAD83 / Montana State Plane)")
    console.print(f"Output directory: {config.download_output_dir}")
    console.print("\nUsing ESRI REST API's native projection capabilities")
    console.print("Data will be downloaded directly in State Plane projection")
    
    # Run download
    download_montana_species_state_plane(overwrite=args.overwrite)