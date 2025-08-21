#!/usr/bin/env python
"""
Create a map showing the dominant tree species at each location in Montana.
"""

import sys
from pathlib import Path
import numpy as np
import zarr
import zarr.storage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def create_dominant_species_map():
    """
    Create a map showing dominant species in northwestern Montana.
    """
    # Load configuration
    config = load_montana_config()
    
    # Use the clipped zarr with dominant species layer
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    
    console.print("[bold cyan]Creating Dominant Species Map[/bold cyan]")
    console.print(f"Zarr path: {zarr_path}")
    
    # Open zarr store
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')
    
    # Get dominant species layer using config index
    biomass = root['biomass']
    dominant_data = biomass[config.dominant_species_idx, :, :]
    
    # Get metadata
    bounds = root.attrs['bounds']
    crs = root.attrs['crs']
    species_mapping = root.attrs.get('dominant_species_mapping', {
        '255': 'No forest',
        '0': 'Douglas-fir',
        '1': 'Ponderosa pine',
        '2': 'Jeffrey pine',
        '3': 'True fir',
        '4': 'Western larch'
    })
    
    console.print(f"\n[cyan]Data Info:[/cyan]")
    console.print(f"  Shape: {dominant_data.shape}")
    console.print(f"  Unique values: {np.unique(dominant_data)}")
    
    # Load county boundaries
    shapefile_path = config.county_shapefile
    all_counties = gpd.read_file(shapefile_path)
    counties_gdf = all_counties[all_counties['STATEFP'] == config.state_fips].copy()
    
    # Reproject to State Plane
    if counties_gdf.crs.to_string() != crs:
        counties_gdf = counties_gdf.to_crs(crs)
    
    # Define colors for each species
    colors_map = {
        255: '#f7f7f7',  # No forest (light gray)
        0: '#1b7837',    # Douglas-fir (dark green)
        1: '#a6611a',    # Ponderosa pine (brown)
        2: '#018571',    # Jeffrey pine (teal) - not present
        3: '#80cdc1',    # True fir (light blue) - not present
        4: '#5aae61',    # Western larch (medium green)
    }
    
    # Create mapping for actual values in data
    unique_vals = np.unique(dominant_data)
    colors = [colors_map.get(val, '#000000') for val in unique_vals]
    
    # Create colormap
    cmap = ListedColormap(colors)
    bounds_list = list(unique_vals) + [unique_vals[-1] + 1]
    norm = BoundaryNorm(bounds_list, len(colors))
    
    # Calculate extent
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # Create output directory
    output_dir = Path(config.maps_dir) / "montana_dominant_species"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create northwestern Montana map
    console.print("\n[cyan]Creating northwestern Montana dominant species map...[/cyan]")
    create_northwestern_zoom(dominant_data, bounds, counties_gdf, cmap, norm, 
                           species_mapping, colors_map, output_dir)
    
    console.print("\n[bold green]✓ Map created successfully![/bold green]")


def create_northwestern_zoom(dominant_data, bounds, counties_gdf, cmap, norm, 
                            species_mapping, colors_map, output_dir):
    """
    Create a zoomed map of northwestern Montana showing species transitions.
    """
    console.print("\n[cyan]Creating northwestern Montana zoom...[/cyan]")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Zoom to northwestern quarter
    zoom_bounds = [
        bounds[0],
        bounds[1] + (bounds[3] - bounds[1]) * 0.5,
        bounds[0] + (bounds[2] - bounds[0]) * 0.4,
        bounds[3]
    ]
    
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # Plot data
    im = ax.imshow(
        dominant_data,
        extent=extent,
        cmap=cmap,
        norm=norm,
        origin='upper',
        interpolation='nearest'
    )
    
    # Add county boundaries
    counties_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.5)
    
    # Add county labels for context
    nw_counties = ['Lincoln', 'Flathead', 'Sanders', 'Lake', 'Missoula']
    for _, county in counties_gdf.iterrows():
        if county['NAME'] in nw_counties:
            centroid = county.geometry.centroid
            if zoom_bounds[0] <= centroid.x <= zoom_bounds[2] and \
               zoom_bounds[1] <= centroid.y <= zoom_bounds[3]:
                ax.annotate(
                    county['NAME'],
                    xy=(centroid.x, centroid.y),
                    fontsize=10,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', alpha=0.7, edgecolor='gray')
                )
    
    # Set zoom extent
    ax.set_xlim(zoom_bounds[0], zoom_bounds[2])
    ax.set_ylim(zoom_bounds[1], zoom_bounds[3])
    
    ax.set_title('Northwestern Montana: Dominant Species Detail', 
                 fontsize=16, pad=15, fontweight='bold')
    ax.set_xlabel('Easting (feet)', fontsize=11)
    ax.set_ylabel('Northing (feet)', fontsize=11)
    
    # Add legend
    legend_elements = []
    for val in np.unique(dominant_data):
        legend_elements.append(
            mpatches.Patch(color=colors_map.get(val, '#000000'), label=species_mapping[str(int(val))])
        )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.ticklabel_format(style='plain', axis='both')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = output_dir / "montana_dominant_species_northwest.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Northwestern zoom saved to: {output_path}[/green]")
    
    plt.close()




if __name__ == "__main__":
    create_dominant_species_map()