#!/usr/bin/env python
"""
Create Montana forest overview map with timber biomass and town labels.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import zarr.storage
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
import pyproj

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def create_montana_overview_map():
    """
    Create overview map of Montana forest with town labels.
    """
    # Load configuration
    config = load_montana_config()
    
    # Use the clipped zarr
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    
    console.print("[bold cyan]Creating Montana Forest Overview Map[/bold cyan]")
    console.print(f"Using clipped zarr: {zarr_path}")
    
    # Load town/city locations
    locations_csv = Path("config/montana_timber_mill_location.csv")
    locations_df = pd.read_csv(locations_csv)
    
    # Transform coordinates to State Plane
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:2256', always_xy=True)
    
    # Transform all locations
    locations_data = []
    for _, loc in locations_df.iterrows():
        x, y = transformer.transform(loc['lon'], loc['lat'])
        locations_data.append({
            'name': loc['name'],
            'x': x,
            'y': y,
            'type': loc['type'],
            'notes': loc.get('notes', '')
        })
    
    # Open zarr store
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')
    
    # Get timber layer using config index
    biomass = root['biomass']
    timber_data = biomass[config.timber_idx, :, :]
    
    # Get metadata
    bounds = root.attrs['bounds']
    crs = root.attrs['crs']
    
    console.print(f"\n[cyan]Data Info:[/cyan]")
    console.print(f"  Shape: {timber_data.shape}")
    console.print(f"  CRS: {crs}")
    console.print(f"  Bounds: {bounds}")
    
    # Load county boundaries
    shapefile_path = config.county_shapefile
    all_counties = gpd.read_file(shapefile_path)
    counties_gdf = all_counties[all_counties['STATEFP'] == config.state_fips].copy()
    
    # Reproject to State Plane
    if counties_gdf.crs.to_string() != crs:
        counties_gdf = counties_gdf.to_crs(crs)
    
    # Calculate statistics
    valid_data = timber_data[timber_data > 0]
    if len(valid_data) > 0:
        vmin = 0
        vmax = np.percentile(valid_data, 98)
        mean_val = np.mean(valid_data)
        max_val = np.max(valid_data)
        
        console.print(f"\n[cyan]Timber Statistics:[/cyan]")
        console.print(f"  Non-zero pixels: {len(valid_data):,} ({len(valid_data)/timber_data.size*100:.1f}%)")
        console.print(f"  Mean biomass: {mean_val:.1f} Mg/ha")
        console.print(f"  Max biomass: {max_val:.1f} Mg/ha")
        console.print(f"  98th percentile: {vmax:.1f} Mg/ha")
    else:
        vmin, vmax = 0, 1
    
    # Create custom colormap
    colors = ['#ffffff', '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', 
              '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
    cmap = LinearSegmentedColormap.from_list('timber', colors, N=100)
    
    # Calculate extent
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # Create output directory
    output_dir = Path(config.maps_dir) / "montana_overview"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Full State Overview
    console.print("\n[cyan]Creating Montana overview map...[/cyan]")
    create_state_overview(timber_data, extent, bounds, counties_gdf, locations_data, 
                         cmap, vmax, mean_val, valid_data, output_dir)
    
    console.print("\n[bold green]✓ Map created successfully![/bold green]")


def create_state_overview(timber_data, extent, bounds, counties_gdf, locations_data, 
                         cmap, vmax, mean_val, valid_data, output_dir):
    """
    Create full state overview map with all towns labeled.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot timber data
    im = ax.imshow(
        timber_data,
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        origin='upper',
        interpolation='nearest'
    )
    
    # Add county boundaries
    counties_gdf.boundary.plot(
        ax=ax,
        color='#333333',
        linewidth=0.5,
        alpha=0.6
    )
    
    # Add state boundary
    state_boundary = counties_gdf.dissolve()
    state_boundary.boundary.plot(
        ax=ax,
        color='black',
        linewidth=2.5,
        alpha=1.0
    )
    
    # Plot all locations with different markers
    for loc in locations_data:
        if loc['type'] == 'mill':
            # Mill site with star marker
            ax.plot(loc['x'], loc['y'], 'r*', markersize=20, markeredgecolor='black', 
                   markeredgewidth=1.5, zorder=10)
        elif loc['type'] == 'city':
            # Cities with larger circles
            ax.plot(loc['x'], loc['y'], 'ko', markersize=10, 
                   markeredgecolor='white', markeredgewidth=1, zorder=9)
            ax.annotate(
                loc['name'],
                xy=(loc['x'], loc['y']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         alpha=0.9, edgecolor='black')
            )
        else:  # towns
            # Towns with smaller circles
            ax.plot(loc['x'], loc['y'], 'ko', markersize=6, 
                   markeredgecolor='white', markeredgewidth=0.5, zorder=8)
            ax.annotate(
                loc['name'],
                xy=(loc['x'], loc['y']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         alpha=0.8, edgecolor='gray')
            )
    
    # Set extent to data bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # Add title and labels
    ax.set_title('Montana Timber Biomass Overview\nwith Cities and Towns', 
                 fontsize=20, pad=20, fontweight='bold')
    ax.set_xlabel('Easting (feet)', fontsize=12)
    ax.set_ylabel('Northing (feet)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Biomass (Mg/ha)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Format axis labels
    ax.ticklabel_format(style='plain', axis='both')
    ax.tick_params(axis='both', labelsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Add statistics box
    stats_text = (
        f'Total Timber Biomass: {np.sum(valid_data)/1e6:.1f} million Mg\n'
        f'Forest Coverage: {len(valid_data)/timber_data.size*100:.1f}%\n'
        f'Mean Density: {mean_val:.1f} Mg/ha'
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Add CRS info
    ax.text(0.98, 0.02, 'NAD83 / Montana State Plane (ft)\nEPSG:2256', 
            transform=ax.transAxes, fontsize=9,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save map
    output_path = output_dir / "montana_timber_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ State overview saved to: {output_path}[/green]")
    
    plt.close()




if __name__ == "__main__":
    create_montana_overview_map()