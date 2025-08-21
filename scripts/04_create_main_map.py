#!/usr/bin/env python
"""
Create the main Montana forest analysis map.
Shows dominant tree species within the 120-minute mill supply area.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
import pyproj
import zarr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def millions_formatter(x, pos):
    """Format axis labels as millions with 'M' suffix."""
    return f'{x/1e6:.1f}M'


def create_isochrone_map(config, zarr_store, counties_gdf, output_dir):
    """Create mill isochrone map with dominant species - MAIN MAP."""
    console.print("\n[bold cyan]Creating Mill Isochrone Map (Main Map)[/bold cyan]")
    
    # Check for isochrone file
    isochrone_path = Path("output/mill_isochrone_120min.geojson")
    if not isochrone_path.exists():
        console.print("[yellow]⚠ Isochrone file not found. Run script 06 first.[/yellow]")
        return
    
    # Load data
    dominant_data = zarr_store['biomass'][config.dominant_species_idx, :, :]
    species_mapping = zarr_store.attrs.get('dominant_species_mapping', {})
    
    # Load isochrone
    isochrone_gdf = gpd.read_file(isochrone_path)
    isochrone_gdf = isochrone_gdf.to_crs(config.target_crs)
    
    # Get the single isochrone record
    iso_120 = isochrone_gdf.iloc[0]
    iso_bounds = iso_120.geometry.bounds
    
    # Define view area with buffer
    buffer_ft = 75000  # 75,000 ft buffer for better context
    view_xmin = iso_bounds[0] - buffer_ft
    view_xmax = iso_bounds[2] + buffer_ft
    view_ymin = iso_bounds[1] - buffer_ft
    view_ymax = iso_bounds[3] + buffer_ft
    
    # Create figure - larger size and higher DPI for main map
    fig, ax = plt.subplots(1, 1, figsize=(16, 14), dpi=300)
    
    # Define species colors (same as dominant species map)
    species_colors = {
        255: '#ffffff',  # No forest
        0: '#1b7837',    # Douglas-fir
        1: '#d95f0e',    # Ponderosa pine
        2: '#7570b3',    # Western larch
        3: '#e7298a',    # Lodgepole pine
        4: '#66a61e',    # Engelmann spruce
        5: '#e6ab02',    # Subalpine fir
    }
    
    # Create colormap
    unique_values = sorted([int(k) for k in species_mapping.keys()])
    colors = [species_colors.get(v, '#cccccc') for v in unique_values]
    cmap = ListedColormap(colors)
    bounds_list = unique_values + [max(unique_values) + 1]
    norm = BoundaryNorm(bounds_list, cmap.N)
    
    # Display dominant species
    bounds = zarr_store.attrs['bounds']
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    im = ax.imshow(dominant_data, extent=extent, origin='upper',
                   cmap=cmap, norm=norm, interpolation='nearest')
    
    # Add isochrone outline only
    ax.add_patch(plt.Polygon(iso_120.geometry.exterior.coords, 
                           fill=False, edgecolor='red', linewidth=3.5,
                           linestyle='--', label='120-minute drive time', zorder=10))
    
    # Set view
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)
    
    # Add county boundaries
    counties_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.5)
    
    # Add county labels for counties in view
    from shapely.geometry import box
    view_box = box(view_xmin, view_ymin, view_xmax, view_ymax)
    for _, county in counties_gdf.iterrows():
        if county.geometry.intersects(view_box):
            centroid = county.geometry.centroid
            if view_xmin <= centroid.x <= view_xmax and view_ymin <= centroid.y <= view_ymax:
                ax.annotate(county.get('NAME', ''), (centroid.x, centroid.y),
                          ha='center', va='center', fontsize=8, 
                          style='italic', alpha=0.6)
    
    # Add mill location
    transformer = pyproj.Transformer.from_crs("EPSG:4326", config.target_crs, always_xy=True)
    mill_coords = transformer.transform(-113.466881, 47.167012)  # from CSV
    ax.scatter(*mill_coords, color='red', s=300, marker='*',
              edgecolor='darkred', linewidth=2, zorder=5, label='Montana Mill Site')
    
    # Add cities and towns from CSV
    csv_path = Path("config/montana_timber_mill_location.csv")
    if csv_path.exists():
        locations_df = pd.read_csv(csv_path)
        for _, loc in locations_df.iterrows():
            if pd.notna(loc['lon']) and pd.notna(loc['lat']) and loc['type'] != 'mill':
                x, y = transformer.transform(loc['lon'], loc['lat'])
                if view_xmin <= x <= view_xmax and view_ymin <= y <= view_ymax:
                    # Different markers for cities vs towns
                    if loc['type'] == 'city':
                        marker_size = 150
                        font_size = 11
                    else:
                        marker_size = 75
                        font_size = 9
                    
                    ax.scatter(x, y, s=marker_size, color='black', marker='o', 
                             edgecolor='white', linewidth=2, zorder=4)
                    ax.annotate(loc['name'], (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=font_size,
                              bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                      edgecolor='black', alpha=0.9, linewidth=1))
    
    # Create legend for species
    legend_elements = []
    for value in unique_values:
        if str(value) in species_mapping and value != 255:
            color = species_colors.get(value, '#cccccc')
            label = species_mapping[str(value)]
            legend_elements.append(mpatches.Patch(color=color, label=label))
    
    # Add isochrone to legend
    legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3,
                                    linestyle='--', label='120-min drive time'))
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                    markerfacecolor='red', markersize=15,
                                    label='Timber Mill'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             title='Legend', title_fontsize=11, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.95)
    
    # Add title and labels
    ax.set_title('Montana Timber Mill Supply Area\nDominant Tree Species within 120-Minute Drive Time', 
                fontsize=20, weight='bold', pad=25, ha='center')
    ax.set_xlabel('State Plane Easting (ft)', fontsize=14)
    ax.set_ylabel('State Plane Northing (ft)', fontsize=14)
    
    # Format axes
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
    
    # Add scale bar
    scale_length_ft = 50000  # 50,000 ft
    scale_length_miles = scale_length_ft / 5280
    scale_x = view_xmin + (view_xmax - view_xmin) * 0.05
    scale_y = view_ymin + (view_ymax - view_ymin) * 0.05
    ax.plot([scale_x, scale_x + scale_length_ft], [scale_y, scale_y],
           'k-', linewidth=4, solid_capstyle='butt')
    ax.text(scale_x + scale_length_ft/2, scale_y + 5000,
           f'{scale_length_miles:.1f} miles', ha='center', va='bottom', 
           fontsize=10, weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='black', alpha=0.9))
    
    # Add north arrow
    arrow_x = view_xmax - (view_xmax - view_xmin) * 0.05
    arrow_y = view_ymax - (view_ymax - view_ymin) * 0.1
    ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 30000),
               ha='center', va='bottom', fontsize=16, weight='bold',
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    
    # Add CRS info
    ax.text(0.99, 0.01, 'CRS: EPSG:2256 - NAD83 / Montana State Plane (ft)', 
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.9))
    
    # Add data source
    ax.text(0.01, 0.01, 'Data: USFS FIA BIGMAP 2018, OpenRouteService', 
            transform=ax.transAxes, fontsize=8, ha='left', va='bottom',
            style='italic', alpha=0.7)
    
    # Save
    output_path = output_dir / "mill_dominant_species_isochrone.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved isochrone map to {output_path}[/green]")


def create_main_map():
    """Create the main mill supply area map with dominant species."""
    # Load configuration
    config = load_montana_config()
    
    # Use output root directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold cyan]Creating Main Montana Forest Analysis Map[/bold cyan]")
    console.print(f"Output directory: {output_dir}")
    
    # Load zarr store
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    if not zarr_path.exists():
        console.print(f"[red]Error: Zarr store not found at {zarr_path}[/red]")
        console.print("[yellow]Run scripts 01-03 first to build the data.[/yellow]")
        return
    
    store = zarr.open(zarr_path, mode='r')
    
    # Load county boundaries
    counties_path = config.county_shapefile
    if not counties_path.exists():
        console.print(f"[red]Error: County shapefile not found at {counties_path}[/red]")
        return
    
    # Load and filter Montana counties
    counties_gdf = gpd.read_file(counties_path)
    counties_gdf = counties_gdf[counties_gdf['STATEFP'] == config.state_fips]
    counties_gdf = counties_gdf.to_crs(config.target_crs)
    
    # Create the main map
    create_isochrone_map(config, store, counties_gdf, output_dir)
    
    console.print("\n[bold green]✓ Main map created successfully![/bold green]")
    console.print(f"Map saved to: {output_dir}/mill_dominant_species_isochrone.png")


if __name__ == "__main__":
    create_main_map()