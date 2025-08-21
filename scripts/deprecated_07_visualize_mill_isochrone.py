#!/usr/bin/env python
"""
Visualize timber mill isochrone with dominant species data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import zarr
import zarr.storage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Point
from rich.console import Console
import pyproj

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def create_mill_isochrone_map():
    """
    Create map showing dominant species with mill location and isochrone coverage.
    """
    # Load configuration
    config = load_montana_config()
    
    console.print("[bold cyan]Creating Mill Isochrone Dominant Species Map[/bold cyan]")
    
    # Load mill locations
    mill_csv = Path("config/montana_timber_mill_location.csv")
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    nearby_towns = mill_df[mill_df['type'] != 'mill']
    
    # Load isochrone
    isochrone_path = Path("output/montana_120min_isochrones.geoparquet")
    isochrone_gdf = gpd.read_parquet(isochrone_path)
    mill_isochrone = isochrone_gdf[isochrone_gdf['poi_id'] == 'custom_0'].iloc[0]
    
    # Transform coordinates to State Plane
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:2256', always_xy=True)
    
    # Transform mill location
    mill_x, mill_y = transformer.transform(mill_site['lon'], mill_site['lat'])
    
    # Transform town locations
    town_coords = []
    for _, town in nearby_towns.iterrows():
        x, y = transformer.transform(town['lon'], town['lat'])
        town_coords.append((x, y, town['name']))
    
    # Transform isochrone
    isochrone_geom = mill_isochrone.geometry
    if isochrone_geom.geom_type == 'Polygon':
        coords = list(isochrone_geom.exterior.coords)
        transformed_coords = [transformer.transform(x, y) for x, y in coords]
        from shapely.geometry import Polygon
        isochrone_state_plane = Polygon(transformed_coords)
    else:
        from shapely.geometry import MultiPolygon, Polygon
        transformed_polygons = []
        for poly in isochrone_geom.geoms:
            coords = list(poly.exterior.coords)
            transformed_coords = [transformer.transform(x, y) for x, y in coords]
            transformed_polygons.append(Polygon(transformed_coords))
        isochrone_state_plane = MultiPolygon(transformed_polygons)
    
    # Load zarr store
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')
    
    biomass = root['biomass']
    bounds = root.attrs['bounds']
    
    # Load county boundaries
    counties_gdf = gpd.read_file(config.county_shapefile)
    counties_gdf = counties_gdf[counties_gdf['STATEFP'] == '30'].copy()
    counties_gdf = counties_gdf.to_crs('EPSG:2256')
    
    # Create output directory
    output_dir = Path(config.maps_dir) / "mill_isochrone"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate extent
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # Calculate isochrone bounds for zooming
    iso_bounds = isochrone_state_plane.bounds
    buffer = 200000  # 200km buffer in feet
    
    # Create Dominant Species Map with Isochrone
    console.print("\n[cyan]Creating dominant species map with isochrone...[/cyan]")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Load dominant species layer using config index
    dominant_data = biomass[config.dominant_species_idx, :, :]
    
    # Define colors for species
    colors_map = {
        255: '#e8e8e8',  # No forest (light gray)
        0: '#1b7837',    # Douglas-fir (dark green)
        1: '#a6611a',    # Ponderosa pine (brown)  
        2: '#018571',    # Jeffrey pine (teal)
        3: '#80cdc1',    # True fir (light blue)
        4: '#5aae61'     # Western larch (medium green)
    }
    
    # Get unique values in the data
    unique_vals = np.unique(dominant_data)
    
    # Create colormap with only the colors we need
    used_colors = [colors_map.get(val, '#000000') for val in unique_vals]
    cmap = ListedColormap(used_colors)
    
    # Create boundaries for discrete color mapping
    from matplotlib.colors import BoundaryNorm
    boundaries = list(unique_vals) + [unique_vals[-1] + 1]
    norm = BoundaryNorm(boundaries, len(used_colors))
    
    # Plot dominant species
    im = ax.imshow(
        dominant_data,
        extent=extent,
        cmap=cmap,
        norm=norm,
        origin='upper',
        alpha=0.8
    )
    
    # Plot isochrone
    if isochrone_state_plane.geom_type == 'Polygon':
        x, y = isochrone_state_plane.exterior.xy
        ax.plot(x, y, 'b-', linewidth=3, label='120-min drive time')
        ax.fill(x, y, 'blue', alpha=0.1)
    else:
        for poly in isochrone_state_plane.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, 'b-', linewidth=3)
            ax.fill(x, y, 'blue', alpha=0.1)
    
    # Add county boundaries
    counties_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
    
    # Plot mill location
    ax.plot(mill_x, mill_y, 'r*', markersize=20, markeredgecolor='black', 
            markeredgewidth=1, label='Proposed mill site')
    
    # Plot nearby towns
    for x, y, name in town_coords:
        ax.plot(x, y, 'ko', markersize=8)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.8))
    
    # Focus on area around isochrone
    ax.set_xlim(iso_bounds[0] - buffer, iso_bounds[2] + buffer)
    ax.set_ylim(iso_bounds[1] - buffer, iso_bounds[3] + buffer)
    
    # Labels and title
    ax.set_title('Timber Mill Site: 120-Minute Drive Time Coverage\nDominant Tree Species', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (feet)', fontsize=12)
    ax.set_ylabel('Northing (feet)', fontsize=12)
    
    # Format axes
    ax.ticklabel_format(style='plain', axis='both')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
    
    # Create species legend
    species_mapping = {
        255: 'No forest',
        0: 'Douglas-fir',
        1: 'Ponderosa pine',
        2: 'Jeffrey pine',
        3: 'True fir',
        4: 'Western larch'
    }
    
    legend_elements = []
    for val in unique_vals:
        legend_elements.append(
            mpatches.Patch(color=colors_map.get(val, '#000000'), label=species_mapping.get(int(val), f'Unknown ({val})'))
        )
    
    # Add mill and isochrone to legend
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                 markerfacecolor='r', markersize=15, 
                                 label='Proposed mill site'))
    legend_elements.append(Line2D([0], [0], color='blue', linewidth=3, 
                                 label='120-min drive time'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    
    output_path = output_dir / "mill_dominant_species_isochrone.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved:[/green] {output_path}")
    
    plt.close()
    
    console.print("\n[bold green]✓ Map created successfully![/bold green]")


if __name__ == "__main__":
    create_mill_isochrone_map()