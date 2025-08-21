#!/usr/bin/env python
"""
Create a boundary overview map showing state boundary, isochrone, county boundaries, and mill location.
"""

import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pyproj
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def millions_formatter(x, pos):
    """Format axis labels as millions with 'M' suffix."""
    return f'{x/1e6:.1f}M'


def create_boundary_overview_map():
    """Create boundary overview map with state, counties, isochrone, and mill location."""
    # Load configuration
    config = load_montana_config()
    
    console.print("[bold cyan]Creating Montana Boundary Overview Map[/bold cyan]")
    
    # Output path
    output_path = Path("output/montana_boundary_overview.png")
    
    # Load county boundaries
    counties_path = config.county_shapefile
    if not counties_path.exists():
        console.print(f"[red]Error: County shapefile not found at {counties_path}[/red]")
        return
    
    # Load and filter Montana counties
    console.print("Loading county boundaries...")
    counties_gdf = gpd.read_file(counties_path)
    counties_gdf = counties_gdf[counties_gdf['STATEFP'] == config.state_fips]
    counties_gdf = counties_gdf.to_crs(config.target_crs)
    
    # Create state boundary by dissolving counties
    console.print("Creating state boundary...")
    state_boundary = counties_gdf.dissolve()
    
    # Load isochrone if available
    isochrone_path = Path("output/mill_isochrone_120min.geojson")
    isochrone_gdf = None
    if isochrone_path.exists():
        console.print("Loading isochrone data...")
        isochrone_gdf = gpd.read_file(isochrone_path)
        isochrone_gdf = isochrone_gdf.to_crs(config.target_crs)
        # Get the single isochrone record
        isochrone_120 = isochrone_gdf.iloc[0]
    else:
        console.print("[yellow]⚠ Isochrone file not found - map will show boundaries only[/yellow]")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)
    
    # Plot county boundaries (filled with light gray)
    counties_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#666666', linewidth=0.5)
    
    # Plot state boundary (thick black line)
    state_boundary.boundary.plot(ax=ax, color='black', linewidth=3, label='State Boundary')
    
    # Plot isochrone if available
    if isochrone_gdf is not None:
        # Fill isochrone area with semi-transparent color
        ax.add_patch(plt.Polygon(isochrone_120.geometry.exterior.coords, 
                               facecolor='lightblue', edgecolor='blue', 
                               linewidth=2.5, alpha=0.3, label='120-minute Drive Time'))
        # Add isochrone boundary
        ax.plot(*isochrone_120.geometry.exterior.xy, color='blue', linewidth=2.5, linestyle='--')
    
    # Add mill location
    transformer = pyproj.Transformer.from_crs("EPSG:4326", config.target_crs, always_xy=True)
    mill_coords = transformer.transform(-113.466881, 47.167012)  # from CSV
    ax.scatter(*mill_coords, color='red', s=400, marker='*',
              edgecolor='darkred', linewidth=2, zorder=5, label='Proposed Mill Site')
    
    # Add mill label
    ax.annotate('Proposed Mill Site', mill_coords, xytext=(10, 10), 
               textcoords='offset points', fontsize=12, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                        edgecolor='darkred', alpha=0.8),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                             color='darkred', linewidth=2))
    
    # Load and add cities/towns from CSV
    csv_path = Path("config/montana_timber_mill_location.csv")
    if csv_path.exists():
        import pandas as pd
        locations_df = pd.read_csv(csv_path)
        for _, loc in locations_df.iterrows():
            if pd.notna(loc['lon']) and pd.notna(loc['lat']) and loc['type'] != 'mill':
                x, y = transformer.transform(loc['lon'], loc['lat'])
                # Different markers for cities vs towns
                if loc['type'] == 'city':
                    marker_size = 120
                    font_size = 11
                    marker = 'o'
                else:
                    marker_size = 60
                    font_size = 9
                    marker = 's'
                
                ax.scatter(x, y, s=marker_size, color='black', marker=marker, 
                         edgecolor='white', linewidth=1.5, zorder=4)
                ax.annotate(loc['name'], (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=font_size,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor='black', alpha=0.8))
    
    # Add county labels for major counties
    major_counties = ['FLATHEAD', 'LINCOLN', 'SANDERS', 'LAKE', 'MISSOULA', 
                     'POWELL', 'LEWIS AND CLARK', 'CASCADE', 'GLACIER']
    for _, county in counties_gdf.iterrows():
        if county.get('NAME', '').upper() in major_counties:
            centroid = county.geometry.centroid
            ax.annotate(county.get('NAME', ''), (centroid.x, centroid.y),
                      ha='center', va='center', fontsize=9, 
                      style='italic', alpha=0.7)
    
    # Set map extent
    bounds = state_boundary.total_bounds  # [minx, miny, maxx, maxy]
    x_buffer = (bounds[2] - bounds[0]) * 0.05
    y_buffer = (bounds[3] - bounds[1]) * 0.05
    ax.set_xlim(bounds[0] - x_buffer, bounds[2] + x_buffer)
    ax.set_ylim(bounds[1] - y_buffer, bounds[3] + y_buffer)
    
    # Add title and labels
    ax.set_title('Montana Forest Analysis - Geographic Overview', 
                fontsize=22, weight='bold', pad=20)
    ax.set_xlabel('State Plane Easting (ft)', fontsize=14)
    ax.set_ylabel('State Plane Northing (ft)', fontsize=14)
    
    # Format axes
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=3, 
                                    label='State Boundary'))
    legend_elements.append(plt.Line2D([0], [0], color='#666666', linewidth=0.5, 
                                    label='County Boundaries'))
    if isochrone_gdf is not None:
        legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=2.5, 
                                        linestyle='--', label='120-minute Drive Time'))
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                    markerfacecolor='red', markersize=15, 
                                    label='Proposed Mill Site'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='black', markersize=10, 
                                    label='Cities'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor='black', markersize=8, 
                                    label='Towns'))
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
             frameon=True, fancybox=True, shadow=True)
    
    # Add scale bar
    scale_length_ft = 200000  # 200,000 ft
    scale_length_miles = scale_length_ft / 5280
    scale_x = bounds[0] + (bounds[2] - bounds[0]) * 0.05
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.05
    ax.plot([scale_x, scale_x + scale_length_ft], [scale_y, scale_y],
           'k-', linewidth=4)
    ax.text(scale_x + scale_length_ft/2, scale_y + 20000,
           f'{scale_length_miles:.0f} miles', ha='center', va='bottom', 
           fontsize=10, weight='bold')
    
    # Add north arrow
    arrow_x = bounds[2] - (bounds[2] - bounds[0]) * 0.05
    arrow_y = bounds[3] - (bounds[3] - bounds[1]) * 0.1
    ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 100000),
               ha='center', va='bottom', fontsize=16, weight='bold',
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Add CRS info
    ax.text(0.99, 0.01, 'CRS: EPSG:2256 - NAD83 / Montana State Plane (ft)', 
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Add data source note
    source_text = "Data Sources: TIGER/Line Shapefiles (Counties), OpenRouteService (Isochrone)"
    ax.text(0.01, 0.01, source_text, transform=ax.transAxes, fontsize=8, 
            ha='left', va='bottom', style='italic', alpha=0.7)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"\n[bold green]✓ Boundary overview map saved to:[/bold green] {output_path}")
    console.print("Map features:")
    console.print("  - Montana state boundary (thick black line)")
    console.print("  - County boundaries (gray lines)")
    if isochrone_gdf is not None:
        console.print("  - 120-minute drive time isochrone (blue dashed line)")
    console.print("  - Proposed mill location (red star)")
    console.print("  - Cities and towns (circles and squares)")
    console.print("  - Scale bar and north arrow")


if __name__ == "__main__":
    create_boundary_overview_map()