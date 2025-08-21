#!/usr/bin/env python
"""
Visualize the generated mill isochrones on a map.
"""

import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()

def visualize_isochrones():
    """Create a visualization of all generated isochrones."""
    
    # Load configuration
    config = load_montana_config()
    
    console.print("[bold cyan]Visualizing Mill Isochrones[/bold cyan]")
    
    # Load county boundaries for context
    counties_path = config.county_shapefile
    counties_gdf = gpd.read_file(counties_path)
    counties_gdf = counties_gdf[counties_gdf['STATEFP'] == config.state_fips]
    
    # Load isochrones
    isochrone_dir = Path("output/isochrones")
    isochrone_files = sorted(isochrone_dir.glob("mill_isochrone_*min.geojson"))
    
    # Also check for the 120min isochrone in the main output directory
    main_120_file = Path("output/mill_isochrone_120min.geojson")
    if main_120_file.exists() and main_120_file not in isochrone_files:
        isochrone_files.append(main_120_file)
        isochrone_files = sorted(isochrone_files, key=lambda x: int(x.stem.split('_')[-1].replace('min', '')))
    
    # Filter out 45min isochrone
    isochrone_files = [f for f in isochrone_files if '45min' not in f.name]
    
    if not isochrone_files:
        console.print("[red]No isochrone files found![/red]")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
    
    # Plot county boundaries
    counties_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
    
    # Load and plot isochrones
    colors = ['#ff0000', '#ff6600', '#ffcc00', '#66ff00', '#0066ff']  # Red to blue gradient
    alphas = [0.6, 0.5, 0.4, 0.3, 0.2]  # Decreasing transparency
    
    for i, iso_file in enumerate(isochrone_files):
        console.print(f"Loading {iso_file.name}")
        iso_gdf = gpd.read_file(iso_file)
        
        # Extract time from properties
        time_min = iso_gdf['travel_time_minutes'].iloc[0]
        
        # Plot with appropriate color and transparency
        iso_gdf.plot(ax=ax, 
                    color=colors[i % len(colors)], 
                    alpha=alphas[i % len(alphas)],
                    edgecolor='black',
                    linewidth=1.5,
                    label=f'{time_min} minutes')
    
    # Load mill location
    mill_csv = Path("config/montana_timber_mill_location.csv")
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    
    # Plot mill location
    ax.scatter(mill_site['lon'], mill_site['lat'], 
              color='red', s=200, marker='*',
              edgecolor='darkred', linewidth=2, 
              zorder=10, label='Mill Site')
    
    # Add mill label
    ax.annotate(mill_site['name'], 
               (mill_site['lon'], mill_site['lat']),
               xytext=(10, 10), textcoords='offset points',
               fontsize=10, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='yellow', 
                        edgecolor='darkred', 
                        alpha=0.8))
    
    # Load and plot nearby towns
    towns = mill_df[mill_df['type'] != 'mill']
    for _, town in towns.iterrows():
        ax.scatter(town['lon'], town['lat'], 
                  s=50, color='black', marker='o',
                  edgecolor='white', linewidth=1)
        ax.annotate(town['name'], 
                   (town['lon'], town['lat']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)
    
    # Set map extent based on largest isochrone
    all_bounds = []
    for iso_file in isochrone_files:
        iso_gdf = gpd.read_file(iso_file)
        all_bounds.append(iso_gdf.total_bounds)
    
    # Get overall bounds with buffer
    all_bounds = pd.DataFrame(all_bounds)
    buffer = 0.5  # degrees
    ax.set_xlim(all_bounds[0].min() - buffer, all_bounds[2].max() + buffer)
    ax.set_ylim(all_bounds[1].min() - buffer, all_bounds[3].max() + buffer)
    
    # Add title and labels
    ax.set_title('Montana Timber Mill Drive-Time Isochrones', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10, 
             title='Drive Time', title_fontsize=11,
             frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add scale note
    ax.text(0.02, 0.02, 
           'Isochrones show areas reachable within specified drive times\n' +
           'Generated using SocialMapper with OpenStreetMap data',
           transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', 
                    edgecolor='gray',
                    alpha=0.8))
    
    # Save figure
    output_path = Path("output/maps/mill_isochrones_visualization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved visualization to {output_path}[/green]")

if __name__ == "__main__":
    visualize_isochrones()