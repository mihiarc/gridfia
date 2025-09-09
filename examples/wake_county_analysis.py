#!/usr/bin/env python3
"""
Complete Wake County Forest Analysis Example
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.plot import show
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console

console = Console()

def add_zarr_metadata():
    """Add required metadata to the Zarr store."""
    print("Adding metadata to Zarr store...")
    
    store_path = Path("output/data/wake_county.zarr")
    z = zarr.open(str(store_path), mode='r+')
    
    # Add required metadata to the group
    z.attrs['species_codes'] = ['0068', '0131', 'TOTAL']
    z.attrs['species_names'] = ['Red maple', 'Loblolly pine', 'Total Biomass']
    z.attrs['crs'] = 'EPSG:3857'  # Web Mercator
    z.attrs['units'] = 'Mg/ha'
    z.attrs['description'] = 'Wake County, NC Forest Biomass Analysis'
    
    # Also add to biomass array if it exists
    if 'biomass' in z:
        z['biomass'].attrs.update(z.attrs)
    
    print(f"✅ Added metadata to {store_path}")
    return z


def analyze_biomass_data():
    """Analyze the biomass data and print statistics."""
    print("\n" + "=" * 60)
    print("Wake County Forest Biomass Analysis")
    print("=" * 60)
    
    # Open the Zarr store
    z = zarr.open("output/data/wake_county.zarr", mode='r')
    
    # Get biomass array
    data = z['biomass'][:]
    
    print(f"\nData shape: {data.shape}")
    print(f"Species: {z.attrs.get('species_names', ['Unknown'])}")
    
    # Calculate statistics for each species
    for i, species_name in enumerate(z.attrs.get('species_names', [])):
        species_data = data[i]
        
        # Filter out no-data values
        valid_data = species_data[species_data > 0]
        
        if len(valid_data) > 0:
            print(f"\n{species_name}:")
            print(f"  Non-zero pixels: {len(valid_data):,} ({100*len(valid_data)/species_data.size:.1f}%)")
            print(f"  Mean biomass: {valid_data.mean():.2f} Mg/ha")
            print(f"  Max biomass: {valid_data.max():.2f} Mg/ha")
            print(f"  Total biomass: {valid_data.sum():,.0f} Mg")
    
    return data


def create_biomass_map():
    """Create a visualization map of forest biomass."""
    print("\n" + "=" * 60)
    print("Creating Biomass Visualization Map")
    print("=" * 60)
    
    # Load data
    z = zarr.open("output/data/wake_county.zarr", mode='r')
    biomass = z['biomass']
    
    # Get total biomass (last layer)
    total_biomass = biomass[2]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Wake County, North Carolina - Forest Biomass Analysis', fontsize=16, fontweight='bold')
    
    # Define colormaps
    biomass_cmap = 'YlGn'
    
    # Plot 1: Red Maple Biomass
    ax1 = axes[0, 0]
    red_maple = biomass[0]
    im1 = ax1.imshow(red_maple, cmap=biomass_cmap, vmin=0, vmax=np.percentile(red_maple[red_maple > 0], 98) if np.any(red_maple > 0) else 1)
    ax1.set_title('Red Maple (Acer rubrum) Biomass', fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Biomass (Mg/ha)', fraction=0.046, pad=0.04)
    
    # Plot 2: Loblolly Pine Biomass
    ax2 = axes[0, 1]
    loblolly = biomass[1]
    im2 = ax2.imshow(loblolly, cmap=biomass_cmap, vmin=0, vmax=np.percentile(loblolly[loblolly > 0], 98) if np.any(loblolly > 0) else 1)
    ax2.set_title('Loblolly Pine (Pinus taeda) Biomass', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Biomass (Mg/ha)', fraction=0.046, pad=0.04)
    
    # Plot 3: Total Biomass
    ax3 = axes[1, 0]
    im3 = ax3.imshow(total_biomass, cmap=biomass_cmap, vmin=0, vmax=np.percentile(total_biomass[total_biomass > 0], 98) if np.any(total_biomass > 0) else 1)
    ax3.set_title('Total Forest Biomass', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='Biomass (Mg/ha)', fraction=0.046, pad=0.04)
    
    # Plot 4: Species Dominance
    ax4 = axes[1, 1]
    # Create dominance map (which species has more biomass)
    dominance = np.zeros_like(red_maple)
    dominance[red_maple > loblolly] = 1  # Red maple dominant
    dominance[loblolly > red_maple] = 2  # Loblolly dominant
    dominance[(red_maple == 0) & (loblolly == 0)] = 0  # No forest
    
    # Create custom colormap
    colors = ['white', 'crimson', 'darkgreen']
    n_bins = 3
    cmap = LinearSegmentedColormap.from_list('dominance', colors, N=n_bins)
    
    im4 = ax4.imshow(dominance, cmap=cmap, vmin=0, vmax=2)
    ax4.set_title('Dominant Species', fontsize=12)
    ax4.axis('off')
    
    # Add custom legend for dominance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', label='No Forest'),
        Patch(facecolor='crimson', label='Red Maple'),
        Patch(facecolor='darkgreen', label='Loblolly Pine')
    ]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    # Add overall statistics
    fig.text(0.5, 0.02, f'Data Source: USDA Forest Service FIA BIGMAP 2018 | Resolution: 30m | CRS: Web Mercator (EPSG:3857)', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("output/maps/wake_county")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wake_county_biomass_analysis.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved map to: {output_file}")
    
    plt.show()
    
    return fig


def create_presentation_summary():
    """Create a single presentation-ready summary map."""
    print("\n" + "=" * 60)
    print("Creating Presentation Summary")
    print("=" * 60)
    
    # Load data
    z = zarr.open("output/data/wake_county.zarr", mode='r')
    total_biomass = z['biomass'][2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot total biomass with enhanced visualization
    valid_data = total_biomass[total_biomass > 0]
    if len(valid_data) > 0:
        vmax = np.percentile(valid_data, 98)
    else:
        vmax = 1
    
    im = ax.imshow(total_biomass, cmap='YlGn', vmin=0, vmax=vmax)
    
    # Add title and labels
    ax.set_title('Wake County, North Carolina\nTotal Forest Biomass from BIGMAP 2018', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Biomass (Mg/ha)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # Add statistics box
    stats_text = f"""
    Forest Coverage: {100 * np.sum(total_biomass > 0) / total_biomass.size:.1f}%
    Mean Biomass: {valid_data.mean():.1f} Mg/ha
    Max Biomass: {valid_data.max():.1f} Mg/ha
    Total Biomass: {valid_data.sum()/1e6:.2f} million Mg
    
    Species Analyzed:
    • Loblolly Pine (Pinus taeda)
    • Red Maple (Acer rubrum)
    """
    
    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Add data source
    fig.text(0.5, 0.02, 
             'Data: USDA Forest Service FIA BIGMAP | Resolution: 30m | Analysis: BigMap Zarr Toolkit', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save presentation figure
    output_file = Path("output/maps/wake_county/wake_county_presentation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved presentation map to: {output_file}")
    
    return fig


def main():
    """Run complete Wake County analysis."""
    # Add metadata
    add_zarr_metadata()
    
    # Analyze data
    data = analyze_biomass_data()
    
    # Create maps
    create_biomass_map()
    create_presentation_summary()
    
    print("\n" + "=" * 60)
    print("✅ Wake County Analysis Complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - Zarr store: output/data/wake_county.zarr")
    print("  - Analysis map: output/maps/wake_county/wake_county_biomass_analysis.png")
    print("  - Presentation: output/maps/wake_county/wake_county_presentation.png")


if __name__ == "__main__":
    main()