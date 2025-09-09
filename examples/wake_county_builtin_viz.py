#!/usr/bin/env python3
"""
Wake County Analysis using Built-in Visualization Module
"""

import matplotlib.pyplot as plt
from pathlib import Path
from bigmap.visualization.mapper import ZarrMapper
from bigmap.visualization.boundaries import load_counties_for_state, plot_boundaries
from bigmap.visualization.plots import set_plot_style, save_figure
from bigmap.console import print_info, print_success

def main():
    """Run Wake County visualization using built-in modules."""
    
    print("=" * 60)
    print("Wake County - Using Built-in Visualization")
    print("=" * 60)
    
    # Set publication style
    set_plot_style('publication')
    
    # Initialize the mapper
    print_info("Initializing ZarrMapper...")
    mapper = ZarrMapper("output/data/wake_county.zarr")
    
    # Create output directory
    output_dir = Path("output/maps/wake_builtin_python")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create individual species maps
    print("\n1. Creating Species Maps")
    print("-" * 40)
    
    # Red Maple (index 0, code 0068)
    fig, ax = mapper.create_species_map(
        species=0,  # First species in array
        cmap='YlGn',
        vmin=0,
        vmax=None,  # Auto scale
        title="Red Maple (Acer rubrum) - Wake County"
    )
    save_figure(fig, str(output_dir / "red_maple.png"), dpi=150)
    print_success("Created red_maple.png")
    plt.close(fig)
    
    # Loblolly Pine (index 1, code 0131)
    fig, ax = mapper.create_species_map(
        species=1,  # Second species in array
        cmap='YlGn',
        title="Loblolly Pine (Pinus taeda) - Wake County"
    )
    save_figure(fig, str(output_dir / "loblolly_pine.png"), dpi=150)
    print_success("Created loblolly_pine.png")
    plt.close(fig)
    
    # Total Biomass (index 2)
    fig, ax = mapper.create_species_map(
        species=2,  # Total biomass
        cmap='YlGn',
        title="Total Forest Biomass - Wake County"
    )
    save_figure(fig, str(output_dir / "total_biomass.png"), dpi=150)
    print_success("Created total_biomass.png")
    plt.close(fig)
    
    # 2. Create diversity maps
    print("\n2. Creating Diversity Maps")
    print("-" * 40)
    
    # Shannon Diversity
    fig, ax = mapper.create_diversity_map(
        diversity_type='shannon',
        cmap='plasma',
        title="Shannon Diversity Index - Wake County"
    )
    save_figure(fig, str(output_dir / "shannon_diversity.png"), dpi=150)
    print_success("Created shannon_diversity.png")
    plt.close(fig)
    
    # Simpson Diversity
    fig, ax = mapper.create_diversity_map(
        diversity_type='simpson',
        cmap='plasma',
        title="Simpson Diversity Index - Wake County"
    )
    save_figure(fig, str(output_dir / "simpson_diversity.png"), dpi=150)
    print_success("Created simpson_diversity.png")
    plt.close(fig)
    
    # 3. Create richness map
    print("\n3. Creating Species Richness Map")
    print("-" * 40)
    
    fig, ax = mapper.create_richness_map(
        cmap='Spectral_r',
        threshold=0.1,
        title="Species Richness - Wake County"
    )
    save_figure(fig, str(output_dir / "species_richness.png"), dpi=150)
    print_success("Created species_richness.png")
    plt.close(fig)
    
    # 4. Create comparison map
    print("\n4. Creating Species Comparison Map")
    print("-" * 40)
    
    # Compare the two actual species (indices 0 and 1)
    fig = mapper.create_comparison_map(
        species_list=[0, 1],  # Red Maple and Loblolly Pine
        cmap='YlGn'
    )
    save_figure(fig, str(output_dir / "species_comparison.png"), dpi=150)
    print_success("Created species_comparison.png")
    plt.close(fig)
    
    # 5. Try to add boundaries (if available)
    print("\n5. Creating Map with Boundaries")
    print("-" * 40)
    
    try:
        # Load Wake County boundary
        counties = load_counties_for_state("North Carolina")
        wake_county = counties[counties['NAME'] == 'Wake']
        
        # Create map with boundary
        fig, ax = mapper.create_species_map(
            species=2,  # Total biomass
            cmap='YlGn',
            title="Total Biomass with County Boundary"
        )
        
        # Add county boundary
        if not wake_county.empty:
            plot_boundaries(
                ax=ax,
                boundaries=wake_county,
                color='red',
                linewidth=2,
                fill=False,
                label='Wake County'
            )
            ax.legend()
        
        save_figure(fig, str(output_dir / "biomass_with_boundary.png"), dpi=150)
        print_success("Created biomass_with_boundary.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"Could not add boundaries: {e}")
    
    # 6. Summary statistics from mapper
    print("\n6. Data Summary")
    print("-" * 40)
    
    species_info = mapper.get_species_info()
    print("Species in dataset:")
    for sp in species_info:
        print(f"  {sp['index']}: {sp['name']} (Code: {sp['code']})")
    
    print("\n" + "=" * 60)
    print("✅ Visualization Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()