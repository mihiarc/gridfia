#!/usr/bin/env python3
"""
Test script to validate visualization capabilities with real output maps.
"""

from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

# Output directory
OUTPUT_DIR = Path("test_output/visualization_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_realistic_sample_zarr(output_path: Path, n_species: int = 5) -> Path:
    """
    Create a more realistic sample zarr with spatial patterns mimicking forest data.
    """
    console.print(f"[bold blue]Creating sample Zarr store at {output_path}[/bold blue]")

    # Create zarr group
    store = zarr.storage.LocalStore(str(output_path))
    root = zarr.open_group(store=store, mode='w')

    # Larger grid for better visualization (300x300 = ~90,000 pixels)
    height, width = 300, 300
    shape = (n_species + 1, height, width)  # +1 for total layer

    biomass_array = root.create_array(
        'biomass',
        shape=shape,
        chunks=(1, 100, 100),
        dtype='float32'
    )

    # Generate realistic spatial patterns
    np.random.seed(42)
    total_biomass = np.zeros((height, width), dtype='float32')

    # Create coordinate grids
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)

    # Species names and codes (common southeastern US forest species)
    species_data = [
        ("0000", "Total Biomass"),
        ("0131", "Loblolly Pine"),
        ("0068", "Red Maple"),
        ("0611", "Sweetgum"),
        ("0802", "White Oak"),
        ("0316", "Eastern Redcedar"),
    ][:n_species + 1]

    console.print(f"  Creating {n_species} species layers...")

    for i in range(1, shape[0]):  # Skip total layer (index 0)
        # Create unique spatial pattern for each species
        # Simulating different habitat preferences

        if i == 1:  # Loblolly Pine - dominant across landscape with gradient
            base = 80 * np.exp(-((X - 5)**2 + (Y - 5)**2) / 20)
            noise = np.random.normal(0, 10, (height, width))
            data = np.maximum(0, base + noise)

        elif i == 2:  # Red Maple - prefers eastern part
            base = 50 * (1 / (1 + np.exp(-(X - 6))))
            noise = np.random.normal(0, 8, (height, width))
            data = np.maximum(0, base + noise)

        elif i == 3:  # Sweetgum - riparian corridors (linear features)
            corridor1 = 40 * np.exp(-((Y - 3)**2) / 2)
            corridor2 = 35 * np.exp(-((X - 7)**2) / 2)
            data = np.maximum(corridor1, corridor2) + np.random.normal(0, 5, (height, width))
            data = np.maximum(0, data)

        elif i == 4:  # White Oak - patchy distribution
            centers = [(2, 2), (7, 3), (4, 8), (8, 7)]
            data = np.zeros((height, width))
            for cx, cy in centers:
                data += 30 * np.exp(-((X - cx)**2 + (Y - cy)**2) / 3)
            data += np.random.normal(0, 5, (height, width))
            data = np.maximum(0, data)

        else:  # Eastern Redcedar - edge habitat
            edge_effect = 25 * (np.sin(X * 1.5) ** 2 + np.cos(Y * 1.5) ** 2)
            data = edge_effect + np.random.normal(0, 5, (height, width))
            data = np.maximum(0, data)

        biomass_array[i, :, :] = data.astype('float32')
        total_biomass += data
        console.print(f"    Layer {i}: {species_data[i][1]} (mean: {data.mean():.1f} Mg/ha)")

    # Store total biomass in first layer
    biomass_array[0, :, :] = total_biomass
    console.print(f"    Layer 0: Total Biomass (mean: {total_biomass.mean():.1f} Mg/ha)")

    # Add metadata
    root.attrs['crs'] = 'EPSG:3857'  # Web Mercator
    root.attrs['num_species'] = n_species + 1
    root.attrs['location'] = 'Test Region (Synthetic Data)'
    root.attrs['year'] = '2025'
    root.attrs['source'] = 'GridFIA Visualization Test'

    # Transform for georeferencing (simulating Wake County area)
    # Affine transform order: [a, b, c, d, e, f]
    # a = pixel width, b = rotation (0), c = x origin
    # d = rotation (0), e = pixel height (negative), f = y origin
    # Place in approximate Wake County, NC location
    x_origin = -8762000.0  # West edge in EPSG:3857
    y_origin = 4314000.0   # North edge in EPSG:3857
    pixel_size = 30.0      # 30m resolution
    root.attrs['transform'] = [pixel_size, 0.0, x_origin, 0.0, -pixel_size, y_origin]

    # Also store explicit bounds for clarity
    left = x_origin
    right = x_origin + width * pixel_size
    top = y_origin
    bottom = y_origin - height * pixel_size
    root.attrs['bounds'] = [left, bottom, right, top]

    # Species codes and names
    codes = [s[0] for s in species_data]
    names = [s[1] for s in species_data]

    root.create_array('species_codes', data=np.array(codes, dtype='U10'))
    root.create_array('species_names', data=np.array(names, dtype='U50'))

    console.print(f"[green]Created Zarr store with shape {shape}[/green]")

    return output_path


def test_visualization():
    """Run full visualization test suite."""
    console.print("\n[bold green]GridFIA Visualization Test Suite[/bold green]")
    console.print("=" * 60)

    # Create sample data
    zarr_path = OUTPUT_DIR / "test_forest.zarr"

    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)

    create_realistic_sample_zarr(zarr_path, n_species=5)

    # Import visualization modules
    from gridfia.visualization.mapper import ZarrMapper
    from gridfia.visualization.plots import set_plot_style, save_figure

    # Set publication style
    set_plot_style('publication')

    # Initialize mapper
    console.print("\n[bold blue]Initializing ZarrMapper...[/bold blue]")
    mapper = ZarrMapper(str(zarr_path))

    # Get species info
    species_info = mapper.get_species_info()
    console.print(f"\nSpecies in store:")
    for info in species_info:
        console.print(f"  {info['index']}: {info['name']} ({info['code']})")

    # Test 1: Species Map
    console.print("\n[bold blue]Test 1: Creating species map (Loblolly Pine)...[/bold blue]")
    fig, ax = mapper.create_species_map(
        species=1,  # Loblolly Pine
        cmap='YlGn',
        title="Loblolly Pine Biomass - Test Region"
    )
    output_path = OUTPUT_DIR / "01_species_loblolly.png"
    save_figure(fig, str(output_path), dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Test 2: Another species
    console.print("\n[bold blue]Test 2: Creating species map (Red Maple)...[/bold blue]")
    fig, ax = mapper.create_species_map(
        species=2,  # Red Maple
        cmap='Reds',
        title="Red Maple Biomass - Test Region"
    )
    output_path = OUTPUT_DIR / "02_species_red_maple.png"
    save_figure(fig, str(output_path), dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Test 3: Shannon Diversity Map
    console.print("\n[bold blue]Test 3: Creating Shannon diversity map...[/bold blue]")
    fig, ax = mapper.create_diversity_map(
        diversity_type='shannon',
        cmap='viridis',
        title="Shannon Diversity Index - Test Region"
    )
    output_path = OUTPUT_DIR / "03_shannon_diversity.png"
    save_figure(fig, str(output_path), dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Test 4: Simpson Diversity Map
    console.print("\n[bold blue]Test 4: Creating Simpson diversity map...[/bold blue]")
    fig, ax = mapper.create_diversity_map(
        diversity_type='simpson',
        cmap='plasma',
        title="Simpson Diversity Index - Test Region"
    )
    output_path = OUTPUT_DIR / "04_simpson_diversity.png"
    save_figure(fig, str(output_path), dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Test 5: Species Richness Map
    console.print("\n[bold blue]Test 5: Creating species richness map...[/bold blue]")
    fig, ax = mapper.create_richness_map(
        threshold=1.0,  # Count species with >1 Mg/ha
        cmap='Spectral_r',
        title="Species Richness - Test Region"
    )
    output_path = OUTPUT_DIR / "05_species_richness.png"
    save_figure(fig, str(output_path), dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Test 6: Comparison Map
    console.print("\n[bold blue]Test 6: Creating species comparison map...[/bold blue]")
    fig = mapper.create_comparison_map(
        species_list=[1, 2, 3, 4],  # Compare 4 species
        ncols=2,
        cmap='YlGn',
        shared_colorbar=True
    )
    output_path = OUTPUT_DIR / "06_species_comparison.png"
    save_figure(fig, str(output_path), dpi=150)
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Test 7: Publication-quality composite figure
    console.print("\n[bold blue]Test 7: Creating publication composite figure...[/bold blue]")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Forest Analysis - Test Region (Synthetic Data)', fontsize=16, fontweight='bold')

    # Load data directly for composite
    root = zarr.open(str(zarr_path), mode='r')
    data = root['biomass'][:]

    # 1. Total Biomass
    ax = axes[0, 0]
    total = data[0]
    vmax = np.percentile(total[total > 0], 98) if np.any(total > 0) else 1.0
    im = ax.imshow(total, cmap='YlGn', vmin=0, vmax=vmax)
    ax.set_title('Total Biomass', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Mg/ha', fraction=0.046)

    # 2. Species Richness
    ax = axes[0, 1]
    richness = np.sum(data[1:] > 1.0, axis=0)
    im = ax.imshow(richness, cmap='Spectral_r', vmin=0, vmax=richness.max())
    ax.set_title('Species Richness', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Count', fraction=0.046)

    # 3. Shannon Diversity
    ax = axes[0, 2]
    forest_mask = total > 0
    shannon = np.zeros_like(total)
    for i in range(1, len(data)):
        p = np.zeros_like(total)
        p[forest_mask] = data[i][forest_mask] / total[forest_mask]
        mask = p > 0
        shannon[mask] -= p[mask] * np.log(p[mask])
    im = ax.imshow(shannon, cmap='viridis', vmin=0, vmax=max(shannon.max(), 0.1))
    ax.set_title('Shannon Diversity', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label="H'", fraction=0.046)

    # 4. Dominant Species
    ax = axes[1, 0]
    dominant = np.argmax(data[1:], axis=0)
    im = ax.imshow(dominant, cmap='tab10', vmin=0, vmax=4)
    ax.set_title('Dominant Species', fontsize=12)
    ax.axis('off')

    # 5. Loblolly Pine proportion
    ax = axes[1, 1]
    pine = data[1]  # Loblolly Pine
    ratio = np.zeros_like(total)
    ratio[forest_mask] = pine[forest_mask] / total[forest_mask]
    im = ax.imshow(ratio, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('Loblolly Pine Proportion', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Ratio', fraction=0.046)

    # 6. Forest Coverage
    ax = axes[1, 2]
    coverage = (total > 0).astype(float)
    im = ax.imshow(coverage, cmap='Greens', vmin=0, vmax=1)
    ax.set_title('Forest Coverage', fontsize=12)
    ax.axis('off')

    # Add footer
    fig.text(0.5, 0.02,
             'Data: Synthetic Test Data | Resolution: 30m | Analysis: GridFIA Python Toolkit',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "07_publication_composite.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    console.print(f"[green]Saved: {output_path}[/green]")

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Visualization Test Complete![/bold green]")
    console.print("=" * 60)
    console.print(f"\nGenerated maps saved to: {OUTPUT_DIR.absolute()}")
    console.print("\nMaps created:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        console.print(f"  - {f.name}")

    # Close mapper
    mapper.close()


if __name__ == "__main__":
    test_visualization()
