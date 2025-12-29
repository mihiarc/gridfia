#!/usr/bin/env python3
"""
Wake County Complete Analysis

A comprehensive case study demonstrating the full GridFIA workflow:
- Data download (ALL species for valid diversity metrics)
- Zarr creation
- All calculations
- Multiple visualizations
- Statistical analysis
- Publication-ready outputs

Takes about 15-20 minutes to run (mostly download time).
"""

from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from gridfia import GridFIA
from gridfia.examples import (
    calculate_basic_stats,
    safe_download_species,
    safe_load_zarr_with_memory_check,
    safe_open_zarr_biomass,
    AnalysisConfig,
    cleanup_example_outputs
)
from gridfia.visualization.mapper import ZarrMapper
from gridfia.visualization.plots import set_plot_style, save_figure
from rich.console import Console

console = Console()


def download_wake_county_data():
    """Download forest data for Wake County."""
    console.print("\n[bold blue]Step 1: Data Download[/bold blue]")
    console.print("-" * 40)

    api = GridFIA()

    # Wake County, NC bounding box (Web Mercator EPSG:3857)
    # Source: US Census Bureau Tiger/Line Shapefiles 2021
    # Validated coordinates for Wake County boundaries
    wake_bbox = (-8792000, 4274000, -8732000, 4334000)  # xmin, ymin, xmax, ymax

    # Validate bounding box makes sense for Wake County
    bbox_width = wake_bbox[2] - wake_bbox[0]  # ~60km
    bbox_height = wake_bbox[3] - wake_bbox[1]  # ~60km

    # Wake County is roughly 55km x 65km, so this should be reasonable
    if not (40000 < bbox_width < 80000 and 40000 < bbox_height < 80000):
        raise ValueError(f"Invalid Wake County bounding box dimensions: {bbox_width/1000:.1f}km x {bbox_height/1000:.1f}km")

    console.print("[yellow]Using validated hardcoded bounding box for Wake County, NC[/yellow]")
    console.print(f"  Bbox: {wake_bbox}")
    console.print(f"  CRS: EPSG:3857 (Web Mercator)")
    console.print(f"  Dimensions: {bbox_width/1000:.1f}km x {bbox_height/1000:.1f}km")
    console.print(f"  [dim]Note: Hardcoded to avoid SSL certificate issues with census.gov[/dim]")

    # Download ALL species for Wake County (required for valid diversity metrics)
    console.print(f"\nDownloading ALL species for Wake County, NC...")
    console.print("[yellow]This takes 10-15 minutes but is required for valid diversity metrics[/yellow]")

    try:
        files = api.download_species(
            bbox=wake_bbox,
            crs="3857",  # Web Mercator
            species_codes=None,  # Download ALL species
            output_dir="examples/wake_county_data"
        )
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        files = []

    console.print(f"\n✅ Downloaded {len(files)} species files")
    return files


def create_wake_zarr():
    """Create Zarr store from downloaded data."""
    console.print("\n[bold blue]Step 2: Zarr Creation[/bold blue]")
    console.print("-" * 40)

    api = GridFIA()

    zarr_path = api.create_zarr(
        input_dir="examples/wake_county_data",
        output_path="examples/wake_county_data/wake_forest.zarr",
        chunk_size=(1, 500, 500)
    )

    # Add comprehensive metadata
    z = zarr.open(str(zarr_path), mode='r+')
    z.attrs['location'] = 'Wake County, North Carolina'
    z.attrs['year'] = '2018'
    z.attrs['source'] = 'USDA Forest Service FIA BIGMAP'

    console.print(f"✅ Created Zarr store: {zarr_path}")
    return Path(zarr_path)


def run_comprehensive_calculations(zarr_path: Path):
    """Run all forest calculations."""
    console.print("\n[bold blue]Step 3: Forest Metrics Calculation[/bold blue]")
    console.print("-" * 40)

    api = GridFIA()

    calculations = [
        "species_richness",
        "shannon_diversity",
        "simpson_diversity",
        "evenness",
        "total_biomass",
        "dominant_species"
    ]

    console.print(f"Running {len(calculations)} calculations...")

    results = api.calculate_metrics(
        zarr_path=zarr_path,
        calculations=calculations,
        output_dir="examples/wake_results/metrics"
    )

    console.print(f"✅ Completed {len(results)} calculations")
    for calc in results:
        console.print(f"   - {calc.name}: {calc.output_path.name}")

    return results


def analyze_forest_statistics(zarr_path: Path):
    """Detailed statistical analysis."""
    console.print("\n[bold blue]Step 4: Statistical Analysis[/bold blue]")
    console.print("-" * 40)

    # Use safe zarr opening utility
    root, z = safe_open_zarr_biomass(zarr_path)
    species_names = z.attrs.get('species_names', [])

    # Load data (sample for memory efficiency)
    data = z[:, :1000, :1000]
    total_biomass = data[0]
    forest_mask = total_biomass > 0
    forest_pixels = np.sum(forest_mask)

    console.print(f"[yellow]Forest Coverage Statistics:[/yellow]")
    console.print(f"  Total pixels: {total_biomass.size:,}")
    console.print(f"  Forest pixels: {forest_pixels:,}")
    console.print(f"  Coverage: {100 * forest_pixels / total_biomass.size:.1f}%")
    console.print(f"  Area: {forest_pixels * 900 / 10000:.1f} hectares")

    # Species statistics
    console.print(f"\n[yellow]Species Statistics:[/yellow]")
    for i in range(1, min(len(species_names), 6)):  # First 5 species
        species_data = data[i]
        valid = species_data[forest_mask]
        present = valid > 0

        if np.any(present):
            console.print(f"\n  {species_names[i]}:")
            console.print(f"    Presence: {100 * np.sum(present) / forest_pixels:.1f}%")
            console.print(f"    Mean biomass: {np.mean(valid[present]):.1f} Mg/ha")
            console.print(f"    Max biomass: {np.max(valid):.1f} Mg/ha")
            console.print(f"    Total biomass: {np.sum(valid) / 1e6:.2f} million Mg")

    # Diversity statistics
    species_count = np.sum(data[1:] > 0, axis=0)
    richness = species_count[forest_mask]

    console.print(f"\n[yellow]Diversity Statistics:[/yellow]")
    console.print(f"  Mean species richness: {np.mean(richness):.1f}")
    console.print(f"  Max species richness: {np.max(richness)}")
    console.print(f"  Areas with 1 species: {100 * np.sum(richness == 1) / len(richness):.1f}%")
    console.print(f"  Areas with 2+ species: {100 * np.sum(richness >= 2) / len(richness):.1f}%")
    console.print(f"  Areas with 3+ species: {100 * np.sum(richness >= 3) / len(richness):.1f}%")


def create_visualization_suite(zarr_path: Path):
    """Create comprehensive visualizations."""
    console.print("\n[bold blue]Step 5: Visualization Suite[/bold blue]")
    console.print("-" * 40)

    # Set publication style
    set_plot_style('publication')

    # Initialize mapper
    mapper = ZarrMapper(str(zarr_path))
    output_dir = Path("examples/wake_results/maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Individual species maps
    console.print("Creating species maps...")
    species_info = mapper.get_species_info()
    for i, species in enumerate(species_info[:3]):  # First 3 species
        fig, ax = mapper.create_species_map(
            species=i,
            cmap='YlGn',
            title=f"{species['name']} - Wake County"
        )
        save_figure(fig, str(output_dir / f"species_{i}.png"), dpi=150)
        plt.close(fig)

    # 2. Diversity maps
    console.print("Creating diversity maps...")
    for diversity_type in ['shannon', 'simpson']:
        fig, ax = mapper.create_diversity_map(
            diversity_type=diversity_type,
            cmap='viridis',
            title=f"{diversity_type.title()} Diversity - Wake County"
        )
        save_figure(fig, str(output_dir / f"{diversity_type}_diversity.png"), dpi=150)
        plt.close(fig)

    # 3. Richness map
    console.print("Creating richness map...")
    fig, ax = mapper.create_richness_map(
        cmap='Spectral_r',
        threshold=0.5,
        title="Species Richness - Wake County"
    )
    save_figure(fig, str(output_dir / "species_richness.png"), dpi=150)
    plt.close(fig)

    # 4. Comparison map
    console.print("Creating comparison map...")
    fig = mapper.create_comparison_map(
        species_list=[0, 1],  # Compare first two species
        cmap='YlGn'
    )
    save_figure(fig, str(output_dir / "species_comparison.png"), dpi=150)
    plt.close(fig)

    console.print(f"✅ Created visualization suite in {output_dir}")


def create_publication_figure(zarr_path: Path):
    """Create a publication-ready composite figure."""
    console.print("\n[bold blue]Step 6: Publication Figure[/bold blue]")
    console.print("-" * 40)

    # Use safe zarr opening utility
    root, z = safe_open_zarr_biomass(zarr_path)
    species_names = z.attrs.get('species_names', [])

    # Create 2x3 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Wake County Forest Analysis - BigMap 2018', fontsize=16, fontweight='bold')

    # Load sample data (adjust for smaller test arrays)
    h, w = z.shape[1], z.shape[2]
    data = z[:, :min(500, h), :min(500, w)]

    # 1. Total Biomass
    ax = axes[0, 0]
    total = data[0]
    # Safe percentile calculation
    if np.any(total > 0):
        vmax_total = np.percentile(total[total > 0], 98)
    else:
        vmax_total = 1.0
    im = ax.imshow(total, cmap='YlGn', vmin=0, vmax=vmax_total)
    ax.set_title('Total Biomass', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Mg/ha', fraction=0.046)

    # 2. Species Richness
    ax = axes[0, 1]
    richness = np.sum(data[1:] > 0, axis=0)
    vmax_richness = max(richness.max(), 1)  # Ensure vmax is at least 1
    im = ax.imshow(richness, cmap='Spectral_r', vmin=0, vmax=vmax_richness)
    ax.set_title('Species Richness', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Count', fraction=0.046)

    # 3. Shannon Diversity
    ax = axes[0, 2]
    # Simple Shannon calculation
    forest_mask = total > 0
    shannon = np.zeros_like(total)
    for i in range(1, len(data)):
        p = np.zeros_like(total)
        p[forest_mask] = data[i][forest_mask] / total[forest_mask]
        mask = p > 0
        shannon[mask] -= p[mask] * np.log(p[mask])

    # Handle case where all values are the same
    vmax = max(shannon.max(), 0.1)  # Ensure vmax is different from vmin
    im = ax.imshow(shannon, cmap='viridis', vmin=0, vmax=vmax)
    ax.set_title('Shannon Diversity', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label="H'", fraction=0.046)

    # 4. Dominant Species
    ax = axes[1, 0]
    dominant = np.argmax(data[1:], axis=0)
    im = ax.imshow(dominant, cmap='tab20', vmin=0, vmax=min(20, len(species_names) - 1))
    ax.set_title('Dominant Species', fontsize=12)
    ax.axis('off')

    # 5. Pine vs Hardwood
    ax = axes[1, 1]
    if len(data) > 2:
        pine = data[1]  # Assuming first species is pine
        hardwood = data[2] if len(data) > 2 else np.zeros_like(pine)
        ratio = np.zeros_like(total)
        mask = (pine + hardwood) > 0
        ratio[mask] = pine[mask] / (pine[mask] + hardwood[mask])

        im = ax.imshow(ratio, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title('Pine Proportion', fontsize=12)
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
             'Data: USDA Forest Service FIA BIGMAP | Resolution: 30m | Analysis: GridFIA Python Toolkit',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    # Save publication figure
    output_path = Path("examples/wake_results/wake_county_publication.png")
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"✅ Created publication figure: {output_path}")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save publication figure: {e}[/yellow]")
    finally:
        plt.close()  # Close figure to free memory


def create_summary_report(zarr_path: Path):
    """Generate a summary report."""
    console.print("\n[bold blue]Step 7: Summary Report[/bold blue]")
    console.print("-" * 40)

    stats = calculate_basic_stats(zarr_path, sample_size=None)

    report = f"""
Wake County Forest Analysis Summary
{'=' * 40}

Location: Wake County, North Carolina
Data Source: USDA FIA BIGMAP 2018
Resolution: 30m x 30m pixels

Forest Coverage
---------------
Total Area: {stats['total_pixels'] * 900 / 1e6:.1f} km²
Forest Area: {stats['forest_pixels'] * 900 / 1e6:.1f} km²
Forest Coverage: {stats['forest_coverage_pct']:.1f}%

Biomass Statistics
------------------
Mean Biomass: {stats['mean_biomass']:.1f} Mg/ha
Maximum Biomass: {stats['max_biomass']:.1f} Mg/ha
Total Biomass: {stats['total_biomass_mg'] / 1e6:.2f} million Mg

Species Diversity
-----------------
Mean Species Richness: {stats['mean_richness']:.1f} species/pixel
Maximum Species Richness: {stats['max_richness']} species/pixel

Analysis Outputs
----------------
- Zarr data store: examples/wake_county_data/wake_forest.zarr
- Metric calculations: examples/wake_results/metrics/
- Visualization maps: examples/wake_results/maps/
- Publication figure: examples/wake_results/wake_county_publication.png

Processing Complete
-------------------
This analysis demonstrates the full GridFIA workflow from
data download through publication-ready visualizations.
"""

    # Save report
    report_path = Path("examples/wake_results/analysis_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)

    console.print(report)
    console.print(f"\n✅ Report saved to: {report_path}")


def main():
    """Run complete Wake County analysis."""
    console.print("[bold green]Wake County Complete Forest Analysis[/bold green]")
    console.print("=" * 60)

    # Check if data exists or download
    zarr_path = Path("examples/wake_county_data/wake_forest.zarr")

    if not zarr_path.exists():
        console.print("\n[yellow]Data not found. Starting download...[/yellow]")
        files = download_wake_county_data()

        # If download failed, use sample data
        if not files:
            console.print("\n[yellow]Download failed. Using sample data instead.[/yellow]")
            from gridfia.examples import create_sample_zarr
            # Create directory if it doesn't exist
            zarr_path.parent.mkdir(parents=True, exist_ok=True)
            zarr_path = create_sample_zarr(zarr_path, n_species=5)
            console.print("[yellow]Note: Using synthetic data for demonstration[/yellow]")
        else:
            zarr_path = create_wake_zarr()
    else:
        console.print(f"\n[green]Using existing data: {zarr_path}[/green]")

    # Run full analysis pipeline
    run_comprehensive_calculations(zarr_path)
    analyze_forest_statistics(zarr_path)
    create_visualization_suite(zarr_path)
    create_publication_figure(zarr_path)
    create_summary_report(zarr_path)

    console.print("\n" + "=" * 60)
    console.print("[bold green]✅ Wake County Analysis Complete![/bold green]")
    console.print("=" * 60)
    console.print("\nThis comprehensive analysis includes:")
    console.print("  • Forest biomass assessment")
    console.print("  • Species diversity analysis")
    console.print("  • Spatial pattern visualization")
    console.print("  • Publication-ready figures")
    console.print("  • Statistical summary report")
    console.print("\nAll outputs saved to examples/wake_results/")


if __name__ == "__main__":
    main()