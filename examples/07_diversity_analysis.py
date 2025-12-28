#!/usr/bin/env python3
"""
Species Diversity Analysis Example

This example demonstrates how to properly calculate species diversity metrics
by downloading ALL species present in a study area. Unlike the quickstart
example which only downloads 2 species for speed, diversity indices require
complete species data to be ecologically meaningful.

Diversity metrics calculated:
- Species Richness: Count of species per pixel
- Shannon Diversity (H'): Information entropy measure
- Simpson Diversity (1-D): Probability-based dominance measure
- Evenness (J): How equally species are distributed

IMPORTANT: Downloading all species takes longer (~10-20 minutes depending on
area size and network speed), but is required for valid diversity calculations.

Takes about 15-30 minutes to run depending on network speed.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from gridfia import GridFIA
from gridfia.examples import print_zarr_info
from gridfia.visualization.mapper import ZarrMapper
from gridfia.visualization.plots import set_plot_style, save_figure
from examples.common_locations import get_location_bbox

console = Console()


def download_all_species(api: GridFIA, bbox: tuple, crs: str, output_dir: Path) -> list:
    """
    Download ALL available species for the study area.

    This is required for accurate diversity calculations. Most species will
    have zero biomass values in any given region - only species native to
    the area will have data.
    """
    console.print("\n[bold blue]Step 1: Download All Species[/bold blue]")
    console.print("-" * 50)

    # List all available species
    all_species = api.list_species()
    console.print(f"Total species in BIGMAP database: [cyan]{len(all_species)}[/cyan]")
    console.print("[yellow]Note: Most species won't be present in this region[/yellow]")
    console.print()

    # Download all species (species_codes=None means all)
    console.print("Downloading all species rasters...")
    console.print("[dim]This may take 10-20 minutes depending on network speed[/dim]")

    files = api.download_species(
        bbox=bbox,
        crs=crs,
        species_codes=None,  # Download ALL species
        output_dir=str(output_dir)
    )

    console.print(f"\n[green]Downloaded {len(files)} species raster files[/green]")
    return files


def analyze_species_presence(zarr_path: Path) -> dict:
    """
    Analyze which species are actually present in the study area.

    Returns statistics about species presence for ecological interpretation.
    """
    console.print("\n[bold blue]Step 3: Analyze Species Presence[/bold blue]")
    console.print("-" * 50)

    import zarr

    # Open zarr store
    root = zarr.open(str(zarr_path), mode='r')
    biomass = root['biomass'][:]
    species_codes = list(root['species_codes'][:])
    species_names = list(root['species_names'][:])

    # Calculate presence statistics
    total_layer = biomass[0]
    forest_mask = total_layer > 0
    forest_pixels = np.sum(forest_mask)

    # Count species with any presence
    species_with_data = []
    for i in range(1, len(species_codes)):  # Skip total layer
        species_data = biomass[i]
        presence_pixels = np.sum(species_data > 0)

        if presence_pixels > 0:
            presence_pct = 100 * presence_pixels / forest_pixels if forest_pixels > 0 else 0
            mean_biomass = np.mean(species_data[species_data > 0])

            species_with_data.append({
                'code': species_codes[i],
                'name': species_names[i],
                'presence_pixels': int(presence_pixels),
                'presence_pct': float(presence_pct),
                'mean_biomass': float(mean_biomass),
                'max_biomass': float(np.max(species_data))
            })

    # Sort by presence percentage
    species_with_data.sort(key=lambda x: x['presence_pct'], reverse=True)

    # Create summary table
    table = Table(title="Species Present in Study Area")
    table.add_column("Code", style="cyan")
    table.add_column("Scientific Name", style="green")
    table.add_column("Presence %", justify="right")
    table.add_column("Mean Biomass", justify="right")

    for sp in species_with_data[:15]:  # Top 15 species
        table.add_row(
            sp['code'],
            sp['name'][:40],
            f"{sp['presence_pct']:.1f}%",
            f"{sp['mean_biomass']:.1f} Mg/ha"
        )

    if len(species_with_data) > 15:
        table.add_row("...", f"... and {len(species_with_data) - 15} more species", "", "")

    console.print(table)

    console.print(f"\n[yellow]Summary:[/yellow]")
    console.print(f"  Total species in database: {len(species_codes) - 1}")
    console.print(f"  Species with presence in study area: [green]{len(species_with_data)}[/green]")
    console.print(f"  Species with no data (zeros): {len(species_codes) - 1 - len(species_with_data)}")

    return {
        'total_species': len(species_codes) - 1,
        'species_present': len(species_with_data),
        'species_data': species_with_data,
        'forest_pixels': int(forest_pixels)
    }


def calculate_diversity_metrics(api: GridFIA, zarr_path: Path, output_dir: Path) -> list:
    """
    Calculate all diversity metrics.

    These metrics are only meaningful when all species are included.
    """
    console.print("\n[bold blue]Step 4: Calculate Diversity Metrics[/bold blue]")
    console.print("-" * 50)

    diversity_calculations = [
        "species_richness",
        "shannon_diversity",
        "simpson_diversity",
        "evenness"
    ]

    console.print("Calculating diversity indices:")
    for calc in diversity_calculations:
        console.print(f"  - {calc}")

    results = api.calculate_metrics(
        zarr_path=zarr_path,
        calculations=diversity_calculations,
        output_dir=str(output_dir)
    )

    console.print(f"\n[green]Completed {len(results)} diversity calculations[/green]")
    return results


def interpret_diversity_statistics(zarr_path: Path, species_stats: dict):
    """
    Calculate and interpret diversity statistics with ecological context.
    """
    console.print("\n[bold blue]Step 5: Ecological Interpretation[/bold blue]")
    console.print("-" * 50)

    import zarr

    root = zarr.open(str(zarr_path), mode='r')
    biomass = root['biomass'][:]

    total_layer = biomass[0]
    species_layers = biomass[1:]

    forest_mask = total_layer > 0

    # Calculate richness per pixel
    richness = np.sum(species_layers > 0, axis=0)
    richness_forest = richness[forest_mask]

    # Calculate Shannon diversity per pixel
    shannon = np.zeros_like(total_layer)
    for i in range(len(species_layers)):
        p = np.zeros_like(total_layer)
        p[forest_mask] = species_layers[i][forest_mask] / total_layer[forest_mask]
        mask = p > 0
        shannon[mask] -= p[mask] * np.log(p[mask])
    shannon_forest = shannon[forest_mask]

    # Calculate Simpson diversity per pixel
    simpson = np.zeros_like(total_layer)
    for i in range(len(species_layers)):
        p = np.zeros_like(total_layer)
        p[forest_mask] = species_layers[i][forest_mask] / total_layer[forest_mask]
        simpson[forest_mask] += p[forest_mask] ** 2
    simpson = 1 - simpson  # Convert to diversity (1 - dominance)
    simpson_forest = simpson[forest_mask]

    console.print("[yellow]Diversity Statistics:[/yellow]\n")

    # Species Richness interpretation
    console.print("[bold]Species Richness[/bold] (count of species per pixel)")
    console.print(f"  Mean: {np.mean(richness_forest):.2f} species")
    console.print(f"  Max:  {np.max(richness_forest)} species")
    console.print(f"  Areas with 1 species:  {100 * np.sum(richness_forest == 1) / len(richness_forest):.1f}%")
    console.print(f"  Areas with 2+ species: {100 * np.sum(richness_forest >= 2) / len(richness_forest):.1f}%")
    console.print(f"  Areas with 5+ species: {100 * np.sum(richness_forest >= 5) / len(richness_forest):.1f}%")

    # Shannon interpretation
    console.print(f"\n[bold]Shannon Diversity Index (H')[/bold]")
    console.print(f"  Mean: {np.mean(shannon_forest):.3f}")
    console.print(f"  Max:  {np.max(shannon_forest):.3f}")
    console.print("  [dim]Interpretation: Higher values = more diverse[/dim]")
    console.print("  [dim]H' = 0: Single species dominates[/dim]")
    console.print(f"  [dim]H' = ln({species_stats['species_present']}) = {np.log(species_stats['species_present']):.2f}: Maximum possible[/dim]")

    # Simpson interpretation
    console.print(f"\n[bold]Simpson Diversity Index (1-D)[/bold]")
    console.print(f"  Mean: {np.mean(simpson_forest):.3f}")
    console.print(f"  Max:  {np.max(simpson_forest):.3f}")
    console.print("  [dim]Interpretation: Probability two random trees are different species[/dim]")
    console.print("  [dim]0 = Single species, 1 = Maximum diversity[/dim]")

    # Evenness
    max_shannon = np.log(richness)
    max_shannon[max_shannon == 0] = 1  # Avoid division by zero
    evenness = np.zeros_like(shannon)
    evenness[forest_mask] = shannon[forest_mask] / max_shannon[forest_mask]
    evenness_forest = evenness[forest_mask]
    evenness_forest = evenness_forest[~np.isnan(evenness_forest) & ~np.isinf(evenness_forest)]

    console.print(f"\n[bold]Evenness (Pielou's J)[/bold]")
    if len(evenness_forest) > 0:
        console.print(f"  Mean: {np.mean(evenness_forest):.3f}")
        console.print(f"  Max:  {np.max(evenness_forest):.3f}")
    console.print("  [dim]Interpretation: How equally biomass is distributed among species[/dim]")
    console.print("  [dim]0 = One species dominates, 1 = All species equal[/dim]")


def create_diversity_maps(zarr_path: Path, output_dir: Path):
    """
    Create publication-quality diversity maps.
    """
    console.print("\n[bold blue]Step 6: Create Diversity Maps[/bold blue]")
    console.print("-" * 50)

    set_plot_style('publication')
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper = ZarrMapper(str(zarr_path))

    # Create richness map
    console.print("Creating species richness map...")
    fig, ax = mapper.create_richness_map(
        cmap='Spectral_r',
        threshold=0.0,
        title="Species Richness"
    )
    save_figure(fig, str(output_dir / "species_richness_map.png"), dpi=200)
    plt.close(fig)

    # Create Shannon diversity map
    console.print("Creating Shannon diversity map...")
    fig, ax = mapper.create_diversity_map(
        diversity_type='shannon',
        cmap='viridis',
        title="Shannon Diversity Index (H')"
    )
    save_figure(fig, str(output_dir / "shannon_diversity_map.png"), dpi=200)
    plt.close(fig)

    # Create Simpson diversity map
    console.print("Creating Simpson diversity map...")
    fig, ax = mapper.create_diversity_map(
        diversity_type='simpson',
        cmap='plasma',
        title="Simpson Diversity Index (1-D)"
    )
    save_figure(fig, str(output_dir / "simpson_diversity_map.png"), dpi=200)
    plt.close(fig)

    # Create composite figure
    console.print("Creating composite diversity figure...")
    create_composite_diversity_figure(zarr_path, output_dir)

    console.print(f"\n[green]Maps saved to {output_dir}[/green]")


def create_composite_diversity_figure(zarr_path: Path, output_dir: Path):
    """Create a 2x2 composite figure showing all diversity metrics."""
    import zarr

    root = zarr.open(str(zarr_path), mode='r')
    biomass = root['biomass'][:]

    total_layer = biomass[0]
    species_layers = biomass[1:]
    forest_mask = total_layer > 0

    # Calculate metrics
    richness = np.sum(species_layers > 0, axis=0).astype(float)
    richness[~forest_mask] = np.nan

    shannon = np.zeros_like(total_layer)
    simpson_d = np.zeros_like(total_layer)

    for i in range(len(species_layers)):
        p = np.zeros_like(total_layer)
        p[forest_mask] = species_layers[i][forest_mask] / total_layer[forest_mask]
        mask = p > 0
        shannon[mask] -= p[mask] * np.log(p[mask])
        simpson_d[forest_mask] += p[forest_mask] ** 2

    simpson = 1 - simpson_d
    shannon[~forest_mask] = np.nan
    simpson[~forest_mask] = np.nan

    # Evenness
    max_shannon = np.log(np.sum(species_layers > 0, axis=0))
    max_shannon[max_shannon == 0] = np.nan
    evenness = shannon / max_shannon
    evenness[~forest_mask] = np.nan

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Forest Species Diversity Analysis', fontsize=16, fontweight='bold')

    # Species Richness
    ax = axes[0, 0]
    im = ax.imshow(richness, cmap='Spectral_r')
    ax.set_title('Species Richness\n(count per pixel)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Species Count', fraction=0.046)

    # Shannon Diversity
    ax = axes[0, 1]
    im = ax.imshow(shannon, cmap='viridis')
    ax.set_title("Shannon Diversity (H')\n(information entropy)", fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label="H'", fraction=0.046)

    # Simpson Diversity
    ax = axes[1, 0]
    im = ax.imshow(simpson, cmap='plasma', vmin=0, vmax=1)
    ax.set_title('Simpson Diversity (1-D)\n(probability measure)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='1-D', fraction=0.046)

    # Evenness
    ax = axes[1, 1]
    im = ax.imshow(evenness, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title("Evenness (Pielou's J)\n(distribution equality)", fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='J', fraction=0.046)

    # Footer
    fig.text(0.5, 0.02,
             'Data: USDA Forest Service FIA BIGMAP 2018 | All species included for valid diversity metrics',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / "diversity_composite.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Run complete diversity analysis workflow."""
    console.print("[bold green]Species Diversity Analysis[/bold green]")
    console.print("=" * 60)
    console.print()
    console.print("[yellow]This example downloads ALL species to calculate valid diversity metrics.[/yellow]")
    console.print("[yellow]This takes longer than the quickstart but produces ecologically meaningful results.[/yellow]")
    console.print()

    # Setup paths
    data_dir = Path("diversity_analysis_data")
    results_dir = Path("diversity_analysis_results")
    zarr_path = data_dir / "forest_all_species.zarr"

    # Initialize API
    api = GridFIA()

    # Get location - using Wake County NC subset
    bbox, crs = get_location_bbox("wake_nc")
    console.print(f"Study Area: Wake County, NC (subset)")
    console.print(f"Bbox: {bbox}")
    console.print(f"CRS: EPSG:{crs}")

    # Check if data already exists
    if zarr_path.exists():
        console.print(f"\n[green]Using existing data: {zarr_path}[/green]")
        console.print("[dim]Delete this folder to re-download[/dim]")
    else:
        # Step 1: Download all species
        files = download_all_species(api, bbox, crs, data_dir)

        if not files:
            console.print("[red]No species data downloaded. Check network connection.[/red]")
            return

        # Step 2: Create Zarr store
        console.print("\n[bold blue]Step 2: Create Zarr Store[/bold blue]")
        console.print("-" * 50)

        zarr_path = api.create_zarr(
            input_dir=str(data_dir),
            output_path=str(zarr_path)
        )
        zarr_path = Path(zarr_path)
        print_zarr_info(zarr_path)

    # Step 3: Analyze species presence
    species_stats = analyze_species_presence(zarr_path)

    # Step 4: Calculate diversity metrics
    results = calculate_diversity_metrics(api, zarr_path, results_dir / "metrics")

    # Step 5: Interpret statistics
    interpret_diversity_statistics(zarr_path, species_stats)

    # Step 6: Create maps
    create_diversity_maps(zarr_path, results_dir / "maps")

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Diversity Analysis Complete![/bold green]")
    console.print("=" * 60)
    console.print()
    console.print(f"Species in database:      {species_stats['total_species']}")
    console.print(f"Species present in area:  [green]{species_stats['species_present']}[/green]")
    console.print()
    console.print("Output files:")
    console.print(f"  Data:    {data_dir}/")
    console.print(f"  Results: {results_dir}/")
    console.print()
    console.print("[yellow]Key insight:[/yellow] True species richness requires ALL species data.")
    console.print("The quickstart example (2 species) cannot calculate meaningful diversity.")


if __name__ == "__main__":
    main()
