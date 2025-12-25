#!/usr/bin/env python3
"""
Species Analysis Example

Comprehensive species analysis including:
- Species proportions and percentages
- Species group analysis (hardwood/softwood)
- Southern Yellow Pine specific analysis
- Diversity metrics and hotspot detection
"""

from pathlib import Path
import numpy as np
import zarr
from gridfia.examples import (
    create_sample_zarr,
    calculate_basic_stats,
    safe_load_zarr_with_memory_check,
    safe_open_zarr_biomass,
    AnalysisConfig
)
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.core.processors.forest_metrics import ForestMetricsProcessor
from gridfia.core.calculations import SpeciesProportion, SpeciesGroupProportion, registry
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Species group definitions
SOUTHERN_YELLOW_PINE = {
    39: {"code": "SPCD0110", "name": "Shortleaf Pine"},
    40: {"code": "SPCD0111", "name": "Slash Pine"},
    44: {"code": "SPCD0121", "name": "Longleaf Pine"},
    50: {"code": "SPCD0131", "name": "Loblolly Pine"}
}

HARDWOOD_INDICES = [3, 8, 11, 14, 18, 22, 25, 28]  # Example oak/maple species
SOFTWOOD_INDICES = [5, 7, 12, 15, 20, 24, 27, 30]  # Example pine/fir species


def analyze_species_proportions(zarr_path: Path):
    """Analyze proportions of individual species."""
    console.print("\n[bold blue]Species Proportion Analysis[/bold blue]")
    console.print("-" * 40)

    # Load zarr data with memory management
    config = AnalysisConfig()
    try:
        root, z = safe_open_zarr_biomass(zarr_path)
        # Get metadata from root (whether it's array or group)
        if hasattr(root, 'attrs'):
            species_codes = root.attrs.get('species_codes', [])
            species_names = root.attrs.get('species_names', [])
        else:
            species_codes = []
            species_names = []

        console.print(f"Analyzing {len(species_codes) - 1} species")  # -1 for TOTAL

        # Sample data for analysis with memory safety
        sample = safe_load_zarr_with_memory_check(zarr_path, config)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return

    # Calculate total biomass and forest mask
    total_biomass = sample[0]
    forest_mask = total_biomass > 0
    forest_pixels = np.sum(forest_mask)

    if forest_pixels == 0:
        console.print("[red]No forest pixels found[/red]")
        return

    # Calculate proportions for each species
    species_stats = []
    for i in range(1, len(species_codes)):  # Skip TOTAL
        species_biomass = sample[i]

        if np.sum(species_biomass) > 0:
            # Calculate proportion
            proportions = np.zeros_like(total_biomass)
            proportions[forest_mask] = species_biomass[forest_mask] / total_biomass[forest_mask]

            # Statistics
            mean_prop = np.mean(proportions[forest_mask])
            max_prop = np.max(proportions[forest_mask])
            coverage = np.sum(species_biomass > 0) / forest_pixels * 100

            species_stats.append({
                'index': i,
                'code': species_codes[i],
                'name': species_names[i],
                'mean_proportion': mean_prop,
                'max_proportion': max_prop,
                'coverage_pct': coverage
            })

    # Sort by mean proportion
    species_stats.sort(key=lambda x: x['mean_proportion'], reverse=True)

    # Display top species
    table = Table(title="Top 10 Species by Proportion")
    table.add_column("Rank", style="dim")
    table.add_column("Species", style="green")
    table.add_column("Mean %", justify="right", style="yellow")
    table.add_column("Max %", justify="right", style="red")
    table.add_column("Coverage %", justify="right", style="cyan")

    for i, stats in enumerate(species_stats[:10], 1):
        table.add_row(
            str(i),
            stats['name'][:30],
            f"{stats['mean_proportion'] * 100:.2f}",
            f"{stats['max_proportion'] * 100:.1f}",
            f"{stats['coverage_pct']:.1f}"
        )

    console.print(table)

    # Dominance analysis
    top_5_proportion = sum(s['mean_proportion'] for s in species_stats[:5])
    console.print(f"\n[yellow]Top 5 species account for {top_5_proportion * 100:.1f}% of biomass[/yellow]")


def analyze_species_groups(zarr_path: Path):
    """Analyze hardwood vs softwood groups."""
    console.print("\n[bold blue]Species Group Analysis[/bold blue]")
    console.print("-" * 40)

    # Register group calculations
    # Create custom classes for hardwood and softwood groups
    class HardwoodProportion(SpeciesGroupProportion):
        def __init__(self, **kwargs):
            super().__init__(
                species_indices=HARDWOOD_INDICES,
                group_name="hardwoods",
                exclude_total_layer=True,
                **kwargs
            )

    class SoftwoodProportion(SpeciesGroupProportion):
        def __init__(self, **kwargs):
            super().__init__(
                species_indices=SOFTWOOD_INDICES,
                group_name="softwoods",
                exclude_total_layer=True,
                **kwargs
            )

    registry.register("species_group_proportion_hardwoods", HardwoodProportion)
    registry.register("species_group_proportion_softwoods", SoftwoodProportion)

    console.print("✅ Registered hardwood and softwood group calculations")

    # Configure calculations
    settings = GridFIASettings(
        output_dir=Path("results/groups"),
        calculations=[
            CalculationConfig(
                name="species_group_proportion_hardwoods",
                enabled=True,
                output_format="geotiff",
                output_name="hardwood_proportion"
            ),
            CalculationConfig(
                name="species_group_proportion_softwoods",
                enabled=True,
                output_format="geotiff",
                output_name="softwood_proportion"
            )
        ]
    )

    # Run analysis
    processor = ForestMetricsProcessor(settings)
    results = processor.run_calculations(str(zarr_path))

    console.print(f"✅ Generated group proportion maps:")
    for name, path in results.items():
        console.print(f"   - {name}: {Path(path).name}")

    # Quick statistics
    root, z = safe_open_zarr_biomass(zarr_path)
    # Safe sample size based on actual array dimensions
    max_h = min(100, z.shape[1])
    max_w = min(100, z.shape[2])
    sample = z[:, :max_h, :max_w]  # Small sample for stats

    hardwood_biomass = np.sum([sample[i] for i in HARDWOOD_INDICES if i < len(sample)], axis=0)
    softwood_biomass = np.sum([sample[i] for i in SOFTWOOD_INDICES if i < len(sample)], axis=0)
    total = sample[0]

    forest_mask = total > 0
    if np.any(forest_mask):
        hw_prop = np.mean(hardwood_biomass[forest_mask] / total[forest_mask])
        sw_prop = np.mean(softwood_biomass[forest_mask] / total[forest_mask])

        console.print(f"\n[yellow]Sample Statistics:[/yellow]")
        console.print(f"  Hardwood proportion: {hw_prop * 100:.1f}%")
        console.print(f"  Softwood proportion: {sw_prop * 100:.1f}%")


def analyze_southern_yellow_pine(zarr_path: Path):
    """Specific analysis for Southern Yellow Pine species."""
    console.print("\n[bold blue]Southern Yellow Pine Analysis[/bold blue]")
    console.print("-" * 40)

    root, z = safe_open_zarr_biomass(zarr_path)
    species_codes = root.attrs.get('species_codes', []) if hasattr(root, 'attrs') else []

    # Check which SYP species are present
    syp_present = []
    for idx, species_info in SOUTHERN_YELLOW_PINE.items():
        if idx < len(species_codes):
            syp_present.append({
                'index': idx,
                'code': species_codes[idx],
                'name': species_info['name']
            })

    if not syp_present:
        console.print("[yellow]No Southern Yellow Pine species found in dataset[/yellow]")
        console.print("This is sample data. In real data, SYP species would be present.")
        return

    console.print(f"Found {len(syp_present)} SYP species in dataset")

    # Register SYP group calculation
    syp_indices = [s['index'] for s in syp_present]

    class SYPGroupProportion(SpeciesGroupProportion):
        def __init__(self, **kwargs):
            super().__init__(
                species_indices=syp_indices,
                group_name="southern_yellow_pine",
                exclude_total_layer=True,
                **kwargs
            )

    registry.register("species_group_proportion_southern_yellow_pine", SYPGroupProportion)

    # Calculate SYP proportion
    settings = GridFIASettings(
        output_dir=Path("results/syp"),
        calculations=[
            CalculationConfig(
                name="species_group_proportion_southern_yellow_pine",
                enabled=True,
                output_format="geotiff",
                output_name="syp_proportion"
            )
        ]
    )

    processor = ForestMetricsProcessor(settings)
    results = processor.run_calculations(str(zarr_path))

    console.print(f"✅ Generated SYP proportion map: {list(results.values())[0]}")

    # Display SYP summary
    panel = Panel(
        "[bold]Southern Yellow Pine Group[/bold]\n\n"
        "Important commercial timber species in the Southeast:\n"
        "• Loblolly Pine - Most planted tree in US\n"
        "• Longleaf Pine - Fire-adapted, conservation priority\n"
        "• Shortleaf Pine - Wide range, declining\n"
        "• Slash Pine - Fast growing, resin production\n\n"
        "These species are key to Southern forestry economy",
        title="🌲 SYP Information",
        border_style="green"
    )
    console.print(panel)


def identify_diversity_hotspots(zarr_path: Path):
    """Identify areas of high species diversity."""
    console.print("\n[bold blue]Diversity Hotspot Analysis[/bold blue]")
    console.print("-" * 40)

    # Calculate Shannon diversity
    settings = GridFIASettings(
        output_dir=Path("results/diversity"),
        calculations=[
            CalculationConfig(
                name="shannon_diversity",
                enabled=True,
                output_format="geotiff",
                output_name="shannon_diversity"
            )
        ]
    )

    processor = ForestMetricsProcessor(settings)
    results = processor.run_calculations(str(zarr_path))

    # Analyze hotspots (simplified for example)
    root, z = safe_open_zarr_biomass(zarr_path)
    # Safe sample size based on actual array dimensions
    max_h = min(200, z.shape[1])
    max_w = min(200, z.shape[2])
    sample = z[:, :max_h, :max_w]

    # Calculate Shannon diversity manually for demonstration
    total = sample[0]
    forest_mask = total > 0

    if np.any(forest_mask):
        # Simple Shannon calculation
        proportions = sample[1:] / (total + 1e-10)

        shannon = np.zeros(forest_mask.shape)
        for i in range(proportions.shape[0]):
            p = proportions[i]
            mask = (p > 0) & forest_mask
            shannon[mask] -= p[mask] * np.log(p[mask])

        # Find hotspots (top 10%)
        threshold = np.percentile(shannon[forest_mask], 90)
        hotspots = shannon > threshold
        hotspot_pixels = np.sum(hotspots)

        console.print(f"[yellow]Diversity Hotspots:[/yellow]")
        console.print(f"  Threshold (90th percentile): {threshold:.3f}")
        console.print(f"  Hotspot pixels: {hotspot_pixels:,}")
        console.print(f"  Hotspot area: {hotspot_pixels * 900 / 10000:.1f} hectares (30m pixels)")

        # Stats
        console.print(f"\n[yellow]Shannon Diversity Statistics:[/yellow]")
        console.print(f"  Mean: {np.mean(shannon[forest_mask]):.3f}")
        console.print(f"  Max: {np.max(shannon):.3f}")
        console.print(f"  Std: {np.std(shannon[forest_mask]):.3f}")


def run_comprehensive_analysis(zarr_path: Path):
    """Run complete species analysis pipeline."""
    console.print("\n[bold blue]Comprehensive Species Analysis[/bold blue]")
    console.print("-" * 40)

    settings = GridFIASettings(
        output_dir=Path("results/comprehensive"),
        calculations=[
            # Diversity metrics
            CalculationConfig(name="species_richness", enabled=True),
            CalculationConfig(name="shannon_diversity", enabled=True),
            CalculationConfig(name="simpson_diversity", enabled=True),
            CalculationConfig(name="evenness", enabled=True),

            # Biomass metrics
            CalculationConfig(name="total_biomass", enabled=True),
            CalculationConfig(name="dominant_species", enabled=True),
        ]
    )

    processor = ForestMetricsProcessor(settings)
    results = processor.run_calculations(str(zarr_path))

    # Display results
    table = Table(title="Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Output File", style="yellow")

    for name, path in results.items():
        table.add_row(name, Path(path).name)

    console.print(table)

    # Basic statistics
    stats = calculate_basic_stats(zarr_path, sample_size=500)
    console.print(f"\n[yellow]Forest Summary:[/yellow]")
    console.print(f"  Coverage: {stats['forest_coverage_pct']:.1f}%")
    console.print(f"  Mean richness: {stats['mean_richness']:.1f} species")
    console.print(f"  Max richness: {stats['max_richness']} species")


def main():
    """Run species analysis examples."""
    console.print("[bold green]Species Analysis Examples[/bold green]")
    console.print("=" * 60)

    # Create sample data with more species
    console.print("\nCreating sample forest data...")
    zarr_path = create_sample_zarr(Path("temp_species.zarr"), n_species=35)

    # Run analyses
    analyze_species_proportions(zarr_path)
    analyze_species_groups(zarr_path)
    analyze_southern_yellow_pine(zarr_path)
    identify_diversity_hotspots(zarr_path)
    run_comprehensive_analysis(zarr_path)

    # Clean up
    import shutil
    shutil.rmtree(zarr_path)
    if Path("results").exists():
        shutil.rmtree("results")

    console.print("\n" + "=" * 60)
    console.print("[bold green]Species Analysis Complete![/bold green]")
    console.print("\nKey capabilities demonstrated:")
    console.print("  - Individual species proportions")
    console.print("  - Species group analysis (hardwood/softwood)")
    console.print("  - Specific species complex (SYP)")
    console.print("  - Diversity hotspot identification")
    console.print("  - Comprehensive metrics calculation")


if __name__ == "__main__":
    main()