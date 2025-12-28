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
import geopandas as gpd

from gridfia import GridFIA
from gridfia.examples import print_zarr_info
from gridfia.visualization.plots import set_plot_style
from gridfia.visualization.boundaries import (
    load_counties_for_state,
    add_basemap,
    plot_boundaries,
    get_basemap_zoom_level
)
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


def create_diversity_maps(zarr_path: Path, output_dir: Path, county_name: str = "Durham"):
    """
    Create publication-quality diversity maps with county boundary and basemap.
    """
    console.print("\n[bold blue]Step 6: Create Diversity Maps[/bold blue]")
    console.print("-" * 50)

    set_plot_style('publication')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create composite figure with all diversity metrics, county boundary, and basemap
    create_composite_diversity_figure(zarr_path, output_dir, county_name=county_name)

    console.print(f"\n[green]Maps saved to {output_dir}[/green]")


def load_county_boundary(county_name: str, state: str = 'NC', crs: str = 'EPSG:3857') -> gpd.GeoDataFrame:
    """
    Load county boundary for clipping and overlay.

    Args:
        county_name: Name of the county (e.g., 'Durham', 'Wake')
        state: State abbreviation (default 'NC')
        crs: Coordinate reference system (default 'EPSG:3857')

    Returns:
        GeoDataFrame with county boundary in the specified CRS.
    """
    try:
        # Load all counties for the state
        counties = load_counties_for_state(state, crs=crs)

        # Filter for the requested county
        county = counties[counties['NAME'].str.lower() == county_name.lower()].copy()

        if len(county) == 0:
            console.print(f"[yellow]Warning: {county_name} County not found[/yellow]")
            return None

        return county
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load county boundary: {e}[/yellow]")
        return None


def create_composite_diversity_figure(zarr_path: Path, output_dir: Path, county_name: str = "Durham"):
    """
    Create a 2x2 composite figure showing all diversity metrics.

    Improved version with:
    - County boundary clipping/masking
    - Basemap for geographic context
    - Professional cartographic styling
    """
    import zarr
    from rasterio.transform import Affine
    from rasterio.features import geometry_mask

    console.print("Loading data and calculating metrics...")

    # Open zarr and get metadata
    root = zarr.open(str(zarr_path), mode='r')
    biomass = root['biomass'][:]

    # Get transform and CRS from zarr attributes
    transform_params = root.attrs.get('transform', None)
    crs_str = root.attrs.get('crs', 'EPSG:3857')

    if transform_params:
        transform = Affine(*transform_params[:6])
    else:
        # Fallback: estimate from data shape and typical Wake County extent
        height, width = biomass.shape[1], biomass.shape[2]
        # Use the bbox from common_locations
        bbox, _ = get_location_bbox("wake_nc")
        xmin, ymin, xmax, ymax = bbox
        pixel_width = (xmax - xmin) / width
        pixel_height = (ymin - ymax) / height  # Negative for north-up
        transform = Affine(pixel_width, 0, xmin, 0, pixel_height, ymax)

    # Calculate data extent for matplotlib (left, right, bottom, top)
    height, width = biomass.shape[1], biomass.shape[2]
    data_left = transform.c
    data_right = transform.c + width * transform.a
    data_top = transform.f
    data_bottom = transform.f + height * transform.e
    data_extent = (data_left, data_right, data_bottom, data_top)

    console.print(f"  Data extent: {data_extent}")
    console.print(f"  CRS: {crs_str}")

    # Load county boundary
    console.print(f"Loading {county_name} County boundary...")
    county_gdf = load_county_boundary(county_name, state='NC', crs=crs_str)

    # Use county bounds for figure extent (shows full county, not just data area)
    county_mask = None
    if county_gdf is not None:
        county_bounds = county_gdf.total_bounds  # (minx, miny, maxx, maxy)
        # Add 5% padding around county
        pad_x = (county_bounds[2] - county_bounds[0]) * 0.05
        pad_y = (county_bounds[3] - county_bounds[1]) * 0.05
        extent = (
            county_bounds[0] - pad_x,  # left
            county_bounds[2] + pad_x,  # right
            county_bounds[1] - pad_y,  # bottom
            county_bounds[3] + pad_y   # top
        )
        console.print(f"  Figure extent (full county): {extent}")

        # Create mask from county boundary to clip raster data
        console.print("Creating county mask for clipping...")
        county_mask = geometry_mask(
            county_gdf.geometry,
            out_shape=(height, width),
            transform=transform,
            invert=True  # True inside county, False outside
        )
        console.print(f"  Pixels inside county: {np.sum(county_mask):,}")
    else:
        extent = data_extent

    # Prepare data layers
    total_layer = biomass[0]
    species_layers = biomass[1:]
    forest_mask = total_layer > 0

    # Calculate diversity metrics
    console.print("Calculating diversity metrics...")

    # Species Richness
    richness = np.sum(species_layers > 0, axis=0).astype(float)
    richness[~forest_mask] = np.nan

    # Shannon Diversity
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
    with np.errstate(divide='ignore', invalid='ignore'):
        max_shannon = np.log(np.sum(species_layers > 0, axis=0))
        max_shannon[max_shannon == 0] = np.nan
        evenness = shannon / max_shannon
        evenness[~forest_mask] = np.nan

    # Apply county mask to clip data to county boundary
    if county_mask is not None:
        console.print("Clipping data to county boundary...")
        richness[~county_mask] = np.nan
        shannon[~county_mask] = np.nan
        simpson[~county_mask] = np.nan
        evenness[~county_mask] = np.nan

    # Create figure with improved layout
    console.print("Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{county_name} County, NC - Forest Species Diversity Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # Define metrics to plot
    metrics = [
        (richness, 'Spectral_r', 'Species Richness\n(count per pixel)', 'Species Count', None, None),
        (shannon, 'viridis', "Shannon Diversity (H')\n(information entropy)", "H'", None, None),
        (simpson, 'plasma', 'Simpson Diversity (1-D)\n(probability measure)', '1-D', 0, 1),
        (evenness, 'RdYlGn', "Evenness (Pielou's J)\n(distribution equality)", 'J', 0, 1),
    ]

    for idx, (data, cmap, title, label, vmin, vmax) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Set extent for proper georeferencing
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add basemap first (behind everything)
        try:
            zoom = get_basemap_zoom_level(extent)
            add_basemap(ax, zoom=zoom, source='CartoDB', crs=crs_str, alpha=0.7)
        except Exception as e:
            console.print(f"[dim]Basemap skipped: {e}[/dim]")

        # Plot raster data with transparency (use data_extent for proper positioning)
        im = ax.imshow(data, cmap=cmap, extent=data_extent, origin='upper',
                      alpha=0.85, vmin=vmin, vmax=vmax, zorder=5)

        # Add county boundary overlay
        if county_gdf is not None:
            try:
                plot_boundaries(ax, county_gdf, color='#333333', linewidth=2.5,
                              alpha=0.9, zorder=15)
            except Exception as e:
                console.print(f"[dim]Boundary overlay skipped: {e}[/dim]")

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    # Footer with data attribution
    fig.text(0.5, 0.01,
             'Data: USDA Forest Service FIA BIGMAP 2018 | Basemap: CartoDB | All species included for valid diversity metrics',
             ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    output_path = output_dir / "diversity_composite.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    console.print(f"[green]Saved: {output_path}[/green]")
    plt.close()


def main():
    """Run complete diversity analysis workflow."""
    console.print("[bold green]Species Diversity Analysis[/bold green]")
    console.print("=" * 60)
    console.print()
    console.print("[yellow]This example downloads ALL species to calculate valid diversity metrics.[/yellow]")
    console.print("[yellow]This takes longer than the quickstart but produces ecologically meaningful results.[/yellow]")
    console.print()

    # Setup paths - using Durham County for full county-level analysis
    county_name = "Durham"
    data_dir = Path("durham_diversity_data")
    results_dir = Path("durham_diversity_results")
    zarr_path = data_dir / "durham_all_species.zarr"

    # Initialize API
    api = GridFIA()

    # Get location - using full Durham County
    bbox, crs = get_location_bbox("durham_nc")
    console.print(f"Study Area: {county_name} County, NC (full county)")
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

    # Step 6: Create maps with county boundary overlay
    create_diversity_maps(zarr_path, results_dir / "maps", county_name=county_name)

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
