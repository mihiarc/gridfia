#!/usr/bin/env python3
"""
Virginia Oak-Maple Transition Study
====================================

Research Workflow for Dr. Sarah Chen
Assistant Professor, Virginia Tech
Department of Forest Resources and Environmental Conservation

Research Question: How does the relative dominance of mesophytic species
(maples, beech) versus xerophytic species (oaks) vary across Virginia's
elevational gradient?

This script demonstrates a complete academic research workflow using GridFIA.

FIA Species Codes for Focal Species:
- Red maple (Acer rubrum): 0316
- Sugar maple (Acer saccharum): 0318
- American beech (Fagus grandifolia): 0531
- Northern red oak (Quercus rubra): 0833
- Chestnut oak (Quercus montana): 0832
- White oak (Quercus alba): 0802
- Yellow-poplar (Liriodendron tulipifera): 0621

Note: This example uses synthetic data for demonstration. In production,
real data would be downloaded from the FIA BIGMAP service.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project directories
PROJECT_DIR = Path("examples/virginia_study")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Focal species for oak-maple transition study
FOCAL_SPECIES = {
    # Mesophytic species (maple-beech group)
    "0316": {"name": "Red Maple", "group": "mesophytic", "color": "#e41a1c"},
    "0318": {"name": "Sugar Maple", "group": "mesophytic", "color": "#377eb8"},
    "0531": {"name": "American Beech", "group": "mesophytic", "color": "#4daf4a"},

    # Xerophytic species (oak group)
    "0833": {"name": "Northern Red Oak", "group": "xerophytic", "color": "#984ea3"},
    "0832": {"name": "Chestnut Oak", "group": "xerophytic", "color": "#ff7f00"},
    "0802": {"name": "White Oak", "group": "xerophytic", "color": "#a65628"},

    # Generalist
    "0621": {"name": "Yellow-poplar", "group": "generalist", "color": "#f781bf"},
}

# For synthetic data demo, use readable indices
SPECIES_LIST = list(FOCAL_SPECIES.keys())


def create_synthetic_virginia_data(output_path: Path, size: int = 400) -> Path:
    """
    Create synthetic forest data for Shenandoah Valley / Blue Ridge region.

    Uses REAL geographic coordinates for the George Washington National Forest
    area in western Virginia, with realistic forest patterns based on:
    - Ridge-valley topography (Appalachian fold structure)
    - Elevation-driven species zonation
    - Aspect effects on moisture availability
    - Realistic patch sizes and spatial autocorrelation
    """
    console.print("\n[bold blue]Creating Forest Data for Shenandoah Region, Virginia[/bold blue]")
    console.print("=" * 60)

    # REAL COORDINATES: George Washington National Forest / Shenandoah Valley
    # Area around Harrisonburg, VA - classic oak-maple transition zone
    # WGS84: approximately 38.4°N to 38.7°N, 79.5°W to 79.1°W
    # Web Mercator bounds:
    x_min = -8860000  # Western edge (Blue Ridge)
    x_max = -8815000  # Eastern edge (Shenandoah Valley)
    y_min = 4640000   # Southern edge
    y_max = 4685000   # Northern edge

    # Calculate dimensions based on 30m pixels
    pixel_size = 30.0
    width = int((x_max - x_min) / pixel_size)
    height = int((y_max - y_min) / pixel_size)

    console.print(f"\n[cyan]Study Area: George Washington National Forest Region[/cyan]")
    console.print(f"  Location: Shenandoah Valley / Blue Ridge, Virginia")
    console.print(f"  Approximate center: 38.55°N, 79.3°W")
    console.print(f"  Grid size: {width} x {height} pixels ({width*30/1000:.1f} x {height*30/1000:.1f} km)")
    console.print(f"  Resolution: {pixel_size}m")

    # Create directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create zarr store
    store = zarr.storage.LocalStore(str(output_path))
    root = zarr.open_group(store=store, mode='w')

    n_species = len(FOCAL_SPECIES)
    shape = (n_species + 1, height, width)  # +1 for total

    biomass_array = root.create_array(
        'biomass',
        shape=shape,
        chunks=(1, 256, 256),
        dtype='float32'
    )

    # Generate REALISTIC spatial patterns using fractal noise
    np.random.seed(2024)
    from scipy.ndimage import gaussian_filter, zoom

    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    def generate_fractal_noise(shape, octaves=6, persistence=0.5, scale=1.0):
        """Generate fractal/Perlin-like noise for natural-looking patterns."""
        result = np.zeros(shape)
        amplitude = 1.0
        frequency = scale

        for _ in range(octaves):
            # Generate noise at this frequency
            noise_shape = (max(2, int(shape[0] * frequency)), max(2, int(shape[1] * frequency)))
            noise = np.random.randn(*noise_shape)
            # Smooth and resize to target shape
            noise = gaussian_filter(noise, sigma=1)
            noise = zoom(noise, (shape[0] / noise_shape[0], shape[1] / noise_shape[1]), order=1)
            # Ensure correct shape after zoom
            noise = noise[:shape[0], :shape[1]]

            result += amplitude * noise
            amplitude *= persistence
            frequency *= 2

        # Normalize to 0-1
        result = (result - result.min()) / (result.max() - result.min())
        return result

    # REALISTIC ELEVATION using fractal noise
    # Base terrain with large-scale features
    elevation_base = generate_fractal_noise((height, width), octaves=4, persistence=0.6, scale=0.05)

    # Add NE-SW trending ridge structure (but irregular, not sinusoidal)
    ridge_angle = 0.35
    ridge_coord = X * np.cos(ridge_angle) + Y * np.sin(ridge_angle)
    ridge_structure = generate_fractal_noise((height, width), octaves=3, persistence=0.5, scale=0.08)
    ridge_structure = ridge_structure * 0.4 + ridge_coord * 0.6  # Blend fractal with linear trend

    # Combine for final elevation
    elevation = elevation_base * 0.3 + ridge_structure * 0.7
    elevation = elevation * 900 + 300  # Scale to 300-1200m range

    # MOISTURE based on local topographic position (valleys are wetter)
    # Use smoothed elevation gradient as proxy
    elevation_smooth = gaussian_filter(elevation, sigma=20)
    moisture = 1 - (elevation - elevation_smooth + 200) / 400  # Valleys wet, ridges dry
    moisture = np.clip(moisture, 0, 1)
    moisture = gaussian_filter(moisture, sigma=5)  # Smooth transitions

    # FOREST MASK with realistic patterns matching Shenandoah Valley
    # Calculate slope from elevation
    slope = np.sqrt(np.gradient(elevation, axis=0)**2 + np.gradient(elevation, axis=1)**2)

    # Normalize elevation for thresholding (0-1 range)
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    # VALLEY AGRICULTURE: Low elevation + relatively flat = farmland
    # Shenandoah Valley is heavily agricultural
    valley_ag = (elev_norm < 0.25) & (slope < 20)

    # Expand valley agriculture with noise for irregular edges
    ag_noise = generate_fractal_noise((height, width), octaves=4, persistence=0.6, scale=0.1)
    valley_ag = valley_ag | ((elev_norm < 0.35) & (ag_noise > 0.6) & (slope < 25))

    # URBAN/DEVELOPED: Towns in valleys (Harrisonburg, Staunton, etc.)
    urban_noise = generate_fractal_noise((height, width), octaves=3, persistence=0.5, scale=0.08)
    # Create clustered urban patches in valley
    urban_centers = (elev_norm < 0.3) & (urban_noise > 0.75)
    # Expand urban areas slightly
    urban_mask = gaussian_filter(urban_centers.astype(float), sigma=8) > 0.3

    # ROADS/POWERLINES: Linear clearings
    road_noise = generate_fractal_noise((height, width), octaves=2, persistence=0.4, scale=0.05)
    roads = (road_noise > 0.48) & (road_noise < 0.52) & (elev_norm < 0.5)

    # NATURAL GAPS: Logging cuts, rock outcrops, etc.
    gap_noise = generate_fractal_noise((height, width), octaves=5, persistence=0.55, scale=0.2)
    natural_gaps = gap_noise > 0.88

    # HIGH ELEVATION: Rocky ridgetops with sparse vegetation
    ridge_sparse = (elev_norm > 0.9) & (gap_noise > 0.5)

    # Combine all non-forest areas
    non_forest = valley_ag | urban_mask | roads | natural_gaps | ridge_sparse

    # Create forest mask with smooth edges
    forest_prob = 1 - non_forest.astype(float)
    forest_prob = gaussian_filter(forest_prob, sigma=3)

    # Add some randomness to edges
    edge_noise = generate_fractal_noise((height, width), octaves=4, persistence=0.5, scale=0.15)
    forest_mask = (forest_prob + edge_noise * 0.2) > 0.5

    console.print(f"\n  Non-forest area: {(~forest_mask).sum() / forest_mask.size * 100:.1f}%")
    console.print(f"    - Valley agriculture: {valley_ag.sum() / forest_mask.size * 100:.1f}%")
    console.print(f"    - Urban/developed: {urban_mask.sum() / forest_mask.size * 100:.1f}%")

    total_biomass = np.zeros((height, width), dtype='float32')

    console.print("\nGenerating species distributions based on topography:")

    for idx, (code, info) in enumerate(FOCAL_SPECIES.items()):
        layer_idx = idx + 1

        # Generate species-specific fractal noise for patchy distribution
        np.random.seed(2024 + idx * 100)  # Different seed per species
        species_noise = generate_fractal_noise((height, width), octaves=5, persistence=0.55, scale=0.12)

        # BASE HABITAT SUITABILITY by functional group
        if info['group'] == 'xerophytic':
            # OAKS: prefer ridges, drier sites, mid-high elevation
            habitat_preference = (
                (elevation - 400) / 500 * 0.5 +  # Higher elevation preferred
                (1 - moisture) * 0.5              # Drier sites preferred
            )
            habitat_preference = np.clip(habitat_preference, 0, 1)

        elif info['group'] == 'mesophytic':
            # MAPLES/BEECH: prefer coves, moist sites, lower-mid elevation
            habitat_preference = (
                (1 - (elevation - 400) / 600) * 0.4 +  # Lower elevation preferred
                moisture * 0.6                          # Moist sites preferred
            )
            habitat_preference = np.clip(habitat_preference, 0, 1)

        else:  # generalist (yellow-poplar)
            # YELLOW-POPLAR: coves and lower slopes, very moist sites
            habitat_preference = (
                (1 - (elevation - 350) / 500) * 0.3 +  # Lower elevation
                moisture * 0.7                          # Very moist preferred
            )
            habitat_preference = np.clip(habitat_preference, 0, 1)

        # Combine habitat preference with species-specific noise
        # This creates patchy distributions within suitable habitat
        suitability = habitat_preference * 0.6 + species_noise * 0.4

        # Convert suitability to biomass with realistic distribution
        # Use a threshold to create presence/absence patterns
        presence_threshold = 0.3 + np.random.rand() * 0.1  # Varies by species
        biomass = np.where(suitability > presence_threshold,
                         suitability * 100 + np.random.exponential(20, (height, width)),
                         np.random.exponential(2, (height, width)))

        # Apply spatial smoothing for realistic autocorrelation
        biomass = gaussian_filter(biomass, sigma=3)

        # Apply forest mask
        biomass = biomass * forest_mask

        # Add fine-scale variation
        fine_noise = generate_fractal_noise((height, width), octaves=3, persistence=0.5, scale=0.3)
        biomass = biomass * (0.7 + fine_noise * 0.6)

        biomass = np.clip(biomass, 0, 150).astype('float32')

        biomass_array[layer_idx, :, :] = biomass
        total_biomass += biomass

        forest_pixels = forest_mask.sum()
        mean_biomass = biomass[forest_mask].mean() if forest_pixels > 0 else 0
        coverage = (biomass > 1).sum() / forest_pixels * 100 if forest_pixels > 0 else 0
        console.print(f"  {info['name']:20s} - Mean: {mean_biomass:5.1f} Mg/ha, Coverage: {coverage:5.1f}%")

    # Store total in first layer
    biomass_array[0, :, :] = total_biomass

    # METADATA with real coordinates
    root.attrs['crs'] = 'EPSG:3857'
    root.attrs['num_species'] = n_species + 1
    root.attrs['location'] = 'George Washington National Forest, Virginia'
    root.attrs['region'] = 'Shenandoah Valley / Blue Ridge'
    root.attrs['center_lat'] = 38.55
    root.attrs['center_lon'] = -79.3
    root.attrs['year'] = '2024'
    root.attrs['source'] = 'GridFIA Synthetic Data (realistic patterns)'
    root.attrs['researcher'] = 'Dr. Sarah Chen, Virginia Tech'
    root.attrs['created'] = datetime.now().isoformat()

    # REAL transform with actual coordinates
    root.attrs['transform'] = [pixel_size, 0.0, x_min, 0.0, -pixel_size, y_max]
    root.attrs['bounds'] = [x_min, y_min, x_max, y_max]

    # Species metadata
    codes = ['0000'] + list(FOCAL_SPECIES.keys())
    names = ['Total Biomass'] + [info['name'] for info in FOCAL_SPECIES.values()]

    root.create_array('species_codes', data=np.array(codes, dtype='U10'))
    root.create_array('species_names', data=np.array(names, dtype='U50'))

    console.print(f"\n[green]Created zarr store: {output_path}[/green]")
    console.print(f"  Shape: {shape}")
    console.print(f"  Bounds: [{x_min}, {y_min}, {x_max}, {y_max}]")
    console.print(f"  Forest coverage: {forest_mask.sum() / forest_mask.size * 100:.1f}%")
    console.print(f"  Total biomass range: {total_biomass[forest_mask].min():.1f} - {total_biomass[forest_mask].max():.1f} Mg/ha")

    return output_path


def calculate_oak_maple_metrics(zarr_path: Path) -> dict:
    """
    Calculate oak-maple transition metrics.

    Metrics calculated:
    - Oak biomass (sum of 3 oak species)
    - Maple biomass (sum of 2 maple species)
    - Oak:Maple ratio (indicator of mesophication)
    - Shannon diversity for the 7 focal species
    - Dominant species at each pixel
    """
    console.print("\n[bold blue]Calculating Oak-Maple Transition Metrics[/bold blue]")
    console.print("=" * 60)

    root = zarr.open(str(zarr_path), mode='r')
    data = root['biomass'][:]
    species_names = list(root['species_names'][:])

    results = {}

    # Get indices for each group
    oak_indices = []
    maple_indices = []
    beech_index = None
    poplar_index = None

    for idx, (code, info) in enumerate(FOCAL_SPECIES.items()):
        layer_idx = idx + 1
        if 'Oak' in info['name']:
            oak_indices.append(layer_idx)
        elif 'Maple' in info['name']:
            maple_indices.append(layer_idx)
        elif 'Beech' in info['name']:
            beech_index = layer_idx
        elif 'poplar' in info['name']:
            poplar_index = layer_idx

    # Calculate group biomass
    console.print("\n1. Calculating group biomass totals...")

    oak_biomass = np.sum(data[oak_indices, :, :], axis=0)
    maple_biomass = np.sum(data[maple_indices, :, :], axis=0)
    mesophytic_biomass = maple_biomass + data[beech_index, :, :]

    results['oak_biomass'] = oak_biomass
    results['maple_biomass'] = maple_biomass
    results['mesophytic_biomass'] = mesophytic_biomass

    console.print(f"   Oak group mean: {oak_biomass.mean():.1f} Mg/ha")
    console.print(f"   Maple group mean: {maple_biomass.mean():.1f} Mg/ha")
    console.print(f"   Mesophytic (maple+beech) mean: {mesophytic_biomass.mean():.1f} Mg/ha")

    # Calculate Oak:Maple ratio
    console.print("\n2. Calculating Oak:Maple ratio (mesophication indicator)...")

    total = oak_biomass + maple_biomass
    oak_maple_ratio = np.zeros_like(oak_biomass)
    mask = total > 0
    oak_maple_ratio[mask] = oak_biomass[mask] / total[mask]

    results['oak_maple_ratio'] = oak_maple_ratio
    results['total_biomass'] = data[0]

    # Interpret ratio
    oak_dominated = (oak_maple_ratio > 0.6) & mask
    maple_dominated = (oak_maple_ratio < 0.4) & mask
    mixed = mask & ~oak_dominated & ~maple_dominated

    console.print(f"   Oak-dominated pixels (>60%): {oak_dominated.sum() / mask.sum() * 100:.1f}%")
    console.print(f"   Maple-dominated pixels (<40%): {maple_dominated.sum() / mask.sum() * 100:.1f}%")
    console.print(f"   Mixed composition: {mixed.sum() / mask.sum() * 100:.1f}%")

    # Shannon diversity for focal species
    console.print("\n3. Calculating Shannon diversity index...")

    species_data = data[1:, :, :]  # Exclude total
    species_total = np.sum(species_data, axis=0)

    shannon = np.zeros_like(oak_biomass)
    for i in range(len(FOCAL_SPECIES)):
        p = np.zeros_like(oak_biomass)
        valid = species_total > 0
        p[valid] = species_data[i][valid] / species_total[valid]

        # Shannon formula: -sum(p * ln(p))
        nonzero = p > 0
        shannon[nonzero] -= p[nonzero] * np.log(p[nonzero])

    results['shannon_diversity'] = shannon
    console.print(f"   Mean Shannon index: {shannon[mask].mean():.2f}")
    console.print(f"   Max Shannon index: {shannon.max():.2f}")

    # Dominant species
    console.print("\n4. Identifying dominant species at each pixel...")

    dominant = np.argmax(species_data, axis=0)
    results['dominant_species'] = dominant

    # Count dominance
    table = Table(title="Species Dominance Summary")
    table.add_column("Species", style="cyan")
    table.add_column("Pixels Dominated", justify="right")
    table.add_column("Percentage", justify="right")

    for idx, (code, info) in enumerate(FOCAL_SPECIES.items()):
        count = (dominant == idx).sum()
        pct = count / dominant.size * 100
        table.add_row(info['name'], str(count), f"{pct:.1f}%")

    console.print(table)

    # Species richness
    console.print("\n5. Calculating species richness...")
    richness = np.sum(species_data > 1, axis=0)  # Species with >1 Mg/ha
    results['species_richness'] = richness
    console.print(f"   Mean richness: {richness.mean():.1f} species/pixel")
    console.print(f"   Max richness: {richness.max()} species")

    return results


def create_publication_figures(zarr_path: Path, metrics: dict, output_dir: Path):
    """
    Create publication-ready figures for the oak-maple transition study.

    Figures created:
    0. Location reference map showing study area
    1. Six-panel species distribution maps with basemap
    2. Oak vs Maple dominance map with geographic context
    3. Diversity and richness summary
    4. Manuscript composite figure
    """
    console.print("\n[bold blue]Creating Publication Figures[/bold blue]")
    console.print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and metadata
    root = zarr.open(str(zarr_path), mode='r')
    data = root['biomass'][:]
    bounds = root.attrs.get('bounds', [-8860000, 4640000, -8815000, 4685000])
    transform = root.attrs.get('transform', [30, 0, bounds[0], 0, -30, bounds[3]])

    # Calculate extent for imshow (left, right, bottom, top)
    height, width = data.shape[1], data.shape[2]
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    console.print(f"  Data bounds: {bounds}")
    console.print(f"  Grid size: {width} x {height}")

    # Import contextily for basemaps
    try:
        import contextily as ctx
        HAS_BASEMAP = True
        console.print("  [green]Basemap support enabled (contextily)[/green]")
    except ImportError:
        HAS_BASEMAP = False
        console.print("  [yellow]Basemap not available (install contextily)[/yellow]")

    # =========================================================================
    # Figure 0: Location Reference Map
    # =========================================================================
    console.print("\n0. Creating location reference map...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Regional context (wider view showing Virginia)
    ax = axes[0]
    # Wider bounds to show regional context (roughly central Virginia)
    regional_bounds = [bounds[0] - 150000, bounds[2] + 150000,
                      bounds[1] - 100000, bounds[3] + 100000]
    ax.set_xlim(regional_bounds[0], regional_bounds[1])
    ax.set_ylim(regional_bounds[2], regional_bounds[3])

    if HAS_BASEMAP:
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik, zoom=9)
        except Exception as e:
            console.print(f"   [yellow]Regional basemap failed: {e}[/yellow]")

    # Draw study area rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((bounds[0], bounds[1]), bounds[2]-bounds[0], bounds[3]-bounds[1],
                     linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
    ax.add_patch(rect)

    # Add label
    ax.text(bounds[0] + (bounds[2]-bounds[0])/2, bounds[3] + 5000,
           'Study Area', ha='center', va='bottom', fontsize=12, fontweight='bold',
           color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title('A) Regional Context\nShenandoah Valley, Virginia', fontsize=12, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))

    # Panel B: Study area with terrain basemap only (no data overlay)
    ax = axes[1]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    if HAS_BASEMAP:
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery, zoom=11)
        except Exception as e:
            console.print(f"   [yellow]Terrain basemap failed: {e}[/yellow]")

    # Add study area boundary
    rect = Rectangle((bounds[0], bounds[1]), bounds[2]-bounds[0], bounds[3]-bounds[1],
                     linewidth=2, edgecolor='yellow', facecolor='none', linestyle='-')
    ax.add_patch(rect)

    ax.set_title('B) Study Area: George Washington National Forest\n'
                'Satellite View (45 × 45 km)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))

    # Add coordinate annotation
    fig.text(0.5, 0.02,
            'Study Area Center: 38.55°N, 79.30°W | George Washington National Forest, Virginia\n'
            'Coordinate System: EPSG:3857 (Web Mercator) | Data Resolution: 30m',
            ha='center', fontsize=10, style='italic')

    fig.suptitle('Study Area Location Reference', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    fig0_path = output_dir / "fig0_location_reference.png"
    plt.savefig(fig0_path, facecolor='white', dpi=200)
    plt.close()
    console.print(f"   Saved: {fig0_path}")

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # =========================================================================
    # Figure 1: Six-panel species distribution with basemap
    # =========================================================================
    console.print("\n1. Creating species distribution panels with basemap...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle('Distribution of Focal Tree Species\nGeorge Washington National Forest Region, Virginia',
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, (code, info) in enumerate(list(FOCAL_SPECIES.items())[:6]):
        ax = axes.flat[idx]
        layer_idx = idx + 1

        # Add basemap first (underneath data)
        if HAS_BASEMAP:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            try:
                ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap,
                               alpha=0.4, attribution_size=6)
            except Exception as e:
                console.print(f"   [yellow]Basemap failed: {e}[/yellow]")

        species_data = data[layer_idx]
        # Mask zero values for transparency
        masked_data = np.ma.masked_where(species_data < 0.5, species_data)
        vmax = np.percentile(species_data[species_data > 0], 98) if (species_data > 0).any() else 100

        # Use group-appropriate colormap
        if info['group'] == 'mesophytic':
            cmap = 'Blues'
        elif info['group'] == 'xerophytic':
            cmap = 'Oranges'
        else:
            cmap = 'Greens'

        im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=vmax,
                      extent=extent, origin='upper', alpha=0.85)
        ax.set_title(f"{info['name']}\n({info['group'].capitalize()})", fontsize=11, fontweight='bold')

        # Format axes with coordinates
        ax.set_xlabel('Easting (m)', fontsize=8)
        ax.set_ylabel('Northing (m)', fontsize=8)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
        ax.tick_params(labelsize=7)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Biomass (Mg/ha)', fontsize=8)

    plt.tight_layout()
    fig1_path = output_dir / "fig1_species_distributions.png"
    plt.savefig(fig1_path, facecolor='white', dpi=200)
    plt.close()
    console.print(f"   Saved: {fig1_path}")

    # =========================================================================
    # Figure 2: Oak-Maple Dominance Map with basemap
    # =========================================================================
    console.print("\n2. Creating oak-maple dominance map with geographic context...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Oak biomass
    ax = axes[0]
    if HAS_BASEMAP:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.3)
        except:
            pass

    oak_data = metrics['oak_biomass']
    masked_oak = np.ma.masked_where(oak_data < 1, oak_data)
    vmax = np.percentile(oak_data[oak_data > 0], 98)
    im = ax.imshow(masked_oak, cmap='Oranges', vmin=0, vmax=vmax,
                  extent=extent, origin='upper', alpha=0.85)
    ax.set_title('A) Oak Group Biomass\n(N. Red Oak + Chestnut Oak + White Oak)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Biomass (Mg/ha)')

    # Panel B: Mesophytic biomass
    ax = axes[1]
    if HAS_BASEMAP:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.3)
        except:
            pass

    meso_data = metrics['mesophytic_biomass']
    masked_meso = np.ma.masked_where(meso_data < 1, meso_data)
    vmax = np.percentile(meso_data[meso_data > 0], 98)
    im = ax.imshow(masked_meso, cmap='Blues', vmin=0, vmax=vmax,
                  extent=extent, origin='upper', alpha=0.85)
    ax.set_title('B) Mesophytic Group Biomass\n(Red Maple + Sugar Maple + Am. Beech)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Biomass (Mg/ha)')

    # Panel C: Oak proportion (mesophication indicator)
    ax = axes[2]
    if HAS_BASEMAP:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.3)
        except:
            pass

    ratio = metrics['oak_maple_ratio']
    # Mask non-forest areas
    masked_ratio = np.ma.masked_where(metrics['total_biomass'] < 10, ratio)

    # Custom diverging colormap: blue (maple) - white (mixed) - orange (oak)
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7',
              '#fddbc7', '#ef8a62', '#b2182b']
    cmap_div = LinearSegmentedColormap.from_list('oak_maple', colors)

    im = ax.imshow(masked_ratio, cmap=cmap_div, vmin=0, vmax=1,
                  extent=extent, origin='upper', alpha=0.9)
    ax.set_title('C) Oak Proportion\n(Mesophication Indicator)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label('Oak / (Oak + Maple)')
    cbar.ax.set_yticklabels(['0.0\nMaple', '0.25', '0.5\nMixed', '0.75', '1.0\nOak'])

    fig.suptitle('Oak-Maple Transition: Shenandoah Valley / Blue Ridge, Virginia\n'
                 'George Washington National Forest Region (38.4°N - 38.7°N, 79.1°W - 79.5°W)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig2_path = output_dir / "fig2_oak_maple_dominance.png"
    plt.savefig(fig2_path, facecolor='white', dpi=200)
    plt.close()
    console.print(f"   Saved: {fig2_path}")

    # =========================================================================
    # Figure 3: Diversity and Richness with basemap
    # =========================================================================
    console.print("\n3. Creating diversity summary figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Shannon diversity
    ax = axes[0]
    if HAS_BASEMAP:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.3)
        except:
            pass

    shannon = metrics['shannon_diversity']
    masked_shannon = np.ma.masked_where(metrics['total_biomass'] < 10, shannon)
    im = ax.imshow(masked_shannon, cmap='viridis', vmin=0, extent=extent, origin='upper', alpha=0.85)
    ax.set_title("A) Shannon Diversity Index (H')", fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("H'")

    # Panel B: Species richness
    ax = axes[1]
    if HAS_BASEMAP:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.3)
        except:
            pass

    richness = metrics['species_richness']
    masked_richness = np.ma.masked_where(metrics['total_biomass'] < 10, richness)
    im = ax.imshow(masked_richness, cmap='Spectral_r', vmin=0, vmax=7,
                  extent=extent, origin='upper', alpha=0.85)
    ax.set_title('B) Species Richness\n(species with >1 Mg/ha)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Species')

    # Panel C: Dominant species
    ax = axes[2]
    if HAS_BASEMAP:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        try:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.3)
        except:
            pass

    dominant = metrics['dominant_species']
    # Mask non-forest
    masked_dominant = np.ma.masked_where(metrics['total_biomass'] < 10, dominant)

    # Create custom colormap for species
    species_colors = [info['color'] for info in FOCAL_SPECIES.values()]
    species_cmap = LinearSegmentedColormap.from_list('species', species_colors, N=7)

    im = ax.imshow(masked_dominant, cmap=species_cmap, vmin=0, vmax=6,
                  extent=extent, origin='upper', alpha=0.9)
    ax.set_title('C) Dominant Species', fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))

    # Add legend
    legend_elements = [Patch(facecolor=info['color'], label=info['name'][:15])
                      for info in FOCAL_SPECIES.values()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)

    fig.suptitle('Forest Diversity Metrics: Shenandoah Valley / Blue Ridge, Virginia',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig3_path = output_dir / "fig3_diversity_metrics.png"
    plt.savefig(fig3_path, facecolor='white', dpi=200)
    plt.close()
    console.print(f"   Saved: {fig3_path}")

    # =========================================================================
    # Figure 4: Summary composite for manuscript with basemaps
    # =========================================================================
    console.print("\n4. Creating manuscript composite figure...")

    fig = plt.figure(figsize=(18, 14))

    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)

    # Helper function to add basemap
    def add_panel_basemap(ax):
        if HAS_BASEMAP:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            try:
                ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldTopoMap, alpha=0.25)
            except:
                pass

    # Row 1: Three main species (one from each group)
    for i, (code, info) in enumerate([
        ('0316', FOCAL_SPECIES['0316']),  # Red Maple
        ('0833', FOCAL_SPECIES['0833']),  # N. Red Oak
        ('0621', FOCAL_SPECIES['0621']),  # Yellow-poplar
    ]):
        ax = fig.add_subplot(gs[0, i])
        add_panel_basemap(ax)

        layer_idx = list(FOCAL_SPECIES.keys()).index(code) + 1
        species_data = data[layer_idx]
        masked_species = np.ma.masked_where(species_data < 0.5, species_data)
        vmax = np.percentile(species_data[species_data > 0], 98) if (species_data > 0).any() else 100

        im = ax.imshow(masked_species, cmap='YlGn', vmin=0, vmax=vmax,
                      extent=extent, origin='upper', alpha=0.85)
        ax.set_title(f"{info['name']}", fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
        plt.colorbar(im, ax=ax, shrink=0.7, label='Mg/ha')

    # Row 1, col 4: Total biomass
    ax = fig.add_subplot(gs[0, 3])
    add_panel_basemap(ax)

    total = metrics['total_biomass']
    masked_total = np.ma.masked_where(total < 10, total)
    vmax = np.percentile(total[total > 0], 98)
    im = ax.imshow(masked_total, cmap='YlGn', vmin=0, vmax=vmax,
                  extent=extent, origin='upper', alpha=0.85)
    ax.set_title('Total Biomass', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    plt.colorbar(im, ax=ax, shrink=0.7, label='Mg/ha')

    # Row 2: Oak-Maple analysis
    ax = fig.add_subplot(gs[1, :2])
    add_panel_basemap(ax)

    ratio = metrics['oak_maple_ratio']
    masked_ratio = np.ma.masked_where(total < 10, ratio)
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7',
              '#fddbc7', '#ef8a62', '#b2182b']
    cmap_oak = LinearSegmentedColormap.from_list('oak_maple', colors)
    im = ax.imshow(masked_ratio, cmap=cmap_oak, vmin=0, vmax=1,
                  extent=extent, origin='upper', alpha=0.9)
    ax.set_title('Oak Proportion (Mesophication Indicator)', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, orientation='horizontal', pad=0.08)
    cbar.set_label('Oak / (Oak + Maple): 0=Maple Dominated, 1=Oak Dominated', fontsize=9)

    ax = fig.add_subplot(gs[1, 2:])
    add_panel_basemap(ax)

    shannon = metrics['shannon_diversity']
    masked_shannon = np.ma.masked_where(total < 10, shannon)
    im = ax.imshow(masked_shannon, cmap='viridis', extent=extent, origin='upper', alpha=0.85)
    ax.set_title('Shannon Diversity Index', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, orientation='horizontal', pad=0.08)
    cbar.set_label("H' (higher = more diverse)", fontsize=9)

    # Row 3: Summary statistics text
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # Calculate summary stats
    mask = total > 0
    oak_mean = metrics['oak_biomass'][mask].mean()
    maple_mean = metrics['maple_biomass'][mask].mean()
    diversity_mean = shannon[mask].mean()
    richness_mean = metrics['species_richness'][mask].mean()

    oak_pct = (ratio[mask] > 0.6).sum() / mask.sum() * 100
    maple_pct = (ratio[mask] < 0.4).sum() / mask.sum() * 100
    mixed_pct = 100 - oak_pct - maple_pct

    summary_text = f"""
    Study Area Summary Statistics
    {'='*50}

    Total Forested Area: {mask.sum():,} pixels ({mask.sum() * 0.09:.1f} ha at 30m resolution)
    Mean Total Biomass: {total[mask].mean():.1f} Mg/ha

    Group Biomass:
      - Oak group (3 species): {oak_mean:.1f} Mg/ha mean
      - Maple-Beech group (3 species): {maple_mean:.1f} Mg/ha mean
      - Yellow-poplar: {data[list(FOCAL_SPECIES.keys()).index('0621')+1][mask].mean():.1f} Mg/ha mean

    Composition Classes:
      - Oak-dominated (>60% oak): {oak_pct:.1f}% of forest area
      - Maple-dominated (<40% oak): {maple_pct:.1f}% of forest area
      - Mixed composition: {mixed_pct:.1f}% of forest area

    Diversity Metrics:
      - Mean Shannon Index: {diversity_mean:.2f}
      - Mean Species Richness: {richness_mean:.1f} species/pixel

    Data Source: Synthetic demonstration data
    Analysis: GridFIA Python Package
    Date: {datetime.now().strftime('%Y-%m-%d')}
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Oak-Maple Transition Analysis: Virginia Study Area\n'
                 'Dr. Sarah Chen, Virginia Tech - Forest Ecology',
                 fontsize=14, fontweight='bold', y=0.98)

    fig4_path = output_dir / "fig4_manuscript_composite.png"
    plt.savefig(fig4_path, facecolor='white', dpi=300)
    plt.close()
    console.print(f"   Saved: {fig4_path}")

    return [fig1_path, fig2_path, fig3_path, fig4_path]


def export_statistics(zarr_path: Path, metrics: dict, output_dir: Path):
    """Export statistics to CSV for further analysis in R."""
    console.print("\n[bold blue]Exporting Statistics for R Analysis[/bold blue]")
    console.print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    root = zarr.open(str(zarr_path), mode='r')
    data = root['biomass'][:]

    # Create summary DataFrame
    import csv

    # 1. Pixel-level data sample (for spatial regression)
    console.print("\n1. Exporting pixel sample for spatial analysis...")

    sample_path = output_dir / "pixel_sample.csv"

    # Sample every 10th pixel for manageable file size
    step = 10
    rows = []

    for i in range(0, data.shape[1], step):
        for j in range(0, data.shape[2], step):
            if metrics['total_biomass'][i, j] > 0:  # Forest pixels only
                row = {
                    'row': i,
                    'col': j,
                    'total_biomass': float(metrics['total_biomass'][i, j]),
                    'oak_biomass': float(metrics['oak_biomass'][i, j]),
                    'maple_biomass': float(metrics['maple_biomass'][i, j]),
                    'oak_proportion': float(metrics['oak_maple_ratio'][i, j]),
                    'shannon_diversity': float(metrics['shannon_diversity'][i, j]),
                    'species_richness': int(metrics['species_richness'][i, j]),
                    'dominant_species': int(metrics['dominant_species'][i, j]),
                }
                # Add individual species
                for idx, (code, info) in enumerate(FOCAL_SPECIES.items()):
                    row[f"biomass_{code}"] = float(data[idx + 1, i, j])
                rows.append(row)

    with open(sample_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"   Saved {len(rows)} pixel samples to: {sample_path}")

    # 2. Species summary statistics
    console.print("\n2. Exporting species summary statistics...")

    summary_path = output_dir / "species_summary.csv"

    mask = metrics['total_biomass'] > 0
    summary_rows = []

    for idx, (code, info) in enumerate(FOCAL_SPECIES.items()):
        layer_idx = idx + 1
        species_data = data[layer_idx]

        present_mask = species_data > 0

        summary_rows.append({
            'species_code': code,
            'common_name': info['name'],
            'functional_group': info['group'],
            'mean_biomass_all': float(species_data[mask].mean()),
            'mean_biomass_present': float(species_data[present_mask].mean()) if present_mask.any() else 0,
            'max_biomass': float(species_data.max()),
            'std_biomass': float(species_data[mask].std()),
            'coverage_pct': float(present_mask.sum() / mask.sum() * 100),
            'pixels_present': int(present_mask.sum()),
            'pixels_dominant': int((metrics['dominant_species'] == idx).sum()),
        })

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    console.print(f"   Saved species summary to: {summary_path}")

    # 3. Study metadata
    console.print("\n3. Exporting study metadata...")

    metadata_path = output_dir / "study_metadata.txt"

    metadata = f"""
Virginia Oak-Maple Transition Study
===================================

Researcher: Dr. Sarah Chen
Institution: Virginia Tech
Department: Forest Resources and Environmental Conservation
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Source
-----------
Type: Synthetic demonstration data (for workflow testing)
Resolution: 30m x 30m pixels
CRS: EPSG:3857 (Web Mercator)
Grid Size: {data.shape[1]} x {data.shape[2]} pixels

Focal Species (n=7)
-------------------
Mesophytic Group:
  - 0316: Red Maple (Acer rubrum)
  - 0318: Sugar Maple (Acer saccharum)
  - 0531: American Beech (Fagus grandifolia)

Xerophytic Group:
  - 0833: Northern Red Oak (Quercus rubra)
  - 0832: Chestnut Oak (Quercus montana)
  - 0802: White Oak (Quercus alba)

Generalist:
  - 0621: Yellow-poplar (Liriodendron tulipifera)

Analysis Metrics
----------------
- Oak:Maple biomass ratio (mesophication indicator)
- Shannon diversity index (H')
- Species richness (species with >1 Mg/ha)
- Dominant species classification

Software
--------
Package: GridFIA (Spatial Forest Analysis Toolkit)
Python Version: 3.12+
Key Dependencies: zarr, numpy, matplotlib, rich

Citation
--------
Chen, S. (2024). Oak-maple transition analysis using GridFIA.
Virginia Tech Forest Dynamics Lab.
"""

    with open(metadata_path, 'w') as f:
        f.write(metadata)

    console.print(f"   Saved metadata to: {metadata_path}")

    return [sample_path, summary_path, metadata_path]


def main():
    """Run complete Virginia oak-maple transition analysis."""

    console.print(Panel.fit(
        "[bold green]Virginia Oak-Maple Transition Study[/bold green]\n"
        "Dr. Sarah Chen | Virginia Tech | Forest Dynamics Lab\n"
        "Research Question: How does oak vs. maple dominance vary\n"
        "across Virginia's elevational gradient?",
        title="GridFIA Research Workflow Demo"
    ))

    # Create project directories
    for d in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    zarr_path = DATA_DIR / "virginia_forest.zarr"

    # Step 1: Create or load data
    if zarr_path.exists():
        console.print(f"\n[yellow]Using existing data: {zarr_path}[/yellow]")
    else:
        zarr_path = create_synthetic_virginia_data(zarr_path)

    # Step 2: Calculate metrics
    metrics = calculate_oak_maple_metrics(zarr_path)

    # Step 3: Create publication figures
    figures = create_publication_figures(zarr_path, metrics, FIGURES_DIR)

    # Step 4: Export statistics
    exports = export_statistics(zarr_path, metrics, OUTPUT_DIR / "statistics")

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Analysis Complete![/bold green]")
    console.print("=" * 60)

    console.print("\n[bold]Output Files:[/bold]")
    console.print("\nFigures (for manuscript):")
    for f in figures:
        console.print(f"  - {f}")

    console.print("\nStatistics (for R analysis):")
    for f in exports:
        console.print(f"  - {f}")

    console.print("\n[dim]To reproduce this analysis:[/dim]")
    console.print("  python examples/virginia_oak_maple_study.py")

    console.print("\n[dim]For real data, replace synthetic data creation with:[/dim]")
    console.print("  from gridfia import GridFIA")
    console.print("  api = GridFIA()")
    console.print("  api.download_species(state='VA', species_codes=['0316', '0318', ...])")


if __name__ == "__main__":
    main()
