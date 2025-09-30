#!/usr/bin/env python3
"""
Create visualizations for Lane County forest data.

This script:
1. Loads calculated metrics from the metrics directory
2. Creates diversity maps (Shannon, Simpson)
3. Creates species richness maps
4. Creates species-specific biomass maps
5. Creates comparison and custom visualizations
"""

from pathlib import Path
import logging
from bigmap import BigMapAPI
import numpy as np
import zarr

# Set matplotlib to use non-interactive backend to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_custom_visualizations(zarr_path: Path, metrics_dir: Path, maps_dir: Path):
    """
    Create custom visualizations specific to Lane County's forest characteristics.
    """
    logger.info("Creating custom visualizations...")

    # Load data
    store = zarr.open(zarr_path, mode='r')
    biomass = store['biomass']
    species_codes = store['species_codes'][:]
    species_names = store['species_names'][:]

    # Load custom metrics if available (check both .zarr and .tif formats)
    dominant_species_path = metrics_dir / "dominant_species.zarr"
    if not dominant_species_path.exists():
        dominant_species_path = metrics_dir / "dominant_species.tif"

    conifer_ratio_path = metrics_dir / "conifer_ratio.zarr"
    if not conifer_ratio_path.exists():
        conifer_ratio_path = metrics_dir / "conifer_ratio.tif"

    if dominant_species_path.exists():
        logger.info("  Creating dominant species map...")
        if dominant_species_path.suffix == '.zarr':
            dominant_species = zarr.open(dominant_species_path, mode='r')[:]
        else:
            import rasterio
            with rasterio.open(dominant_species_path) as src:
                dominant_species = src.read(1)

        # Create a custom colormap for top species
        # Get unique species and their counts
        unique_species, counts = np.unique(dominant_species[dominant_species >= 0], return_counts=True)

        # Get top 10 species by area
        top_indices = np.argsort(counts)[-10:][::-1]
        top_species = unique_species[top_indices]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Create a discrete colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_species)))

        # Create display data (only show top species, others as gray)
        display_data = np.full_like(dominant_species, -1, dtype=np.float32)
        for i, sp_idx in enumerate(top_species):
            display_data[dominant_species == sp_idx] = i

        # Plot
        im = ax.imshow(display_data, cmap=ListedColormap(colors), vmin=0, vmax=len(top_species)-1)
        ax.set_title("Dominant Species Distribution - Lane County", fontsize=16, fontweight='bold')
        ax.axis('off')

        # Create legend
        legend_elements = []
        for i, sp_idx in enumerate(top_species[:10]):
            if sp_idx < len(species_codes):
                code = species_codes[sp_idx].decode() if isinstance(species_codes[sp_idx], bytes) else species_codes[sp_idx]
                name = species_names[sp_idx].decode() if isinstance(species_names[sp_idx], bytes) else species_names[sp_idx]
                if not name:
                    name = f"Species {code}"
                legend_elements.append(mpatches.Patch(color=colors[i], label=f"{code}: {name[:25]}"))

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        output_path = maps_dir / "custom" / "dominant_species_map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"    Saved to {output_path}")

    if conifer_ratio_path.exists():
        logger.info("  Creating conifer dominance map...")
        if conifer_ratio_path.suffix == '.zarr':
            conifer_ratio = zarr.open(conifer_ratio_path, mode='r')[:]
        else:
            import rasterio
            with rasterio.open(conifer_ratio_path) as src:
                conifer_ratio = src.read(1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Mask out non-forest areas
        masked_ratio = np.ma.masked_where(conifer_ratio == 0, conifer_ratio)

        im = ax.imshow(masked_ratio, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("Conifer vs Hardwood Ratio - Lane County", fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Conifer Ratio (0=Hardwood, 1=Conifer)', fontsize=12)

        # Add text annotations
        ax.text(0.02, 0.98, f"Mean Conifer Ratio: {np.mean(conifer_ratio[conifer_ratio > 0]):.1%}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = maps_dir / "custom" / "conifer_dominance_map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"    Saved to {output_path}")

    # Create elevation zone visualization (if we had elevation data)
    # This would show species distribution across elevation gradients


def main():
    # Configuration
    ZARR_PATH = Path("data/lane_county/lane_county.zarr")
    METRICS_DIR = Path("output/lane_county/metrics")
    MAPS_DIR = Path("output/lane_county/maps")

    # Ensure directories exist
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if Zarr store and metrics exist
    if not ZARR_PATH.exists():
        logger.error(f"Zarr store not found at {ZARR_PATH}")
        logger.error("Please run download_lane_county.py first")
        return

    if not METRICS_DIR.exists():
        logger.error(f"Metrics directory not found: {METRICS_DIR}")
        logger.error("Please run calculate_metrics_lane_county.py first")
        return

    # Check for metric files (can be .tif or .zarr)
    metric_files = list(METRICS_DIR.glob("*.tif")) + list(METRICS_DIR.glob("*.zarr"))
    if not metric_files:
        logger.error(f"No metric files found in {METRICS_DIR}")
        logger.error("Please run calculate_metrics_lane_county.py first")
        return

    logger.info(f"Found {len(metric_files)} metric files:")
    for mf in metric_files:
        logger.info(f"  - {mf.name}")

    # Initialize API
    api = BigMapAPI()

    logger.info("="*60)
    logger.info("Lane County Forest Visualization")
    logger.info("="*60)

    # Step 1: Create diversity maps
    logger.info("-"*60)
    logger.info("Creating diversity maps...")

    try:
        diversity_maps = api.create_maps(
            zarr_path=ZARR_PATH,
            map_type="diversity",
            output_dir=MAPS_DIR / "diversity",
            format="png",
            dpi=300
        )
        logger.info(f"  Created {len(diversity_maps)} diversity maps")

    except Exception as e:
        logger.error(f"Diversity map creation failed: {e}")

    # Step 2: Create species richness map
    logger.info("-"*60)
    logger.info("Creating species richness map...")

    try:
        richness_maps = api.create_maps(
            zarr_path=ZARR_PATH,
            map_type="richness",
            output_dir=MAPS_DIR / "richness",
            format="png",
            dpi=300
        )
        logger.info(f"  Created {len(richness_maps)} richness maps")

    except Exception as e:
        logger.error(f"Richness map creation failed: {e}")

    # Step 3: Create species-specific maps for key Oregon species
    logger.info("-"*60)
    logger.info("Creating species-specific maps...")

    # Key Oregon species to visualize
    oregon_species = {
        "0202": "Douglas-fir",
        "0122": "Ponderosa pine",
        "0263": "Western hemlock",
        "0242": "Western redcedar",
        "0017": "Grand fir",
        "0015": "White fir"
    }

    # Check which species are available
    try:
        all_species = api.list_species()
        available_codes = [s.species_code for s in all_species]

        species_to_map = []
        for code, name in oregon_species.items():
            if code in available_codes:
                species_to_map.append(code)
                logger.info(f"  Will map: {code} - {name}")

        if species_to_map:
            # Map top 4 species individually
            for species_code in species_to_map[:4]:
                try:
                    species_maps = api.create_maps(
                        zarr_path=ZARR_PATH,
                        map_type="species",
                        species=[species_code],
                        output_dir=MAPS_DIR / "species",
                        format="png",
                        dpi=300,
                        cmap="YlGn"  # Green colormap for biomass
                    )
                    logger.info(f"    ✓ Created map for species {species_code}")
                except Exception as e:
                    logger.error(f"    ✗ Failed to map species {species_code}: {e}")

    except Exception as e:
        logger.error(f"Species map creation failed: {e}")

    # Step 4: Create comparison maps
    logger.info("-"*60)
    logger.info("Creating species comparison map...")

    try:
        # Compare major conifer species
        comparison_species = ["0202", "0263", "0122"]  # Douglas-fir, W. hemlock, Ponderosa

        comparison_maps = api.create_maps(
            zarr_path=ZARR_PATH,
            map_type="comparison",
            species=comparison_species,
            output_dir=MAPS_DIR / "comparison",
            format="png",
            dpi=300
        )
        logger.info(f"  Created {len(comparison_maps)} comparison maps")

    except Exception as e:
        logger.error(f"Comparison map creation failed: {e}")

    # Step 5: Create custom visualizations
    logger.info("-"*60)
    logger.info("Creating custom Lane County visualizations...")

    try:
        # Only create custom visualizations if custom metrics exist
        custom_metrics_exist = False
        for metric_name in ["dominant_species", "conifer_ratio"]:
            if (METRICS_DIR / f"{metric_name}.zarr").exists() or (METRICS_DIR / f"{metric_name}.tif").exists():
                custom_metrics_exist = True
                break

        if custom_metrics_exist:
            create_custom_visualizations(ZARR_PATH, METRICS_DIR, MAPS_DIR)
        else:
            logger.info("  No custom metrics found - skipping custom visualizations")
            logger.info("  (Run calculate_metrics_lane_county.py to generate custom metrics)")
    except Exception as e:
        logger.error(f"Custom visualization creation failed: {e}")

    # Step 6: Generate visualization summary
    logger.info("-"*60)
    logger.info("Generating visualization summary...")

    summary_file = MAPS_DIR / "visualization_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Lane County Forest Visualizations\n")
        f.write("="*60 + "\n\n")

        f.write("Generated Maps:\n")
        f.write("-"*30 + "\n")

        # Count maps by type
        for map_type in ["diversity", "richness", "species", "comparison", "custom"]:
            type_dir = MAPS_DIR / map_type
            if type_dir.exists():
                map_files = list(type_dir.glob("*.png"))
                if map_files:
                    f.write(f"\n{map_type.title()} Maps ({len(map_files)} files):\n")
                    for map_file in map_files[:5]:  # Show first 5
                        f.write(f"  - {map_file.name}\n")
                    if len(map_files) > 5:
                        f.write(f"  ... and {len(map_files) - 5} more\n")

        f.write("\n\nVisualization Features:\n")
        f.write("-"*30 + "\n")
        f.write("• Diversity maps: Shannon and Simpson indices\n")
        f.write("• Richness maps: Species count per pixel\n")
        f.write("• Species maps: Individual species biomass distribution\n")
        f.write("• Comparison maps: Multi-species overlay\n")
        f.write("• Custom maps: Dominant species, conifer ratio\n")

    logger.info(f"Summary saved to: {summary_file}")

    logger.info("="*60)
    logger.info("Visualization complete!")
    logger.info(f"All maps saved to: {MAPS_DIR}")

    # List total files created
    all_maps = list(MAPS_DIR.rglob("*.png"))
    logger.info(f"Total visualizations created: {len(all_maps)}")


if __name__ == "__main__":
    main()