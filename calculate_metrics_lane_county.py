#!/usr/bin/env python3
"""
Calculate forest metrics for Lane County.

This script:
1. Loads the Lane County Zarr store
2. Calculates diversity indices (Shannon, Simpson)
3. Calculates species richness
4. Calculates total biomass
5. Saves metric arrays for visualization
"""

from pathlib import Path
import logging
from bigmap import BigMapAPI
import numpy as np
import zarr
import xarray as xr
from datetime import datetime

# Set matplotlib to use non-interactive backend if imported by any module
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_custom_metrics(zarr_path: Path, output_dir: Path):
    """
    Calculate additional custom metrics beyond the API defaults.

    These metrics provide deeper insights into Lane County's forest composition.
    """
    logger.info("Calculating custom metrics...")

    # Open Zarr store
    store = zarr.open(zarr_path, mode='r')
    biomass = store['biomass']
    species_codes = store['species_codes'][:]

    # Define key Oregon species indices
    oregon_species = {
        '0202': 'Douglas-fir',
        '0263': 'Western hemlock',
        '0242': 'Western redcedar',
        '0017': 'Grand fir',
        '0122': 'Ponderosa pine'
    }

    # Find indices for key species
    species_indices = {}
    for i, code in enumerate(species_codes):
        code_str = code.decode() if isinstance(code, bytes) else code
        if code_str in oregon_species:
            species_indices[code_str] = i

    # Calculate dominant species map
    logger.info("  Computing dominant species map...")
    # Get the species with maximum biomass at each pixel
    dominant_species = np.argmax(biomass[:], axis=0)
    dominant_biomass = np.max(biomass[:], axis=0)

    # Save dominant species map
    dominant_path = output_dir / "dominant_species.zarr"
    zarr.save_array(dominant_path, dominant_species, chunks=(1000, 1000))
    logger.info(f"  Saved dominant species map to {dominant_path}")

    # Calculate conifer vs hardwood ratio
    logger.info("  Computing conifer dominance...")
    # Douglas-fir + Western hemlock + Grand fir + Western redcedar
    conifer_biomass = np.zeros(biomass.shape[1:], dtype=np.float32)
    for code in ['0202', '0263', '0242', '0017', '0081', '0015', '0122', '0117', '0119']:
        if code in species_indices:
            conifer_biomass += biomass[species_indices[code], :, :]

    total_biomass = biomass[0, :, :]  # Index 0 is total biomass
    conifer_ratio = np.where(total_biomass > 0, conifer_biomass / total_biomass, 0)

    # Save conifer ratio
    conifer_path = output_dir / "conifer_ratio.zarr"
    zarr.save_array(conifer_path, conifer_ratio, chunks=(1000, 1000))
    logger.info(f"  Saved conifer ratio to {conifer_path}")

    # Calculate elevation gradient metrics (if we had elevation data)
    # This would show species distribution by elevation zones

    # Generate summary statistics
    stats = {
        'dominant_species_diversity': len(np.unique(dominant_species[dominant_biomass > 0])),
        'mean_conifer_ratio': float(np.mean(conifer_ratio[total_biomass > 0])),
        'douglas_fir_dominance': float(np.sum(dominant_species == species_indices.get('0202', -1)) / np.sum(dominant_biomass > 0))
    }

    return stats


def main():
    # Configuration
    ZARR_PATH = Path("data/lane_county/lane_county.zarr")
    METRICS_DIR = Path("output/lane_county/metrics")

    # Ensure output directory exists
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if Zarr store exists
    if not ZARR_PATH.exists():
        logger.error(f"Zarr store not found at {ZARR_PATH}")
        logger.error("Please run download_lane_county.py first to create the Zarr store")
        return

    # Initialize API
    api = BigMapAPI()

    # Validate Zarr store
    logger.info("="*60)
    logger.info("Lane County Forest Metrics Calculation")
    logger.info("="*60)

    try:
        zarr_info = api.validate_zarr(ZARR_PATH)
        logger.info(f"Zarr store validated:")
        logger.info(f"  - Shape: {zarr_info['shape']}")
        logger.info(f"  - Species count: {zarr_info['num_species']}")
    except Exception as e:
        logger.error(f"Zarr validation failed: {e}")
        raise

    # Step 1: Calculate standard forest metrics
    logger.info("-"*60)
    logger.info("Calculating standard forest metrics...")

    calculations = [
        "species_richness",     # Number of species per pixel
        "shannon_diversity",    # Shannon diversity index
        "simpson_diversity",    # Simpson diversity index
        "total_biomass",       # Total biomass across all species
    ]

    try:
        results = api.calculate_metrics(
            zarr_path=ZARR_PATH,
            calculations=calculations,
            output_dir=METRICS_DIR
        )

        logger.info(f"Successfully calculated {len(results)} standard metrics:")
        for result in results:
            logger.info(f"  ✓ {result.name}: {result.output_path}")

    except Exception as e:
        logger.error(f"Metric calculation failed: {e}")
        raise

    # Step 2: Calculate custom Lane County metrics
    logger.info("-"*60)
    logger.info("Calculating custom Lane County metrics...")

    try:
        custom_stats = calculate_custom_metrics(ZARR_PATH, METRICS_DIR)

        logger.info("Custom metrics computed:")
        logger.info(f"  - Dominant species diversity: {custom_stats['dominant_species_diversity']} species")
        logger.info(f"  - Mean conifer ratio: {custom_stats['mean_conifer_ratio']:.1%}")
        logger.info(f"  - Douglas-fir dominance: {custom_stats['douglas_fir_dominance']:.1%} of forested pixels")

    except Exception as e:
        logger.error(f"Custom metric calculation failed: {e}")
        raise

    # Step 3: Generate metrics summary report
    logger.info("-"*60)
    logger.info("Generating metrics summary report...")

    summary_file = METRICS_DIR / "metrics_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Lane County Forest Metrics Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

        f.write("Data Source:\n")
        f.write(f"  Zarr store: {ZARR_PATH}\n")
        f.write(f"  Shape: {zarr_info['shape']}\n")
        f.write(f"  Number of species: {zarr_info['num_species']}\n\n")

        f.write("Standard Metrics Calculated:\n")
        for result in results:
            f.write(f"  - {result.name}\n")
            f.write(f"    Output: {result.output_path}\n")

        f.write("\nCustom Metrics:\n")
        f.write(f"  - Dominant species diversity: {custom_stats['dominant_species_diversity']} species\n")
        f.write(f"  - Mean conifer ratio: {custom_stats['mean_conifer_ratio']:.1%}\n")
        f.write(f"  - Douglas-fir dominance: {custom_stats['douglas_fir_dominance']:.1%}\n")

        f.write("\nMetric Files:\n")
        for metric_file in METRICS_DIR.glob("*.zarr"):
            f.write(f"  - {metric_file.name}\n")

    logger.info(f"Summary saved to: {summary_file}")

    # Step 4: Create a quick statistical summary
    logger.info("-"*60)
    logger.info("Computing statistical summary...")

    # Load and analyze key metrics
    store = zarr.open(ZARR_PATH, mode='r')
    biomass = store['biomass']
    total_biomass = biomass[0, :, :]

    # Forest coverage statistics
    forested_pixels = np.sum(total_biomass > 0)
    total_pixels = total_biomass.size
    forest_coverage = forested_pixels / total_pixels

    # Biomass statistics
    biomass_values = total_biomass[total_biomass > 0]

    logger.info("Forest Statistics:")
    logger.info(f"  - Forest coverage: {forest_coverage:.1%} of county")
    logger.info(f"  - Forested pixels: {forested_pixels:,} / {total_pixels:,}")
    logger.info(f"  - Mean biomass: {biomass_values.mean():.1f} tons/acre")
    logger.info(f"  - Median biomass: {np.median(biomass_values):.1f} tons/acre")
    logger.info(f"  - Biomass range: {biomass_values.min():.1f} - {biomass_values.max():.1f} tons/acre")

    logger.info("="*60)
    logger.info("Metrics calculation complete!")
    logger.info(f"All metrics saved to: {METRICS_DIR}")
    logger.info("Ready for visualization with visualize_lane_county.py")


if __name__ == "__main__":
    main()