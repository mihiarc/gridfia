#!/usr/bin/env python3
"""
GridFIA Quickstart Example

The simplest possible example to get started with GridFIA.
Downloads data for one county, creates a zarr store, and calculates species richness.

Takes about 2 minutes to run.
"""

from pathlib import Path
from gridfia import GridFIA
from gridfia.examples import print_zarr_info, calculate_basic_stats
from examples.common_locations import get_location_bbox


def main():
    print("=" * 60)
    print("GridFIA Quickstart - Wake County, NC")
    print("=" * 60)

    # Initialize API
    api = GridFIA()

    # 1. Download species data (just 2 species for speed)
    print("\n1. Downloading forest data...")
    print("   Location: Wake County, NC")

    # Get predefined bounding box for Wake County
    bbox, crs = get_location_bbox("wake_nc")

    files = api.download_species(
        bbox=bbox,
        crs=crs,
        species_codes=["0131", "0068"],  # Loblolly Pine, Red Maple
        output_dir="quickstart_data"
    )
    print(f"   Downloaded {len(files)} species files")

    # 2. Create Zarr store
    print("\n2. Creating Zarr store...")
    zarr_path = api.create_zarr(
        input_dir="quickstart_data",
        output_path="quickstart_data/wake_forest.zarr"
    )
    print_zarr_info(Path(zarr_path))

    # 3. Calculate species richness
    print("\n3. Calculating species richness...")
    results = api.calculate_metrics(
        zarr_path=zarr_path,
        calculations=["species_richness"],
        output_dir="quickstart_results"
    )

    # 4. Print basic statistics
    print("\n4. Forest Statistics:")
    stats = calculate_basic_stats(Path(zarr_path), sample_size=None)
    print(f"   Forest coverage: {stats['forest_coverage_pct']:.1f}%")
    print(f"   Mean biomass: {stats['mean_biomass']:.1f} Mg/ha")
    print(f"   Total biomass: {stats['total_biomass_mg']/1e6:.2f} million Mg")

    print("\n✅ Quickstart complete!")
    print(f"   Results saved to: quickstart_results/")
    print("\nNext steps:")
    print("  - Run 02_api_overview.py to see all API features")
    print("  - Run 06_wake_county_full.py for complete analysis")


if __name__ == "__main__":
    main()