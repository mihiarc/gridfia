#!/usr/bin/env python3
"""
GridFIA API Overview

Demonstrates all major API features and patterns.
Each example is self-contained and can be run independently.
"""

from pathlib import Path
from gridfia import GridFIA
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.examples import create_sample_zarr, print_zarr_info
from examples.common_locations import get_location_bbox, COUNTIES, STATES


def example_1_list_species():
    """List all available species in the BIGMAP dataset."""
    print("\n" + "=" * 60)
    print("Example 1: List Available Species")
    print("=" * 60)

    api = GridFIA()
    species = api.list_species()

    print(f"Total species available: {len(species)}")
    print("\nFirst 5 species:")
    for s in species[:5]:
        print(f"  {s.species_code}: {s.common_name} ({s.scientific_name})")

    # Find specific species
    pine_species = [s for s in species if "pine" in s.common_name.lower()]
    print(f"\nFound {len(pine_species)} pine species")


def example_2_location_config():
    """Demonstrate using predefined location bounding boxes."""
    print("\n" + "=" * 60)
    print("Example 2: Location Configurations")
    print("=" * 60)

    # Using predefined bounding boxes to avoid external dependencies
    print("Using predefined bounding boxes (no external downloads required)")

    # County example - using predefined bbox
    harris_bbox, harris_crs = get_location_bbox("harris_tx")
    print(f"\nCounty: Harris County, Texas")
    print(f"  Bbox: {harris_bbox}")
    print(f"  CRS: {harris_crs}")
    if "harris_tx" in COUNTIES:
        print(f"  Description: {COUNTIES['harris_tx']['description']}")

    # Another county example
    wake_bbox, wake_crs = get_location_bbox("wake_nc")
    print(f"\nCounty: Wake County, North Carolina")
    print(f"  Bbox: {wake_bbox}")
    print(f"  CRS: {wake_crs}")

    # Custom bounding box - no external dependencies needed
    custom_bbox = (-104.5, 44.0, -104.0, 44.5)
    print(f"\nCustom area:")
    print(f"  Bbox (WGS84): {custom_bbox}")
    print(f"  CRS: 4326")
    print("  Note: Custom bboxes work directly without boundary downloads")


def example_3_download_patterns():
    """Different patterns for downloading species data."""
    print("\n" + "=" * 60)
    print("Example 3: Download Patterns")
    print("=" * 60)

    api = GridFIA()

    # Note: These are examples - uncomment to actually download
    print("Download patterns using bounding boxes (not executed):")

    print("\n1. Single species, single county (using predefined bbox):")
    print('   bbox, crs = get_location_bbox("wake_nc")')
    print('   api.download_species(bbox=bbox, crs=crs, species_codes=["0131"])')

    print("\n2. Multiple species for a location:")
    print('   bbox, crs = get_location_bbox("harris_tx")')
    print('   api.download_species(bbox=bbox, crs=crs, species_codes=["0202", "0122"])')

    print("\n3. Custom bounding box:")
    print('   api.download_species(bbox=(-104.5, 44.0, -104.0, 44.5), crs="4326")')

    print("\n4. Using small test area:")
    print('   bbox, crs = get_location_bbox("raleigh_downtown")  # Small area for testing')
    print('   api.download_species(bbox=bbox, crs=crs, species_codes=["0068"])')


def example_4_zarr_operations():
    """Working with Zarr stores."""
    print("\n" + "=" * 60)
    print("Example 4: Zarr Operations")
    print("=" * 60)

    api = GridFIA()

    # Create sample data for demonstration
    sample_path = create_sample_zarr(Path("temp_sample.zarr"))

    # Validate zarr
    info = api.validate_zarr(sample_path)
    print(f"Zarr validation:")
    print(f"  Valid: {info.get('valid', False)}")
    print(f"  Shape: {info['shape']}")
    print(f"  Species: {info['num_species']}")

    # Get detailed info
    print_zarr_info(sample_path)

    # Clean up
    import shutil
    shutil.rmtree(sample_path)


def example_5_calculations():
    """Different calculation configurations."""
    print("\n" + "=" * 60)
    print("Example 5: Calculation Patterns")
    print("=" * 60)

    # Create sample data
    sample_path = create_sample_zarr(Path("temp_sample.zarr"))

    # Method 1: Simple calculation list
    api = GridFIA()
    results = api.calculate_metrics(
        zarr_path=sample_path,
        calculations=["species_richness", "shannon_diversity"]
    )
    print(f"Simple: Calculated {len(results)} metrics")

    # Method 2: Custom configuration
    settings = GridFIASettings(
        output_dir=Path("custom_output"),
        calculations=[
            CalculationConfig(
                name="species_richness",
                parameters={"biomass_threshold": 2.0},
                output_format="geotiff"
            ),
            CalculationConfig(
                name="total_biomass",
                output_format="geotiff"  # Changed from netcdf to geotiff
            )
        ]
    )
    api_custom = GridFIA(config=settings)
    results = api_custom.calculate_metrics(zarr_path=sample_path)
    print(f"Custom: Calculated {len(results)} metrics with custom settings")

    # Clean up
    import shutil
    shutil.rmtree(sample_path)
    if Path("custom_output").exists():
        shutil.rmtree("custom_output")


def example_6_visualization():
    """Creating visualizations (demonstration with sample data)."""
    print("\n" + "=" * 60)
    print("Example 6: Visualization Options (Sample Data Demo)")
    print("=" * 60)

    print("Note: This example uses synthetic data to demonstrate the API.")
    print("For real forest visualizations, see examples 01 or 06 which use")
    print("actual BIGMAP data downloads.\n")

    # Create sample data for demonstration
    sample_path = create_sample_zarr(Path("temp_sample.zarr"))
    api = GridFIA()

    # Different map types
    map_types = ["diversity", "species", "richness", "comparison"]

    print("Demonstrating visualization API with sample data:")
    for map_type in map_types:
        if map_type == "species":
            maps = api.create_maps(
                zarr_path=sample_path,
                map_type=map_type,
                output_dir=f"maps_{map_type}",
                show_all=True
            )
        elif map_type == "comparison":
            maps = api.create_maps(
                zarr_path=sample_path,
                map_type=map_type,
                output_dir=f"maps_{map_type}",
                species=["0001", "0002"]  # Compare first two species
            )
        else:
            maps = api.create_maps(
                zarr_path=sample_path,
                map_type=map_type,
                output_dir=f"maps_{map_type}"
            )
        print(f"  {map_type}: Created {len(maps)} maps (sample data)")

    # Clean up - remove sample visualizations as they're not real data
    import shutil
    shutil.rmtree(sample_path)
    for map_type in map_types:
        output_dir = Path(f"maps_{map_type}")
        if output_dir.exists():
            shutil.rmtree(output_dir)

    print("\n💡 To create visualizations with real forest data:")
    print("   Run examples/01_quickstart.py or examples/06_wake_county_full.py")


def example_7_batch_processing():
    """Batch processing multiple locations."""
    print("\n" + "=" * 60)
    print("Example 7: Batch Processing")
    print("=" * 60)

    api = GridFIA()

    locations = [
        {"state": "North Carolina", "counties": ["Wake", "Durham"]},
        {"state": "Montana", "counties": ["Missoula"]},
    ]

    print("Batch processing pattern:")
    for location in locations:
        state = location["state"]
        for county in location["counties"]:
            print(f"\n  Processing {county} County, {state}:")
            print(f"    1. Download species data")
            print(f"    2. Create zarr store")
            print(f"    3. Calculate metrics")
            print(f"    4. Generate visualizations")


def main():
    """Run all API examples."""
    print("\n" + "🌲" * 30)
    print("GridFIA API Overview")
    print("Complete API Feature Demonstration")
    print("🌲" * 30)

    # Run examples
    example_1_list_species()
    example_2_location_config()
    example_3_download_patterns()
    example_4_zarr_operations()
    example_5_calculations()
    example_6_visualization()
    example_7_batch_processing()

    print("\n" + "=" * 60)
    print("API Overview Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run specific examples for detailed workflows")
    print("  - See examples/README.md for full documentation")
    print("  - Check docs/tutorials/ for step-by-step guides")


if __name__ == "__main__":
    main()