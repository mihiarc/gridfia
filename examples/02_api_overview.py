#!/usr/bin/env python3
"""
BigMap API Overview

Demonstrates all major API features and patterns.
Each example is self-contained and can be run independently.
"""

from pathlib import Path
from bigmap import BigMapAPI
from bigmap.config import BigMapSettings, CalculationConfig
from bigmap.examples import create_sample_zarr, print_zarr_info


def example_1_list_species():
    """List all available species in the BIGMAP dataset."""
    print("\n" + "=" * 60)
    print("Example 1: List Available Species")
    print("=" * 60)

    api = BigMapAPI()
    species = api.list_species()

    print(f"Total species available: {len(species)}")
    print("\nFirst 5 species:")
    for s in species[:5]:
        print(f"  {s.species_code}: {s.common_name} ({s.scientific_name})")

    # Find specific species
    pine_species = [s for s in species if "pine" in s.common_name.lower()]
    print(f"\nFound {len(pine_species)} pine species")


def example_2_location_config():
    """Get location configurations for different geographic areas."""
    print("\n" + "=" * 60)
    print("Example 2: Location Configurations")
    print("=" * 60)

    api = BigMapAPI()

    # State configuration
    state_config = api.get_location_config(state="Montana")
    print(f"State: {state_config.location_name}")
    print(f"  CRS: {state_config.target_crs}")
    print(f"  Bbox: {state_config.web_mercator_bbox}")

    # County configuration
    county_config = api.get_location_config(
        state="Texas",
        county="Harris"
    )
    print(f"\nCounty: {county_config.location_name}")
    print(f"  Type: {county_config.location_type}")

    # Custom bounding box
    custom_config = api.get_location_config(
        bbox=(-104.5, 44.0, -104.0, 44.5),
        crs="EPSG:4326"
    )
    print(f"\nCustom area created with bbox")


def example_3_download_patterns():
    """Different patterns for downloading species data."""
    print("\n" + "=" * 60)
    print("Example 3: Download Patterns")
    print("=" * 60)

    api = BigMapAPI()

    # Note: These are examples - uncomment to actually download
    print("Download patterns (not executed):")

    print("\n1. Single species, single county:")
    print('   api.download_species(state="NC", county="Wake", species_codes=["0131"])')

    print("\n2. Multiple species, state-level:")
    print('   api.download_species(state="Montana", species_codes=["0202", "0122"])')

    print("\n3. All species for small area:")
    print('   api.download_species(state="RI", species_codes="all")')

    print("\n4. Using location config:")
    print('   config = api.get_location_config(state="VT")')
    print('   api.download_from_config(config, species_codes=["0068"])')


def example_4_zarr_operations():
    """Working with Zarr stores."""
    print("\n" + "=" * 60)
    print("Example 4: Zarr Operations")
    print("=" * 60)

    api = BigMapAPI()

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
    api = BigMapAPI()
    results = api.calculate_metrics(
        zarr_path=sample_path,
        calculations=["species_richness", "shannon_diversity"]
    )
    print(f"Simple: Calculated {len(results)} metrics")

    # Method 2: Custom configuration
    settings = BigMapSettings(
        output_dir=Path("custom_output"),
        calculations=[
            CalculationConfig(
                name="species_richness",
                parameters={"biomass_threshold": 2.0},
                output_format="geotiff"
            ),
            CalculationConfig(
                name="total_biomass",
                output_format="netcdf"
            )
        ]
    )
    api_custom = BigMapAPI(config=settings)
    results = api_custom.calculate_metrics(zarr_path=sample_path)
    print(f"Custom: Calculated {len(results)} metrics with custom settings")

    # Clean up
    import shutil
    shutil.rmtree(sample_path)
    if Path("custom_output").exists():
        shutil.rmtree("custom_output")


def example_6_visualization():
    """Creating visualizations."""
    print("\n" + "=" * 60)
    print("Example 6: Visualization Options")
    print("=" * 60)

    # Create sample data
    sample_path = create_sample_zarr(Path("temp_sample.zarr"))
    api = BigMapAPI()

    # Different map types
    map_types = ["diversity", "species", "richness", "comparison"]

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
        print(f"{map_type}: Created {len(maps)} maps")

    # Clean up
    import shutil
    shutil.rmtree(sample_path)
    for map_type in map_types:
        output_dir = Path(f"maps_{map_type}")
        if output_dir.exists():
            shutil.rmtree(output_dir)


def example_7_batch_processing():
    """Batch processing multiple locations."""
    print("\n" + "=" * 60)
    print("Example 7: Batch Processing")
    print("=" * 60)

    api = BigMapAPI()

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
    print("BigMap API Overview")
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