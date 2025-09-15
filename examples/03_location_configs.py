#!/usr/bin/env python3
"""
Location Configuration Examples

Demonstrates how to work with different geographic locations:
- States
- Counties
- Custom bounding boxes
- Batch processing multiple locations
"""

from pathlib import Path
from bigmap.utils.location_config import LocationConfig
from bigmap.visualization.boundaries import STATE_ABBR
from rich.console import Console
from rich.table import Table

console = Console()


def create_state_configs():
    """Create configurations for multiple states."""
    console.print("\n[bold blue]State Configurations[/bold blue]")
    console.print("-" * 40)

    states = ['North Carolina', 'Texas', 'California', 'Montana', 'Georgia']
    configs = []

    table = Table(title="State Configurations")
    table.add_column("State", style="cyan")
    table.add_column("CRS", style="yellow")
    table.add_column("Bbox (Web Mercator)", style="green")

    for state in states:
        try:
            config = LocationConfig.from_state(state)
            configs.append(config)

            # Save configuration
            output_path = Path(f"configs/{state.lower().replace(' ', '_')}.yaml")
            output_path.parent.mkdir(exist_ok=True)
            config.save(output_path)

            # Add to table
            bbox = config.web_mercator_bbox
            bbox_str = f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            table.add_row(state, config.target_crs, bbox_str)

        except Exception as e:
            console.print(f"[red]Failed for {state}: {e}[/red]")

    console.print(table)
    return configs


def create_county_configs():
    """Create configurations for specific counties."""
    console.print("\n[bold blue]County Configurations[/bold blue]")
    console.print("-" * 40)

    counties = [
        ('Wake', 'North Carolina'),
        ('Harris', 'Texas'),
        ('Los Angeles', 'California'),
        ('Cook', 'Illinois'),
        ('King', 'Washington')
    ]

    table = Table(title="County Configurations")
    table.add_column("County", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("CRS", style="green")
    table.add_column("Status", style="magenta")

    configs = []
    for county, state in counties:
        try:
            # Create configuration using public method
            config = LocationConfig.from_county(county, state)
            configs.append(config)

            # Save
            filename = f"{county.lower()}_{state.lower().replace(' ', '_')}.yaml"
            output_path = Path(f"configs/counties/{filename}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            config.save(output_path)

            table.add_row(county, state, config.target_crs, "✅ Created")

        except Exception as e:
            table.add_row(county, state, "N/A", f"❌ {str(e)[:20]}")

    console.print(table)
    return configs


def create_custom_bbox_configs():
    """Create configurations for custom bounding boxes."""
    console.print("\n[bold blue]Custom Bounding Box Configurations[/bold blue]")
    console.print("-" * 40)

    custom_areas = [
        {
            "name": "Yellowstone Region",
            "bbox": (-111.2, 44.0, -109.8, 45.2),
            "crs": "EPSG:4326"
        },
        {
            "name": "Great Smoky Mountains",
            "bbox": (-84.0, 35.4, -83.0, 36.0),
            "crs": "EPSG:4326"
        },
        {
            "name": "Olympic Peninsula",
            "bbox": (-125.0, 47.5, -123.0, 48.5),
            "crs": "EPSG:4326"
        }
    ]

    configs = []
    for area in custom_areas:
        config = LocationConfig.from_bbox(
            bbox=area["bbox"],
            name=area["name"]
        )

        # Save
        filename = area["name"].lower().replace(' ', '_') + ".yaml"
        output_path = Path(f"configs/custom/{filename}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.save(output_path)

        configs.append(config)
        console.print(f"✅ Created config for {area['name']}")
        console.print(f"   Bbox: {area['bbox']}")

    return configs


def batch_process_locations():
    """Example of batch processing multiple locations."""
    console.print("\n[bold blue]Batch Processing Example[/bold blue]")
    console.print("-" * 40)

    # Define batch of locations
    batch = [
        {"type": "state", "name": "Vermont"},
        {"type": "county", "name": "Orange", "state": "California"},
        {"type": "county", "name": "Durham", "state": "North Carolina"},
        {"type": "custom", "name": "Mt. Hood", "bbox": (-122.0, 45.2, -121.4, 45.6)}
    ]

    console.print(f"Processing {len(batch)} locations:")

    for loc in batch:
        console.print(f"\n  {loc['name']}:")

        if loc["type"] == "state":
            config = LocationConfig.from_state(loc["name"])
        elif loc["type"] == "county":
            config = LocationConfig.from_county(
                county=loc['name'],
                state=loc['state']
            )
        else:  # custom
            config = LocationConfig.from_bbox(
                bbox=loc["bbox"],
                name=loc["name"]
            )

        console.print(f"    Type: {config.location_type}")
        console.print(f"    CRS: {config.target_crs}")

        # Here you would typically:
        # 1. Download species data using the bbox
        # 2. Create zarr store
        # 3. Run calculations
        # 4. Generate visualizations


def show_location_usage():
    """Show how to use location configs with the API."""
    console.print("\n[bold blue]Using Location Configurations[/bold blue]")
    console.print("-" * 40)

    console.print("\n[yellow]Python API Usage:[/yellow]")
    console.print("""
    from bigmap import BigMapAPI
    from bigmap.utils.location_config import LocationConfig

    # Load saved configuration
    config = LocationConfig("configs/wake_north_carolina.yaml")

    # Use with API
    api = BigMapAPI()

    # Download using config bounds
    files = api.download_species(
        bbox=config.web_mercator_bbox,
        species_codes=['0131', '0068'],
        output_dir="data/wake"
    )

    # Or use the high-level method
    api.download_from_config(
        config=config,
        species_codes=['0131'],
        output_dir="data"
    )
    """)

    console.print("\n[yellow]Command Line Usage:[/yellow]")
    console.print("""
    # Download using location config
    bigmap download --location-config configs/montana.yaml --species 0202

    # Or specify directly
    bigmap download --state Texas --county Harris --species 0131
    """)


def main():
    """Run all location configuration examples."""
    console.print("[bold green]Location Configuration Examples[/bold green]")
    console.print("=" * 60)

    # Create different types of configs
    state_configs = create_state_configs()
    console.print(f"\n✅ Created {len(state_configs)} state configurations")

    county_configs = create_county_configs()
    console.print(f"✅ Created {len(county_configs)} county configurations")

    custom_configs = create_custom_bbox_configs()
    console.print(f"✅ Created {len(custom_configs)} custom area configurations")

    # Show batch processing
    batch_process_locations()

    # Show usage examples
    show_location_usage()

    console.print("\n" + "=" * 60)
    console.print("[bold green]Location Configuration Complete![/bold green]")
    console.print("\nConfiguration files saved to:")
    console.print("  - configs/           (states)")
    console.print("  - configs/counties/  (counties)")
    console.print("  - configs/custom/    (custom areas)")


if __name__ == "__main__":
    main()