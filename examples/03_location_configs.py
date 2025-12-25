#!/usr/bin/env python3
"""
Location Configuration Examples

Demonstrates how to work with different geographic locations:
- States (using predefined bounding boxes)
- Counties (using predefined bounding boxes)
- Custom bounding boxes
- Batch processing multiple locations

Note: This example uses predefined bounding boxes to avoid downloading
external boundary files, making it more reliable and faster to run.
"""

from pathlib import Path
from gridfia.utils.location_config import LocationConfig
from rich.console import Console
from rich.table import Table
import warnings

# Suppress boundary download warnings since we'll use predefined boxes
warnings.filterwarnings('ignore', message='.*boundaries.*')

console = Console()

# Predefined bounding boxes for common locations (WGS84)
# These avoid the need to download boundary files
STATE_BBOXES = {
    'North Carolina': (-84.32, 33.84, -75.46, 36.59),
    'Texas': (-106.65, 25.84, -93.51, 36.50),
    'California': (-124.48, 32.53, -114.13, 42.01),
    'Montana': (-116.05, 44.36, -104.04, 49.00),
    'Georgia': (-85.61, 30.36, -80.84, 35.00),
    'Vermont': (-73.44, 42.73, -71.46, 45.02),
}

COUNTY_BBOXES = {
    ('Wake', 'North Carolina'): (-78.97, 35.57, -78.25, 36.08),
    ('Harris', 'Texas'): (-95.91, 29.52, -95.01, 30.17),
    ('Los Angeles', 'California'): (-118.95, 33.70, -117.65, 34.82),
    ('Cook', 'Illinois'): (-88.26, 41.47, -87.52, 42.15),
    ('King', 'Washington'): (-122.54, 47.08, -121.06, 47.78),
    ('Orange', 'California'): (-118.15, 33.38, -117.41, 33.95),
    ('Durham', 'North Carolina'): (-79.11, 35.87, -78.67, 36.24),
}

# State Plane CRS codes for states
STATE_CRS = {
    'North Carolina': 'EPSG:32119',  # NAD83 / North Carolina
    'Texas': 'EPSG:32139',  # NAD83 / Texas Central
    'California': 'EPSG:32610',  # WGS 84 / UTM zone 10N
    'Montana': 'EPSG:32100',  # NAD83 / Montana
    'Georgia': 'EPSG:32616',  # WGS 84 / UTM zone 16N
    'Vermont': 'EPSG:32145',  # NAD83 / Vermont
}


def create_state_configs():
    """Create configurations for multiple states using predefined bounding boxes."""
    console.print("\n[bold blue]State Configurations (Using Predefined Bounding Boxes)[/bold blue]")
    console.print("-" * 40)

    configs = []

    table = Table(title="State Configurations")
    table.add_column("State", style="cyan")
    table.add_column("CRS", style="yellow")
    table.add_column("Bbox (WGS84)", style="green")
    table.add_column("Status", style="magenta")

    for state, bbox in STATE_BBOXES.items():
        try:
            # Create configuration using predefined bbox
            config = LocationConfig.from_bbox(
                bbox=bbox,
                name=state
            )

            # Set the target CRS
            config._config['crs']['target'] = STATE_CRS.get(state, 'EPSG:3857')
            config._config['location']['type'] = 'state'
            config._config['location']['name'] = state
            configs.append(config)

            # Save configuration
            output_path = Path(f"configs/{state.lower().replace(' ', '_')}.yaml")
            output_path.parent.mkdir(exist_ok=True)
            config.save(output_path)

            # Add to table with formatted bbox
            bbox_str = f"({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})"
            table.add_row(state, STATE_CRS.get(state, 'EPSG:3857'), bbox_str, "✅ Created")

        except Exception as e:
            table.add_row(state, "N/A", "N/A", f"❌ {str(e)[:20]}")

    console.print(table)
    return configs


def create_county_configs():
    """Create configurations for specific counties using predefined bounding boxes."""
    console.print("\n[bold blue]County Configurations (Using Predefined Bounding Boxes)[/bold blue]")
    console.print("-" * 40)

    table = Table(title="County Configurations")
    table.add_column("County", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("Bbox (WGS84)", style="green")
    table.add_column("Status", style="magenta")

    configs = []
    for (county, state), bbox in COUNTY_BBOXES.items():
        if (county, state) not in [('Orange', 'California'), ('Durham', 'North Carolina')]:  # Skip extras for demo
            try:
                # Create configuration using predefined bbox
                config = LocationConfig.from_bbox(
                    bbox=bbox,
                    name=f"{county}, {state}"
                )

                # Set metadata
                config._config['location']['type'] = 'county'
                config._config['location']['name'] = county
                config._config['location']['state'] = state
                config._config['crs']['target'] = STATE_CRS.get(state, 'EPSG:3857')
                configs.append(config)

                # Save
                filename = f"{county.lower()}_{state.lower().replace(' ', '_')}.yaml"
                output_path = Path(f"configs/counties/{filename}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                config.save(output_path)

                bbox_str = f"({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})"
                table.add_row(county, state, bbox_str, "✅ Created")

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
        {"type": "state", "name": "Vermont", "bbox": STATE_BBOXES.get('Vermont')},
        {"type": "county", "name": "Orange", "state": "California",
         "bbox": COUNTY_BBOXES.get(('Orange', 'California'))},
        {"type": "county", "name": "Durham", "state": "North Carolina",
         "bbox": COUNTY_BBOXES.get(('Durham', 'North Carolina'))},
        {"type": "custom", "name": "Mt. Hood", "bbox": (-122.0, 45.2, -121.4, 45.6)}
    ]

    console.print(f"Processing {len(batch)} locations:")

    for loc in batch:
        console.print(f"\n  {loc['name']}:")

        # All locations now use bbox-based configuration
        config = LocationConfig.from_bbox(
            bbox=loc["bbox"],
            name=loc["name"] if loc["type"] == "custom" else
                 loc["name"] if loc["type"] == "state" else
                 f"{loc['name']}, {loc['state']}"
        )

        # Set appropriate metadata
        config._config['location']['type'] = loc["type"]
        if loc["type"] == "county":
            config._config['location']['state'] = loc['state']
            config._config['crs']['target'] = STATE_CRS.get(loc['state'], 'EPSG:3857')
        elif loc["type"] == "state":
            config._config['crs']['target'] = STATE_CRS.get(loc['name'], 'EPSG:3857')

        console.print(f"    Type: {loc['type']}")
        console.print(f"    Bbox: {loc['bbox']}")
        console.print(f"    CRS: {config._config['crs'].get('target', 'EPSG:3857')}")

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
    from gridfia import GridFIA
    from gridfia.utils.location_config import LocationConfig

    # Method 1: Load saved configuration
    config = LocationConfig("configs/wake_county.yaml")

    # Method 2: Create configuration from predefined bbox
    config = LocationConfig.from_bbox(
        bbox=(-78.97, 35.57, -78.25, 36.08),  # Wake County, NC
        name="Wake County, NC"
    )

    # Use with API
    api = GridFIA()

    # Download using config bounds
    files = api.download_species(
        bbox=config.wgs84_bbox,  # or web_mercator_bbox
        species_codes=['0131', '0068'],
        output_dir="data/wake"
    )

    # Process the downloaded data
    zarr_path = api.create_zarr("data/wake", "wake.zarr")
    results = api.calculate_metrics(zarr_path)
    """)

    console.print("\n[yellow]Creating Custom Location Configs:[/yellow]")
    console.print("""
    # For any custom area - just provide the bounding box!
    config = LocationConfig.from_bbox(
        bbox=(-122.0, 45.2, -121.4, 45.6),  # Mt. Hood area
        name="Mt. Hood Region"
    )

    # Save for later use
    config.save("configs/mt_hood.yaml")
    """)


def main():
    """Run all location configuration examples."""
    console.print("[bold green]Location Configuration Examples[/bold green]")
    console.print("=" * 60)

    console.print("\n[yellow]Note:[/yellow] This example uses predefined bounding boxes")
    console.print("to avoid downloading external boundary files. The same")
    console.print("approach works for any location - just provide the bbox!\n")

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
    console.print("\n[cyan]Tip:[/cyan] You can find bounding boxes for any location at:")
    console.print("  - https://boundingbox.klokantech.com/")
    console.print("  - https://www.openstreetmap.org/ (export feature)")


if __name__ == "__main__":
    main()