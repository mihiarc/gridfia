#!/usr/bin/env python
"""
Generate isochrones for Montana timber mill at multiple time intervals.
Uses SocialMapper package to create drive-time isochrones.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add SocialMapper to path if needed
socialmapper_path = Path("/Users/mihiarc/repos/socialmapper")
if socialmapper_path.exists():
    sys.path.insert(0, str(socialmapper_path))

# Import SocialMapper
from socialmapper.isochrone import create_isochrone_from_poi, TravelMode

# Add parent directory to path for bigmap imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.console import print_info, print_success, print_error, print_warning

console = Console()


def load_mill_location():
    """Load mill location from CSV configuration file."""
    mill_csv = Path("config/montana_timber_mill_location.csv")
    if not mill_csv.exists():
        raise FileNotFoundError(f"Mill location file not found: {mill_csv}")
    
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    
    return {
        'name': mill_site['name'],
        'lat': mill_site['lat'],
        'lon': mill_site['lon'],
        'notes': mill_site['notes']
    }


def create_poi_dict(name: str, lat: float, lon: float, poi_id: str = None):
    """Create POI dictionary in SocialMapper format."""
    return {
        'id': poi_id or f"mill_{name.lower().replace(' ', '_')}",
        'lat': lat,
        'lon': lon,
        'tags': {
            'name': name,
            'amenity': 'industrial'  # Tag for mill/industrial site
        }
    }


def generate_mill_isochrones(time_intervals: list[int] = None, 
                           travel_mode: str = 'drive',
                           output_dir: str = None,
                           simplify: bool = True):
    """
    Generate isochrones for the mill location at specified time intervals.
    
    Args:
        time_intervals: List of time intervals in minutes (default: [30, 45, 60, 90, 120])
        travel_mode: Travel mode - 'drive', 'walk', or 'bike' (default: 'drive')
        output_dir: Output directory for isochrone files (default: 'output/isochrones')
        simplify: Whether to simplify geometries (default: True)
    
    Returns:
        Dictionary mapping time intervals to file paths
    """
    # Default time intervals
    if time_intervals is None:
        time_intervals = [30, 45, 60, 90, 120]
    
    # Default output directory
    if output_dir is None:
        output_dir = Path("output/isochrones")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mill location
    console.print("[bold cyan]Loading mill location...[/bold cyan]")
    mill = load_mill_location()
    
    console.print(f"\n[cyan]Mill Details:[/cyan]")
    console.print(f"  Name: {mill['name']}")
    console.print(f"  Location: {mill['lat']:.6f}°N, {mill['lon']:.6f}°W")
    console.print(f"  Notes: {mill['notes']}")
    
    # Convert travel mode string to enum
    travel_mode_map = {
        'drive': TravelMode.DRIVE,
        'walk': TravelMode.WALK,
        'bike': TravelMode.BIKE
    }
    
    if travel_mode not in travel_mode_map:
        raise ValueError(f"Invalid travel mode: {travel_mode}. Must be one of: {list(travel_mode_map.keys())}")
    
    travel_mode_enum = travel_mode_map[travel_mode]
    
    # Create POI dictionary
    poi = create_poi_dict(
        name=mill['name'],
        lat=mill['lat'],
        lon=mill['lon'],
        poi_id='custom_0'  # Match the existing isochrone ID format
    )
    
    # Generate isochrones for each time interval
    console.print(f"\n[bold cyan]Generating {travel_mode} isochrones for time intervals: {time_intervals} minutes[/bold cyan]")
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for time_limit in time_intervals:
            task = progress.add_task(f"Generating {time_limit}-minute isochrone...", total=None)
            
            try:
                # Generate isochrone
                isochrone_path = create_isochrone_from_poi(
                    poi=poi,
                    travel_time_limit=time_limit,
                    output_dir=str(output_dir),
                    save_file=True,
                    simplify_tolerance=0.001 if simplify else None,  # ~100m simplification
                    use_parquet=False,  # Use GeoJSON for compatibility
                    travel_mode=travel_mode_enum
                )
                
                # Also save a simplified version with standard naming
                if isinstance(isochrone_path, Path):
                    # Create standardized filename
                    standard_name = f"mill_isochrone_{time_limit}min.geojson"
                    standard_path = output_dir / standard_name
                    
                    # Read and resave with standard naming
                    gdf = gpd.read_file(isochrone_path)
                    
                    # Ensure proper attributes
                    gdf['poi_id'] = 'custom_0'
                    gdf['poi_name'] = f'poi_{poi["id"]}'
                    gdf['travel_time_minutes'] = time_limit
                    gdf['travel_mode'] = travel_mode
                    
                    # Save with standard name
                    gdf.to_file(standard_path, driver='GeoJSON')
                    
                    results[time_limit] = standard_path
                    print_success(f"Created {time_limit}-minute isochrone: {standard_path}")
                
            except Exception as e:
                print_error(f"Failed to generate {time_limit}-minute isochrone: {e}")
                results[time_limit] = None
            
            progress.update(task, completed=True)
    
    # Create summary table
    console.print("\n[bold green]Isochrone Generation Summary[/bold green]")
    
    table = Table(title="Generated Isochrones")
    table.add_column("Time (min)", style="cyan", justify="right")
    table.add_column("File", style="yellow")
    table.add_column("Status", style="green")
    
    for time_limit, path in results.items():
        if path:
            table.add_row(
                str(time_limit),
                path.name if isinstance(path, Path) else str(path),
                "✓ Success"
            )
        else:
            table.add_row(
                str(time_limit),
                "-",
                "✗ Failed"
            )
    
    console.print(table)
    
    # Create combined GeoParquet file with all isochrones
    successful_results = {t: p for t, p in results.items() if p is not None}
    if len(successful_results) > 1:
        console.print("\n[cyan]Creating combined GeoParquet file...[/cyan]")
        
        gdfs = []
        for time_limit, path in successful_results.items():
            gdf = gpd.read_file(path)
            gdf['travel_time_minutes'] = time_limit
            gdfs.append(gdf)
        
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        combined_path = output_dir / f"montana_{travel_mode}_isochrones_combined.geoparquet"
        combined_gdf.to_parquet(combined_path)
        
        print_success(f"Created combined file: {combined_path}")
    
    return results


def main():
    """Main function to run isochrone generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate isochrones for Montana timber mill using SocialMapper"
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=int,
        default=[30, 45, 60, 90, 120],
        help="Time intervals in minutes (default: 30 45 60 90 120)"
    )
    parser.add_argument(
        "--mode",
        choices=['drive', 'walk', 'bike'],
        default='drive',
        help="Travel mode (default: drive)"
    )
    parser.add_argument(
        "--output",
        default="output/isochrones",
        help="Output directory (default: output/isochrones)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Don't simplify geometries"
    )
    
    args = parser.parse_args()
    
    console.print("[bold]Montana Timber Mill Isochrone Generator[/bold]")
    console.print("Using SocialMapper for isochrone generation\n")
    
    # Generate isochrones
    results = generate_mill_isochrones(
        time_intervals=args.times,
        travel_mode=args.mode,
        output_dir=args.output,
        simplify=not args.no_simplify
    )
    
    # Summary
    successful = sum(1 for r in results.values() if r is not None)
    console.print(f"\n[bold green]✓ Successfully generated {successful}/{len(results)} isochrones[/bold green]")
    
    if successful < len(results):
        failed_times = [t for t, r in results.items() if r is None]
        console.print(f"[yellow]⚠ Failed time intervals: {failed_times}[/yellow]")


if __name__ == "__main__":
    main()