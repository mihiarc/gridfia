#!/usr/bin/env python3
"""
Create and Update State Data Manifest

This script creates a manifest.json file that indexes all available state Zarr stores.
The manifest enables the GridFIA API to discover and load state-specific data from cloud storage.

Usage:
    # Create manifest from local state data
    python scripts/create_manifest.py --input-dir ./us_forest_data --output manifest.json

    # Create manifest with cloud URLs
    python scripts/create_manifest.py --input-dir ./us_forest_data --base-url https://pub-xxx.r2.dev

Manifest Format:
{
    "version": "1.0",
    "created": "2025-01-01T00:00:00",
    "data_source": "BIGMAP 2018",
    "resolution_m": 30,
    "crs": "EPSG:3857",
    "total_states": 50,
    "states": {
        "NC": {
            "name": "North Carolina",
            "url": "https://...",
            "local_path": "./us_forest_data/states/nc/nc_forest.zarr",
            "num_species": 326,
            "shape": [326, 2000, 3000],
            "bounds": [-84.32, 33.84, -75.46, 36.59],
            "size_mb": 450
        },
        ...
    }
}
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table

console = Console()


def collect_state_info(state_dir: Path) -> Optional[Dict]:
    """
    Collect metadata from a state Zarr store.

    Args:
        state_dir: Path to state directory containing the Zarr store

    Returns:
        Dict with state metadata or None if invalid
    """
    import zarr

    state_abbr = state_dir.name.upper()
    zarr_path = state_dir / f'{state_abbr.lower()}_forest.zarr'

    if not zarr_path.exists():
        return None

    try:
        root = zarr.open(str(zarr_path), mode='r')

        # Get metadata
        if 'biomass' in root:
            shape = list(root['biomass'].shape)
            chunks = list(root['biomass'].chunks)
        else:
            shape = list(root.shape)
            chunks = list(root.chunks)

        bounds = root.attrs.get('bounds', [])
        crs = root.attrs.get('crs', 'EPSG:3857')
        num_species = root.attrs.get('num_species', shape[0] if shape else 0)

        # Calculate size
        size_bytes = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
        size_mb = size_bytes / (1024 * 1024)

        return {
            'abbr': state_abbr,
            'local_path': str(zarr_path),
            'num_species': num_species,
            'shape': shape,
            'chunks': chunks,
            'bounds': bounds,
            'crs': crs,
            'size_mb': round(size_mb, 2)
        }

    except Exception as e:
        console.print(f"[red]Error reading {zarr_path}: {e}[/red]")
        return None


# State names lookup
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming'
}


def create_manifest(
    input_dir: Path,
    output_path: Path,
    base_url: Optional[str] = None,
    include_local: bool = True
) -> Dict:
    """
    Create manifest from processed state data.

    Args:
        input_dir: Directory containing state subdirectories
        output_path: Path to write manifest.json
        base_url: Base URL for cloud storage (e.g., https://pub-xxx.r2.dev/states)
        include_local: Include local paths in manifest

    Returns:
        The manifest dictionary
    """
    states_dir = input_dir / 'states'

    if not states_dir.exists():
        console.print(f"[red]States directory not found: {states_dir}[/red]")
        return {}

    manifest = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'data_source': 'USDA Forest Service BIGMAP 2018',
        'resolution_m': 30,
        'default_crs': 'EPSG:3857',
        'chunk_size': [1, 512, 512],
        'compression': 'lz4',
        'states': {}
    }

    total_size_mb = 0
    total_species = 0

    # Scan for state directories
    for state_path in sorted(states_dir.iterdir()):
        if not state_path.is_dir():
            continue

        info = collect_state_info(state_path)
        if not info:
            continue

        state_abbr = info['abbr']
        state_name = STATE_NAMES.get(state_abbr, state_abbr)

        state_entry = {
            'name': state_name,
            'num_species': info['num_species'],
            'shape': info['shape'],
            'bounds': info['bounds'],
            'size_mb': info['size_mb']
        }

        if include_local:
            state_entry['local_path'] = info['local_path']

        if base_url:
            state_entry['url'] = f"{base_url}/{state_abbr.lower()}/{state_abbr.lower()}_forest.zarr"

        manifest['states'][state_abbr] = state_entry
        total_size_mb += info['size_mb']
        total_species = max(total_species, info['num_species'])

        console.print(f"  [green]Added {state_abbr}: {info['num_species']} species, {info['size_mb']:.1f} MB[/green]")

    manifest['total_states'] = len(manifest['states'])
    manifest['total_size_mb'] = round(total_size_mb, 2)
    manifest['max_species'] = total_species

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[bold green]Manifest written to {output_path}[/bold green]")

    return manifest


def print_manifest_summary(manifest: Dict):
    """Print a summary table of the manifest."""
    table = Table(title="State Data Manifest")
    table.add_column("State", style="cyan")
    table.add_column("Name")
    table.add_column("Species")
    table.add_column("Shape")
    table.add_column("Size (MB)")
    table.add_column("URL", style="dim")

    for abbr, info in sorted(manifest.get('states', {}).items()):
        table.add_row(
            abbr,
            info.get('name', 'N/A'),
            str(info.get('num_species', 'N/A')),
            str(info.get('shape', 'N/A')),
            f"{info.get('size_mb', 0):.1f}",
            info.get('url', 'N/A')[:40] + '...' if info.get('url') else 'N/A'
        )

    console.print(table)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total states: {manifest.get('total_states', 0)}")
    console.print(f"  Total size: {manifest.get('total_size_mb', 0):.1f} MB ({manifest.get('total_size_mb', 0)/1024:.2f} GB)")
    console.print(f"  Max species: {manifest.get('max_species', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create state data manifest for GridFIA"
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('./us_forest_data'),
        help='Directory containing processed state data'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./us_forest_data/manifest.json'),
        help='Output manifest file path'
    )
    parser.add_argument(
        '--base-url',
        help='Base URL for cloud storage (e.g., https://pub-xxx.r2.dev/states)'
    )
    parser.add_argument(
        '--no-local',
        action='store_true',
        help='Exclude local paths from manifest'
    )

    args = parser.parse_args()

    console.print("[bold]Creating State Data Manifest[/bold]\n")

    manifest = create_manifest(
        input_dir=args.input_dir,
        output_path=args.output,
        base_url=args.base_url,
        include_local=not args.no_local
    )

    if manifest.get('states'):
        print_manifest_summary(manifest)


if __name__ == '__main__':
    main()
