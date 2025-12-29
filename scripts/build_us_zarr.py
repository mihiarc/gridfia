#!/usr/bin/env python3
"""
Build US-Wide Zarr Store for GridFIA

This script creates a country-wide Zarr store from BIGMAP 2018 forest data.
It processes state-by-state to manage memory and provides checkpoint/resume capability.

Pipeline Stages:
1. Download all species GeoTIFFs for each state
2. Create per-state Zarr stores with consistent chunking
3. Track progress for resume capability

Usage:
    # Process specific states
    python scripts/build_us_zarr.py --states NC VA --output-dir ./us_data

    # Process all states (will take many hours)
    python scripts/build_us_zarr.py --all-states --output-dir ./us_data

    # Resume interrupted processing
    python scripts/build_us_zarr.py --resume --output-dir ./us_data

    # Dry run to see what would be processed
    python scripts/build_us_zarr.py --all-states --dry-run

Configuration:
    The script uses optimal settings for cloud streaming:
    - Chunk size: (1, 512, 512) - efficient spatial subsetting
    - Compression: LZ4 level 5 - fast decompression
    - CRS: EPSG:3857 (Web Mercator) - matches BIGMAP source
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# US States with FIPS codes (CONUS + AK, HI)
US_STATES = {
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

# States to prioritize (have significant forest cover)
PRIORITY_STATES = [
    'NC', 'VA', 'GA', 'SC',  # Southeast
    'OR', 'WA', 'CA',         # Pacific Northwest
    'ME', 'NH', 'VT', 'NY',   # Northeast
    'MI', 'WI', 'MN',         # Great Lakes
    'MT', 'ID', 'CO'          # Mountain West
]


class ProgressTracker:
    """Track pipeline progress with checkpoint/resume capability."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / 'pipeline_progress.json'
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load progress from checkpoint file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {
            'started': datetime.now().isoformat(),
            'completed_states': [],
            'failed_states': [],
            'in_progress': None,
            'total_species_downloaded': 0,
            'total_zarr_size_mb': 0
        }

    def save(self):
        """Save progress to checkpoint file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def mark_started(self, state: str):
        """Mark a state as in-progress."""
        self.progress['in_progress'] = state
        self.save()

    def mark_completed(self, state: str, num_species: int, zarr_size_mb: float):
        """Mark a state as completed."""
        if state not in self.progress['completed_states']:
            self.progress['completed_states'].append(state)
        self.progress['total_species_downloaded'] += num_species
        self.progress['total_zarr_size_mb'] += zarr_size_mb
        self.progress['in_progress'] = None
        if state in self.progress['failed_states']:
            self.progress['failed_states'].remove(state)
        self.save()

    def mark_failed(self, state: str, error: str):
        """Mark a state as failed."""
        if state not in self.progress['failed_states']:
            self.progress['failed_states'].append(state)
        self.progress['in_progress'] = None
        self.progress[f'error_{state}'] = error
        self.save()

    def get_pending_states(self, requested_states: List[str]) -> List[str]:
        """Get states that haven't been completed yet."""
        completed = set(self.progress['completed_states'])
        return [s for s in requested_states if s not in completed]

    def print_summary(self):
        """Print progress summary."""
        table = Table(title="Pipeline Progress")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Started", self.progress.get('started', 'N/A'))
        table.add_row("Completed States", str(len(self.progress['completed_states'])))
        table.add_row("Failed States", str(len(self.progress['failed_states'])))
        table.add_row("Total Species Downloaded", str(self.progress['total_species_downloaded']))
        table.add_row("Total Zarr Size (MB)", f"{self.progress['total_zarr_size_mb']:.1f}")

        if self.progress['completed_states']:
            table.add_row("Completed", ', '.join(sorted(self.progress['completed_states'])))
        if self.progress['failed_states']:
            table.add_row("Failed", ', '.join(sorted(self.progress['failed_states'])))

        console.print(table)


def process_state(
    state_abbr: str,
    output_dir: Path,
    chunk_size: tuple = (1, 512, 512),
    compression: str = 'lz4',
    compression_level: int = 5,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    Process a single state: download all species and create Zarr store.

    Returns dict with stats on success, None on failure.
    """
    from gridfia import GridFIA

    state_name = US_STATES[state_abbr]
    state_dir = output_dir / 'states' / state_abbr.lower()
    downloads_dir = state_dir / 'downloads'
    zarr_path = state_dir / f'{state_abbr.lower()}_forest.zarr'

    console.print(f"\n[bold blue]Processing {state_name} ({state_abbr})[/bold blue]")

    if dry_run:
        console.print(f"  [dim]Would download to: {downloads_dir}[/dim]")
        console.print(f"  [dim]Would create Zarr: {zarr_path}[/dim]")
        return {'state': state_abbr, 'dry_run': True}

    try:
        api = GridFIA()

        # Check if Zarr already exists
        if zarr_path.exists():
            console.print(f"  [yellow]Zarr already exists, validating...[/yellow]")
            info = api.validate_zarr(zarr_path)
            size_mb = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024 * 1024)
            return {
                'state': state_abbr,
                'num_species': info['num_species'],
                'shape': info['shape'],
                'zarr_size_mb': size_mb
            }

        # Step 1: Download all species
        console.print(f"  [cyan]Downloading all species for {state_name}...[/cyan]")
        downloads_dir.mkdir(parents=True, exist_ok=True)

        files = api.download_species(
            state=state_name,
            species_codes=None,  # All species
            output_dir=str(downloads_dir)
        )

        console.print(f"  [green]Downloaded {len(files)} species files[/green]")

        # Step 2: Create Zarr store with optimized chunking
        console.print(f"  [cyan]Creating Zarr store with chunks {chunk_size}...[/cyan]")

        zarr_path = api.create_zarr(
            input_dir=str(downloads_dir),
            output_path=str(zarr_path),
            chunk_size=chunk_size,
            compression=compression,
            compression_level=compression_level
        )

        # Get stats
        info = api.validate_zarr(zarr_path)
        size_mb = sum(f.stat().st_size for f in Path(zarr_path).rglob('*') if f.is_file()) / (1024 * 1024)

        console.print(f"  [green]Created Zarr: {info['shape']}, {size_mb:.1f} MB[/green]")

        # Optionally clean up downloads to save space
        # shutil.rmtree(downloads_dir)

        return {
            'state': state_abbr,
            'num_species': info['num_species'],
            'shape': info['shape'],
            'zarr_size_mb': size_mb
        }

    except Exception as e:
        console.print(f"  [red]Error processing {state_abbr}: {e}[/red]")
        logger.exception(f"Failed to process {state_abbr}")
        return None


def run_pipeline(
    states: List[str],
    output_dir: Path,
    dry_run: bool = False,
    resume: bool = True
):
    """
    Run the US-wide data pipeline.

    Args:
        states: List of state abbreviations to process
        output_dir: Base output directory
        dry_run: If True, just show what would be done
        resume: If True, skip already completed states
    """
    output_dir = Path(output_dir)
    tracker = ProgressTracker(output_dir)

    # Show current progress
    if tracker.progress['completed_states']:
        console.print("\n[bold]Current Progress:[/bold]")
        tracker.print_summary()

    # Get states to process
    if resume:
        pending_states = tracker.get_pending_states(states)
        if len(pending_states) < len(states):
            console.print(f"\n[yellow]Resuming: {len(states) - len(pending_states)} states already completed[/yellow]")
    else:
        pending_states = states

    if not pending_states:
        console.print("\n[green]All requested states have been processed![/green]")
        return

    console.print(Panel(
        f"Processing {len(pending_states)} states: {', '.join(pending_states)}",
        title="US-Wide Pipeline",
        border_style="blue"
    ))

    # Process each state
    results = []
    for i, state in enumerate(pending_states, 1):
        console.print(f"\n[bold]State {i}/{len(pending_states)}[/bold]")

        tracker.mark_started(state)
        result = process_state(state, output_dir, dry_run=dry_run)

        if result:
            if not dry_run:
                tracker.mark_completed(
                    state,
                    result.get('num_species', 0),
                    result.get('zarr_size_mb', 0)
                )
            results.append(result)
        else:
            tracker.mark_failed(state, "Processing failed")

    # Final summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Pipeline Complete[/bold green]")
    tracker.print_summary()

    # Show per-state details
    if results and not dry_run:
        table = Table(title="Processed States")
        table.add_column("State", style="cyan")
        table.add_column("Species")
        table.add_column("Shape")
        table.add_column("Size (MB)")

        for r in results:
            table.add_row(
                r['state'],
                str(r.get('num_species', 'N/A')),
                str(r.get('shape', 'N/A')),
                f"{r.get('zarr_size_mb', 0):.1f}"
            )
        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Build US-Wide Zarr Store for GridFIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process North Carolina and Virginia
    python scripts/build_us_zarr.py --states NC VA --output-dir ./us_data

    # Process priority states (highest forest coverage)
    python scripts/build_us_zarr.py --priority --output-dir ./us_data

    # Process all 50 states
    python scripts/build_us_zarr.py --all-states --output-dir ./us_data

    # Resume interrupted processing
    python scripts/build_us_zarr.py --resume --output-dir ./us_data
        """
    )

    parser.add_argument(
        '--states',
        nargs='+',
        help='State abbreviations to process (e.g., NC VA GA)'
    )
    parser.add_argument(
        '--priority',
        action='store_true',
        help='Process priority states (high forest coverage)'
    )
    parser.add_argument(
        '--all-states',
        action='store_true',
        help='Process all 50 US states'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./us_forest_data'),
        help='Output directory (default: ./us_forest_data)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint (default: True)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without processing'
    )

    args = parser.parse_args()

    # Determine which states to process
    if args.all_states:
        states = list(US_STATES.keys())
    elif args.priority:
        states = PRIORITY_STATES
    elif args.states:
        # Validate state abbreviations
        invalid = [s for s in args.states if s.upper() not in US_STATES]
        if invalid:
            console.print(f"[red]Invalid state abbreviations: {invalid}[/red]")
            console.print(f"[dim]Valid states: {', '.join(sorted(US_STATES.keys()))}[/dim]")
            sys.exit(1)
        states = [s.upper() for s in args.states]
    else:
        parser.print_help()
        console.print("\n[yellow]Please specify states with --states, --priority, or --all-states[/yellow]")
        sys.exit(1)

    resume = args.resume and not args.no_resume

    run_pipeline(
        states=states,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=resume
    )


if __name__ == '__main__':
    main()
