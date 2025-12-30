#!/usr/bin/env python
"""
CONUS Tile Pipeline - Build national forest data tiles.

This script processes the continental US forest data into a fixed grid
of tiles, uploading each tile to R2 and cleaning up local storage.

Usage:
    # Test with 3 tiles
    python scripts/build_conus_tiles.py --test

    # Process all tiles (streaming mode)
    python scripts/build_conus_tiles.py --all

    # Process specific tiles
    python scripts/build_conus_tiles.py --tiles conus_027_015 conus_027_016

    # Resume from checkpoint
    python scripts/build_conus_tiles.py --resume
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridfia.conus import (
    TileIndex,
    TileDownloader,
    TileProcessor,
    CloudStorage,
    CONUSGrid,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()

# Default paths
DEFAULT_WORK_DIR = Path("conus_pipeline_work")
DEFAULT_CHECKPOINT = Path("conus_pipeline_checkpoint.json")


class PipelineCheckpoint:
    """Manages pipeline checkpoint for resume capability."""

    def __init__(self, path: Path = DEFAULT_CHECKPOINT):
        self.path = path
        self.data = {
            "version": "1.0.0",
            "started": None,
            "mode": "streaming",
            "tiles_completed": [],
            "tiles_uploaded": [],
            "tiles_failed": [],
            "current_tile": None,
            "total_tiles": 0,
            "last_updated": None,
        }

    def load(self) -> bool:
        """Load checkpoint from file. Returns True if loaded."""
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)
            return True
        return False

    def save(self) -> None:
        """Save checkpoint to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def start(self, total_tiles: int) -> None:
        """Initialize a new pipeline run."""
        self.data["started"] = datetime.now().isoformat()
        self.data["total_tiles"] = total_tiles
        self.data["tiles_completed"] = []
        self.data["tiles_uploaded"] = []
        self.data["tiles_failed"] = []
        self.save()

    def mark_completed(self, tile_id: str) -> None:
        """Mark a tile as completed (processed to Zarr)."""
        if tile_id not in self.data["tiles_completed"]:
            self.data["tiles_completed"].append(tile_id)
        self.data["current_tile"] = None
        self.save()

    def mark_uploaded(self, tile_id: str) -> None:
        """Mark a tile as uploaded to R2."""
        if tile_id not in self.data["tiles_uploaded"]:
            self.data["tiles_uploaded"].append(tile_id)
        self.save()

    def mark_failed(self, tile_id: str) -> None:
        """Mark a tile as failed."""
        if tile_id not in self.data["tiles_failed"]:
            self.data["tiles_failed"].append(tile_id)
        self.data["current_tile"] = None
        self.save()

    def set_current(self, tile_id: str) -> None:
        """Set the current tile being processed."""
        self.data["current_tile"] = tile_id
        self.save()

    def get_pending(self, all_tiles: List[str]) -> List[str]:
        """Get tiles that haven't been uploaded yet."""
        done = set(self.data["tiles_uploaded"])
        failed = set(self.data["tiles_failed"])
        return [t for t in all_tiles if t not in done and t not in failed]

    def summary(self) -> dict:
        """Get pipeline summary."""
        return {
            "started": self.data["started"],
            "total": self.data["total_tiles"],
            "completed": len(self.data["tiles_completed"]),
            "uploaded": len(self.data["tiles_uploaded"]),
            "failed": len(self.data["tiles_failed"]),
            "current": self.data["current_tile"],
        }


def process_single_tile(
    tile_id: str,
    downloader: TileDownloader,
    processor: TileProcessor,
    storage: CloudStorage,
    checkpoint: PipelineCheckpoint,
    skip_upload: bool = False,
    species_codes: Optional[List[str]] = None,
) -> bool:
    """
    Process a single tile: download -> zarr -> upload -> cleanup.

    Returns True on success, False on failure.
    """
    grid = CONUSGrid()
    col, row = grid.parse_tile_id(tile_id)
    tile_info = grid.get_tile_info(col, row)

    console.print(f"\n[bold cyan]Processing tile: {tile_id}[/bold cyan]")
    console.print(f"  Bbox: {tile_info.bbox_3857}")

    checkpoint.set_current(tile_id)

    try:
        # Step 1: Download species data
        console.print("\n[yellow]Step 1: Downloading species data...[/yellow]")
        tile_dir, files_downloaded = downloader.download_tile(
            tile_info,
            species_codes=species_codes,
            show_progress=True,
        )
        console.print(f"  Downloaded {files_downloaded} species files")

        # Step 2: Process to Zarr
        console.print("\n[yellow]Step 2: Processing to Zarr...[/yellow]")
        zarr_path, metadata = processor.process_tile(
            tile_dir,
            tile_info,
            show_progress=True,
        )
        console.print(f"  Created Zarr: {metadata['size_mb']:.1f} MB")
        console.print(f"  Valid data: {metadata['valid_percent']:.1f}%")

        checkpoint.mark_completed(tile_id)

        # Step 3: Upload to R2
        if not skip_upload:
            console.print("\n[yellow]Step 3: Uploading to R2...[/yellow]")
            upload_result = storage.upload_tile(
                tile_id,
                zarr_path,
                show_progress=True,
            )
            console.print(f"  Uploaded {upload_result['files_uploaded']} files")
            console.print(f"  Total size: {upload_result['total_size_mb']:.1f} MB")

            # Verify upload
            if storage.verify_tile(tile_id):
                console.print("  [green]Upload verified[/green]")
                checkpoint.mark_uploaded(tile_id)
            else:
                console.print("  [red]Upload verification failed![/red]")
                return False
        else:
            console.print("\n[dim]Step 3: Skipping upload (--skip-upload)[/dim]")
            checkpoint.mark_uploaded(tile_id)  # Mark as done anyway for testing

        # Step 4: Cleanup local files
        console.print("\n[yellow]Step 4: Cleaning up local files...[/yellow]")
        if tile_dir.exists():
            shutil.rmtree(tile_dir)
            console.print(f"  Removed: {tile_dir}")

        console.print(f"\n[bold green]✓ Tile {tile_id} completed successfully[/bold green]")
        return True

    except Exception as e:
        logger.exception(f"Failed to process tile {tile_id}")
        checkpoint.mark_failed(tile_id)
        console.print(f"\n[bold red]✗ Tile {tile_id} failed: {e}[/bold red]")
        return False


def run_pipeline(
    tiles: List[str],
    work_dir: Path,
    checkpoint: PipelineCheckpoint,
    skip_upload: bool = False,
    species_codes: Optional[List[str]] = None,
) -> None:
    """Run the pipeline on a list of tiles."""
    # Initialize components
    downloader = TileDownloader(output_dir=work_dir)
    processor = TileProcessor()
    storage = CloudStorage()

    # Get pending tiles
    pending = checkpoint.get_pending(tiles)

    if not pending:
        console.print("[green]All tiles have been processed![/green]")
        return

    console.print(Panel(
        f"[bold]CONUS Tile Pipeline[/bold]\n\n"
        f"Total tiles: {len(tiles)}\n"
        f"Pending: {len(pending)}\n"
        f"Completed: {len(tiles) - len(pending)}",
        title="Pipeline Status",
    ))

    # Process each tile
    success_count = 0
    fail_count = 0

    for i, tile_id in enumerate(pending):
        console.print(f"\n[bold]Tile {i + 1}/{len(pending)}[/bold]")

        success = process_single_tile(
            tile_id,
            downloader,
            processor,
            storage,
            checkpoint,
            skip_upload=skip_upload,
            species_codes=species_codes,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Print summary
    console.print("\n" + "=" * 50)
    table = Table(title="Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Total tiles", str(len(tiles)))
    table.add_row("Processed this run", str(success_count + fail_count))
    table.add_row("Succeeded", str(success_count))
    table.add_row("Failed", str(fail_count))
    table.add_row("Total uploaded", str(len(checkpoint.data["tiles_uploaded"])))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="CONUS Tile Pipeline - Build national forest data tiles"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 3 tiles",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all CONUS tiles",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        help="Specific tile IDs to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip R2 upload (for local testing)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Working directory for downloads",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--species-subset",
        type=int,
        default=None,
        help="Use subset of species (for faster testing)",
    )

    args = parser.parse_args()

    # Initialize grid and checkpoint
    grid = CONUSGrid()
    checkpoint = PipelineCheckpoint(args.checkpoint)

    # Determine tiles to process
    if args.test:
        # Test with 3 tiles in different regions
        tiles = ["conus_027_015", "conus_035_010", "conus_020_020"]
        console.print("[yellow]Test mode: Processing 3 sample tiles[/yellow]")
    elif args.tiles:
        tiles = args.tiles
    elif args.all or args.resume:
        # All tiles
        all_indices = grid.all_tiles()
        tiles = [grid.get_tile_id(col, row) for col, row in all_indices]
    else:
        parser.print_help()
        console.print("\n[red]Please specify --test, --all, --tiles, or --resume[/red]")
        sys.exit(1)

    # Load or initialize checkpoint
    if args.resume and checkpoint.load():
        console.print(f"[green]Resuming from checkpoint: {args.checkpoint}[/green]")
        summary = checkpoint.summary()
        console.print(f"  Started: {summary['started']}")
        console.print(f"  Uploaded: {summary['uploaded']}/{summary['total']}")
        if summary['failed']:
            console.print(f"  [red]Failed: {summary['failed']}[/red]")
    else:
        checkpoint.start(len(tiles))

    # Get species subset if requested
    species_codes = None
    if args.species_subset:
        # Get first N species for testing
        downloader = TileDownloader(output_dir=args.work_dir)
        all_species = downloader.get_species_list()
        species_codes = [s["species_code"] for s in all_species[:args.species_subset]]
        console.print(f"[yellow]Using {len(species_codes)} species subset[/yellow]")

    # Create work directory
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    run_pipeline(
        tiles=tiles,
        work_dir=args.work_dir,
        checkpoint=checkpoint,
        skip_upload=args.skip_upload,
        species_codes=species_codes,
    )


if __name__ == "__main__":
    main()
