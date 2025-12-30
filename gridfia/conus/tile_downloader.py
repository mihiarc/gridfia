"""
Tile Downloader for CONUS Grid System.

Downloads species biomass data from FIA BIGMAP REST API for individual tiles.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import logging
import time

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from ..external.fia_client import BigMapRestClient
from .tile_index import TileInfo, CONUSGrid

logger = logging.getLogger(__name__)
console = Console()


class TileDownloader:
    """
    Downloads species data for individual tiles from BIGMAP API.

    Handles:
    - Rate limiting and circuit breaker
    - Progress tracking
    - Error recovery
    """

    def __init__(
        self,
        output_dir: Path,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize tile downloader.

        Args:
            output_dir: Base directory for downloaded files
            rate_limit_delay: Delay between API requests (seconds)
            max_retries: Maximum retry attempts per request
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = BigMapRestClient(
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )
        self.grid = CONUSGrid()
        self._species_list = None

    def get_species_list(self) -> List[dict]:
        """Get list of available species from API."""
        if self._species_list is None:
            self._species_list = self.client.list_available_species()
        return self._species_list

    def download_tile(
        self,
        tile_info: TileInfo,
        species_codes: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Tuple[Path, int]:
        """
        Download all species data for a single tile.

        Args:
            tile_info: Tile information including bbox
            species_codes: Specific species to download (None = all)
            show_progress: Show progress bar

        Returns:
            Tuple of (output_directory, files_downloaded)
        """
        # Create tile output directory
        tile_dir = self.output_dir / tile_info.tile_id / "downloads"
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Get species to download
        if species_codes is None:
            species = self.get_species_list()
            species_codes = [s["species_code"] for s in species]

        bbox = tile_info.bbox_3857
        files_downloaded = 0
        failed_species = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Downloading {tile_info.tile_id}",
                    total=len(species_codes),
                )

                for species_code in species_codes:
                    try:
                        output_file = tile_dir / f"species_{species_code}.tif"

                        if output_file.exists():
                            logger.debug(f"Skipping existing file: {output_file}")
                            files_downloaded += 1
                            progress.update(task, advance=1)
                            continue

                        self.client.export_species_raster(
                            species_code=species_code,
                            bbox=bbox,
                            output_path=output_file,
                            bbox_srs="3857",
                            output_srs="3857",
                        )
                        files_downloaded += 1

                    except Exception as e:
                        logger.warning(f"Failed to download {species_code}: {e}")
                        failed_species.append(species_code)

                    progress.update(task, advance=1)
        else:
            for species_code in species_codes:
                try:
                    output_file = tile_dir / f"species_{species_code}.tif"

                    if output_file.exists():
                        files_downloaded += 1
                        continue

                    self.client.export_species_raster(
                        species_code=species_code,
                        bbox=bbox,
                        output_path=output_file,
                        bbox_srs="3857",
                        output_srs="3857",
                    )
                    files_downloaded += 1

                except Exception as e:
                    logger.warning(f"Failed to download {species_code}: {e}")
                    failed_species.append(species_code)

        if failed_species:
            logger.warning(
                f"Failed to download {len(failed_species)} species for {tile_info.tile_id}"
            )

        return tile_dir.parent, files_downloaded

    def download_tile_by_id(
        self,
        tile_id: str,
        species_codes: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Tuple[Path, int]:
        """
        Download tile by ID.

        Args:
            tile_id: Tile identifier (e.g., conus_027_015)
            species_codes: Specific species to download
            show_progress: Show progress bar

        Returns:
            Tuple of (output_directory, files_downloaded)
        """
        col, row = self.grid.parse_tile_id(tile_id)
        tile_info = self.grid.get_tile_info(col, row)
        return self.download_tile(tile_info, species_codes, show_progress)

    def estimate_tile_size(self, num_species: int = 326) -> float:
        """
        Estimate download size for a tile in MB.

        Args:
            num_species: Number of species to download

        Returns:
            Estimated size in MB
        """
        # Each species GeoTIFF is approximately:
        # 4096 x 4096 pixels x 4 bytes (float32) = 64 MB uncompressed
        # With LZW compression, typically 5-15 MB per species with data
        # Empty/sparse tiles are much smaller (~16 KB)
        # Average estimate: ~1-2 MB per species considering mixed coverage
        return num_species * 1.5

    def cleanup_tile_downloads(self, tile_id: str) -> None:
        """
        Remove downloaded files for a tile.

        Used after successful Zarr creation and upload.
        """
        tile_dir = self.output_dir / tile_id
        if tile_dir.exists():
            import shutil

            shutil.rmtree(tile_dir)
            logger.info(f"Cleaned up downloads for {tile_id}")
