"""
Tile Processor for CONUS Grid System.

Converts downloaded GeoTIFF files to Zarr format with consistent chunking.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
import json

import numpy as np
import rasterio
import zarr
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from .tile_index import TileInfo, TILE_SIZE_PX

logger = logging.getLogger(__name__)
console = Console()

# Zarr chunk size for cloud streaming
CHUNK_SIZE = (1, 512, 512)  # (species, height, width)

# NoData value
NODATA_VALUE = -9999.0


class TileProcessor:
    """
    Processes downloaded GeoTIFFs into Zarr stores.

    Creates Zarr stores with:
    - Consistent chunking for cloud streaming
    - LZ4 compression for fast decompression
    - Species metadata arrays
    """

    def __init__(self, compression: str = "lz4", compression_level: int = 5):
        """
        Initialize tile processor.

        Args:
            compression: Compression algorithm (lz4, zstd, gzip)
            compression_level: Compression level (1-9)
        """
        self.compression = compression
        self.compression_level = compression_level

    def process_tile(
        self,
        tile_dir: Path,
        tile_info: TileInfo,
        output_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Process downloaded GeoTIFFs into a Zarr store.

        Args:
            tile_dir: Directory containing downloaded TIFFs
            tile_info: Tile metadata
            output_path: Output Zarr path (default: tile_dir/biomass.zarr)
            show_progress: Show progress bar

        Returns:
            Tuple of (zarr_path, metadata_dict)
        """
        downloads_dir = tile_dir / "downloads"
        if not downloads_dir.exists():
            raise FileNotFoundError(f"Downloads directory not found: {downloads_dir}")

        # Find all species TIFFs
        tiff_files = sorted(downloads_dir.glob("species_*.tif"))
        if not tiff_files:
            raise FileNotFoundError(f"No species TIFFs found in {downloads_dir}")

        # Extract species codes from filenames
        species_codes = []
        for f in tiff_files:
            code = f.stem.replace("species_", "")
            species_codes.append(code)

        # Determine output path
        if output_path is None:
            output_path = tile_dir / "biomass.zarr"

        # Read first file to get dimensions and transform
        with rasterio.open(tiff_files[0]) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs

        num_species = len(species_codes)
        shape = (num_species, height, width)

        logger.info(f"Creating Zarr store: {output_path}")
        logger.info(f"Shape: {shape}, Chunks: {CHUNK_SIZE}")

        # Create Zarr store
        store = zarr.open_group(str(output_path), mode="w")

        # Create biomass array with chunking and compression
        compressor = zarr.codecs.BloscCodec(
            cname=self.compression,
            clevel=self.compression_level,
        )

        biomass = store.create_array(
            "biomass",
            shape=shape,
            chunks=CHUNK_SIZE,
            dtype=np.float32,
            compressors=[compressor],
            fill_value=NODATA_VALUE,
        )

        # Create species metadata arrays
        store.create_array(
            "species_codes",
            data=np.array(species_codes, dtype="U10"),
        )

        # Process each species file
        valid_pixels = 0
        total_pixels = height * width * num_species

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[green]Processing {tile_info.tile_id}",
                    total=len(tiff_files),
                )

                for i, tiff_file in enumerate(tiff_files):
                    with rasterio.open(tiff_file) as src:
                        data = src.read(1).astype(np.float32)

                        # Replace nodata with our standard value
                        if src.nodata is not None:
                            data[data == src.nodata] = NODATA_VALUE

                        # Count valid pixels
                        valid_pixels += np.sum(data > 0)

                        # Write to Zarr
                        biomass[i, :, :] = data

                    progress.update(task, advance=1)
        else:
            for i, tiff_file in enumerate(tiff_files):
                with rasterio.open(tiff_file) as src:
                    data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        data[data == src.nodata] = NODATA_VALUE
                    valid_pixels += np.sum(data > 0)
                    biomass[i, :, :] = data

        # Calculate valid percent
        valid_percent = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Store metadata as attributes
        store.attrs.update({
            "tile_id": tile_info.tile_id,
            "crs": str(crs),
            "transform": list(transform)[:6],
            "bounds": list(tile_info.bbox_3857),
            "width": width,
            "height": height,
            "num_species": num_species,
            "valid_percent": valid_percent,
            "nodata": NODATA_VALUE,
        })

        # Calculate size
        size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)

        metadata = {
            "zarr_path": str(output_path),
            "shape": shape,
            "valid_percent": valid_percent,
            "size_mb": size_mb,
            "species_count": num_species,
        }

        logger.info(f"Created Zarr store: {output_path}")
        logger.info(f"Valid data: {valid_percent:.1f}%, Size: {size_mb:.1f} MB")

        return output_path, metadata

    def validate_zarr(self, zarr_path: Path) -> bool:
        """
        Validate a Zarr store has correct structure.

        Args:
            zarr_path: Path to Zarr store

        Returns:
            True if valid, False otherwise
        """
        try:
            store = zarr.open_group(str(zarr_path), mode="r")

            # Check required arrays
            if "biomass" not in store:
                logger.error("Missing 'biomass' array")
                return False

            if "species_codes" not in store:
                logger.error("Missing 'species_codes' array")
                return False

            # Check attributes
            required_attrs = ["tile_id", "crs", "transform", "bounds"]
            for attr in required_attrs:
                if attr not in store.attrs:
                    logger.error(f"Missing attribute: {attr}")
                    return False

            # Check dimensions match
            biomass = store["biomass"]
            species_codes = store["species_codes"]
            if biomass.shape[0] != len(species_codes):
                logger.error("Species count mismatch")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def cleanup_zarr(self, zarr_path: Path) -> None:
        """Remove a Zarr store."""
        if zarr_path.exists():
            import shutil

            shutil.rmtree(zarr_path)
            logger.info(f"Cleaned up Zarr: {zarr_path}")
