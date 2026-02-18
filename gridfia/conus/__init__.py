"""
CONUS Tile Grid System for GridFIA.

This module provides a fixed grid tile system for efficiently processing
and serving BIGMAP forest data across the continental United States.

Key Components:
- TileIndex: Grid calculations and tile lookups
- TileDownloader: Download species data from BIGMAP API
- TileProcessor: Convert downloads to Zarr format
- CloudStorage: Upload/download tiles to/from R2 (requires boto3)
"""

from .tile_index import TileIndex, TileInfo, CONUSGrid
from .tile_downloader import TileDownloader
from .tile_processor import TileProcessor


def __getattr__(name: str):
    """Lazy import CloudStorage to avoid requiring boto3 at import time."""
    if name == "CloudStorage":
        from .cloud_storage import CloudStorage
        return CloudStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TileIndex",
    "TileInfo",
    "CONUSGrid",
    "TileDownloader",
    "TileProcessor",
    "CloudStorage",
]
