"""
CONUS Tile Grid System for GridFIA.

This module provides a fixed grid tile system for efficiently processing
and serving BIGMAP forest data across the continental United States.

Key Components:
- TileIndex: Grid calculations and tile lookups
- TileDownloader: Download species data from BIGMAP API
- TileProcessor: Convert downloads to Zarr format
- CloudStorage: Upload/download tiles to/from R2
"""

from .tile_index import TileIndex, TileInfo, CONUSGrid
from .tile_downloader import TileDownloader
from .tile_processor import TileProcessor
from .cloud_storage import CloudStorage

__all__ = [
    "TileIndex",
    "TileInfo",
    "CONUSGrid",
    "TileDownloader",
    "TileProcessor",
    "CloudStorage",
]
