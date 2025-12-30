"""
Tile Index for CONUS Grid System.

Provides grid calculations, tile lookups, and spatial indexing
for the continental US forest data tile system.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from pydantic import BaseModel, Field


# CONUS bounds in EPSG:3857 (Web Mercator)
CONUS_BOUNDS_3857 = {
    "xmin": -13914936,
    "ymin": 2814455,
    "xmax": -7402746,
    "ymax": 6360131,
}

# Tile specifications
TILE_SIZE_PX = 4096  # pixels
PIXEL_SIZE_M = 30  # meters
TILE_SIZE_M = TILE_SIZE_PX * PIXEL_SIZE_M  # 122,880 meters


class TileInfo(BaseModel):
    """Information about a single tile."""

    tile_id: str = Field(description="Tile identifier (e.g., conus_027_015)")
    col: int = Field(ge=0, description="Column index")
    row: int = Field(ge=0, description="Row index")
    bbox_3857: Tuple[float, float, float, float] = Field(
        description="Bounding box in EPSG:3857 (xmin, ymin, xmax, ymax)"
    )
    states: List[str] = Field(default_factory=list, description="States intersecting this tile")
    valid_percent: float = Field(default=0.0, ge=0, le=100, description="Percent of tile with valid data")
    size_mb: float = Field(default=0.0, ge=0, description="Size in megabytes")
    status: str = Field(default="pending", description="Processing status")


@dataclass
class CONUSGrid:
    """
    CONUS tile grid definition.

    The grid covers the continental United States with fixed-size tiles
    in Web Mercator projection (EPSG:3857).
    """

    # Grid origin (southwest corner)
    origin_x: float = CONUS_BOUNDS_3857["xmin"]
    origin_y: float = CONUS_BOUNDS_3857["ymin"]

    # Grid extent
    extent_x: float = CONUS_BOUNDS_3857["xmax"] - CONUS_BOUNDS_3857["xmin"]
    extent_y: float = CONUS_BOUNDS_3857["ymax"] - CONUS_BOUNDS_3857["ymin"]

    # Tile dimensions
    tile_size_px: int = TILE_SIZE_PX
    tile_size_m: float = TILE_SIZE_M
    pixel_size_m: float = PIXEL_SIZE_M

    # Computed grid dimensions
    num_cols: int = field(init=False)
    num_rows: int = field(init=False)
    total_tiles: int = field(init=False)

    def __post_init__(self):
        """Calculate grid dimensions."""
        import math

        self.num_cols = math.ceil(self.extent_x / self.tile_size_m)
        self.num_rows = math.ceil(self.extent_y / self.tile_size_m)
        self.total_tiles = self.num_cols * self.num_rows

    def get_tile_id(self, col: int, row: int) -> str:
        """Generate tile ID from column and row indices."""
        return f"conus_{col:03d}_{row:03d}"

    def parse_tile_id(self, tile_id: str) -> Tuple[int, int]:
        """Parse column and row from tile ID."""
        parts = tile_id.split("_")
        if len(parts) != 3 or parts[0] != "conus":
            raise ValueError(f"Invalid tile ID format: {tile_id}")
        return int(parts[1]), int(parts[2])

    def get_tile_bbox(self, col: int, row: int) -> Tuple[float, float, float, float]:
        """
        Get bounding box for a tile in EPSG:3857.

        Returns:
            Tuple of (xmin, ymin, xmax, ymax)
        """
        xmin = self.origin_x + col * self.tile_size_m
        ymin = self.origin_y + row * self.tile_size_m
        xmax = xmin + self.tile_size_m
        ymax = ymin + self.tile_size_m
        return (xmin, ymin, xmax, ymax)

    def get_tile_info(self, col: int, row: int) -> TileInfo:
        """Get TileInfo for a specific tile."""
        tile_id = self.get_tile_id(col, row)
        bbox = self.get_tile_bbox(col, row)
        return TileInfo(tile_id=tile_id, col=col, row=row, bbox_3857=bbox)

    def get_tiles_for_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[Tuple[int, int]]:
        """
        Get all tile indices that intersect a bounding box.

        Args:
            bbox: (xmin, ymin, xmax, ymax) in EPSG:3857

        Returns:
            List of (col, row) tuples
        """
        xmin, ymin, xmax, ymax = bbox

        # Calculate tile range
        col_min = max(0, int((xmin - self.origin_x) / self.tile_size_m))
        col_max = min(self.num_cols - 1, int((xmax - self.origin_x) / self.tile_size_m))
        row_min = max(0, int((ymin - self.origin_y) / self.tile_size_m))
        row_max = min(self.num_rows - 1, int((ymax - self.origin_y) / self.tile_size_m))

        tiles = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                tiles.append((col, row))
        return tiles

    def all_tiles(self) -> List[Tuple[int, int]]:
        """Get all tile indices in row-major order."""
        tiles = []
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                tiles.append((col, row))
        return tiles

    def to_dict(self) -> Dict[str, Any]:
        """Convert grid definition to dictionary."""
        return {
            "crs": "EPSG:3857",
            "resolution_m": self.pixel_size_m,
            "tile_size_px": self.tile_size_px,
            "tile_size_m": self.tile_size_m,
            "grid_origin": [self.origin_x, self.origin_y],
            "num_cols": self.num_cols,
            "num_rows": self.num_rows,
            "total_tiles": self.total_tiles,
            "bounds": {
                "xmin": self.origin_x,
                "ymin": self.origin_y,
                "xmax": self.origin_x + self.num_cols * self.tile_size_m,
                "ymax": self.origin_y + self.num_rows * self.tile_size_m,
            },
        }


class TileIndex:
    """
    Manages the CONUS tile index with spatial lookups.

    Provides methods for:
    - Loading/saving tile index from JSON
    - State and county to tile mappings
    - Tracking tile processing status
    """

    def __init__(self):
        """Initialize tile index."""
        self.grid = CONUSGrid()
        self.tiles: Dict[str, TileInfo] = {}
        self.state_mapping: Dict[str, List[str]] = {}
        self.county_mapping: Dict[str, List[str]] = {}
        self.version = "1.0.0"

    def initialize_tiles(self) -> None:
        """Initialize all tiles in the grid."""
        for col, row in self.grid.all_tiles():
            tile_info = self.grid.get_tile_info(col, row)
            self.tiles[tile_info.tile_id] = tile_info

    def get_tile(self, tile_id: str) -> Optional[TileInfo]:
        """Get tile info by ID."""
        return self.tiles.get(tile_id)

    def get_tiles_for_state(self, state: str) -> List[str]:
        """Get tile IDs for a state."""
        return self.state_mapping.get(state.upper(), [])

    def get_tiles_for_county(self, fips: str) -> List[str]:
        """Get tile IDs for a county (by FIPS code)."""
        return self.county_mapping.get(fips, [])

    def update_tile_status(
        self,
        tile_id: str,
        status: str,
        valid_percent: float = None,
        size_mb: float = None,
        states: List[str] = None,
    ) -> None:
        """Update tile processing status."""
        if tile_id in self.tiles:
            self.tiles[tile_id].status = status
            if valid_percent is not None:
                self.tiles[tile_id].valid_percent = valid_percent
            if size_mb is not None:
                self.tiles[tile_id].size_mb = size_mb
            if states is not None:
                self.tiles[tile_id].states = states

    def get_pending_tiles(self) -> List[str]:
        """Get list of tiles with pending status."""
        return [tid for tid, t in self.tiles.items() if t.status == "pending"]

    def get_completed_tiles(self) -> List[str]:
        """Get list of completed tiles."""
        return [tid for tid, t in self.tiles.items() if t.status == "completed"]

    def save(self, path: Path) -> None:
        """Save tile index to JSON file."""
        data = {
            "version": self.version,
            "grid": self.grid.to_dict(),
            "tiles": {tid: t.model_dump() for tid, t in self.tiles.items()},
            "state_mapping": self.state_mapping,
            "county_mapping": self.county_mapping,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TileIndex":
        """Load tile index from JSON file."""
        with open(path) as f:
            data = json.load(f)

        index = cls()
        index.version = data.get("version", "1.0.0")
        index.state_mapping = data.get("state_mapping", {})
        index.county_mapping = data.get("county_mapping", {})

        for tid, tile_data in data.get("tiles", {}).items():
            index.tiles[tid] = TileInfo(**tile_data)

        return index

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        statuses = {}
        for tile in self.tiles.values():
            statuses[tile.status] = statuses.get(tile.status, 0) + 1

        total_size = sum(t.size_mb for t in self.tiles.values())

        return {
            "total_tiles": len(self.tiles),
            "status_counts": statuses,
            "total_size_mb": total_size,
            "states_indexed": len(self.state_mapping),
            "counties_indexed": len(self.county_mapping),
        }
