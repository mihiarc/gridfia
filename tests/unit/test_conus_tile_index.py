"""Tests for CONUS tile index module.

Tests grid calculations, tile lookups, and spatial indexing
for the continental US forest data tile system.
"""

import pytest
import json
from pathlib import Path

from gridfia.conus.tile_index import (
    TileInfo,
    CONUSGrid,
    TileIndex,
    CONUS_BOUNDS_3857,
    TILE_SIZE_PX,
    PIXEL_SIZE_M,
    TILE_SIZE_M,
)


class TestTileInfoModel:
    """Tests for the TileInfo Pydantic model."""

    def test_tile_info_creation_minimal(self):
        """Test creating TileInfo with minimal required fields."""
        tile = TileInfo(
            tile_id="conus_027_015",
            col=27,
            row=15,
            bbox_3857=(-10595496, 4657335, -10472616, 4780215),
        )

        assert tile.tile_id == "conus_027_015"
        assert tile.col == 27
        assert tile.row == 15
        assert tile.bbox_3857 == (-10595496, 4657335, -10472616, 4780215)
        assert tile.states == []  # Default
        assert tile.valid_percent == 0.0  # Default
        assert tile.size_mb == 0.0  # Default
        assert tile.status == "pending"  # Default

    def test_tile_info_creation_full(self):
        """Test creating TileInfo with all fields."""
        tile = TileInfo(
            tile_id="conus_000_000",
            col=0,
            row=0,
            bbox_3857=(-13914936, 2814455, -13792056, 2937335),
            states=["CA", "NV"],
            valid_percent=75.5,
            size_mb=125.3,
            status="completed",
        )

        assert tile.states == ["CA", "NV"]
        assert tile.valid_percent == 75.5
        assert tile.size_mb == 125.3
        assert tile.status == "completed"

    def test_tile_info_validation_col_non_negative(self):
        """Test that col must be non-negative."""
        with pytest.raises(ValueError):
            TileInfo(
                tile_id="invalid",
                col=-1,  # Invalid
                row=0,
                bbox_3857=(0, 0, 1, 1),
            )

    def test_tile_info_validation_row_non_negative(self):
        """Test that row must be non-negative."""
        with pytest.raises(ValueError):
            TileInfo(
                tile_id="invalid",
                col=0,
                row=-1,  # Invalid
                bbox_3857=(0, 0, 1, 1),
            )

    def test_tile_info_validation_valid_percent_range(self):
        """Test that valid_percent must be 0-100."""
        # Valid at boundaries
        tile = TileInfo(
            tile_id="test",
            col=0,
            row=0,
            bbox_3857=(0, 0, 1, 1),
            valid_percent=0.0,
        )
        assert tile.valid_percent == 0.0

        tile = TileInfo(
            tile_id="test",
            col=0,
            row=0,
            bbox_3857=(0, 0, 1, 1),
            valid_percent=100.0,
        )
        assert tile.valid_percent == 100.0

        # Invalid - over 100
        with pytest.raises(ValueError):
            TileInfo(
                tile_id="test",
                col=0,
                row=0,
                bbox_3857=(0, 0, 1, 1),
                valid_percent=101.0,
            )

    def test_tile_info_serialization(self):
        """Test TileInfo can be serialized to dict."""
        tile = TileInfo(
            tile_id="conus_010_020",
            col=10,
            row=20,
            bbox_3857=(1000, 2000, 3000, 4000),
            states=["TX"],
            valid_percent=50.0,
            size_mb=100.0,
            status="processing",
        )

        data = tile.model_dump()

        assert data["tile_id"] == "conus_010_020"
        assert data["col"] == 10
        assert data["row"] == 20
        assert data["bbox_3857"] == (1000, 2000, 3000, 4000)
        assert data["states"] == ["TX"]
        assert data["valid_percent"] == 50.0
        assert data["size_mb"] == 100.0
        assert data["status"] == "processing"


class TestCONUSGrid:
    """Tests for the CONUSGrid dataclass."""

    @pytest.fixture
    def grid(self):
        """Create a CONUSGrid instance."""
        return CONUSGrid()

    def test_grid_initialization_defaults(self, grid):
        """Test grid initializes with correct default values."""
        assert grid.origin_x == CONUS_BOUNDS_3857["xmin"]
        assert grid.origin_y == CONUS_BOUNDS_3857["ymin"]
        assert grid.tile_size_px == TILE_SIZE_PX
        assert grid.tile_size_m == TILE_SIZE_M
        assert grid.pixel_size_m == PIXEL_SIZE_M

    def test_grid_calculates_dimensions(self, grid):
        """Test grid calculates correct number of columns and rows."""
        assert grid.num_cols > 0
        assert grid.num_rows > 0
        assert grid.total_tiles == grid.num_cols * grid.num_rows

    def test_grid_dimensions_are_reasonable(self, grid):
        """Test grid dimensions are reasonable for CONUS coverage."""
        # CONUS is roughly 4000km x 2500km, with ~123km tiles
        # Should be roughly 33 cols x 20 rows
        assert 20 <= grid.num_cols <= 100
        assert 15 <= grid.num_rows <= 50

    def test_get_tile_id_format(self, grid):
        """Test tile ID generation format."""
        tile_id = grid.get_tile_id(0, 0)
        assert tile_id == "conus_000_000"

        tile_id = grid.get_tile_id(27, 15)
        assert tile_id == "conus_027_015"

        tile_id = grid.get_tile_id(99, 99)
        assert tile_id == "conus_099_099"

    def test_parse_tile_id_valid(self, grid):
        """Test parsing valid tile IDs."""
        col, row = grid.parse_tile_id("conus_000_000")
        assert col == 0
        assert row == 0

        col, row = grid.parse_tile_id("conus_027_015")
        assert col == 27
        assert row == 15

    def test_parse_tile_id_invalid_format(self, grid):
        """Test parsing invalid tile ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tile ID format"):
            grid.parse_tile_id("invalid")

        with pytest.raises(ValueError, match="Invalid tile ID format"):
            grid.parse_tile_id("tiles_001_002")  # Wrong prefix

        with pytest.raises(ValueError, match="Invalid tile ID format"):
            grid.parse_tile_id("conus_001")  # Missing row

    def test_get_tile_id_and_parse_roundtrip(self, grid):
        """Test that get_tile_id and parse_tile_id are inverses."""
        for col in [0, 10, 27, 50]:
            for row in [0, 5, 15, 29]:
                tile_id = grid.get_tile_id(col, row)
                parsed_col, parsed_row = grid.parse_tile_id(tile_id)
                assert parsed_col == col
                assert parsed_row == row

    def test_get_tile_bbox_origin_tile(self, grid):
        """Test bounding box for origin tile (0, 0)."""
        bbox = grid.get_tile_bbox(0, 0)

        assert bbox[0] == grid.origin_x  # xmin
        assert bbox[1] == grid.origin_y  # ymin
        assert bbox[2] == grid.origin_x + grid.tile_size_m  # xmax
        assert bbox[3] == grid.origin_y + grid.tile_size_m  # ymax

    def test_get_tile_bbox_adjacent_tiles(self, grid):
        """Test that adjacent tiles share edges."""
        bbox_00 = grid.get_tile_bbox(0, 0)
        bbox_10 = grid.get_tile_bbox(1, 0)
        bbox_01 = grid.get_tile_bbox(0, 1)

        # Tile (1,0) should start where (0,0) ends (x-axis)
        assert bbox_10[0] == bbox_00[2]

        # Tile (0,1) should start where (0,0) ends (y-axis)
        assert bbox_01[1] == bbox_00[3]

    def test_get_tile_bbox_size_consistent(self, grid):
        """Test all tiles have consistent size."""
        for col in [0, 5, 10]:
            for row in [0, 3, 7]:
                bbox = grid.get_tile_bbox(col, row)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                assert width == grid.tile_size_m
                assert height == grid.tile_size_m

    def test_get_tile_info_creates_tile_info(self, grid):
        """Test get_tile_info returns proper TileInfo object."""
        info = grid.get_tile_info(27, 15)

        assert isinstance(info, TileInfo)
        assert info.tile_id == "conus_027_015"
        assert info.col == 27
        assert info.row == 15
        assert info.bbox_3857 == grid.get_tile_bbox(27, 15)

    def test_get_tiles_for_bbox_single_tile(self, grid):
        """Test getting tiles for a bbox within a single tile."""
        # Get the bbox of tile (0, 0) and shrink it
        bbox = grid.get_tile_bbox(0, 0)
        inner_bbox = (bbox[0] + 100, bbox[1] + 100, bbox[2] - 100, bbox[3] - 100)

        tiles = grid.get_tiles_for_bbox(inner_bbox)

        assert len(tiles) == 1
        assert tiles[0] == (0, 0)

    def test_get_tiles_for_bbox_multiple_tiles(self, grid):
        """Test getting tiles for a bbox spanning multiple tiles."""
        # Create a bbox that spans exactly tiles (5,5) through (6,6)
        # by using points inside the tiles, not at edges
        bbox_55 = grid.get_tile_bbox(5, 5)
        bbox_66 = grid.get_tile_bbox(6, 6)
        # Use inner points to avoid floating point edge issues
        spanning_bbox = (
            bbox_55[0] + 100,  # xmin inside tile 5
            bbox_55[1] + 100,  # ymin inside tile 5
            bbox_66[2] - 100,  # xmax inside tile 6
            bbox_66[3] - 100,  # ymax inside tile 6
        )

        tiles = grid.get_tiles_for_bbox(spanning_bbox)

        assert len(tiles) == 4
        assert (5, 5) in tiles
        assert (6, 5) in tiles
        assert (5, 6) in tiles
        assert (6, 6) in tiles

    def test_get_tiles_for_bbox_clamps_to_grid(self, grid):
        """Test that bbox queries are clamped to valid grid range."""
        # Bbox extending beyond grid boundaries
        huge_bbox = (-999999999, -999999999, 999999999, 999999999)

        tiles = grid.get_tiles_for_bbox(huge_bbox)

        # Should return all tiles in the grid
        assert len(tiles) == grid.total_tiles

    def test_all_tiles_returns_complete_list(self, grid):
        """Test all_tiles returns all grid tiles."""
        tiles = grid.all_tiles()

        assert len(tiles) == grid.total_tiles

        # Check all coordinates are valid
        for col, row in tiles:
            assert 0 <= col < grid.num_cols
            assert 0 <= row < grid.num_rows

    def test_all_tiles_row_major_order(self, grid):
        """Test all_tiles returns tiles in row-major order."""
        tiles = grid.all_tiles()

        # First tiles should be row 0
        assert tiles[0] == (0, 0)
        assert tiles[1] == (1, 0)

        # After all columns in row 0, next should be row 1
        assert tiles[grid.num_cols] == (0, 1)

    def test_to_dict_includes_all_fields(self, grid):
        """Test to_dict includes all grid information."""
        data = grid.to_dict()

        assert data["crs"] == "EPSG:3857"
        assert data["resolution_m"] == grid.pixel_size_m
        assert data["tile_size_px"] == grid.tile_size_px
        assert data["tile_size_m"] == grid.tile_size_m
        assert data["grid_origin"] == [grid.origin_x, grid.origin_y]
        assert data["num_cols"] == grid.num_cols
        assert data["num_rows"] == grid.num_rows
        assert data["total_tiles"] == grid.total_tiles
        assert "bounds" in data


class TestTileIndex:
    """Tests for the TileIndex class."""

    @pytest.fixture
    def index(self):
        """Create a TileIndex instance."""
        return TileIndex()

    @pytest.fixture
    def initialized_index(self, index):
        """Create an initialized TileIndex with all tiles."""
        index.initialize_tiles()
        return index

    def test_tile_index_initialization(self, index):
        """Test TileIndex initializes with correct defaults."""
        assert isinstance(index.grid, CONUSGrid)
        assert index.tiles == {}
        assert index.state_mapping == {}
        assert index.county_mapping == {}
        assert index.version == "1.0.0"

    def test_initialize_tiles_creates_all_tiles(self, index):
        """Test initialize_tiles populates all grid tiles."""
        index.initialize_tiles()

        assert len(index.tiles) == index.grid.total_tiles

        # Check each tile has correct structure
        for tile_id, tile_info in index.tiles.items():
            assert isinstance(tile_info, TileInfo)
            assert tile_info.tile_id == tile_id
            assert tile_info.status == "pending"

    def test_get_tile_existing(self, initialized_index):
        """Test getting an existing tile."""
        tile = initialized_index.get_tile("conus_000_000")

        assert tile is not None
        assert tile.tile_id == "conus_000_000"
        assert tile.col == 0
        assert tile.row == 0

    def test_get_tile_nonexistent(self, index):
        """Test getting a nonexistent tile returns None."""
        tile = index.get_tile("conus_999_999")
        assert tile is None

    def test_get_tiles_for_state(self, index):
        """Test getting tiles for a state."""
        # Set up state mapping
        index.state_mapping = {
            "NC": ["conus_050_020", "conus_050_021"],
            "CA": ["conus_005_015", "conus_006_015"],
        }

        nc_tiles = index.get_tiles_for_state("NC")
        assert nc_tiles == ["conus_050_020", "conus_050_021"]

        ca_tiles = index.get_tiles_for_state("CA")
        assert ca_tiles == ["conus_005_015", "conus_006_015"]

    def test_get_tiles_for_state_case_insensitive(self, index):
        """Test state lookup is case-insensitive."""
        index.state_mapping = {"NC": ["conus_050_020"]}

        assert index.get_tiles_for_state("nc") == ["conus_050_020"]
        assert index.get_tiles_for_state("Nc") == ["conus_050_020"]
        assert index.get_tiles_for_state("NC") == ["conus_050_020"]

    def test_get_tiles_for_state_unknown(self, index):
        """Test getting tiles for unknown state returns empty list."""
        tiles = index.get_tiles_for_state("XX")
        assert tiles == []

    def test_get_tiles_for_county(self, index):
        """Test getting tiles for a county by FIPS code."""
        index.county_mapping = {
            "37183": ["conus_050_020"],  # Wake County, NC
            "06037": ["conus_005_015", "conus_006_015"],  # Los Angeles County, CA
        }

        wake_tiles = index.get_tiles_for_county("37183")
        assert wake_tiles == ["conus_050_020"]

    def test_get_tiles_for_county_unknown(self, index):
        """Test getting tiles for unknown county returns empty list."""
        tiles = index.get_tiles_for_county("00000")
        assert tiles == []

    def test_update_tile_status(self, initialized_index):
        """Test updating tile status."""
        tile_id = "conus_000_000"

        initialized_index.update_tile_status(
            tile_id,
            status="completed",
            valid_percent=85.5,
            size_mb=150.2,
            states=["CA"],
        )

        tile = initialized_index.get_tile(tile_id)
        assert tile.status == "completed"
        assert tile.valid_percent == 85.5
        assert tile.size_mb == 150.2
        assert tile.states == ["CA"]

    def test_update_tile_status_partial(self, initialized_index):
        """Test updating only some fields."""
        tile_id = "conus_000_000"

        # Only update status
        initialized_index.update_tile_status(tile_id, status="processing")

        tile = initialized_index.get_tile(tile_id)
        assert tile.status == "processing"
        assert tile.valid_percent == 0.0  # Unchanged
        assert tile.size_mb == 0.0  # Unchanged

    def test_update_tile_status_nonexistent(self, index):
        """Test updating nonexistent tile does nothing (no error)."""
        # Should not raise
        index.update_tile_status("conus_999_999", status="completed")

    def test_get_pending_tiles(self, initialized_index):
        """Test getting list of pending tiles."""
        # All should be pending initially
        pending = initialized_index.get_pending_tiles()
        assert len(pending) == initialized_index.grid.total_tiles

        # Mark some as completed
        initialized_index.update_tile_status("conus_000_000", status="completed")
        initialized_index.update_tile_status("conus_001_000", status="completed")

        pending = initialized_index.get_pending_tiles()
        assert len(pending) == initialized_index.grid.total_tiles - 2
        assert "conus_000_000" not in pending
        assert "conus_001_000" not in pending

    def test_get_completed_tiles(self, initialized_index):
        """Test getting list of completed tiles."""
        # None completed initially
        completed = initialized_index.get_completed_tiles()
        assert len(completed) == 0

        # Mark some as completed
        initialized_index.update_tile_status("conus_000_000", status="completed")
        initialized_index.update_tile_status("conus_001_000", status="completed")

        completed = initialized_index.get_completed_tiles()
        assert len(completed) == 2
        assert "conus_000_000" in completed
        assert "conus_001_000" in completed

    def test_save_and_load_roundtrip(self, initialized_index, tmp_path):
        """Test saving and loading tile index."""
        # Set up some state
        initialized_index.update_tile_status(
            "conus_000_000", status="completed", valid_percent=75.0
        )
        initialized_index.state_mapping = {"NC": ["conus_050_020"]}
        initialized_index.county_mapping = {"37183": ["conus_050_020"]}

        # Save
        save_path = tmp_path / "tile_index.json"
        initialized_index.save(save_path)

        assert save_path.exists()

        # Load
        loaded = TileIndex.load(save_path)

        # Verify
        assert loaded.version == initialized_index.version
        assert len(loaded.tiles) == len(initialized_index.tiles)
        assert loaded.state_mapping == initialized_index.state_mapping
        assert loaded.county_mapping == initialized_index.county_mapping

        # Check specific tile was preserved
        tile = loaded.get_tile("conus_000_000")
        assert tile.status == "completed"
        assert tile.valid_percent == 75.0

    def test_save_creates_parent_dirs(self, initialized_index, tmp_path):
        """Test save creates parent directories if they don't exist."""
        save_path = tmp_path / "nested" / "path" / "tile_index.json"

        initialized_index.save(save_path)

        assert save_path.exists()

    def test_load_file_not_found(self, tmp_path):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            TileIndex.load(tmp_path / "nonexistent.json")

    def test_summary_returns_stats(self, initialized_index):
        """Test summary returns correct statistics."""
        # Mark some tiles
        initialized_index.update_tile_status("conus_000_000", status="completed", size_mb=100.0)
        initialized_index.update_tile_status("conus_001_000", status="completed", size_mb=150.0)
        initialized_index.update_tile_status("conus_002_000", status="failed")

        # Set up mappings
        initialized_index.state_mapping = {"NC": ["conus_050_020"], "CA": ["conus_005_015"]}
        initialized_index.county_mapping = {"37183": ["conus_050_020"]}

        summary = initialized_index.summary()

        assert summary["total_tiles"] == initialized_index.grid.total_tiles
        assert summary["status_counts"]["completed"] == 2
        assert summary["status_counts"]["failed"] == 1
        assert summary["status_counts"]["pending"] == initialized_index.grid.total_tiles - 3
        assert summary["total_size_mb"] == 250.0
        assert summary["states_indexed"] == 2
        assert summary["counties_indexed"] == 1


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_conus_bounds_valid(self):
        """Test CONUS bounds are valid Web Mercator coordinates."""
        assert CONUS_BOUNDS_3857["xmin"] < CONUS_BOUNDS_3857["xmax"]
        assert CONUS_BOUNDS_3857["ymin"] < CONUS_BOUNDS_3857["ymax"]

        # Web Mercator x should be negative for western hemisphere
        assert CONUS_BOUNDS_3857["xmin"] < 0
        assert CONUS_BOUNDS_3857["xmax"] < 0

        # Web Mercator y should be positive for northern hemisphere
        assert CONUS_BOUNDS_3857["ymin"] > 0
        assert CONUS_BOUNDS_3857["ymax"] > 0

    def test_tile_size_calculation(self):
        """Test tile size is correctly calculated from pixels and resolution."""
        assert TILE_SIZE_M == TILE_SIZE_PX * PIXEL_SIZE_M

    def test_pixel_size_is_30m(self):
        """Test pixel size matches BIGMAP 30m resolution."""
        assert PIXEL_SIZE_M == 30

    def test_tile_size_is_4096px(self):
        """Test tile size is 4096 pixels (standard tile size)."""
        assert TILE_SIZE_PX == 4096
