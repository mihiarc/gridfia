"""Tests for CONUS tile processor module.

Tests the TileProcessor class that converts GeoTIFFs to Zarr stores.
"""

import pytest
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import zarr
import shutil

from gridfia.conus.tile_processor import (
    TileProcessor,
    CHUNK_SIZE,
    NODATA_VALUE,
)
from gridfia.conus.tile_index import TileInfo


class TestTileProcessorInit:
    """Tests for TileProcessor initialization."""

    def test_default_compression(self):
        """Test default compression settings."""
        processor = TileProcessor()

        assert processor.compression == "lz4"
        assert processor.compression_level == 5

    def test_custom_compression(self):
        """Test custom compression settings."""
        processor = TileProcessor(compression="zstd", compression_level=9)

        assert processor.compression == "zstd"
        assert processor.compression_level == 9

    def test_gzip_compression(self):
        """Test gzip compression option."""
        processor = TileProcessor(compression="gzip", compression_level=6)

        assert processor.compression == "gzip"
        assert processor.compression_level == 6


class TestTileProcessorProcessTile:
    """Tests for process_tile method."""

    @pytest.fixture
    def sample_tile_info(self):
        """Create a sample TileInfo for testing."""
        return TileInfo(
            tile_id="r000_c000",
            row=0,
            col=0,
            bbox_3857=(-8006582.0, 4852835.0, -7782990.0, 5076428.0),
        )

    @pytest.fixture
    def tile_with_tiffs(self, tmp_path, sample_tile_info):
        """Create a tile directory with sample GeoTIFF files."""
        tile_dir = tmp_path / "tile_r000_c000"
        downloads_dir = tile_dir / "downloads"
        downloads_dir.mkdir(parents=True)

        # Create sample GeoTIFFs for 3 species
        height, width = 100, 100
        bbox = sample_tile_info.bbox_3857
        transform = from_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3], width, height
        )
        crs = CRS.from_epsg(3857)

        species_codes = ["0131", "0316", "0202"]
        for i, code in enumerate(species_codes):
            # Create different patterns for each species
            data = np.random.rand(height, width).astype(np.float32) * 100
            # Add some nodata values
            data[0:10, 0:10] = -9999.0

            tiff_path = downloads_dir / f"species_{code}.tif"
            with rasterio.open(
                tiff_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=transform,
                nodata=-9999.0,
            ) as dst:
                dst.write(data, 1)

        return tile_dir, sample_tile_info

    def test_process_tile_basic(self, tile_with_tiffs):
        """Test basic tile processing."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor()

        zarr_path, metadata = processor.process_tile(
            tile_dir, tile_info, show_progress=False
        )

        assert zarr_path.exists()
        assert zarr_path.name == "biomass.zarr"
        assert "shape" in metadata
        assert "valid_percent" in metadata
        assert "size_mb" in metadata
        assert "species_count" in metadata
        assert metadata["species_count"] == 3

    def test_process_tile_custom_output(self, tile_with_tiffs, tmp_path):
        """Test tile processing with custom output path."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor()
        custom_path = tmp_path / "custom_output.zarr"

        zarr_path, metadata = processor.process_tile(
            tile_dir, tile_info, output_path=custom_path, show_progress=False
        )

        assert zarr_path == custom_path
        assert zarr_path.exists()

    def test_process_tile_creates_zarr_structure(self, tile_with_tiffs):
        """Test that processed tile has correct Zarr structure."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor()

        zarr_path, _ = processor.process_tile(
            tile_dir, tile_info, show_progress=False
        )

        store = zarr.open_group(str(zarr_path), mode="r")

        # Check arrays exist
        assert "biomass" in store
        assert "species_codes" in store

        # Check shape
        assert store["biomass"].shape[0] == 3  # 3 species
        assert store["biomass"].shape[1] == 100
        assert store["biomass"].shape[2] == 100

        # Check species codes
        codes = list(store["species_codes"][:])
        assert "0131" in codes
        assert "0316" in codes
        assert "0202" in codes

    def test_process_tile_stores_metadata(self, tile_with_tiffs):
        """Test that processed tile has correct metadata."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor()

        zarr_path, _ = processor.process_tile(
            tile_dir, tile_info, show_progress=False
        )

        store = zarr.open_group(str(zarr_path), mode="r")
        attrs = dict(store.attrs)

        assert attrs["tile_id"] == "r000_c000"
        assert "crs" in attrs
        assert "transform" in attrs
        assert "bounds" in attrs
        assert attrs["width"] == 100
        assert attrs["height"] == 100
        assert attrs["num_species"] == 3
        assert attrs["nodata"] == NODATA_VALUE

    def test_process_tile_with_zstd_compression(self, tile_with_tiffs):
        """Test processing with zstd compression."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor(compression="zstd", compression_level=9)

        zarr_path, metadata = processor.process_tile(
            tile_dir, tile_info, show_progress=False
        )

        assert zarr_path.exists()
        # Verify compression worked (file size should be reasonable)
        assert metadata["size_mb"] < 1.0  # Small test data should compress well

    def test_process_tile_valid_percent(self, tile_with_tiffs):
        """Test that valid percent is calculated correctly."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor()

        _, metadata = processor.process_tile(
            tile_dir, tile_info, show_progress=False
        )

        # We have 100 nodata pixels per species (10x10) out of 10000 total
        # So valid_percent should be around 99%
        assert 0 <= metadata["valid_percent"] <= 100

    def test_process_tile_missing_downloads_dir(self, tmp_path, sample_tile_info):
        """Test error when downloads directory is missing."""
        tile_dir = tmp_path / "empty_tile"
        tile_dir.mkdir()

        processor = TileProcessor()

        with pytest.raises(FileNotFoundError, match="Downloads directory not found"):
            processor.process_tile(tile_dir, sample_tile_info, show_progress=False)

    def test_process_tile_no_tiffs(self, tmp_path, sample_tile_info):
        """Test error when no TIFFs are found."""
        tile_dir = tmp_path / "empty_tile"
        downloads_dir = tile_dir / "downloads"
        downloads_dir.mkdir(parents=True)

        processor = TileProcessor()

        with pytest.raises(FileNotFoundError, match="No species TIFFs found"):
            processor.process_tile(tile_dir, sample_tile_info, show_progress=False)

    def test_process_tile_with_progress(self, tile_with_tiffs, capsys):
        """Test processing with progress bar enabled."""
        tile_dir, tile_info = tile_with_tiffs
        processor = TileProcessor()

        zarr_path, metadata = processor.process_tile(
            tile_dir, tile_info, show_progress=True
        )

        assert zarr_path.exists()
        assert metadata["species_count"] == 3


class TestTileProcessorValidateZarr:
    """Tests for validate_zarr method."""

    @pytest.fixture
    def valid_zarr_store(self, tmp_path):
        """Create a valid Zarr store for testing."""
        zarr_path = tmp_path / "valid.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")

        # Create biomass array
        store.create_array(
            "biomass",
            data=np.random.rand(3, 100, 100).astype(np.float32),
        )

        # Create species codes
        store.create_array(
            "species_codes",
            data=np.array(["0131", "0316", "0202"]),
        )

        # Add required attributes
        store.attrs.update({
            "tile_id": "r000_c000",
            "crs": "EPSG:3857",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "bounds": [-8006582.0, 4852835.0, -7782990.0, 5076428.0],
        })

        return zarr_path

    def test_validate_zarr_valid_store(self, valid_zarr_store):
        """Test validation of a valid Zarr store."""
        processor = TileProcessor()

        result = processor.validate_zarr(valid_zarr_store)

        assert result is True

    def test_validate_zarr_missing_biomass(self, tmp_path):
        """Test validation fails without biomass array."""
        zarr_path = tmp_path / "invalid.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")

        store.create_array(
            "species_codes",
            data=np.array(["0131", "0316", "0202"]),
        )
        store.attrs.update({
            "tile_id": "r000_c000",
            "crs": "EPSG:3857",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "bounds": [-8006582.0, 4852835.0, -7782990.0, 5076428.0],
        })

        processor = TileProcessor()
        result = processor.validate_zarr(zarr_path)

        assert result is False

    def test_validate_zarr_missing_species_codes(self, tmp_path):
        """Test validation fails without species_codes array."""
        zarr_path = tmp_path / "invalid.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")

        store.create_array(
            "biomass",
            data=np.random.rand(3, 100, 100).astype(np.float32),
        )
        store.attrs.update({
            "tile_id": "r000_c000",
            "crs": "EPSG:3857",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "bounds": [-8006582.0, 4852835.0, -7782990.0, 5076428.0],
        })

        processor = TileProcessor()
        result = processor.validate_zarr(zarr_path)

        assert result is False

    def test_validate_zarr_missing_attributes(self, tmp_path):
        """Test validation fails without required attributes."""
        zarr_path = tmp_path / "invalid.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")

        store.create_array(
            "biomass",
            data=np.random.rand(3, 100, 100).astype(np.float32),
        )
        store.create_array(
            "species_codes",
            data=np.array(["0131", "0316", "0202"]),
        )
        # Missing attributes

        processor = TileProcessor()
        result = processor.validate_zarr(zarr_path)

        assert result is False

    def test_validate_zarr_dimension_mismatch(self, tmp_path):
        """Test validation fails when dimensions don't match."""
        zarr_path = tmp_path / "invalid.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")

        # Biomass has 3 species but species_codes has 2
        store.create_array(
            "biomass",
            data=np.random.rand(3, 100, 100).astype(np.float32),
        )
        store.create_array(
            "species_codes",
            data=np.array(["0131", "0316"]),  # Only 2 codes
        )
        store.attrs.update({
            "tile_id": "r000_c000",
            "crs": "EPSG:3857",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "bounds": [-8006582.0, 4852835.0, -7782990.0, 5076428.0],
        })

        processor = TileProcessor()
        result = processor.validate_zarr(zarr_path)

        assert result is False

    def test_validate_zarr_nonexistent_path(self, tmp_path):
        """Test validation fails for nonexistent path."""
        processor = TileProcessor()

        result = processor.validate_zarr(tmp_path / "nonexistent.zarr")

        assert result is False


class TestTileProcessorCleanupZarr:
    """Tests for cleanup_zarr method."""

    def test_cleanup_zarr_existing(self, tmp_path):
        """Test cleanup of existing Zarr store."""
        zarr_path = tmp_path / "to_cleanup.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")
        store.create_array("data", data=np.zeros((10, 10)))

        assert zarr_path.exists()

        processor = TileProcessor()
        processor.cleanup_zarr(zarr_path)

        assert not zarr_path.exists()

    def test_cleanup_zarr_nonexistent(self, tmp_path):
        """Test cleanup of nonexistent path doesn't error."""
        zarr_path = tmp_path / "nonexistent.zarr"

        processor = TileProcessor()
        # Should not raise an error
        processor.cleanup_zarr(zarr_path)


class TestModuleConstants:
    """Tests for module constants."""

    def test_chunk_size_format(self):
        """Test CHUNK_SIZE has correct format."""
        assert len(CHUNK_SIZE) == 3
        assert CHUNK_SIZE[0] == 1  # Single species per chunk
        assert CHUNK_SIZE[1] == 512
        assert CHUNK_SIZE[2] == 512

    def test_nodata_value(self):
        """Test NODATA_VALUE is correct."""
        assert NODATA_VALUE == -9999.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def sample_tile_info(self):
        """Create a sample TileInfo for testing."""
        return TileInfo(
            tile_id="r001_c001",
            row=1,
            col=1,
            bbox_3857=(-8006582.0, 4852835.0, -7782990.0, 5076428.0),
        )

    def test_single_species(self, tmp_path, sample_tile_info):
        """Test processing with only one species."""
        tile_dir = tmp_path / "single_species"
        downloads_dir = tile_dir / "downloads"
        downloads_dir.mkdir(parents=True)

        # Create one GeoTIFF
        height, width = 50, 50
        data = np.random.rand(height, width).astype(np.float32) * 100
        bbox = sample_tile_info.bbox_3857
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

        tiff_path = downloads_dir / "species_0131.tif"
        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(3857),
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        processor = TileProcessor()
        zarr_path, metadata = processor.process_tile(
            tile_dir, sample_tile_info, show_progress=False
        )

        assert metadata["species_count"] == 1

        store = zarr.open_group(str(zarr_path), mode="r")
        assert store["biomass"].shape[0] == 1

    def test_many_species(self, tmp_path, sample_tile_info):
        """Test processing with many species."""
        tile_dir = tmp_path / "many_species"
        downloads_dir = tile_dir / "downloads"
        downloads_dir.mkdir(parents=True)

        # Create 10 GeoTIFFs
        height, width = 50, 50
        bbox = sample_tile_info.bbox_3857
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

        for i in range(10):
            data = np.random.rand(height, width).astype(np.float32) * 100
            code = f"{i:04d}"
            tiff_path = downloads_dir / f"species_{code}.tif"
            with rasterio.open(
                tiff_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs=CRS.from_epsg(3857),
                transform=transform,
            ) as dst:
                dst.write(data, 1)

        processor = TileProcessor()
        zarr_path, metadata = processor.process_tile(
            tile_dir, sample_tile_info, show_progress=False
        )

        assert metadata["species_count"] == 10

    def test_all_nodata(self, tmp_path, sample_tile_info):
        """Test processing when data is all nodata."""
        tile_dir = tmp_path / "all_nodata"
        downloads_dir = tile_dir / "downloads"
        downloads_dir.mkdir(parents=True)

        height, width = 50, 50
        data = np.full((height, width), -9999.0, dtype=np.float32)
        bbox = sample_tile_info.bbox_3857
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

        tiff_path = downloads_dir / "species_0131.tif"
        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(3857),
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        processor = TileProcessor()
        zarr_path, metadata = processor.process_tile(
            tile_dir, sample_tile_info, show_progress=False
        )

        # Valid percent should be 0 since all data is nodata
        assert metadata["valid_percent"] == 0.0
