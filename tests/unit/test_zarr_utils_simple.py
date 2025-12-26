"""
Simplified tests for gridfia.utils.zarr_utils module focused on maximum coverage.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import rasterio
from rasterio.transform import from_bounds
import zarr
import zarr.storage

from gridfia.utils.zarr_utils import (
    create_expandable_zarr_from_base_raster,
    append_species_to_zarr,
    batch_append_species_from_dir,
    create_zarr_from_geotiffs,
    validate_zarr_store
)
from gridfia.exceptions import (
    InvalidZarrStructure, SpeciesNotFound, CalculationFailed,
    APIConnectionError, InvalidLocationConfig, DownloadError
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def simple_raster(temp_dir):
    """Create a simple test raster file."""
    raster_path = temp_dir / "test_raster.tif"

    # Create simple test data
    height, width = 50, 50
    data = np.random.rand(height, width).astype(np.float32) * 100
    data[data < 20] = 0  # Some no-data pixels

    # Spatial properties
    bounds = (-2000000, -1000000, -1900000, -900000)
    transform = from_bounds(*bounds, width, height)

    # Write raster
    with rasterio.open(
        str(raster_path),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs='ESRI:102039',
        transform=transform
    ) as dst:
        dst.write(data, 1)

    return raster_path


class TestCreateExpandableZarrFromBaseRaster:
    """Test zarr creation functionality."""

    def test_create_basic_zarr(self, temp_dir, simple_raster):
        """Test basic zarr creation."""
        zarr_path = temp_dir / "basic.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path
        )

        # Verify basic structure
        assert isinstance(result, zarr.Group)
        assert zarr_path.exists()
        assert 'biomass' in result
        assert 'species_codes' in result
        assert 'species_names' in result
        assert result.attrs['num_species'] == 1

    def test_create_with_custom_params(self, temp_dir, simple_raster):
        """Test zarr creation with custom parameters."""
        zarr_path = temp_dir / "custom.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path,
            max_species=5,
            chunk_size=(1, 25, 25),
            compression='zstd',
            compression_level=3
        )

        assert result['biomass'].shape[0] == 5
        assert result['biomass'].chunks == (1, 25, 25)

    def test_create_with_invalid_path(self, temp_dir):
        """Test error handling with invalid raster path."""
        zarr_path = temp_dir / "error.zarr"
        invalid_path = temp_dir / "nonexistent.tif"

        with pytest.raises(rasterio.RasterioIOError):
            create_expandable_zarr_from_base_raster(
                base_raster_path=invalid_path,
                zarr_path=zarr_path
            )


class TestAppendSpeciesToZarr:
    """Test species appending functionality."""

    @pytest.fixture
    def base_zarr_store(self, temp_dir, simple_raster):
        """Create base zarr for append tests."""
        zarr_path = temp_dir / "append_test.zarr"
        root = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path,
            max_species=10
        )
        return root, zarr_path

    @pytest.fixture
    def species_raster(self, temp_dir):
        """Create a species raster with matching properties."""
        raster_path = temp_dir / "species.tif"

        height, width = 50, 50
        data = np.random.rand(height, width).astype(np.float32) * 30

        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        with rasterio.open(
            str(raster_path),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='ESRI:102039',
            transform=transform
        ) as dst:
            dst.write(data, 1)

        return raster_path

    def test_append_species_success(self, base_zarr_store, species_raster):
        """Test successful species append."""
        root, zarr_path = base_zarr_store

        index = append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=species_raster,
            species_code='SP001',
            species_name='Test Species'
        )

        assert index == 1

        # Verify data was added
        store = zarr.storage.LocalStore(zarr_path)
        updated_root = zarr.open_group(store=store, mode='r')
        assert updated_root.attrs['num_species'] == 2
        assert updated_root['species_codes'][1] == 'SP001'

    def test_append_without_validation(self, base_zarr_store, species_raster):
        """Test append without validation."""
        root, zarr_path = base_zarr_store

        index = append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=species_raster,
            species_code='SP002',
            species_name='No Validation Species',
            validate_alignment=False
        )

        assert index == 1

    def test_append_with_transform_mismatch(self, base_zarr_store, temp_dir):
        """Test error with transform mismatch."""
        root, zarr_path = base_zarr_store

        # Create raster with different transform
        bad_raster = temp_dir / "bad_transform.tif"
        height, width = 50, 50
        data = np.random.rand(height, width).astype(np.float32) * 30

        # Different bounds
        bounds = (-1500000, -800000, -1400000, -700000)
        transform = from_bounds(*bounds, width, height)

        with rasterio.open(
            str(bad_raster),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='ESRI:102039',
            transform=transform
        ) as dst:
            dst.write(data, 1)

        with pytest.raises(InvalidZarrStructure, match="Transform mismatch"):
            append_species_to_zarr(
                zarr_path=zarr_path,
                species_raster_path=bad_raster,
                species_code='BAD',
                species_name='Bad Species'
            )


class TestBatchAppendSpeciesFromDir:
    """Test batch append functionality."""

    @pytest.fixture
    def batch_setup(self, temp_dir, simple_raster):
        """Create setup for batch tests."""
        # Create base zarr
        zarr_path = temp_dir / "batch.zarr"
        root = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path,
            max_species=10
        )

        # Create species directory with rasters
        species_dir = temp_dir / "species"
        species_dir.mkdir()

        species_mapping = {'SP001': 'Species 1', 'SP002': 'Species 2'}

        # Create species rasters
        height, width = 50, 50
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        for code in species_mapping:
            raster_path = species_dir / f"biomass_{code}.tif"
            data = np.random.rand(height, width).astype(np.float32) * 25

            with rasterio.open(
                str(raster_path),
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs='ESRI:102039',
                transform=transform
            ) as dst:
                dst.write(data, 1)

        return zarr_path, species_dir, species_mapping

    def test_batch_append_success(self, batch_setup):
        """Test successful batch append."""
        zarr_path, species_dir, species_mapping = batch_setup

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping=species_mapping
        )

        # Verify results
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 3  # 1 base + 2 species

    def test_batch_with_no_files(self, temp_dir, simple_raster):
        """Test batch append with empty directory."""
        zarr_path = temp_dir / "no_files.zarr"
        create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path
        )

        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=empty_dir,
            species_mapping={'SP001': 'Test'}
        )

        # Should remain unchanged
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 1


class TestCreateZarrFromGeotiffs:
    """Test zarr creation from geotiffs."""

    @pytest.fixture
    def geotiff_list(self, temp_dir):
        """Create list of geotiff files."""
        files = []
        codes = ['SP001', 'SP002']
        names = ['Species 1', 'Species 2']

        height, width = 40, 40
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        for i, (code, name) in enumerate(zip(codes, names)):
            file_path = temp_dir / f"{code}.tif"
            data = np.random.rand(height, width).astype(np.float32) * (30 + i * 10)

            with rasterio.open(
                str(file_path),
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs='ESRI:102039',
                transform=transform
            ) as dst:
                dst.write(data, 1)

            files.append(file_path)

        return files, codes, names

    def test_create_from_geotiffs_with_total(self, temp_dir, geotiff_list):
        """Test zarr creation from geotiffs including total."""
        files, codes, names = geotiff_list
        zarr_path = temp_dir / "from_geotiffs.zarr"

        create_zarr_from_geotiffs(
            output_zarr_path=zarr_path,
            geotiff_paths=files,
            species_codes=codes,
            species_names=names,
            include_total=True
        )

        # Verify zarr
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')

        assert root['biomass'].shape == (3, 40, 40)  # 2 species + 1 total
        assert root.attrs['num_species'] == 3
        assert root['species_codes'][0] == '0000'  # Total biomass

    def test_create_from_geotiffs_without_total(self, temp_dir, geotiff_list):
        """Test zarr creation without total."""
        files, codes, names = geotiff_list
        zarr_path = temp_dir / "no_total.zarr"

        create_zarr_from_geotiffs(
            output_zarr_path=zarr_path,
            geotiff_paths=files,
            species_codes=codes,
            species_names=names,
            include_total=False
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')

        assert root['biomass'].shape == (2, 40, 40)  # 2 species only
        assert root.attrs['num_species'] == 2

    def test_create_mismatched_lengths(self, temp_dir, geotiff_list):
        """Test error with mismatched input lengths."""
        files, codes, names = geotiff_list
        zarr_path = temp_dir / "mismatch.zarr"

        with pytest.raises(InvalidZarrStructure, match="must match"):
            create_zarr_from_geotiffs(
                output_zarr_path=zarr_path,
                geotiff_paths=files,
                species_codes=codes,
                species_names=names[:-1]  # One fewer name
            )


class TestValidateZarrStore:
    """Test zarr validation functionality."""

    def test_validate_basic_store(self, temp_dir, simple_raster):
        """Test validation of basic zarr store."""
        zarr_path = temp_dir / "validate.zarr"
        create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path
        )

        result = validate_zarr_store(zarr_path)

        assert result['path'] == str(zarr_path)
        assert result['shape'] == (350, 50, 50)  # Default max_species
        assert result['num_species'] == 1
        assert len(result['species']) == 1
        assert result['species'][0]['code'] == '0000'

    def test_validate_missing_metadata(self, temp_dir):
        """Test validation with missing metadata."""
        zarr_path = temp_dir / "minimal.zarr"

        # Create minimal zarr store
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('biomass', shape=(2, 30, 30), dtype='f4')

        result = validate_zarr_store(zarr_path)

        assert result['num_species'] == 0
        assert result['crs'] is None
        assert result['species'] == []

    def test_validate_empty_species_entries(self, temp_dir):
        """Test validation with empty species entries."""
        zarr_path = temp_dir / "empty_species.zarr"

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('biomass', shape=(3, 30, 30), dtype='f4')
        root.create_array('species_codes', shape=(3,), dtype='<U10', fill_value='')
        root.create_array('species_names', shape=(3,), dtype='<U100', fill_value='')
        root.attrs['num_species'] = 3

        # Only fill first entry
        root['species_codes'][0] = 'SP001'
        root['species_names'][0] = 'Valid Species'

        result = validate_zarr_store(zarr_path)

        # Should only include non-empty species
        assert len(result['species']) == 1
        assert result['species'][0]['code'] == 'SP001'


class TestZarrUtilsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_large_species_allocation(self, temp_dir, simple_raster):
        """Test creating zarr with large max_species."""
        zarr_path = temp_dir / "large.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path,
            max_species=1000
        )

        assert result['biomass'].shape[0] == 1000
        assert result['species_codes'].shape[0] == 1000

    def test_different_compression_algorithms(self, temp_dir, simple_raster):
        """Test different compression algorithms."""
        zarr_path = temp_dir / "compressed.zarr"

        # Test lz4 compression (default)
        result1 = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path,
            compression='lz4'
        )
        assert isinstance(result1, zarr.Group)

        # Test with different compression
        zarr_path2 = temp_dir / "zstd.zarr"
        result2 = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path2,
            compression='zstd'
        )
        assert isinstance(result2, zarr.Group)

    def test_string_vs_path_inputs(self, temp_dir, simple_raster):
        """Test string vs Path object inputs."""
        zarr_path_str = str(temp_dir / "string.zarr")

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=str(simple_raster),  # String path
            zarr_path=zarr_path_str  # String path
        )

        assert isinstance(result, zarr.Group)
        assert Path(zarr_path_str).exists()

    @patch('gridfia.utils.zarr_utils.console')
    def test_console_output(self, mock_console, temp_dir, simple_raster):
        """Test console output during operations."""
        zarr_path = temp_dir / "console.zarr"

        create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path
        )

        # Verify console.print was called
        assert mock_console.print.called
        call_args = [str(call[0][0]) for call in mock_console.print.call_args_list]
        assert any("Creating Zarr store" in arg for arg in call_args)

    def test_zarr_v3_compatibility(self, temp_dir, simple_raster):
        """Test Zarr v3 API compatibility."""
        zarr_path = temp_dir / "v3.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=simple_raster,
            zarr_path=zarr_path
        )

        # Verify we can open with v3 API
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert 'biomass' in root