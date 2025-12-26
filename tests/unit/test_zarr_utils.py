"""
Comprehensive tests for gridfia.utils.zarr_utils module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import rasterio
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS
import zarr
import zarr.storage
from rich.console import Console

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


class TestCreateExpandableZarrFromBaseRaster:
    """Test the create_expandable_zarr_from_base_raster function."""

    def test_create_zarr_success(self, temp_dir: Path, sample_raster: Path):
        """Test successful creation of expandable zarr store."""
        zarr_path = temp_dir / "test.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=10,
            chunk_size=(1, 50, 50),
            compression='lz4',
            compression_level=5
        )

        # Verify zarr group is created
        assert isinstance(result, zarr.Group)
        assert zarr_path.exists()

        # Check main data array
        assert 'biomass' in result
        biomass_array = result['biomass']
        assert biomass_array.shape == (10, 100, 100)  # max_species, height, width
        assert biomass_array.chunks == (1, 50, 50)
        assert biomass_array.dtype == np.float32

        # Check metadata arrays
        assert 'species_codes' in result
        assert 'species_names' in result
        assert result['species_codes'].shape == (10,)
        assert result['species_names'].shape == (10,)

        # Check attributes
        assert 'crs' in result.attrs
        assert 'transform' in result.attrs
        assert 'bounds' in result.attrs
        assert result.attrs['num_species'] == 1

        # Check first layer contains base data
        assert np.any(biomass_array[0, :, :] != 0)

        # Check metadata for first layer
        assert result['species_codes'][0] == '0000'
        assert result['species_names'][0] == 'Total Biomass'

    def test_create_zarr_custom_parameters(self, temp_dir: Path, sample_raster: Path):
        """Test zarr creation with custom parameters."""
        zarr_path = temp_dir / "custom.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=5,
            chunk_size=(2, 25, 25),
            compression='zstd',
            compression_level=3
        )

        biomass_array = result['biomass']
        assert biomass_array.shape == (5, 100, 100)
        assert biomass_array.chunks == (2, 25, 25)
        assert result['species_codes'].shape == (5,)
        assert result['species_names'].shape == (5,)

    def test_create_zarr_different_compression(self, temp_dir: Path, sample_raster: Path):
        """Test zarr creation with different compression algorithms."""
        zarr_path = temp_dir / "compressed.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            compression='zlib',
            compression_level=6
        )

        assert isinstance(result, zarr.Group)
        assert 'biomass' in result

    def test_create_zarr_invalid_raster_path(self, temp_dir: Path):
        """Test error handling with invalid raster path."""
        zarr_path = temp_dir / "test.zarr"
        invalid_raster = temp_dir / "nonexistent.tif"

        with pytest.raises(rasterio.RasterioIOError):
            create_expandable_zarr_from_base_raster(
                base_raster_path=invalid_raster,
                zarr_path=zarr_path
            )

    def test_create_zarr_path_as_string(self, temp_dir: Path, sample_raster: Path):
        """Test zarr creation with string paths."""
        zarr_path = str(temp_dir / "string_path.zarr")

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=str(sample_raster),
            zarr_path=zarr_path
        )

        assert isinstance(result, zarr.Group)
        assert Path(zarr_path).exists()

    @patch('gridfia.utils.zarr_utils.console')
    def test_console_output(self, mock_console, temp_dir: Path, sample_raster: Path):
        """Test console output during zarr creation."""
        zarr_path = temp_dir / "console_test.zarr"

        create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path
        )

        # Verify console.print was called
        assert mock_console.print.call_count >= 3
        call_args = [call[0][0] for call in mock_console.print.call_args_list]
        assert any("Creating Zarr store" in str(arg) for arg in call_args)


class TestAppendSpeciesToZarr:
    """Test the append_species_to_zarr function."""

    @pytest.fixture
    def base_zarr(self, temp_dir: Path, sample_raster: Path):
        """Create a base zarr store for testing append operations."""
        zarr_path = temp_dir / "base.zarr"
        return create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=5
        ), zarr_path

    @pytest.fixture
    def species_raster(self, temp_dir: Path):
        """Create a species raster file for testing."""
        raster_path = temp_dir / "species_001.tif"

        # Create sample data with same dimensions as sample_raster
        height, width = 100, 100
        data = np.random.rand(height, width) * 50
        data[data < 10] = 0

        # Same spatial properties as sample_raster
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
            transform=transform,
            nodata=None
        ) as dst:
            dst.write(data.astype(np.float32), 1)

        return raster_path

    def test_append_species_success(self, base_zarr, species_raster):
        """Test successful species append."""
        root, zarr_path = base_zarr

        result_index = append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=species_raster,
            species_code='SP001',
            species_name='Test Pine',
            validate_alignment=True
        )

        assert result_index == 1  # Second layer (after total biomass)

        # Verify data was added
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')

        assert root.attrs['num_species'] == 2
        assert root['species_codes'][1] == 'SP001'
        assert root['species_names'][1] == 'Test Pine'
        assert np.any(root['biomass'][1, :, :] != 0)

    def test_append_species_no_validation(self, base_zarr, species_raster):
        """Test species append without validation."""
        root, zarr_path = base_zarr

        result_index = append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=species_raster,
            species_code='SP002',
            species_name='Test Oak',
            validate_alignment=False
        )

        assert result_index == 1

        # Verify data was added
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root['species_codes'][1] == 'SP002'

    def test_append_species_transform_mismatch(self, base_zarr, temp_dir: Path):
        """Test error handling with transform mismatch."""
        root, zarr_path = base_zarr

        # Create raster with different transform
        raster_path = temp_dir / "mismatched.tif"
        height, width = 100, 100
        data = np.random.rand(height, width) * 50

        # Different bounds
        bounds = (-1500000, -800000, -1400000, -700000)
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
            dst.write(data.astype(np.float32), 1)

        with pytest.raises(InvalidZarrStructure, match="Transform mismatch"):
            append_species_to_zarr(
                zarr_path=zarr_path,
                species_raster_path=raster_path,
                species_code='SP003',
                species_name='Mismatched Species',
                validate_alignment=True
            )

    def test_append_species_bounds_mismatch(self, base_zarr, temp_dir: Path):
        """Test error handling with bounds/transform mismatch.

        Note: When bounds differ, transform also differs (transform is derived from bounds),
        so the transform check triggers first.
        """
        root, zarr_path = base_zarr

        # Create raster with different bounds - use different actual bounds
        raster_path = temp_dir / "bounds_mismatch.tif"
        height, width = 100, 100  # Same dimensions but different bounds
        data = np.random.rand(height, width) * 50

        # Different bounds from the base raster
        bounds = (-1500000, -800000, -1400000, -700000)
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
            dst.write(data.astype(np.float32), 1)

        # When bounds differ, transform also differs, so transform check triggers first
        with pytest.raises(InvalidZarrStructure, match="Transform mismatch"):
            append_species_to_zarr(
                zarr_path=zarr_path,
                species_raster_path=raster_path,
                species_code='SP004',
                species_name='Bounds Mismatch Species',
                validate_alignment=True
            )

    @patch('gridfia.utils.zarr_utils.console')
    def test_append_species_crs_warning(self, mock_console, base_zarr, temp_dir: Path):
        """Test CRS mismatch warning."""
        root, zarr_path = base_zarr

        # Create raster with different CRS
        raster_path = temp_dir / "crs_mismatch.tif"
        height, width = 100, 100
        data = np.random.rand(height, width) * 50

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
            crs='EPSG:4326',  # Different CRS
            transform=transform
        ) as dst:
            dst.write(data.astype(np.float32), 1)

        append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=raster_path,
            species_code='SP005',
            species_name='CRS Warning Species',
            validate_alignment=True
        )

        # Check for warning message
        call_args = [str(call[0][0]) for call in mock_console.print.call_args_list]
        assert any("Warning: CRS mismatch" in arg for arg in call_args)

    def test_append_multiple_species(self, base_zarr, species_raster):
        """Test appending multiple species sequentially."""
        root, zarr_path = base_zarr

        # Append first species
        index1 = append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=species_raster,
            species_code='SP001',
            species_name='First Pine'
        )

        # Append second species
        index2 = append_species_to_zarr(
            zarr_path=zarr_path,
            species_raster_path=species_raster,
            species_code='SP002',
            species_name='Second Oak'
        )

        assert index1 == 1
        assert index2 == 2

        # Verify final state
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 3


class TestBatchAppendSpeciesFromDir:
    """Test the batch_append_species_from_dir function."""

    @pytest.fixture
    def base_zarr_for_batch(self, temp_dir: Path, sample_raster: Path):
        """Create a base zarr store for batch testing."""
        zarr_path = temp_dir / "batch_test.zarr"
        return create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=10
        ), zarr_path

    @pytest.fixture
    def species_directory(self, temp_dir: Path):
        """Create directory with multiple species raster files."""
        species_dir = temp_dir / "species_rasters"
        species_dir.mkdir()

        # Create species mapping
        species_mapping = {
            'SP001': 'Douglas Fir',
            'SP002': 'Ponderosa Pine',
            'SP003': 'White Oak'
        }

        # Create raster files
        height, width = 100, 100
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        for code, name in species_mapping.items():
            raster_path = species_dir / f"biomass_{code}.tif"
            data = np.random.rand(height, width) * 30
            data[data < 5] = 0

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
                dst.write(data.astype(np.float32), 1)

        return species_dir, species_mapping

    def test_batch_append_success(self, base_zarr_for_batch, species_directory):
        """Test successful batch append operation."""
        root, zarr_path = base_zarr_for_batch
        species_dir, species_mapping = species_directory

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping=species_mapping,
            pattern="*.tif",
            validate_alignment=True
        )

        # Verify all species were added
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 4  # 1 + 3 species

        # Check species codes and names
        added_codes = []
        for i in range(1, 4):  # Skip total biomass at index 0
            code = root['species_codes'][i]
            if code:
                added_codes.append(str(code))

        assert len(added_codes) == 3
        assert all(code in species_mapping.keys() for code in added_codes)

    def test_batch_append_no_files_found(self, base_zarr_for_batch, temp_dir: Path):
        """Test batch append with no matching files."""
        root, zarr_path = base_zarr_for_batch
        empty_dir = temp_dir / "empty_dir"
        empty_dir.mkdir()

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=empty_dir,
            species_mapping={'SP001': 'Test Species'},
            pattern="*.tif"
        )

        # Should remain unchanged
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 1

    def test_batch_append_custom_pattern(self, base_zarr_for_batch, species_directory):
        """Test batch append with custom file pattern."""
        root, zarr_path = base_zarr_for_batch
        species_dir, species_mapping = species_directory

        # Create additional file with different extension
        other_file = species_dir / "SP001_data.img"
        other_file.write_text("dummy")

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping=species_mapping,
            pattern="*.tif"  # Should only match .tif files
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 4  # Only .tif files processed

    def test_batch_append_no_validation(self, base_zarr_for_batch, species_directory):
        """Test batch append without alignment validation."""
        root, zarr_path = base_zarr_for_batch
        species_dir, species_mapping = species_directory

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping=species_mapping,
            validate_alignment=False
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 4

    @patch('gridfia.utils.zarr_utils.console')
    def test_batch_append_unknown_species(self, mock_console, base_zarr_for_batch, temp_dir: Path):
        """Test batch append with files containing unknown species codes."""
        root, zarr_path = base_zarr_for_batch
        species_dir = temp_dir / "unknown_species"
        species_dir.mkdir()

        # Create file with unknown species code
        unknown_file = species_dir / "biomass_UNKNOWN.tif"
        height, width = 100, 100
        data = np.random.rand(height, width) * 30
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        with rasterio.open(
            str(unknown_file),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='ESRI:102039',
            transform=transform
        ) as dst:
            dst.write(data.astype(np.float32), 1)

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping={'SP001': 'Known Species'},
            pattern="*.tif"
        )

        # Check for warning message
        call_args = [str(call[0][0]) for call in mock_console.print.call_args_list]
        assert any("Could not find species code" in arg for arg in call_args)

    @patch('gridfia.utils.zarr_utils.console')
    def test_batch_append_error_handling(self, mock_console, base_zarr_for_batch, species_directory, temp_dir: Path):
        """Test error handling during batch append."""
        root, zarr_path = base_zarr_for_batch
        species_dir, species_mapping = species_directory

        # Create a file with invalid raster data to trigger an error
        invalid_file = species_dir / "SP001_invalid.tif"
        invalid_file.write_text("This is not a valid TIFF file")

        # Adjust species mapping to include the invalid file
        species_mapping['SP001_invalid'] = 'Invalid Species'

        # Should handle errors gracefully and continue
        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping=species_mapping,
            pattern="*invalid.tif"
        )

        # Should have printed error messages
        assert mock_console.print.called


class TestCreateZarrFromGeotiffs:
    """Test the create_zarr_from_geotiffs function."""

    @pytest.fixture
    def geotiff_files(self, temp_dir: Path):
        """Create multiple GeoTIFF files for testing."""
        files = []
        codes = ['SP001', 'SP002', 'SP003']
        names = ['Douglas Fir', 'Ponderosa Pine', 'White Oak']

        height, width = 80, 80
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        for i, (code, name) in enumerate(zip(codes, names)):
            file_path = temp_dir / f"{code}.tif"
            # Create distinct data patterns for each species
            data = np.random.rand(height, width) * (30 + i * 10)
            data[data < 10] = 0

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
                dst.write(data.astype(np.float32), 1)

            files.append(file_path)

        return files, codes, names

    def test_create_zarr_from_geotiffs_with_total(self, temp_dir: Path, geotiff_files):
        """Test creating zarr from geotiffs including total biomass."""
        files, codes, names = geotiff_files
        zarr_path = temp_dir / "from_geotiffs.zarr"

        create_zarr_from_geotiffs(
            output_zarr_path=zarr_path,
            geotiff_paths=files,
            species_codes=codes,
            species_names=names,
            include_total=True
        )

        # Verify zarr store
        assert zarr_path.exists()
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')

        # Check dimensions (3 species + 1 total)
        assert root['biomass'].shape == (4, 80, 80)
        assert root.attrs['num_species'] == 4

        # Check total biomass layer (index 0)
        assert root['species_codes'][0] == '0000'
        assert root['species_names'][0] == 'Total Biomass'

        # Check individual species
        for i in range(1, 4):
            assert root['species_codes'][i] == codes[i-1]
            assert root['species_names'][i] == names[i-1]

        # Verify total biomass is sum of species
        total_layer = np.array(root['biomass'][0, :, :])
        species_sum = np.sum([np.array(root['biomass'][i, :, :]) for i in range(1, 4)], axis=0)
        np.testing.assert_array_almost_equal(total_layer, species_sum)

    def test_create_zarr_from_geotiffs_without_total(self, temp_dir: Path, geotiff_files):
        """Test creating zarr from geotiffs without total biomass."""
        files, codes, names = geotiff_files
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

        # Check dimensions (3 species only)
        assert root['biomass'].shape == (3, 80, 80)
        assert root.attrs['num_species'] == 3

        # Check species data starts at index 0
        for i in range(3):
            assert root['species_codes'][i] == codes[i]
            assert root['species_names'][i] == names[i]

    def test_create_zarr_custom_parameters(self, temp_dir: Path, geotiff_files):
        """Test zarr creation with custom parameters."""
        files, codes, names = geotiff_files
        zarr_path = temp_dir / "custom_params.zarr"

        create_zarr_from_geotiffs(
            output_zarr_path=zarr_path,
            geotiff_paths=files,
            species_codes=codes,
            species_names=names,
            chunk_size=(2, 40, 40),
            compression='zstd',
            compression_level=3,
            include_total=False
        )

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')

        assert root['biomass'].chunks == (2, 40, 40)

    def test_create_zarr_mismatched_lengths(self, temp_dir: Path, geotiff_files):
        """Test error handling with mismatched list lengths."""
        files, codes, names = geotiff_files
        zarr_path = temp_dir / "mismatch.zarr"

        # Remove one name to create mismatch
        with pytest.raises(InvalidZarrStructure, match="must match"):
            create_zarr_from_geotiffs(
                output_zarr_path=zarr_path,
                geotiff_paths=files,
                species_codes=codes,
                species_names=names[:-1]  # One fewer name
            )

    def test_create_zarr_dimension_mismatch(self, temp_dir: Path, geotiff_files):
        """Test error handling with dimension mismatch."""
        files, codes, names = geotiff_files

        # Create file with different dimensions
        mismatched_file = temp_dir / "mismatched.tif"
        height, width = 50, 50  # Different size
        data = np.random.rand(height, width) * 30
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        with rasterio.open(
            str(mismatched_file),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='ESRI:102039',
            transform=transform
        ) as dst:
            dst.write(data.astype(np.float32), 1)

        zarr_path = temp_dir / "dimension_mismatch.zarr"

        with pytest.raises(InvalidZarrStructure, match="Dimension mismatch"):
            create_zarr_from_geotiffs(
                output_zarr_path=zarr_path,
                geotiff_paths=[files[0], mismatched_file],
                species_codes=['SP001', 'SP002'],
                species_names=['Species 1', 'Species 2']
            )

    def test_create_zarr_transform_mismatch(self, temp_dir: Path, geotiff_files):
        """Test error handling with transform mismatch."""
        files, codes, names = geotiff_files

        # Create file with different transform
        mismatched_file = temp_dir / "transform_mismatch.tif"
        height, width = 80, 80
        data = np.random.rand(height, width) * 30

        # Different bounds
        bounds = (-1500000, -800000, -1400000, -700000)
        transform = from_bounds(*bounds, width, height)

        with rasterio.open(
            str(mismatched_file),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='ESRI:102039',
            transform=transform
        ) as dst:
            dst.write(data.astype(np.float32), 1)

        zarr_path = temp_dir / "transform_mismatch.zarr"

        with pytest.raises(InvalidZarrStructure, match="Transform mismatch"):
            create_zarr_from_geotiffs(
                output_zarr_path=zarr_path,
                geotiff_paths=[files[0], mismatched_file],
                species_codes=['SP001', 'SP002'],
                species_names=['Species 1', 'Species 2']
            )


class TestValidateZarrStore:
    """Test the validate_zarr_store function."""

    @pytest.fixture
    def complete_zarr_store(self, temp_dir: Path, sample_raster: Path):
        """Create a complete zarr store for validation testing."""
        zarr_path = temp_dir / "complete.zarr"

        # Create store with multiple species
        root = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=5
        )

        # Add a few more species
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r+')

        # Simulate adding species data
        root['species_codes'][1] = 'SP001'
        root['species_names'][1] = 'Douglas Fir'
        root['species_codes'][2] = 'SP002'
        root['species_names'][2] = 'Ponderosa Pine'
        root.attrs['num_species'] = 3

        return zarr_path

    def test_validate_complete_store(self, complete_zarr_store):
        """Test validation of complete zarr store."""
        result = validate_zarr_store(complete_zarr_store)

        # Check basic info
        assert result['path'] == str(complete_zarr_store)
        assert result['shape'] == (5, 100, 100)  # max_species, height, width
        assert result['chunks'] == (1, 1000, 1000)  # Default chunk size
        assert result['dtype'] == 'float32'
        assert result['num_species'] == 3
        assert result['crs'] is not None
        assert result['bounds'] is not None

        # Check species information
        assert len(result['species']) == 3
        species_codes = [s['code'] for s in result['species']]
        assert '0000' in species_codes  # Total biomass
        assert 'SP001' in species_codes
        assert 'SP002' in species_codes

        # Check species details
        total_species = next(s for s in result['species'] if s['code'] == '0000')
        assert total_species['name'] == 'Total Biomass'
        assert total_species['index'] == 0

    def test_validate_minimal_store(self, temp_dir: Path, sample_raster: Path):
        """Test validation of minimal zarr store."""
        zarr_path = temp_dir / "minimal.zarr"

        # Create minimal store
        create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path
        )

        result = validate_zarr_store(zarr_path)

        assert result['num_species'] == 1
        assert len(result['species']) == 1
        assert result['species'][0]['code'] == '0000'

    def test_validate_store_missing_metadata(self, temp_dir: Path):
        """Test validation of zarr store with missing metadata."""
        zarr_path = temp_dir / "incomplete.zarr"

        # Create store with minimal metadata
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')

        # Create basic array without full metadata
        root.create_array(
            'biomass',
            shape=(3, 50, 50),
            chunks=(1, 50, 50),
            dtype='f4'
        )

        result = validate_zarr_store(zarr_path)

        # Should handle missing attributes gracefully
        assert result['num_species'] == 0
        assert result['crs'] is None
        assert result['bounds'] is None
        assert result['species'] == []

    def test_validate_store_empty_species(self, temp_dir: Path):
        """Test validation with empty species entries."""
        zarr_path = temp_dir / "empty_species.zarr"

        # Create store with empty species entries
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')

        root.create_array('biomass', shape=(3, 50, 50), dtype='f4')
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

    def test_validate_store_no_species_arrays(self, temp_dir: Path):
        """Test validation of store without species metadata arrays."""
        zarr_path = temp_dir / "no_species_arrays.zarr"

        # Create store without species arrays
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')

        root.create_array('biomass', shape=(2, 50, 50), dtype='f4')
        root.attrs['num_species'] = 2

        result = validate_zarr_store(zarr_path)

        # Should handle missing species arrays
        assert result['species'] == []

    def test_validate_store_string_path(self, complete_zarr_store):
        """Test validation with string path input."""
        result = validate_zarr_store(str(complete_zarr_store))

        assert result['path'] == str(complete_zarr_store)
        assert result['num_species'] == 3


class TestZarrUtilsEdgeCases:
    """Test edge cases and error conditions."""

    def test_zarr_v3_compatibility(self, temp_dir: Path, sample_raster: Path):
        """Test compatibility with Zarr v3 API."""
        zarr_path = temp_dir / "v3_test.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path
        )

        # Verify we can open with v3 API
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')

        assert 'biomass' in root
        assert hasattr(root['biomass'], 'compressors')

    def test_large_max_species_allocation(self, temp_dir: Path, sample_raster: Path):
        """Test creating zarr with large max_species value."""
        zarr_path = temp_dir / "large_allocation.zarr"

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=1000
        )

        assert result['biomass'].shape[0] == 1000
        assert result['species_codes'].shape[0] == 1000
        assert result['species_names'].shape[0] == 1000

    def test_different_data_types(self, temp_dir: Path):
        """Test zarr creation with different data types."""
        zarr_path = temp_dir / "int_type.zarr"
        raster_path = temp_dir / "int_raster.tif"

        # Create integer raster
        height, width = 50, 50
        data = np.random.randint(0, 100, (height, width))
        bounds = (-2000000, -1000000, -1900000, -900000)
        transform = from_bounds(*bounds, width, height)

        with rasterio.open(
            str(raster_path),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.int32,
            crs='ESRI:102039',
            transform=transform
        ) as dst:
            dst.write(data.astype(np.int32), 1)

        result = create_expandable_zarr_from_base_raster(
            base_raster_path=raster_path,
            zarr_path=zarr_path
        )

        assert result['biomass'].dtype == np.int32

    def test_progress_tracking_batch_operations(self, temp_dir: Path, sample_raster: Path):
        """Test progress tracking during batch operations."""
        zarr_path = temp_dir / "progress_test.zarr"

        # Create base zarr
        create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path
        )

        # Create multiple raster files
        species_dir = temp_dir / "species"
        species_dir.mkdir()

        files = []
        for i in range(3):  # Reduced number for faster test
            file_path = species_dir / f"SP{i:03d}.tif"
            height, width = 100, 100
            data = np.random.rand(height, width) * 30
            bounds = (-2000000, -1000000, -1900000, -900000)
            transform = from_bounds(*bounds, width, height)

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
                dst.write(data.astype(np.float32), 1)

            files.append(file_path)

        # Test batch append with progress
        species_mapping = {f'SP{i:03d}': f'Species {i}' for i in range(3)}

        batch_append_species_from_dir(
            zarr_path=zarr_path,
            raster_dir=species_dir,
            species_mapping=species_mapping
        )

        # Verify final state
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r')
        assert root.attrs['num_species'] == 4  # 1 base + 3 species

    def test_memory_efficiency_large_arrays(self, temp_dir: Path):
        """Test memory efficiency with larger arrays."""
        zarr_path = temp_dir / "large_array.zarr"
        raster_path = temp_dir / "large_raster.tif"

        # Create larger test raster
        height, width = 1000, 1000
        data = np.random.rand(height, width).astype(np.float32) * 100
        bounds = (-2000000, -1000000, -1000000, 0)
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

        # Create zarr with appropriate chunking
        result = create_expandable_zarr_from_base_raster(
            base_raster_path=raster_path,
            zarr_path=zarr_path,
            chunk_size=(1, 500, 500)
        )

        assert result['biomass'].shape == (350, height, width)  # Default max_species
        assert result['biomass'].chunks == (1, 500, 500)

        # Verify data integrity
        original_data = data
        zarr_data = np.array(result['biomass'][0, :, :])
        np.testing.assert_array_equal(original_data, zarr_data)


class TestSafeOpenZarrBiomass:
    """Test the safe_open_zarr_biomass utility function from examples.utils."""

    def test_safe_open_zarr_array_format(self, temp_dir: Path, sample_raster: Path):
        """Test opening legacy zarr array format."""
        from gridfia.examples.utils import safe_open_zarr_biomass

        zarr_path = temp_dir / "array_format.zarr"

        # Create legacy array format (single array, not group)
        z = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(3, 100, 100),
            chunks=(1, 50, 50),
            dtype='float32'
        )

        # Add some test data
        test_data = np.random.rand(3, 100, 100).astype(np.float32)
        z[:] = test_data

        # Test opening with utility function
        root, biomass = safe_open_zarr_biomass(zarr_path)

        # For array format, root and biomass should be the same
        assert root is biomass
        assert biomass.shape == (3, 100, 100)
        np.testing.assert_array_equal(biomass[:], test_data)

    def test_safe_open_zarr_group_format(self, temp_dir: Path, sample_raster: Path):
        """Test opening group-based zarr format."""
        from gridfia.examples.utils import safe_open_zarr_biomass

        zarr_path = temp_dir / "group_format.zarr"

        # Create group-based format
        result = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path
        )

        # Test opening with utility function
        root, biomass = safe_open_zarr_biomass(zarr_path)

        # Should return group and biomass array separately
        assert root != biomass
        assert hasattr(root, 'attrs')  # Group has attributes
        assert 'biomass' in root
        assert biomass.shape[1:] == (100, 100)  # From sample raster

    def test_safe_open_zarr_missing_biomass_array(self, temp_dir: Path):
        """Test error handling when biomass array is missing from group."""
        from gridfia.examples.utils import safe_open_zarr_biomass

        zarr_path = temp_dir / "no_biomass.zarr"

        # Create group without biomass array
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('other_data', shape=(10, 10), dtype='f4')

        # Should raise ValueError (KeyError is caught and wrapped in ValueError)
        with pytest.raises(ValueError, match="'biomass' array not found"):
            safe_open_zarr_biomass(zarr_path)

    def test_safe_open_zarr_nonexistent_path(self, temp_dir: Path):
        """Test error handling with nonexistent path."""
        from gridfia.examples.utils import safe_open_zarr_biomass

        nonexistent_path = temp_dir / "does_not_exist.zarr"

        # Should raise ValueError (from examples/utils.py which still uses ValueError)
        with pytest.raises(ValueError, match="Cannot open zarr store"):
            safe_open_zarr_biomass(nonexistent_path)


class TestZarrStoreClass:
    """
    Comprehensive tests for the ZarrStore class.

    Tests cover all public methods and properties:
    - from_path() class method
    - is_valid_store() class method
    - open() context manager
    - biomass property
    - species_codes, species_names properties
    - crs, transform, bounds properties
    - num_species, shape, height, width properties
    - get_species_index() method
    - get_species_layer() method
    - get_species_info() method
    - get_extent() method
    - summary() method
    - close() method
    """

    from gridfia.utils.zarr_utils import ZarrStore

    @pytest.fixture
    def zarr_store_path(self, temp_dir: Path, sample_raster: Path):
        """Create a complete Zarr store for ZarrStore testing."""
        zarr_path = temp_dir / "zarr_store_test.zarr"

        # Create store with multiple species using the existing function
        root = create_expandable_zarr_from_base_raster(
            base_raster_path=sample_raster,
            zarr_path=zarr_path,
            max_species=10
        )

        # Add additional species data
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r+')

        # Add test species
        np.random.seed(42)
        height, width = 100, 100

        root['biomass'][1, :, :] = np.random.rand(height, width) * 50
        root['biomass'][2, :, :] = np.random.rand(height, width) * 30
        root['biomass'][3, :, :] = np.random.rand(height, width) * 20

        root['species_codes'][1] = '0202'
        root['species_names'][1] = 'Douglas-fir'
        root['species_codes'][2] = '0122'
        root['species_names'][2] = 'Ponderosa Pine'
        root['species_codes'][3] = '0746'
        root['species_names'][3] = 'Quaking Aspen'
        root.attrs['num_species'] = 4

        return zarr_path

    # Tests for from_path() class method

    def test_from_path_success(self, zarr_store_path: Path):
        """Test successful ZarrStore creation from path."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)

        assert store is not None
        assert store.path == zarr_store_path
        assert store.num_species == 4

        store.close()

    def test_from_path_with_string(self, zarr_store_path: Path):
        """Test from_path accepts string paths."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(str(zarr_store_path))

        assert store is not None
        assert store.path == zarr_store_path

        store.close()

    def test_from_path_nonexistent(self, temp_dir: Path):
        """Test from_path raises FileNotFoundError for missing path."""
        from gridfia.utils.zarr_utils import ZarrStore

        nonexistent_path = temp_dir / "does_not_exist.zarr"

        with pytest.raises(FileNotFoundError, match="Zarr store not found"):
            ZarrStore.from_path(nonexistent_path)

    def test_from_path_invalid_zarr(self, temp_dir: Path):
        """Test from_path raises InvalidZarrStructure for invalid zarr."""
        from gridfia.utils.zarr_utils import ZarrStore

        # Create a directory that is not a valid zarr store
        invalid_path = temp_dir / "invalid.zarr"
        invalid_path.mkdir()
        (invalid_path / "some_file.txt").write_text("not a zarr store")

        with pytest.raises(InvalidZarrStructure):
            ZarrStore.from_path(invalid_path)

    def test_from_path_missing_biomass_array(self, temp_dir: Path):
        """Test from_path raises InvalidZarrStructure when biomass array missing."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "no_biomass.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('other_array', shape=(10, 10), dtype='f4')

        with pytest.raises(InvalidZarrStructure, match="missing required 'biomass' array"):
            ZarrStore.from_path(zarr_path)

    def test_from_path_read_write_mode(self, zarr_store_path: Path):
        """Test from_path with read/write mode."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path, mode='r+')

        assert store is not None
        # Verify we can access the store
        assert store.num_species >= 1

        store.close()

    # Tests for is_valid_store() class method

    def test_is_valid_store_true(self, zarr_store_path: Path):
        """Test is_valid_store returns True for valid store."""
        from gridfia.utils.zarr_utils import ZarrStore

        assert ZarrStore.is_valid_store(zarr_store_path) is True

    def test_is_valid_store_false_nonexistent(self, temp_dir: Path):
        """Test is_valid_store returns False for nonexistent path."""
        from gridfia.utils.zarr_utils import ZarrStore

        nonexistent_path = temp_dir / "nonexistent.zarr"
        assert ZarrStore.is_valid_store(nonexistent_path) is False

    def test_is_valid_store_false_no_biomass(self, temp_dir: Path):
        """Test is_valid_store returns False when biomass array missing."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "no_biomass_check.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('other_data', shape=(5, 5), dtype='f4')

        assert ZarrStore.is_valid_store(zarr_path) is False

    def test_is_valid_store_false_wrong_dimensions(self, temp_dir: Path):
        """Test is_valid_store returns False when biomass is not 3D."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "wrong_dims.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        # Create 2D array instead of 3D
        root.create_array('biomass', shape=(100, 100), dtype='f4')

        assert ZarrStore.is_valid_store(zarr_path) is False

    def test_is_valid_store_with_string_path(self, zarr_store_path: Path):
        """Test is_valid_store accepts string paths."""
        from gridfia.utils.zarr_utils import ZarrStore

        assert ZarrStore.is_valid_store(str(zarr_store_path)) is True

    # Tests for open() context manager

    def test_open_context_manager(self, zarr_store_path: Path):
        """Test open() context manager properly opens and closes store."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            assert store is not None
            assert store.num_species == 4
            # Store should be accessible inside context
            biomass = store.biomass
            assert biomass is not None

    def test_open_context_manager_closes_on_exit(self, zarr_store_path: Path):
        """Test open() context manager properly closes store on exit."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            pass

        # After exiting context, store should be closed
        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store.biomass

    def test_open_context_manager_closes_on_exception(self, zarr_store_path: Path):
        """Test open() context manager closes store even on exception."""
        from gridfia.utils.zarr_utils import ZarrStore

        store_ref = None
        try:
            with ZarrStore.open(zarr_store_path) as store:
                store_ref = store
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Store should still be closed after exception
        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store_ref.biomass

    def test_open_read_write_mode(self, zarr_store_path: Path):
        """Test open() context manager with read/write mode."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path, mode='r+') as store:
            assert store.num_species >= 1

    # Tests for biomass property

    def test_biomass_property(self, zarr_store_path: Path):
        """Test biomass property returns the biomass array."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            biomass = store.biomass

            assert biomass is not None
            assert isinstance(biomass, zarr.Array)
            assert biomass.ndim == 3

    def test_biomass_property_closed_store(self, zarr_store_path: Path):
        """Test biomass property raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store.biomass

    def test_biomass_data_integrity(self, zarr_store_path: Path):
        """Test biomass data can be read correctly."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            # Access first layer (total biomass)
            total_biomass = np.array(store.biomass[0, :, :])

            assert total_biomass.shape == (100, 100)
            assert total_biomass.dtype == np.float32

    # Tests for species_codes and species_names properties

    def test_species_codes_property(self, zarr_store_path: Path):
        """Test species_codes property returns list of codes."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            codes = store.species_codes

            assert isinstance(codes, list)
            assert len(codes) >= 4
            assert '0000' in codes  # Total biomass code
            assert '0202' in codes  # Douglas-fir
            assert '0122' in codes  # Ponderosa Pine
            assert '0746' in codes  # Quaking Aspen

    def test_species_names_property(self, zarr_store_path: Path):
        """Test species_names property returns list of names."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            names = store.species_names

            assert isinstance(names, list)
            assert len(names) >= 4
            assert 'Total Biomass' in names
            assert 'Douglas-fir' in names
            assert 'Ponderosa Pine' in names
            assert 'Quaking Aspen' in names

    def test_species_codes_caching(self, zarr_store_path: Path):
        """Test species_codes are cached after first access."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            codes1 = store.species_codes
            codes2 = store.species_codes

            # Should return same cached list
            assert codes1 is codes2

    def test_species_names_caching(self, zarr_store_path: Path):
        """Test species_names are cached after first access."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            names1 = store.species_names
            names2 = store.species_names

            # Should return same cached list
            assert names1 is names2

    def test_species_codes_closed_store(self, zarr_store_path: Path):
        """Test species_codes raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store.species_codes

    # Tests for crs, transform, bounds properties

    def test_crs_property(self, zarr_store_path: Path):
        """Test crs property returns CRS object."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            crs = store.crs

            assert crs is not None
            assert isinstance(crs, CRS)
            assert 'ESRI:102039' in crs.to_string() or '102039' in str(crs)

    def test_transform_property(self, zarr_store_path: Path):
        """Test transform property returns Affine transform."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            transform = store.transform

            assert transform is not None
            assert isinstance(transform, Affine)
            # Transform should have 6 parameters
            assert len(list(transform)[:6]) == 6

    def test_bounds_property(self, zarr_store_path: Path):
        """Test bounds property returns tuple of bounds."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            bounds = store.bounds

            assert bounds is not None
            assert isinstance(bounds, tuple)
            assert len(bounds) == 4
            # Bounds should be (left, bottom, right, top)
            left, bottom, right, top = bounds
            assert left < right
            assert bottom < top

    def test_crs_caching(self, zarr_store_path: Path):
        """Test CRS is cached after first access."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            crs1 = store.crs
            crs2 = store.crs

            assert crs1 is crs2

    def test_transform_caching(self, zarr_store_path: Path):
        """Test transform is cached after first access."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            transform1 = store.transform
            transform2 = store.transform

            assert transform1 is transform2

    def test_bounds_caching(self, zarr_store_path: Path):
        """Test bounds are cached after first access."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            bounds1 = store.bounds
            bounds2 = store.bounds

            assert bounds1 is bounds2

    def test_crs_closed_store(self, zarr_store_path: Path):
        """Test crs raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store.crs

    # Tests for num_species, shape, height, width properties

    def test_num_species_property(self, zarr_store_path: Path):
        """Test num_species property returns correct count."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            num_species = store.num_species

            assert num_species == 4

    def test_shape_property(self, zarr_store_path: Path):
        """Test shape property returns tuple of dimensions."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            shape = store.shape

            assert isinstance(shape, tuple)
            assert len(shape) == 3
            # Shape is (species, height, width)
            assert shape[0] == 10  # max_species
            assert shape[1] == 100  # height
            assert shape[2] == 100  # width

    def test_height_property(self, zarr_store_path: Path):
        """Test height property returns correct value."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            height = store.height

            assert height == 100

    def test_width_property(self, zarr_store_path: Path):
        """Test width property returns correct value."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            width = store.width

            assert width == 100

    def test_shape_consistency(self, zarr_store_path: Path):
        """Test height and width are consistent with shape."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            shape = store.shape

            assert store.height == shape[1]
            assert store.width == shape[2]

    def test_num_species_closed_store(self, zarr_store_path: Path):
        """Test num_species raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store.num_species

    # Tests for get_species_index() method

    def test_get_species_index_valid_code(self, zarr_store_path: Path):
        """Test get_species_index returns correct index for valid code."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            idx = store.get_species_index('0202')

            assert idx == 1  # Douglas-fir is at index 1

    def test_get_species_index_total_biomass(self, zarr_store_path: Path):
        """Test get_species_index returns 0 for total biomass."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            idx = store.get_species_index('0000')

            assert idx == 0

    def test_get_species_index_invalid_code(self, zarr_store_path: Path):
        """Test get_species_index raises SpeciesNotFound for invalid code."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            with pytest.raises(SpeciesNotFound, match="not found"):
                store.get_species_index('9999')

    def test_get_species_index_provides_available_species(self, zarr_store_path: Path):
        """Test SpeciesNotFound includes available species list."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            try:
                store.get_species_index('9999')
            except SpeciesNotFound as e:
                assert e.species_code == '9999'
                assert e.available_species is not None
                assert len(e.available_species) > 0

    # Tests for get_species_layer() method

    def test_get_species_layer_valid_code(self, zarr_store_path: Path):
        """Test get_species_layer returns correct data for valid code."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            layer = store.get_species_layer('0202')

            assert isinstance(layer, np.ndarray)
            assert layer.shape == (100, 100)

    def test_get_species_layer_total_biomass(self, zarr_store_path: Path):
        """Test get_species_layer returns total biomass layer."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            layer = store.get_species_layer('0000')

            assert layer.shape == (100, 100)
            # Total biomass should have non-zero values
            assert np.any(layer > 0)

    def test_get_species_layer_invalid_code(self, zarr_store_path: Path):
        """Test get_species_layer raises SpeciesNotFound for invalid code."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            with pytest.raises(SpeciesNotFound):
                store.get_species_layer('INVALID')

    def test_get_species_layer_data_values(self, zarr_store_path: Path):
        """Test get_species_layer returns actual data values."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            # Douglas-fir was set with random values * 50
            layer = store.get_species_layer('0202')

            # Should have values in expected range
            assert layer.min() >= 0
            assert layer.max() <= 50

    # Tests for get_species_info() method

    def test_get_species_info_returns_list(self, zarr_store_path: Path):
        """Test get_species_info returns list of species info dictionaries."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            info = store.get_species_info()

            assert isinstance(info, list)
            assert len(info) == 4  # num_species

    def test_get_species_info_dict_structure(self, zarr_store_path: Path):
        """Test get_species_info dictionaries have correct keys."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            info = store.get_species_info()

            for species_info in info:
                assert 'index' in species_info
                assert 'code' in species_info
                assert 'name' in species_info

    def test_get_species_info_content(self, zarr_store_path: Path):
        """Test get_species_info returns correct content."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            info = store.get_species_info()

            # Check first species (total biomass)
            total_info = next(s for s in info if s['code'] == '0000')
            assert total_info['index'] == 0
            assert total_info['name'] == 'Total Biomass'

            # Check Douglas-fir
            df_info = next(s for s in info if s['code'] == '0202')
            assert df_info['index'] == 1
            assert df_info['name'] == 'Douglas-fir'

    def test_get_species_info_closed_store(self, zarr_store_path: Path):
        """Test get_species_info raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            store.get_species_info()

    # Tests for get_extent() method

    def test_get_extent_returns_tuple(self, zarr_store_path: Path):
        """Test get_extent returns tuple of 4 values."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            extent = store.get_extent()

            assert isinstance(extent, tuple)
            assert len(extent) == 4

    def test_get_extent_format(self, zarr_store_path: Path):
        """Test get_extent returns (left, right, bottom, top) format."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            left, right, bottom, top = store.get_extent()

            # Left should be less than right
            assert left < right
            # For this CRS (transform.e is negative), bottom < top
            # The transform has negative e, so bottom should be less than top
            # Actually depends on the transform direction

    def test_get_extent_uses_transform(self, zarr_store_path: Path):
        """Test get_extent calculation uses transform correctly."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            extent = store.get_extent()
            transform = store.transform
            shape = store.shape

            # Verify extent calculation
            expected_left = transform.c
            expected_right = transform.c + shape[2] * transform.a

            assert extent[0] == expected_left
            assert extent[1] == expected_right

    def test_get_extent_closed_store(self, zarr_store_path: Path):
        """Test get_extent raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            store.get_extent()

    # Tests for summary() method

    def test_summary_returns_dict(self, zarr_store_path: Path):
        """Test summary returns dictionary with expected keys."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            summary = store.summary()

            assert isinstance(summary, dict)
            assert 'path' in summary
            assert 'shape' in summary
            assert 'chunks' in summary
            assert 'dtype' in summary
            assert 'num_species' in summary
            assert 'crs' in summary
            assert 'bounds' in summary
            assert 'transform' in summary
            assert 'species_codes' in summary
            assert 'species_names' in summary

    def test_summary_content(self, zarr_store_path: Path):
        """Test summary contains correct values."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            summary = store.summary()

            assert summary['path'] == str(zarr_store_path)
            assert summary['shape'] == (10, 100, 100)
            assert summary['num_species'] == 4
            assert len(summary['species_codes']) >= 4
            assert len(summary['species_names']) >= 4

    def test_summary_transform_format(self, zarr_store_path: Path):
        """Test summary transform is a list of 6 values."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            summary = store.summary()

            assert isinstance(summary['transform'], list)
            assert len(summary['transform']) == 6

    def test_summary_closed_store(self, zarr_store_path: Path):
        """Test summary raises error on closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            store.summary()

    # Tests for close() method

    def test_close_method(self, zarr_store_path: Path):
        """Test close method marks store as closed."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        assert store._closed is False

        store.close()
        assert store._closed is True

    def test_close_clears_caches(self, zarr_store_path: Path):
        """Test close method clears cached values."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)

        # Access properties to populate cache
        _ = store.species_codes
        _ = store.species_names
        _ = store.crs
        _ = store.transform
        _ = store.bounds

        store.close()

        # Verify caches are cleared
        assert store._species_codes is None
        assert store._species_names is None
        assert store._crs is None
        assert store._transform is None
        assert store._bounds is None

    def test_close_prevents_access(self, zarr_store_path: Path):
        """Test close prevents further access to store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        with pytest.raises(InvalidZarrStructure, match="Cannot access closed"):
            _ = store.biomass

    def test_close_idempotent(self, zarr_store_path: Path):
        """Test close can be called multiple times without error."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)

        # Calling close multiple times should not raise
        store.close()
        store.close()
        store.close()

    # Tests for additional properties

    def test_chunks_property(self, zarr_store_path: Path):
        """Test chunks property returns chunk dimensions."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            chunks = store.chunks

            assert chunks is not None
            assert isinstance(chunks, tuple)
            assert len(chunks) == 3

    def test_dtype_property(self, zarr_store_path: Path):
        """Test dtype property returns numpy dtype."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            dtype = store.dtype

            assert dtype is not None
            assert dtype == np.float32

    def test_attrs_property(self, zarr_store_path: Path):
        """Test attrs property returns dictionary of attributes."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            attrs = store.attrs

            assert isinstance(attrs, dict)
            assert 'crs' in attrs
            assert 'transform' in attrs
            assert 'bounds' in attrs
            assert 'num_species' in attrs

    def test_path_property(self, zarr_store_path: Path):
        """Test path property returns the store path."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            path = store.path

            assert path == zarr_store_path

    # Tests for __repr__ method

    def test_repr_open_store(self, zarr_store_path: Path):
        """Test __repr__ for open store."""
        from gridfia.utils.zarr_utils import ZarrStore

        with ZarrStore.open(zarr_store_path) as store:
            repr_str = repr(store)

            assert 'ZarrStore' in repr_str
            assert str(zarr_store_path) in repr_str
            assert 'shape=' in repr_str
            assert 'species=' in repr_str

    def test_repr_closed_store(self, zarr_store_path: Path):
        """Test __repr__ for closed store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.close()

        repr_str = repr(store)
        assert 'ZarrStore(closed)' in repr_str

    # Tests for __enter__ and __exit__ methods (context manager protocol)

    def test_enter_returns_self(self, zarr_store_path: Path):
        """Test __enter__ returns the store instance."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)

        result = store.__enter__()
        assert result is store

        store.__exit__(None, None, None)

    def test_exit_closes_store(self, zarr_store_path: Path):
        """Test __exit__ closes the store."""
        from gridfia.utils.zarr_utils import ZarrStore

        store = ZarrStore.from_path(zarr_store_path)
        store.__enter__()
        store.__exit__(None, None, None)

        assert store._closed is True

    # Edge case tests

    def test_store_without_species_arrays(self, temp_dir: Path):
        """Test ZarrStore handles store without species_codes/species_names arrays."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "no_species_arrays.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('biomass', shape=(3, 50, 50), dtype='f4')
        root.attrs['num_species'] = 3
        root.attrs['crs'] = 'EPSG:4326'
        root.attrs['transform'] = [1, 0, 0, 0, -1, 0]
        root.attrs['bounds'] = [0, 0, 50, 50]

        with ZarrStore.open(zarr_path) as zarr_store:
            # Should handle missing species arrays gracefully
            codes = zarr_store.species_codes
            names = zarr_store.species_names

            assert isinstance(codes, list)
            assert isinstance(names, list)

    def test_store_with_default_crs(self, temp_dir: Path):
        """Test ZarrStore uses default CRS when not specified."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "no_crs.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('biomass', shape=(3, 50, 50), dtype='f4')
        # No crs attribute set

        with ZarrStore.open(zarr_path) as zarr_store:
            crs = zarr_store.crs

            # Should default to EPSG:3857
            assert crs is not None
            assert '3857' in crs.to_string()

    def test_store_with_default_transform(self, temp_dir: Path):
        """Test ZarrStore uses default transform when not specified."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "no_transform.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('biomass', shape=(3, 50, 50), dtype='f4')
        # No transform attribute set

        with ZarrStore.open(zarr_path) as zarr_store:
            transform = zarr_store.transform

            # Should return identity or default transform
            assert transform is not None
            assert isinstance(transform, Affine)

    def test_store_bounds_calculated_from_transform(self, temp_dir: Path):
        """Test ZarrStore calculates bounds from transform when not stored."""
        from gridfia.utils.zarr_utils import ZarrStore

        zarr_path = temp_dir / "no_bounds.zarr"
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='w')
        root.create_array('biomass', shape=(3, 50, 50), dtype='f4')
        root.attrs['transform'] = [30, 0, -2000000, 0, -30, -900000]
        # No bounds attribute set

        with ZarrStore.open(zarr_path) as zarr_store:
            bounds = zarr_store.bounds

            # Should calculate bounds from transform
            assert bounds is not None
            assert isinstance(bounds, tuple)
            assert len(bounds) == 4