"""
Comprehensive tests for gridfia.utils.zarr_utils module.
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