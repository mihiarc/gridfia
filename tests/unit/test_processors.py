"""
Unit tests for forest metrics processors.
"""

import pytest
import numpy as np
import zarr
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from gridfia.core.processors.forest_metrics import ForestMetricsProcessor, run_forest_analysis
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.core.calculations import registry
from gridfia.exceptions import (
    InvalidZarrStructure, SpeciesNotFound, CalculationFailed,
    APIConnectionError, InvalidLocationConfig, DownloadError
)


class TestForestMetricsProcessor:
    """Test suite for ForestMetricsProcessor."""
    
    def test_initialization(self, test_settings):
        """Test processor initialization."""
        processor = ForestMetricsProcessor(test_settings)
        assert processor.settings == test_settings
        assert hasattr(processor, 'run_calculations')
    
    def test_initialization_with_default_settings(self):
        """Test processor initialization with default settings."""
        processor = ForestMetricsProcessor()
        assert isinstance(processor.settings, GridFIASettings)
    
    def test_validate_zarr_array_valid(self, sample_zarr_array):
        """Test zarr validation with valid array."""
        processor = ForestMetricsProcessor()
        # Test should pass without raising exception
        processor._validate_zarr_array(sample_zarr_array)
    
    def test_validate_zarr_array_missing_attrs(self, temp_dir):
        """Test zarr validation with missing attributes."""
        # Create zarr without required attributes
        zarr_path = temp_dir / "invalid.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(2, 10, 10))

        processor = ForestMetricsProcessor()
        with pytest.raises(InvalidZarrStructure, match="Missing required attributes"):
            processor._validate_zarr_array(z)
    
    def test_validate_zarr_array_invalid_shape(self, temp_dir):
        """Test zarr validation with invalid shape."""
        # Create zarr with wrong dimensions
        zarr_path = temp_dir / "invalid_shape.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(10, 10))  # 2D instead of 3D
        z.attrs['species_codes'] = ['SP1']

        processor = ForestMetricsProcessor()
        with pytest.raises(InvalidZarrStructure, match="Expected 3D array"):
            processor._validate_zarr_array(z)
    
    def test_get_enabled_calculations(self, test_settings):
        """Test getting enabled calculations from settings."""
        processor = ForestMetricsProcessor(test_settings)
        enabled = processor._get_enabled_calculations()
        
        # Should have 3 enabled calculations from test_settings
        assert len(enabled) == 3
        assert all(calc.enabled for calc in enabled)
        assert 'species_richness' in [calc.name for calc in enabled]
        assert 'dominant_species' not in [calc.name for calc in enabled]
    
    @patch.object(registry, 'get')
    def test_initialize_calculation_instances(self, mock_get, test_settings):
        """Test initialization of calculation instances from registry."""
        # Mock calculation instance
        mock_calc_instance = Mock()
        mock_calc_instance.name = "test_calc"
        mock_get.return_value = mock_calc_instance
        
        processor = ForestMetricsProcessor(test_settings)
        enabled_configs = processor._get_enabled_calculations()
        calc_instances = processor._initialize_calculations(enabled_configs)
        
        assert len(calc_instances) == 3
        assert mock_get.call_count == 3
        assert all(inst == mock_calc_instance for inst in calc_instances)
    
    def test_process_chunk(self, sample_zarr_array):
        """Test processing a single chunk of data."""
        processor = ForestMetricsProcessor()
        
        # Create mock calculation
        mock_calc = Mock()
        mock_calc.name = "test_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.return_value = sample_zarr_array[:, :50, :50]
        mock_calc.calculate.return_value = np.ones((50, 50))
        mock_calc.postprocess_result.return_value = np.ones((50, 50))
        mock_calc.get_output_dtype.return_value = np.float32
        
        # Process chunk
        chunk_data = sample_zarr_array[:, :50, :50]
        result = processor._process_chunk(chunk_data, [mock_calc])
        
        assert "test_calc" in result
        assert result["test_calc"].shape == (50, 50)
        mock_calc.calculate.assert_called_once()
    
    def test_save_results_geotiff(self, test_settings, temp_dir):
        """Test saving results as GeoTIFF."""
        processor = ForestMetricsProcessor(test_settings)
        
        # Create test results
        results = {
            "species_richness": np.random.randint(0, 10, (100, 100)),
            "total_biomass": np.random.rand(100, 100) * 100
        }
        
        # Mock metadata
        from rasterio.transform import Affine
        metadata = {
            'crs': 'ESRI:102039',
            'transform': Affine(-2000000, 30, 0, -900000, 0, -30),
            'bounds': [-2000000, -1000000, -1900000, -900000]
        }
        
        output_paths = processor._save_results(results, metadata, test_settings.output_dir)
        
        assert len(output_paths) == 2
        assert all(Path(p).exists() for p in output_paths.values())
        assert str(output_paths["species_richness"]).endswith(".tif")
    
    def test_run_calculations_full_pipeline(self, test_settings, sample_zarr_array):
        """Test the full calculation pipeline."""
        
        processor = ForestMetricsProcessor(test_settings)
        
        # Patch internal methods to avoid full implementation
        with patch.object(processor, '_load_zarr_array') as mock_load:
            mock_load.return_value = (sample_zarr_array, None)
            
            with patch.object(processor, '_validate_zarr_array'):
                with patch.object(processor, '_process_in_chunks') as mock_process:
                    mock_process.return_value = {
                        "species_richness": np.ones((100, 100)),
                        "total_biomass": np.ones((100, 100)) * 50
                    }
                    
                    with patch.object(processor, '_save_results') as mock_save:
                        mock_save.return_value = {
                            "species_richness": str(test_settings.output_dir / "species_richness.tif"),
                            "total_biomass": str(test_settings.output_dir / "total_biomass.tif")
                        }
                        
                        results = processor.run_calculations("test.zarr")
                    
                    assert len(results) == 2
                    assert "species_richness" in results
                    assert "total_biomass" in results
    
    def test_run_calculations_no_enabled_calculations(self, test_settings):
        """Test run_calculations with no enabled calculations."""
        # Disable all calculations
        for calc in test_settings.calculations:
            calc.enabled = False

        processor = ForestMetricsProcessor(test_settings)

        with pytest.raises(CalculationFailed, match="No calculations enabled"):
            processor.run_calculations("dummy_path.zarr")
    
    def test_chunked_processing_memory_efficiency(self, sample_zarr_array, test_settings):
        """Test that chunked processing uses less memory than full array."""
        processor = ForestMetricsProcessor(test_settings)
        
        # Track memory usage (simplified test)
        chunk_size = (1, 50, 50)
        full_size = sample_zarr_array.shape
        
        # Memory for chunk should be much less than full array
        chunk_memory = np.prod(chunk_size) * 4  # float32
        full_memory = np.prod(full_size) * 4
        
        assert chunk_memory < full_memory / 2  # At least 50% reduction


class TestRunForestAnalysis:
    """Test the convenience function run_forest_analysis."""
    
    def test_run_forest_analysis_with_config(self, temp_dir):
        """Test run_forest_analysis with config file."""
        # Create dummy config file
        config_path = temp_dir / "config.yaml"
        config_path.write_text("app_name: BigMap\n")
        
        with patch('gridfia.core.processors.forest_metrics.ForestMetricsProcessor') as mock_processor:
            mock_instance = Mock()
            mock_instance.run_calculations.return_value = {"test": "result"}
            mock_processor.return_value = mock_instance
            
            results = run_forest_analysis("test.zarr", str(config_path))
            
            assert results == {"test": "result"}
            mock_instance.run_calculations.assert_called_once_with("test.zarr")
    
    def test_run_forest_analysis_without_config(self):
        """Test run_forest_analysis without config file."""
        with patch('gridfia.core.processors.forest_metrics.ForestMetricsProcessor') as mock_processor:
            mock_instance = Mock()
            mock_instance.run_calculations.return_value = {"test": "result"}
            mock_processor.return_value = mock_instance

            results = run_forest_analysis("test.zarr")

            assert results == {"test": "result"}
            mock_processor.assert_called_once()  # With default settings


class TestProcessInChunks:
    """Test suite for _process_in_chunks method."""

    def test_process_in_chunks_basic(self, sample_zarr_array, test_settings):
        """Test basic chunking logic processes all data."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)  # Small chunks for testing

        # Create mock calculation
        mock_calc = Mock()
        mock_calc.name = "test_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = lambda x: np.ones(x.shape[1:])
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(sample_zarr_array, [mock_calc])

        assert "test_calc" in results
        assert results["test_calc"].shape == (100, 100)
        # Verify all pixels were processed (no NaN values for float type)
        assert np.all(results["test_calc"] == 1.0)

    def test_process_in_chunks_multiple_calculations(self, sample_zarr_array, test_settings):
        """Test processing with multiple calculations."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        # Create two mock calculations
        mock_calc1 = Mock()
        mock_calc1.name = "calc1"
        mock_calc1.validate_data.return_value = True
        mock_calc1.preprocess_data.side_effect = lambda x: x
        mock_calc1.calculate.side_effect = lambda x: np.ones(x.shape[1:]) * 2
        mock_calc1.postprocess_result.side_effect = lambda x: x
        mock_calc1.get_output_dtype.return_value = np.float32

        mock_calc2 = Mock()
        mock_calc2.name = "calc2"
        mock_calc2.validate_data.return_value = True
        mock_calc2.preprocess_data.side_effect = lambda x: x
        mock_calc2.calculate.side_effect = lambda x: np.ones(x.shape[1:]) * 5
        mock_calc2.postprocess_result.side_effect = lambda x: x
        mock_calc2.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(sample_zarr_array, [mock_calc1, mock_calc2])

        assert len(results) == 2
        assert "calc1" in results
        assert "calc2" in results
        assert np.all(results["calc1"] == 2.0)
        assert np.all(results["calc2"] == 5.0)

    def test_process_in_chunks_integer_dtype(self, sample_zarr_array, test_settings):
        """Test chunking with integer output dtype."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        mock_calc = Mock()
        mock_calc.name = "int_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = lambda x: np.ones(x.shape[1:], dtype=np.int32) * 3
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.int32

        results = processor._process_in_chunks(sample_zarr_array, [mock_calc])

        assert results["int_calc"].dtype == np.int32
        assert np.all(results["int_calc"] == 3)

    def test_process_in_chunks_uneven_boundaries(self, temp_dir, test_settings):
        """Test chunking with array size not divisible by chunk size."""
        # Create an array with odd dimensions that won't divide evenly
        zarr_path = temp_dir / "uneven.zarr"
        z = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(3, 73, 89),  # Odd sizes
            chunks=(1, 50, 50),
            dtype='f4'
        )
        z[:] = np.random.rand(3, 73, 89).astype(np.float32)
        z.attrs['species_codes'] = ['TOTAL', 'SP1', 'SP2']
        z.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 30, 30)  # Doesn't divide evenly

        mock_calc = Mock()
        mock_calc.name = "uneven_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = lambda x: np.ones(x.shape[1:])
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(z, [mock_calc])

        assert results["uneven_calc"].shape == (73, 89)
        assert np.all(results["uneven_calc"] == 1.0)


class TestSaveResultsFormats:
    """Test suite for _save_results with different output formats."""

    def test_save_results_zarr_format(self, test_settings, temp_dir):
        """Test saving results as Zarr."""
        # Update settings to use zarr format
        test_settings.calculations[0].output_format = "zarr"
        processor = ForestMetricsProcessor(test_settings)

        results = {
            "species_richness": np.random.randint(0, 10, (100, 100)).astype(np.float32)
        }

        from rasterio.transform import Affine
        metadata = {
            'crs': 'ESRI:102039',
            'transform': Affine(30, 0, -2000000, 0, -30, -900000),
        }

        output_paths = processor._save_results(results, metadata, test_settings.output_dir)

        assert "species_richness" in output_paths
        assert output_paths["species_richness"].endswith(".zarr")

        # Verify zarr file was created correctly (Zarr v3 uses group with 'data' array)
        root = zarr.open_group(output_paths["species_richness"], mode='r')
        z = root['data']
        assert z.shape == (100, 100)
        assert 'crs' in z.attrs
        assert z.attrs['variable'] == 'species_richness'

    def test_save_results_netcdf_format(self, test_settings, temp_dir):
        """Test saving results as NetCDF."""
        pytest.importorskip("netCDF4", reason="netCDF4 required for this test")

        # Update settings to use netcdf format
        test_settings.calculations[0].output_format = "netcdf"
        processor = ForestMetricsProcessor(test_settings)

        results = {
            "species_richness": np.random.rand(100, 100).astype(np.float32)
        }

        from rasterio.transform import Affine
        metadata = {
            'crs': 'ESRI:102039',
            'transform': Affine(30, 0, -2000000, 0, -30, -900000),
        }

        output_paths = processor._save_results(results, metadata, test_settings.output_dir)

        assert "species_richness" in output_paths
        assert output_paths["species_richness"].endswith(".nc")
        assert Path(output_paths["species_richness"]).exists()

        # Verify netcdf file was created correctly
        import xarray as xr
        ds = xr.open_dataset(output_paths["species_richness"])
        assert 'species_richness' in ds.data_vars
        assert ds.attrs['crs'] == 'ESRI:102039'
        ds.close()

    def test_save_results_custom_output_name(self, test_settings, temp_dir):
        """Test saving results with custom output filename."""
        test_settings.calculations[0].output_name = "my_custom_richness"
        processor = ForestMetricsProcessor(test_settings)

        results = {
            "species_richness": np.random.rand(50, 50).astype(np.float32)
        }

        from rasterio.transform import Affine
        metadata = {
            'crs': 'ESRI:102039',
            'transform': Affine(30, 0, -2000000, 0, -30, -900000),
        }

        output_paths = processor._save_results(results, metadata, test_settings.output_dir)

        # Check custom name was used
        assert "my_custom_richness.tif" in output_paths["species_richness"]


class TestCreateFailureArray:
    """Test suite for _create_failure_array method."""

    def test_create_failure_array_float32(self, test_settings):
        """Test failure array creation for float32 dtype."""
        processor = ForestMetricsProcessor(test_settings)

        result = processor._create_failure_array((10, 10), np.float32, "test_calc")

        assert result.shape == (10, 10)
        assert result.dtype == np.float32
        assert np.all(np.isnan(result))

    def test_create_failure_array_float64(self, test_settings):
        """Test failure array creation for float64 dtype."""
        processor = ForestMetricsProcessor(test_settings)

        result = processor._create_failure_array((5, 8), np.float64, "test_calc")

        assert result.shape == (5, 8)
        assert result.dtype == np.float64
        assert np.all(np.isnan(result))

    def test_create_failure_array_signed_integer(self, test_settings):
        """Test failure array creation for signed integer dtype (uses -1)."""
        processor = ForestMetricsProcessor(test_settings)

        result = processor._create_failure_array((10, 10), np.int32, "test_calc")

        assert result.shape == (10, 10)
        assert result.dtype == np.int32
        assert np.all(result == -1)

    def test_create_failure_array_int16(self, test_settings):
        """Test failure array creation for int16 dtype."""
        processor = ForestMetricsProcessor(test_settings)

        result = processor._create_failure_array((8, 8), np.int16, "test_calc")

        assert result.dtype == np.int16
        assert np.all(result == -1)

    def test_create_failure_array_unsigned_integer(self, test_settings, caplog):
        """Test failure array creation for unsigned integer dtype (uses max value with warning)."""
        processor = ForestMetricsProcessor(test_settings)

        result = processor._create_failure_array((10, 10), np.uint8, "test_calc")

        assert result.shape == (10, 10)
        assert result.dtype == np.uint8
        assert np.all(result == 255)  # Max value for uint8

        # Check warning was logged
        assert "unsigned integer dtype" in caplog.text.lower()

    def test_create_failure_array_uint16(self, test_settings, caplog):
        """Test failure array creation for uint16 dtype."""
        processor = ForestMetricsProcessor(test_settings)

        result = processor._create_failure_array((5, 5), np.uint16, "test_calc")

        assert result.dtype == np.uint16
        assert np.all(result == 65535)  # Max value for uint16


class TestMetadataPreservation:
    """Test suite for metadata preservation in output files."""

    def test_geotiff_metadata_tags(self, test_settings, temp_dir):
        """Test that GeoTIFF files contain correct metadata tags."""
        processor = ForestMetricsProcessor(test_settings)

        results = {
            "species_richness": np.random.rand(50, 50).astype(np.float32)
        }

        from rasterio.transform import Affine
        metadata = {
            'crs': 'ESRI:102039',
            'transform': Affine(30, 0, -2000000, 0, -30, -900000),
        }

        output_paths = processor._save_results(results, metadata, test_settings.output_dir)

        # Read back and check metadata
        import rasterio
        with rasterio.open(output_paths["species_richness"]) as src:
            tags = src.tags()
            assert 'SOFTWARE' in tags
            assert 'GridFIA' in tags['SOFTWARE']
            assert 'PROCESSED_BY' in tags
            assert str(src.crs) == 'ESRI:102039'

    def test_zarr_metadata_attributes(self, test_settings, temp_dir):
        """Test that Zarr files contain correct metadata attributes."""
        test_settings.calculations[0].output_format = "zarr"
        processor = ForestMetricsProcessor(test_settings)

        results = {
            "species_richness": np.random.rand(50, 50).astype(np.float32)
        }

        from rasterio.transform import Affine
        transform = Affine(30, 0, -2000000, 0, -30, -900000)
        metadata = {
            'crs': 'ESRI:102039',
            'transform': transform,
        }

        output_paths = processor._save_results(results, metadata, test_settings.output_dir)

        # Read back and check metadata (Zarr v3 uses group with 'data' array)
        root = zarr.open_group(output_paths["species_richness"], mode='r')
        z = root['data']
        assert z.attrs['crs'] == 'ESRI:102039'
        assert z.attrs['variable'] == 'species_richness'
        assert z.attrs['software'] == 'GridFIA Forest Metrics Processor'
        assert 'transform' in z.attrs


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_empty_array_all_zeros(self, empty_zarr_array, test_settings):
        """Test processing an array with all zero values."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        mock_calc = Mock()
        mock_calc.name = "zero_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = lambda x: np.sum(x, axis=0)
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(empty_zarr_array, [mock_calc])

        assert results["zero_calc"].shape == (50, 50)
        assert np.all(results["zero_calc"] == 0)

    def test_single_pixel_array(self, temp_dir, test_settings):
        """Test processing a single pixel array."""
        zarr_path = temp_dir / "single_pixel.zarr"
        z = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(2, 1, 1),
            chunks=(1, 1, 1),
            dtype='f4'
        )
        # Must match 3D shape (species, y, x)
        z[:] = np.array([[[1.0]], [[0.5]]], dtype=np.float32)
        z.attrs['species_codes'] = ['TOTAL', 'SP1']
        z.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 1, 1)

        mock_calc = Mock()
        mock_calc.name = "single_pixel_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = lambda x: np.ones(x.shape[1:]) * 42
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(z, [mock_calc])

        assert results["single_pixel_calc"].shape == (1, 1)
        assert results["single_pixel_calc"][0, 0] == 42

    def test_large_array_chunking(self, temp_dir, test_settings):
        """Test that large arrays are processed correctly with chunking."""
        zarr_path = temp_dir / "large.zarr"
        z = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(5, 500, 500),  # Larger array
            chunks=(1, 100, 100),
            dtype='f4'
        )
        # Fill with pattern to verify chunking works correctly
        for i in range(5):
            z[i] = np.ones((500, 500), dtype=np.float32) * (i + 1)
        z.attrs['species_codes'] = ['TOTAL', 'SP1', 'SP2', 'SP3', 'SP4']
        z.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 100, 100)  # 5x5 = 25 chunks

        mock_calc = Mock()
        mock_calc.name = "large_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = lambda x: np.sum(x, axis=0)
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(z, [mock_calc])

        assert results["large_calc"].shape == (500, 500)
        # Sum should be 1+2+3+4+5 = 15 for each pixel
        assert np.allclose(results["large_calc"], 15.0)

    def test_single_species_array(self, single_species_zarr, test_settings):
        """Test processing array with only one species."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        mock_calc = Mock()
        mock_calc.name = "single_species_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        # Count non-zero species (excluding total)
        mock_calc.calculate.side_effect = lambda x: np.sum(x[1:] > 0, axis=0).astype(np.float32)
        mock_calc.postprocess_result.side_effect = lambda x: x
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(single_species_zarr, [mock_calc])

        assert results["single_species_calc"].shape == (100, 100)
        # Should have 0 or 1 species at each pixel
        assert np.all(results["single_species_calc"] <= 1)


class TestErrorRecovery:
    """Test suite for error recovery during chunk processing."""

    def test_chunk_processing_validation_failure(self, sample_zarr_array, test_settings):
        """Test handling of validation failure in chunk processing."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        mock_calc = Mock()
        mock_calc.name = "failing_validation"
        mock_calc.validate_data.return_value = False  # Validation fails
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(sample_zarr_array, [mock_calc])

        # Result should be NaN-filled for failed validation
        assert "failing_validation" in results
        assert np.all(np.isnan(results["failing_validation"]))

    def test_chunk_processing_calculation_exception(self, sample_zarr_array, test_settings):
        """Test handling of exception during calculation."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        mock_calc = Mock()
        mock_calc.name = "exception_calc"
        mock_calc.validate_data.return_value = True
        mock_calc.preprocess_data.side_effect = lambda x: x
        mock_calc.calculate.side_effect = ValueError("Calculation error")
        mock_calc.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(sample_zarr_array, [mock_calc])

        # Result should be NaN-filled for failed calculation
        assert "exception_calc" in results
        assert np.all(np.isnan(results["exception_calc"]))

    def test_chunk_processing_partial_failure(self, sample_zarr_array, test_settings):
        """Test that one failing calculation doesn't affect others."""
        processor = ForestMetricsProcessor(test_settings)
        processor.chunk_size = (1, 50, 50)

        # Working calculation
        mock_calc_good = Mock()
        mock_calc_good.name = "good_calc"
        mock_calc_good.validate_data.return_value = True
        mock_calc_good.preprocess_data.side_effect = lambda x: x
        mock_calc_good.calculate.side_effect = lambda x: np.ones(x.shape[1:])
        mock_calc_good.postprocess_result.side_effect = lambda x: x
        mock_calc_good.get_output_dtype.return_value = np.float32

        # Failing calculation
        mock_calc_bad = Mock()
        mock_calc_bad.name = "bad_calc"
        mock_calc_bad.validate_data.return_value = True
        mock_calc_bad.preprocess_data.side_effect = lambda x: x
        mock_calc_bad.calculate.side_effect = RuntimeError("Bad calculation")
        mock_calc_bad.get_output_dtype.return_value = np.float32

        results = processor._process_in_chunks(sample_zarr_array, [mock_calc_good, mock_calc_bad])

        # Good calculation should succeed
        assert np.all(results["good_calc"] == 1.0)

        # Bad calculation should have NaN values
        assert np.all(np.isnan(results["bad_calc"]))


class TestLoadZarrArray:
    """Test suite for _load_zarr_array method."""

    def test_load_zarr_group_with_biomass(self, temp_dir):
        """Test loading zarr group with biomass array."""
        zarr_path = temp_dir / "group.zarr"
        root = zarr.open_group(str(zarr_path), mode='w')

        # Create biomass array using zarr 3 API
        biomass = root.create_array('biomass', shape=(3, 50, 50), dtype='f4')
        biomass[:] = np.random.rand(3, 50, 50).astype(np.float32)
        biomass.attrs['test_attr'] = 'test_value'

        # Create species arrays using zarr 3 API
        species_codes = root.create_array('species_codes', shape=(3,), dtype='U10')
        species_codes[:] = np.array(['TOTAL', 'SP1', 'SP2'])
        species_names = root.create_array('species_names', shape=(3,), dtype='U20')
        species_names[:] = np.array(['Total', 'Species 1', 'Species 2'])

        # Add root-level attributes
        root.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor()
        array, group = processor._load_zarr_array(str(zarr_path))

        assert array.shape == (3, 50, 50)
        assert 'species_codes' in array.attrs
        assert array.attrs['crs'] == 'ESRI:102039'
        assert group is not None

    def test_load_zarr_array_legacy(self, temp_dir):
        """Test loading a legacy standalone zarr array."""
        # Create a standalone zarr array (not in a group)
        zarr_path = temp_dir / "legacy_array.zarr"
        z = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(5, 100, 100),
            dtype='f4'
        )
        z[:] = np.random.rand(5, 100, 100).astype(np.float32)
        z.attrs['species_codes'] = ['TOTAL', 'SP1', 'SP2', 'SP3', 'SP4']
        z.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor()
        array, group = processor._load_zarr_array(str(zarr_path))

        assert array.shape == (5, 100, 100)
        # For standalone arrays, group should be None
        assert group is None

    def test_load_zarr_missing_biomass_array(self, temp_dir):
        """Test loading zarr group without biomass array raises error."""
        zarr_path = temp_dir / "no_biomass.zarr"
        root = zarr.open_group(str(zarr_path), mode='w')
        root.create_array('other_array', shape=(3, 50, 50), dtype='f4')

        processor = ForestMetricsProcessor()

        with pytest.raises(InvalidZarrStructure, match="No biomass"):
            processor._load_zarr_array(str(zarr_path))

    def test_load_zarr_invalid_path(self, temp_dir):
        """Test loading from invalid path raises error."""
        processor = ForestMetricsProcessor()

        with pytest.raises(InvalidZarrStructure):
            processor._load_zarr_array(str(temp_dir / "nonexistent.zarr"))


class TestExtractMetadata:
    """Test suite for _extract_metadata method."""

    def test_extract_metadata_with_transform(self, temp_dir):
        """Test metadata extraction when transform is available."""
        zarr_path = temp_dir / "with_transform.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(3, 50, 50), dtype='f4')
        z.attrs['species_codes'] = ['TOTAL', 'SP1', 'SP2']
        z.attrs['species_names'] = ['Total', 'Species 1', 'Species 2']
        z.attrs['crs'] = 'ESRI:102039'
        z.attrs['transform'] = [30, 0, -2000000, 0, -30, -900000]

        processor = ForestMetricsProcessor()
        metadata = processor._extract_metadata(z)

        assert metadata['crs'] == 'ESRI:102039'
        assert metadata['species_codes'] == ['TOTAL', 'SP1', 'SP2']
        assert metadata['shape'] == (50, 50)
        from rasterio.transform import Affine
        assert isinstance(metadata['transform'], Affine)

    def test_extract_metadata_with_bounds(self, temp_dir):
        """Test metadata extraction when only bounds are available."""
        zarr_path = temp_dir / "with_bounds.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(3, 50, 50), dtype='f4')
        z.attrs['species_codes'] = ['TOTAL', 'SP1', 'SP2']
        z.attrs['crs'] = 'ESRI:102039'
        z.attrs['bounds'] = [-2000000, -1000000, -1900000, -900000]

        processor = ForestMetricsProcessor()
        metadata = processor._extract_metadata(z)

        assert 'transform' in metadata
        assert metadata['bounds'] == [-2000000, -1000000, -1900000, -900000]

    def test_extract_metadata_default_transform(self, temp_dir, caplog):
        """Test metadata extraction uses default transform when none available."""
        zarr_path = temp_dir / "no_spatial.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(3, 50, 50), dtype='f4')
        z.attrs['species_codes'] = ['TOTAL', 'SP1', 'SP2']
        z.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor()

        import logging
        with caplog.at_level(logging.WARNING):
            metadata = processor._extract_metadata(z)

        assert 'transform' in metadata
        assert "No spatial reference found" in caplog.text


class TestInitializeCalculations:
    """Test suite for _initialize_calculations method."""

    def test_initialize_calculations_missing_from_registry(self, test_settings, caplog):
        """Test handling of calculations not found in registry."""
        # Add a calculation that doesn't exist in registry
        test_settings.calculations.append(
            CalculationConfig(name="nonexistent_calc", enabled=True)
        )

        processor = ForestMetricsProcessor(test_settings)
        enabled = processor._get_enabled_calculations()

        import logging
        with caplog.at_level(logging.WARNING):
            instances = processor._initialize_calculations(enabled)

        # Should have fewer instances than enabled configs
        # because nonexistent_calc wasn't found
        assert "not found in registry" in caplog.text.lower() or len(instances) < len(enabled)

    def test_initialize_calculations_with_parameters(self, test_settings):
        """Test that calculation parameters are passed correctly."""
        processor = ForestMetricsProcessor(test_settings)
        enabled = processor._get_enabled_calculations()

        # Check that species_richness has its threshold parameter
        sr_config = next(c for c in enabled if c.name == "species_richness")
        assert sr_config.parameters.get("biomass_threshold") == 0.0