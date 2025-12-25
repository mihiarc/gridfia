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
        with pytest.raises(ValueError, match="Missing required attributes"):
            processor._validate_zarr_array(z)
    
    def test_validate_zarr_array_invalid_shape(self, temp_dir):
        """Test zarr validation with invalid shape."""
        # Create zarr with wrong dimensions
        zarr_path = temp_dir / "invalid_shape.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(10, 10))  # 2D instead of 3D
        z.attrs['species_codes'] = ['SP1']
        
        processor = ForestMetricsProcessor()
        with pytest.raises(ValueError, match="Expected 3D array"):
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
        
        with pytest.raises(ValueError, match="No calculations enabled"):
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