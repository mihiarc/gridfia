"""
Integration tests for the forest metrics processing pipeline.
"""

import pytest
import numpy as np
import zarr
import rasterio
import xarray as xr
from pathlib import Path

# Check if netCDF4 is available
try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

from gridfia.core.processors.forest_metrics import ForestMetricsProcessor, run_forest_analysis
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.core.calculations import registry
from gridfia.exceptions import (
    InvalidZarrStructure, SpeciesNotFound, CalculationFailed,
    APIConnectionError, InvalidLocationConfig, DownloadError
)


def get_zarr_path(zarr_array):
    """Get the path from a zarr array, handling different zarr versions."""
    if hasattr(zarr_array.store, 'path'):
        return str(zarr_array.store.path)
    elif hasattr(zarr_array.store, 'directory'):
        # For zarr 3.x LocalStore
        return str(zarr_array.store.directory)
    elif hasattr(zarr_array.store, 'root'):
        # For zarr 3.x with root directory
        return str(zarr_array.store.root)
    else:
        # Try to get from store's string representation
        store_str = str(zarr_array.store)
        if store_str.startswith("file://"):
            return store_str[7:]  # Remove file:// prefix
        # For zarr 3.x LocalStore, extract path from repr
        import re
        match = re.search(r"path=PosixPath\('([^']+)'\)", store_str)
        if match:
            return match.group(1)
        return store_str


class TestForestMetricsPipeline:
    """Integration tests for the complete forest metrics pipeline."""
    
    def test_full_pipeline_with_sample_data(self, sample_zarr_array, test_settings):
        """Test the full pipeline with sample zarr data."""
        # Get zarr path from fixture
        zarr_path = get_zarr_path(sample_zarr_array)
        
        # Run the processor
        processor = ForestMetricsProcessor(test_settings)
        results = processor.run_calculations(zarr_path)
        
        # Verify results
        assert len(results) == 3  # Three enabled calculations
        assert "species_richness" in results
        assert "total_biomass" in results
        assert "shannon_diversity" in results
        
        # Check that files were created
        for calc_name, file_path in results.items():
            assert Path(file_path).exists()
            assert file_path.endswith(".tif")  # Default format
            
            # Verify GeoTIFF can be read
            with rasterio.open(file_path) as src:
                assert src.count == 1
                assert src.crs is not None
                assert src.transform is not None
                data = src.read(1)
                assert data.shape == (100, 100)  # Same as input
    
    @pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not installed")
    def test_pipeline_with_different_output_formats(self, sample_zarr_array, temp_dir):
        """Test saving results in different formats."""
        zarr_path = get_zarr_path(sample_zarr_array)

        # Configure different output formats
        settings = GridFIASettings(
            output_dir=temp_dir / "multi_format_output",
            calculations=[
                CalculationConfig(
                    name="species_richness",
                    enabled=True,
                    output_format="geotiff"
                ),
                CalculationConfig(
                    name="total_biomass",
                    enabled=True,
                    output_format="zarr"
                ),
                CalculationConfig(
                    name="shannon_diversity",
                    enabled=True,
                    output_format="netcdf"
                )
            ]
        )
        
        processor = ForestMetricsProcessor(settings)
        results = processor.run_calculations(zarr_path)
        
        # Verify different formats
        assert results["species_richness"].endswith(".tif")
        assert results["total_biomass"].endswith(".zarr")
        assert results["shannon_diversity"].endswith(".nc")
        
        # Verify each format can be read
        # GeoTIFF
        with rasterio.open(results["species_richness"]) as src:
            assert src.read(1).shape == (100, 100)
        
        # Zarr
        z = zarr.open_array(results["total_biomass"], mode='r')
        assert z.shape == (100, 100)
        assert 'crs' in z.attrs
        
        # NetCDF
        ds = xr.open_dataset(results["shannon_diversity"])
        assert 'shannon_diversity' in ds.data_vars
        assert ds.shannon_diversity.shape == (100, 100)
        ds.close()
    
    def test_pipeline_with_empty_data(self, empty_zarr_array, test_settings):
        """Test pipeline handles empty (all zero) data gracefully."""
        zarr_path = get_zarr_path(empty_zarr_array)
        
        processor = ForestMetricsProcessor(test_settings)
        results = processor.run_calculations(zarr_path)
        
        # Should still produce results, even if all zeros
        assert len(results) > 0
        
        # Check species richness is all zeros
        with rasterio.open(results["species_richness"]) as src:
            data = src.read(1)
            assert np.all(data == 0)
    
    def test_pipeline_with_single_species(self, single_species_zarr, test_settings):
        """Test pipeline with single species data."""
        zarr_path = get_zarr_path(single_species_zarr)
        
        processor = ForestMetricsProcessor(test_settings)
        results = processor.run_calculations(zarr_path)
        
        # Check results
        assert len(results) == 3
        
        # Species richness should be 1 where biomass > 0
        with rasterio.open(results["species_richness"]) as src:
            data = src.read(1)
            assert np.max(data) == 1  # Only one species
    
    def test_run_forest_analysis_convenience_function(self, sample_zarr_array, temp_dir):
        """Test the convenience function run_forest_analysis."""
        zarr_path = get_zarr_path(sample_zarr_array)
        
        # Create a simple config file
        config_path = temp_dir / "test_config.yaml"
        config_content = """
app_name: BigMap Test
output_dir: test_output
calculations:
  - name: species_richness
    enabled: true
  - name: total_biomass
    enabled: false
"""
        config_path.write_text(config_content)
        
        # Run analysis
        results = run_forest_analysis(zarr_path, str(config_path))
        
        # Should only have species_richness enabled
        assert len(results) == 1
        assert "species_richness" in results
        assert "total_biomass" not in results
    
    def test_chunked_processing_consistency(self, sample_zarr_array, test_settings):
        """Test that chunked processing produces same results as full processing."""
        zarr_path = get_zarr_path(sample_zarr_array)
        
        # Process with small chunks
        processor1 = ForestMetricsProcessor(test_settings)
        processor1.chunk_size = (1, 25, 25)  # Small chunks
        results1 = processor1.run_calculations(zarr_path)
        
        # Process with large chunks (essentially full array)
        processor2 = ForestMetricsProcessor(test_settings)
        processor2.chunk_size = (1, 100, 100)  # Full array
        results2 = processor2.run_calculations(zarr_path)
        
        # Compare results
        for calc_name in results1:
            with rasterio.open(results1[calc_name]) as src1:
                data1 = src1.read(1)
            with rasterio.open(results2[calc_name]) as src2:
                data2 = src2.read(1)
            
            # Results should be identical
            np.testing.assert_array_almost_equal(data1, data2, decimal=5)
    
    def test_custom_output_names(self, sample_zarr_array, test_settings):
        """Test using custom output names for calculations."""
        zarr_path = get_zarr_path(sample_zarr_array)
        
        # Configure custom names
        test_settings.calculations[0].output_name = "custom_richness"
        test_settings.calculations[1].output_name = "custom_biomass"
        
        processor = ForestMetricsProcessor(test_settings)
        results = processor.run_calculations(zarr_path)
        
        # Check custom names were used
        assert any("custom_richness" in path for path in results.values())
        assert any("custom_biomass" in path for path in results.values())
    
    def test_error_handling_invalid_calculation(self, sample_zarr_array, test_settings):
        """Test handling of invalid calculation names."""
        zarr_path = get_zarr_path(sample_zarr_array)
        
        # Add invalid calculation
        test_settings.calculations.append(
            CalculationConfig(name="invalid_calculation", enabled=True)
        )
        
        processor = ForestMetricsProcessor(test_settings)
        # Should still run valid calculations
        results = processor.run_calculations(zarr_path)
        
        # Should have results for valid calculations only
        assert len(results) == 3  # Only valid ones
        assert "invalid_calculation" not in results
    
    def test_spatial_metadata_preservation(self, sample_zarr_array, test_settings):
        """Test that spatial metadata is preserved in outputs."""
        zarr_path = get_zarr_path(sample_zarr_array)
        
        processor = ForestMetricsProcessor(test_settings)
        results = processor.run_calculations(zarr_path)
        
        # Check that CRS and transform are preserved
        with rasterio.open(results["species_richness"]) as src:
            assert src.crs.to_string() == 'ESRI:102039'
            assert src.transform is not None
            assert src.bounds is not None
            
            # Check transform values match input
            input_transform = sample_zarr_array.attrs['transform']
            output_transform = list(src.transform)[:6]
            np.testing.assert_array_almost_equal(
                input_transform, output_transform, decimal=2
            )
    
    @pytest.mark.parametrize("calc_name,expected_range", [
        ("species_richness", (0, 5)),  # 0-5 species
        ("total_biomass", (0, 200)),   # Biomass values
        ("shannon_diversity", (0, 2))   # Shannon index typically 0-3
    ])
    def test_calculation_value_ranges(self, sample_zarr_array, test_settings, calc_name, expected_range):
        """Test that calculation outputs are in expected ranges."""
        zarr_path = get_zarr_path(sample_zarr_array)
        
        processor = ForestMetricsProcessor(test_settings)
        results = processor.run_calculations(zarr_path)
        
        if calc_name in results:
            with rasterio.open(results[calc_name]) as src:
                data = src.read(1)
                
                # Check value range
                min_val, max_val = expected_range
                assert np.min(data) >= min_val - 0.01  # Small tolerance
                assert np.max(data) <= max_val + 10  # Larger tolerance for biomass


class TestErrorConditions:
    """Test error handling in the pipeline."""

    def test_no_enabled_calculations(self, sample_zarr_array, test_settings):
        """Test error when no calculations are enabled."""
        zarr_path = get_zarr_path(sample_zarr_array)

        # Disable all calculations
        for calc in test_settings.calculations:
            calc.enabled = False

        processor = ForestMetricsProcessor(test_settings)

        with pytest.raises(CalculationFailed, match="No calculations enabled"):
            processor.run_calculations(zarr_path)

    def test_invalid_zarr_path(self, test_settings):
        """Test error handling for invalid zarr path."""
        processor = ForestMetricsProcessor(test_settings)

        with pytest.raises(InvalidZarrStructure, match="Cannot open"):
            processor.run_calculations("/path/does/not/exist.zarr")

    def test_missing_required_attributes(self, temp_dir, test_settings):
        """Test error when zarr is missing required attributes."""
        # Create zarr without required attributes
        zarr_path = temp_dir / "invalid.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(2, 10, 10))
        # No species_codes or crs attributes

        processor = ForestMetricsProcessor(test_settings)

        with pytest.raises(InvalidZarrStructure, match="Missing required attributes"):
            processor.run_calculations(str(zarr_path))

    def test_dimension_mismatch(self, temp_dir, test_settings):
        """Test error when species dimension doesn't match metadata."""
        zarr_path = temp_dir / "mismatch.zarr"
        z = zarr.open_array(str(zarr_path), mode='w', shape=(3, 10, 10))
        z.attrs['species_codes'] = ['SP1', 'SP2']  # Only 2 codes for 3 layers
        z.attrs['crs'] = 'ESRI:102039'

        processor = ForestMetricsProcessor(test_settings)

        with pytest.raises(InvalidZarrStructure, match="doesn't match"):
            processor.run_calculations(str(zarr_path))