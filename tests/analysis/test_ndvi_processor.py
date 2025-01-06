import pytest
import numpy as np
import rasterio
from pathlib import Path
from src.analysis.ndvi_processor import NDVIProcessor

def create_test_raster(path, year):
    """Create a test NDVI raster file."""
    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    data = np.random.uniform(-1, 1, (100, 100))
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='EPSG:2264',
        transform=transform
    ) as dst:
        dst.write(data, 1)

def test_ndvi_processor_init(tmp_path):
    """Test NDVIProcessor initialization."""
    processor = NDVIProcessor(ndvi_dir=str(tmp_path))
    assert processor.target_crs == "EPSG:2264"
    assert set(processor.years) == {'2018', '2020', '2022'}
    assert isinstance(processor.ndvi_files, dict)

def test_load_ndvi_files(tmp_path):
    """Test NDVI file loading."""
    # Create test NDVI files
    for year in ['2018', '2020', '2022']:
        create_test_raster(tmp_path / f"ndvi_{year}.tif", year)
    
    processor = NDVIProcessor(ndvi_dir=str(tmp_path))
    assert all(year in processor.ndvi_files for year in processor.years)
    assert all(len(processor.ndvi_files[year]) > 0 for year in processor.years)

def test_extract_ndvi(tmp_path, sample_geometry):
    """Test NDVI extraction for a single property."""
    # Create test NDVI file
    create_test_raster(tmp_path / "ndvi_2020.tif", '2020')
    
    processor = NDVIProcessor(ndvi_dir=str(tmp_path))
    ndvi = processor.extract_ndvi(sample_geometry, '2020')
    
    if ndvi is not None:
        assert -1 <= ndvi <= 1

def test_process_properties(tmp_path, sample_heirs_gdf, sample_parcels_gdf):
    """Test property processing."""
    # Create test NDVI files
    for year in ['2018', '2020', '2022']:
        create_test_raster(tmp_path / f"ndvi_{year}.tif", year)
    
    # Setup processor
    processor = NDVIProcessor(ndvi_dir=str(tmp_path))
    
    # Process sample properties
    property_ids = sample_heirs_gdf.index[:2].tolist()
    results = processor.process_properties(property_ids=property_ids)
    
    # Verify results
    assert 'property_id' in results.columns
    assert 'property_type' in results.columns
    for year in processor.years:
        assert f'ndvi_{year}' in results.columns 