"""Tests for the neighbor NDVI processor module."""

import pytest
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.coords import BoundingBox

from src.analysis.core.ndvi.neighbor_processor import (
    NeighborSelector,
    NDVIExtractor,
    ResultsManager,
    NeighborNDVIProcessor
)

@pytest.fixture
def sample_matches(tmp_path):
    """Create sample matches data."""
    matches_data = {
        'heir_id': [1, 1, 2, 2],
        'neighbor_id': [3, 4, 5, 6],
        'distance': [100, 200, 150, 250],
        'area_ratio': [1.0, 1.1, 0.9, 1.0]
    }
    matches_file = tmp_path / "matches.csv"
    pd.DataFrame(matches_data).to_csv(matches_file, index=False)
    return matches_file

@pytest.fixture
def sample_parcels(tmp_path):
    """Create sample parcels data."""
    # Create simple polygons for testing
    polygons = [
        Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
        Polygon([(20, 20), (20, 30), (30, 30), (30, 20)]),
        Polygon([(40, 40), (40, 50), (50, 50), (50, 40)]),
        Polygon([(60, 60), (60, 70), (70, 70), (70, 60)])
    ]
    
    parcels_data = {
        'property_id': [3, 4, 5, 6],
        'county_nam': ['VANCE'] * 4,
        'geometry': polygons,
        'area': [100] * 4
    }
    
    parcels_gdf = gpd.GeoDataFrame(parcels_data, crs='EPSG:2264')
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    parcels_file = processed_dir / "parcels_processed.parquet"
    parcels_gdf.to_parquet(parcels_file)
    return processed_dir

@pytest.fixture
def sample_ndvi_files(tmp_path):
    """Create sample NDVI raster files."""
    ndvi_dir = tmp_path / "ndvi"
    ndvi_dir.mkdir()
    
    # Create two sample NDVI files with consistent bounds
    bounds = BoundingBox(left=0, bottom=0, right=100, top=100)
    
    for part in ['p1', 'p2']:
        filename = f"ndvi_NAIP_Vance_County_2018-{part}.tif"
        with rasterio.open(
            ndvi_dir / filename,
            'w',
            driver='GTiff',
            height=10,
            width=10,
            count=1,
            dtype='float32',
            crs='EPSG:2264',
            transform=rasterio.transform.from_bounds(*bounds, 10, 10)
        ) as dst:
            dst.write(1, 1)
    
    return ndvi_dir

def test_neighbor_selector(sample_matches, sample_parcels):
    """Test NeighborSelector functionality."""
    selector = NeighborSelector(sample_matches, sample_parcels)
    bounds_poly = box(0, 0, 100, 100)
    
    neighbors = selector.get_neighbors_within_bounds(bounds_poly, "EPSG:2264")
    assert isinstance(neighbors, gpd.GeoDataFrame)
    assert len(neighbors) > 0
    assert all(neighbors.geometry.intersects(bounds_poly))

def test_ndvi_extractor(sample_ndvi_files):
    """Test NDVIExtractor functionality."""
    extractor = NDVIExtractor(sample_ndvi_files)
    bounds_poly, crs = extractor.get_ndvi_bounds()
    
    assert isinstance(bounds_poly, box)
    assert crs == 'EPSG:2264'
    assert bounds_poly.bounds == (0, 0, 100, 100)

def test_results_manager(tmp_path):
    """Test ResultsManager functionality."""
    output_dir = tmp_path / "output"
    manager = ResultsManager(output_dir)
    
    # Create sample results
    results_data = {
        'property_id': [1, 2],
        'ndvi_mean': [0.5, 0.6]
    }
    results_df = pd.DataFrame(results_data)
    
    output_file = manager.save_results(results_df)
    assert output_file.exists()
    loaded_results = pd.read_parquet(output_file)
    pd.testing.assert_frame_equal(results_df, loaded_results)

def test_neighbor_ndvi_processor(sample_matches, sample_parcels, sample_ndvi_files, tmp_path):
    """Test NeighborNDVIProcessor end-to-end functionality."""
    output_dir = tmp_path / "output"
    
    processor = NeighborNDVIProcessor(
        ndvi_dir=sample_ndvi_files,
        output_dir=output_dir,
        matches_file=sample_matches,
        processed_dir=sample_parcels
    )
    
    result_file = processor.process()
    assert result_file is not None
    assert result_file.exists()
    
    # Verify results structure
    results = pd.read_parquet(result_file)
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0 