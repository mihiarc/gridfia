"""Test configuration and fixtures."""
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import numpy as np

@pytest.fixture
def test_data_dir():
    """Create a temporary test data directory."""
    return Path("tests/data")

@pytest.fixture
def sample_geometry():
    """Create a sample polygon geometry."""
    return Polygon([
        (0, 0), (0, 100), 
        (100, 100), (100, 0)
    ])

@pytest.fixture
def sample_heirs_gdf():
    """Create a sample GeoDataFrame of heir properties.
    Using NC State Plane (meters) EPSG:32119 for accurate area calculations.
    """
    # Create two simple polygons - coordinates in meters
    polygons = [
        Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]),
        Polygon([(2000, 2000), (2000, 2100), (2100, 2100), (2100, 2000)])
    ]
    
    # Create GeoDataFrame with pre-calculated areas
    data = {
        'property_id': [1, 2],
        'county_nam': ['VANCE', 'VANCE'],
        'geometry': polygons
    }
    
    gdf = gpd.GeoDataFrame(data, crs='EPSG:32119')
    # Calculate areas after setting CRS
    gdf['area'] = gdf.geometry.area
    
    return gdf

@pytest.fixture
def sample_parcels_gdf():
    """Create a sample GeoDataFrame of all parcels.
    Using NC State Plane (meters) EPSG:32119 for accurate area calculations.
    """
    # Create two simple polygons - coordinates in meters
    polygons = [
        Polygon([(500, 500), (500, 600), (600, 600), (600, 500)]),
        Polygon([(2500, 2500), (2500, 2600), (2600, 2600), (2600, 2500)])
    ]
    
    # Create GeoDataFrame with pre-calculated areas
    data = {
        'property_id': [3, 4],
        'county_nam': ['VANCE', 'VANCE'],
        'geometry': polygons
    }
    
    gdf = gpd.GeoDataFrame(data, crs='EPSG:32119')
    # Calculate areas after setting CRS
    gdf['area'] = gdf.geometry.area
    
    return gdf

@pytest.fixture
def sample_ndvi_data():
    """Create sample NDVI data."""
    data = {
        'property_id': [1, 2, 3, 4],
        'property_type': ['heir', 'heir', 'neighbor', 'neighbor'],
        'ndvi_2018': [0.5, 0.6, 0.4, 0.5],
        'ndvi_2020': [0.4, 0.5, 0.3, 0.4],
        'ndvi_2022': [0.6, 0.7, 0.5, 0.6]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_matches():
    """Create sample property matches."""
    data = {
        'heir_id': [1, 1, 2, 2],
        'neighbor_id': [3, 4, 3, 4],
        'distance': [100, 200, 150, 250],
        'area_ratio': [1.0, 1.1, 0.9, 1.0]
    }
    return pd.DataFrame(data) 