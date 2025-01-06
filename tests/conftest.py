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
    """Create a sample heirs property GeoDataFrame."""
    geometries = [
        Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]),
        Polygon([(200, 200), (200, 300), (300, 300), (300, 200)])
    ]
    data = {
        'geometry': geometries,
        'area': [10000, 10000],
        'county_nam': ['VANCE', 'VANCE'],
        'property_id': [1, 2]
    }
    return gpd.GeoDataFrame(data, crs="EPSG:2264")

@pytest.fixture
def sample_parcels_gdf():
    """Create a sample parcels GeoDataFrame."""
    geometries = [
        Polygon([(50, 50), (50, 150), (150, 150), (150, 50)]),
        Polygon([(250, 250), (250, 350), (350, 350), (350, 250)])
    ]
    data = {
        'geometry': geometries,
        'area': [10000, 10000],
        'county_nam': ['VANCE', 'VANCE'],
        'property_id': [3, 4]
    }
    return gpd.GeoDataFrame(data, crs="EPSG:2264")

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