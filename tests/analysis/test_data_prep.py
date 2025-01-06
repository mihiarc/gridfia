import pytest
from pathlib import Path
import pandas as pd
import geopandas as gpd
from src.analysis.data_prep import DataPreparator

def test_data_preparator_init():
    """Test DataPreparator initialization."""
    prep = DataPreparator()
    assert prep.data_dir == Path("data")
    assert prep.target_crs == "EPSG:2264"

def test_data_preparator_with_custom_dir():
    """Test DataPreparator with custom directory."""
    custom_dir = "custom_data"
    prep = DataPreparator(data_dir=custom_dir)
    assert prep.data_dir == Path(custom_dir)

def test_load_and_prepare_data(sample_heirs_gdf, sample_parcels_gdf, tmp_path):
    """Test data loading and preparation."""
    # Setup test data directory structure
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw/gis"
    raw_dir.mkdir(parents=True)
    
    # Save test data
    heirs_file = raw_dir / "nc-hp_v2.geojson"
    parcels_file = raw_dir / "nc-parcels.geojson"
    
    # Convert to target CRS before saving
    sample_heirs_gdf = sample_heirs_gdf.to_crs("EPSG:2264")
    sample_parcels_gdf = sample_parcels_gdf.to_crs("EPSG:2264")

    # Save test data
    sample_heirs_gdf.to_file(heirs_file, driver='GeoJSON')
    sample_parcels_gdf.to_file(parcels_file, driver='GeoJSON')
    
    # Test data preparation
    prep = DataPreparator(data_dir=str(data_dir))
    heirs_gdf, parcels_gdf = prep.load_and_prepare_data()
    
    # Verify results
    assert isinstance(heirs_gdf, gpd.GeoDataFrame)
    assert isinstance(parcels_gdf, gpd.GeoDataFrame)
    assert heirs_gdf.crs == "EPSG:2264"
    assert parcels_gdf.crs == "EPSG:2264"
    assert all(heirs_gdf.geometry.is_valid)
    assert all(parcels_gdf.geometry.is_valid)
    assert all(heirs_gdf.area > 0)
    assert all(parcels_gdf.area > 0)
    
    # Verify data content
    assert len(heirs_gdf) == len(sample_heirs_gdf)
    assert len(parcels_gdf) == len(sample_parcels_gdf)
    assert set(heirs_gdf.columns) == set(sample_heirs_gdf.columns)
    assert set(parcels_gdf.columns) == set(sample_parcels_gdf.columns) 