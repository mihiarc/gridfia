import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

def test_data_integration(sample_heirs_gdf, sample_parcels_gdf, sample_ndvi_data, sample_matches):
    """Test integration of heirs properties, parcels, and NDVI data."""
    
    # Verify input data structure
    assert isinstance(sample_heirs_gdf, gpd.GeoDataFrame)
    assert isinstance(sample_parcels_gdf, gpd.GeoDataFrame)
    assert isinstance(sample_ndvi_data, pd.DataFrame)
    assert isinstance(sample_matches, pd.DataFrame)
    
    # Verify CRS consistency
    assert sample_heirs_gdf.crs == "EPSG:2264"
    assert sample_parcels_gdf.crs == "EPSG:2264"
    
    # Verify required columns
    heir_columns = {'geometry', 'area', 'county_nam', 'property_id'}
    assert all(col in sample_heirs_gdf.columns for col in heir_columns)
    assert all(col in sample_parcels_gdf.columns for col in heir_columns)
    
    ndvi_columns = {'property_id', 'property_type', 'ndvi_2018', 'ndvi_2020', 'ndvi_2022'}
    assert all(col in sample_ndvi_data.columns for col in ndvi_columns)
    
    match_columns = {'heir_id', 'neighbor_id', 'distance', 'area_ratio'}
    assert all(col in sample_matches.columns for col in match_columns)
    
    # Test data consistency
    # 1. All heir properties in matches should exist in heirs GDF
    heir_ids_in_matches = sample_matches['heir_id'].unique()
    assert all(hid in sample_heirs_gdf.property_id.values for hid in heir_ids_in_matches)
    
    # 2. All neighbor properties in matches should exist in parcels GDF
    neighbor_ids_in_matches = sample_matches['neighbor_id'].unique()
    assert all(nid in sample_parcels_gdf.property_id.values for nid in neighbor_ids_in_matches)
    
    # 3. All properties should have NDVI values
    all_property_ids = pd.concat([
        pd.Series(heir_ids_in_matches),
        pd.Series(neighbor_ids_in_matches)
    ]).unique()
    assert all(pid in sample_ndvi_data.property_id.values for pid in all_property_ids)
    
    # Test NDVI value ranges
    ndvi_years = ['ndvi_2018', 'ndvi_2020', 'ndvi_2022']
    for year in ndvi_years:
        values = sample_ndvi_data[year].dropna()
        assert all(values >= -1)
        assert all(values <= 1)
    
    # Test property type consistency
    heir_ndvi = sample_ndvi_data[sample_ndvi_data.property_id.isin(heir_ids_in_matches)]
    assert all(heir_ndvi.property_type == 'heir')
    
    neighbor_ndvi = sample_ndvi_data[sample_ndvi_data.property_id.isin(neighbor_ids_in_matches)]
    assert all(neighbor_ndvi.property_type == 'neighbor')
    
    # Test match quality metrics
    assert all(sample_matches.distance > 0)
    assert all(sample_matches.area_ratio >= 0.6)
    assert all(sample_matches.area_ratio <= 1.4)
    
    # Test data completeness
    # Each heir property should have at least one match
    match_counts = sample_matches.groupby('heir_id').size()
    assert all(match_counts > 0)
    
    # Each property should have NDVI values for all years
    complete_ndvi = sample_ndvi_data.dropna(subset=ndvi_years)
    assert len(complete_ndvi) == len(sample_ndvi_data)

def test_data_merge_workflow(sample_heirs_gdf, sample_parcels_gdf, sample_ndvi_data, sample_matches):
    """Test the complete data merging workflow."""
    
    # 1. Merge matches with heir properties
    heir_merge = sample_matches.merge(
        sample_heirs_gdf,
        left_on='heir_id',
        right_on='property_id',
        validate='many_to_one'
    )
    assert len(heir_merge) == len(sample_matches)
    
    # 2. Merge with neighbor properties
    neighbor_merge = heir_merge.merge(
        sample_parcels_gdf,
        left_on='neighbor_id',
        right_on='property_id',
        validate='many_to_one',
        suffixes=('_heir', '_neighbor')
    )
    assert len(neighbor_merge) == len(sample_matches)
    
    # 3. Merge with NDVI data
    # First, prepare NDVI data with property type prefix
    heir_ndvi = sample_ndvi_data[sample_ndvi_data.property_type == 'heir'].add_prefix('heir_')
    neighbor_ndvi = sample_ndvi_data[sample_ndvi_data.property_type == 'neighbor'].add_prefix('neighbor_')
    
    # Merge heir NDVI data
    final_merge = neighbor_merge.merge(
        heir_ndvi,
        left_on='heir_id',
        right_on='heir_property_id',
        validate='many_to_one'
    )
    assert len(final_merge) == len(sample_matches)
    
    # Merge neighbor NDVI data
    final_merge = final_merge.merge(
        neighbor_ndvi,
        left_on='neighbor_id',
        right_on='neighbor_property_id',
        validate='many_to_one'
    )
    assert len(final_merge) == len(sample_matches)
    
    # Verify merged data structure
    expected_columns = {
        'heir_id', 'neighbor_id', 'distance', 'area_ratio',
        'geometry_heir', 'geometry_neighbor',
        'heir_ndvi_2018', 'heir_ndvi_2020', 'heir_ndvi_2022',
        'neighbor_ndvi_2018', 'neighbor_ndvi_2020', 'neighbor_ndvi_2022'
    }
    assert all(col in final_merge.columns for col in expected_columns)
    
    # Verify data consistency
    assert all(final_merge.heir_ndvi_2018.notna())
    assert all(final_merge.neighbor_ndvi_2018.notna())
    assert all(final_merge.distance > 0)
    assert all(final_merge.area_ratio.between(0.6, 1.4)) 