import pytest
import pandas as pd
import geopandas as gpd
from src.analysis.property_matcher import PropertyMatcher

def test_property_matcher_init():
    """Test PropertyMatcher initialization."""
    matcher = PropertyMatcher()
    assert matcher.min_size_ratio == 0.6
    assert matcher.max_size_ratio == 1.4
    assert matcher.distance_threshold == 2000

def test_property_matcher_custom_params():
    """Test PropertyMatcher with custom parameters."""
    matcher = PropertyMatcher(min_size_ratio=0.7, max_size_ratio=1.3, distance_threshold=1500)
    assert matcher.min_size_ratio == 0.7
    assert matcher.max_size_ratio == 1.3
    assert matcher.distance_threshold == 1500

def test_find_matches(sample_heirs_gdf, sample_parcels_gdf):
    """Test finding matches for a single property."""
    matcher = PropertyMatcher()
    
    # Get first heir property
    heir_property = sample_heirs_gdf.iloc[[0]]
    
    # Ensure the property has valid geometry
    assert heir_property.geometry.is_valid.iloc[0]
    assert heir_property.area.iloc[0] > 0
    
    # Find matches
    matches = matcher.find_matches(heir_property, sample_parcels_gdf)
    
    # Verify results
    assert isinstance(matches, gpd.GeoDataFrame)
    if len(matches) > 0:
        distances = matches.geometry.distance(heir_property.geometry.iloc[0])
        assert all(distances <= matcher.distance_threshold)
        assert all(matches.area_ratio >= matcher.min_size_ratio)
        assert all(matches.area_ratio <= matcher.max_size_ratio)
    else:
        # If no matches found, verify that this is expected
        heir_area = heir_property.area.iloc[0]
        min_size = heir_area * matcher.min_size_ratio
        max_size = heir_area * matcher.max_size_ratio
        
        # Check if any parcels meet the size criteria
        size_matches = sample_parcels_gdf[
            (sample_parcels_gdf.area >= min_size) &
            (sample_parcels_gdf.area <= max_size)
        ]
        
        if len(size_matches) > 0:
            # If size matches exist, they must be too far away
            distances = size_matches.geometry.distance(heir_property.geometry.iloc[0])
            assert all(distances > matcher.distance_threshold)

def test_find_all_matches(sample_heirs_gdf, sample_parcels_gdf):
    """Test finding matches for multiple properties."""
    matcher = PropertyMatcher()
    
    # Find matches for all properties
    matches = matcher.find_all_matches(sample_heirs_gdf, sample_parcels_gdf)
    
    # Verify results
    assert isinstance(matches, pd.DataFrame)
    if len(matches) > 0:
        assert 'heir_id' in matches.columns
        assert 'neighbor_id' in matches.columns
        assert 'distance' in matches.columns
        assert 'area_ratio' in matches.columns
        assert all(matches.distance <= matcher.distance_threshold)
        assert all(matches.area_ratio >= matcher.min_size_ratio)
        assert all(matches.area_ratio <= matcher.max_size_ratio)

def test_find_matches_with_sample(sample_heirs_gdf, sample_parcels_gdf):
    """Test finding matches with sample size."""
    matcher = PropertyMatcher()
    sample_size = 1
    
    # Find matches for sample
    matches = matcher.find_all_matches(sample_heirs_gdf, sample_parcels_gdf, sample_size=sample_size)
    
    # Verify results
    if len(matches) > 0:
        assert len(matches['heir_id'].unique()) <= sample_size 