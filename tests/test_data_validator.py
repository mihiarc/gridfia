"""Tests for the data validation functionality."""
import json
from pathlib import Path
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from shapely.wkt import loads

from processing.data_validator import DataValidator

@pytest.fixture
def sample_polygon():
    """Create a valid polygon for testing."""
    return Polygon([
        (0, 0), (0, 1), (1, 1), (1, 0), (0, 0)
    ])

@pytest.fixture
def invalid_polygon():
    """Create an invalid self-intersecting polygon."""
    return Polygon([
        (0, 0), (1, 1), (1, 0), (0, 1), (0, 0)
    ])

@pytest.fixture
def sample_geodataframe(sample_polygon):
    """Create a sample GeoDataFrame for testing."""
    data = {
        'id': [1, 2],
        'name': ['Test1', 'Test2'],
        'area': [100.0, 200.0],
        'geometry': [
            sample_polygon,
            Polygon([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)])
        ]
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=2264, inplace=True)
    return gdf

@pytest.fixture
def validator():
    """Create a DataValidator instance with test configuration."""
    return DataValidator(
        required_fields=['id', 'name', 'area'],
        geometry_type='Polygon',
        srid=2264
    )

def test_validate_geodataframe_success(validator, sample_geodataframe):
    """Test successful validation of a valid GeoDataFrame."""
    success, report = validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    
    assert success is True
    assert len(report["issues"]) == 0
    assert report["record_count"] == 2

def test_validate_missing_fields(validator):
    """Test validation with missing required fields."""
    data = {
        'id': [1],
        'geometry': [Point(0, 0)]
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=2264, inplace=True)
    
    success, report = validator.validate_geodataframe(gdf, "test_data")
    
    assert success is False
    assert any(issue["type"] == "missing_fields" for issue in report["issues"])
    missing_fields = next(
        issue["fields"] 
        for issue in report["issues"] 
        if issue["type"] == "missing_fields"
    )
    assert "name" in missing_fields
    assert "area" in missing_fields

def test_validate_wrong_crs(validator, sample_geodataframe):
    """Test validation with incorrect CRS."""
    sample_geodataframe.set_crs(epsg=4326, inplace=True)
    
    success, report = validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    
    assert success is False
    assert any(issue["type"] == "wrong_crs" for issue in report["issues"])
    crs_issue = next(
        issue for issue in report["issues"] 
        if issue["type"] == "wrong_crs"
    )
    assert crs_issue["expected"] == 2264
    assert crs_issue["found"] == 4326

def test_validate_invalid_geometry_type(validator):
    """Test validation with incorrect geometry type."""
    data = {
        'id': [1],
        'name': ['Test1'],
        'area': [100.0],
        'geometry': [Point(0, 0)]
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=2264, inplace=True)
    
    success, report = validator.validate_geodataframe(gdf, "test_data")
    
    assert success is False
    assert any(
        issue["type"] == "invalid_geometry_type" 
        for issue in report["issues"]
    )
    geom_issue = next(
        issue for issue in report["issues"] 
        if issue["type"] == "invalid_geometry_type"
    )
    assert geom_issue["expected"] == "Polygon"
    assert "Point" in geom_issue["found"]

def test_validate_invalid_geometry(validator, invalid_polygon):
    """Test validation with invalid geometry."""
    data = {
        'id': [1],
        'name': ['Test1'],
        'area': [100.0],
        'geometry': [invalid_polygon]
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=2264, inplace=True)
    
    success, report = validator.validate_geodataframe(gdf, "test_data")
    
    assert success is False
    assert any(
        issue["type"] == "invalid_geometries" 
        for issue in report["issues"]
    )
    geom_issue = next(
        issue for issue in report["issues"] 
        if issue["type"] == "invalid_geometries"
    )
    assert geom_issue["count"] == 1

def test_validate_null_geometries(validator):
    """Test validation with null geometries."""
    data = {
        'id': [1, 2],
        'name': ['Test1', 'Test2'],
        'area': [100.0, 200.0],
        'geometry': [None, None]
    }
    gdf = gpd.GeoDataFrame(data)
    gdf.set_crs(epsg=2264, inplace=True)
    
    success, report = validator.validate_geodataframe(gdf, "test_data")
    
    assert success is False
    assert any(issue["type"] == "null_geometries" for issue in report["issues"])
    null_issue = next(
        issue for issue in report["issues"] 
        if issue["type"] == "null_geometries"
    )
    assert null_issue["count"] == 2

def test_validate_file_parquet(validator, sample_geodataframe, tmp_path):
    """Test validation of a Parquet file."""
    file_path = tmp_path / "test.parquet"
    sample_geodataframe.to_parquet(file_path)
    
    success, report = validator.validate_file(file_path)
    
    assert success is True
    assert len(report["issues"]) == 0

def test_validate_file_error(validator):
    """Test validation with non-existent file."""
    success, report = validator.validate_file(Path("nonexistent.parquet"))
    
    assert success is False
    assert any(
        issue["type"] == "validation_error" 
        for issue in report["issues"]
    )

def test_generate_report(validator, sample_geodataframe, tmp_path):
    """Test report generation."""
    # Validate some data first
    validator.validate_geodataframe(sample_geodataframe, "test_data")
    
    # Generate and save report
    report_path = tmp_path / "validation_report.json"
    report = validator.generate_report(report_path)
    
    # Check report contents
    assert "validation_time" in report
    assert "datasets" in report
    assert "test_data" in report["datasets"]
    
    # Check saved file
    assert report_path.exists()
    with open(report_path) as f:
        saved_report = json.load(f)
    assert saved_report["datasets"]["test_data"]["record_count"] == 2 