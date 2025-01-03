"""
Unit tests for the DataValidator class.
"""
import pytest
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, Point

from processing.data_validator import DataValidator

def test_validate_required_fields(test_validator, sample_geodataframe):
    """Test validation of required fields."""
    # Test with valid data
    success, report = test_validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    assert success is True
    assert len(report["issues"]) == 0
    
    # Test with missing fields
    invalid_gdf = sample_geodataframe.drop(columns=['area'])
    success, report = test_validator.validate_geodataframe(
        invalid_gdf,
        "test_data"
    )
    assert success is False
    assert any(issue["type"] == "missing_fields" for issue in report["issues"])
    assert "area" in next(
        issue["fields"] 
        for issue in report["issues"] 
        if issue["type"] == "missing_fields"
    )

def test_validate_geometry_type(test_validator, sample_geodataframe):
    """Test geometry type validation."""
    # Test with valid geometry type
    success, report = test_validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    assert success is True
    assert len(report["issues"]) == 0
    
    # Test with invalid geometry type
    invalid_geom = sample_geodataframe.copy()
    invalid_geom.geometry = [Point(0, 0), Point(1, 1)]
    success, report = test_validator.validate_geodataframe(
        invalid_geom,
        "test_data"
    )
    assert success is False
    assert any(
        issue["type"] == "invalid_geometry_type" 
        for issue in report["issues"]
    )

def test_validate_crs(test_validator, sample_geodataframe):
    """Test coordinate reference system validation."""
    # Test with valid CRS
    success, report = test_validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    assert success is True
    assert len(report["issues"]) == 0
    
    # Test with wrong CRS
    wrong_crs = sample_geodataframe.copy()
    wrong_crs.set_crs(epsg=4326, inplace=True)
    success, report = test_validator.validate_geodataframe(
        wrong_crs,
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

def test_validate_topology(test_validator, invalid_polygon):
    """Test topology validation."""
    # Create GeoDataFrame with invalid topology
    invalid_gdf = gpd.GeoDataFrame({
        'parcel_id': ['P001'],
        'area': [100.0],
        'county': ['County1'],
        'geometry': [invalid_polygon]
    })
    invalid_gdf.set_crs(epsg=2264, inplace=True)
    
    success, report = test_validator.validate_geodataframe(
        invalid_gdf,
        "test_data"
    )
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

def test_validate_null_geometries(test_validator):
    """Test validation of null geometries."""
    # Create GeoDataFrame with null geometries
    null_gdf = gpd.GeoDataFrame({
        'parcel_id': ['P001', 'P002'],
        'area': [100.0, 200.0],
        'county': ['County1', 'County2'],
        'geometry': [None, None]
    })
    null_gdf.set_crs(epsg=2264, inplace=True)
    
    success, report = test_validator.validate_geodataframe(
        null_gdf,
        "test_data"
    )
    assert success is False
    assert any(issue["type"] == "null_geometries" for issue in report["issues"])
    null_issue = next(
        issue for issue in report["issues"] 
        if issue["type"] == "null_geometries"
    )
    assert null_issue["count"] == 2

def test_validation_report_format(test_validator, sample_geodataframe):
    """Test validation report structure."""
    success, report = test_validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    
    # Check report structure
    assert isinstance(report, dict)
    assert "dataset" in report
    assert "timestamp" in report
    assert "record_count" in report
    assert "issues" in report
    
    # Check report content
    assert report["dataset"] == "test_data"
    assert report["record_count"] == len(sample_geodataframe)
    assert isinstance(report["issues"], list)

def test_validate_file(test_validator, sample_geodataframe, test_data_dir):
    """Test validation of a file."""
    # Save test data to parquet
    parquet_path = test_data_dir / "test.parquet"
    sample_geodataframe.to_parquet(parquet_path)
    
    # Test validation
    success, report = test_validator.validate_file(parquet_path)
    assert success is True
    assert len(report["issues"]) == 0
    
    # Test with non-existent file
    success, report = test_validator.validate_file(
        test_data_dir / "nonexistent.parquet"
    )
    assert success is False
    assert any(
        issue["type"] == "validation_error" 
        for issue in report["issues"]
    )

def test_custom_validation_rules(sample_geodataframe):
    """Test validator with custom validation rules."""
    # Create validator with custom rules
    validator = DataValidator(
        required_fields=['parcel_id', 'area', 'county'],
        geometry_type='Polygon',
        srid=2264,
        custom_rules={
            'area': lambda x: x > 0,  # Area must be positive
            'county': lambda x: x.startswith('County')  # County name format
        }
    )
    
    # Test with valid data
    success, report = validator.validate_geodataframe(
        sample_geodataframe,
        "test_data"
    )
    assert success is True
    
    # Test with invalid data
    invalid_gdf = sample_geodataframe.copy()
    invalid_gdf.loc[0, 'area'] = -100  # Invalid area
    invalid_gdf.loc[1, 'county'] = 'Invalid'  # Invalid county name
    
    success, report = validator.validate_geodataframe(
        invalid_gdf,
        "test_data"
    )
    assert success is False
    assert any(
        issue["type"] == "custom_validation_error" 
        for issue in report["issues"]
    )

def test_batch_validation(test_validator, large_geodataframe):
    """Test validation of large datasets."""
    # Split data into batches
    batch_size = 100
    batches = [
        large_geodataframe.iloc[i:i + batch_size]
        for i in range(0, len(large_geodataframe), batch_size)
    ]
    
    # Validate each batch
    for i, batch in enumerate(batches):
        success, report = test_validator.validate_geodataframe(
            batch,
            f"batch_{i}"
        )
        assert success is True
        assert report["record_count"] == len(batch)
        assert len(report["issues"]) == 0

def test_validation_performance(test_validator, large_geodataframe):
    """Test validation performance with large datasets."""
    import time
    
    # Measure validation time
    start_time = time.time()
    success, report = test_validator.validate_geodataframe(
        large_geodataframe,
        "test_data"
    )
    end_time = time.time()
    
    # Performance assertions
    validation_time = end_time - start_time
    assert validation_time < 5.0  # Should validate 1000 records in under 5 seconds
    assert success is True
    assert report["record_count"] == len(large_geodataframe) 