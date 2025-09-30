"""Tests for polygon utilities and clipping functionality."""

import pytest
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.transform import from_bounds

from bigmap.utils.polygon_utils import (
    load_polygon,
    clip_geotiff_to_polygon,
    clip_geotiffs_batch,
    get_polygon_bounds
)
from bigmap.utils.location_config import LocationConfig


@pytest.fixture
def sample_polygon():
    """Create a simple test polygon."""
    coords = [
        (-124.0, 42.0),
        (-123.0, 42.0),
        (-123.0, 43.0),
        (-124.0, 43.0),
        (-124.0, 42.0)
    ]
    poly = Polygon(coords)
    gdf = gpd.GeoDataFrame([{'id': 1, 'geometry': poly}], crs="EPSG:4326")
    return gdf


@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a sample GeoTIFF for testing."""
    # Create a simple 100x100 raster
    data = np.random.rand(1, 100, 100) * 100

    # Define bounds (larger than test polygon)
    bounds = (-125, 41, -122, 44)
    transform = from_bounds(*bounds, 100, 100)

    output_path = tmp_path / "test_raster.tif"

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=100,
        width=100,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(data)

    return output_path


def test_load_polygon_from_geodataframe(sample_polygon):
    """Test loading polygon from GeoDataFrame."""
    result = load_polygon(sample_polygon)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert result.crs == sample_polygon.crs


def test_load_polygon_from_file(tmp_path, sample_polygon):
    """Test loading polygon from GeoJSON file."""
    geojson_path = tmp_path / "test_polygon.geojson"
    sample_polygon.to_file(geojson_path, driver="GeoJSON")

    result = load_polygon(geojson_path)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1


def test_load_polygon_crs_transform(sample_polygon):
    """Test CRS transformation when loading polygon."""
    result = load_polygon(sample_polygon, target_crs="EPSG:3857")
    assert result.crs == "EPSG:3857"


def test_get_polygon_bounds(sample_polygon):
    """Test getting bounding box from polygon."""
    bounds = get_polygon_bounds(sample_polygon)
    assert len(bounds) == 4
    assert bounds[0] < bounds[2]  # xmin < xmax
    assert bounds[1] < bounds[3]  # ymin < ymax


def test_clip_geotiff_to_polygon(sample_geotiff, sample_polygon, tmp_path):
    """Test clipping a single GeoTIFF to polygon."""
    output_path = tmp_path / "clipped.tif"

    clipped_data, meta = clip_geotiff_to_polygon(
        sample_geotiff,
        sample_polygon,
        output_path=output_path
    )

    assert isinstance(clipped_data, np.ndarray)
    assert output_path.exists()
    assert clipped_data.shape[1] < 100  # Should be smaller than original
    assert clipped_data.shape[2] < 100


def test_clip_geotiff_without_saving(sample_geotiff, sample_polygon):
    """Test clipping without saving output."""
    clipped_data, meta = clip_geotiff_to_polygon(
        sample_geotiff,
        sample_polygon,
        output_path=None
    )

    assert isinstance(clipped_data, np.ndarray)
    assert 'transform' in meta
    assert 'height' in meta
    assert 'width' in meta


def test_clip_geotiffs_batch(tmp_path, sample_polygon):
    """Test batch clipping of multiple GeoTIFFs."""
    # Create multiple test rasters
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    # Create 3 test rasters
    for i in range(3):
        data = np.random.rand(1, 100, 100) * 100
        bounds = (-125, 41, -122, 44)
        transform = from_bounds(*bounds, 100, 100)

        raster_path = input_dir / f"species_{i:04d}.tif"
        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=100,
            width=100,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(data)

    # Clip all rasters
    clipped_files = clip_geotiffs_batch(
        input_dir,
        sample_polygon,
        output_dir
    )

    assert len(clipped_files) == 3
    assert all(f.exists() for f in clipped_files)
    assert all(f.parent == output_dir for f in clipped_files)


def test_location_config_from_polygon(tmp_path, sample_polygon):
    """Test creating LocationConfig from polygon."""
    geojson_path = tmp_path / "test_region.geojson"
    sample_polygon.to_file(geojson_path, driver="GeoJSON")

    config = LocationConfig.from_polygon(geojson_path, name="Test Region")

    assert config.location_name == "Test Region"
    assert config.has_polygon
    assert config.polygon_geojson is not None
    assert config.wgs84_bbox is not None


def test_location_config_polygon_properties(sample_polygon):
    """Test polygon-related properties of LocationConfig."""
    config = LocationConfig.from_polygon(sample_polygon, name="Test Area")

    # Test has_polygon
    assert config.has_polygon

    # Test polygon_geojson
    geojson = config.polygon_geojson
    assert isinstance(geojson, dict)
    assert 'type' in geojson
    assert 'coordinates' in geojson

    # Test polygon_gdf
    gdf = config.polygon_gdf
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.crs == "EPSG:4326"


def test_location_config_from_county_with_boundary():
    """Test creating LocationConfig for county with boundary storage."""
    # This test requires actual county data, so it may be skipped if data unavailable
    try:
        config = LocationConfig.from_county(
            "Lane", "Oregon",
            store_boundary=True
        )
        assert config.has_polygon
        assert config.polygon_geojson is not None
    except Exception as e:
        pytest.skip(f"County boundary data not available: {e}")


def test_location_config_save_and_load_with_polygon(tmp_path, sample_polygon):
    """Test saving and loading LocationConfig with polygon."""
    geojson_path = tmp_path / "test_region.geojson"
    sample_polygon.to_file(geojson_path, driver="GeoJSON")

    # Create and save config
    config_path = tmp_path / "config.yaml"
    config = LocationConfig.from_polygon(geojson_path)
    config.save(config_path)

    # Load config
    loaded_config = LocationConfig(config_path)

    assert loaded_config.has_polygon
    assert loaded_config.polygon_geojson is not None
    assert loaded_config.location_name == config.location_name


def test_clip_with_different_crs(tmp_path, sample_polygon):
    """Test clipping when polygon and raster have different CRS."""
    # Transform polygon to Web Mercator
    polygon_3857 = sample_polygon.to_crs("EPSG:3857")

    # Create raster in WGS84
    data = np.random.rand(1, 100, 100) * 100
    bounds = (-125, 41, -122, 44)
    transform = from_bounds(*bounds, 100, 100)

    raster_path = tmp_path / "test_raster_wgs84.tif"
    with rasterio.open(
        raster_path, 'w',
        driver='GTiff',
        height=100,
        width=100,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(data)

    # Clip should handle CRS transformation
    output_path = tmp_path / "clipped.tif"
    clipped_data, meta = clip_geotiff_to_polygon(
        raster_path,
        polygon_3857,
        output_path=output_path
    )

    assert isinstance(clipped_data, np.ndarray)
    assert output_path.exists()
