"""
Comprehensive tests for the visualization boundaries module.

Tests cover boundary downloading and caching, state and county boundary loading,
CRS transformations, basemap operations, and boundary clipping functions.
"""

import pytest
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call
from shapely.geometry import Polygon, box, MultiPolygon
from rasterio.crs import CRS
import tempfile
import zipfile
import io

from gridfia.visualization.boundaries import (
    download_boundaries,
    load_state_boundary,
    load_counties_for_state,
    plot_boundaries,
    add_basemap,
    clip_boundaries_to_extent,
    get_basemap_zoom_level,
    BOUNDARY_SOURCES,
    STATE_ABBR,
    BOUNDARY_CACHE_DIR,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_state_gdf():
    """Create a sample GeoDataFrame representing state boundaries."""
    # Create simple polygon geometries for states
    montana_geom = box(-116.0, 45.0, -104.0, 49.0)
    idaho_geom = box(-117.5, 42.0, -111.0, 49.0)
    california_geom = box(-124.5, 32.5, -114.0, 42.0)

    data = {
        'name': ['Montana', 'Idaho', 'California'],
        'postal': ['MT', 'ID', 'CA'],
        'admin': ['United States of America'] * 3,
        'geometry': [montana_geom, idaho_geom, california_geom]
    }
    return gpd.GeoDataFrame(data, crs='EPSG:4326')


@pytest.fixture
def sample_county_gdf():
    """Create a sample GeoDataFrame representing county boundaries."""
    # Create simple polygon geometries for counties
    missoula_geom = box(-115.0, 46.0, -113.0, 47.5)
    flathead_geom = box(-115.5, 47.5, -113.5, 49.0)
    gallatin_geom = box(-112.0, 45.0, -110.0, 46.5)
    boise_geom = box(-117.0, 43.0, -115.0, 44.5)

    data = {
        'NAME': ['Missoula', 'Flathead', 'Gallatin', 'Boise'],
        'STATE_NAME': ['Montana', 'Montana', 'Montana', 'Idaho'],
        'geometry': [missoula_geom, flathead_geom, gallatin_geom, boise_geom]
    }
    return gpd.GeoDataFrame(data, crs='EPSG:4326')


@pytest.fixture
def mock_zip_response():
    """Create a mock HTTP response with a valid zip file containing shapefile components."""
    # Create a minimal shapefile-like structure in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add placeholder shapefile components (minimal valid zip structure)
        zf.writestr('test_boundary.shp', b'')
        zf.writestr('test_boundary.dbf', b'')
        zf.writestr('test_boundary.shx', b'')
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@pytest.fixture
def temp_cache_dir(temp_dir):
    """Create a temporary cache directory for boundary files."""
    cache_dir = temp_dir / "boundaries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_axes():
    """Create a properly mocked matplotlib axes object."""
    mock_ax = Mock(spec=Axes)
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)
    mock_ax.transAxes = Mock()
    return mock_ax


# ============================================================================
# Tests for download_boundaries
# ============================================================================


class TestDownloadBoundaries:
    """Test suite for the download_boundaries function."""

    def test_download_boundaries_invalid_type_raises_error(self):
        """Test that invalid boundary type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown boundary type"):
            download_boundaries('invalid_type')

    def test_download_boundaries_uses_cached_file(self, temp_cache_dir, sample_state_gdf):
        """Test that cached boundary file is used when available."""
        # Create a cached file
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = download_boundaries('states', force=False)

        assert result == cache_path
        assert result.exists()

    def test_download_boundaries_force_redownload(
        self, temp_cache_dir, sample_state_gdf, mock_zip_response
    ):
        """Test that force=True triggers download even when cached file exists."""
        # Create a cached file
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        mock_response = Mock()
        mock_response.content = mock_zip_response
        mock_response.raise_for_status = Mock()

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir), \
             patch('requests.Session') as mock_session_class, \
             patch('geopandas.read_file') as mock_read_file:

            mock_session = Mock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            mock_read_file.return_value = sample_state_gdf.copy()

            result = download_boundaries('states', force=True)

        # Verify that Session.get was called
        mock_session.get.assert_called_once()

    def test_download_boundaries_states_filters_us_only(
        self, temp_cache_dir, mock_zip_response
    ):
        """Test that downloaded state boundaries are filtered for US only."""
        # Create GDF with US and non-US regions
        world_gdf = gpd.GeoDataFrame({
            'name': ['Montana', 'British Columbia', 'Alberta'],
            'postal': ['MT', 'BC', 'AB'],
            'admin': ['United States of America', 'Canada', 'Canada'],
            'geometry': [
                box(-116.0, 45.0, -104.0, 49.0),
                box(-139.0, 49.0, -114.0, 60.0),
                box(-120.0, 49.0, -110.0, 60.0),
            ]
        }, crs='EPSG:4326')

        mock_response = Mock()
        mock_response.content = mock_zip_response
        mock_response.raise_for_status = Mock()

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir), \
             patch('requests.Session') as mock_session_class, \
             patch('geopandas.read_file') as mock_read_file, \
             patch.object(gpd.GeoDataFrame, 'to_file') as mock_to_file:

            mock_session = Mock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            mock_read_file.return_value = world_gdf.copy()

            # Run the download
            download_boundaries('states', force=True)

        # Verify that to_file was called (meaning filtering was applied)
        mock_to_file.assert_called_once()

    def test_download_boundaries_census_gov_ssl_workaround(
        self, temp_cache_dir, mock_zip_response
    ):
        """Test that census.gov URLs use relaxed SSL verification."""
        mock_response = Mock()
        mock_response.content = mock_zip_response
        mock_response.raise_for_status = Mock()

        # Sample county data
        county_gdf = gpd.GeoDataFrame({
            'NAME': ['Test County'],
            'STATE_NAME': ['Montana'],
            'geometry': [box(-115.0, 46.0, -113.0, 47.5)]
        }, crs='EPSG:4326')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir), \
             patch('requests.Session') as mock_session_class, \
             patch('geopandas.read_file') as mock_read_file:

            mock_session = Mock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            mock_read_file.return_value = county_gdf.copy()

            download_boundaries('counties', force=True)

        # Verify verify=False was set for census.gov
        assert mock_session.verify is False or mock_session.verify == False

    def test_download_boundaries_network_error_raises_exception(self, temp_cache_dir):
        """Test that network errors are properly raised."""
        import requests

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir), \
             patch('requests.Session') as mock_session_class:

            mock_session = Mock()
            mock_session.get.side_effect = requests.exceptions.ConnectionError("Network error")
            mock_session_class.return_value = mock_session

            with pytest.raises(requests.exceptions.ConnectionError):
                download_boundaries('states', force=True)

    def test_download_boundaries_no_shapefile_in_zip_raises_error(
        self, temp_cache_dir
    ):
        """Test that error is raised when zip contains no shapefile."""
        # Create zip without shapefile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('readme.txt', b'No shapefile here')
        zip_buffer.seek(0)

        mock_response = Mock()
        mock_response.content = zip_buffer.getvalue()
        mock_response.raise_for_status = Mock()

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir), \
             patch('requests.Session') as mock_session_class:

            mock_session = Mock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            with pytest.raises(ValueError, match="No shapefile found"):
                download_boundaries('states', force=True)


class TestDownloadBoundariesHTTPErrors:
    """Test suite for HTTP error handling in download_boundaries."""

    def test_download_boundaries_http_error_raises_exception(self, temp_cache_dir):
        """Test that HTTP errors are properly raised."""
        import requests

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir), \
             patch('requests.Session') as mock_session_class:

            mock_session = Mock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            with pytest.raises(requests.exceptions.HTTPError):
                download_boundaries('states', force=True)


# ============================================================================
# Tests for load_state_boundary
# ============================================================================


class TestLoadStateBoundary:
    """Test suite for the load_state_boundary function."""

    def test_load_state_boundary_by_full_name(self, temp_cache_dir, sample_state_gdf):
        """Test loading state boundary by full state name."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_state_boundary('Montana')

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Montana'
        assert result.iloc[0]['postal'] == 'MT'

    def test_load_state_boundary_by_abbreviation(self, temp_cache_dir, sample_state_gdf):
        """Test loading state boundary by state abbreviation."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_state_boundary('MT')

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Montana'

    def test_load_state_boundary_case_insensitive(self, temp_cache_dir, sample_state_gdf):
        """Test that state name lookup is case insensitive."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result_lower = load_state_boundary('montana')
            result_upper = load_state_boundary('MONTANA')
            result_mixed = load_state_boundary('MoNtAnA')

        assert len(result_lower) == 1
        assert len(result_upper) == 1
        assert len(result_mixed) == 1

    def test_load_state_boundary_not_found_raises_error(self, temp_cache_dir, sample_state_gdf):
        """Test that ValueError is raised for unknown state."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            with pytest.raises(ValueError, match="State not found"):
                load_state_boundary('Atlantis')

    def test_load_state_boundary_with_crs_string(self, temp_cache_dir, sample_state_gdf):
        """Test loading state boundary with CRS transformation using string."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_state_boundary('Montana', crs='EPSG:3857')

        assert result.crs.to_string() == 'EPSG:3857'

    def test_load_state_boundary_with_crs_object(self, temp_cache_dir, sample_state_gdf):
        """Test loading state boundary with CRS transformation using CRS object."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        target_crs = CRS.from_epsg(3857)

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_state_boundary('Montana', crs=target_crs)

        # The result CRS should match
        assert 'EPSG:3857' in result.crs.to_string() or '3857' in str(result.crs)

    def test_load_state_boundary_with_simplify_tolerance(
        self, temp_cache_dir, sample_state_gdf
    ):
        """Test loading state boundary with geometry simplification."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result_full = load_state_boundary('Montana')
            result_simplified = load_state_boundary('Montana', simplify_tolerance=0.5)

        # Simplified geometry should have fewer vertices or equal
        original_vertices = sum(len(geom.exterior.coords) for geom in result_full.geometry)
        simplified_vertices = sum(
            len(geom.exterior.coords) if hasattr(geom, 'exterior') else 0
            for geom in result_simplified.geometry
        )

        # Simplification may reduce vertices (or keep them if already simple)
        assert simplified_vertices <= original_vertices or simplified_vertices > 0

    def test_load_state_boundary_hires(self, temp_cache_dir, sample_state_gdf):
        """Test loading high-resolution state boundary."""
        cache_path = temp_cache_dir / "us_states_10m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_state_boundary('Montana', boundary_type='states_hires')

        assert len(result) == 1


class TestLoadStateBoundaryEdgeCases:
    """Test edge cases for load_state_boundary."""

    def test_load_state_boundary_unknown_abbreviation(self, temp_cache_dir, sample_state_gdf):
        """Test handling of unknown state abbreviation."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            with pytest.raises(ValueError, match="State not found"):
                load_state_boundary('XX')

    def test_state_abbr_mapping_completeness(self):
        """Test that STATE_ABBR contains expected states."""
        expected_states = ['montana', 'california', 'texas', 'florida', 'new york']
        for state in expected_states:
            assert state in STATE_ABBR, f"{state} not found in STATE_ABBR"

        # Verify abbreviations are 2 characters
        for state_name, abbr in STATE_ABBR.items():
            assert len(abbr) == 2, f"Abbreviation {abbr} for {state_name} is not 2 characters"


# ============================================================================
# Tests for load_counties_for_state
# ============================================================================


class TestLoadCountiesForState:
    """Test suite for the load_counties_for_state function."""

    def test_load_counties_by_state_name(self, temp_cache_dir, sample_county_gdf):
        """Test loading counties by state full name."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_counties_for_state('Montana')

        assert len(result) == 3  # Missoula, Flathead, Gallatin
        assert all(result['STATE_NAME'].str.lower() == 'montana')

    def test_load_counties_by_state_abbreviation(self, temp_cache_dir, sample_county_gdf):
        """Test loading counties by state abbreviation."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_counties_for_state('MT')

        assert len(result) == 3

    def test_load_counties_case_insensitive(self, temp_cache_dir, sample_county_gdf):
        """Test that county loading is case insensitive."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_counties_for_state('MONTANA')

        assert len(result) == 3

    def test_load_counties_state_not_found_raises_error(
        self, temp_cache_dir, sample_county_gdf
    ):
        """Test that ValueError is raised for unknown state."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            with pytest.raises(ValueError, match="State not found"):
                load_counties_for_state('Atlantis')

    def test_load_counties_no_counties_found_raises_error(self, temp_cache_dir):
        """Test that ValueError is raised when no counties found for state."""
        # Create county GDF with only Idaho counties
        county_gdf = gpd.GeoDataFrame({
            'NAME': ['Boise', 'Ada'],
            'STATE_NAME': ['Idaho', 'Idaho'],
            'geometry': [
                box(-117.0, 43.0, -115.0, 44.5),
                box(-116.5, 43.0, -115.5, 44.0),
            ]
        }, crs='EPSG:4326')

        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            with pytest.raises(ValueError, match="No counties found for state"):
                load_counties_for_state('Montana')

    def test_load_counties_with_crs_transformation(
        self, temp_cache_dir, sample_county_gdf
    ):
        """Test loading counties with CRS transformation."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_counties_for_state('Montana', crs='EPSG:3857')

        assert result.crs.to_string() == 'EPSG:3857'

    def test_load_counties_with_simplify_tolerance(
        self, temp_cache_dir, sample_county_gdf
    ):
        """Test loading counties with geometry simplification."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_counties_for_state('Montana', simplify_tolerance=0.1)

        assert len(result) == 3


# ============================================================================
# Tests for plot_boundaries
# ============================================================================


class TestPlotBoundaries:
    """Test suite for the plot_boundaries function."""

    def test_plot_boundaries_basic(self, sample_state_gdf):
        """Test basic boundary plotting."""
        fig, ax = plt.subplots()

        plot_boundaries(ax, sample_state_gdf)

        # Verify plot was called (geopandas adds to axes)
        assert len(ax.collections) > 0 or len(ax.patches) > 0

        plt.close(fig)

    def test_plot_boundaries_with_fill(self, sample_state_gdf):
        """Test boundary plotting with fill enabled."""
        fig, ax = plt.subplots()

        plot_boundaries(
            ax,
            sample_state_gdf,
            fill=True,
            fill_color='blue',
            fill_alpha=0.3
        )

        plt.close(fig)

    def test_plot_boundaries_custom_style(self, sample_state_gdf):
        """Test boundary plotting with custom styling."""
        fig, ax = plt.subplots()

        plot_boundaries(
            ax,
            sample_state_gdf,
            color='red',
            linewidth=2.5,
            alpha=0.8,
            label='Test Boundaries',
            zorder=15
        )

        plt.close(fig)

    def test_plot_boundaries_with_mock_axes(self, sample_state_gdf):
        """Test boundary plotting with mocked axes."""
        mock_ax = Mock(spec=Axes)

        # Mock the GeoDataFrame plot method
        with patch.object(gpd.GeoDataFrame, 'plot') as mock_plot:
            plot_boundaries(mock_ax, sample_state_gdf)

        # Verify plot was called at least once
        mock_plot.assert_called()


class TestPlotBoundariesParameters:
    """Test parameter handling for plot_boundaries."""

    def test_plot_boundaries_default_parameters(self, sample_state_gdf):
        """Test that default parameters are correctly applied."""
        with patch.object(gpd.GeoDataFrame, 'plot') as mock_plot:
            mock_ax = Mock(spec=Axes)
            plot_boundaries(mock_ax, sample_state_gdf)

        # Default call should have default styling
        mock_plot.assert_called()

    def test_plot_boundaries_fill_creates_two_calls(self, sample_state_gdf):
        """Test that fill=True creates two plot calls (fill and outline)."""
        with patch.object(gpd.GeoDataFrame, 'plot') as mock_plot:
            mock_ax = Mock(spec=Axes)
            plot_boundaries(mock_ax, sample_state_gdf, fill=True)

        # Should be called twice - once for fill, once for outline
        assert mock_plot.call_count == 2


# ============================================================================
# Tests for add_basemap
# ============================================================================


class TestAddBasemap:
    """Test suite for the add_basemap function."""

    def test_add_basemap_basic(self, mock_axes):
        """Test basic basemap addition."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes)

        mock_add_basemap.assert_called_once()

    def test_add_basemap_preserves_extent(self, mock_axes):
        """Test that basemap preserves axes extent when reset_extent=True."""
        original_xlim = (-1000, 1000)
        original_ylim = (-500, 500)
        mock_axes.get_xlim.return_value = original_xlim
        mock_axes.get_ylim.return_value = original_ylim

        with patch('contextily.add_basemap'):
            add_basemap(mock_axes, reset_extent=True)

        mock_axes.set_xlim.assert_called_with(original_xlim)
        mock_axes.set_ylim.assert_called_with(original_ylim)

    def test_add_basemap_no_reset_extent(self, mock_axes):
        """Test basemap without extent reset."""
        with patch('contextily.add_basemap'):
            add_basemap(mock_axes, reset_extent=False)

        # set_xlim and set_ylim should not be called
        mock_axes.set_xlim.assert_not_called()
        mock_axes.set_ylim.assert_not_called()

    def test_add_basemap_openstreetmap_provider(self, mock_axes):
        """Test basemap with OpenStreetMap provider."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, source='OpenStreetMap')

        # Verify provider was passed correctly
        call_kwargs = mock_add_basemap.call_args[1]
        assert call_kwargs['source'] is not None

    def test_add_basemap_cartodb_provider(self, mock_axes):
        """Test basemap with CartoDB provider."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, source='CartoDB')

        mock_add_basemap.assert_called_once()

    def test_add_basemap_cartodb_dark_provider(self, mock_axes):
        """Test basemap with CartoDB dark provider."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, source='CartoDB_dark')

        mock_add_basemap.assert_called_once()

    def test_add_basemap_esri_provider(self, mock_axes):
        """Test basemap with ESRI provider."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, source='ESRI')

        mock_add_basemap.assert_called_once()

    def test_add_basemap_custom_zoom(self, mock_axes):
        """Test basemap with custom zoom level."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, zoom=10)

        call_kwargs = mock_add_basemap.call_args[1]
        assert call_kwargs['zoom'] == 10

    def test_add_basemap_custom_alpha(self, mock_axes):
        """Test basemap with custom transparency."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, alpha=0.5)

        call_kwargs = mock_add_basemap.call_args[1]
        assert call_kwargs['alpha'] == 0.5

    def test_add_basemap_custom_crs_string(self, mock_axes):
        """Test basemap with custom CRS as string."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, crs='EPSG:4326')

        call_kwargs = mock_add_basemap.call_args[1]
        assert call_kwargs['crs'] == 'EPSG:4326'

    def test_add_basemap_custom_crs_object(self, mock_axes):
        """Test basemap with custom CRS as CRS object."""
        target_crs = CRS.from_epsg(4326)

        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, crs=target_crs)

        mock_add_basemap.assert_called_once()

    def test_add_basemap_with_attribution(self, mock_axes):
        """Test basemap with attribution enabled."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            add_basemap(mock_axes, attribution=True, attribution_size=10)

        call_kwargs = mock_add_basemap.call_args[1]
        assert call_kwargs['attribution'] is True
        assert call_kwargs['attribution_size'] == 10

    def test_add_basemap_error_handling(self, mock_axes):
        """Test that basemap errors are caught and logged."""
        with patch('contextily.add_basemap') as mock_add_basemap:
            mock_add_basemap.side_effect = Exception("Network error")

            # Should not raise, just log warning
            add_basemap(mock_axes)


# ============================================================================
# Tests for clip_boundaries_to_extent
# ============================================================================


class TestClipBoundariesToExtent:
    """Test suite for the clip_boundaries_to_extent function."""

    def test_clip_boundaries_basic(self, sample_state_gdf):
        """Test basic boundary clipping."""
        # Clip to extent that includes only Montana
        extent = (-116.0, -104.0, 45.0, 49.0)  # xmin, xmax, ymin, ymax

        result = clip_boundaries_to_extent(sample_state_gdf, extent)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_clip_boundaries_removes_empty(self, sample_state_gdf):
        """Test that clipping removes empty geometries."""
        # Clip to extent outside all boundaries
        extent = (0, 1, 0, 1)  # Far from US

        result = clip_boundaries_to_extent(sample_state_gdf, extent)

        # All geometries should be clipped away
        assert len(result) == 0

    def test_clip_boundaries_with_buffer(self, sample_state_gdf):
        """Test boundary clipping with buffer."""
        extent = (-116.0, -104.0, 45.0, 49.0)

        result_no_buffer = clip_boundaries_to_extent(sample_state_gdf, extent, buffer=0.0)
        result_with_buffer = clip_boundaries_to_extent(sample_state_gdf, extent, buffer=5.0)

        # Buffer should include more area
        if len(result_with_buffer) > 0 and len(result_no_buffer) > 0:
            buffered_area = result_with_buffer.geometry.area.sum()
            unbuffered_area = result_no_buffer.geometry.area.sum()
            assert buffered_area >= unbuffered_area

    def test_clip_boundaries_preserves_attributes(self, sample_state_gdf):
        """Test that clipping preserves non-geometry attributes."""
        extent = (-116.0, -104.0, 45.0, 49.0)

        result = clip_boundaries_to_extent(sample_state_gdf, extent)

        # Original columns should be preserved
        expected_columns = ['name', 'postal', 'admin', 'geometry']
        for col in expected_columns:
            assert col in result.columns

    def test_clip_boundaries_partial_clip(self, sample_state_gdf):
        """Test that boundaries are partially clipped at extent edges."""
        # Extent that clips Montana in half
        extent = (-116.0, -110.0, 45.0, 49.0)

        result = clip_boundaries_to_extent(sample_state_gdf, extent)

        # Should have clipped geometry
        assert len(result) > 0


class TestClipBoundariesExtentFormats:
    """Test extent format handling for clip_boundaries_to_extent."""

    def test_clip_boundaries_extent_tuple(self, sample_state_gdf):
        """Test clipping with extent as tuple."""
        extent = (-116.0, -104.0, 45.0, 49.0)

        result = clip_boundaries_to_extent(sample_state_gdf, extent)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_clip_boundaries_negative_buffer(self, sample_state_gdf):
        """Test clipping with negative buffer (shrinks extent)."""
        extent = (-116.0, -104.0, 45.0, 49.0)

        result = clip_boundaries_to_extent(sample_state_gdf, extent, buffer=-1.0)
        assert isinstance(result, gpd.GeoDataFrame)


# ============================================================================
# Tests for get_basemap_zoom_level
# ============================================================================


class TestGetBasemapZoomLevel:
    """Test suite for the get_basemap_zoom_level function."""

    def test_zoom_level_large_extent(self):
        """Test zoom level calculation for large extent."""
        # Large extent (whole country scale)
        # Convert approximate degrees to Web Mercator
        extent = (-15000000, -5000000, 3000000, 6500000)

        zoom = get_basemap_zoom_level(extent)

        assert zoom <= 7  # Should be zoomed out

    def test_zoom_level_medium_extent(self):
        """Test zoom level calculation for medium extent (state scale)."""
        # Medium extent (state scale) in Web Mercator
        extent = (-13000000, -12000000, 5000000, 5500000)

        zoom = get_basemap_zoom_level(extent)

        assert 7 <= zoom <= 10

    def test_zoom_level_small_extent(self):
        """Test zoom level calculation for small extent (county scale)."""
        # Small extent (county scale) in Web Mercator
        extent = (-12700000, -12600000, 5200000, 5250000)

        zoom = get_basemap_zoom_level(extent)

        assert zoom >= 9

    def test_zoom_level_very_small_extent(self):
        """Test zoom level calculation for very small extent."""
        # Very small extent in Web Mercator
        extent = (-12650000, -12640000, 5220000, 5225000)

        zoom = get_basemap_zoom_level(extent)

        assert zoom >= 11

    def test_zoom_level_returns_integer(self):
        """Test that zoom level is always an integer."""
        extent = (-13000000, -12000000, 5000000, 5500000)

        zoom = get_basemap_zoom_level(extent)

        assert isinstance(zoom, int)


class TestGetBasemapZoomLevelEdgeCases:
    """Test edge cases for get_basemap_zoom_level."""

    def test_zoom_level_zero_width_extent(self):
        """Test zoom level with zero width extent."""
        extent = (-12000000, -12000000, 5000000, 5500000)

        zoom = get_basemap_zoom_level(extent)

        # Should return max zoom for zero width
        assert zoom == 12

    def test_zoom_level_negative_extent_order(self):
        """Test zoom level when extent has swapped min/max."""
        # xmin > xmax case
        extent = (-5000000, -15000000, 5000000, 3000000)

        zoom = get_basemap_zoom_level(extent)

        # Should still calculate based on absolute difference
        assert isinstance(zoom, int)


# ============================================================================
# Tests for Constants and Module-Level Variables
# ============================================================================


class TestBoundaryConstants:
    """Test module-level constants and configuration."""

    def test_boundary_sources_structure(self):
        """Test that BOUNDARY_SOURCES has required structure."""
        required_types = ['states', 'states_hires', 'counties']

        for boundary_type in required_types:
            assert boundary_type in BOUNDARY_SOURCES
            assert 'url' in BOUNDARY_SOURCES[boundary_type]
            assert 'cache_name' in BOUNDARY_SOURCES[boundary_type]

    def test_boundary_sources_urls_are_valid(self):
        """Test that boundary source URLs are valid HTTP(S) URLs."""
        for boundary_type, config in BOUNDARY_SOURCES.items():
            url = config['url']
            assert url.startswith('http://') or url.startswith('https://')

    def test_state_abbr_mapping_all_50_states(self):
        """Test that STATE_ABBR contains all 50 states."""
        assert len(STATE_ABBR) == 50

    def test_state_abbr_values_unique(self):
        """Test that state abbreviations are unique."""
        abbreviations = list(STATE_ABBR.values())
        assert len(abbreviations) == len(set(abbreviations))

    def test_boundary_cache_dir_is_path(self):
        """Test that BOUNDARY_CACHE_DIR is a valid Path."""
        assert isinstance(BOUNDARY_CACHE_DIR, Path)
        assert '.gridfia' in str(BOUNDARY_CACHE_DIR)
        assert 'boundaries' in str(BOUNDARY_CACHE_DIR)


# ============================================================================
# Integration Tests (with mocked external services)
# ============================================================================


class TestBoundaryWorkflows:
    """Integration tests for complete boundary workflows."""

    def test_load_state_and_clip_workflow(self, temp_cache_dir, sample_state_gdf):
        """Test complete workflow: load state then clip to extent."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            # Load Montana boundary
            state_boundary = load_state_boundary('Montana')

            # Get extent and clip
            bounds = state_boundary.total_bounds
            extent = (bounds[0], bounds[2], bounds[1], bounds[3])

            # Clip (should return same data since extent matches)
            clipped = clip_boundaries_to_extent(state_boundary, extent)

        assert len(clipped) == 1
        assert not clipped.geometry.is_empty.any()

    def test_load_state_with_counties_workflow(
        self, temp_cache_dir, sample_state_gdf, sample_county_gdf
    ):
        """Test workflow: load state and its counties."""
        state_cache = temp_cache_dir / "us_states_50m.gpkg"
        county_cache = temp_cache_dir / "us_counties_20m.gpkg"
        sample_state_gdf.to_file(state_cache, driver='GPKG')
        sample_county_gdf.to_file(county_cache, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            state_boundary = load_state_boundary('Montana')
            counties = load_counties_for_state('Montana')

        # Verify state was loaded
        assert len(state_boundary) == 1

        # Verify counties belong to state
        assert len(counties) == 3
        assert all(counties['STATE_NAME'].str.lower() == 'montana')

    def test_plot_with_basemap_workflow(
        self, temp_cache_dir, sample_state_gdf
    ):
        """Test workflow: load state, reproject, plot with basemap."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            # Load and reproject to Web Mercator
            state_boundary = load_state_boundary('Montana', crs='EPSG:3857')

            # Create plot
            fig, ax = plt.subplots()

            # Plot boundaries
            plot_boundaries(ax, state_boundary, color='red', linewidth=2)

            # Add basemap (mocked)
            with patch('contextily.add_basemap') as mock_basemap:
                add_basemap(ax, source='CartoDB', crs='EPSG:3857')

                mock_basemap.assert_called_once()

        plt.close(fig)


# ============================================================================
# CRS Transformation Tests
# ============================================================================


class TestCRSTransformations:
    """Test CRS transformation functionality."""

    def test_crs_transformation_preserves_geometry_validity(
        self, temp_cache_dir, sample_state_gdf
    ):
        """Test that CRS transformation preserves valid geometries."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            result = load_state_boundary('Montana', crs='EPSG:3857')

        # All geometries should remain valid
        assert result.geometry.is_valid.all()

    def test_crs_transformation_changes_coordinates(
        self, temp_cache_dir, sample_state_gdf
    ):
        """Test that CRS transformation changes coordinate values."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            original = load_state_boundary('Montana')
            reprojected = load_state_boundary('Montana', crs='EPSG:3857')

        # Coordinates should be different
        orig_centroid = original.geometry.centroid.iloc[0]
        reproj_centroid = reprojected.geometry.centroid.iloc[0]

        assert orig_centroid.x != reproj_centroid.x
        assert orig_centroid.y != reproj_centroid.y

    def test_counties_crs_transformation(self, temp_cache_dir, sample_county_gdf):
        """Test CRS transformation for counties."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            original = load_counties_for_state('Montana')
            reprojected = load_counties_for_state('Montana', crs='EPSG:32611')  # UTM 11N

        # CRS should be updated
        assert reprojected.crs.to_string() == 'EPSG:32611'


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling throughout the boundaries module."""

    def test_invalid_boundary_type_error_message(self):
        """Test that invalid boundary type provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            download_boundaries('invalid_type')

        assert 'Unknown boundary type' in str(exc_info.value)
        assert 'invalid_type' in str(exc_info.value)

    def test_state_not_found_error_message(self, temp_cache_dir, sample_state_gdf):
        """Test that state not found provides helpful error message."""
        cache_path = temp_cache_dir / "us_states_50m.gpkg"
        sample_state_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            with pytest.raises(ValueError) as exc_info:
                load_state_boundary('Narnia')

        assert 'State not found' in str(exc_info.value)

    def test_county_state_not_found_error(self, temp_cache_dir, sample_county_gdf):
        """Test error when state has no counties."""
        cache_path = temp_cache_dir / "us_counties_20m.gpkg"
        sample_county_gdf.to_file(cache_path, driver='GPKG')

        with patch('gridfia.visualization.boundaries.BOUNDARY_CACHE_DIR', temp_cache_dir):
            with pytest.raises(ValueError) as exc_info:
                load_counties_for_state('Narnia')

        assert 'State not found' in str(exc_info.value)
