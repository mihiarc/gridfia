"""
Comprehensive tests for the visualization mapper module.

Tests cover map creation, visualization functions, matplotlib integration,
color mapping, legend generation, and spatial data visualization.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import zarr
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import warnings

from gridfia.visualization.mapper import ZarrMapper


def setup_mock_axes():
    """Helper function to create properly mocked matplotlib axes."""
    mock_ax = Mock(spec=Axes)
    mock_ax.transAxes = Mock()  # Required for text annotations
    return mock_ax


@pytest.fixture
def complete_zarr_store(temp_dir):
    """Create a complete zarr store with all metadata for mapper testing."""
    zarr_path = temp_dir / "complete_zarr_store.zarr"

    # Create zarr group (not just array)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='w')

    # Create biomass array with realistic data
    n_species, height, width = 6, 100, 100
    biomass = root.create_array(
        'biomass',
        shape=(n_species, height, width),
        chunks=(1, 50, 50),
        dtype='f4',
        fill_value=0.0
    )

    # Generate realistic test data
    np.random.seed(42)
    total_biomass = np.zeros((height, width), dtype=np.float32)

    # Create species data with different spatial patterns
    for i in range(1, n_species):
        if i == 1:  # Dominant species
            data = np.random.exponential(20, (height, width)).astype(np.float32)
            data[data > 100] = 100  # Cap at reasonable values
        elif i == 2:  # Common species
            data = np.random.gamma(2, 10, (height, width)).astype(np.float32)
        elif i == 3:  # Rare species - clustered
            data = np.zeros((height, width), dtype=np.float32)
            center_y, center_x = height//2, width//2
            data[center_y-15:center_y+15, center_x-15:center_x+15] = np.random.exponential(5, (30, 30))
        elif i == 4:  # Edge species
            data = np.zeros((height, width), dtype=np.float32)
            data[:5, :] = np.random.exponential(8, (5, width))
            data[-5:, :] = np.random.exponential(8, (5, width))
        else:  # Scattered species
            data = np.random.exponential(3, (height, width)).astype(np.float32)
            data[np.random.rand(height, width) < 0.7] = 0  # Make sparse

        biomass[i] = data
        total_biomass += data

    # Set total biomass (species 0)
    biomass[0] = total_biomass

    # Create species codes and names arrays
    species_codes_data = ['0000', '0202', '0122', '0318', '0541', '0802']
    species_names_data = [
        'All Species Combined',
        'Douglas-fir',
        'Ponderosa Pine',
        'Sugar Maple',
        'Paper Birch',
        'Quaking Aspen'
    ]

    species_codes = root.create_array('species_codes', shape=(len(species_codes_data),), dtype='U10')
    species_codes[:] = species_codes_data

    species_names = root.create_array('species_names', shape=(len(species_names_data),), dtype='U50')
    species_names[:] = species_names_data

    # Add required metadata to root
    root.attrs['crs'] = 'EPSG:3857'
    root.attrs['transform'] = [30.0, 0.0, -2000000.0, 0.0, -30.0, 1000000.0]  # 30m resolution
    root.attrs['bounds'] = [-2000000.0, 997000.0, -1997000.0, 1000000.0]
    root.attrs['num_species'] = n_species
    root.attrs['description'] = 'Test forest biomass data'
    root.attrs['units'] = 'Mg/ha'

    return zarr_path


@pytest.fixture
def minimal_zarr_store(temp_dir):
    """Create a minimal zarr store for testing edge cases."""
    zarr_path = temp_dir / "minimal_zarr_store.zarr"

    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='w')

    # Single species plus total
    biomass = root.create_array(
        'biomass',
        shape=(2, 50, 50),
        chunks=(1, 50, 50),
        dtype='f4',
        fill_value=0.0
    )

    # Simple data
    data = np.ones((50, 50), dtype=np.float32) * 25
    biomass[0] = data  # Total
    biomass[1] = data  # Single species

    # Include species codes and names (required for diversity/richness calculations)
    species_codes_data = ['0000', '0202']
    species_names_data = ['All Species Combined', 'Douglas-fir']

    species_codes = root.create_array('species_codes', shape=(len(species_codes_data),), dtype='U10')
    species_codes[:] = species_codes_data

    species_names = root.create_array('species_names', shape=(len(species_names_data),), dtype='U50')
    species_names[:] = species_names_data

    # Minimal metadata
    root.attrs['crs'] = 'EPSG:4326'
    root.attrs['transform'] = [1.0, 0.0, 0.0, 0.0, -1.0, 50.0]
    root.attrs['bounds'] = [0.0, 0.0, 50.0, 50.0]
    root.attrs['num_species'] = 2

    return zarr_path


@pytest.fixture
def empty_zarr_store(temp_dir):
    """Create an empty zarr store for zero data testing."""
    zarr_path = temp_dir / "empty_zarr_store.zarr"

    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='w')

    # All zeros
    biomass = root.create_array(
        'biomass',
        shape=(3, 20, 20),
        chunks=(1, 20, 20),
        dtype='f4',
        fill_value=0.0
    )
    biomass[:] = 0  # All zeros

    # Include species codes and names (required for diversity/richness calculations)
    species_codes_data = ['0000', '0202', '0122']
    species_names_data = ['All Species Combined', 'Douglas-fir', 'Ponderosa Pine']

    species_codes = root.create_array('species_codes', shape=(len(species_codes_data),), dtype='U10')
    species_codes[:] = species_codes_data

    species_names = root.create_array('species_names', shape=(len(species_names_data),), dtype='U50')
    species_names[:] = species_names_data

    root.attrs['crs'] = 'EPSG:4326'
    root.attrs['transform'] = [1.0, 0.0, 0.0, 0.0, -1.0, 20.0]
    root.attrs['bounds'] = [0.0, 0.0, 20.0, 20.0]
    root.attrs['num_species'] = 3

    return zarr_path


class TestZarrMapperInitialization:
    """Test suite for ZarrMapper initialization and validation."""

    def test_successful_initialization(self, complete_zarr_store):
        """Test successful mapper initialization with complete zarr store."""
        mapper = ZarrMapper(complete_zarr_store)

        assert mapper.zarr_path == Path(complete_zarr_store)
        assert mapper._store is not None  # Now uses ZarrStore internally
        assert mapper.biomass is not None
        assert mapper.num_species == 6
        assert mapper.crs == CRS.from_string('EPSG:3857')
        assert len(mapper.species_codes) == 6  # Now returns a list
        assert len(mapper.species_names) == 6  # Now returns a list
        assert isinstance(mapper.transform, Affine)
        assert len(mapper.bounds) == 4
        assert mapper._diversity_cache == {}

    def test_initialization_with_string_path(self, complete_zarr_store):
        """Test initialization with string path instead of Path object."""
        mapper = ZarrMapper(str(complete_zarr_store))
        assert mapper.zarr_path == Path(complete_zarr_store)
        assert mapper.num_species == 6

    def test_initialization_nonexistent_path(self, temp_dir):
        """Test initialization with non-existent zarr path."""
        nonexistent_path = temp_dir / "nonexistent.zarr"

        with pytest.raises(FileNotFoundError, match="Zarr store not found"):
            ZarrMapper(nonexistent_path)

    def test_initialization_minimal_metadata(self, minimal_zarr_store):
        """Test initialization with minimal metadata."""
        mapper = ZarrMapper(minimal_zarr_store)

        assert mapper.num_species == 2
        assert mapper.crs == CRS.from_string('EPSG:4326')
        # Species codes/names are now returned as lists by ZarrStore
        assert len(mapper.species_codes) == 2
        assert len(mapper.species_names) == 2

    @patch('gridfia.visualization.mapper.console')
    def test_console_output_during_initialization(self, mock_console, complete_zarr_store):
        """Test that appropriate console output is generated during initialization."""
        mapper = ZarrMapper(complete_zarr_store)

        # Check console.print was called with expected messages
        assert mock_console.print.call_count >= 4
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Loaded Zarr store" in call for call in calls)
        assert any("Shape:" in call for call in calls)
        assert any("CRS:" in call for call in calls)
        assert any("Species:" in call for call in calls)


class TestSpeciesInfo:
    """Test suite for species information retrieval."""

    def test_get_species_info_complete(self, complete_zarr_store):
        """Test getting species info from complete store."""
        mapper = ZarrMapper(complete_zarr_store)
        species_info = mapper.get_species_info()

        assert len(species_info) == 6
        assert all(isinstance(info, dict) for info in species_info)
        assert all('index' in info for info in species_info)
        assert all('code' in info for info in species_info)
        assert all('name' in info for info in species_info)

        # Check first species
        assert species_info[0]['index'] == 0
        assert species_info[0]['code'] == '0000'
        assert species_info[0]['name'] == 'All Species Combined'

        # Check last species
        assert species_info[-1]['index'] == 5
        assert species_info[-1]['code'] == '0802'
        assert species_info[-1]['name'] == 'Quaking Aspen'

    def test_get_species_info_missing_metadata(self, minimal_zarr_store):
        """Test species info with missing species codes/names."""
        mapper = ZarrMapper(minimal_zarr_store)
        species_info = mapper.get_species_info()

        assert len(species_info) == 2

        # Should generate correct species info from the minimal store
        for i, info in enumerate(species_info):
            assert info['index'] == i
            assert 'code' in info
            assert 'name' in info
            # Check actual values from the minimal store (species codes are now lists)
            if i < len(mapper.species_codes):
                assert info['code'] == mapper.species_codes[i]
                assert info['name'] == mapper.species_names[i]


class TestDataNormalization:
    """Test suite for data normalization functionality."""

    def test_normalize_data_default_percentiles(self, complete_zarr_store):
        """Test data normalization with default percentile clipping."""
        mapper = ZarrMapper(complete_zarr_store)

        # Create test data with outliers
        data = np.array([[1, 2, 3, 100], [4, 5, 6, 200]], dtype=np.float32)
        normalized = mapper._normalize_data(data)

        assert normalized.shape == data.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert np.all(np.isfinite(normalized))

    def test_normalize_data_explicit_vmin_vmax(self, complete_zarr_store):
        """Test data normalization with explicit vmin/vmax values."""
        mapper = ZarrMapper(complete_zarr_store)

        data = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        normalized = mapper._normalize_data(data, vmin=20, vmax=60)

        assert normalized.min() >= 0
        assert normalized.max() <= 1
        # Values below vmin should be 0, above vmax should be 1
        assert normalized[0, 0] == 0  # 10 < 20
        assert normalized[1, 3] == 1  # 80 > 60

    def test_normalize_data_with_nans_and_infs(self, complete_zarr_store):
        """Test normalization with NaN and infinite values."""
        mapper = ZarrMapper(complete_zarr_store)

        data = np.array([[1, 2, np.nan, 4], [5, np.inf, 7, -np.inf]], dtype=np.float32)
        normalized = mapper._normalize_data(data)

        assert normalized.shape == data.shape
        # NaN and inf should be handled gracefully
        finite_mask = np.isfinite(data)
        assert np.all(np.isfinite(normalized[finite_mask]))

    def test_normalize_data_all_zeros(self, complete_zarr_store):
        """Test normalization with all zero data."""
        mapper = ZarrMapper(complete_zarr_store)

        data = np.zeros((10, 10), dtype=np.float32)
        normalized = mapper._normalize_data(data)

        assert normalized.shape == data.shape
        assert np.all(normalized == 0)

    def test_normalize_data_custom_percentiles(self, complete_zarr_store):
        """Test normalization with custom percentile values."""
        mapper = ZarrMapper(complete_zarr_store)

        # Data with more extreme outliers
        data = np.concatenate([
            np.ones(90),  # 90% of data is 1
            np.array([100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000])  # 10% outliers
        ]).reshape(10, 10)

        # Use tighter percentiles
        normalized = mapper._normalize_data(data, percentile=(5, 95))

        assert normalized.shape == data.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestExtentCalculation:
    """Test suite for extent calculation functionality."""

    def test_get_extent_default_transform(self, complete_zarr_store):
        """Test extent calculation with default transform."""
        mapper = ZarrMapper(complete_zarr_store)
        extent = mapper._get_extent()

        assert len(extent) == 4
        left, right, bottom, top = extent

        # Check order and relationships
        assert left < right
        assert bottom < top

        # Should match bounds from transform
        expected_left = mapper.transform.c
        expected_right = expected_left + mapper.biomass.shape[2] * mapper.transform.a
        assert abs(extent[0] - expected_left) < 1e-6
        assert abs(extent[1] - expected_right) < 1e-6

    def test_get_extent_custom_transform(self, complete_zarr_store):
        """Test extent calculation with custom transform."""
        mapper = ZarrMapper(complete_zarr_store)

        # Custom transform with different resolution
        custom_transform = Affine(60.0, 0.0, -3000000.0, 0.0, -60.0, 2000000.0)
        extent = mapper._get_extent(custom_transform)

        assert len(extent) == 4
        left, right, bottom, top = extent

        # Verify calculations with custom transform
        expected_left = custom_transform.c
        expected_right = expected_left + mapper.biomass.shape[2] * custom_transform.a
        expected_top = custom_transform.f
        expected_bottom = expected_top + mapper.biomass.shape[1] * custom_transform.e

        assert abs(extent[0] - expected_left) < 1e-6
        assert abs(extent[1] - expected_right) < 1e-6
        assert abs(extent[2] - expected_bottom) < 1e-6
        assert abs(extent[3] - expected_top) < 1e-6


@patch('matplotlib.pyplot.subplots')
@patch('matplotlib.pyplot.colorbar')
@patch('matplotlib.pyplot.tight_layout')
class TestSpeciesMapCreation:
    """Test suite for species map creation functionality."""

    def test_create_species_map_by_index(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test creating species map using species index."""
        # Setup mocks
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_species_map(species=1)

        assert fig is mock_fig
        assert ax is mock_ax
        mock_subplots.assert_called_once()
        mock_ax.imshow.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_tight_layout.assert_called_once()

    def test_create_species_map_by_code(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test creating species map using species code."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_species_map(species='0202')  # Douglas-fir

        assert fig is mock_fig
        assert ax is mock_ax
        mock_ax.imshow.assert_called_once()

        # Verify image was created with correct parameters
        imshow_call = mock_ax.imshow.call_args
        args, kwargs = imshow_call
        assert 'cmap' in kwargs
        assert 'extent' in kwargs
        assert 'origin' in kwargs
        assert kwargs['origin'] == 'upper'
        assert kwargs['interpolation'] == 'nearest'
        assert kwargs['aspect'] == 'equal'

    def test_create_species_map_invalid_code(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test error handling for invalid species code."""
        mapper = ZarrMapper(complete_zarr_store)

        with pytest.raises(ValueError, match="Species code 'INVALID' not found"):
            mapper.create_species_map(species='INVALID')

    def test_create_species_map_invalid_index(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test error handling for out-of-range species index."""
        mapper = ZarrMapper(complete_zarr_store)

        with pytest.raises(ValueError, match="Species index 10 out of range"):
            mapper.create_species_map(species=10)

    def test_create_species_map_custom_parameters(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test species map creation with custom parameters."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_species_map(
            species=1,
            cmap='plasma',
            vmin=5.0,
            vmax=50.0,
            title='Custom Title',
            colorbar=True,
            colorbar_label='Custom Biomass (t/ha)',
            show_bounds=False
        )

        # Verify imshow called with custom colormap
        imshow_call = mock_ax.imshow.call_args
        args, kwargs = imshow_call
        assert kwargs['cmap'] == 'plasma'

        # Verify title setting
        mock_ax.set_title.assert_called_once()
        title_call = mock_ax.set_title.call_args[0][0]
        assert title_call == 'Custom Title'

        # Verify colorbar creation with custom label
        mock_colorbar.assert_called_once()
        colorbar_call = mock_colorbar.call_args
        args, kwargs = colorbar_call
        assert kwargs['label'] == 'Custom Biomass (t/ha)'

    def test_create_species_map_no_colorbar(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test species map creation without colorbar."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_species_map(species=1, colorbar=False)

        mock_colorbar.assert_not_called()

    def test_create_species_map_provided_fig_ax(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test species map creation with provided figure and axes."""
        # Don't create new figure
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_species_map(species=1, fig_ax=(mock_fig, mock_ax))

        assert fig is mock_fig
        assert ax is mock_ax
        mock_subplots.assert_not_called()  # Should not create new figure

    @patch('gridfia.visualization.mapper.load_state_boundary')
    @patch('gridfia.visualization.mapper.plot_boundaries')
    @patch('gridfia.visualization.mapper.clip_boundaries_to_extent')
    def test_create_species_map_with_state_boundary(self, mock_clip, mock_plot, mock_load,
                                                  mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test species map creation with state boundary overlay."""
        # Setup mocks
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mock_boundary_gdf = Mock()
        mock_load.return_value = mock_boundary_gdf
        mock_clip.return_value = mock_boundary_gdf

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_species_map(species=1, state_boundary='California')

        # Verify boundary functions were called
        mock_load.assert_called_once()
        mock_clip.assert_called_once()
        mock_plot.assert_called_once()

    @patch('gridfia.visualization.mapper.get_basemap_zoom_level')
    @patch('gridfia.visualization.mapper.add_basemap')
    def test_create_species_map_with_basemap(self, mock_add_basemap, mock_get_zoom,
                                           mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test species map creation with basemap."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im
        mock_get_zoom.return_value = 10

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_species_map(species=1, basemap='OpenStreetMap', data_alpha=0.7)

        # Verify basemap functions were called
        mock_get_zoom.assert_called_once()
        mock_add_basemap.assert_called_once()

        # Verify alpha was applied to imshow
        imshow_call = mock_ax.imshow.call_args
        args, kwargs = imshow_call
        assert kwargs['alpha'] == 0.7

    @patch('gridfia.visualization.mapper.console')
    def test_create_species_map_boundary_error_handling(self, mock_console, mock_tight_layout,
                                                       mock_colorbar, mock_subplots, complete_zarr_store):
        """Test error handling when boundary loading fails."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        with patch('gridfia.visualization.mapper.load_state_boundary', side_effect=Exception('Boundary error')):
            mapper = ZarrMapper(complete_zarr_store)
            # Should not raise exception, but should print warning
            mapper.create_species_map(species=1, state_boundary='California')

            # Check that warning was printed
            warning_calls = [str(call) for call in mock_console.print.call_args_list]
            assert any('Warning: Could not add state boundary' in call for call in warning_calls)


@patch('matplotlib.pyplot.subplots')
@patch('matplotlib.pyplot.colorbar')
@patch('matplotlib.pyplot.tight_layout')
class TestDiversityMapCreation:
    """Test suite for diversity map creation functionality."""

    def test_create_shannon_diversity_map(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test Shannon diversity map creation."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_diversity_map(diversity_type='shannon')

        assert fig is mock_fig
        assert ax is mock_ax
        mock_ax.imshow.assert_called_once()

        # Verify colorbar has correct label
        mock_colorbar.assert_called_once()
        colorbar_call = mock_colorbar.call_args
        args, kwargs = colorbar_call
        assert kwargs['label'] == 'Shannon Index'

        # Verify title
        mock_ax.set_title.assert_called_once()
        title = mock_ax.set_title.call_args[0][0]
        assert title == 'Shannon Diversity'

    def test_create_simpson_diversity_map(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test Simpson diversity map creation."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_diversity_map(diversity_type='simpson')

        # Verify colorbar has correct label
        mock_colorbar.assert_called_once()
        colorbar_call = mock_colorbar.call_args
        args, kwargs = colorbar_call
        assert kwargs['label'] == 'Simpson Index'

        # Verify title
        mock_ax.set_title.assert_called_once()
        title = mock_ax.set_title.call_args[0][0]
        assert title == 'Simpson Diversity'

    def test_create_diversity_map_invalid_type(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test error handling for invalid diversity type."""
        mapper = ZarrMapper(complete_zarr_store)

        with pytest.raises(ValueError, match="diversity_type must be 'shannon' or 'simpson'"):
            mapper.create_diversity_map(diversity_type='invalid')

    def test_diversity_map_caching(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test that diversity calculations are cached."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)

        # Create map twice with same parameters
        mapper.create_diversity_map(diversity_type='shannon', vmin=0, vmax=2)
        assert len(mapper._diversity_cache) == 1

        # Second call should use cache
        mapper.create_diversity_map(diversity_type='shannon', vmin=0, vmax=2)
        assert len(mapper._diversity_cache) == 1  # Still just one entry

        # Different parameters should create new cache entry
        mapper.create_diversity_map(diversity_type='simpson', vmin=0, vmax=1)
        assert len(mapper._diversity_cache) == 2

    def test_diversity_map_custom_parameters(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test diversity map with custom parameters."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_diversity_map(
            diversity_type='shannon',
            cmap='viridis',
            vmin=0.5,
            vmax=2.5,
            title='Custom Shannon Map',
            colorbar=False
        )

        # Verify custom colormap
        imshow_call = mock_ax.imshow.call_args
        args, kwargs = imshow_call
        assert kwargs['cmap'] == 'viridis'

        # Verify custom title
        mock_ax.set_title.assert_called_once()
        title = mock_ax.set_title.call_args[0][0]
        assert title == 'Custom Shannon Map'

        # Verify no colorbar
        mock_colorbar.assert_not_called()

    @patch('gridfia.visualization.mapper.console')
    def test_diversity_calculation_console_output(self, mock_console, mock_tight_layout,
                                                 mock_colorbar, mock_subplots, complete_zarr_store):
        """Test console output during diversity calculation."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_diversity_map(diversity_type='shannon')

        # Check that calculation message was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any('Calculating shannon diversity index' in call for call in calls)


@patch('matplotlib.pyplot.subplots')
@patch('matplotlib.pyplot.colorbar')
@patch('matplotlib.pyplot.tight_layout')
class TestRichnessMapCreation:
    """Test suite for richness map creation functionality."""

    def test_create_richness_map_default_threshold(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test richness map creation with default threshold."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        fig, ax = mapper.create_richness_map()

        assert fig is mock_fig
        assert ax is mock_ax
        mock_ax.imshow.assert_called_once()

        # Verify colorbar label
        mock_colorbar.assert_called_once()
        colorbar_call = mock_colorbar.call_args
        args, kwargs = colorbar_call
        assert kwargs['label'] == 'Number of Species'

        # Verify default title
        mock_ax.set_title.assert_called_once()
        title = mock_ax.set_title.call_args[0][0]
        assert title == 'Species Richness'

    def test_create_richness_map_custom_threshold(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test richness map with custom biomass threshold."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_richness_map(threshold=5.0)

        mock_ax.imshow.assert_called_once()
        imshow_call = mock_ax.imshow.call_args
        args, kwargs = imshow_call

        # The richness data should be passed to imshow
        richness_data = args[0]
        assert isinstance(richness_data, np.ndarray)
        assert richness_data.dtype == np.uint8

    def test_create_richness_map_custom_parameters(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test richness map with all custom parameters."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_richness_map(
            threshold=2.5,
            cmap='RdYlBu',
            vmin=0,
            vmax=10,
            title='Custom Richness Map',
            colorbar=True
        )

        # Verify custom parameters
        imshow_call = mock_ax.imshow.call_args
        args, kwargs = imshow_call
        assert kwargs['cmap'] == 'RdYlBu'
        assert kwargs['vmin'] == 0
        assert kwargs['vmax'] == 10

        # Verify title
        mock_ax.set_title.assert_called_once()
        title = mock_ax.set_title.call_args[0][0]
        assert title == 'Custom Richness Map'

    @patch('gridfia.visualization.mapper.console')
    def test_richness_calculation_console_output(self, mock_console, mock_tight_layout,
                                                mock_colorbar, mock_subplots, complete_zarr_store):
        """Test console output during richness calculation."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)
        mapper.create_richness_map(threshold=1.5)

        # Check calculation message was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any('Calculating species richness (threshold=1.5)' in call for call in calls)

    def test_richness_map_integer_colorbar_ticks(self, mock_tight_layout, mock_colorbar, mock_subplots, complete_zarr_store):
        """Test that richness map colorbar uses integer ticks for small values."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im
        mock_cbar = Mock()
        mock_colorbar.return_value = mock_cbar

        # Mock richness data to have low maximum
        with patch.object(np, 'max', return_value=5):
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_richness_map()

            # Verify colorbar ticks were set to integers
            mock_cbar.set_ticks.assert_called_once()
            ticks = mock_cbar.set_ticks.call_args[0][0]
            expected_ticks = list(range(0, 6))  # 0 to 5
            assert list(ticks) == expected_ticks


@patch('matplotlib.pyplot.subplots')
@patch('matplotlib.pyplot.tight_layout')
class TestComparisonMapCreation:
    """Test suite for comparison map creation functionality."""

    def test_create_comparison_map_basic(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test basic comparison map creation."""
        # Mock subplot creation
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes(), setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Mock the species map creation
        with patch.object(ZarrMapper, 'create_species_map') as mock_create:
            mapper = ZarrMapper(complete_zarr_store)
            fig = mapper.create_comparison_map(species_list=[1, 2])

            assert fig is mock_fig
            mock_subplots.assert_called_once()

            # Should have called create_species_map for each species
            assert mock_create.call_count == 2

            # Check remove was called for unused subplots (if any)
            mock_tight_layout.assert_called_once()

    def test_create_comparison_map_custom_grid(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test comparison map with custom grid layout."""
        # 3x2 grid for 4 species
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([
            [setup_mock_axes(), setup_mock_axes(), setup_mock_axes()],
            [setup_mock_axes(), setup_mock_axes(), setup_mock_axes()]
        ])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map') as mock_create:
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, 2, 3, 4], ncols=3)

            # Should create 2x3 grid
            mock_subplots.assert_called_once()
            args, kwargs = mock_subplots.call_args
            assert args[0] == 2  # nrows
            assert args[1] == 3  # ncols

            # Should create 4 species maps
            assert mock_create.call_count == 4

            # Should remove 2 empty subplots (6 - 4 = 2)
            assert mock_axes[1, 1].remove.called
            assert mock_axes[1, 2].remove.called

    def test_create_comparison_map_shared_colorbar(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test comparison map with shared colorbar."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes(), setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Mock add_axes for colorbar
        mock_cbar_ax = Mock()
        mock_fig.add_axes.return_value = mock_cbar_ax

        with patch.object(ZarrMapper, 'create_species_map') as mock_create:
            with patch('matplotlib.pyplot.cm.ScalarMappable') as mock_sm_class:
                with patch.object(mock_fig, 'colorbar') as mock_colorbar:
                    mock_sm = Mock()
                    mock_sm_class.return_value = mock_sm

                    mapper = ZarrMapper(complete_zarr_store)
                    mapper.create_comparison_map(species_list=[1, 2], shared_colorbar=True)

                    # Should create ScalarMappable for shared colorbar
                    mock_sm_class.assert_called_once()
                    mock_sm.set_array.assert_called_once()

                    # Should add colorbar axes and create colorbar
                    mock_fig.add_axes.assert_called_once()
                    mock_colorbar.assert_called_once()

    def test_create_comparison_map_no_shared_colorbar(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test comparison map without shared colorbar."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes(), setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map') as mock_create:
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, 2], shared_colorbar=False)

            # Each species map should have its own colorbar
            for call_args in mock_create.call_args_list:
                args, kwargs = call_args
                assert kwargs['colorbar'] is True

    def test_create_comparison_map_mixed_species_identifiers(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test comparison map with mixed indices and codes."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes(), setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map') as mock_create:
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, '0122'])  # Mix of index and code

            assert mock_create.call_count == 2

            # Verify species were passed correctly
            calls = mock_create.call_args_list
            assert calls[0][1]['species'] == 1
            assert calls[1][1]['species'] == '0122'

    def test_create_comparison_map_auto_figsize(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test automatic figure size calculation."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([
            [setup_mock_axes(), setup_mock_axes()],
            [setup_mock_axes(), setup_mock_axes()]
        ])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map'):
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, 2, 3, 4], ncols=2)

            # Should calculate figsize: 2 cols * 6 width, 2 rows * 5 height
            args, kwargs = mock_subplots.call_args
            assert kwargs['figsize'] == (12, 10)

    def test_create_comparison_map_custom_figsize(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test comparison map with custom figure size."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes(), setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map'):
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, 2], figsize=(16, 8))

            args, kwargs = mock_subplots.call_args
            assert kwargs['figsize'] == (16, 8)

    def test_create_comparison_map_single_column(self, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test comparison map with single column layout."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes()], [setup_mock_axes()], [setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map'):
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, 2, 3], ncols=1)

            args, kwargs = mock_subplots.call_args
            assert args[0] == 3  # nrows
            assert args[1] == 1  # ncols

    @patch('gridfia.visualization.mapper.console')
    def test_comparison_map_global_minmax_calculation(self, mock_console, mock_tight_layout, mock_subplots, complete_zarr_store):
        """Test global min/max calculation for shared colorbar."""
        mock_fig = Mock(spec=Figure)
        mock_axes = np.array([[setup_mock_axes(), setup_mock_axes()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        with patch.object(ZarrMapper, 'create_species_map'):
            mapper = ZarrMapper(complete_zarr_store)
            mapper.create_comparison_map(species_list=[1, 2], shared_colorbar=True)

            # Check that global min/max calculation message was printed
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any('Calculating global min/max for shared colorbar' in call for call in calls)


class TestMapExportFunctionality:
    """Test suite for map export functionality."""

    @patch('gridfia.visualization.mapper.console')
    def test_export_map_basic(self, mock_console, temp_dir, complete_zarr_store):
        """Test basic map export functionality."""
        mapper = ZarrMapper(complete_zarr_store)

        mock_fig = Mock(spec=Figure)
        output_path = temp_dir / "test_map.png"

        mapper.export_map(mock_fig, output_path)

        # Verify savefig was called on the figure with correct parameters
        mock_fig.savefig.assert_called_once()
        args, kwargs = mock_fig.savefig.call_args
        assert args[0] == output_path
        assert kwargs['dpi'] == 300
        assert kwargs['bbox_inches'] == 'tight'
        assert kwargs['transparent'] is False
        assert kwargs['facecolor'] == 'white'

        # Verify console messages
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any('Exporting map to' in call for call in calls)
        assert any('Map saved to' in call for call in calls)

    def test_export_map_custom_parameters(self, temp_dir, complete_zarr_store):
        """Test map export with custom parameters."""
        mapper = ZarrMapper(complete_zarr_store)

        mock_fig = Mock(spec=Figure)
        output_path = temp_dir / "custom_map.jpg"

        mapper.export_map(
            mock_fig,
            output_path,
            dpi=150,
            bbox_inches='standard',
            transparent=True
        )

        # Verify custom parameters
        mock_fig.savefig.assert_called_once()
        args, kwargs = mock_fig.savefig.call_args
        assert kwargs['dpi'] == 150
        assert kwargs['bbox_inches'] == 'standard'
        assert kwargs['transparent'] is True

    def test_export_map_string_path(self, temp_dir, complete_zarr_store):
        """Test map export with string path instead of Path object."""
        mapper = ZarrMapper(complete_zarr_store)

        mock_fig = Mock(spec=Figure)
        output_path = str(temp_dir / "string_path.png")

        mapper.export_map(mock_fig, output_path)

        mock_fig.savefig.assert_called_once()
        args, kwargs = mock_fig.savefig.call_args
        # Should convert string to Path
        assert args[0] == Path(output_path)

    def test_export_map_creates_directories(self, temp_dir, complete_zarr_store):
        """Test that export creates necessary directories."""
        mapper = ZarrMapper(complete_zarr_store)

        mock_fig = Mock(spec=Figure)
        # Create nested directory path that doesn't exist
        output_path = temp_dir / "nested" / "directories" / "map.png"

        mapper.export_map(mock_fig, output_path)

        # Directory should be created
        assert output_path.parent.exists()
        mock_fig.savefig.assert_called_once()


class TestErrorHandlingAndEdgeCases:
    """Test suite for error conditions and edge cases."""

    def test_empty_zarr_data_handling(self, empty_zarr_store):
        """Test handling of zarr stores with all zero data."""
        mapper = ZarrMapper(empty_zarr_store)

        # Should initialize successfully
        assert mapper.num_species == 3

        # Should handle empty data gracefully in normalization
        data = mapper.biomass[0, :, :]
        normalized = mapper._normalize_data(data)
        assert normalized.shape == data.shape
        assert np.all(normalized == 0)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.tight_layout')
    def test_single_species_zarr(self, mock_tight_layout, mock_colorbar, mock_subplots, minimal_zarr_store):
        """Test handling of zarr stores with minimal species data."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(minimal_zarr_store)

        # Should handle species maps
        fig, ax = mapper.create_species_map(species=1)
        assert fig is mock_fig

        # Should handle diversity maps
        fig, ax = mapper.create_diversity_map()
        assert fig is mock_fig

    def test_data_with_all_nans(self, complete_zarr_store):
        """Test normalization with all NaN data."""
        mapper = ZarrMapper(complete_zarr_store)

        data = np.full((10, 10), np.nan, dtype=np.float32)
        normalized = mapper._normalize_data(data)

        # Should handle gracefully
        assert normalized.shape == data.shape
        # NaN values are now preserved in the output (making failed calculations visible)
        assert np.all(np.isnan(normalized))

    def test_data_with_single_value(self, complete_zarr_store):
        """Test normalization when all valid data has same value."""
        mapper = ZarrMapper(complete_zarr_store)

        data = np.full((10, 10), 5.0, dtype=np.float32)
        normalized = mapper._normalize_data(data)

        # When vmin == vmax, should return zeros
        assert normalized.shape == data.shape
        assert np.all(normalized == 0)

    def test_extreme_percentile_values(self, complete_zarr_store):
        """Test normalization with extreme percentile settings."""
        mapper = ZarrMapper(complete_zarr_store)

        data = np.random.rand(100, 100).astype(np.float32)

        # Very tight percentiles
        normalized = mapper._normalize_data(data, percentile=(49, 51))
        assert normalized.shape == data.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.tight_layout')
    def test_diversity_map_with_zero_biomass(self, mock_tight_layout, mock_colorbar, mock_subplots, empty_zarr_store):
        """Test diversity calculation with zero biomass everywhere."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(empty_zarr_store)

        fig, ax = mapper.create_diversity_map()

        # Should complete without error
        assert fig is mock_fig
        mock_ax.imshow.assert_called_once()

        # Diversity data should be all zeros (zero biomass = zero diversity)
        imshow_call = mock_ax.imshow.call_args
        diversity_data = imshow_call[0][0]  # First positional argument
        # After normalization, zero diversity should remain zero
        assert np.all(diversity_data == 0)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.tight_layout')
    def test_richness_map_with_zero_biomass(self, mock_tight_layout, mock_colorbar, mock_subplots, empty_zarr_store):
        """Test richness calculation with zero biomass everywhere."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(empty_zarr_store)

        fig, ax = mapper.create_richness_map()

        # Should complete without error
        assert fig is mock_fig
        mock_ax.imshow.assert_called_once()

        # Richness should be all zeros
        imshow_call = mock_ax.imshow.call_args
        richness_data = imshow_call[0][0]
        assert np.all(richness_data == 0)
        assert richness_data.dtype == np.uint8

    def test_comparison_map_empty_species_list(self, complete_zarr_store):
        """Test comparison map with empty species list."""
        mapper = ZarrMapper(complete_zarr_store)

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock(spec=Figure)
            # No axes needed for empty list
            mock_subplots.return_value = (mock_fig, np.array([]))

            fig = mapper.create_comparison_map(species_list=[])

            # Should handle gracefully
            assert fig is mock_fig


class TestMatplotlibIntegration:
    """Test suite for matplotlib integration and visual components."""

    @patch('matplotlib.pyplot.subplots')
    def test_axes_configuration(self, mock_subplots, complete_zarr_store):
        """Test that matplotlib axes are configured correctly."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)

        with patch('matplotlib.pyplot.colorbar'):
            with patch('matplotlib.pyplot.tight_layout'):
                mapper.create_species_map(species=1)

                # Verify axes configuration calls
                mock_ax.set_xlabel.assert_called_once()
                mock_ax.set_ylabel.assert_called_once()
                mock_ax.set_title.assert_called_once()
                mock_ax.ticklabel_format.assert_called_once()
                mock_ax.grid.assert_called_once()

                # Check specific parameter values
                xlabel_call = mock_ax.set_xlabel.call_args[0][0]
                ylabel_call = mock_ax.set_ylabel.call_args[0][0]
                assert 'Easting' in xlabel_call
                assert 'Northing' in ylabel_call

                # Check grid settings
                grid_call = mock_ax.grid.call_args
                assert grid_call[0][0] is True  # grid enabled
                assert grid_call[1]['alpha'] == 0.3

    @patch('matplotlib.pyplot.subplots')
    def test_colorbar_configuration(self, mock_subplots, complete_zarr_store):
        """Test colorbar creation and configuration."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)

        with patch('matplotlib.pyplot.colorbar') as mock_colorbar:
            mock_cbar = Mock()
            mock_colorbar.return_value = mock_cbar

            with patch('matplotlib.pyplot.tight_layout'):
                mapper.create_species_map(species=1, vmin=10, vmax=50)

                # Verify colorbar creation
                mock_colorbar.assert_called_once()
                args, kwargs = mock_colorbar.call_args
                assert args[0] is mock_im  # image mappable
                assert kwargs['ax'] is mock_ax
                assert 'label' in kwargs
                assert kwargs['shrink'] == 0.8

                # Verify clim setting
                mock_cbar.mappable.set_clim.assert_called_once_with(10, 50)

    @patch('matplotlib.pyplot.subplots')
    def test_bounds_annotation(self, mock_subplots, complete_zarr_store):
        """Test bounds annotation display."""
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_ax.transAxes = Mock()  # Add the required attribute
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        mapper = ZarrMapper(complete_zarr_store)

        with patch('matplotlib.pyplot.colorbar'):
            with patch('matplotlib.pyplot.tight_layout'):
                mapper.create_species_map(species=1, show_bounds=True)

                # Verify text annotation was added
                mock_ax.text.assert_called_once()
                text_call = mock_ax.text.call_args
                args, kwargs = text_call

                # Check position and content
                assert args[0] == 0.02  # x position
                assert args[1] == 0.98  # y position
                bounds_text = args[2]
                assert 'Bounds:' in bounds_text
                assert kwargs['transform'] is mock_ax.transAxes
                assert 'bbox' in kwargs

    def test_colormap_handling(self, complete_zarr_store):
        """Test various colormap options."""
        mapper = ZarrMapper(complete_zarr_store)

        # Test that different colormaps are accepted
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'RdYlBu', 'Spectral_r']

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = setup_mock_axes()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_im = Mock()
            mock_ax.imshow.return_value = mock_im

            with patch('matplotlib.pyplot.colorbar'):
                with patch('matplotlib.pyplot.tight_layout'):
                    for cmap in colormaps:
                        mapper.create_species_map(species=1, cmap=cmap)

                        # Verify colormap was passed to imshow
                        imshow_call = mock_ax.imshow.call_args
                        args, kwargs = imshow_call
                        assert kwargs['cmap'] == cmap

    @patch('matplotlib.pyplot.subplots')
    def test_figure_size_handling(self, mock_subplots, complete_zarr_store):
        """Test figure size configuration."""
        mapper = ZarrMapper(complete_zarr_store)

        # Default figure size for species map
        mock_fig = Mock(spec=Figure)
        mock_ax = setup_mock_axes()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im

        with patch('matplotlib.pyplot.colorbar'):
            with patch('matplotlib.pyplot.tight_layout'):
                mapper.create_species_map(species=1)

                # Check default figsize
                args, kwargs = mock_subplots.call_args
                assert kwargs['figsize'] == (12, 10)


# Mark the TodoWrite task as completed
@pytest.fixture(autouse=True)
def complete_fixture_creation():
    """Auto-run fixture to mark todo completion."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])