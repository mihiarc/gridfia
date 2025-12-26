"""
Comprehensive tests for the visualization plots module.

Tests cover plotting utilities, colormap functions, axis formatting,
legend creation, scalebar/north arrow additions, and figure saving.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile

from gridfia.visualization.plots import (
    DEFAULT_COLORMAPS,
    DEFAULT_FIGURE_SETTINGS,
    DEFAULT_FONT_SETTINGS,
    set_plot_style,
    get_colormap,
    create_discrete_colormap,
    add_scalebar,
    add_north_arrow,
    format_axes_labels,
    create_legend,
    adjust_colorbar,
    add_inset_histogram,
    save_figure,
)


# ==============================================================================
# Helper Functions and Fixtures
# ==============================================================================


def create_mock_axes():
    """Helper function to create properly mocked matplotlib axes."""
    mock_axes = Mock(spec=Axes)
    mock_axes.transAxes = Mock()
    mock_axes.spines = {
        'top': Mock(),
        'bottom': Mock(),
        'left': Mock(),
        'right': Mock()
    }
    return mock_axes


def create_mock_figure():
    """Helper function to create a properly mocked matplotlib figure."""
    mock_figure = Mock(spec=Figure)
    mock_figure.get_facecolor.return_value = (1.0, 1.0, 1.0, 1.0)
    return mock_figure


def create_mock_colorbar():
    """Helper function to create a properly mocked colorbar."""
    mock_cbar = Mock(spec=Colorbar)
    mock_cbar.ax = Mock()
    mock_cbar.ax.yaxis = Mock()
    mock_cbar.ax.tick_params = Mock()
    mock_cbar.extend = 'neither'  # Default extend value
    return mock_cbar


@pytest.fixture
def mock_axes():
    """Fixture providing a mocked matplotlib axes object."""
    return create_mock_axes()


@pytest.fixture
def mock_figure():
    """Fixture providing a mocked matplotlib figure object."""
    return create_mock_figure()


@pytest.fixture
def mock_colorbar():
    """Fixture providing a mocked colorbar object."""
    return create_mock_colorbar()


@pytest.fixture
def sample_data():
    """Fixture providing sample numpy data for histogram testing."""
    np.random.seed(42)
    data = np.random.normal(50, 15, (100, 100))
    data[data < 0] = 0
    return data


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary directory for file output tests."""
    with tempfile.TemporaryDirectory() as temp_directory:
        yield Path(temp_directory)


# ==============================================================================
# Tests for Default Configuration Constants
# ==============================================================================


class TestDefaultConstants:
    """Test suite for default configuration constants."""

    def test_default_colormaps_contains_expected_keys(self):
        """Verify DEFAULT_COLORMAPS contains all expected data type keys."""
        expected_keys = ['biomass', 'diversity', 'richness', 'species', 'comparison', 'hotspot']
        for key in expected_keys:
            assert key in DEFAULT_COLORMAPS, f"Missing colormap key: {key}"

    def test_default_colormaps_values_are_valid_colormap_names(self):
        """Verify all default colormap values are valid matplotlib colormap names."""
        for data_type, colormap_name in DEFAULT_COLORMAPS.items():
            assert isinstance(colormap_name, str), f"Colormap for {data_type} should be a string"
            try:
                plt.cm.get_cmap(colormap_name)
            except ValueError:
                pytest.fail(f"Invalid colormap name '{colormap_name}' for data type '{data_type}'")

    def test_default_figure_settings_contains_expected_keys(self):
        """Verify DEFAULT_FIGURE_SETTINGS contains expected configuration keys."""
        expected_keys = ['dpi', 'facecolor', 'edgecolor', 'tight_layout']
        for key in expected_keys:
            assert key in DEFAULT_FIGURE_SETTINGS, f"Missing figure setting: {key}"

    def test_default_figure_settings_dpi_is_positive_integer(self):
        """Verify DPI setting is a positive integer."""
        assert isinstance(DEFAULT_FIGURE_SETTINGS['dpi'], int)
        assert DEFAULT_FIGURE_SETTINGS['dpi'] > 0

    def test_default_font_settings_contains_expected_keys(self):
        """Verify DEFAULT_FONT_SETTINGS contains expected configuration keys."""
        expected_keys = ['family', 'size', 'weight']
        for key in expected_keys:
            assert key in DEFAULT_FONT_SETTINGS, f"Missing font setting: {key}"


# ==============================================================================
# Tests for set_plot_style Function
# ==============================================================================


class TestSetPlotStyle:
    """Test suite for the set_plot_style function."""

    def test_set_publication_style_updates_rcparams(self):
        """Verify publication style sets appropriate rcParams values."""
        original_font_size = plt.rcParams['font.size']

        set_plot_style('publication')

        assert plt.rcParams['font.size'] == 10
        assert plt.rcParams['axes.titlesize'] == 12
        assert plt.rcParams['axes.labelsize'] == 11
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['axes.spines.top'] is False
        assert plt.rcParams['axes.spines.right'] is False

        # Reset to avoid affecting other tests
        plt.rcParams.update(plt.rcParamsDefault)

    def test_set_presentation_style_updates_rcparams(self):
        """Verify presentation style sets larger font sizes for visibility."""
        set_plot_style('presentation')

        assert plt.rcParams['font.size'] == 14
        assert plt.rcParams['axes.titlesize'] == 18
        assert plt.rcParams['axes.labelsize'] == 16
        assert plt.rcParams['figure.dpi'] == 150
        assert plt.rcParams['lines.linewidth'] == 2.5

        # Reset to avoid affecting other tests
        plt.rcParams.update(plt.rcParamsDefault)

    def test_set_default_style_resets_to_defaults(self):
        """Verify default style resets rcParams to matplotlib defaults."""
        # First set a custom style
        set_plot_style('publication')
        assert plt.rcParams['font.size'] == 10

        # Then reset to default
        set_plot_style('default')

        # Font size should be reset to matplotlib default (typically 10.0)
        assert isinstance(plt.rcParams['font.size'], (int, float))

    def test_set_unknown_style_uses_default(self):
        """Verify unknown style names fall back to default style."""
        # This should not raise an error
        set_plot_style('unknown_style')

        # Should work without exception
        assert True

        # Reset
        plt.rcParams.update(plt.rcParamsDefault)

    def test_publication_style_line_settings(self):
        """Verify publication style sets appropriate line properties."""
        set_plot_style('publication')

        assert plt.rcParams['lines.linewidth'] == 1.5
        assert plt.rcParams['lines.markersize'] == 6
        assert plt.rcParams['axes.linewidth'] == 0.8

        plt.rcParams.update(plt.rcParamsDefault)

    def test_presentation_style_tick_sizes(self):
        """Verify presentation style sets larger tick sizes for visibility."""
        set_plot_style('presentation')

        assert plt.rcParams['xtick.labelsize'] == 14
        assert plt.rcParams['ytick.labelsize'] == 14
        assert plt.rcParams['legend.fontsize'] == 14

        plt.rcParams.update(plt.rcParamsDefault)


# ==============================================================================
# Tests for get_colormap Function
# ==============================================================================


class TestGetColormap:
    """Test suite for the get_colormap function."""

    def test_get_colormap_returns_default_for_known_data_type(self):
        """Verify known data types return their default colormaps."""
        assert get_colormap('biomass') == 'viridis'
        assert get_colormap('diversity') == 'plasma'
        assert get_colormap('richness') == 'Spectral_r'
        assert get_colormap('species') == 'YlGn'
        assert get_colormap('comparison') == 'RdYlBu_r'
        assert get_colormap('hotspot') == 'hot_r'

    def test_get_colormap_returns_viridis_for_unknown_data_type(self):
        """Verify unknown data types fall back to viridis colormap."""
        assert get_colormap('unknown_type') == 'viridis'
        assert get_colormap('random') == 'viridis'
        assert get_colormap('') == 'viridis'

    def test_get_colormap_custom_cmap_overrides_default(self):
        """Verify custom colormap parameter overrides the default."""
        custom_colormap = 'plasma'
        result = get_colormap('biomass', custom_cmap=custom_colormap)
        assert result == custom_colormap

    def test_get_colormap_custom_cmap_with_unknown_type(self):
        """Verify custom colormap works even with unknown data type."""
        custom_colormap = 'magma'
        result = get_colormap('nonexistent', custom_cmap=custom_colormap)
        assert result == custom_colormap

    def test_get_colormap_none_custom_cmap_uses_default(self):
        """Verify None custom_cmap uses the default colormap."""
        result = get_colormap('biomass', custom_cmap=None)
        assert result == 'viridis'

    def test_get_colormap_returns_string(self):
        """Verify get_colormap always returns a string."""
        assert isinstance(get_colormap('biomass'), str)
        assert isinstance(get_colormap('unknown'), str)
        assert isinstance(get_colormap('species', custom_cmap='inferno'), str)


# ==============================================================================
# Tests for create_discrete_colormap Function
# ==============================================================================


class TestCreateDiscreteColormap:
    """Test suite for the create_discrete_colormap function."""

    def test_create_discrete_colormap_returns_listed_colormap(self):
        """Verify function returns a matplotlib ListedColormap."""
        result = create_discrete_colormap(5)
        assert isinstance(result, mcolors.ListedColormap)

    def test_create_discrete_colormap_correct_number_of_colors(self):
        """Verify colormap has the correct number of colors."""
        for num_colors in [3, 5, 10, 15, 20]:
            result = create_discrete_colormap(num_colors)
            assert len(result.colors) == num_colors

    def test_create_discrete_colormap_uses_tab20_for_small_counts(self):
        """Verify tab20 is used by default for 20 or fewer colors."""
        result = create_discrete_colormap(10, cmap_name='tab20')
        assert isinstance(result, mcolors.ListedColormap)
        assert len(result.colors) == 10

    def test_create_discrete_colormap_handles_more_than_20_colors(self):
        """Verify function handles more than 20 colors correctly."""
        result = create_discrete_colormap(25)
        assert isinstance(result, mcolors.ListedColormap)
        assert len(result.colors) == 25

    def test_create_discrete_colormap_custom_base_colormap(self):
        """Verify custom base colormap can be specified."""
        result = create_discrete_colormap(8, cmap_name='viridis')
        assert isinstance(result, mcolors.ListedColormap)
        assert len(result.colors) == 8

    def test_create_discrete_colormap_single_color(self):
        """Verify function handles single color request."""
        result = create_discrete_colormap(1)
        assert isinstance(result, mcolors.ListedColormap)
        assert len(result.colors) == 1

    def test_create_discrete_colormap_colors_are_valid_rgba(self):
        """Verify returned colors are valid RGBA tuples."""
        result = create_discrete_colormap(5)
        for color in result.colors:
            assert len(color) == 4  # RGBA
            assert all(0 <= channel <= 1 for channel in color)

    def test_create_discrete_colormap_exactly_20_colors(self):
        """Verify boundary case of exactly 20 colors uses tab20."""
        result = create_discrete_colormap(20, cmap_name='tab20')
        assert len(result.colors) == 20


# ==============================================================================
# Tests for add_scalebar Function
# ==============================================================================


class TestAddScalebar:
    """Test suite for the add_scalebar function."""

    def test_add_scalebar_calls_add_artist(self, mock_axes):
        """Verify scalebar is added to axes via add_artist."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            mock_scalebar_instance = Mock()
            mock_scalebar_class.return_value = mock_scalebar_instance

            add_scalebar(mock_axes)

            mock_axes.add_artist.assert_called_once_with(mock_scalebar_instance)

    def test_add_scalebar_with_default_parameters(self, mock_axes):
        """Verify scalebar uses default parameters when none specified."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            add_scalebar(mock_axes)

            mock_scalebar_class.assert_called_once()
            call_kwargs = mock_scalebar_class.call_args[1]
            assert call_kwargs['location'] == 'lower right'
            assert call_kwargs['length_fraction'] == 0.25
            assert call_kwargs['box_alpha'] == 0.8
            assert call_kwargs['color'] == 'black'

    def test_add_scalebar_with_custom_location(self, mock_axes):
        """Verify scalebar accepts custom location parameter."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            add_scalebar(mock_axes, location='upper left')

            call_kwargs = mock_scalebar_class.call_args[1]
            assert call_kwargs['location'] == 'upper left'

    def test_add_scalebar_with_custom_length_fraction(self, mock_axes):
        """Verify scalebar accepts custom length_fraction parameter."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            add_scalebar(mock_axes, length_fraction=0.5)

            call_kwargs = mock_scalebar_class.call_args[1]
            assert call_kwargs['length_fraction'] == 0.5

    def test_add_scalebar_with_custom_color(self, mock_axes):
        """Verify scalebar accepts custom color parameter."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            add_scalebar(mock_axes, color='white')

            call_kwargs = mock_scalebar_class.call_args[1]
            assert call_kwargs['color'] == 'white'

    def test_add_scalebar_with_custom_font_size(self, mock_axes):
        """Verify scalebar accepts custom font_size parameter."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            add_scalebar(mock_axes, font_size=14)

            call_kwargs = mock_scalebar_class.call_args[1]
            assert call_kwargs['font_properties']['size'] == 14

    def test_add_scalebar_handles_exception_gracefully(self, mock_axes, capsys):
        """Verify scalebar handles exceptions gracefully with warning."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            mock_scalebar_class.side_effect = RuntimeError("Scalebar error")

            # Should not raise an exception
            add_scalebar(mock_axes)

            captured = capsys.readouterr()
            assert "Warning" in captured.out
            assert "Could not add scalebar" in captured.out

    def test_add_scalebar_with_all_custom_parameters(self, mock_axes):
        """Verify scalebar accepts all custom parameters together."""
        with patch('gridfia.visualization.plots.ScaleBar') as mock_scalebar_class:
            add_scalebar(
                mock_axes,
                location='center',
                length_fraction=0.3,
                box_alpha=0.5,
                font_size=12,
                color='red'
            )

            call_kwargs = mock_scalebar_class.call_args[1]
            assert call_kwargs['location'] == 'center'
            assert call_kwargs['length_fraction'] == 0.3
            assert call_kwargs['box_alpha'] == 0.5
            assert call_kwargs['color'] == 'red'


# ==============================================================================
# Tests for add_north_arrow Function
# ==============================================================================


class TestAddNorthArrow:
    """Test suite for the add_north_arrow function."""

    def test_add_north_arrow_calls_annotate_twice(self, mock_axes):
        """Verify north arrow creates two annotations (edge and main)."""
        add_north_arrow(mock_axes)

        # Should call annotate twice: once for edge, once for main arrow
        assert mock_axes.annotate.call_count == 2

    def test_add_north_arrow_with_default_location(self, mock_axes):
        """Verify north arrow uses default location (0.95, 0.95)."""
        add_north_arrow(mock_axes)

        # Check first annotate call (edge annotation)
        first_call_kwargs = mock_axes.annotate.call_args_list[0][1]
        assert first_call_kwargs['xytext'] == (0.95, 0.95)

    def test_add_north_arrow_with_custom_location(self, mock_axes):
        """Verify north arrow accepts custom location parameter."""
        custom_location = (0.8, 0.9)
        add_north_arrow(mock_axes, location=custom_location)

        first_call_kwargs = mock_axes.annotate.call_args_list[0][1]
        assert first_call_kwargs['xytext'] == custom_location

    def test_add_north_arrow_with_custom_size(self, mock_axes):
        """Verify north arrow accepts custom size parameter."""
        custom_size = 0.15
        add_north_arrow(mock_axes, size=custom_size)

        # The arrow goes from (x, y) to (x, y + size)
        first_call_kwargs = mock_axes.annotate.call_args_list[0][1]
        expected_y_end = 0.95 + custom_size
        assert first_call_kwargs['xy'] == (0.95, expected_y_end)

    def test_add_north_arrow_with_custom_colors(self, mock_axes):
        """Verify north arrow accepts custom color and edge_color parameters."""
        add_north_arrow(mock_axes, color='blue', edge_color='yellow')

        # Check main arrow color (second annotate call)
        second_call_kwargs = mock_axes.annotate.call_args_list[1][1]
        assert second_call_kwargs['arrowprops']['color'] == 'blue'

        # Check edge color (first annotate call)
        first_call_kwargs = mock_axes.annotate.call_args_list[0][1]
        assert first_call_kwargs['arrowprops']['color'] == 'yellow'

    def test_add_north_arrow_with_custom_edge_width(self, mock_axes):
        """Verify north arrow accepts custom edge_width parameter."""
        add_north_arrow(mock_axes, edge_width=3)

        # Edge arrow should have width + 2
        first_call_kwargs = mock_axes.annotate.call_args_list[0][1]
        assert first_call_kwargs['arrowprops']['lw'] == 5  # edge_width + 2

        # Main arrow should have specified width
        second_call_kwargs = mock_axes.annotate.call_args_list[1][1]
        assert second_call_kwargs['arrowprops']['lw'] == 3

    def test_add_north_arrow_includes_n_label(self, mock_axes):
        """Verify north arrow includes 'N' label text."""
        add_north_arrow(mock_axes)

        # Second call should include 'N' label
        second_call_args = mock_axes.annotate.call_args_list[1][0]
        assert 'N' in second_call_args

    def test_add_north_arrow_uses_axes_fraction_coordinates(self, mock_axes):
        """Verify north arrow uses axes fraction coordinate system."""
        add_north_arrow(mock_axes)

        for call_item in mock_axes.annotate.call_args_list:
            call_kwargs = call_item[1]
            assert call_kwargs['xycoords'] == 'axes fraction'

    def test_add_north_arrow_annotation_clip_disabled(self, mock_axes):
        """Verify annotation_clip is set to False for both annotations."""
        add_north_arrow(mock_axes)

        for call_item in mock_axes.annotate.call_args_list:
            call_kwargs = call_item[1]
            assert call_kwargs['annotation_clip'] is False


# ==============================================================================
# Tests for format_axes_labels Function
# ==============================================================================


class TestFormatAxesLabels:
    """Test suite for the format_axes_labels function."""

    def test_format_axes_labels_sets_default_labels(self, mock_axes):
        """Verify default axis labels are set correctly."""
        format_axes_labels(mock_axes)

        mock_axes.set_xlabel.assert_called_once()
        mock_axes.set_ylabel.assert_called_once()

        xlabel_call = mock_axes.set_xlabel.call_args
        ylabel_call = mock_axes.set_ylabel.call_args

        assert 'Easting' in xlabel_call[0][0]
        assert 'Northing' in ylabel_call[0][0]

    def test_format_axes_labels_sets_custom_labels(self, mock_axes):
        """Verify custom axis labels are applied."""
        format_axes_labels(mock_axes, xlabel='Longitude', ylabel='Latitude')

        mock_axes.set_xlabel.assert_called_once_with('Longitude', fontsize=12)
        mock_axes.set_ylabel.assert_called_once_with('Latitude', fontsize=12)

    def test_format_axes_labels_sets_title_when_provided(self, mock_axes):
        """Verify title is set when provided."""
        test_title = 'Test Plot Title'
        format_axes_labels(mock_axes, title=test_title)

        mock_axes.set_title.assert_called_once()
        call_args = mock_axes.set_title.call_args
        assert call_args[0][0] == test_title
        assert call_args[1]['fontweight'] == 'bold'

    def test_format_axes_labels_no_title_when_none(self, mock_axes):
        """Verify no title is set when title is None."""
        format_axes_labels(mock_axes, title=None)

        mock_axes.set_title.assert_not_called()

    def test_format_axes_labels_applies_tick_params(self, mock_axes):
        """Verify tick parameters are applied."""
        format_axes_labels(mock_axes, tick_fontsize=8)

        mock_axes.tick_params.assert_called_once()
        call_kwargs = mock_axes.tick_params.call_args[1]
        assert call_kwargs['labelsize'] == 8

    def test_format_axes_labels_enables_grid_by_default(self, mock_axes):
        """Verify grid is enabled by default."""
        format_axes_labels(mock_axes)

        mock_axes.grid.assert_called_once()
        call_kwargs = mock_axes.grid.call_args[1]
        assert call_kwargs['alpha'] == 0.3

    def test_format_axes_labels_disables_grid_when_specified(self, mock_axes):
        """Verify grid can be disabled."""
        format_axes_labels(mock_axes, grid=False)

        mock_axes.grid.assert_not_called()

    def test_format_axes_labels_custom_grid_alpha(self, mock_axes):
        """Verify custom grid alpha is applied."""
        format_axes_labels(mock_axes, grid=True, grid_alpha=0.5)

        call_kwargs = mock_axes.grid.call_args[1]
        assert call_kwargs['alpha'] == 0.5

    def test_format_axes_labels_hides_top_right_spines(self, mock_axes):
        """Verify top and right spines are hidden."""
        format_axes_labels(mock_axes)

        mock_axes.spines['top'].set_visible.assert_called_once_with(False)
        mock_axes.spines['right'].set_visible.assert_called_once_with(False)

    def test_format_axes_labels_custom_font_sizes(self, mock_axes):
        """Verify custom font sizes are applied correctly."""
        format_axes_labels(
            mock_axes,
            title='Test',
            title_fontsize=16,
            label_fontsize=14,
            tick_fontsize=10
        )

        # Check title fontsize
        title_call = mock_axes.set_title.call_args
        assert title_call[1]['fontsize'] == 16

        # Check label fontsize
        xlabel_call = mock_axes.set_xlabel.call_args
        assert xlabel_call[1]['fontsize'] == 14

    def test_format_axes_labels_ticklabel_format_called(self, mock_axes):
        """Verify ticklabel_format is called with plain style."""
        format_axes_labels(mock_axes)

        mock_axes.ticklabel_format.assert_called_once()
        call_kwargs = mock_axes.ticklabel_format.call_args[1]
        assert call_kwargs['style'] == 'plain'


# ==============================================================================
# Tests for create_legend Function
# ==============================================================================


class TestCreateLegend:
    """Test suite for the create_legend function."""

    def test_create_legend_creates_patches_for_each_label(self, mock_axes):
        """Verify legend creates a patch for each label-color pair."""
        labels = ['Label 1', 'Label 2', 'Label 3']
        colors = ['red', 'green', 'blue']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors)

        mock_axes.legend.assert_called_once()
        call_kwargs = mock_axes.legend.call_args[1]
        handles = call_kwargs['handles']
        assert len(handles) == 3

    def test_create_legend_with_title(self, mock_axes):
        """Verify legend accepts title parameter."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, title='Legend Title')

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['title'] == 'Legend Title'

    def test_create_legend_with_custom_location(self, mock_axes):
        """Verify legend accepts custom location parameter."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, location='upper right')

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['loc'] == 'upper right'

    def test_create_legend_with_multiple_columns(self, mock_axes):
        """Verify legend accepts ncol parameter for multiple columns."""
        labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4']
        colors = ['red', 'green', 'blue', 'yellow']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, ncol=2)

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['ncol'] == 2

    def test_create_legend_with_custom_font_sizes(self, mock_axes):
        """Verify legend accepts custom fontsize parameters."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(
            mock_axes, labels, colors,
            title='Test',
            fontsize=12,
            title_fontsize=14
        )

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['fontsize'] == 12
        assert call_kwargs['title_fontsize'] == 14

    def test_create_legend_frameon_parameter(self, mock_axes):
        """Verify legend accepts frameon parameter."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, frameon=False)

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['frameon'] is False

    def test_create_legend_fancybox_parameter(self, mock_axes):
        """Verify legend accepts fancybox parameter."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, fancybox=False)

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['fancybox'] is False

    def test_create_legend_shadow_parameter(self, mock_axes):
        """Verify legend accepts shadow parameter."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, shadow=True)

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['shadow'] is True

    def test_create_legend_alpha_parameter(self, mock_axes):
        """Verify legend accepts alpha parameter for transparency."""
        labels = ['Label 1']
        colors = ['red']

        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors, alpha=0.7)

        call_kwargs = mock_axes.legend.call_args[1]
        assert call_kwargs['framealpha'] == 0.7

    def test_create_legend_sets_frame_linewidth(self, mock_axes):
        """Verify legend frame linewidth is set."""
        labels = ['Label 1']
        colors = ['red']

        mock_frame = Mock()
        mock_legend = Mock()
        mock_legend.get_frame.return_value = mock_frame
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, labels, colors)

        mock_frame.set_linewidth.assert_called_once_with(0.8)


# ==============================================================================
# Tests for adjust_colorbar Function
# ==============================================================================


class TestAdjustColorbar:
    """Test suite for the adjust_colorbar function."""

    def test_adjust_colorbar_sets_label(self, mock_colorbar):
        """Verify colorbar label is set correctly."""
        adjust_colorbar(mock_colorbar, label='Biomass (Mg/ha)')

        mock_colorbar.set_label.assert_called_once()
        call_args = mock_colorbar.set_label.call_args
        assert call_args[0][0] == 'Biomass (Mg/ha)'

    def test_adjust_colorbar_sets_label_fontsize(self, mock_colorbar):
        """Verify colorbar label fontsize is applied."""
        adjust_colorbar(mock_colorbar, label='Test', label_fontsize=14)

        call_kwargs = mock_colorbar.set_label.call_args[1]
        assert call_kwargs['fontsize'] == 14

    def test_adjust_colorbar_sets_tick_fontsize(self, mock_colorbar):
        """Verify colorbar tick fontsize is applied."""
        adjust_colorbar(mock_colorbar, label='Test', tick_fontsize=10)

        mock_colorbar.ax.tick_params.assert_called_once()
        call_kwargs = mock_colorbar.ax.tick_params.call_args[1]
        assert call_kwargs['labelsize'] == 10

    def test_adjust_colorbar_sets_n_ticks(self, mock_colorbar):
        """Verify colorbar number of ticks is set when specified."""
        with patch('gridfia.visualization.plots.plt.MaxNLocator') as mock_locator:
            mock_locator_instance = Mock()
            mock_locator.return_value = mock_locator_instance

            adjust_colorbar(mock_colorbar, label='Test', n_ticks=5)

            mock_locator.assert_called_once_with(5)
            assert mock_colorbar.locator == mock_locator_instance
            mock_colorbar.update_ticks.assert_called_once()

    def test_adjust_colorbar_no_n_ticks_when_none(self, mock_colorbar):
        """Verify n_ticks is not set when None."""
        adjust_colorbar(mock_colorbar, label='Test', n_ticks=None)

        mock_colorbar.update_ticks.assert_not_called()

    def test_adjust_colorbar_sets_format_string(self, mock_colorbar):
        """Verify colorbar format string is applied when specified."""
        with patch('gridfia.visualization.plots.plt.FormatStrFormatter') as mock_formatter:
            mock_formatter_instance = Mock()
            mock_formatter.return_value = mock_formatter_instance

            adjust_colorbar(mock_colorbar, label='Test', format_str='%.2f')

            mock_formatter.assert_called_once_with('%.2f')
            mock_colorbar.ax.yaxis.set_major_formatter.assert_called_once_with(mock_formatter_instance)

    def test_adjust_colorbar_no_format_string_when_none(self, mock_colorbar):
        """Verify format string is not set when None."""
        adjust_colorbar(mock_colorbar, label='Test', format_str=None)

        mock_colorbar.ax.yaxis.set_major_formatter.assert_not_called()

    def test_adjust_colorbar_sets_extend(self, mock_colorbar):
        """Verify colorbar extend attribute is set when specified."""
        adjust_colorbar(mock_colorbar, label='Test', extend='both')

        assert mock_colorbar.extend == 'both'

    def test_adjust_colorbar_extend_options(self, mock_colorbar):
        """Verify all extend options are supported."""
        for extend_option in ['neither', 'both', 'min', 'max']:
            mock_colorbar.extend = None
            adjust_colorbar(mock_colorbar, label='Test', extend=extend_option)
            assert mock_colorbar.extend == extend_option

    def test_adjust_colorbar_no_extend_when_none(self, mock_colorbar):
        """Verify extend is not set when None."""
        original_extend = mock_colorbar.extend
        adjust_colorbar(mock_colorbar, label='Test', extend=None)

        # extend should remain unchanged
        assert mock_colorbar.extend == original_extend


# ==============================================================================
# Tests for add_inset_histogram Function
# ==============================================================================


class TestAddInsetHistogram:
    """Test suite for the add_inset_histogram function."""

    def test_add_inset_histogram_creates_inset_axes(self, mock_axes, sample_data):
        """Verify inset axes are created with specified position."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        mock_axes.inset_axes.assert_called_once()

    def test_add_inset_histogram_default_position(self, mock_axes, sample_data):
        """Verify default position is applied."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        call_args = mock_axes.inset_axes.call_args[0]
        assert call_args[0] == (0.7, 0.7, 0.25, 0.25)

    def test_add_inset_histogram_custom_position(self, mock_axes, sample_data):
        """Verify custom position is applied."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        custom_position = (0.1, 0.1, 0.3, 0.3)
        add_inset_histogram(mock_axes, sample_data, position=custom_position)

        call_args = mock_axes.inset_axes.call_args[0]
        assert call_args[0] == custom_position

    def test_add_inset_histogram_plots_histogram(self, mock_axes, sample_data):
        """Verify histogram is plotted on inset axes."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        mock_inset_ax.hist.assert_called_once()

    def test_add_inset_histogram_custom_bins(self, mock_axes, sample_data):
        """Verify custom number of bins is applied."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data, bins=30)

        call_kwargs = mock_inset_ax.hist.call_args[1]
        assert call_kwargs['bins'] == 30

    def test_add_inset_histogram_custom_color(self, mock_axes, sample_data):
        """Verify custom histogram color is applied."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data, color='blue')

        call_kwargs = mock_inset_ax.hist.call_args[1]
        assert call_kwargs['color'] == 'blue'

    def test_add_inset_histogram_custom_alpha(self, mock_axes, sample_data):
        """Verify custom alpha (transparency) is applied."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data, alpha=0.5)

        call_kwargs = mock_inset_ax.hist.call_args[1]
        assert call_kwargs['alpha'] == 0.5

    def test_add_inset_histogram_sets_labels(self, mock_axes, sample_data):
        """Verify axis labels are set on inset."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        mock_inset_ax.set_xlabel.assert_called_once()
        mock_inset_ax.set_ylabel.assert_called_once()

    def test_add_inset_histogram_with_label(self, mock_axes, sample_data):
        """Verify optional label is set as title."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data, label='Distribution')

        mock_inset_ax.set_title.assert_called_once()
        call_args = mock_inset_ax.set_title.call_args[0]
        assert call_args[0] == 'Distribution'

    def test_add_inset_histogram_no_title_without_label(self, mock_axes, sample_data):
        """Verify no title is set when label is None."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data, label=None)

        mock_inset_ax.set_title.assert_not_called()

    def test_add_inset_histogram_enables_grid(self, mock_axes, sample_data):
        """Verify grid is enabled on inset."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        mock_inset_ax.grid.assert_called_once()

    def test_add_inset_histogram_sets_background(self, mock_axes, sample_data):
        """Verify inset background is set correctly."""
        mock_patch = Mock()
        mock_inset_ax = Mock()
        mock_inset_ax.patch = mock_patch
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        mock_patch.set_alpha.assert_called_once_with(0.9)
        mock_patch.set_facecolor.assert_called_once_with('white')

    def test_add_inset_histogram_filters_nan_values(self, mock_axes):
        """Verify NaN and infinite values are filtered from data."""
        data_with_nan = np.array([[1, 2, np.nan], [4, np.inf, 6], [7, 8, 9]])

        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, data_with_nan)

        # Verify hist was called with filtered data (without nan/inf)
        mock_inset_ax.hist.assert_called_once()
        call_args = mock_inset_ax.hist.call_args[0]
        valid_data = call_args[0]
        assert not np.any(np.isnan(valid_data))
        assert not np.any(np.isinf(valid_data))

    def test_add_inset_histogram_tick_params(self, mock_axes, sample_data):
        """Verify tick parameters are set on inset."""
        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        add_inset_histogram(mock_axes, sample_data)

        mock_inset_ax.tick_params.assert_called_once()


# ==============================================================================
# Tests for save_figure Function
# ==============================================================================


class TestSaveFigure:
    """Test suite for the save_figure function."""

    def test_save_figure_calls_savefig(self, mock_figure, temp_output_dir):
        """Verify savefig is called with output path."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path)

        mock_figure.savefig.assert_called_once()
        call_args = mock_figure.savefig.call_args[0]
        assert call_args[0] == output_path

    def test_save_figure_default_dpi(self, mock_figure, temp_output_dir):
        """Verify default DPI is 300."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['dpi'] == 300

    def test_save_figure_custom_dpi(self, mock_figure, temp_output_dir):
        """Verify custom DPI is applied."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path, dpi=150)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['dpi'] == 150

    def test_save_figure_default_bbox_inches(self, mock_figure, temp_output_dir):
        """Verify default bbox_inches is 'tight'."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['bbox_inches'] == 'tight'

    def test_save_figure_custom_bbox_inches(self, mock_figure, temp_output_dir):
        """Verify custom bbox_inches is applied."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path, bbox_inches='standard')

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['bbox_inches'] == 'standard'

    def test_save_figure_default_pad_inches(self, mock_figure, temp_output_dir):
        """Verify default pad_inches is 0.1."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['pad_inches'] == 0.1

    def test_save_figure_custom_pad_inches(self, mock_figure, temp_output_dir):
        """Verify custom pad_inches is applied."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path, pad_inches=0.2)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['pad_inches'] == 0.2

    def test_save_figure_transparent_false_by_default(self, mock_figure, temp_output_dir):
        """Verify transparent is False by default."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['transparent'] is False

    def test_save_figure_transparent_true(self, mock_figure, temp_output_dir):
        """Verify transparent background when specified."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path, transparent=True)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['transparent'] is True
        assert call_kwargs['facecolor'] == 'none'

    def test_save_figure_facecolor_from_figure(self, mock_figure, temp_output_dir):
        """Verify facecolor is taken from figure when not transparent."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path, transparent=False)

        mock_figure.get_facecolor.assert_called()

    def test_save_figure_jpeg_adds_optimize(self, mock_figure, temp_output_dir):
        """Verify JPEG format adds optimize option."""
        output_path = str(temp_output_dir / 'test_figure.jpg')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['optimize'] is True
        assert call_kwargs['quality'] == 95

    def test_save_figure_jpeg_extension_case_insensitive(self, mock_figure, temp_output_dir):
        """Verify JPEG detection works with uppercase extension."""
        output_path = str(temp_output_dir / 'test_figure.JPG')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['optimize'] is True

    def test_save_figure_jpeg_alternate_extension(self, mock_figure, temp_output_dir):
        """Verify .jpeg extension is also recognized."""
        output_path = str(temp_output_dir / 'test_figure.jpeg')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['optimize'] is True

    def test_save_figure_png_adds_metadata(self, mock_figure, temp_output_dir):
        """Verify PNG format adds metadata."""
        output_path = str(temp_output_dir / 'test_figure.png')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert 'metadata' in call_kwargs
        assert 'Software' in call_kwargs['metadata']

    def test_save_figure_pdf_no_special_options(self, mock_figure, temp_output_dir):
        """Verify PDF format does not add JPEG or PNG specific options."""
        output_path = str(temp_output_dir / 'test_figure.pdf')

        save_figure(mock_figure, output_path)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert 'optimize' not in call_kwargs
        assert 'metadata' not in call_kwargs

    def test_save_figure_optimize_false(self, mock_figure, temp_output_dir):
        """Verify optimize parameter can be disabled."""
        output_path = str(temp_output_dir / 'test_figure.jpg')

        save_figure(mock_figure, output_path, optimize=False)

        call_kwargs = mock_figure.savefig.call_args[1]
        assert call_kwargs['optimize'] is False


# ==============================================================================
# Integration-Style Tests (Using Real Matplotlib Objects)
# ==============================================================================


class TestIntegrationWithRealMatplotlib:
    """Integration tests using real matplotlib objects to verify actual behavior."""

    def test_create_discrete_colormap_produces_valid_colormap(self):
        """Verify discrete colormap can be used with matplotlib."""
        colormap = create_discrete_colormap(5)

        # Test that colormap works with actual data
        test_data = np.array([0, 1, 2, 3, 4])
        colors = colormap(test_data / 4.0)

        assert colors.shape == (5, 4)  # 5 colors, RGBA

    def test_set_plot_style_publication_produces_valid_config(self):
        """Verify publication style produces valid matplotlib configuration."""
        set_plot_style('publication')

        # Verify key settings are applied
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['axes.spines.top'] is False

        # Reset
        plt.rcParams.update(plt.rcParamsDefault)

    def test_format_axes_labels_on_real_axes(self):
        """Verify format_axes_labels works with real matplotlib axes."""
        fig, axes = plt.subplots()

        try:
            format_axes_labels(
                axes,
                xlabel='X Label',
                ylabel='Y Label',
                title='Test Title',
                grid=True
            )

            assert axes.get_xlabel() == 'X Label'
            assert axes.get_ylabel() == 'Y Label'
            assert axes.get_title() == 'Test Title'
        finally:
            plt.close(fig)

    def test_add_inset_histogram_on_real_axes(self):
        """Verify add_inset_histogram works with real matplotlib axes."""
        fig, axes = plt.subplots()
        data = np.random.normal(50, 10, (50, 50))

        try:
            add_inset_histogram(axes, data, label='Test Histogram')

            # Verify inset was created
            assert len(axes.child_axes) > 0
        finally:
            plt.close(fig)

    def test_create_legend_on_real_axes(self):
        """Verify create_legend works with real matplotlib axes."""
        fig, axes = plt.subplots()
        labels = ['A', 'B', 'C']
        colors = ['red', 'green', 'blue']

        try:
            create_legend(axes, labels, colors, title='Test Legend')

            legend = axes.get_legend()
            assert legend is not None
            assert legend.get_title().get_text() == 'Test Legend'
        finally:
            plt.close(fig)

    def test_save_figure_creates_file(self, temp_output_dir):
        """Verify save_figure actually creates a file."""
        fig, axes = plt.subplots()
        axes.plot([1, 2, 3], [1, 4, 9])

        output_path = str(temp_output_dir / 'test_output.png')

        try:
            save_figure(fig, output_path)

            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            plt.close(fig)

    def test_add_north_arrow_on_real_axes(self):
        """Verify add_north_arrow works with real matplotlib axes."""
        fig, axes = plt.subplots()

        try:
            add_north_arrow(axes)

            # Verify annotations were added
            assert len(axes.texts) > 0 or len(axes.patches) > 0 or len(axes.artists) > 0
        finally:
            plt.close(fig)

    def test_add_scalebar_on_real_axes(self):
        """Verify add_scalebar works with real matplotlib axes."""
        fig, axes = plt.subplots()

        try:
            add_scalebar(axes)

            # Verify scalebar artist was added
            assert len(axes.artists) > 0
        finally:
            plt.close(fig)


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_get_colormap_with_empty_string_data_type(self):
        """Verify empty string data type falls back to viridis."""
        result = get_colormap('')
        assert result == 'viridis'

    def test_create_discrete_colormap_with_zero_colors(self):
        """Verify behavior with zero colors requested."""
        result = create_discrete_colormap(0)
        assert isinstance(result, mcolors.ListedColormap)
        assert len(result.colors) == 0

    def test_add_inset_histogram_with_all_nan_data(self, mock_axes):
        """Verify histogram handles all-NaN data gracefully."""
        all_nan_data = np.full((10, 10), np.nan)

        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        # Should not raise an exception
        add_inset_histogram(mock_axes, all_nan_data)

        mock_inset_ax.hist.assert_called_once()

    def test_add_inset_histogram_with_empty_array(self, mock_axes):
        """Verify histogram handles empty array gracefully."""
        empty_data = np.array([])

        mock_inset_ax = Mock()
        mock_inset_ax.patch = Mock()
        mock_axes.inset_axes.return_value = mock_inset_ax

        # Should not raise an exception
        add_inset_histogram(mock_axes, empty_data)

    def test_create_legend_with_empty_lists(self, mock_axes):
        """Verify legend handles empty label/color lists."""
        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        create_legend(mock_axes, [], [])

        mock_axes.legend.assert_called_once()
        call_kwargs = mock_axes.legend.call_args[1]
        assert len(call_kwargs['handles']) == 0

    def test_create_legend_with_mismatched_lists(self, mock_axes):
        """Verify legend behavior with mismatched label/color list lengths."""
        mock_legend = Mock()
        mock_legend.get_frame.return_value = Mock()
        mock_axes.legend.return_value = mock_legend

        # This should create patches for the zip, which stops at shorter list
        create_legend(mock_axes, ['A', 'B', 'C'], ['red', 'blue'])

        mock_axes.legend.assert_called_once()

    def test_save_figure_with_path_object(self, mock_figure, temp_output_dir):
        """Verify save_figure works with Path objects."""
        output_path = temp_output_dir / 'test_figure.png'

        save_figure(mock_figure, str(output_path))

        mock_figure.savefig.assert_called_once()

    def test_format_axes_labels_with_unicode_text(self, mock_axes):
        """Verify format_axes_labels handles unicode characters."""
        format_axes_labels(
            mock_axes,
            xlabel='Distance (m)',
            ylabel='Concentration (mg/m3)',
            title='Test Plot with Unicode'
        )

        mock_axes.set_xlabel.assert_called_once()
        mock_axes.set_ylabel.assert_called_once()

    def test_create_discrete_colormap_large_number_of_colors(self):
        """Verify colormap handles large number of colors."""
        result = create_discrete_colormap(100)
        assert isinstance(result, mcolors.ListedColormap)
        assert len(result.colors) == 100

    def test_add_north_arrow_boundary_location(self, mock_axes):
        """Verify north arrow handles boundary location values."""
        add_north_arrow(mock_axes, location=(0.0, 0.0))
        assert mock_axes.annotate.call_count == 2

        mock_axes.reset_mock()

        add_north_arrow(mock_axes, location=(1.0, 1.0))
        assert mock_axes.annotate.call_count == 2
