"""
Visualization module for BigMap.

This module provides comprehensive mapping and visualization capabilities:
- Forest biomass heatmaps
- Species distribution maps
- Diversity index visualizations (Shannon, Simpson)
- Species richness maps
- Multi-species comparison maps
- Publication-quality figure export
"""

from .mapper import ZarrMapper
from .plots import (
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
    DEFAULT_COLORMAPS,
    DEFAULT_FIGURE_SETTINGS,
    DEFAULT_FONT_SETTINGS
)

__all__ = [
    'ZarrMapper',
    'set_plot_style',
    'get_colormap',
    'create_discrete_colormap',
    'add_scalebar',
    'add_north_arrow',
    'format_axes_labels',
    'create_legend',
    'adjust_colorbar',
    'add_inset_histogram',
    'save_figure',
    'DEFAULT_COLORMAPS',
    'DEFAULT_FIGURE_SETTINGS',
    'DEFAULT_FONT_SETTINGS'
]