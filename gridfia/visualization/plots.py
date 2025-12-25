"""
Plotting utilities for consistent visualization styling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar


# Default colormaps for different data types
DEFAULT_COLORMAPS = {
    'biomass': 'viridis',
    'diversity': 'plasma',
    'richness': 'Spectral_r',
    'species': 'YlGn',
    'comparison': 'RdYlBu_r',
    'hotspot': 'hot_r'
}

# Default figure settings
DEFAULT_FIGURE_SETTINGS = {
    'dpi': 100,
    'facecolor': 'white',
    'edgecolor': 'none',
    'tight_layout': True
}

# Default font settings
DEFAULT_FONT_SETTINGS = {
    'family': 'sans-serif',
    'size': 12,
    'weight': 'normal'
}


def set_plot_style(style: str = 'publication') -> None:
    """
    Set a consistent plotting style.
    
    Args:
        style: Style name ('publication', 'presentation', 'default')
    """
    if style == 'publication':
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'lines.linewidth': 2.5,
            'lines.markersize': 10,
            'axes.linewidth': 1.5,
            'grid.linewidth': 1.0
        })
    else:  # default
        plt.rcParams.update(plt.rcParamsDefault)


def get_colormap(data_type: str, custom_cmap: Optional[str] = None) -> str:
    """
    Get the appropriate colormap for a data type.
    
    Args:
        data_type: Type of data being plotted
        custom_cmap: Custom colormap name (overrides default)
        
    Returns:
        Colormap name
    """
    if custom_cmap is not None:
        return custom_cmap
    
    return DEFAULT_COLORMAPS.get(data_type, 'viridis')


def create_discrete_colormap(n_colors: int, cmap_name: str = 'tab20') -> mcolors.ListedColormap:
    """
    Create a discrete colormap with n colors.
    
    Args:
        n_colors: Number of discrete colors
        cmap_name: Base colormap name
        
    Returns:
        Discrete colormap
    """
    if n_colors <= 20 and cmap_name == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
    else:
        cmap = plt.cm.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, n_colors))
    
    return mcolors.ListedColormap(colors)


def add_scalebar(ax, 
                 location: str = 'lower right',
                 length_fraction: float = 0.25,
                 box_alpha: float = 0.8,
                 font_size: int = 10,
                 color: str = 'black') -> None:
    """
    Add a scale bar to the map.
    
    Args:
        ax: Matplotlib axes
        location: Location of the scalebar
        length_fraction: Length as fraction of axes
        box_alpha: Alpha value for background box
        font_size: Font size for label
        color: Color of the scalebar
    """
    try:
        scalebar = ScaleBar(
            1.0,  # 1 meter per unit
            location=location,
            length_fraction=length_fraction,
            box_alpha=box_alpha,
            font_properties={'size': font_size},
            color=color,
            scale_loc='top'
        )
        ax.add_artist(scalebar)
    except Exception as e:
        print(f"Warning: Could not add scalebar: {e}")


def add_north_arrow(ax,
                   location: Tuple[float, float] = (0.95, 0.95),
                   size: float = 0.1,
                   color: str = 'black',
                   edge_color: str = 'white',
                   edge_width: float = 2) -> None:
    """
    Add a north arrow to the map.
    
    Args:
        ax: Matplotlib axes
        location: (x, y) position in axes coordinates
        size: Size of the arrow as fraction of axes
        color: Color of the arrow
        edge_color: Edge color
        edge_width: Edge width
    """
    x, y = location
    dx = 0
    dy = size
    
    # Add white edge for visibility
    ax.annotate('', xy=(x, y + dy), xytext=(x, y),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=edge_width + 2, color=edge_color),
                annotation_clip=False)
    
    # Add arrow
    ax.annotate('N', xy=(x, y + dy), xytext=(x, y),
                xycoords='axes fraction',
                fontsize=12, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', lw=edge_width, color=color),
                annotation_clip=False)


def format_axes_labels(ax,
                      xlabel: str = 'Easting (m)',
                      ylabel: str = 'Northing (m)',
                      title: Optional[str] = None,
                      title_fontsize: int = 14,
                      label_fontsize: int = 12,
                      tick_fontsize: int = 10,
                      grid: bool = True,
                      grid_alpha: float = 0.3) -> None:
    """
    Apply consistent formatting to axes.
    
    Args:
        ax: Matplotlib axes
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        title_fontsize: Title font size
        label_fontsize: Axis label font size
        tick_fontsize: Tick label font size
        grid: Whether to show grid
        grid_alpha: Grid transparency
    """
    # Set labels
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=15)
    
    # Format tick labels
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.ticklabel_format(style='plain', axis='both')
    
    # Add grid
    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_legend(ax,
                 labels: List[str],
                 colors: List[str],
                 title: Optional[str] = None,
                 location: str = 'best',
                 ncol: int = 1,
                 fontsize: int = 10,
                 title_fontsize: int = 11,
                 frameon: bool = True,
                 fancybox: bool = True,
                 shadow: bool = False,
                 alpha: float = 0.9) -> None:
    """
    Create a formatted legend.
    
    Args:
        ax: Matplotlib axes
        labels: Legend labels
        colors: Colors for each label
        title: Legend title
        location: Legend location
        ncol: Number of columns
        fontsize: Font size for labels
        title_fontsize: Font size for title
        frameon: Whether to show frame
        fancybox: Whether to use fancy box
        shadow: Whether to add shadow
        alpha: Legend transparency
    """
    # Create patches
    patches = [mpatches.Patch(color=color, label=label) 
              for color, label in zip(colors, labels)]
    
    # Create legend
    legend = ax.legend(
        handles=patches,
        title=title,
        loc=location,
        ncol=ncol,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        frameon=frameon,
        fancybox=fancybox,
        shadow=shadow,
        framealpha=alpha
    )
    
    # Adjust spacing
    legend.get_frame().set_linewidth(0.8)


def adjust_colorbar(cbar,
                   label: str,
                   label_fontsize: int = 11,
                   tick_fontsize: int = 9,
                   n_ticks: Optional[int] = None,
                   format_str: Optional[str] = None,
                   extend: Optional[str] = None) -> None:
    """
    Adjust colorbar appearance.
    
    Args:
        cbar: Colorbar object
        label: Colorbar label
        label_fontsize: Label font size
        tick_fontsize: Tick label font size
        n_ticks: Number of ticks (auto if None)
        format_str: Format string for tick labels
        extend: Extension style ('neither', 'both', 'min', 'max')
    """
    # Set label
    cbar.set_label(label, fontsize=label_fontsize)
    
    # Set tick label size
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    # Set number of ticks
    if n_ticks is not None:
        cbar.locator = plt.MaxNLocator(n_ticks)
        cbar.update_ticks()
    
    # Format tick labels
    if format_str is not None:
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(format_str))
    
    # Set extension
    if extend is not None:
        cbar.extend = extend


def add_inset_histogram(ax,
                       data: np.ndarray,
                       position: Tuple[float, float, float, float] = (0.7, 0.7, 0.25, 0.25),
                       bins: int = 50,
                       color: str = 'gray',
                       alpha: float = 0.7,
                       label: Optional[str] = None) -> None:
    """
    Add an inset histogram to show data distribution.
    
    Args:
        ax: Main axes
        data: Data to plot
        position: Position as (left, bottom, width, height) in axes coords
        bins: Number of histogram bins
        color: Histogram color
        alpha: Transparency
        label: Histogram label
    """
    # Create inset axes
    inset_ax = ax.inset_axes(position)
    
    # Plot histogram
    valid_data = data[np.isfinite(data)]
    inset_ax.hist(valid_data, bins=bins, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    # Format inset
    inset_ax.set_xlabel('Value', fontsize=8)
    inset_ax.set_ylabel('Count', fontsize=8)
    if label:
        inset_ax.set_title(label, fontsize=9)
    
    inset_ax.tick_params(labelsize=7)
    inset_ax.grid(True, alpha=0.3)
    
    # Add background
    inset_ax.patch.set_alpha(0.9)
    inset_ax.patch.set_facecolor('white')


def save_figure(fig,
               output_path: str,
               dpi: int = 300,
               bbox_inches: str = 'tight',
               pad_inches: float = 0.1,
               transparent: bool = False,
               optimize: bool = True) -> None:
    """
    Save figure with consistent settings.
    
    Args:
        fig: Figure to save
        output_path: Output file path
        dpi: Resolution
        bbox_inches: Bounding box setting
        pad_inches: Padding around figure
        transparent: Transparent background
        optimize: Optimize file size (for JPEG)
    """
    save_kwargs = {
        'dpi': dpi,
        'bbox_inches': bbox_inches,
        'pad_inches': pad_inches,
        'transparent': transparent,
        'facecolor': fig.get_facecolor() if not transparent else 'none'
    }
    
    # Add format-specific options
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        save_kwargs['optimize'] = optimize
        save_kwargs['quality'] = 95
    elif output_path.lower().endswith('.png'):
        save_kwargs['metadata'] = {'Software': 'BigMap Visualization'}
    
    fig.savefig(output_path, **save_kwargs)