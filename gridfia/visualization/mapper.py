"""
Core mapping functionality for visualizing forest data from Zarr stores.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from rasterio.transform import Affine
from rasterio.crs import CRS
from rich.console import Console
import warnings
from .boundaries import (
    load_state_boundary, plot_boundaries, add_basemap,
    get_basemap_zoom_level, clip_boundaries_to_extent
)
from ..utils.zarr_utils import ZarrStore

console = Console()


class ZarrMapper:
    """
    Main class for creating maps from Zarr stores.

    This class uses the unified ZarrStore interface for accessing Zarr data,
    providing consistent handling of both Zarr v2 and v3 formats.
    """

    def __init__(self, zarr_path: Union[str, Path, ZarrStore]):
        """
        Initialize the mapper with a Zarr store.

        Args:
            zarr_path: Path to the Zarr store or an existing ZarrStore instance
        """
        # Accept either a path or an existing ZarrStore
        if isinstance(zarr_path, ZarrStore):
            self._store = zarr_path
            self.zarr_path = zarr_path.path or Path("unknown")
            self._owns_store = False  # Don't close a store we didn't create
        else:
            self.zarr_path = Path(zarr_path)
            self._store = ZarrStore.from_path(self.zarr_path)
            self._owns_store = True  # We created this store, so we manage it

        # Cache for computed indices
        self._diversity_cache = {}

        console.print(f"[green]Loaded Zarr store:[/green] {self.zarr_path}")
        console.print(f"  Shape: {self._store.shape}")
        console.print(f"  CRS: {self._store.crs}")
        console.print(f"  Species: {self._store.num_species}")

    def close(self) -> None:
        """Close the underlying Zarr store if we own it."""
        if self._owns_store and hasattr(self, '_store'):
            self._store.close()

    def __enter__(self) -> 'ZarrMapper':
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close store when exiting context."""
        self.close()

    # Property accessors that delegate to ZarrStore
    @property
    def biomass(self):
        """The main biomass array (3D: species x height x width)."""
        return self._store.biomass

    @property
    def species_codes(self) -> List[str]:
        """List of species codes."""
        return self._store.species_codes

    @property
    def species_names(self) -> List[str]:
        """List of species names."""
        return self._store.species_names

    @property
    def crs(self) -> CRS:
        """Coordinate reference system."""
        return self._store.crs

    @property
    def transform(self) -> Affine:
        """Affine transform for georeferencing."""
        return self._store.transform

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds."""
        return self._store.bounds

    @property
    def num_species(self) -> int:
        """Number of species in the store."""
        return self._store.num_species

    def get_species_info(self) -> List[Dict[str, Any]]:
        """Get information about all species in the store."""
        return self._store.get_species_info()

    def _get_extent(self, transform: Optional[Affine] = None) -> Tuple[float, float, float, float]:
        """Get the extent for matplotlib plotting."""
        if transform is None:
            # Use the ZarrStore's get_extent method for default transform
            return self._store.get_extent()

        # If a custom transform is provided, calculate manually
        height, width = self.biomass.shape[1], self.biomass.shape[2]

        # Calculate corners
        left = transform.c
        right = transform.c + width * transform.a
        top = transform.f
        bottom = transform.f + height * transform.e

        return (left, right, bottom, top)
    
    def _normalize_data(self, data: np.ndarray, vmin: Optional[float] = None,
                       vmax: Optional[float] = None, percentile: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Normalize data for visualization.

        NaN values in the input data (which indicate failed calculations) are
        preserved in the output. They will render as blank/transparent areas
        in the visualization, making failed calculations visible to users.
        """
        # Handle NaN and infinite values - these indicate failed calculations
        valid_mask = np.isfinite(data)

        if vmin is None or vmax is None:
            valid_data = data[valid_mask]
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.percentile(valid_data, percentile[0])
                if vmax is None:
                    vmax = np.percentile(valid_data, percentile[1])
            else:
                vmin, vmax = 0, 1

        # Clip and normalize, preserving NaN values
        normalized = np.clip(data, vmin, vmax)
        if vmax > vmin:
            normalized = (normalized - vmin) / (vmax - vmin)
        else:
            # When vmax == vmin, set valid values to 0 but preserve NaN
            normalized = np.where(valid_mask, 0.0, np.nan)

        return normalized
    
    def create_species_map(self, 
                          species: Union[int, str],
                          fig_ax: Optional[Tuple[Figure, Axes]] = None,
                          cmap: str = 'viridis',
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          title: Optional[str] = None,
                          colorbar: bool = True,
                          colorbar_label: str = 'Biomass (Mg/ha)',
                          show_bounds: bool = True,
                          state_boundary: Optional[str] = None,
                          basemap: Optional[str] = None,
                          data_alpha: float = 0.8) -> Tuple[Figure, Axes]:
        """
        Create a map for a single species.
        
        Args:
            species: Species index or code
            fig_ax: Optional (figure, axes) tuple to plot on
            cmap: Colormap name
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            title: Map title (auto-generated if None)
            colorbar: Whether to add a colorbar
            colorbar_label: Label for the colorbar
            show_bounds: Whether to show map bounds
            state_boundary: State name or abbreviation to overlay boundary
            basemap: Basemap provider name (e.g., 'OpenStreetMap', 'CartoDB', 'ESRI')
            data_alpha: Transparency of data layer when using basemap
            
        Returns:
            Tuple of (figure, axes)
        """
        # Find species index
        if isinstance(species, str):
            species_idx = None
            for i in range(self.num_species):
                if str(self.species_codes[i]) == species:
                    species_idx = i
                    break
            if species_idx is None:
                raise ValueError(f"Species code '{species}' not found")
        else:
            species_idx = species
        
        if species_idx >= self.num_species:
            raise ValueError(f"Species index {species_idx} out of range (0-{self.num_species-1})")
        
        # Get species info
        try:
            species_name = str(self.species_names[species_idx])
            species_code = str(self.species_codes[species_idx])
        except (IndexError, KeyError):
            species_name = f"Species {species_idx}"
            species_code = f"{species_idx:04d}"
        
        # Create figure if not provided
        if fig_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        else:
            fig, ax = fig_ax
        
        # Load data
        console.print(f"Loading data for {species_name}...")
        data = self.biomass[species_idx, :, :]
        
        # Normalize data
        data_norm = self._normalize_data(data, vmin, vmax)
        
        # Get extent
        extent = self._get_extent()
        
        # Add basemap if requested
        if basemap:
            # Calculate zoom level
            zoom = get_basemap_zoom_level(extent)
            add_basemap(ax, zoom=zoom, source=basemap, crs=self.crs, alpha=0.8)
            
            # Use transparency for data overlay
            alpha = data_alpha
        else:
            alpha = 1.0
        
        # Create the map
        im = ax.imshow(data_norm, cmap=cmap, extent=extent, origin='upper', 
                      interpolation='nearest', aspect='equal', alpha=alpha)
        
        # Add state boundary if requested
        if state_boundary:
            try:
                # Load boundary
                state_gdf = load_state_boundary(
                    state_boundary, 
                    crs=self.crs,
                    boundary_type='states',
                    simplify_tolerance=1000  # Simplify for performance
                )
                
                # Clip to extent with buffer
                state_gdf = clip_boundaries_to_extent(
                    state_gdf,
                    (extent[0], extent[1], extent[2], extent[3]),
                    buffer=50000  # 50km buffer
                )
                
                # Plot boundary
                plot_boundaries(
                    ax, state_gdf,
                    color='black',
                    linewidth=2,
                    alpha=0.8,
                    zorder=15
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not add state boundary: {e}[/yellow]")
        
        # Add colorbar
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)
            # Set colorbar ticks to actual data values
            if vmin is not None and vmax is not None:
                cbar.mappable.set_clim(vmin, vmax)
        
        # Set title
        if title is None:
            title = f"{species_name} ({species_code})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        
        # Format axes
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        
        # Add bounds annotation if requested
        if show_bounds:
            bounds_text = f"Bounds: [{self.bounds[0]:.0f}, {self.bounds[1]:.0f}, {self.bounds[2]:.0f}, {self.bounds[3]:.0f}]"
            ax.text(0.02, 0.98, bounds_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig, ax
    
    def create_diversity_map(self,
                            diversity_type: str = 'shannon',
                            fig_ax: Optional[Tuple[Figure, Axes]] = None,
                            cmap: str = 'plasma',
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            title: Optional[str] = None,
                            colorbar: bool = True,
                            state_boundary: Optional[str] = None,
                            basemap: Optional[str] = None,
                            data_alpha: float = 0.8) -> Tuple[Figure, Axes]:
        """
        Create a diversity map (Shannon or Simpson index).
        
        Args:
            diversity_type: Type of diversity index ('shannon' or 'simpson')
            fig_ax: Optional (figure, axes) tuple
            cmap: Colormap name
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            title: Map title
            colorbar: Whether to add colorbar
            
        Returns:
            Tuple of (figure, axes)
        """
        if diversity_type not in ['shannon', 'simpson']:
            raise ValueError("diversity_type must be 'shannon' or 'simpson'")
        
        # Check cache
        cache_key = f"{diversity_type}_{vmin}_{vmax}"
        if cache_key in self._diversity_cache:
            diversity_index = self._diversity_cache[cache_key]
        else:
            console.print(f"Calculating {diversity_type} diversity index...")
            
            # Calculate diversity index
            # Skip first layer if it's total biomass
            start_idx = 1 if str(self.species_codes[0]) == '0000' else 0
            
            # Initialize diversity array
            diversity_index = np.zeros((self.biomass.shape[1], self.biomass.shape[2]), dtype=np.float32)
            
            # Process in chunks for memory efficiency
            chunk_size = 1000
            for i in range(0, self.biomass.shape[1], chunk_size):
                for j in range(0, self.biomass.shape[2], chunk_size):
                    # Get chunk bounds
                    i_end = min(i + chunk_size, self.biomass.shape[1])
                    j_end = min(j + chunk_size, self.biomass.shape[2])
                    
                    # Load chunk data for all species
                    chunk_data = self.biomass[start_idx:self.num_species, i:i_end, j:j_end]
                    
                    # Calculate total biomass per pixel
                    total_biomass = np.sum(chunk_data, axis=0)
                    
                    # Calculate proportions
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        proportions = chunk_data / total_biomass[np.newaxis, :, :]
                        proportions[~np.isfinite(proportions)] = 0
                    
                    if diversity_type == 'shannon':
                        # Shannon diversity: -sum(p * log(p))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            shannon = -np.sum(proportions * np.log(proportions + 1e-10), axis=0)
                            shannon[total_biomass == 0] = 0
                        diversity_index[i:i_end, j:j_end] = shannon
                    else:  # simpson
                        # Simpson diversity: 1 - sum(p^2)
                        simpson = 1 - np.sum(proportions ** 2, axis=0)
                        simpson[total_biomass == 0] = 0
                        diversity_index[i:i_end, j:j_end] = simpson
            
            # Cache the result
            self._diversity_cache[cache_key] = diversity_index
        
        # Create figure if not provided
        if fig_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        else:
            fig, ax = fig_ax
        
        # Normalize data
        data_norm = self._normalize_data(diversity_index, vmin, vmax)
        
        # Get extent
        extent = self._get_extent()
        
        # Add basemap if requested
        if basemap:
            zoom = get_basemap_zoom_level(extent)
            add_basemap(ax, zoom=zoom, source=basemap, crs=self.crs, alpha=0.8)
            alpha = data_alpha
        else:
            alpha = 1.0
        
        # Create the map
        im = ax.imshow(data_norm, cmap=cmap, extent=extent, origin='upper',
                      interpolation='nearest', aspect='equal', alpha=alpha)
        
        # Add state boundary if requested
        if state_boundary:
            try:
                state_gdf = load_state_boundary(
                    state_boundary, 
                    crs=self.crs,
                    boundary_type='states',
                    simplify_tolerance=1000
                )
                state_gdf = clip_boundaries_to_extent(
                    state_gdf,
                    (extent[0], extent[1], extent[2], extent[3]),
                    buffer=50000
                )
                plot_boundaries(
                    ax, state_gdf,
                    color='black',
                    linewidth=2,
                    alpha=0.8,
                    zorder=15
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not add state boundary: {e}[/yellow]")
        
        # Add colorbar
        if colorbar:
            label = 'Shannon Index' if diversity_type == 'shannon' else 'Simpson Index'
            cbar = plt.colorbar(im, ax=ax, label=label, shrink=0.8)
            if vmin is not None and vmax is not None:
                cbar.mappable.set_clim(vmin, vmax)
        
        # Set title
        if title is None:
            title = f"{diversity_type.capitalize()} Diversity"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        
        # Format axes
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax
    
    def create_richness_map(self,
                           threshold: float = 0.1,
                           fig_ax: Optional[Tuple[Figure, Axes]] = None,
                           cmap: str = 'Spectral_r',
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           title: Optional[str] = None,
                           colorbar: bool = True,
                           state_boundary: Optional[str] = None,
                           basemap: Optional[str] = None,
                           data_alpha: float = 0.8) -> Tuple[Figure, Axes]:
        """
        Create a species richness map.
        
        Args:
            threshold: Minimum biomass to count species as present
            fig_ax: Optional (figure, axes) tuple
            cmap: Colormap name
            vmin: Minimum value for color scaling
            vmax: Maximum value for color scaling
            title: Map title
            colorbar: Whether to add colorbar
            
        Returns:
            Tuple of (figure, axes)
        """
        console.print(f"Calculating species richness (threshold={threshold})...")
        
        # Skip first layer if it's total biomass
        start_idx = 1 if str(self.species_codes[0]) == '0000' else 0
        
        # Calculate richness
        richness = np.zeros((self.biomass.shape[1], self.biomass.shape[2]), dtype=np.uint8)
        
        # Process in chunks
        chunk_size = 1000
        for i in range(0, self.biomass.shape[1], chunk_size):
            for j in range(0, self.biomass.shape[2], chunk_size):
                i_end = min(i + chunk_size, self.biomass.shape[1])
                j_end = min(j + chunk_size, self.biomass.shape[2])
                
                # Count species above threshold
                chunk_data = self.biomass[start_idx:self.num_species, i:i_end, j:j_end]
                richness[i:i_end, j:j_end] = np.sum(chunk_data > threshold, axis=0)
        
        # Create figure if not provided
        if fig_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        else:
            fig, ax = fig_ax
        
        # Get extent
        extent = self._get_extent()
        
        # Add basemap if requested
        if basemap:
            zoom = get_basemap_zoom_level(extent)
            add_basemap(ax, zoom=zoom, source=basemap, crs=self.crs, alpha=0.8)
            alpha = data_alpha
        else:
            alpha = 1.0
        
        # Create the map
        im = ax.imshow(richness, cmap=cmap, extent=extent, origin='upper',
                      interpolation='nearest', aspect='equal',
                      vmin=vmin, vmax=vmax, alpha=alpha)
        
        # Add state boundary if requested
        if state_boundary:
            try:
                state_gdf = load_state_boundary(
                    state_boundary, 
                    crs=self.crs,
                    boundary_type='states',
                    simplify_tolerance=1000
                )
                state_gdf = clip_boundaries_to_extent(
                    state_gdf,
                    (extent[0], extent[1], extent[2], extent[3]),
                    buffer=50000
                )
                plot_boundaries(
                    ax, state_gdf,
                    color='black',
                    linewidth=2,
                    alpha=0.8,
                    zorder=15
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not add state boundary: {e}[/yellow]")
        
        # Add colorbar
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, label='Number of Species', shrink=0.8)
            # Set integer ticks
            max_species = np.max(richness)
            if max_species <= 10:
                cbar.set_ticks(range(0, int(max_species) + 1))
        
        # Set title
        if title is None:
            title = "Species Richness"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        
        # Format axes
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax
    
    def create_comparison_map(self,
                             species_list: List[Union[int, str]],
                             ncols: int = 2,
                             figsize: Optional[Tuple[float, float]] = None,
                             cmap: str = 'viridis',
                             shared_colorbar: bool = True) -> Figure:
        """
        Create a comparison map showing multiple species side by side.
        
        Args:
            species_list: List of species indices or codes
            ncols: Number of columns in the subplot grid
            figsize: Figure size (auto-calculated if None)
            cmap: Colormap name
            shared_colorbar: Whether to use shared color scale
            
        Returns:
            Figure object
        """
        n_species = len(species_list)
        nrows = (n_species + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        # Find global min/max if using shared colorbar
        if shared_colorbar:
            console.print("Calculating global min/max for shared colorbar...")
            global_min = np.inf
            global_max = -np.inf
            
            for species in species_list:
                if isinstance(species, str):
                    for i in range(self.num_species):
                        if str(self.species_codes[i]) == species:
                            species_idx = i
                            break
                else:
                    species_idx = species
                
                data = self.biomass[species_idx, :, :]
                valid_data = data[np.isfinite(data)]
                if len(valid_data) > 0:
                    global_min = min(global_min, np.percentile(valid_data, 2))
                    global_max = max(global_max, np.percentile(valid_data, 98))
        else:
            global_min = None
            global_max = None
        
        # Create individual maps
        for idx, species in enumerate(species_list):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            self.create_species_map(
                species=species,
                fig_ax=(fig, ax),
                cmap=cmap,
                vmin=global_min,
                vmax=global_max,
                colorbar=not shared_colorbar,
                show_bounds=False
            )
        
        # Remove empty subplots
        for idx in range(n_species, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].remove()
        
        # Add shared colorbar if requested
        if shared_colorbar and n_species > 0:
            # Create a ScalarMappable for the colorbar
            norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax, label='Biomass (Mg/ha)')
        
        fig.suptitle('Species Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def export_map(self,
                  fig: Figure,
                  output_path: Union[str, Path],
                  dpi: int = 300,
                  bbox_inches: str = 'tight',
                  transparent: bool = False) -> None:
        """
        Export a figure to file.
        
        Args:
            fig: Figure to export
            output_path: Output file path
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
            transparent: Whether to use transparent background
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        console.print(f"Exporting map to {output_path} at {dpi} DPI...")
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, 
                   transparent=transparent, facecolor='white')
        console.print(f"[green]✓ Map saved to {output_path}[/green]")