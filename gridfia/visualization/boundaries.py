"""
Utilities for loading and plotting geographic boundaries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import contextily as ctx
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import requests
import ssl
import certifi
import zipfile
import io
from rich.console import Console

console = Console()

# Default cache directory for boundary files
BOUNDARY_CACHE_DIR = Path.home() / ".gridfia" / "boundaries"
BOUNDARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# URLs for boundary data sources
BOUNDARY_SOURCES = {
    'states': {
        'url': 'https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip',
        'name_field': 'name',
        'abbr_field': 'postal',
        'cache_name': 'us_states_50m.gpkg'
    },
    'states_hires': {
        'url': 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip',
        'name_field': 'name',
        'abbr_field': 'postal',
        'cache_name': 'us_states_10m.gpkg'
    },
    'counties': {
        'url': 'https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip',
        'name_field': 'NAME',
        'state_field': 'STATE_NAME',
        'cache_name': 'us_counties_20m.gpkg'
    }
}

# State name to abbreviation mapping
STATE_ABBR = {
    'montana': 'MT', 'idaho': 'ID', 'wyoming': 'WY', 'north dakota': 'ND',
    'south dakota': 'SD', 'washington': 'WA', 'oregon': 'OR', 'california': 'CA',
    'nevada': 'NV', 'utah': 'UT', 'colorado': 'CO', 'arizona': 'AZ',
    'new mexico': 'NM', 'texas': 'TX', 'oklahoma': 'OK', 'kansas': 'KS',
    'nebraska': 'NE', 'iowa': 'IA', 'missouri': 'MO', 'arkansas': 'AR',
    'louisiana': 'LA', 'mississippi': 'MS', 'alabama': 'AL', 'tennessee': 'TN',
    'kentucky': 'KY', 'illinois': 'IL', 'indiana': 'IN', 'ohio': 'OH',
    'west virginia': 'WV', 'virginia': 'VA', 'north carolina': 'NC',
    'south carolina': 'SC', 'georgia': 'GA', 'florida': 'FL', 'michigan': 'MI',
    'wisconsin': 'WI', 'minnesota': 'MN', 'pennsylvania': 'PA', 'new york': 'NY',
    'vermont': 'VT', 'new hampshire': 'NH', 'maine': 'ME', 'massachusetts': 'MA',
    'rhode island': 'RI', 'connecticut': 'CT', 'new jersey': 'NJ',
    'delaware': 'DE', 'maryland': 'MD', 'alaska': 'AK', 'hawaii': 'HI'
}


def download_boundaries(boundary_type: str = 'states', force: bool = False) -> Path:
    """
    Download and cache boundary files.
    
    Args:
        boundary_type: Type of boundaries ('states', 'states_hires', 'counties')
        force: Force re-download even if cached
        
    Returns:
        Path to cached boundary file
    """
    if boundary_type not in BOUNDARY_SOURCES:
        raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    source = BOUNDARY_SOURCES[boundary_type]
    cache_path = BOUNDARY_CACHE_DIR / source['cache_name']
    
    if cache_path.exists() and not force:
        console.print(f"[green]Using cached boundaries:[/green] {cache_path}")
        return cache_path
    
    console.print(f"[cyan]Downloading {boundary_type} boundaries...[/cyan]")
    
    try:
        # Download the zip file with custom SSL context for gov sites
        session = requests.Session()

        # Create custom SSL context that's more lenient for government sites
        if 'census.gov' in source['url']:
            # For census.gov, we need to handle their certificate chain differently
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            session.verify = False  # Temporary workaround for census.gov SSL issues
            console.print("[yellow]Warning: Using relaxed SSL verification for census.gov[/yellow]")
        else:
            session.verify = certifi.where()

        response = session.get(source['url'], stream=True)
        response.raise_for_status()
        
        # Save response content to a temporary file for geopandas to read
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        try:
            # Extract and convert to GeoPackage
            with zipfile.ZipFile(tmp_path) as zf:
                # Find the shapefile
                shp_file = None
                for name in zf.namelist():
                    if name.endswith('.shp'):
                        shp_file = name.replace('.shp', '')
                        break

                if not shp_file:
                    raise ValueError("No shapefile found in archive")

                # Read directly from zip
                gdf = gpd.read_file(f"zip://{tmp_path}/{shp_file}.shp")
            
            # Filter for US states if using Natural Earth data
            if 'admin' in gdf.columns and 'states' in boundary_type:
                gdf = gdf[gdf['admin'] == 'United States of America'].copy()
            
            # Save as GeoPackage for faster access
            gdf.to_file(cache_path, driver='GPKG')

            console.print(f"[green]✓ Downloaded and cached boundaries[/green]")
            return cache_path
        finally:
            # Clean up temporary file
            import os
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    except Exception as e:
        console.print(f"[red]Error downloading boundaries: {e}[/red]")
        raise


def load_state_boundary(
    state: str,
    crs: Optional[Union[str, CRS]] = None,
    boundary_type: str = 'states',
    simplify_tolerance: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Load boundary for a specific state.
    
    Args:
        state: State name or abbreviation
        crs: Target CRS (if None, keeps original)
        boundary_type: Type of boundaries to use
        simplify_tolerance: Simplification tolerance in map units
        
    Returns:
        GeoDataFrame with state boundary
    """
    # Normalize state name
    state_lower = state.lower()
    if state_lower in STATE_ABBR:
        state_abbr = STATE_ABBR[state_lower]
        state_name = state_lower.title()
    else:
        state_abbr = state.upper()
        # Find full name from abbreviation
        state_name = None
        for name, abbr in STATE_ABBR.items():
            if abbr == state_abbr:
                state_name = name.title()
                break
    
    # Download/load boundaries
    boundary_path = download_boundaries(boundary_type)
    gdf = gpd.read_file(boundary_path)
    
    # Filter for state
    source = BOUNDARY_SOURCES[boundary_type]
    if state_name:
        state_gdf = gdf[
            (gdf[source['name_field']].str.lower() == state_name.lower()) |
            (gdf[source['abbr_field']] == state_abbr)
        ].copy()
    else:
        state_gdf = gdf[gdf[source['abbr_field']] == state_abbr].copy()
    
    if state_gdf.empty:
        raise ValueError(f"State not found: {state}")
    
    # Simplify if requested
    if simplify_tolerance is not None:
        state_gdf['geometry'] = state_gdf['geometry'].simplify(simplify_tolerance)
    
    # Reproject if needed
    if crs is not None:
        if isinstance(crs, str):
            crs = CRS.from_string(crs)
        state_gdf = state_gdf.to_crs(crs.to_string())
    
    return state_gdf


def load_counties_for_state(
    state: str,
    crs: Optional[Union[str, CRS]] = None,
    simplify_tolerance: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Load county boundaries for a specific state.
    
    Args:
        state: State name or abbreviation
        crs: Target CRS
        simplify_tolerance: Simplification tolerance
        
    Returns:
        GeoDataFrame with county boundaries
    """
    # Get state info
    state_lower = state.lower()
    if state_lower in STATE_ABBR:
        state_name = state_lower.title()
    else:
        # Find full name from abbreviation
        state_abbr = state.upper()
        state_name = None
        for name, abbr in STATE_ABBR.items():
            if abbr == state_abbr:
                state_name = name.title()
                break
    
    if not state_name:
        raise ValueError(f"State not found: {state}")
    
    # Download/load boundaries
    boundary_path = download_boundaries('counties')
    gdf = gpd.read_file(boundary_path)
    
    # Filter for state
    source = BOUNDARY_SOURCES['counties']
    counties_gdf = gdf[gdf[source['state_field']].str.lower() == state_name.lower()].copy()
    
    if counties_gdf.empty:
        raise ValueError(f"No counties found for state: {state}")
    
    # Simplify if requested
    if simplify_tolerance is not None:
        counties_gdf['geometry'] = counties_gdf['geometry'].simplify(simplify_tolerance)
    
    # Reproject if needed
    if crs is not None:
        if isinstance(crs, str):
            crs = CRS.from_string(crs)
        counties_gdf = counties_gdf.to_crs(crs.to_string())
    
    return counties_gdf


def plot_boundaries(
    ax: Axes,
    boundaries: gpd.GeoDataFrame,
    color: str = 'black',
    linewidth: float = 1.5,
    alpha: float = 1.0,
    fill: bool = False,
    fill_color: Optional[str] = None,
    fill_alpha: float = 0.1,
    label: Optional[str] = None,
    zorder: int = 10
) -> None:
    """
    Plot boundaries on axes.
    
    Args:
        ax: Matplotlib axes
        boundaries: GeoDataFrame with boundaries
        color: Line color
        linewidth: Line width
        alpha: Line transparency
        fill: Whether to fill polygons
        fill_color: Fill color (if None, uses line color)
        fill_alpha: Fill transparency
        label: Legend label
        zorder: Plot order
    """
    if fill:
        boundaries.plot(
            ax=ax,
            facecolor=fill_color or color,
            edgecolor='none',
            alpha=fill_alpha,
            zorder=zorder - 1
        )
    
    boundaries.plot(
        ax=ax,
        facecolor='none',
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder
    )


def add_basemap(
    ax: Axes,
    zoom: Optional[int] = None,
    source: str = 'OpenStreetMap',
    alpha: float = 1.0,
    crs: Optional[Union[str, CRS]] = None,
    attribution: bool = False,
    attribution_size: int = 8,
    reset_extent: bool = True
) -> None:
    """
    Add a basemap to the axes.
    
    Args:
        ax: Matplotlib axes
        zoom: Zoom level (auto if None)
        source: Basemap source or provider
        alpha: Basemap transparency
        crs: CRS of the axes (defaults to EPSG:3857)
        attribution: Whether to add attribution text
        attribution_size: Font size for attribution
        reset_extent: Whether to reset extent after adding basemap
    """
    # Get current extent before adding basemap
    if reset_extent:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    
    # Set default CRS if not provided
    if crs is None:
        crs = 'EPSG:3857'
    elif isinstance(crs, CRS):
        crs = crs.to_string()
    
    # Get tile provider
    if isinstance(source, str):
        if source == 'OpenStreetMap':
            provider = ctx.providers.OpenStreetMap.Mapnik
        elif source == 'CartoDB':
            provider = ctx.providers.CartoDB.Positron
        elif source == 'CartoDB_dark':
            provider = ctx.providers.CartoDB.DarkMatter
        elif source == 'Stamen':
            provider = ctx.providers.Stamen.Terrain
        elif source == 'ESRI':
            provider = ctx.providers.Esri.WorldImagery
        else:
            # Try to get from ctx.providers
            provider = eval(f"ctx.providers.{source}")
    else:
        provider = source
    
    # Add basemap
    try:
        ctx.add_basemap(
            ax,
            zoom=zoom,
            source=provider,
            alpha=alpha,
            crs=crs,
            attribution=attribution,
            attribution_size=attribution_size
        )
        
        # Reset extent if requested
        if reset_extent:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
    except Exception as e:
        console.print(f"[yellow]Warning: Could not add basemap: {e}[/yellow]")


def clip_boundaries_to_extent(
    boundaries: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    buffer: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Clip boundaries to a specific extent.
    
    Args:
        boundaries: GeoDataFrame with boundaries
        extent: (xmin, xmax, ymin, ymax)
        buffer: Buffer to add around extent
        
    Returns:
        Clipped GeoDataFrame
    """
    xmin, xmax, ymin, ymax = extent
    
    # Create clipping box
    clip_box = box(
        xmin - buffer,
        ymin - buffer,
        xmax + buffer,
        ymax + buffer
    )
    
    # Clip boundaries
    clipped = boundaries.copy()
    clipped['geometry'] = boundaries.intersection(clip_box)
    
    # Remove empty geometries
    clipped = clipped[~clipped['geometry'].is_empty]
    
    return clipped


def get_basemap_zoom_level(extent: Tuple[float, float, float, float]) -> int:
    """
    Calculate appropriate zoom level for extent.
    
    Args:
        extent: (xmin, xmax, ymin, ymax) in Web Mercator
        
    Returns:
        Zoom level
    """
    xmin, xmax, ymin, ymax = extent
    
    # Calculate extent in degrees (approximate)
    # Web Mercator to degrees conversion
    lon_min = xmin * 180 / 20037508.34
    lon_max = xmax * 180 / 20037508.34
    
    # Calculate appropriate zoom level
    lon_diff = abs(lon_max - lon_min)
    
    if lon_diff > 10:
        return 6
    elif lon_diff > 5:
        return 7
    elif lon_diff > 2:
        return 8
    elif lon_diff > 1:
        return 9
    elif lon_diff > 0.5:
        return 10
    elif lon_diff > 0.25:
        return 11
    else:
        return 12