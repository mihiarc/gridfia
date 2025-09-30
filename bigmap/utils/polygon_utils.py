"""
Utilities for working with polygon boundaries and clipping raster data.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import logging

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def load_polygon(
    polygon_input: Union[str, Path, gpd.GeoDataFrame, dict],
    target_crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load polygon from various input formats.

    Parameters
    ----------
    polygon_input : str, Path, GeoDataFrame, or dict
        Polygon source - can be a file path (GeoJSON, Shapefile),
        GeoDataFrame, or GeoJSON-like dict.
    target_crs : str, optional
        Target CRS to transform polygon to.

    Returns
    -------
    GeoDataFrame
        Loaded polygon as GeoDataFrame.

    Examples
    --------
    >>> poly = load_polygon("boundary.geojson", target_crs="EPSG:3857")
    >>> poly = load_polygon(gdf)
    """
    if isinstance(polygon_input, gpd.GeoDataFrame):
        gdf = polygon_input.copy()
    elif isinstance(polygon_input, (str, Path)):
        path = Path(polygon_input)
        if not path.exists():
            raise FileNotFoundError(f"Polygon file not found: {path}")
        gdf = gpd.read_file(path)
        logger.info(f"Loaded polygon from {path}")
    elif isinstance(polygon_input, dict):
        # GeoJSON-like dict
        if 'type' in polygon_input and 'coordinates' in polygon_input:
            gdf = gpd.GeoDataFrame([{'geometry': shape(polygon_input)}], crs="EPSG:4326")
        else:
            raise ValueError("Dict input must be GeoJSON-like with 'type' and 'coordinates'")
    else:
        raise TypeError(f"Unsupported polygon input type: {type(polygon_input)}")

    if gdf.empty:
        raise ValueError("Polygon is empty")

    # Transform to target CRS if specified
    if target_crs and gdf.crs != target_crs:
        logger.info(f"Transforming polygon from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    return gdf


def clip_geotiff_to_polygon(
    input_path: Union[str, Path],
    polygon: Union[gpd.GeoDataFrame, dict],
    output_path: Optional[Union[str, Path]] = None,
    nodata: float = -9999,
    crop: bool = True,
    all_touched: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Clip a GeoTIFF to a polygon boundary.

    Parameters
    ----------
    input_path : str or Path
        Path to input GeoTIFF file.
    polygon : GeoDataFrame or dict
        Polygon to clip to. If dict, must be GeoJSON-like geometry.
    output_path : str or Path, optional
        Path to save clipped GeoTIFF. If None, doesn't save.
    nodata : float, default=-9999
        NoData value for output.
    crop : bool, default=True
        Whether to crop the output to the polygon extent.
    all_touched : bool, default=False
        Whether to include all pixels touched by polygon.

    Returns
    -------
    np.ndarray
        Clipped raster array.
    dict
        Output metadata/transform.

    Examples
    --------
    >>> data, meta = clip_geotiff_to_polygon(
    ...     "biomass.tif",
    ...     polygon_gdf,
    ...     output_path="clipped_biomass.tif"
    ... )
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with rasterio.open(input_path) as src:
        # Load polygon and ensure CRS matches
        if isinstance(polygon, dict):
            # GeoJSON-like geometry dict
            polygon_gdf = gpd.GeoDataFrame([{'geometry': shape(polygon)}], crs="EPSG:4326")
        else:
            polygon_gdf = polygon

        # Transform polygon to match raster CRS
        if polygon_gdf.crs != src.crs:
            polygon_gdf = polygon_gdf.to_crs(src.crs)

        # Get geometries for masking
        geometries = [mapping(geom) for geom in polygon_gdf.geometry]

        # Perform clipping
        try:
            clipped_data, clipped_transform = mask(
                src,
                geometries,
                crop=crop,
                nodata=nodata,
                all_touched=all_touched
            )
        except ValueError as e:
            logger.error(f"Failed to clip {input_path}: {e}")
            raise

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform,
            "nodata": nodata
        })

        # Save if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(clipped_data)

            logger.info(f"Saved clipped raster to {output_path}")

        return clipped_data, out_meta


def clip_geotiffs_batch(
    input_dir: Union[str, Path],
    polygon: Union[gpd.GeoDataFrame, dict, str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.tif",
    nodata: float = -9999,
    crop: bool = True,
    all_touched: bool = False,
    overwrite: bool = False
) -> list[Path]:
    """
    Clip multiple GeoTIFF files to a polygon boundary.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input GeoTIFF files.
    polygon : GeoDataFrame, dict, str, or Path
        Polygon to clip to. Can be GeoDataFrame, GeoJSON dict, or file path.
    output_dir : str or Path
        Directory to save clipped files.
    pattern : str, default="*.tif"
        Glob pattern to match input files.
    nodata : float, default=-9999
        NoData value for outputs.
    crop : bool, default=True
        Whether to crop outputs to polygon extent.
    all_touched : bool, default=False
        Whether to include all pixels touched by polygon.
    overwrite : bool, default=False
        Whether to overwrite existing output files.

    Returns
    -------
    list[Path]
        Paths to clipped output files.

    Examples
    --------
    >>> clipped = clip_geotiffs_batch(
    ...     "downloads/species/",
    ...     "county_boundary.geojson",
    ...     "clipped_species/"
    ... )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load polygon if it's a file path
    if isinstance(polygon, (str, Path)):
        polygon = load_polygon(polygon)

    # Find all matching files
    input_files = list(input_dir.glob(pattern))

    if not input_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
        return []

    console.print(f"[cyan]Clipping {len(input_files)} files to polygon boundary...[/cyan]")

    clipped_files = []

    for input_file in input_files:
        output_file = output_dir / input_file.name

        # Skip if exists and not overwriting
        if output_file.exists() and not overwrite:
            logger.info(f"Skipping existing file: {output_file}")
            clipped_files.append(output_file)
            continue

        try:
            clip_geotiff_to_polygon(
                input_file,
                polygon,
                output_path=output_file,
                nodata=nodata,
                crop=crop,
                all_touched=all_touched
            )
            clipped_files.append(output_file)

        except Exception as e:
            logger.error(f"Failed to clip {input_file}: {e}")
            continue

    console.print(f"[green]Successfully clipped {len(clipped_files)} files[/green]")

    return clipped_files


def get_polygon_bounds(
    polygon: Union[gpd.GeoDataFrame, dict, str, Path],
    crs: str = "EPSG:4326"
) -> Tuple[float, float, float, float]:
    """
    Get bounding box from a polygon.

    Parameters
    ----------
    polygon : GeoDataFrame, dict, str, or Path
        Polygon to get bounds from.
    crs : str, default="EPSG:4326"
        CRS for output bounds.

    Returns
    -------
    Tuple[float, float, float, float]
        Bounding box as (xmin, ymin, xmax, ymax).

    Examples
    --------
    >>> bbox = get_polygon_bounds("boundary.geojson", crs="EPSG:3857")
    """
    if not isinstance(polygon, gpd.GeoDataFrame):
        polygon = load_polygon(polygon)

    # Transform to target CRS if needed
    if polygon.crs != crs:
        polygon = polygon.to_crs(crs)

    bounds = polygon.total_bounds
    return tuple(bounds)
