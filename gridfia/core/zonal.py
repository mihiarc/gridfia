"""
Zonal statistics for forest metrics using exactextract.

Computes sub-pixel accurate zonal statistics from raster data
within polygon zones. Supports GeoTIFF files, xarray DataArrays,
and Zarr-backed datasets for cloud-native workflows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import geopandas as gpd
import numpy as np
from rich.console import Console

from ..utils.polygon_utils import load_polygon

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_STATS = ["mean", "sum", "min", "max", "stdev", "count"]


def _sanitize_name(name: str) -> str:
    """Convert a species name to a column-safe label.

    'Loblolly Pine' -> 'loblolly_pine'
    'Douglas-fir'   -> 'douglas_fir'
    """
    return name.lower().replace(" ", "_").replace("-", "_")


def calculate_zonal_stats(
    rasters: Union[str, Path, List[Union[str, Path]], Dict[str, Union[str, Path]]],
    zones: Union[str, Path, gpd.GeoDataFrame],
    stats: Optional[List[str]] = None,
    include_cols: Optional[List[str]] = None,
    output_csv: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics for raster data within polygon zones.

    Uses exactextract for sub-pixel accurate statistics. Each raster cell
    is weighted by the exact fraction covered by each polygon, providing
    more accurate results than center-of-pixel approaches.

    Parameters
    ----------
    rasters : str, Path, list, or dict
        Raster input(s). Can be:
        - Single file path to a GeoTIFF
        - List of file paths (columns named from filenames)
        - Dict mapping names to file paths (columns named from keys)
    zones : str, Path, or GeoDataFrame
        Polygon zones to aggregate within. Can be a file path
        (GeoJSON, GeoPackage, Shapefile) or a GeoDataFrame.
    stats : list of str, optional
        Statistics to compute. Default: mean, sum, min, max, stdev, count.
        Supported: mean, sum, min, max, stdev, variance, count, median,
        minority, majority, variety, coefficient_of_variation, frac,
        weighted_mean, weighted_sum, and more.
        See exactextract documentation for the full list.
    include_cols : list of str, optional
        Columns from zones to include in the output DataFrame.
        If None, all non-geometry columns are included.
    output_csv : str or Path, optional
        Path to save results as CSV. Geometry is excluded from CSV output.

    Returns
    -------
    GeoDataFrame
        Zones with computed statistics as additional columns.
        Column names follow the pattern ``{raster_name}_{stat}``
        for multi-raster input, or just ``{stat}`` for single raster.

    Examples
    --------
    >>> from gridfia.core.zonal import calculate_zonal_stats
    >>> result = calculate_zonal_stats(
    ...     "output/species_richness.tif",
    ...     "zones.geojson",
    ...     stats=["mean", "sum", "count"]
    ... )

    >>> # Multiple rasters with named columns
    >>> result = calculate_zonal_stats(
    ...     {"richness": "output/species_richness.tif",
    ...      "diversity": "output/shannon_diversity.tif"},
    ...     counties_gdf,
    ...     output_csv="zonal_results.csv"
    ... )
    """
    import rasterio

    try:
        from exactextract import exact_extract
    except ImportError:
        raise ImportError(
            "The 'exactextract' package is required for zonal statistics. "
            "Install with: uv pip install 'gridfia[zonal]'"
        )

    if stats is None:
        stats = DEFAULT_STATS.copy()

    # Load zones
    if not isinstance(zones, gpd.GeoDataFrame):
        zones_gdf = load_polygon(zones)
    else:
        zones_gdf = zones.copy()

    if zones_gdf.empty:
        raise ValueError("Zones GeoDataFrame is empty")

    # Determine which columns to carry through
    non_geom_cols = [c for c in zones_gdf.columns if c != "geometry"]
    if include_cols is not None:
        missing_cols = [c for c in include_cols if c not in non_geom_cols]
        if missing_cols:
            logger.warning(f"Requested columns not found in zones: {missing_cols}")
        carry_cols = [c for c in include_cols if c in non_geom_cols]
    else:
        carry_cols = non_geom_cols

    # Normalize raster input
    raster_dict = _normalize_raster_input(rasters)

    # Validate raster files exist
    for name, raster_path in raster_dict.items():
        path = Path(raster_path)
        if not path.exists():
            raise FileNotFoundError(f"Raster file not found: {path}")

    # Reproject zones to match the first raster's CRS if needed
    first_raster = next(iter(raster_dict.values()))
    with rasterio.open(first_raster) as src:
        raster_crs = src.crs
    if zones_gdf.crs is not None and not zones_gdf.crs.equals(raster_crs):
        console.print(
            f"[yellow]Reprojecting zones from {zones_gdf.crs} "
            f"to raster CRS {raster_crs}[/yellow]"
        )
        zones_gdf = zones_gdf.to_crs(raster_crs)

    single_raster = len(raster_dict) == 1

    console.print(
        f"[cyan]Computing zonal stats for {len(raster_dict)} raster(s) "
        f"across {len(zones_gdf)} zone(s)...[/cyan]"
    )

    # Run exactextract for each raster and collect results
    all_results = {}
    for raster_name, raster_path in raster_dict.items():
        result_df = exact_extract(
            str(raster_path),
            zones_gdf,
            stats,
            include_cols=carry_cols if carry_cols else None,
            output="pandas",
        )

        # Rename stat columns with raster prefix for multi-raster input
        for stat in stats:
            col_name = stat
            if col_name in result_df.columns:
                if single_raster:
                    all_results[stat] = result_df[col_name].values
                else:
                    all_results[f"{raster_name}_{stat}"] = result_df[col_name].values

    # Build output GeoDataFrame
    output_data = {}
    for col in carry_cols:
        if col in zones_gdf.columns:
            output_data[col] = zones_gdf[col].values

    output_data.update(all_results)

    result_gdf = gpd.GeoDataFrame(
        output_data,
        geometry=zones_gdf.geometry.values,
        crs=zones_gdf.crs,
    )

    console.print(
        f"[green]Computed {len(stats)} stats for {len(raster_dict)} raster(s) "
        f"across {len(result_gdf)} zone(s)[/green]"
    )

    # Save to CSV if requested
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.drop(columns="geometry").to_csv(output_csv, index=False)
        console.print(f"[green]Saved zonal stats to {output_csv}[/green]")
        logger.info(f"Saved zonal stats CSV to {output_csv}")

    return result_gdf


def calculate_zonal_stats_from_zarr(
    zarr_path: Union[str, Path],
    zones: Union[str, Path, gpd.GeoDataFrame],
    layers: Optional[List[str]] = None,
    stats: Optional[List[str]] = None,
    include_cols: Optional[List[str]] = None,
    output_csv: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics directly from a GridFIA Zarr store.

    Reads species biomass layers from the Zarr store as xarray DataArrays
    and computes zonal statistics using exactextract's XArrayRasterSource.
    This is the cloud-native path — no intermediate GeoTIFFs required.

    Parameters
    ----------
    zarr_path : str or Path
        Path to GridFIA Zarr store (local or cloud via fsspec).
    zones : str, Path, or GeoDataFrame
        Polygon zones to aggregate within.
    layers : list of str, optional
        Species codes to include. If None, all species layers are used
        (excluding the total biomass layer at index 0).
    stats : list of str, optional
        Statistics to compute. Default: mean, sum, min, max, stdev, count.
    include_cols : list of str, optional
        Columns from zones to include in the output.
    output_csv : str or Path, optional
        Path to save results as CSV.

    Returns
    -------
    GeoDataFrame
        Zones with computed statistics as additional columns.
        Column names use human-readable species names:
        ``{species_name}_{stat}`` (e.g., ``douglas_fir_mean``).
        Falls back to species code if name is unavailable.
        The ``attrs["species_code_to_name"]`` dict maps codes to names.

    Examples
    --------
    >>> from gridfia.core.zonal import calculate_zonal_stats_from_zarr
    >>> result = calculate_zonal_stats_from_zarr(
    ...     "data/forest.zarr",
    ...     "zones.geojson",
    ...     layers=["0202", "0122"],
    ...     stats=["mean", "sum"]
    ... )
    >>> # Columns: douglas_fir_mean, douglas_fir_sum, ponderosa_pine_mean, ...
    >>> result.attrs["species_code_to_name"]
    {'0202': 'Douglas-fir', '0122': 'Ponderosa Pine'}
    """
    import xarray as xr

    try:
        from exactextract import exact_extract
    except ImportError:
        raise ImportError(
            "The 'exactextract' package is required for zonal statistics. "
            "Install with: uv pip install 'gridfia[zonal]'"
        )
    try:
        import rioxarray  # noqa: F401 — registers the .rio accessor on xarray
    except ImportError:
        raise ImportError(
            "The 'rioxarray' package is required for Zarr-based zonal statistics. "
            "Install with: uv pip install 'gridfia[zonal]'"
        )

    from ..utils.zarr_utils import ZarrStore

    if stats is None:
        stats = DEFAULT_STATS.copy()

    # Load zones
    if not isinstance(zones, gpd.GeoDataFrame):
        zones_gdf = load_polygon(zones)
    else:
        zones_gdf = zones.copy()

    if zones_gdf.empty:
        raise ValueError("Zones GeoDataFrame is empty")

    # Determine columns to carry through
    non_geom_cols = [c for c in zones_gdf.columns if c != "geometry"]
    if include_cols is not None:
        carry_cols = [c for c in include_cols if c in non_geom_cols]
    else:
        carry_cols = non_geom_cols

    # Open Zarr store
    store = ZarrStore.from_path(zarr_path)
    try:
        species_codes = store.species_codes
        species_names = store.species_names
        crs = store.crs
        transform = store.transform
        biomass = store.biomass

        # Reproject zones to match raster CRS if needed
        if zones_gdf.crs is not None and not zones_gdf.crs.equals(crs):
            console.print(
                f"[yellow]Reprojecting zones from {zones_gdf.crs} "
                f"to raster CRS {crs}[/yellow]"
            )
            zones_gdf = zones_gdf.to_crs(crs)

        # Determine which layers to process
        if layers is not None:
            layer_indices = []
            layer_codes = []
            for code in layers:
                idx = store.get_species_index(code)
                layer_indices.append(idx)
                layer_codes.append(code)
        else:
            # Skip index 0 (total biomass) unless explicitly requested
            layer_indices = list(range(1, len(species_codes)))
            layer_codes = [species_codes[i] for i in layer_indices]

        # Build human-readable column labels from species names, fall back to codes
        layer_labels = []
        code_to_name = {}
        for idx, code in zip(layer_indices, layer_codes):
            if idx < len(species_names) and species_names[idx]:
                label = _sanitize_name(species_names[idx])
                code_to_name[code] = species_names[idx]
            else:
                label = code
                code_to_name[code] = code
            layer_labels.append(label)

        console.print(
            f"[cyan]Computing zonal stats for {len(layer_indices)} layer(s) "
            f"from Zarr store across {len(zones_gdf)} zone(s)...[/cyan]"
        )
        for code, name in code_to_name.items():
            console.print(f"  [dim]{code}[/dim] -> [bold]{name}[/bold]")

        # Build xarray DataArrays with spatial metadata for exactextract
        height, width = biomass.shape[1], biomass.shape[2]
        x_coords = np.array(
            [transform.c + (col + 0.5) * transform.a for col in range(width)]
        )
        y_coords = np.array(
            [transform.f + (row + 0.5) * transform.e for row in range(height)]
        )

        all_results = {}
        for layer_idx, label in zip(layer_indices, layer_labels):
            data = biomass[layer_idx, :, :]
            if isinstance(data, np.ndarray):
                layer_data = data
            else:
                layer_data = np.array(data)

            da = xr.DataArray(
                layer_data,
                dims=["y", "x"],
                coords={"y": y_coords, "x": x_coords},
            )
            da.rio.write_crs(crs, inplace=True)
            da.rio.write_transform(transform, inplace=True)

            result_df = exact_extract(
                da,
                zones_gdf,
                stats,
                include_cols=carry_cols if carry_cols else None,
                output="pandas",
            )

            for stat in stats:
                if stat in result_df.columns:
                    all_results[f"{label}_{stat}"] = result_df[stat].values
    finally:
        store.close()

    # Build output GeoDataFrame
    output_data = {}
    for col in carry_cols:
        if col in zones_gdf.columns:
            output_data[col] = zones_gdf[col].values

    output_data.update(all_results)

    result_gdf = gpd.GeoDataFrame(
        output_data,
        geometry=zones_gdf.geometry.values,
        crs=zones_gdf.crs,
    )

    # Attach species code-to-name mapping as DataFrame metadata
    result_gdf.attrs["species_code_to_name"] = code_to_name

    console.print(
        f"[green]Computed {len(stats)} stats for {len(layer_indices)} layer(s) "
        f"across {len(result_gdf)} zone(s)[/green]"
    )

    # Save to CSV if requested
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.drop(columns="geometry").to_csv(output_csv, index=False)
        console.print(f"[green]Saved zonal stats to {output_csv}[/green]")
        logger.info(f"Saved zonal stats CSV to {output_csv}")

    return result_gdf


def _normalize_raster_input(
    rasters: Union[str, Path, List[Union[str, Path]], Dict[str, Union[str, Path]]],
) -> Dict[str, Path]:
    """Normalize raster input into a dict mapping names to paths."""
    if isinstance(rasters, dict):
        return {name: Path(path) for name, path in rasters.items()}
    elif isinstance(rasters, (str, Path)):
        path = Path(rasters)
        return {path.stem: path}
    elif isinstance(rasters, list):
        return {Path(p).stem: Path(p) for p in rasters}
    else:
        raise TypeError(f"Unsupported raster input type: {type(rasters)}")
