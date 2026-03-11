"""
Zonal statistics for forest metrics using exactextract.

Computes sub-pixel accurate zonal statistics from raster data
within polygon zones. Supports GeoTIFF files, xarray DataArrays,
and Zarr-backed datasets for cloud-native workflows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import geopandas as gpd
import numpy as np
from rich.console import Console

from ..utils.polygon_utils import load_polygon

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_STATS = ["mean", "sum", "min", "max", "stdev", "count"]

# GridFIA Zarr stores use fill_value=0 for non-forested pixels.
# Setting this as nodata ensures exactextract excludes zero-biomass
# cells from statistics (e.g., mean won't be diluted by non-forest).
_ZARR_NODATA = 0.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sanitize_name(name: str) -> str:
    """Convert a species name to a column-safe label.

    'Loblolly Pine' -> 'loblolly_pine'
    'Douglas-fir'   -> 'douglas_fir'
    """
    return name.lower().replace(" ", "_").replace("-", "_")


def _load_and_prepare_zones(
    zones: Union[str, Path, gpd.GeoDataFrame],
    target_crs,
    include_cols: Optional[List[str]] = None,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """Load zones, validate, reproject to target CRS, and select carry columns."""
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
            logger.warning("Requested columns not found in zones: %s", missing_cols)
        carry_cols = [c for c in include_cols if c in non_geom_cols]
    else:
        carry_cols = non_geom_cols

    # Reproject zones to match raster CRS if needed
    if (
        zones_gdf.crs is not None
        and target_crs is not None
        and not zones_gdf.crs.equals(target_crs)
    ):
        console.print(
            f"[yellow]Reprojecting zones from {zones_gdf.crs} "
            f"to raster CRS {target_crs}[/yellow]"
        )
        zones_gdf = zones_gdf.to_crs(target_crs)

    return zones_gdf, carry_cols


def _build_output_gdf(
    zones_gdf: gpd.GeoDataFrame,
    carry_cols: List[str],
    stat_results: Dict[str, np.ndarray],
    attrs: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """Build output GeoDataFrame from zones and computed stat columns."""
    output_data = {}
    for col in carry_cols:
        if col in zones_gdf.columns:
            output_data[col] = zones_gdf[col].values

    output_data.update(stat_results)

    result_gdf = gpd.GeoDataFrame(
        output_data,
        geometry=zones_gdf.geometry.values,
        crs=zones_gdf.crs,
    )

    if attrs:
        result_gdf.attrs.update(attrs)

    return result_gdf


def _save_csv(gdf: gpd.GeoDataFrame, output_csv: Union[str, Path]) -> None:
    """Save GeoDataFrame to CSV, excluding geometry."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").to_csv(output_csv, index=False)
    console.print(f"[green]Saved zonal stats to {output_csv}[/green]")
    logger.info("Saved zonal stats CSV to %s", output_csv)


def _resolve_layer_labels(
    layer_indices: List[int],
    layer_codes: List[str],
    species_names: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Build unique, human-readable labels for each layer.

    Detects collisions from sanitization (e.g. "Red Cedar" and "Red-Cedar"
    both becoming "red_cedar") and disambiguates by appending the species code.
    """
    labels = []
    code_to_name = {}
    seen_labels: Dict[str, str] = {}  # label -> first code that used it

    for idx, code in zip(layer_indices, layer_codes):
        if idx < len(species_names) and species_names[idx]:
            label = _sanitize_name(species_names[idx])
            code_to_name[code] = species_names[idx]
        else:
            label = code
            code_to_name[code] = code

        # Disambiguate duplicate labels by appending the species code
        if label in seen_labels:
            # Also fix the first occurrence if not already fixed
            first_code = seen_labels[label]
            if first_code is not None:
                first_idx = next(
                    i for i, l in enumerate(labels) if l == label
                )
                labels[first_idx] = f"{label}_{first_code}"
                seen_labels[label] = None  # mark as already fixed
            label = f"{label}_{code}"
        else:
            seen_labels[label] = code

        labels.append(label)

    return labels, code_to_name


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


# ---------------------------------------------------------------------------
# GeoTIFF-based zonal statistics
# ---------------------------------------------------------------------------


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

    # Normalize raster input
    raster_dict = _normalize_raster_input(rasters)

    # Validate raster files exist
    for name, raster_path in raster_dict.items():
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")

    # Get CRS from first raster for zone reprojection
    first_raster = next(iter(raster_dict.values()))
    with rasterio.open(first_raster) as src:
        raster_crs = src.crs

    zones_gdf, carry_cols = _load_and_prepare_zones(zones, raster_crs, include_cols)

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

        for stat in stats:
            if stat in result_df.columns:
                key = stat if single_raster else f"{raster_name}_{stat}"
                all_results[key] = result_df[stat].values

    console.print(
        f"[green]Computed {len(stats)} stats for {len(raster_dict)} raster(s) "
        f"across {len(zones_gdf)} zone(s)[/green]"
    )

    result_gdf = _build_output_gdf(zones_gdf, carry_cols, all_results)

    if output_csv is not None:
        _save_csv(result_gdf, output_csv)

    return result_gdf


# ---------------------------------------------------------------------------
# Zarr-based zonal statistics
# ---------------------------------------------------------------------------

# Maximum layers per exact_extract call. Limits peak memory while still
# batching enough layers to amortise the spatial-index build on zones.
_BATCH_SIZE = 50


def _extract_zarr_layers_batched(
    biomass,
    layer_indices: List[int],
    layer_labels: List[str],
    stats: List[str],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    crs,
    transform,
    zones_gdf: gpd.GeoDataFrame,
    carry_cols: List[str],
    nodata: Optional[float],
) -> Dict[str, np.ndarray]:
    """Process Zarr layers in batches using xr.Dataset for spatial-index reuse.

    Instead of calling exact_extract once per layer, layers are packed into
    an xr.Dataset and processed in a single call. This avoids rebuilding the
    spatial index on zones for every layer.
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
        import rioxarray  # noqa: F401 — registers the .rio accessor
    except ImportError:
        raise ImportError(
            "The 'rioxarray' package is required for Zarr-based zonal statistics. "
            "Install with: uv pip install 'gridfia[zonal]'"
        )

    all_results: Dict[str, np.ndarray] = {}

    for batch_start in range(0, len(layer_indices), _BATCH_SIZE):
        batch_end = min(batch_start + _BATCH_SIZE, len(layer_indices))
        batch_indices = layer_indices[batch_start:batch_end]
        batch_labels = layer_labels[batch_start:batch_end]

        # Build xr.Dataset: variable names become column prefixes in output
        data_vars = {}
        for layer_idx, label in zip(batch_indices, batch_labels):
            layer_data = np.asarray(biomass[layer_idx, :, :], dtype=np.float32)

            da = xr.DataArray(
                layer_data,
                dims=["y", "x"],
                coords={"y": y_coords, "x": x_coords},
            )
            da.rio.write_crs(crs, inplace=True)
            da.rio.write_transform(transform, inplace=True)
            if nodata is not None:
                da.rio.write_nodata(nodata, inplace=True)
            data_vars[label] = da

        ds = xr.Dataset(data_vars)
        single_var = len(data_vars) == 1

        result_df = exact_extract(
            ds,
            zones_gdf,
            stats,
            include_cols=carry_cols if carry_cols else None,
            output="pandas",
        )

        # exactextract prefixes columns with var name only for multi-variable
        # Datasets. For single-variable, columns are just the stat names.
        for label in batch_labels:
            for stat in stats:
                if single_var:
                    src_col = stat
                else:
                    src_col = f"{label}_{stat}"
                if src_col in result_df.columns:
                    all_results[f"{label}_{stat}"] = result_df[src_col].values

    return all_results


def calculate_zonal_stats_from_zarr(
    zarr_path: Union[str, Path],
    zones: Union[str, Path, gpd.GeoDataFrame],
    layers: Optional[List[str]] = None,
    stats: Optional[List[str]] = None,
    include_cols: Optional[List[str]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    nodata: Optional[float] = _ZARR_NODATA,
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics directly from a GridFIA Zarr store.

    Reads species biomass layers from the Zarr store as xarray DataArrays
    and computes zonal statistics using exactextract's XArrayRasterSource.
    This is the cloud-native path — no intermediate GeoTIFFs required.

    By default, pixels with value 0 are treated as nodata (non-forested)
    and excluded from statistics. This matches the GridFIA Zarr convention
    where ``fill_value=0`` represents non-forested cells. Pass
    ``nodata=None`` to include zero-value pixels.

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
    nodata : float or None, optional
        Value to treat as nodata (excluded from stats). Default: 0.0.
        Pass None to include all pixel values.

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
    from ..utils.zarr_utils import ZarrStore

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    if stats is None:
        stats = DEFAULT_STATS.copy()

    with ZarrStore.open(zarr_path) as store:
        species_codes = store.species_codes
        species_names = store.species_names
        crs = store.crs
        transform = store.transform
        biomass = store.biomass

        zones_gdf, carry_cols = _load_and_prepare_zones(zones, crs, include_cols)

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

        # Build labels with collision detection
        layer_labels, code_to_name = _resolve_layer_labels(
            layer_indices, layer_codes, species_names
        )

        console.print(
            f"[cyan]Computing zonal stats for {len(layer_indices)} layer(s) "
            f"from Zarr store across {len(zones_gdf)} zone(s)...[/cyan]"
        )
        for code, name in code_to_name.items():
            console.print(f"  [dim]{code}[/dim] -> [bold]{name}[/bold]")

        # Precompute coordinate arrays for xarray DataArrays
        height, width = biomass.shape[1], biomass.shape[2]
        x_coords = np.array(
            [transform.c + (col + 0.5) * transform.a for col in range(width)]
        )
        y_coords = np.array(
            [transform.f + (row + 0.5) * transform.e for row in range(height)]
        )

        # Batch layers into xr.Datasets for efficient spatial-index reuse
        all_results = _extract_zarr_layers_batched(
            biomass,
            layer_indices,
            layer_labels,
            stats,
            x_coords,
            y_coords,
            crs,
            transform,
            zones_gdf,
            carry_cols,
            nodata,
        )

    result_gdf = _build_output_gdf(
        zones_gdf,
        carry_cols,
        all_results,
        attrs={"species_code_to_name": code_to_name},
    )

    console.print(
        f"[green]Computed {len(stats)} stats for {len(layer_indices)} layer(s) "
        f"across {len(result_gdf)} zone(s)[/green]"
    )

    if output_csv is not None:
        _save_csv(result_gdf, output_csv)

    return result_gdf
