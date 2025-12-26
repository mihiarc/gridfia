#!/usr/bin/env python3
"""
Shared utilities for GridFIA examples.

This module contains common functions used across multiple examples
to avoid code duplication and provide consistent functionality.
"""

from pathlib import Path
import numpy as np
import zarr
import zarr.storage
import zarr.codecs
import rasterio
import shutil
from typing import List, Optional, Dict, Any, Tuple
from rich.console import Console
from dataclasses import dataclass
from typing import Any

console = Console()


def _validate_bbox(bbox: Tuple[float, float, float, float], crs: str) -> None:
    """
    Validate bounding box coordinates for given CRS.

    Args:
        bbox: Bounding box coordinates (xmin, ymin, xmax, ymax)
        crs: Coordinate reference system

    Raises:
        ValueError: If bbox is invalid
    """
    xmin, ymin, xmax, ymax = bbox

    # Basic geometric validation
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid bbox: min values must be less than max values")

    # CRS-specific validation
    if crs in ["4326", "4269"]:  # WGS84/NAD83
        if not (-180 <= xmin <= 180 and -180 <= xmax <= 180):
            raise ValueError("Longitude out of range for geographic CRS")
        if not (-90 <= ymin <= 90 and -90 <= ymax <= 90):
            raise ValueError("Latitude out of range for geographic CRS")
    elif crs in ["3857"]:  # Web Mercator
        # Approximate bounds for Web Mercator (covers most of Earth)
        web_mercator_max = 20037508.34  # ~180 degrees
        if not (-web_mercator_max <= xmin <= web_mercator_max and
                -web_mercator_max <= xmax <= web_mercator_max):
            raise ValueError("X coordinates out of range for Web Mercator")
        if not (-web_mercator_max <= ymin <= web_mercator_max and
                -web_mercator_max <= ymax <= web_mercator_max):
            raise ValueError("Y coordinates out of range for Web Mercator")


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters to avoid magic numbers."""
    biomass_threshold: float = 1.0
    diversity_percentile: int = 90
    richness_threshold: float = 0.5
    chunk_size: Tuple[int, int, int] = (1, 1000, 1000)
    max_pixels: int = 1_000_000  # Maximum pixels to load in memory
    sample_ratio: float = 0.1  # Default sampling ratio for large arrays
    nodata_value: float = -9999.0
    presence_threshold: float = 1.0


def cleanup_example_outputs(directories: Optional[List[str]] = None) -> None:
    """
    Remove example output directories to clean up after running examples.

    Args:
        directories: List of directory names to remove. If None, removes common ones.
    """
    if directories is None:
        directories = [
            "quickstart_data",
            "quickstart_results",
            "wake_results",
            "configs",
            "analysis_results",
            "species_data",
            "output"
        ]

    # Security: Only allow relative paths in current directory
    current_dir = Path.cwd()

    for dir_name in directories:
        # Validate path is safe
        if ".." in dir_name or dir_name.startswith("/") or dir_name.startswith("~"):
            console.print(f"[red]Security: Skipping unsafe path: {dir_name}[/red]")
            continue

        dir_path = current_dir / dir_name

        # Additional safety check: ensure path is within current directory
        try:
            dir_path.resolve().relative_to(current_dir)
        except ValueError:
            console.print(f"[red]Security: Path outside current directory: {dir_name}[/red]")
            continue

        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                console.print(f"[green]Cleaned up:[/green] {dir_name}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not remove {dir_name}: {e}[/yellow]")


def safe_download_species(api, state: Optional[str] = None, county: Optional[str] = None,
                         bbox: Optional[Tuple[float, float, float, float]] = None,
                         crs: str = "4326",
                         species_codes: List[str] = None,
                         output_dir: str = "species_data",
                         max_retries: int = 3) -> List[Path]:
    """
    Download species data with error handling and retry logic.

    Args:
        api: GridFIA instance
        state: State name (optional if bbox provided)
        county: County name (optional)
        bbox: Bounding box coordinates (xmin, ymin, xmax, ymax)
        crs: Coordinate reference system for bbox
        species_codes: List of species codes to download
        output_dir: Output directory for downloads
        max_retries: Maximum number of retry attempts

    Returns:
        List of downloaded file paths

    Raises:
        ConnectionError: If download fails after all retries
    """
    # Validate bbox if provided
    if bbox:
        _validate_bbox(bbox, crs)

    attempt = 0
    while attempt < max_retries:
        try:
            files = api.download_species(
                state=state,
                county=county,
                bbox=bbox,
                crs=crs,
                species_codes=species_codes,
                output_dir=output_dir
            )
            return files
        except ConnectionError as e:
            attempt += 1
            if attempt >= max_retries:
                console.print(f"[red]Download failed after {max_retries} attempts: {e}[/red]")
                raise
            console.print(f"[yellow]Download attempt {attempt} failed, retrying...[/yellow]")
        except Exception as e:
            console.print(f"[red]Unexpected error during download: {e}[/red]")
            raise

    return []


def safe_open_zarr_biomass(zarr_path: Path) -> Tuple[zarr.Group, zarr.Array]:
    """
    Safely open zarr store and return both group and biomass array.

    Supports both Zarr v3 group format (with 'biomass' array) and legacy
    array format (single array at root level).

    Args:
        zarr_path: Path to zarr store

    Returns:
        Tuple of (group_root, biomass_array). For legacy array format,
        both elements point to the same array.

    Raises:
        ValueError: If zarr store cannot be opened or biomass array not found
    """
    try:
        # Use Zarr v3 API with LocalStore
        store = zarr.storage.LocalStore(str(zarr_path))

        # First try to open as group (Zarr v3 format)
        try:
            root = zarr.open_group(store=store, mode='r')

            # Check if this is a group with biomass array
            if 'biomass' in root:
                return root, root['biomass']
            else:
                raise ValueError(f"'biomass' array not found in zarr group: {zarr_path}")
        except (zarr.errors.NodeTypeValidationError, zarr.errors.ContainsArrayError):
            # This is a legacy array format (single array, not group)
            # Open as array instead
            arr = zarr.open_array(store=store, mode='r')
            return arr, arr

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        raise ValueError(f"Cannot open zarr store {zarr_path}: {e}")


def safe_load_zarr_with_memory_check(zarr_path: Path,
                                    config: Optional[AnalysisConfig] = None) -> np.ndarray:
    """
    Load zarr array with memory management.

    Args:
        zarr_path: Path to zarr array
        config: Analysis configuration

    Returns:
        Numpy array (possibly downsampled if too large)
    """
    if config is None:
        config = AnalysisConfig()

    try:
        root, z = safe_open_zarr_biomass(zarr_path)

        # Calculate total pixels
        total_pixels = z.shape[1] * z.shape[2]

        if total_pixels > config.max_pixels:
            # Calculate safe sample size
            sample_ratio = np.sqrt(config.max_pixels / total_pixels)
            h = int(z.shape[1] * sample_ratio)
            w = int(z.shape[2] * sample_ratio)

            console.print(f"[yellow]Large array detected ({total_pixels:,} pixels)[/yellow]")
            console.print(f"[yellow]Downsampling to {h}x{w} for memory safety[/yellow]")

            # Sample the array
            sample = z[:, ::int(1/sample_ratio), ::int(1/sample_ratio)]
            return sample
        else:
            return z[:]

    except Exception as e:
        console.print(f"[red]Error loading zarr array: {e}[/red]")
        raise


def create_zarr_from_rasters(
    raster_dir: Path,
    output_path: Path,
    config: Optional[AnalysisConfig] = None
) -> Path:
    """
    Create a zarr array from species raster files with error handling.

    Args:
        raster_dir: Directory containing GeoTIFF files
        output_path: Output path for zarr array
        config: Analysis configuration

    Returns:
        Path to created zarr array

    Raises:
        ValueError: If no raster files found
        IOError: If raster reading fails
    """
    if config is None:
        config = AnalysisConfig()

    raster_files = sorted(Path(raster_dir).glob("*.tif"))
    console.print(f"Found {len(raster_files)} species rasters")

    if not raster_files:
        raise ValueError(f"No .tif files found in {raster_dir}")

    try:
        # Read first raster for dimensions and metadata
        with rasterio.open(raster_files[0]) as src:
            height, width = src.shape
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        # Create zarr group with biomass array using Zarr v3 API
        n_layers = len(raster_files) + 1  # +1 for total biomass
        store = zarr.storage.LocalStore(str(output_path))
        root = zarr.open_group(store=store, mode='w')

        # Use BloscCodec for compression (Zarr v3 pattern)
        blosc_codec = zarr.codecs.BloscCodec(cname='lz4', clevel=5, shuffle='shuffle')

        z = root.create_array(
            'biomass',
            shape=(n_layers, height, width),
            chunks=config.chunk_size,
            dtype='float32',
            fill_value=config.nodata_value,
            compressors=[blosc_codec]
        )

        # Store metadata on the group
        root.attrs['crs'] = str(crs)
        root.attrs['transform'] = transform.to_gdal()
        root.attrs['bounds'] = bounds
        root.attrs['layer_names'] = ['total_biomass'] + [f.stem for f in raster_files]
        root.attrs['nodata'] = config.nodata_value

        # Load species data
        total = np.zeros((height, width), dtype='float32')
        for i, raster_file in enumerate(raster_files, start=1):
            console.print(f"Processing {raster_file.name}...")
            try:
                with rasterio.open(raster_file) as src:
                    data = src.read(1).astype('float32')
                    data[data < 0] = 0  # Clean nodata
                    z[i, :, :] = data
                    total += data
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to read {raster_file.name}: {e}[/yellow]")
                continue

        # Store total biomass in first layer
        z[0, :, :] = total

        console.print(f"[green]Created zarr array:[/green] {output_path}")
        return output_path

    except Exception as e:
        console.print(f"[red]Error creating zarr array: {e}[/red]")
        raise


def create_sample_zarr(output_path: Path, n_species: int = 3) -> Path:
    """
    Create a sample zarr group for testing with error handling.

    Args:
        output_path: Output path for zarr
        n_species: Number of species layers (plus total)

    Returns:
        Path to created zarr group
    """
    try:
        # Create zarr group (not just array)
        store = zarr.storage.LocalStore(str(output_path))
        root = zarr.open_group(store=store, mode='w')

        shape = (n_species + 1, 100, 100)  # +1 for total layer
        biomass_array = root.create_array(
            'biomass',
            shape=shape,
            chunks=(1, 50, 50),
            dtype='float32'
        )

        # Generate sample data
        np.random.seed(42)
        total_biomass = np.zeros((shape[1], shape[2]), dtype='float32')

        for i in range(shape[0]):
            # Create species distribution with spatial pattern
            x = np.linspace(0, 10, shape[1])
            y = np.linspace(0, 10, shape[2])
            X, Y = np.meshgrid(x, y)

            # Different pattern for each species
            if i == 0:  # Skip total for now
                continue
            else:
                freq = i * 0.5
                data = np.abs(np.sin(X * freq) * np.cos(Y * freq) * 50)
                biomass_array[i, :, :] = data
                total_biomass += data

        # Store total biomass in first layer
        biomass_array[0, :, :] = total_biomass

        # Add metadata to the group
        root.attrs['crs'] = 'EPSG:32617'
        root.attrs['num_species'] = n_species + 1

        # Create species metadata arrays
        species_codes = ['0000'] + [f'{i:04d}' for i in range(1, n_species + 1)]
        species_names = ['All Species Combined'] + [f'Sample Species {i}' for i in range(1, n_species + 1)]

        root.create_array('species_codes', data=np.array(species_codes, dtype='U10'))
        root.create_array('species_names', data=np.array(species_names, dtype='U50'))

        console.print(f"[green]Created sample zarr:[/green] {output_path}")
        return output_path

    except Exception as e:
        console.print(f"[red]Error creating sample zarr: {e}[/red]")
        raise


def print_zarr_info(zarr_path: Path) -> None:
    """Print information about a zarr store with error handling."""
    try:
        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.open_group(store=store, mode='r')
        biomass_array = root['biomass']
        console.print(f"\n[cyan]Zarr Store Info:[/cyan]")
        console.print(f"  Shape: {biomass_array.shape}")
        console.print(f"  Chunks: {biomass_array.chunks}")
        console.print(f"  Dtype: {biomass_array.dtype}")
        console.print(f"  Size: {biomass_array.nbytes / 1e6:.2f} MB")

        # Display species information if available
        if 'species_codes' in root and 'species_names' in root:
            num_species = root.attrs.get('num_species', 0)
            console.print(f"  Species: {num_species}")
            if num_species > 0:
                species_list = []
                for i in range(min(3, num_species)):  # Show first 3
                    code = root['species_codes'][i]
                    name = root['species_names'][i]
                    if code:
                        species_list.append(f"{code} ({name})")
                if species_list:
                    console.print(f"    {', '.join(species_list)}{'...' if num_species > 3 else ''}")
    except Exception as e:
        console.print(f"[red]Error reading zarr store: {e}[/red]")


def calculate_basic_stats(zarr_path: Path, sample_size: Optional[int] = 1000) -> Dict[str, Any]:
    """
    Calculate basic statistics from a zarr store.

    Args:
        zarr_path: Path to zarr store group
        sample_size: Size of sample to use (None for full array)

    Returns:
        Dictionary of statistics
    """
    try:
        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.open_group(store=store, mode='r')
        z = root['biomass']

        # Sample data if specified
        if sample_size and z.shape[1] > sample_size:
            data = z[:, :sample_size, :sample_size]
            console.print(f"Sampling {sample_size}x{sample_size} pixels for statistics")
        else:
            data = z[:]

        # Calculate stats
        total_biomass = data[0]
        forest_mask = total_biomass > 0
        forest_pixels = np.sum(forest_mask)

        stats = {
            'total_pixels': total_biomass.size,
            'forest_pixels': int(forest_pixels),
            'forest_coverage_pct': 100 * forest_pixels / total_biomass.size,
            'mean_biomass': float(np.mean(total_biomass[forest_mask])) if forest_pixels > 0 else 0,
            'max_biomass': float(np.max(total_biomass)) if forest_pixels > 0 else 0,
            'total_biomass_mg': float(np.sum(total_biomass))
        }

        # Species richness
        if data.shape[0] > 1:
            species_present = np.sum(data[1:] > 0, axis=0)  # Skip TOTAL layer
            stats['mean_richness'] = float(np.mean(species_present[forest_mask])) if forest_pixels > 0 else 0
            stats['max_richness'] = int(np.max(species_present)) if forest_pixels > 0 else 0
        else:
            stats['mean_richness'] = 0
            stats['max_richness'] = 0

        # Print summary
        console.print(f"\n[bold]Forest Statistics:[/bold]")
        console.print(f"  Forest coverage: {stats['forest_coverage_pct']:.1f}%")
        console.print(f"  Mean biomass: {stats['mean_biomass']:.2f} Mg/ha")
        console.print(f"  Max biomass: {stats['max_biomass']:.2f} Mg/ha")
        if data.shape[0] > 1:
            console.print(f"  Mean species richness: {stats['mean_richness']:.2f}")
            console.print(f"  Max species richness: {stats['max_richness']}")

        return stats

    except Exception as e:
        console.print(f"[red]Error calculating statistics: {e}[/red]")
        return {}


def add_zarr_metadata(
    zarr_array: zarr.Array,
    species_codes: List[str],
    species_names: List[str],
    crs: Any = None,
    transform: Any = None,
    bounds: Any = None
) -> None:
    """Add standard metadata to a zarr array.

    Args:
        zarr_array: Zarr array to add metadata to
        species_codes: List of species codes
        species_names: List of species names
        crs: Coordinate reference system
        transform: Affine transform
        bounds: Bounding box
    """
    zarr_array.attrs.update({
        'species_codes': species_codes,
        'species_names': species_names,
    })

    if crs is not None:
        zarr_array.attrs['crs'] = str(crs)
    if transform is not None:
        zarr_array.attrs['transform'] = list(transform) if hasattr(transform, '__iter__') else transform
    if bounds is not None:
        zarr_array.attrs['bounds'] = list(bounds) if hasattr(bounds, '__iter__') else bounds

    zarr_array.attrs['description'] = 'Forest biomass by species'
    zarr_array.attrs['units'] = 'Mg/ha'


def validate_species_codes(api, species_codes: List[str]) -> List[str]:
    """
    Validate species codes against available species.

    Args:
        api: GridFIA instance
        species_codes: List of species codes to validate

    Returns:
        List of valid species codes
    """
    try:
        all_species = api.list_species()
        valid_codes = {s.species_code for s in all_species}

        validated = []
        for code in species_codes:
            if code in valid_codes:
                validated.append(code)
            else:
                console.print(f"[yellow]Warning: Species code {code} not found[/yellow]")

        return validated

    except Exception as e:
        console.print(f"[red]Error validating species codes: {e}[/red]")
        return species_codes  # Return original if validation fails