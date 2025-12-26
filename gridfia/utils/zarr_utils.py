"""
Utilities for creating and managing Zarr stores for forest species data.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import zarr
import zarr.storage
import zarr.codecs
import numcodecs
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import xarray as xr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from ..exceptions import InvalidZarrStructure, SpeciesNotFound

console = Console()


def create_expandable_zarr_from_base_raster(
    base_raster_path: Union[str, Path],
    zarr_path: Union[str, Path],
    max_species: int = 350,
    chunk_size: Tuple[int, int, int] = (1, 1000, 1000),
    compression: str = 'lz4',
    compression_level: int = 5
) -> zarr.Group:
    """
    Create an expandable Zarr store from a base raster file.
    
    Args:
        base_raster_path: Path to the base raster (e.g., total biomass or first species)
        zarr_path: Path where the Zarr store will be created
        max_species: Maximum number of species to allocate space for
        chunk_size: Chunk dimensions (species, height, width)
        compression: Compression algorithm to use
        compression_level: Compression level
        
    Returns:
        zarr.Group: The created Zarr group
    """
    console.print(f"[cyan]Creating Zarr store from base raster: {base_raster_path}")
    
    # Read base raster metadata
    with rasterio.open(base_raster_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        dtype = src.dtypes[0]
        
        # Read the data
        base_data = src.read(1)
    
    # Create Zarr store (Zarr v3 API)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='w')
    
    # Create the main data array
    # Use Zarr v3 codec instead of numcodecs
    if compression == 'lz4':
        codec = zarr.codecs.BloscCodec(cname='lz4', clevel=compression_level, shuffle='shuffle')
    else:
        codec = zarr.codecs.BloscCodec(cname=compression, clevel=compression_level, shuffle='shuffle')
    
    # Initialize with zeros
    data_array = root.create_array(
        'biomass',
        shape=(max_species, height, width),
        chunks=chunk_size,
        dtype=dtype,
        compressors=[codec],
        fill_value=0
    )
    
    # Add the base data as the first layer (index 0 for total biomass)
    data_array[0, :, :] = base_data
    
    # Store metadata
    root.attrs['crs'] = crs.to_string()
    root.attrs['transform'] = list(transform)
    root.attrs['bounds'] = list(bounds)
    root.attrs['width'] = width
    root.attrs['height'] = height
    root.attrs['num_species'] = 1  # Will be updated as species are added
    
    # Create species metadata arrays
    root.create_array(
        'species_codes',
        shape=(max_species,),
        dtype='<U10',
        fill_value=''
    )
    
    root.create_array(
        'species_names',
        shape=(max_species,),
        dtype='<U100',
        fill_value=''
    )
    
    # Set first entry as total biomass
    root['species_codes'][0] = '0000'
    root['species_names'][0] = 'Total Biomass'
    
    console.print(f"[green]✓ Created Zarr store with shape: {data_array.shape}")
    console.print(f"[green]✓ Chunk size: {chunk_size}")
    console.print(f"[green]✓ Compression: {compression} (level {compression_level})")
    
    return root


def append_species_to_zarr(
    zarr_path: Union[str, Path],
    species_raster_path: Union[str, Path],
    species_code: str,
    species_name: str,
    validate_alignment: bool = True
) -> int:
    """
    Append a species raster to an existing Zarr store.
    
    Args:
        zarr_path: Path to the existing Zarr store
        species_raster_path: Path to the species raster file
        species_code: Species code (e.g., '0202')
        species_name: Species common name (e.g., 'Douglas-fir')
        validate_alignment: Whether to validate spatial alignment
        
    Returns:
        int: The index where the species was added
    """
    console.print(f"[cyan]Adding species {species_code} - {species_name}")
    
    # Open Zarr store (Zarr v3 API)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r+')
    
    # Get current number of species
    current_num = root.attrs['num_species']
    
    # Read species raster
    with rasterio.open(species_raster_path) as src:
        species_data = src.read(1)
        
        if validate_alignment:
            # Validate spatial alignment
            zarr_transform = Affine(*root.attrs['transform'])
            zarr_bounds = root.attrs['bounds']
            zarr_crs = CRS.from_string(root.attrs['crs'])
            
            if not np.allclose(src.transform, zarr_transform, rtol=1e-5):
                raise InvalidZarrStructure(
                    f"Transform mismatch for species {species_code}",
                    zarr_path=str(zarr_path)
                )

            if not np.allclose(src.bounds, zarr_bounds, rtol=1e-5):
                raise InvalidZarrStructure(
                    f"Bounds mismatch for species {species_code}",
                    zarr_path=str(zarr_path)
                )
            
            if src.crs != zarr_crs:
                console.print(f"[yellow]Warning: CRS mismatch. Expected {zarr_crs}, got {src.crs}")
    
    # Add species data
    root['biomass'][current_num, :, :] = species_data
    
    # Update metadata
    root['species_codes'][current_num] = species_code
    root['species_names'][current_num] = species_name
    root.attrs['num_species'] = current_num + 1
    
    console.print(f"[green]✓ Added {species_name} at index {current_num}")
    
    return current_num


def batch_append_species_from_dir(
    zarr_path: Union[str, Path],
    raster_dir: Union[str, Path],
    species_mapping: Dict[str, str],
    pattern: str = "*.tif",
    validate_alignment: bool = True
) -> None:
    """
    Batch append multiple species rasters from a directory.
    
    Args:
        zarr_path: Path to the existing Zarr store
        raster_dir: Directory containing species raster files
        species_mapping: Dictionary mapping species codes to names
        pattern: File pattern to match
        validate_alignment: Whether to validate spatial alignment
    """
    raster_dir = Path(raster_dir)
    raster_files = sorted(raster_dir.glob(pattern))
    
    if not raster_files:
        console.print(f"[red]No files found matching pattern {pattern} in {raster_dir}")
        return
    
    console.print(f"[cyan]Found {len(raster_files)} raster files to process")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Adding species to Zarr", total=len(raster_files))
        
        for raster_file in raster_files:
            # Extract species code from filename
            filename = raster_file.stem
            species_code = None
            
            # Try to find species code in filename
            for code in species_mapping:
                if code in filename:
                    species_code = code
                    break
            
            if species_code:
                species_name = species_mapping[species_code]
                try:
                    append_species_to_zarr(
                        zarr_path,
                        raster_file,
                        species_code,
                        species_name,
                        validate_alignment
                    )
                except Exception as e:
                    console.print(f"[red]Error adding {species_code}: {e}")
            else:
                console.print(f"[yellow]Warning: Could not find species code in {filename}")
            
            progress.update(task, advance=1)


def create_zarr_from_geotiffs(
    output_zarr_path: Union[str, Path],
    geotiff_paths: List[Union[str, Path]],
    species_codes: List[str],
    species_names: List[str],
    chunk_size: Tuple[int, int, int] = (1, 1000, 1000),
    compression: str = 'lz4',
    compression_level: int = 5,
    include_total: bool = True
) -> None:
    """
    Create a Zarr store from multiple GeoTIFF files.
    
    Args:
        output_zarr_path: Path for the output Zarr store
        geotiff_paths: List of paths to GeoTIFF files
        species_codes: List of species codes corresponding to each GeoTIFF
        species_names: List of species names corresponding to each GeoTIFF
        chunk_size: Chunk dimensions (species, height, width)
        compression: Compression algorithm
        compression_level: Compression level
        include_total: Whether to calculate and include total biomass as first layer
    """
    if len(geotiff_paths) != len(species_codes) or len(geotiff_paths) != len(species_names):
        raise InvalidZarrStructure(
            f"Number of paths ({len(geotiff_paths)}), codes ({len(species_codes)}), "
            f"and names ({len(species_names)}) must match",
            zarr_path=str(output_zarr_path)
        )
    
    console.print(f"[cyan]Creating Zarr store from {len(geotiff_paths)} GeoTIFF files")
    
    # Read first raster to get dimensions and metadata
    with rasterio.open(geotiff_paths[0]) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        dtype = src.dtypes[0]
    
    # Determine number of layers
    num_layers = len(geotiff_paths) + (1 if include_total else 0)
    
    # Create Zarr store (Zarr v3 API)
    store = zarr.storage.LocalStore(output_zarr_path)
    root = zarr.open_group(store=store, mode='w')
    
    # Create main data array
    # Use Zarr v3 codec
    if compression == 'lz4':
        codec = zarr.codecs.BloscCodec(cname='lz4', clevel=compression_level, shuffle='shuffle')
    else:
        codec = zarr.codecs.BloscCodec(cname=compression, clevel=compression_level, shuffle='shuffle')
    
    data_array = root.create_array(
        'biomass',
        shape=(num_layers, height, width),
        chunks=chunk_size,
        dtype=dtype,
        compressors=[codec],
        fill_value=0
    )
    
    # Create metadata arrays
    codes_array = root.create_array(
        'species_codes',
        shape=(num_layers,),
        dtype='<U10',
        fill_value=''
    )
    
    names_array = root.create_array(
        'species_names',
        shape=(num_layers,),
        dtype='<U100',
        fill_value=''
    )
    
    # Store spatial metadata
    root.attrs['crs'] = crs.to_string()
    root.attrs['transform'] = list(transform)
    root.attrs['bounds'] = list(bounds)
    root.attrs['width'] = width
    root.attrs['height'] = height
    
    # Process each species
    start_idx = 1 if include_total else 0
    total_biomass = np.zeros((height, width), dtype=dtype)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing species", total=len(geotiff_paths))
        
        for i, (path, code, name) in enumerate(zip(geotiff_paths, species_codes, species_names)):
            with rasterio.open(path) as src:
                data = src.read(1)
                
                # Validate alignment
                if src.height != height or src.width != width:
                    raise InvalidZarrStructure(
                        f"Dimension mismatch for {name}: expected ({height}, {width}), "
                        f"got ({src.height}, {src.width})",
                        zarr_path=str(output_zarr_path),
                        expected_shape=(None, height, width),
                        actual_shape=(None, src.height, src.width)
                    )
                if not np.allclose(src.transform, transform, rtol=1e-5):
                    raise InvalidZarrStructure(
                        f"Transform mismatch for {name}",
                        zarr_path=str(output_zarr_path)
                    )
                
                # Add to zarr
                idx = start_idx + i
                data_array[idx, :, :] = data
                codes_array[idx] = code
                names_array[idx] = name
                
                # Accumulate for total
                if include_total:
                    total_biomass += data
                
                progress.update(task, advance=1)
    
    # Add total biomass if requested
    if include_total:
        data_array[0, :, :] = total_biomass
        codes_array[0] = '0000'
        names_array[0] = 'Total Biomass'
    
    root.attrs['num_species'] = num_layers
    
    console.print(f"[green]✓ Created Zarr store at {output_zarr_path}")
    console.print(f"[green]✓ Shape: {data_array.shape}")
    console.print(f"[green]✓ Species: {', '.join(species_names)}")


def validate_zarr_store(zarr_path: Union[str, Path]) -> Dict:
    """
    Validate and summarize a Zarr store.
    
    Args:
        zarr_path: Path to the Zarr store
        
    Returns:
        Dict: Summary information about the Zarr store
    """
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')
    
    info = {
        'path': str(zarr_path),
        'shape': root['biomass'].shape,
        'chunks': root['biomass'].chunks,
        'dtype': str(root['biomass'].dtype),
        'compression': 'blosc' if hasattr(root['biomass'], 'codecs') else None,
        'num_species': root.attrs.get('num_species', 0),
        'crs': root.attrs.get('crs'),
        'bounds': root.attrs.get('bounds'),
        'species': []
    }
    
    # Get species information
    if 'species_codes' in root and 'species_names' in root:
        for i in range(info['num_species']):
            code = root['species_codes'][i]
            name = root['species_names'][i]
            if code:  # Skip empty entries
                info['species'].append({
                    'index': i,
                    'code': str(code),
                    'name': str(name)
                })
    
    return info