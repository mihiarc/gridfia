"""Module for handling large-scale geospatial data processing."""
import os
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
import dask.array as da
import dask_geopandas as dgpd
import geopandas as gpd
from shapely.prepared import prep
import logging

logger = logging.getLogger(__name__)

class LargeDataProcessor:
    """Handles processing of large geospatial datasets efficiently."""
    
    def __init__(self, chunk_size: Tuple[int, int] = (1024, 1024)):
        """Initialize processor with chunk size for tiled processing."""
        self.chunk_size = chunk_size
        
    def process_raster_in_chunks(
        self, 
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        process_func: callable,
        dtype: np.dtype = np.float32
    ) -> None:
        """Process a large raster file in chunks to manage memory usage.
        
        Args:
            input_path: Path to input raster
            output_path: Path to save processed raster
            process_func: Function to apply to each chunk
            dtype: Output data type
        """
        with rasterio.open(input_path) as src:
            # Create output raster with same properties
            profile = src.profile.copy()
            profile.update(dtype=dtype)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Process by chunks
                for ji, window in src.block_windows(1):
                    data = src.read(1, window=window)
                    processed = process_func(data)
                    dst.write(processed, 1, window=window)
    
    def calculate_ndvi_chunked(
        self,
        nir_path: Union[str, Path],
        red_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> None:
        """Calculate NDVI using chunked processing for large images.
        
        Args:
            nir_path: Path to NIR band raster
            red_path: Path to Red band raster
            output_path: Path to save NDVI raster
        """
        def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
            """Calculate NDVI from NIR and Red bands."""
            mask = (nir > 0) & (red > 0)
            ndvi = np.zeros_like(nir, dtype=np.float32)
            ndvi[mask] = (nir[mask] - red[mask]) / (nir[mask] + red[mask])
            return ndvi
        
        with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
            profile = nir_src.profile.copy()
            profile.update(dtype='float32', count=1)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                for ji, window in nir_src.block_windows(1):
                    nir_data = nir_src.read(1, window=window)
                    red_data = red_src.read(1, window=window)
                    ndvi = calculate_ndvi(nir_data, red_data)
                    dst.write(ndvi, 1, window=window)
    
    def process_vector_chunked(
        self,
        input_path: Union[str, Path],
        process_func: callable,
        output_path: Optional[Union[str, Path]] = None,
        partition_col: Optional[str] = None
    ) -> Union[gpd.GeoDataFrame, None]:
        """Process vector data in chunks using dask-geopandas.
        
        Args:
            input_path: Path to input vector data
            process_func: Function to apply to each chunk
            output_path: Optional path to save processed data
            partition_col: Column to use for partitioning
            
        Returns:
            Processed GeoDataFrame if output_path is None
        """
        # Read data with dask-geopandas
        ddf = dgpd.read_parquet(input_path)
        
        if partition_col:
            ddf = ddf.set_index(partition_col)
        
        # Apply processing function
        processed = ddf.map_partitions(process_func)
        
        if output_path:
            processed.to_parquet(output_path)
            return None
        else:
            return processed.compute()
    
    def spatial_join_optimized(
        self,
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        predicate: str = 'intersects'
    ) -> gpd.GeoDataFrame:
        """Perform optimized spatial join using prepared geometries.
        
        Args:
            left_gdf: Left GeoDataFrame
            right_gdf: Right GeoDataFrame
            predicate: Spatial predicate to use
            
        Returns:
            Joined GeoDataFrame
        """
        # Prepare geometries for faster operations
        prepared_geoms = [prep(geom) for geom in right_gdf.geometry]
        
        # Create spatial index
        spatial_index = right_gdf.sindex
        
        def find_matches(geometry):
            # Get candidate matches
            possible_matches_index = list(spatial_index.intersection(geometry.bounds))
            possible_matches = right_gdf.iloc[possible_matches_index]
            
            # Test against prepared geometries
            match_index = [
                possible_matches.index[i] for i, prepared_geom 
                in enumerate(prepared_geoms) 
                if getattr(prepared_geom, predicate)(geometry)
            ]
            
            return match_index
        
        # Apply spatial join
        left_gdf['match_index'] = left_gdf.geometry.apply(find_matches)
        
        # Explode to get one row per match
        result = left_gdf.explode('match_index')
        
        # Merge with right GeoDataFrame
        return result.merge(
            right_gdf,
            left_on='match_index',
            right_index=True,
            suffixes=('_left', '_right')
        ) 