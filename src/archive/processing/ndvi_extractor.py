"""
NDVI data extraction and processing functionality.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping

from src.utils.error_handler import ErrorHandler, ProcessingError

class NDVIExtractor:
    """Handles extraction and processing of NDVI values from raster data."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the NDVI extractor.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
    def extract_ndvi_timeseries(self,
                               property_geom: Dict,
                               ndvi_files: List[Path],
                               scale_factor: float = 0.0001) -> pd.DataFrame:
        """
        Extract NDVI time series for a property.
        
        Args:
            property_geom: GeoJSON-like geometry object
            ndvi_files: List of paths to NDVI files
            scale_factor: Factor to scale NDVI values by (default: 0.0001)
            
        Returns:
            DataFrame with NDVI statistics for each timestamp
            
        Raises:
            ProcessingError: If extraction fails
        """
        try:
            results = []
            
            for file_path in ndvi_files:
                ndvi_stats = self.extract_ndvi_stats(
                    file_path,
                    property_geom,
                    scale_factor
                )
                
                if ndvi_stats:
                    results.append(ndvi_stats)
            
            if not results:
                self.logger.warning("No valid NDVI values extracted")
                return pd.DataFrame()
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context="extracting NDVI time series"
            )
    
    def extract_ndvi_stats(self,
                          ndvi_path: Path,
                          geometry: Dict,
                          scale_factor: float = 0.0001) -> Optional[Dict[str, Any]]:
        """
        Extract NDVI statistics for a single timestamp and geometry.
        
        Args:
            ndvi_path: Path to NDVI file
            geometry: GeoJSON-like geometry object
            scale_factor: Factor to scale NDVI values by
            
        Returns:
            Dictionary containing NDVI statistics or None if extraction fails
            
        Raises:
            ProcessingError: If extraction fails
        """
        try:
            with rasterio.open(ndvi_path) as src:
                # Mask the raster with the geometry
                out_image, out_transform = mask(src, [geometry], crop=True)
                
                # Get valid data (exclude nodata values)
                ndvi_data = out_image[0]
                if src.nodata is not None:
                    ndvi_data = ndvi_data[ndvi_data != src.nodata]
                
                if len(ndvi_data) == 0:
                    self.logger.warning(f"No valid NDVI values found in {ndvi_path}")
                    return None
                
                # Scale the values
                ndvi_data = ndvi_data * scale_factor
                
                # Calculate statistics
                stats = {
                    'date': self._extract_date(ndvi_path),
                    'mean_ndvi': float(np.mean(ndvi_data)),
                    'median_ndvi': float(np.median(ndvi_data)),
                    'min_ndvi': float(np.min(ndvi_data)),
                    'max_ndvi': float(np.max(ndvi_data)),
                    'std_ndvi': float(np.std(ndvi_data)),
                    'pixel_count': len(ndvi_data)
                }
                
                return stats
                
        except Exception as e:
            self.logger.warning(f"Failed to extract NDVI from {ndvi_path}: {str(e)}")
            return None
    
    def process_property_batch(self,
                             properties: gpd.GeoDataFrame,
                             ndvi_files: List[Path],
                             scale_factor: float = 0.0001) -> pd.DataFrame:
        """
        Process NDVI extraction for a batch of properties.
        
        Args:
            properties: GeoDataFrame containing property geometries
            ndvi_files: List of paths to NDVI files
            scale_factor: Factor to scale NDVI values by
            
        Returns:
            DataFrame with NDVI statistics for each property and timestamp
            
        Raises:
            ProcessingError: If batch processing fails
        """
        try:
            results = []
            
            for idx, row in properties.iterrows():
                self.logger.info(f"Processing property {idx}")
                
                property_geom = mapping(row.geometry)
                property_stats = self.extract_ndvi_timeseries(
                    property_geom,
                    ndvi_files,
                    scale_factor
                )
                
                if not property_stats.empty:
                    # Add property identifier
                    property_stats['property_id'] = row.name
                    results.append(property_stats)
                
            if not results:
                return pd.DataFrame()
            
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context="processing property batch"
            )
    
    def _extract_date(self, file_path: Path) -> datetime:
        """
        Extract date from NDVI filename.
        Assumes filename contains date in format YYYYMMDD or YYYY_MM_DD.
        
        Args:
            file_path: Path to NDVI file
            
        Returns:
            datetime object
        """
        try:
            # Extract date from filename
            filename = file_path.stem
            date_str = ''.join(filter(str.isdigit, filename))[:8]
            return datetime.strptime(date_str, '%Y%m%d')
            
        except Exception:
            self.logger.warning(
                f"Could not extract date from {file_path}, using file modification time"
            )
            return datetime.fromtimestamp(file_path.stat().st_mtime) 