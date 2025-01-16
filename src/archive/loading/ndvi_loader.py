"""
NDVI data loading and preprocessing functionality.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import numpy as np
import rasterio
from rasterio.mask import mask

from src.utils.error_handler import ErrorHandler, ProcessingError

class NDVILoader:
    """Handles loading and basic preprocessing of NDVI data."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the NDVI loader.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
    def load_ndvi_files(self, ndvi_dir: Path, years: List[int]) -> Dict[int, List[Path]]:
        """
        Load and validate NDVI files for specified years.
        
        Args:
            ndvi_dir: Directory containing NDVI files
            years: List of years to process
            
        Returns:
            Dictionary mapping years to lists of NDVI file paths
            
        Raises:
            ProcessingError: If there are issues finding or validating NDVI files
        """
        try:
            ndvi_files: Dict[int, List[Path]] = {}
            
            for year in years:
                year_pattern = f"*{year}*.tif"
                year_files = list(ndvi_dir.glob(year_pattern))
                
                if not year_files:
                    self.logger.warning(f"No NDVI files found for year {year}")
                    continue
                    
                self.logger.info(f"Found {len(year_files)} NDVI files for year {year}")
                ndvi_files[year] = sorted(year_files)
                
            return ndvi_files
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context=f"loading NDVI files from {ndvi_dir}"
            )
    
    def extract_ndvi_for_geometry(self, 
                                ndvi_path: Path,
                                geometry,
                                scale_factor: float = 0.0001) -> Tuple[np.ndarray, dict]:
        """
        Extract NDVI values for a specific geometry.
        
        Args:
            ndvi_path: Path to the NDVI file
            geometry: GeoJSON-like geometry object
            scale_factor: Factor to scale NDVI values by (default: 0.0001)
            
        Returns:
            Tuple of (NDVI array, metadata)
            
        Raises:
            ProcessingError: If there are issues extracting NDVI data
        """
        try:
            with rasterio.open(ndvi_path) as src:
                # Mask the raster with the geometry
                out_image, out_transform = mask(src, [geometry], crop=True)
                
                # Scale the values
                ndvi_values = out_image[0] * scale_factor
                
                # Create metadata
                metadata = {
                    'transform': out_transform,
                    'crs': src.crs,
                    'nodata': src.nodata,
                    'datetime': self._extract_datetime(ndvi_path)
                }
                
                return ndvi_values, metadata
                
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context=f"extracting NDVI from {ndvi_path}"
            )
    
    def _extract_datetime(self, file_path: Path) -> datetime:
        """
        Extract datetime from NDVI filename.
        Assumes filename contains date in format YYYYMMDD or YYYY_MM_DD.
        
        Args:
            file_path: Path to NDVI file
            
        Returns:
            datetime object
        """
        try:
            # Extract date from filename
            filename = file_path.stem
            # This is a simplified version - adjust based on your actual filename format
            date_str = ''.join(filter(str.isdigit, filename))[:8]
            return datetime.strptime(date_str, '%Y%m%d')
            
        except Exception as e:
            self.logger.warning(f"Could not extract date from {file_path}, using file modification time")
            return datetime.fromtimestamp(file_path.stat().st_mtime) 