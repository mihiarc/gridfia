"""
Property data loading and preprocessing functionality.
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from src.utils.error_handler import ErrorHandler, ProcessingError

class PropertyLoader:
    """Handles loading and basic preprocessing of property data."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the property loader.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
    def load_property_data(self, file_path: Path) -> gpd.GeoDataFrame:
        """
        Load property data from various file formats (Shapefile, GeoJSON, etc.).
        
        Args:
            file_path: Path to the property data file
            
        Returns:
            GeoDataFrame containing the property data
            
        Raises:
            ProcessingError: If there are issues loading the data
        """
        try:
            self.logger.info(f"Loading property data from {file_path}")
            gdf = gpd.read_file(file_path)
            self.logger.info(f"Successfully loaded {len(gdf)} properties")
            return gdf
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context=f"loading property data from {file_path}"
            )
    
    def ensure_crs(self, gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Ensure the GeoDataFrame is in the specified coordinate reference system.
        
        Args:
            gdf: Input GeoDataFrame
            target_crs: Target CRS (default: EPSG:4326 - WGS84)
            
        Returns:
            GeoDataFrame in the target CRS
            
        Raises:
            ProcessingError: If there are issues with CRS transformation
        """
        try:
            if gdf.crs is None:
                raise ValueError("Input GeoDataFrame has no CRS defined")
                
            if gdf.crs != target_crs:
                self.logger.info(f"Converting CRS from {gdf.crs} to {target_crs}")
                gdf = gdf.to_crs(target_crs)
                
            return gdf
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context=f"ensuring CRS {target_crs}"
            )
    
    def clean_property_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clean and standardize property data.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            Cleaned GeoDataFrame
        """
        try:
            # Remove any rows with invalid geometries
            gdf = gdf[~gdf.geometry.isna()]
            gdf = gdf[gdf.geometry.is_valid]
            
            # Convert column names to lowercase and remove spaces
            gdf.columns = [col.lower().replace(" ", "_") for col in gdf.columns]
            
            # Remove duplicates if any
            gdf = gdf.drop_duplicates()
            
            self.logger.info(f"Cleaned property data: {len(gdf)} valid properties remaining")
            return gdf
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context="cleaning property data"
            ) 