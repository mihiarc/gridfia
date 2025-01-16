"""
Vance County specific data filtering functionality.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.utils.error_handler import ErrorHandler, ProcessingError

class VanceFilter:
    """Handles Vance County specific data filtering."""
    
    # Vance County bounding box coordinates (approximate)
    VANCE_BOUNDS = {
        'minx': -78.7,
        'miny': 36.2,
        'maxx': -78.1,
        'maxy': 36.5
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Vance filter.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        self.vance_bbox = box(**self.VANCE_BOUNDS)
        
    def filter_properties(self, 
                         gdf: gpd.GeoDataFrame,
                         county_field: str = 'county') -> gpd.GeoDataFrame:
        """
        Filter properties to include only those in Vance County.
        
        Args:
            gdf: Input GeoDataFrame
            county_field: Name of the field containing county information
            
        Returns:
            Filtered GeoDataFrame containing only Vance County properties
            
        Raises:
            ProcessingError: If filtering fails
        """
        try:
            # First try filtering by county field if it exists
            if county_field in gdf.columns:
                vance_mask = gdf[county_field].str.lower().str.contains('vance')
                filtered_gdf = gdf[vance_mask].copy()
                self.logger.info(f"Filtered {len(filtered_gdf)} properties using county field")
                
            # If no county field or no properties found, use spatial filter
            if county_field not in gdf.columns or len(filtered_gdf) == 0:
                self.logger.info("Using spatial filter for Vance County properties")
                filtered_gdf = self.filter_by_bounds(gdf)
                
            return filtered_gdf
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context="filtering Vance County properties"
            )
    
    def filter_by_bounds(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter properties using Vance County bounding box.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            Filtered GeoDataFrame
            
        Raises:
            ProcessingError: If spatial filtering fails
        """
        try:
            # Ensure the GeoDataFrame is in the same CRS as the bounds (WGS84)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Create spatial index for efficient filtering
            spatial_index = gdf.sindex
            
            # Find candidates that might intersect with Vance County
            candidates = list(spatial_index.intersection(self.vance_bbox.bounds))
            
            if not candidates:
                self.logger.warning("No properties found within Vance County bounds")
                return gdf.iloc[0:0]  # Return empty GeoDataFrame with same schema
            
            # Verify intersection with actual geometry
            mask = gdf.iloc[candidates].intersects(self.vance_bbox)
            filtered_gdf = gdf.iloc[candidates][mask].copy()
            
            self.logger.info(f"Found {len(filtered_gdf)} properties within Vance County bounds")
            return filtered_gdf
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context="spatial filtering for Vance County"
            )
    
    def validate_vance_properties(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Validate that properties are within expected Vance County bounds.
        
        Args:
            gdf: Input GeoDataFrame
            
        Raises:
            ProcessingError: If properties are found outside expected bounds
        """
        try:
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Get bounds of all properties
            total_bounds = gdf.total_bounds
            
            # Check if bounds are reasonable for Vance County
            if (total_bounds[0] < self.VANCE_BOUNDS['minx'] - 0.1 or
                total_bounds[1] < self.VANCE_BOUNDS['miny'] - 0.1 or
                total_bounds[2] > self.VANCE_BOUNDS['maxx'] + 0.1 or
                total_bounds[3] > self.VANCE_BOUNDS['maxy'] + 0.1):
                
                raise ProcessingError(
                    message="Properties found outside expected Vance County bounds",
                    context={
                        "expected_bounds": self.VANCE_BOUNDS,
                        "actual_bounds": {
                            "minx": total_bounds[0],
                            "miny": total_bounds[1],
                            "maxx": total_bounds[2],
                            "maxy": total_bounds[3]
                        }
                    }
                )
                
        except Exception as e:
            if not isinstance(e, ProcessingError):
                e = self.error_handler.handle_processing_error(
                    error=e,
                    context="validating Vance County properties"
                )
            raise e 