"""
Vance County Property Filtering Module
"""
from pathlib import Path
from typing import List, Optional
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import logging

from ..config.vance_config import (
    VANCE_BOUNDS,
    MAX_PROPERTIES,
    DATA_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VancePropertyFilter:
    """Handles filtering and validation of Vance County properties."""
    
    def __init__(self, heirs_file: Path, parcels_file: Path):
        """Initialize the property filter.
        
        Args:
            heirs_file: Path to heirs property data
            parcels_file: Path to NC parcels data
        """
        self.heirs_file = heirs_file
        self.parcels_file = parcels_file
        self.vance_bounds = box(
            VANCE_BOUNDS['minx'],
            VANCE_BOUNDS['miny'],
            VANCE_BOUNDS['maxx'],
            VANCE_BOUNDS['maxy']
        )
        
    def load_and_filter_properties(self) -> gpd.GeoDataFrame:
        """Load and filter properties within Vance County NDVI coverage.
        
        Returns:
            GeoDataFrame of filtered properties
        """
        logger.info("Loading heirs properties...")
        heirs_gdf = gpd.read_parquet(self.heirs_file)
        
        # Filter to Vance County bounds
        logger.info("Filtering to Vance County bounds...")
        vance_mask = heirs_gdf.intersects(self.vance_bounds)
        vance_properties = heirs_gdf[vance_mask].copy()
        
        logger.info(f"Found {len(vance_properties)} properties in Vance County")
        if len(vance_properties) != MAX_PROPERTIES:
            logger.warning(
                f"Expected {MAX_PROPERTIES} properties, found {len(vance_properties)}"
            )
        
        return vance_properties
    
    def validate_properties(self, properties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate properties for analysis.
        
        Args:
            properties: GeoDataFrame of properties to validate
            
        Returns:
            GeoDataFrame of validated properties
        """
        logger.info("Validating properties...")
        
        # Check for invalid geometries
        invalid_mask = ~properties.is_valid
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} invalid geometries")
            properties = properties[~invalid_mask].copy()
        
        # Check for missing required fields
        required_fields = ['property_id', 'geometry', 'area_m2']
        missing_fields = [f for f in required_fields if f not in properties.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Calculate area if missing
        if 'area_m2' not in properties.columns:
            logger.info("Calculating property areas...")
            properties['area_m2'] = properties.geometry.area
        
        return properties
    
    def save_filtered_properties(
        self,
        properties: gpd.GeoDataFrame,
        output_file: Optional[Path] = None
    ) -> Path:
        """Save filtered properties to file.
        
        Args:
            properties: GeoDataFrame to save
            output_file: Optional output path
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = DATA_DIR / "vance_properties.parquet"
            
        logger.info(f"Saving {len(properties)} properties to {output_file}")
        properties.to_parquet(output_file)
        return output_file
    
    def run_filtering(self) -> Path:
        """Run the complete filtering process.
        
        Returns:
            Path to filtered properties file
        """
        properties = self.load_and_filter_properties()
        validated = self.validate_properties(properties)
        output_path = self.save_filtered_properties(validated)
        
        logger.info("Property filtering complete")
        return output_path 